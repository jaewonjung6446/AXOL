"""
AXOL AI Home-Ground Benchmark: Digit Classification
====================================================
Task: sklearn `digits` (MNIST-like, 8x8=64 features, 10 classes, 1797 samples)
  → AXOL's native setting: fixed input dim, fixed output class count.

Comparisons:
  - Logistic Regression (sklearn)
  - MLP (sklearn, 2-layer, ~1K params)
  - Random Forest (sklearn)
  - k-NN (sklearn, k=5)
  - AXOL (DeclarationBuilder + fit_data + observe)

Metrics:
  - Test accuracy (held-out 30%)
  - Train time
  - Inference time per sample
  - AXOL-specific: mean Omega, mean Phi
"""

from __future__ import annotations
import os
import sys
import time
import gc

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axol.core.types import FloatVec
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe


def bench_inference(fn, warmup=5, repeats=30):
    for _ in range(warmup):
        fn()
    gc.collect(); gc.disable()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    gc.enable()
    ts.sort()
    return ts[len(ts) // 2]


def next_pow2(x):
    p = 1
    while p < x: p *= 2
    return p


def main():
    print("=" * 80)
    print("  AXOL AI Home-Ground Benchmark: Digit Classification")
    print("=" * 80)

    # Load data
    data = load_digits()
    X, y = data.data.astype(np.float32), data.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    n_classes = 10
    n_features = X_train_s.shape[1]  # 64
    n_train, n_test = len(X_train_s), len(X_test_s)
    print(f"  Dataset: {n_train} train / {n_test} test, dim={n_features}, classes={n_classes}")

    results = []

    # -------------------- Baselines --------------------
    baselines = [
        ("LogReg", LogisticRegression(max_iter=1000, random_state=42)),
        ("MLP(64-32-10)", MLPClassifier(hidden_layer_sizes=(32,), max_iter=500,
                                         random_state=42, early_stopping=False)),
        ("RF(100)", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
        ("kNN(k=5)", KNeighborsClassifier(n_neighbors=5)),
    ]

    for name, clf in baselines:
        t0 = time.perf_counter()
        clf.fit(X_train_s, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = clf.predict(X_test_s)
        pred_time_total = time.perf_counter() - t0
        acc = float((preds == y_test).mean())

        # Per-sample inference
        def single_pred():
            _ = clf.predict(X_test_s[:1])
        t_single = bench_inference(single_pred)

        results.append({
            "name": name,
            "train_s": train_time,
            "infer_total_s": pred_time_total,
            "infer_per_sample_us": t_single / 1000,
            "accuracy": acc,
            "omega": None,
            "phi": None,
        })
        print(f"  [{name:<14}] acc={acc:.4f}  train={train_time*1000:7.1f}ms  "
              f"infer/item={t_single/1000:7.1f}us")

    # -------------------- AXOL (multiple dim configurations) --------------------
    for dim_choice in [64, 128, 256, 512]:
        dim = dim_choice

        def pad(X, d=dim):
            if X.shape[1] < d:
                padded = np.zeros((X.shape[0], d), dtype=np.float32)
                padded[:, :X.shape[1]] = X
                return padded
            return X[:, :d]

        X_train_pad = pad(X_train_s)
        X_test_pad = pad(X_test_s)

        b = DeclarationBuilder(f"digit_classify_d{dim}")
        b.input("feat", dim)
        b.relate("class", ["feat"], RelationKind.PROPORTIONAL)
        b.output("class")
        b.quality(omega=0.9, phi=0.85)
        decl = b.build()

        t0 = time.perf_counter()
        tap = weave(decl, quantum=False, seed=42,
                    fit_data={"input": X_train_pad, "target": y_train})
        axol_train_time = time.perf_counter() - t0

        omega_sum, phi_sum = 0.0, 0.0
        preds = np.empty(n_test, dtype=np.int64)
        t0 = time.perf_counter()
        for i in range(n_test):
            obs = observe(tap, {"feat": FloatVec(data=X_test_pad[i])})
            preds[i] = obs.value_index
            omega_sum += obs.omega
            phi_sum += obs.phi
        axol_infer_total = time.perf_counter() - t0
        axol_acc = float((preds == y_test).mean())

        single_fv = FloatVec(data=X_test_pad[0])
        def axol_single():
            _ = observe(tap, {"feat": single_fv})
        t_axol_single = bench_inference(axol_single)

        results.append({
            "name": f"AXOL(dim={dim})",
            "train_s": axol_train_time,
            "infer_total_s": axol_infer_total,
            "infer_per_sample_us": t_axol_single / 1000,
            "accuracy": axol_acc,
            "omega": omega_sum / n_test,
            "phi": phi_sum / n_test,
            "dim": dim,
        })
        print(f"  [AXOL(dim={dim:>3})   ] acc={axol_acc:.4f}  train(weave)={axol_train_time*1000:7.1f}ms  "
              f"infer/item={t_axol_single/1000:7.1f}us  Omega={omega_sum/n_test:.3f}  Phi={phi_sum/n_test:.3f}")

    # -------------------- Summary --------------------
    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print(f"  {'method':<16} {'acc':>7} {'train':>10} {'infer/item':>12} {'Omega':>7} {'Phi':>7}")
    print("  " + "-" * 64)
    for r in results:
        omega_s = f"{r['omega']:.3f}" if r['omega'] is not None else "  -  "
        phi_s = f"{r['phi']:.3f}" if r['phi'] is not None else "  -  "
        print(f"  {r['name']:<16} {r['accuracy']:>7.4f} "
              f"{r['train_s']*1000:>8.1f}ms {r['infer_per_sample_us']:>10.1f}us "
              f"{omega_s:>7} {phi_s:>7}")

    # Save JSON
    import json
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "benchmark_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ai_classification.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
