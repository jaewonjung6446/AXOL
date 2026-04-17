"""
AXOL Sort Similarity Benchmark
===============================
Claim under test: for non-iterative probabilistic sort (AXOL observe),
  - exact position accuracy is theoretically bounded (~1 - 1/e floor)
  - but STRUCTURAL similarity of output vs true sort can be ~99%

We measure both families of metrics on arrays of size n in {100, 300, 1000, 3000}
(and optionally 10000). AXOL is trained via fit_data on (embedded_value, rank)
pairs drawn from the same distribution; rank prediction is then a single
non-iterative observe() per element. No gradient descent, no comparisons.

Metrics:
  - exact_accuracy:            mean(argmax(pred_rank) == true_rank)
  - kendall_tau:               rank correlation of predicted permutation vs truth
  - spearman_rho:              rank correlation (Spearman)
  - normalized_displacement:   mean(|pred_rank - true_rank|) / n
  - value_cosine_similarity:   cos(axol_sorted_values, truly_sorted_values)
  - neighbor_preservation:     fraction of true-adjacent pairs that remain adjacent
  - inversion_ratio:           1 - inversions / max_inversions

Reference baselines:
  - random:        shuffled array (no information)
  - quicksort:     reference 100%
  - fixed_point_floor: 1 - 1/e ~ 0.6321 (classical derangement bound for >=1 fp)
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axol.core.types import FloatVec
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.observatory import observe
from axol.quantum.weaver import weave


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall rank correlation via O(n log n) merge-sort count."""
    n = len(a)
    if n < 2:
        return 1.0
    order = np.argsort(a, kind="mergesort")
    b_sorted = b[order]
    inv = _count_inversions(b_sorted.copy())
    max_pairs = n * (n - 1) / 2
    return 1.0 - 2.0 * inv / max_pairs


def _count_inversions(arr: np.ndarray) -> int:
    def merge_count(a):
        if len(a) <= 1:
            return a, 0
        mid = len(a) // 2
        left, cl = merge_count(a[:mid])
        right, cr = merge_count(a[mid:])
        merged = np.empty(len(a), dtype=a.dtype)
        i = j = k = 0
        inv = cl + cr
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged[k] = left[i]
                i += 1
            else:
                merged[k] = right[j]
                j += 1
                inv += len(left) - i
            k += 1
        while i < len(left):
            merged[k] = left[i]; i += 1; k += 1
        while j < len(right):
            merged[k] = right[j]; j += 1; k += 1
        return merged, inv
    _, total = merge_count(arr)
    return int(total)


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        return 0.0
    return float((ra * rb).sum() / denom)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def neighbor_preservation(true_order: np.ndarray, pred_order: np.ndarray) -> float:
    """Fraction of (i, i+1) pairs in true sort that remain adjacent in predicted sort."""
    n = len(true_order)
    if n < 2:
        return 1.0
    pos_in_pred = np.empty(n, dtype=np.int64)
    pos_in_pred[pred_order] = np.arange(n)
    kept = 0
    for i in range(n - 1):
        if abs(pos_in_pred[true_order[i]] - pos_in_pred[true_order[i + 1]]) == 1:
            kept += 1
    return kept / (n - 1)


# ---------------------------------------------------------------------------
# Value embedding (scalar -> dim-vector)
# ---------------------------------------------------------------------------

def make_embedder(dim: int, seed: int = 0):
    """Random-feature embedding: scalar x -> cos/sin of random frequencies.
    This requires AXOL to *learn* the inverse (rank) from a non-trivial feature map.
    """
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.5, dim / 2.0, size=dim // 2)
    phases = rng.uniform(0.0, 2 * math.pi, size=dim // 2)

    def embed(x: float) -> np.ndarray:
        v = np.empty(dim, dtype=np.float32)
        v[: dim // 2] = np.cos(2 * math.pi * freqs * x + phases)
        v[dim // 2 :] = np.sin(2 * math.pi * freqs * x + phases)
        n = np.linalg.norm(v) + 1e-10
        return (v / n).astype(np.float32)

    return embed


# ---------------------------------------------------------------------------
# AXOL sort
# ---------------------------------------------------------------------------

def next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def axol_sort(
    X: np.ndarray,
    n_train: int = 2000,
    seed: int = 42,
    quantum: bool = True,
):
    """Sort X by asking AXOL to predict rank for each element, non-iteratively."""
    n = len(X)
    dim = max(16, next_pow2(n))  # output space must be >= n
    embed = make_embedder(dim, seed=seed)

    # Training: uniform samples -> rank bin (quantile * n).
    rng = np.random.default_rng(seed)
    t_vals = rng.uniform(0.0, 1.0, size=n_train)
    t_X = np.stack([embed(v) for v in t_vals])
    t_ranks = np.clip((t_vals * n).astype(np.int64), 0, n - 1)

    b = DeclarationBuilder("sort")
    b.input("value", dim)
    b.relate("rank", ["value"], RelationKind.PROPORTIONAL)
    b.output("rank")
    b.quality(omega=0.85, phi=0.7)
    decl = b.build()

    t0 = time.perf_counter()
    tap = weave(
        decl,
        quantum=quantum,
        seed=seed,
        fit_data={"input": t_X, "target": t_ranks},
    )
    weave_time = time.perf_counter() - t0

    pred_ranks = np.empty(n, dtype=np.int64)
    omega_sum = 0.0
    phi_sum = 0.0
    t0 = time.perf_counter()
    for i, x in enumerate(X):
        obs = observe(tap, {"value": FloatVec(data=embed(float(x)))})
        pred_ranks[i] = obs.value_index
        omega_sum += obs.omega
        phi_sum += obs.phi
    observe_time = time.perf_counter() - t0

    return pred_ranks, {
        "weave_s": weave_time,
        "observe_s": observe_time,
        "mean_omega": omega_sum / n,
        "mean_phi": phi_sum / n,
    }


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def evaluate_sort(X: np.ndarray, pred_ranks: np.ndarray):
    n = len(X)
    true_ranks = np.argsort(np.argsort(X))
    pred_order = np.argsort(pred_ranks, kind="mergesort")
    true_order = np.argsort(X, kind="mergesort")

    exact_acc = float(np.mean(pred_ranks == true_ranks))

    tau = kendall_tau(pred_ranks.astype(np.float64), true_ranks.astype(np.float64))
    rho = spearman_rho(pred_ranks.astype(np.float64), true_ranks.astype(np.float64))
    disp = float(np.mean(np.abs(pred_ranks - true_ranks)) / n)

    axol_sorted_values = X[pred_order]
    true_sorted_values = np.sort(X)
    cos_val = cosine_sim(axol_sorted_values, true_sorted_values)

    neighbor = neighbor_preservation(true_order, pred_order)

    return {
        "exact_accuracy": exact_acc,
        "kendall_tau": tau,
        "spearman_rho": rho,
        "norm_displacement": disp,
        "value_cosine": cos_val,
        "neighbor_preservation": neighbor,
    }


def random_baseline(X: np.ndarray, seed: int):
    n = len(X)
    rng = np.random.default_rng(seed)
    pred = rng.permutation(n)
    return evaluate_sort(X, pred)


def run_once(n: int, n_train: int, seed: int, quantum: bool = True):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=n).astype(np.float32)

    pred_ranks, timing = axol_sort(X, n_train=n_train, seed=seed, quantum=quantum)
    axol = evaluate_sort(X, pred_ranks)
    rnd = random_baseline(X, seed=seed + 1)

    return {
        "n": n,
        "n_train": n_train,
        "axol": axol,
        "random": rnd,
        "timing": timing,
    }


def format_result(res: dict) -> str:
    n = res["n"]
    a = res["axol"]; r = res["random"]; t = res["timing"]
    return (
        f"\n[n={n}, n_train={res['n_train']}]\n"
        f"  weave={t['weave_s']*1000:.1f}ms  observe={t['observe_s']*1000:.1f}ms  "
        f"Omega={t['mean_omega']:.3f}  Phi={t['mean_phi']:.3f}\n"
        f"  {'metric':<28}{'AXOL':>10}{'random':>10}{'diff':>10}\n"
        f"  {'-'*58}\n"
        f"  {'exact_accuracy':<28}{a['exact_accuracy']:>10.4f}{r['exact_accuracy']:>10.4f}"
        f"{a['exact_accuracy']-r['exact_accuracy']:>10.4f}\n"
        f"  {'kendall_tau':<28}{a['kendall_tau']:>10.4f}{r['kendall_tau']:>10.4f}"
        f"{a['kendall_tau']-r['kendall_tau']:>10.4f}\n"
        f"  {'spearman_rho':<28}{a['spearman_rho']:>10.4f}{r['spearman_rho']:>10.4f}"
        f"{a['spearman_rho']-r['spearman_rho']:>10.4f}\n"
        f"  {'norm_displacement (lower=better)':<28}{a['norm_displacement']:>10.4f}"
        f"{r['norm_displacement']:>10.4f}"
        f"{a['norm_displacement']-r['norm_displacement']:>10.4f}\n"
        f"  {'value_cosine':<28}{a['value_cosine']:>10.4f}{r['value_cosine']:>10.4f}"
        f"{a['value_cosine']-r['value_cosine']:>10.4f}\n"
        f"  {'neighbor_preservation':<28}{a['neighbor_preservation']:>10.4f}"
        f"{r['neighbor_preservation']:>10.4f}"
        f"{a['neighbor_preservation']-r['neighbor_preservation']:>10.4f}\n"
    )


def main():
    # Allow --sizes flag to override, e.g. --sizes 100,300,1000
    sizes = [100, 300, 1000]
    explicit_ntrain: int | None = None
    for i, arg in enumerate(sys.argv):
        if arg == "--sizes" and i + 1 < len(sys.argv):
            sizes = [int(s) for s in sys.argv[i + 1].split(",")]
        if arg == "--ntrain" and i + 1 < len(sys.argv):
            explicit_ntrain = int(sys.argv[i + 1])
    if "--big" in sys.argv:
        sizes.append(10000)
    quantum = "--classical" not in sys.argv

    print("=" * 80)
    print("  AXOL Sort Similarity Benchmark")
    print("  Claim: exact accuracy bounded, structural similarity ~99%")
    print("=" * 80)
    print(f"  Theoretical floor (1 - 1/e) = {1 - 1/math.e:.4f}")
    print(f"  Random perm expected exact match rate = 1/n")

    all_results = []
    for n in sizes:
        n_train = explicit_ntrain if explicit_ntrain is not None else min(5000, max(500, 5 * n))
        try:
            t0 = time.perf_counter()
            res = run_once(n=n, n_train=n_train, seed=42, quantum=quantum)
            elapsed = time.perf_counter() - t0
            print(format_result(res), flush=True)
            print(f"  [n={n}] elapsed={elapsed:.1f}s", flush=True)
            all_results.append(res)
        except Exception as exc:
            print(f"\n[n={n}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            import traceback; traceback.print_exc()

    # Summary table
    print("\n" + "=" * 80)
    print("  Summary (AXOL)")
    print("=" * 80)
    hdr = f"  {'n':>6} {'exact':>8} {'kendall':>8} {'spearman':>9} {'disp':>8} {'cos_val':>8} {'neigh':>8}"
    print(hdr)
    print("  " + "-" * 60)
    for r in all_results:
        a = r["axol"]
        print(
            f"  {r['n']:>6} {a['exact_accuracy']:>8.3f} {a['kendall_tau']:>8.3f} "
            f"{a['spearman_rho']:>9.3f} {a['norm_displacement']:>8.3f} "
            f"{a['value_cosine']:>8.3f} {a['neighbor_preservation']:>8.3f}"
        )

    # Append JSON report (merge with prior runs, keep latest per n)
    import json
    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmark_results",
        "sort_similarity.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    prior = []
    if os.path.exists(out):
        try:
            with open(out) as f:
                prior = json.load(f)
        except Exception:
            prior = []
    by_n = {r.get("n"): r for r in prior}
    for r in all_results:
        by_n[r["n"]] = r
    merged = [by_n[k] for k in sorted(by_n.keys())]
    with open(out, "w") as f:
        json.dump(merged, f, indent=2, default=float)
    print(f"\n  Saved: {out}  ({len(merged)} sizes total)")


if __name__ == "__main__":
    main()
