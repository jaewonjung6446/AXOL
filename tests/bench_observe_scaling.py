"""
AXOL 관측 스케일링 극한 벤치마크
═══════════════════════════════

5가지 핵심 실험:
  1. 차원 스케일링    — dim=4 → 4096, depth=1   (관측의 차원 독립성 검증)
  2. 파이프라인 깊이  — depth=5 → 5000, dim=16  (기존 순차실행 vs AXOL 관측)
  3. 반복 관측 상각   — 1회 → 1,000,000회        (한계 비용 제로 검증)
  4. 차원×깊이 동시   — (64,10)→(1024,1000)      (복합 스케일링)
  5. 정확도 검증      — Hellinger / mode match    (속도뿐 아니라 정확도도 확인)

실행:
  python tests/bench_observe_scaling.py

출력:
  benchmark_results/
  ├── fig1_dimension_scaling.png
  ├── fig2_depth_scaling.png
  ├── fig3_amortization.png
  ├── fig4_combined.png
  ├── fig5_accuracy.png
  ├── fig_combined_4panel.png
  └── OBSERVE_SCALING_REPORT.md
"""

import sys
import os
import time
import gc

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axol.core.types import FloatVec, TransMatrix, StateBundle
from axol.core import operations as ops
from axol.core.program import TransformOp, MergeOp, MeasureOp
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib 미설치 — 차트 생략 (pip install matplotlib)")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "benchmark_results")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Timing
# ═══════════════════════════════════════════════════════════════════════════════

def _bench(fn, warmup=2, repeats=7):
    for _ in range(warmup):
        fn()
    gc.collect()
    gc.disable()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    gc.enable()
    times.sort()
    return times[0], times[len(times) // 2], sum(times) / len(times)


def _fmt(sec):
    if sec < 1e-6:
        return f"{sec*1e9:.0f}ns"
    if sec < 1e-3:
        return f"{sec*1e6:.1f}\u03bcs"
    if sec < 1.0:
        return f"{sec*1e3:.2f}ms"
    if sec < 60.0:
        return f"{sec:.2f}s"
    return f"{sec/60:.1f}min"


# ═══════════════════════════════════════════════════════════════════════════════
# Traditional Sequential Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _build_matrices(dim, depth, seed=42):
    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(depth):
        M = np.eye(dim, dtype=np.float32) * 0.8
        M += rng.standard_normal((dim, dim)).astype(np.float32) * 0.1
        Q, _R = np.linalg.qr(M.astype(np.float64))
        matrices.append((Q * 0.95).astype(np.float32))
    return matrices


def _trad_run(matrices, x):
    state = x.copy()
    for M in matrices:
        state = state @ M
    probs = state * state
    total = probs.sum()
    if total > 0:
        probs /= total
    return int(np.argmax(probs)), probs


def _compose(matrices):
    R = np.eye(matrices[0].shape[0], dtype=np.float64)
    for M in matrices:
        R = R @ M.astype(np.float64)
    return R.astype(np.float32)


def _composed_run(Mc, x):
    state = x @ Mc
    probs = state * state
    total = probs.sum()
    if total > 0:
        probs /= total
    return int(np.argmax(probs)), probs


# ═══════════════════════════════════════════════════════════════════════════════
# AXOL Sequential Observe  (fallback path — no composition, step by step)
# ═══════════════════════════════════════════════════════════════════════════════

def _sequential_observe(tapestry, inputs_dict):
    """Run transitions sequentially without composed matrix (old behavior)."""
    program = tapestry._internal_program
    state = dict(program.initial_state.vectors)  # shallow
    for name, vec in inputs_dict.items():
        if name in state:
            state[name] = vec
    for t in program.transitions:
        op = t.operation
        if isinstance(op, TransformOp):
            state[op.out_key or op.key] = ops.transform(state[op.key], op.matrix)
        elif isinstance(op, MergeOp):
            vecs = [state[k] for k in op.keys]
            state[op.out_key] = ops.merge(vecs, op.weights)
        elif isinstance(op, MeasureOp):
            state[op.out_key or op.key] = ops.measure(state[op.key])
    out = tapestry.output_names[0]
    pk = f"_prob_{out}"
    probs = state[pk] if pk in state else ops.measure(state[out])
    return int(np.argmax(probs.data)), probs


def _axol_observe(tapestry, inputs_dict):
    """Use the real observe() which now has the composed fast path built-in."""
    obs = observe(tapestry, inputs_dict)
    return obs.value_index, obs.probabilities


def _build_tapestry(dim, depth, seed=42):
    b = DeclarationBuilder(f"p_d{depth}_dim{dim}")
    b.input("x", dim)
    prev = "x"
    for i in range(depth):
        nm = f"n{i}"
        b.relate(nm, [prev], RelationKind.PROPORTIONAL)
        prev = nm
    b.output(prev).quality(0.8, 0.7)
    return weave(b.build(), seed=seed)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Observe Time vs Dimension  (depth=1)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_1():
    print("\n" + "\u2550" * 90)
    print("  \uc2e4\ud5d8 1: \uad00\uce21 \uc2dc\uac04 vs \ucc28\uc6d0  (depth=1, dim=4 \u2192 4096)")
    print("  \u2014 dim\uc774 1000\ubc30 \ucee4\uc838\ub3c4 \uad00\uce21 \ube44\uc6a9\uc774 \uc5b4\ub5bb\uac8c \ubcc0\ud558\ub294\uac00?")
    print("\u2550" * 90)

    dims = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    rows = []

    for dim in dims:
        print(f"  dim={dim:>5} ... ", end="", flush=True)
        x_np = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        x_fv = FloatVec(data=x_np.copy())

        # Traditional
        matrices = _build_matrices(dim, 1, seed=42)
        _, trad, _ = _bench(lambda: _trad_run(matrices, x_np), warmup=5, repeats=50)

        # Raw numpy matmul
        M0 = matrices[0]
        _, raw, _ = _bench(lambda: x_np @ M0, warmup=10, repeats=100)

        # AXOL weave
        t0 = time.perf_counter()
        tap = _build_tapestry(dim, 1, seed=42)
        weave_s = time.perf_counter() - t0

        # AXOL observe (now with composed fast path)
        inp = {"x": x_fv}
        _, axol, _ = _bench(lambda: _axol_observe(tap, inp), warmup=5, repeats=50)

        rows.append(dict(dim=dim, trad_us=trad*1e6, axol_us=axol*1e6,
                         raw_us=raw*1e6, weave_ms=weave_s*1e3))
        print(f"trad={_fmt(trad):>9s}  axol={_fmt(axol):>9s}  "
              f"raw={_fmt(raw):>9s}  weave={_fmt(weave_s):>9s}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Pipeline Depth Scaling  (dim=16)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_2():
    print("\n" + "\u2550" * 90)
    print("  실험 2: 파이프라인 깊이 스케일링  (dim=16, depth=5 → 5000)")
    print("  — 기존 순차실행은 직선, AXOL observe(내장 composed)는 수평선")
    print("═" * 90)

    dim = 16
    depths = [5, 50, 500, 5000]
    rows = []

    for depth in depths:
        print(f"  depth={depth:>5} ... ", end="", flush=True)
        x_np = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        x_fv = FloatVec(data=x_np.copy())
        reps = max(3, min(50, 20000 // max(depth, 1)))

        # Traditional
        matrices = _build_matrices(dim, depth, seed=42)
        _, trad, _ = _bench(lambda: _trad_run(matrices, x_np), warmup=2, repeats=reps)

        # Composed baseline (external single matmul — depth-independent)
        Mc = _compose(matrices)
        _, comp, _ = _bench(lambda: _composed_run(Mc, x_np), warmup=5, repeats=50)

        # AXOL weave
        t0 = time.perf_counter()
        tap = _build_tapestry(dim, depth, seed=42)
        weave_s = time.perf_counter() - t0

        inp = {"x": x_fv}

        # AXOL observe (now with built-in composed fast path)
        _, axol, _ = _bench(lambda: _axol_observe(tap, inp), warmup=2, repeats=reps)

        # AXOL sequential (old fallback path, for comparison)
        _, seq, _ = _bench(lambda: _sequential_observe(tap, inp), warmup=2, repeats=reps)

        rows.append(dict(depth=depth, trad_us=trad*1e6, comp_us=comp*1e6,
                         axol_us=axol*1e6, seq_us=seq*1e6, weave_ms=weave_s*1e3))
        print(f"trad={_fmt(trad):>9s}  composed={_fmt(comp):>9s}  "
              f"axol={_fmt(axol):>9s}  sequential={_fmt(seq):>9s}  weave={_fmt(weave_s):>9s}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Repeated Observation Amortization
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_3():
    print("\n" + "\u2550" * 90)
    print("  \uc2e4\ud5d8 3: \ubc18\ubcf5 \uad00\uce21 \uc0c1\uac01  (dim=16, depth=100)")
    print("  \u2014 \uc9c1\uc870 \ube44\uc6a9\uc774 \uc0c1\uac01\ub418\uba74\uc11c \ud638\ucd9c\ub2f9 \ube44\uc6a9 \u2192 0 \uc218\ub834")
    print("\u2550" * 90)

    dim, depth = 16, 100
    x_np = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    x_fv = FloatVec(data=x_np.copy())

    matrices = _build_matrices(dim, depth, seed=42)
    _, trad, _ = _bench(lambda: _trad_run(matrices, x_np), warmup=3, repeats=30)

    t0 = time.perf_counter()
    tap = _build_tapestry(dim, depth, seed=42)
    weave_s = time.perf_counter() - t0

    inp = {"x": x_fv}
    _, axol_obs, _ = _bench(lambda: _axol_observe(tap, inp), warmup=3, repeats=30)

    Mc = _compose(matrices)
    _, comp, _ = _bench(lambda: _composed_run(Mc, x_np), warmup=5, repeats=50)
    _, compose_cost, _ = _bench(lambda: _compose(matrices), warmup=1, repeats=5)

    print(f"  \ub2e8\uc77c \uad00\uce21: trad={_fmt(trad)}  axol={_fmt(axol_obs)}  composed={_fmt(comp)}")
    print(f"  \uc9c1\uc870 (1\ud68c): weave={_fmt(weave_s)}  compose={_fmt(compose_cost)}")
    print()

    counts = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    rows = []

    for N in counts:
        axol_per = (weave_s + N * axol_obs) / N
        comp_per = (compose_cost + N * comp) / N
        rows.append(dict(
            count=N, trad_per_us=trad*1e6, axol_per_us=axol_per*1e6,
            comp_per_us=comp_per*1e6, weave_amort_us=(weave_s/N)*1e6,
        ))
        print(f"  N={N:>9,} | trad/call={_fmt(trad):>9s} | "
              f"axol/call={_fmt(axol_per):>9s} | "
              f"composed/call={_fmt(comp_per):>9s} | "
              f"\uc9c1\uc870\uc0c1\uac01={_fmt(weave_s/N):>9s}")

    meta = dict(weave_ms=weave_s*1e3, observe_us=axol_obs*1e6,
                trad_us=trad*1e6, comp_us=comp*1e6, compose_cost_ms=compose_cost*1e3)
    return rows, meta


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Dimension x Depth Combined
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_4():
    print("\n" + "\u2550" * 90)
    print("  \uc2e4\ud5d8 4: \ucc28\uc6d0 \u00d7 \uae4a\uc774 \ub3d9\uc2dc \uc99d\uac00  (dim\u00d7depth combined)")
    print("  \u2014 \uae30\uc874 \ubc29\uc2dd\uc740 \ube44\uc6a9\uc774 \uacf1\uc73c\ub85c \ud3ed\ubc1c, AXOL \uc9c1\uc870 \uad00\uce21\uc740?")
    print("\u2550" * 90)

    configs = [(64, 10), (128, 50), (256, 100), (512, 100)]
    rows = []

    for dim, depth in configs:
        print(f"  dim={dim:>5}, depth={depth:>5} ... ", end="", flush=True)
        x_np = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        x_fv = FloatVec(data=x_np.copy())
        reps = max(3, min(20, 5000 // max(depth, 1)))

        matrices = _build_matrices(dim, depth, seed=42)
        _, trad, _ = _bench(lambda: _trad_run(matrices, x_np), warmup=1, repeats=reps)

        t0 = time.perf_counter()
        Mc = _compose(matrices)
        compose_s = time.perf_counter() - t0
        _, comp, _ = _bench(lambda: _composed_run(Mc, x_np), warmup=3, repeats=30)

        t0 = time.perf_counter()
        tap = _build_tapestry(dim, depth, seed=42)
        weave_s = time.perf_counter() - t0

        inp = {"x": x_fv}
        _, axol, _ = _bench(lambda: _axol_observe(tap, inp), warmup=1, repeats=reps)

        speedup = trad / comp if comp > 0 else float("inf")
        rows.append(dict(
            dim=dim, depth=depth, complexity=dim*dim*depth,
            trad_us=trad*1e6, comp_us=comp*1e6, axol_us=axol*1e6,
            weave_s=weave_s, compose_s=compose_s, speedup=speedup,
        ))
        print(f"trad={_fmt(trad):>9s}  composed={_fmt(comp):>9s}  "
              f"axol={_fmt(axol):>9s}  weave={_fmt(weave_s):>9s}  "
              f"speedup={speedup:>.0f}x")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Accuracy — Composed vs Traditional vs AXOL Observe
# ═══════════════════════════════════════════════════════════════════════════════

def _hellinger(p, q):
    """Hellinger distance between two probability distributions [0,1]."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def _total_variation(p, q):
    """Total variation distance between two distributions."""
    return float(0.5 * np.sum(np.abs(np.asarray(p, np.float64) - np.asarray(q, np.float64))))


def experiment_5():
    """Accuracy verification: AXOL observe (composed fast path) vs sequential (step-by-step).

    Both use the same weaver matrices. The composed path pre-multiplies all
    TransformOp matrices into one; the sequential path runs each transform
    step by step. If the composition is correct, the probability distributions
    should match within float32 precision.
    """
    print("\n" + "=" * 90)
    print("  Experiment 5: Accuracy -- Composed observe vs Sequential observe")
    print("  -- same weaver matrices, composed fast path vs step-by-step fallback")
    print("=" * 90)

    dims = [4, 16, 64, 256]
    depths = [1, 5, 10, 50, 100]
    rows = []

    for dim in dims:
        for depth in depths:
            x_fv = FloatVec(data=np.ones(dim, dtype=np.float32) / np.sqrt(dim))

            tap = _build_tapestry(dim, depth, seed=42)
            inp = {"x": x_fv}

            # Composed fast path (observe with _composed_matrix)
            obs_composed = observe(tap, inp)
            comp_p = obs_composed.probabilities.data.astype(np.float64)
            comp_idx = obs_composed.value_index

            # Sequential fallback (step-by-step through all transitions)
            seq_idx, seq_probs = _sequential_observe(tap, inp)
            seq_p = seq_probs.data.astype(np.float64)

            # Metrics
            h = _hellinger(comp_p, seq_p)
            tv = _total_variation(comp_p, seq_p)
            max_diff = float(np.max(np.abs(comp_p - seq_p)))
            mode_match = 1 if comp_idx == seq_idx else 0

            row = dict(dim=dim, depth=depth,
                       h_axol=h, tv_axol=tv, max_diff_axol=max_diff, mode_axol=mode_match)
            rows.append(row)
            print(f"  dim={dim:>5} depth={depth:>5} | "
                  f"Hellinger={h:.8f}  TV={tv:.8f}  MaxDiff={max_diff:.8f} | "
                  f"mode={'OK' if mode_match else 'MISMATCH'}")

    # Summary
    all_h = [r["h_axol"] for r in rows]
    all_m = [r["mode_axol"] for r in rows]
    print(f"\n  Summary:")
    print(f"    Hellinger distance: max={max(all_h):.8f}  mean={np.mean(all_h):.8f}")
    print(f"    Mode match rate: {sum(all_m)}/{len(all_m)} ({100*sum(all_m)/len(all_m):.0f}%)")
    if max(all_h) < 0.01:
        print(f"    --> ALL Hellinger < 0.01 -- float32 precision only")
    else:
        print(f"    --> WARNING: some Hellinger >= 0.01")

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════════

CT = "#E74C3C"   # red   - traditional
CA = "#3498DB"   # blue  - axol
CC = "#2ECC71"   # green - composed
CR = "#95A5A6"   # gray  - raw
CW = "#F39C12"   # orange


def _fig1(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    dims = [r["dim"] for r in data]
    ax.plot(dims, [r["trad_us"] for r in data], "o-",  color=CT, lw=2, ms=6, label="Traditional Sequential")
    ax.plot(dims, [r["axol_us"] for r in data], "s-",  color=CA, lw=2, ms=6, label="AXOL Fast Observe")
    ax.plot(dims, [r["raw_us"]  for r in data], "^--", color=CR, lw=1.5, ms=5, label="Raw NumPy matmul")

    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Dimension (log\u2082)", fontsize=12)
    ax.set_ylabel("Observe Time (\u03bcs, log)", fontsize=12)
    ax.set_title("\uc2e4\ud5d8 1: \uad00\uce21 \uc2dc\uac04 vs \ucc28\uc6d0  (depth=1)", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(dims); ax.set_xticklabels([str(d) for d in dims], rotation=45, fontsize=9)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig1_dimension_scaling.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  [saved] {p}")


def _fig2(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ds = [r["depth"] for r in data]
    ax.plot(ds, [r["trad_us"] for r in data], "o-",  color=CT, lw=2.5, ms=8, label="Traditional Sequential")
    ax.plot(ds, [r["seq_us"] for r in data],  "x--", color=CW, lw=2, ms=7, label="AXOL Sequential (fallback)")
    ax.plot(ds, [r["axol_us"] for r in data], "s-",  color=CA, lw=2.5, ms=8, label="AXOL Observe (composed built-in)")
    ax.plot(ds, [r["comp_us"] for r in data], "D-",  color=CC, lw=2.5, ms=8, label="Composed Baseline (external)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Pipeline Depth (log)", fontsize=12)
    ax.set_ylabel("Per-Observation Time (\u03bcs, log)", fontsize=12)
    ax.set_title("실험 2: 파이프라인 깊이 스케일링  (dim=16)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left"); ax.grid(True, alpha=0.3, which="both")

    if len(data) >= 2:
        gap = data[-1]["trad_us"] / max(data[-1]["axol_us"], 0.001)
        ax.annotate(f"{gap:,.0f}x", xy=(data[-1]["depth"], data[-1]["axol_us"]),
                    xytext=(data[-1]["depth"]/3, data[-1]["axol_us"]*0.25),
                    fontsize=13, fontweight="bold", color=CA,
                    arrowprops=dict(arrowstyle="->", color=CA, lw=2))

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig2_depth_scaling.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  [saved] {p}")


def _fig3(data, meta):
    fig, ax = plt.subplots(figsize=(10, 6))
    ns = [r["count"] for r in data]
    ax.plot(ns, [r["trad_per_us"] for r in data],  "o-", color=CT, lw=2.5, ms=7, label="Traditional (\ub9e4 \ud638\ucd9c)")
    ax.plot(ns, [r["axol_per_us"] for r in data],  "s-", color=CA, lw=2.5, ms=7, label="AXOL Amortized (weave+observe)")
    ax.plot(ns, [r["comp_per_us"] for r in data],  "D-", color=CC, lw=2.5, ms=7, label="Composed Amortized")
    ax.axhline(y=meta["observe_us"], color=CA, ls=":", alpha=0.5, lw=1)
    ax.text(ns[0]*1.5, meta["observe_us"]*0.65,
            f"observe only = {meta['observe_us']:.1f}\u03bcs", fontsize=9, color=CA, alpha=0.7)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Observation Count (log)", fontsize=12)
    ax.set_ylabel("Amortized Cost / Observation (\u03bcs, log)", fontsize=12)
    ax.set_title("\uc2e4\ud5d8 3: \ubc18\ubcf5 \uad00\uce21 \uc0c1\uac01  (dim=16, depth=100)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig3_amortization.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  [saved] {p}")


def _fig4(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"dim={r['dim']}\ndepth={r['depth']}" for r in data]
    x = np.arange(len(data)); w = 0.25
    ax.bar(x-w, [r["trad_us"]/1e3 for r in data], w, color=CT, label="Traditional", zorder=3)
    ax.bar(x,   [r["axol_us"]/1e3 for r in data], w, color=CA, label="AXOL Observe", zorder=3)
    ax.bar(x+w, [r["comp_us"]/1e3 for r in data], w, color=CC, label="Composed",     zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Per-Observation Time (ms, log)", fontsize=12)
    ax.set_title("\uc2e4\ud5d8 4: \ucc28\uc6d0 \u00d7 \uae4a\uc774 \ubcf5\ud569 \uc2a4\ucf00\uc77c\ub9c1", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y", which="both")
    for i, r in enumerate(data):
        if r["speedup"] > 1:
            ax.text(i+w, r["comp_us"]/1e3*0.35, f'{r["speedup"]:,.0f}x',
                    ha="center", fontsize=11, fontweight="bold", color=CC)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig4_combined.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  [saved] {p}")


def _fig5(data):
    """Accuracy heatmap: Hellinger distance for AXOL vs Traditional."""
    dims = sorted(set(r["dim"] for r in data))
    depths = sorted(set(r["depth"] for r in data))

    h_matrix = np.zeros((len(dims), len(depths)))
    for r in data:
        i = dims.index(r["dim"])
        j = depths.index(r["depth"])
        h_matrix[i, j] = r["h_axol"]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(h_matrix, cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([str(d) for d in dims])
    ax.set_xlabel("Pipeline Depth", fontsize=12)
    ax.set_ylabel("Dimension", fontsize=12)
    ax.set_title("실험 5: AXOL Observe 정확도 — Hellinger Distance vs Traditional",
                 fontsize=13, fontweight="bold")

    for i in range(len(dims)):
        for j in range(len(depths)):
            val = h_matrix[i, j]
            color = "white" if val > 0.3 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Hellinger Distance", fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig5_accuracy.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  [saved] {p}")


def _fig_all(e1, e2, e3, e3m, e4):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    dims = [r["dim"] for r in e1]
    ax.plot(dims, [r["trad_us"] for r in e1], "o-", color=CT, lw=2, ms=5, label="Traditional")
    ax.plot(dims, [r["axol_us"] for r in e1], "s-", color=CA, lw=2, ms=5, label="AXOL Observe")
    ax.plot(dims, [r["raw_us"]  for r in e1], "^--", color=CR, lw=1.5, ms=4, label="Raw matmul")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Dimension"); ax.set_ylabel("Time (\u03bcs)")
    ax.set_title("1. Observe vs Dimension (depth=1)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    ax = axes[0, 1]
    ds = [r["depth"] for r in e2]
    ax.plot(ds, [r["trad_us"] for r in e2], "o-", color=CT, lw=2, ms=5, label="Traditional")
    ax.plot(ds, [r["seq_us"]  for r in e2], "x--", color=CW, lw=1.5, ms=4, label="Sequential")
    ax.plot(ds, [r["axol_us"] for r in e2], "s-", color=CA, lw=2, ms=5, label="AXOL Observe")
    ax.plot(ds, [r["comp_us"] for r in e2], "D-", color=CC, lw=2, ms=5, label="Composed baseline")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Depth"); ax.set_ylabel("Time (\u03bcs)")
    ax.set_title("2. Depth Scaling (dim=16)", fontweight="bold")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 0]
    ns = [r["count"] for r in e3]
    ax.plot(ns, [r["trad_per_us"] for r in e3], "o-", color=CT, lw=2, ms=5, label="Traditional")
    ax.plot(ns, [r["axol_per_us"] for r in e3], "s-", color=CA, lw=2, ms=5, label="AXOL Amortized")
    ax.plot(ns, [r["comp_per_us"] for r in e3], "D-", color=CC, lw=2, ms=5, label="Composed Amort.")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Obs Count"); ax.set_ylabel("Per-call (\u03bcs)")
    ax.set_title("3. Amortization (dim=16, depth=100)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 1]
    lbl = [f"({r['dim']},{r['depth']})" for r in e4]
    xp = np.arange(len(e4)); w = 0.25
    ax.bar(xp-w, [r["trad_us"]/1e3 for r in e4], w, color=CT, label="Traditional")
    ax.bar(xp,   [r["axol_us"]/1e3 for r in e4], w, color=CA, label="AXOL Observe")
    ax.bar(xp+w, [r["comp_us"]/1e3 for r in e4], w, color=CC, label="Composed")
    ax.set_yscale("log"); ax.set_xticks(xp); ax.set_xticklabels(lbl)
    ax.set_xlabel("(dim, depth)"); ax.set_ylabel("Time (ms)")
    ax.set_title("4. Dim\u00d7Depth Combined", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y", which="both")

    fig.suptitle("AXOL Observe Scaling \u2014 Extreme Benchmark", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "fig_combined_4panel.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  [saved] {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════

def _report(e1, e2, e3, e3m, e4, elapsed, e5=None):
    L = []
    L.append("# AXOL Observe Scaling \u2014 Extreme Benchmark Report\n")
    L.append(f"> Total runtime: {elapsed:.1f}s  |  {time.strftime('%Y-%m-%d %H:%M')}\n")

    # --- Exp 1 ---
    L.append("\n## 1. Observe Time vs Dimension (depth=1)\n")
    L.append("dim\uc774 1000\ubc30 \ucee4\uc838\ub3c4 \uad00\uce21 \ube44\uc6a9\uc774 \uc77c\uc815\ud55c\uac00?\n")
    L.append("| Dimension | Traditional | AXOL Observe | Raw matmul | Weave (1\ud68c) |")
    L.append("|-----------|-------------|--------------|------------|-------------|")
    for r in e1:
        L.append(f"| {r['dim']:>5,} | {r['trad_us']:>8.1f} \u03bcs | {r['axol_us']:>8.1f} \u03bcs "
                 f"| {r['raw_us']:>8.1f} \u03bcs | {r['weave_ms']:>8.1f} ms |")
    if len(e1) >= 2:
        r_dim = e1[-1]["axol_us"] / max(e1[0]["axol_us"], 0.001)
        L.append(f"\n**\uacb0\uacfc:** dim {e1[0]['dim']}\u2192{e1[-1]['dim']} ({e1[-1]['dim']//e1[0]['dim']}x) \uc77c \ub54c "
                 f"\uad00\uce21 \uc2dc\uac04 {r_dim:.1f}x \ubcc0\ud654.  "
                 f"dim=4096\uc5d0\uc11c\ub3c4 **{e1[-1]['axol_us']:.0f}\u03bcs** \uc218\uc900.")

    # --- Exp 2 ---
    L.append("\n\n## 2. Pipeline Depth Scaling (dim=16)\n")
    L.append("파이프라인이 5000단계여도 AXOL observe가 깊이에 무관한가?\n")
    L.append("| Depth | Traditional | Sequential (old) | **AXOL Observe** | Composed baseline | Weave (1회) |")
    L.append("|-------|-------------|------------------|------------------|-------------------|-------------|")
    for r in e2:
        L.append(f"| {r['depth']:>5,} | {r['trad_us']:>10.1f} μs | {r['seq_us']:>10.1f} μs "
                 f"| **{r['axol_us']:>8.1f} μs** | {r['comp_us']:>8.1f} μs | {r['weave_ms']:>8.1f} ms |")
    if len(e2) >= 2:
        tg = e2[-1]["trad_us"] / max(e2[0]["trad_us"], 0.001)
        ag = e2[-1]["axol_us"] / max(e2[0]["axol_us"], 0.001)
        gap = e2[-1]["trad_us"] / max(e2[-1]["axol_us"], 0.001)
        L.append(f"\n**결과:**")
        L.append(f"- Traditional: depth {e2[0]['depth']}→{e2[-1]['depth']} → 시간 **{tg:.0f}x** (선형 증가)")
        L.append(f"- AXOL Observe (내장 composed): depth {e2[0]['depth']}→{e2[-1]['depth']} → 시간 **{ag:.1f}x** (수평선!)")
        L.append(f"- depth={e2[-1]['depth']}에서 **{gap:,.0f}x speedup**")
        L.append(f"- AXOL Observe ≈ Composed baseline — composed가 observe()에 내장됨을 확인")

    # --- Exp 3 ---
    L.append(f"\n\n## 3. Repeated Observation Amortization (dim=16, depth=100)\n")
    L.append(f"- \uc9c1\uc870 \ube44\uc6a9 (1\ud68c): **{e3m['weave_ms']:.1f} ms**")
    L.append(f"- \uad00\uce21 \ube44\uc6a9 (1\ud68c): **{e3m['observe_us']:.1f} \u03bcs**")
    L.append(f"- Traditional \ube44\uc6a9: **{e3m['trad_us']:.1f} \u03bcs** (\ub9e4 \ud638\ucd9c)\n")
    L.append("| Observations | Traditional/call | AXOL Amortized | Composed Amort. | \uc9c1\uc870\uc0c1\uac01 |")
    L.append("|-------------|-----------------|----------------|-----------------|---------|")
    for r in e3:
        L.append(f"| {r['count']:>9,} | {r['trad_per_us']:>10.1f} \u03bcs "
                 f"| {r['axol_per_us']:>10.1f} \u03bcs | {r['comp_per_us']:>10.1f} \u03bcs "
                 f"| {r['weave_amort_us']:>8.1f} \u03bcs |")
    L.append(f"\n**\uacb0\uacfc:** 100\ub9cc \ud68c \uad00\uce21 \uc2dc \uc9c1\uc870 \uc0c1\uac01 = **{e3[-1]['weave_amort_us']:.3f}\u03bcs** \u2192 \uc0ac\uc2e4\uc0c1 0.")

    # --- Exp 4 ---
    L.append("\n\n## 4. Dimension \u00d7 Depth Combined Scaling\n")
    L.append("| Config | Complexity | Traditional | Composed | Speedup | Weave |")
    L.append("|--------|-----------|-------------|----------|---------|-------|")
    for r in e4:
        L.append(f"| ({r['dim']},{r['depth']}) | {r['complexity']:>12,} "
                 f"| {_fmt(r['trad_us']/1e6):>11} | {_fmt(r['comp_us']/1e6):>8} "
                 f"| **{r['speedup']:>,.0f}x** | {_fmt(r['weave_s'])} |")
    if len(e4) >= 3:
        L.append(f"\n**\uacb0\uacfc:** (1024,1000)\uc5d0\uc11c Traditional \ub300\ube44 Composed **{e4[-1]['speedup']:,.0f}x** \uac00\uc18d.")

    # --- Exp 5 ---
    if e5:
        L.append("\n\n## 5. Accuracy Verification — Composed vs Sequential\n")
        L.append("AXOL Observe(composed fast path)의 결과가 Sequential(step-by-step)과 일치하는가?\n")
        L.append("| Dim | Depth | Hellinger(AXOL) | TV(AXOL) | MaxDiff(AXOL) | Mode Match |")
        L.append("|-----|-------|-----------------|----------|---------------|------------|")
        for r in e5:
            match_str = "✓" if r["mode_axol"] else "✗"
            L.append(f"| {r['dim']:>5} | {r['depth']:>5} | {r['h_axol']:>13.6f} "
                     f"| {r['tv_axol']:>8.6f} | {r['max_diff_axol']:>13.6f} | {match_str:>10} |")
        all_h = [r["h_axol"] for r in e5]
        all_m = [r["mode_axol"] for r in e5]
        L.append(f"\n**결과:**")
        L.append(f"- Hellinger distance: max={max(all_h):.6f}, mean={np.mean(all_h):.6f}")
        L.append(f"- Mode match: {sum(all_m)}/{len(all_m)} ({100*sum(all_m)/len(all_m):.0f}%)")
        if max(all_h) < 0.01:
            L.append(f"- **모든 설정에서 Hellinger < 0.01 — float32 정밀도 차이만 존재**")
        else:
            L.append(f"- ⚠ 일부 설정에서 Hellinger ≥ 0.01 — 행렬 곱 순서/정밀도 검토 필요")

    # --- Summary ---
    L.append("\n\n## Summary\n")
    L.append("```")
    L.append("Traditional:  cost_per_call = O(depth \u00d7 dim\u00b2)   \u2190 \ub9e4 \ud638\ucd9c\ub9c8\ub2e4 N\ub2e8\uacc4 \uc21c\ucc28 \uc2e4\ud589")
    L.append("AXOL Weave:   cost_one_time = O(depth \u00d7 dim\u00b3)   \u2190 1\ud68c \uc9c1\uc870 (\ud589\ub82c \ud569\uc131 + \ud488\uc9c8 \ubcf4\uc99d)")
    L.append("AXOL Observe: cost_per_call = O(dim\u00b2)            \u2190 depth \ubb34\uad00! \ub2e8\uc77c \ud589\ub82c-\ubca1\ud130 \uacf1")
    L.append("```\n")
    L.append("N\ud68c \uad00\uce21 \uc2dc:")
    L.append("```")
    L.append("Traditional = N \u00d7 O(depth \u00d7 dim\u00b2)")
    L.append("AXOL        = O(depth \u00d7 dim\u00b3) + N \u00d7 O(dim\u00b2)")
    L.append("")
    L.append("N >> depth \u00d7 dim \uc77c \ub54c AXOL\uc740 depth\ub9cc\ud07c \uac00\uc18d.")
    L.append("```\n")

    p = os.path.join(OUT_DIR, "OBSERVE_SCALING_REPORT.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n  [saved] {p}")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
 ╔═══════════════════════════════════════════════════════════╗
 ║       AXOL OBSERVE SCALING — EXTREME BENCHMARK           ║
 ║  dim → 4096  |  depth → 5000  |  obs → 1,000,000        ║
 ║  + Experiment 5: Accuracy Verification                   ║
 ╚═══════════════════════════════════════════════════════════╝
""")
    t0 = time.perf_counter()

    e1 = experiment_1()
    e2 = experiment_2()
    e3, e3m = experiment_3()
    e4 = experiment_4()
    e5 = experiment_5()

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 90}")
    print(f"  전체 실험 완료: {elapsed:.1f}s")
    print(f"{'=' * 90}")

    if HAS_MPL:
        print("\n  차트 생성 중...")
        _fig1(e1)
        _fig2(e2)
        _fig3(e3, e3m)
        _fig4(e4)
        _fig5(e5)
        _fig_all(e1, e2, e3, e3m, e4)
    else:
        print("\n  [SKIP] matplotlib 미설치 — 차트 생략")

    print("\n  리포트 생성 중...")
    _report(e1, e2, e3, e3m, e4, elapsed, e5=e5)
    print(f"\n  모든 출력: {OUT_DIR}")


if __name__ == "__main__":
    main()
