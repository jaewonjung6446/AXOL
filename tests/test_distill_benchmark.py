"""End-to-end distillation prototype + benchmark.

Idea: instead of per-step estimation + composition,
run the full fallback pipeline N times, collect (input, output) pairs,
fit a single dim x dim matrix via lstsq.

Measures argmax agreement and H-dist vs fallback ground truth.
"""

from __future__ import annotations

import time
import numpy as np

from axol.core.types import FloatVec, TransMatrix
from axol.core import operations as ops
from axol.core.program import Program, run_program
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe
from axol.quantum.unitary import nearest_unitary


def hellinger_distance(p, q):
    p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    q = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
    sp, sq = p.sum(), q.sum()
    if sp > 0: p = p / sp
    if sq > 0: q = q / sq
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def _relu(x):
    return np.maximum(x, 0.0)

def _hard_threshold(x):
    return np.where(x > 0, 1.0, 0.0)


def _build_chain(depth, dim=4, transform_fn=None):
    builder = DeclarationBuilder(f"chain_d{depth}")
    builder.input("x", dim)
    prev = "x"
    for i in range(depth):
        node = f"h{i}"
        builder.relate(node, [prev], RelationKind.PROPORTIONAL, transform_fn=transform_fn)
        prev = node
    builder.output(prev)
    return builder.build()


def _run_fallback(tapestry, input_vec):
    """Run the full internal program (fallback path) and return output FloatVec."""
    program = tapestry._internal_program
    state = program.initial_state.copy()
    state["x"] = input_vec
    injected = Program(
        name=program.name,
        initial_state=state,
        transitions=program.transitions,
    )
    result = run_program(injected)
    output_name = tapestry.output_names[0]
    if output_name in result.final_state:
        v = result.final_state[output_name]
        if not isinstance(v, FloatVec):
            v = FloatVec(data=v.data.astype(np.float32))
        return v
    return None


def distill_matrix(tapestry, dim, n_samples=500, seed=42):
    """End-to-end distillation: run fallback N times, fit single dim x dim matrix."""
    rng = np.random.default_rng(seed)
    n_samples = max(n_samples, dim + 10)

    X = rng.standard_normal((n_samples, dim)).astype(np.float64) * 0.5
    Y = np.empty_like(X)

    for i in range(n_samples):
        x_fv = FloatVec(data=X[i].astype(np.float32))
        out = _run_fallback(tapestry, x_fv)
        Y[i] = out.data.astype(np.float64)

    A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    A = np.nan_to_num(A, nan=0.0, posinf=100.0, neginf=-100.0)
    return TransMatrix(data=A.astype(np.float32))


def distill_observe(distilled_matrix, input_vec):
    """Observe using the distilled matrix -- single mat-mul + Born rule."""
    value = ops.transform(input_vec, distilled_matrix)
    probs = ops.measure(value)
    value_index = int(np.argmax(probs.data))
    return value, probs, value_index


class TestDistillBenchmark:

    def test_distill_benchmark(self):
        rng = np.random.default_rng(42)
        dim = 4

        # ============================================================
        # A. Accuracy per function (depth=3)
        # ============================================================
        functions = [
            ("Sigmoid", _sigmoid),
            ("ReLU", _relu),
            ("HardThresh", _hard_threshold),
        ]
        print()
        print("=" * 100)
        print("  A. Accuracy -- Distilled vs Koopman vs Hybrid (dim=4, depth=3)")
        print("=" * 100)
        print(f"  {'Func':<12}  {'D H-dist':>8}  {'K H-dist':>8}  {'H H-dist':>8}"
              f"  {'D argmax':>8}  {'K argmax':>8}  {'H argmax':>8}")
        print("-" * 100)

        for fn_name, fn in functions:
            decl = _build_chain(3, dim=dim, transform_fn=fn)
            tap_f = weave(decl, seed=42, optimize=False)
            tap_k = weave(decl, seed=42, nonlinear_method="koopman")
            tap_h = weave(decl, seed=42, nonlinear_method="hybrid")

            # Distill
            distilled = distill_matrix(tap_f, dim, n_samples=500, seed=42)

            hd, hk, hh, md, mk, mh = 0.0, 0.0, 0.0, 0, 0, 0
            N = 30
            for _ in range(N):
                x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
                of = observe(tap_f, {"x": x})
                ok = observe(tap_k, {"x": x})
                oh = observe(tap_h, {"x": x})
                _, pd, id_ = distill_observe(distilled, x)

                hd += hellinger_distance(pd.data, of.probabilities.data)
                hk += hellinger_distance(ok.probabilities.data, of.probabilities.data)
                hh += hellinger_distance(oh.probabilities.data, of.probabilities.data)
                if id_ == of.value_index: md += 1
                if ok.value_index == of.value_index: mk += 1
                if oh.value_index == of.value_index: mh += 1

            hd /= N; hk /= N; hh /= N
            print(f"  {fn_name:<12}  {hd:>8.4f}  {hk:>8.4f}  {hh:>8.4f}"
                  f"  {md/N:>7.0%}  {mk/N:>7.0%}  {mh/N:>7.0%}")

        print("=" * 100)

        # ============================================================
        # B. Depth Scaling (sigmoid)
        # ============================================================
        depths = [1, 2, 5, 10, 20]
        print()
        print("=" * 110)
        print("  B. Depth Scaling -- sigmoid, dim=4")
        print("=" * 110)
        print(f"  {'Depth':>6}  {'D H-dist':>9}  {'K H-dist':>9}"
              f"  {'D argmax':>9}  {'K argmax':>9}"
              f"  {'D(us)':>7}  {'K(us)':>7}  {'F(us)':>7}"
              f"  {'Distill(ms)':>12}")
        print("-" * 110)

        for depth in depths:
            decl = _build_chain(depth, dim=dim, transform_fn=_sigmoid)
            tap_f = weave(decl, seed=42, optimize=False)
            tap_k = weave(decl, seed=42, nonlinear_method="koopman")

            # Distill (timed)
            t0 = time.perf_counter()
            distilled = distill_matrix(tap_f, dim, n_samples=500, seed=42)
            t_distill = (time.perf_counter() - t0) * 1e3

            hd, hk, md, mk = 0.0, 0.0, 0, 0
            N = 20
            for _ in range(N):
                x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
                of = observe(tap_f, {"x": x})
                ok = observe(tap_k, {"x": x})
                _, pd, id_ = distill_observe(distilled, x)

                hd += hellinger_distance(pd.data, of.probabilities.data)
                hk += hellinger_distance(ok.probabilities.data, of.probabilities.data)
                if id_ == of.value_index: md += 1
                if ok.value_index == of.value_index: mk += 1
            hd /= N; hk /= N

            # Timing observe (30 iters)
            x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
            observe(tap_k, {"x": x})
            distill_observe(distilled, x)
            observe(tap_f, {"x": x})

            times = {}
            t0 = time.perf_counter()
            for _ in range(30):
                distill_observe(distilled, x)
            times["D"] = (time.perf_counter() - t0) / 30 * 1e6
            t0 = time.perf_counter()
            for _ in range(30):
                observe(tap_k, {"x": x})
            times["K"] = (time.perf_counter() - t0) / 30 * 1e6
            t0 = time.perf_counter()
            for _ in range(30):
                observe(tap_f, {"x": x})
            times["F"] = (time.perf_counter() - t0) / 30 * 1e6

            print(f"  {depth:>6}  {hd:>9.4f}  {hk:>9.4f}"
                  f"  {md/N:>8.0%}  {mk/N:>8.0%}"
                  f"  {times['D']:>7.1f}  {times['K']:>7.1f}  {times['F']:>7.1f}"
                  f"  {t_distill:>12.1f}")

        print("=" * 110)

        # ============================================================
        # C. Sample count vs accuracy (depth=5, sigmoid)
        # ============================================================
        print()
        print("=" * 70)
        print("  C. Sample Count vs Accuracy -- sigmoid, dim=4, depth=5")
        print("=" * 70)
        print(f"  {'Samples':>8}  {'H-dist':>8}  {'Argmax':>8}  {'Distill(ms)':>12}")
        print("-" * 70)

        decl = _build_chain(5, dim=dim, transform_fn=_sigmoid)
        tap_f = weave(decl, seed=42, optimize=False)

        for n_samp in [50, 100, 200, 500, 1000]:
            t0 = time.perf_counter()
            distilled = distill_matrix(tap_f, dim, n_samples=n_samp, seed=42)
            t_d = (time.perf_counter() - t0) * 1e3

            hd, md = 0.0, 0
            N = 30
            for _ in range(N):
                x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
                of = observe(tap_f, {"x": x})
                _, pd, id_ = distill_observe(distilled, x)
                hd += hellinger_distance(pd.data, of.probabilities.data)
                if id_ == of.value_index: md += 1
            hd /= N
            print(f"  {n_samp:>8}  {hd:>8.4f}  {md/N:>7.0%}  {t_d:>12.1f}")

        print("=" * 70)
