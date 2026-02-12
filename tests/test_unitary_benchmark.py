"""Distill vs Hybrid vs Unitary vs Koopman vs Fallback -- lightweight benchmark."""

from __future__ import annotations

import time
import numpy as np
import pytest

from axol.core.types import FloatVec
from axol.core import operations as ops
from axol.quantum.koopman import lifted_dim
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe


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


class TestBenchmark:
    """Single test: prints all benchmark tables."""

    def test_full_benchmark(self):
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
        print("=" * 140)
        print("  A. Accuracy -- Distill vs Hybrid vs Unitary vs Koopman vs Fallback (dim=4, depth=3)")
        print("=" * 140)
        print(f"  {'Func':<12}  {'D H-dist':>8}  {'H H-dist':>8}  {'U H-dist':>8}  {'K H-dist':>8}"
              f"  {'D argmax':>8}  {'H argmax':>8}  {'U argmax':>8}  {'K argmax':>8}  {'Winner':>8}")
        print("-" * 140)

        for fn_name, fn in functions:
            decl = _build_chain(3, dim=dim, transform_fn=fn)
            tap_d = weave(decl, seed=42, nonlinear_method="distill")
            tap_h = weave(decl, seed=42, nonlinear_method="hybrid")
            tap_u = weave(decl, seed=42, nonlinear_method="unitary")
            tap_k = weave(decl, seed=42, nonlinear_method="koopman")
            tap_f = weave(decl, seed=42, optimize=False)

            hd, hh, hu, hk, md, mh, mu, mk = 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0
            N = 30
            for _ in range(N):
                x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
                od = observe(tap_d, {"x": x})
                oh = observe(tap_h, {"x": x})
                ou = observe(tap_u, {"x": x})
                ok = observe(tap_k, {"x": x})
                of = observe(tap_f, {"x": x})
                hd += hellinger_distance(od.probabilities.data, of.probabilities.data)
                hh += hellinger_distance(oh.probabilities.data, of.probabilities.data)
                hu += hellinger_distance(ou.probabilities.data, of.probabilities.data)
                hk += hellinger_distance(ok.probabilities.data, of.probabilities.data)
                if od.value_index == of.value_index: md += 1
                if oh.value_index == of.value_index: mh += 1
                if ou.value_index == of.value_index: mu += 1
                if ok.value_index == of.value_index: mk += 1

            hd /= N; hh /= N; hu /= N; hk /= N
            best = min(hd, hh, hu, hk)
            winner = ("DISTILL" if best == hd else
                      "HYBRID" if best == hh else
                      ("UNITARY" if best == hu else "KOOPMAN"))
            print(f"  {fn_name:<12}  {hd:>8.4f}  {hh:>8.4f}  {hu:>8.4f}  {hk:>8.4f}"
                  f"  {md/N:>7.0%}  {mh/N:>7.0%}  {mu/N:>7.0%}  {mk/N:>7.0%}  {winner:>8}")

        print("=" * 140)

        # ============================================================
        # B. Depth Scaling (sigmoid, dim=4)
        # ============================================================
        depths = [1, 2, 5, 10, 20]
        print()
        print("=" * 155)
        print("  B. Depth Scaling -- sigmoid, dim=4")
        print("=" * 155)
        print(f"  {'Depth':>6}  {'D H-dist':>9}  {'H H-dist':>9}  {'U H-dist':>9}  {'K H-dist':>9}"
              f"  {'D argmax':>9}  {'H argmax':>9}  {'U argmax':>9}  {'K argmax':>9}"
              f"  {'D(us)':>7}  {'H(us)':>7}  {'U(us)':>7}  {'K(us)':>7}  {'F(us)':>7}")
        print("-" * 155)

        for depth in depths:
            decl = _build_chain(depth, dim=dim, transform_fn=_sigmoid)
            tap_d = weave(decl, seed=42, nonlinear_method="distill")
            tap_h = weave(decl, seed=42, nonlinear_method="hybrid")
            tap_u = weave(decl, seed=42, nonlinear_method="unitary")
            tap_k = weave(decl, seed=42, nonlinear_method="koopman")
            tap_f = weave(decl, seed=42, optimize=False)

            hd, hh, hu, hk, md, mh, mu, mk = 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0
            N = 20
            for _ in range(N):
                x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
                od = observe(tap_d, {"x": x})
                oh = observe(tap_h, {"x": x})
                ou = observe(tap_u, {"x": x})
                ok = observe(tap_k, {"x": x})
                of = observe(tap_f, {"x": x})
                hd += hellinger_distance(od.probabilities.data, of.probabilities.data)
                hh += hellinger_distance(oh.probabilities.data, of.probabilities.data)
                hu += hellinger_distance(ou.probabilities.data, of.probabilities.data)
                hk += hellinger_distance(ok.probabilities.data, of.probabilities.data)
                if od.value_index == of.value_index: md += 1
                if oh.value_index == of.value_index: mh += 1
                if ou.value_index == of.value_index: mu += 1
                if ok.value_index == of.value_index: mk += 1
            hd /= N; hh /= N; hu /= N; hk /= N

            # Timing (30 iters)
            x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
            for tap in (tap_d, tap_h, tap_u, tap_k, tap_f):
                observe(tap, {"x": x})
            times = {}
            for label, tap in [("D", tap_d), ("H", tap_h), ("U", tap_u), ("K", tap_k), ("F", tap_f)]:
                t0 = time.perf_counter()
                for _ in range(30):
                    observe(tap, {"x": x})
                times[label] = (time.perf_counter() - t0) / 30 * 1e6

            print(f"  {depth:>6}  {hd:>9.4f}  {hh:>9.4f}  {hu:>9.4f}  {hk:>9.4f}"
                  f"  {md/N:>8.0%}  {mh/N:>8.0%}  {mu/N:>8.0%}  {mk/N:>8.0%}"
                  f"  {times['D']:>7.1f}  {times['H']:>7.1f}  {times['U']:>7.1f}  {times['K']:>7.1f}  {times['F']:>7.1f}")

        print("=" * 155)

        # ============================================================
        # C. Memory: matrix size comparison
        # ============================================================
        print()
        print("=" * 78)
        print("  C. Matrix Size -- Distill/Hybrid/Unitary (dxd) vs Koopman (lifted x lifted)")
        print("=" * 78)
        print(f"  {'dim':>5}  {'Distill/H/U':>15}  {'Koopman':>12}  {'Ratio':>8}")
        print("-" * 78)
        for d in [1, 4, 8, 16, 32, 64]:
            u = d * d
            ld = lifted_dim(d, degree=2)
            k = ld * ld
            print(f"  {d:>5}  {d}x{d}={u:>5}         {ld}x{ld}={k:>6}  {k/u:>7.1f}x")
        print("=" * 78)

        # ============================================================
        # D. Weave cost (compilation time)
        # ============================================================
        print()
        print("=" * 100)
        print("  D. Weave Cost -- sigmoid, dim=4")
        print("=" * 100)
        print(f"  {'Depth':>6}  {'Distill(ms)':>12}  {'Hybrid(ms)':>12}  {'Unitary(ms)':>12}  {'Koopman(ms)':>12}  {'NoOpt(ms)':>11}")
        print("-" * 100)
        for depth in [3, 10, 20]:
            decl = _build_chain(depth, dim=4, transform_fn=_sigmoid)
            t0 = time.perf_counter()
            weave(decl, seed=42, nonlinear_method="distill")
            td = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            weave(decl, seed=42, nonlinear_method="hybrid")
            th = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            weave(decl, seed=42, nonlinear_method="unitary")
            tu = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            weave(decl, seed=42, nonlinear_method="koopman")
            tk = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            weave(decl, seed=42, optimize=False)
            tf = (time.perf_counter() - t0) * 1e3
            print(f"  {depth:>6}  {td:>12.1f}  {th:>12.1f}  {tu:>12.1f}  {tk:>12.1f}  {tf:>11.1f}")
        print("=" * 100)
