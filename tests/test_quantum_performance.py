"""Quantum module performance measurement and report generation.

Measures three axes:
  1. Accuracy  — Lyapunov/Fractal estimation vs known systems, Omega/Phi correctness,
                 composition rules, observation consistency
  2. Speed     — Weave/Observe/Lyapunov/Fractal/DSL timing at various scales
  3. Token Efficiency — Quantum DSL vs equivalent Python code

Generates QUANTUM_PERFORMANCE_REPORT.md at project root.
"""

import os
import time
import textwrap
import math

import pytest
import numpy as np
import tiktoken

from axol.core.types import FloatVec, TransMatrix, StateBundle
from axol.core.program import Program, Transition, TransformOp, run_program

from axol.quantum.types import SuperposedState, Attractor, Observation
from axol.quantum.declare import DeclarationBuilder, RelationKind, QualityTarget
from axol.quantum.lyapunov import (
    estimate_lyapunov, lyapunov_spectrum, omega_from_lyapunov, omega_from_observations,
)
from axol.quantum.fractal import (
    estimate_fractal_dim, phi_from_fractal, phi_from_entropy,
)
from axol.quantum.cost import estimate_cost
from axol.quantum.compose import compose_serial, compose_parallel
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe, reobserve
from axol.quantum.dsl import parse_quantum


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

_results: dict[str, object] = {}
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _time_fn(fn, n=20):
    """Time a function call, return average seconds."""
    # Warmup
    fn()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n


# ═══════════════════════════════════════════════════════════════════════════
# 1. ACCURACY — Lyapunov Estimation
# ═══════════════════════════════════════════════════════════════════════════

class TestAccuracyLyapunov:
    """Lyapunov exponent estimation accuracy against known systems."""

    def test_01_known_systems(self, capsys):
        rows = []

        # (a) Contracting: M = 0.5*I => lambda = ln(0.5) ≈ -0.693
        M_contract = TransMatrix(data=np.eye(4, dtype=np.float32) * 0.5)
        lam = estimate_lyapunov(M_contract, steps=200)
        expected = math.log(0.5)
        error = abs(lam - expected)
        rows.append(("Contracting (0.5*I, dim=4)", expected, lam, error))

        # (b) Expanding: M = 2.0*I => lambda = ln(2) ≈ 0.693
        M_expand = TransMatrix(data=np.eye(4, dtype=np.float32) * 2.0)
        lam = estimate_lyapunov(M_expand, steps=200)
        expected = math.log(2.0)
        error = abs(lam - expected)
        rows.append(("Expanding (2.0*I, dim=4)", expected, lam, error))

        # (c) Identity: lambda = 0
        M_id = TransMatrix.identity(4)
        lam = estimate_lyapunov(M_id, steps=200)
        expected = 0.0
        error = abs(lam - expected)
        rows.append(("Identity (dim=4)", expected, lam, error))

        # (d) Lorenz-like: construct with known eigenvalues
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(A)
        D = np.diag([np.exp(0.91), np.exp(-0.5), np.exp(-1.2)])
        M_lorenz = TransMatrix(data=(Q @ D @ Q.T).astype(np.float32))
        lam = estimate_lyapunov(M_lorenz, steps=500)
        expected = 0.91
        error = abs(lam - expected)
        rows.append(("Lorenz-like (lambda~0.91, dim=3)", expected, lam, error))

        # (e) Strongly contracting: 0.01*I => lambda = ln(0.01) ≈ -4.605
        M_strong = TransMatrix(data=np.eye(4, dtype=np.float32) * 0.01)
        lam = estimate_lyapunov(M_strong, steps=200)
        expected = math.log(0.01)
        error = abs(lam - expected)
        rows.append(("Strong contraction (0.01*I)", expected, lam, error))

        # (f) Rotation (orthogonal): lambda ≈ 0
        theta = 0.5
        R = np.array([[math.cos(theta), -math.sin(theta)],
                       [math.sin(theta), math.cos(theta)]], dtype=np.float32)
        M_rot = TransMatrix(data=R)
        lam = estimate_lyapunov(M_rot, steps=500)
        expected = 0.0
        error = abs(lam - expected)
        rows.append(("Pure rotation (2D)", expected, lam, error))

        _results["lyapunov_accuracy"] = rows

        with capsys.disabled():
            print(f"\n{'='*78}")
            print("  1.1 Lyapunov Estimation Accuracy")
            print(f"{'='*78}")
            print(f"  {'System':<38} {'Expected':>10} {'Measured':>10} {'Error':>10}")
            print(f"{'-'*78}")
            for name, exp, meas, err in rows:
                print(f"  {name:<38} {exp:>10.4f} {meas:>10.4f} {err:>10.4f}")
            avg_err = sum(r[3] for r in rows) / len(rows)
            print(f"{'-'*78}")
            print(f"  {'Average absolute error':<38} {'':>10} {'':>10} {avg_err:>10.4f}")
            print(f"{'='*78}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. ACCURACY — Fractal Dimension Estimation
# ═══════════════════════════════════════════════════════════════════════════

class TestAccuracyFractal:
    def test_02_known_geometries(self, capsys):
        rows = []
        rng = np.random.default_rng(42)

        # (a) Single point cluster: D ≈ 0
        pts = np.tile([0.5, 0.5], 200)
        noise = rng.standard_normal(len(pts)) * 0.001
        d = estimate_fractal_dim(FloatVec.from_list((pts + noise).tolist()), phase_space_dim=2)
        rows.append(("Point cluster (2D)", 0.0, d, abs(d - 0.0)))

        # (b) Line segment: D ≈ 1
        t = np.linspace(0, 1, 500)
        pts = np.column_stack([t, t * 0.7]).flatten()
        d = estimate_fractal_dim(FloatVec.from_list(pts.tolist()), phase_space_dim=2)
        rows.append(("Line segment (2D)", 1.0, d, abs(d - 1.0)))

        # (c) Filled square: D ≈ 2
        pts = rng.uniform(0, 1, (1000, 2)).flatten()
        d = estimate_fractal_dim(FloatVec.from_list(pts.tolist()), phase_space_dim=2)
        rows.append(("Filled square (2D)", 2.0, d, abs(d - 2.0)))

        # (d) Filled cube: D ≈ 3
        pts = rng.uniform(0, 1, (1000, 3)).flatten()
        d = estimate_fractal_dim(FloatVec.from_list(pts.tolist()), phase_space_dim=3)
        rows.append(("Filled cube (3D)", 3.0, d, abs(d - 3.0)))

        # (e) Cantor-like set: D ≈ 0.63 (ln2/ln3)
        # Generate Cantor set approximation
        cantor = [0.0, 1.0]
        for _ in range(8):
            new = []
            for x in cantor:
                new.append(x / 3.0)
                new.append(x / 3.0 + 2.0 / 3.0)
            cantor = new
        cantor_pts = np.array([[x, 0.0] for x in cantor]).flatten()
        d = estimate_fractal_dim(FloatVec.from_list(cantor_pts.tolist()), phase_space_dim=2)
        cantor_expected = math.log(2) / math.log(3)  # ≈ 0.631
        rows.append(("Cantor-like set", cantor_expected, d, abs(d - cantor_expected)))

        # (f) Sierpinski-like: D ≈ 1.585 (ln3/ln2)
        # Approximate via random IFS
        pts_sierpinski = []
        x, y = 0.0, 0.0
        vertices = [(0, 0), (1, 0), (0.5, math.sqrt(3)/2)]
        for _ in range(5000):
            vi = rng.integers(0, 3)
            x = (x + vertices[vi][0]) / 2.0
            y = (y + vertices[vi][1]) / 2.0
            pts_sierpinski.extend([x, y])
        d = estimate_fractal_dim(FloatVec.from_list(pts_sierpinski), phase_space_dim=2)
        sierpinski_expected = math.log(3) / math.log(2)  # ≈ 1.585
        rows.append(("Sierpinski triangle", sierpinski_expected, d, abs(d - sierpinski_expected)))

        _results["fractal_accuracy"] = rows

        with capsys.disabled():
            print(f"\n{'='*78}")
            print("  1.2 Fractal Dimension Estimation Accuracy")
            print(f"{'='*78}")
            print(f"  {'Geometry':<30} {'Expected':>10} {'Measured':>10} {'Error':>10}")
            print(f"{'-'*78}")
            for name, exp, meas, err in rows:
                print(f"  {name:<30} {exp:>10.3f} {meas:>10.3f} {err:>10.3f}")
            avg_err = sum(r[3] for r in rows) / len(rows)
            print(f"{'-'*78}")
            print(f"  {'Average absolute error':<30} {'':>10} {'':>10} {avg_err:>10.3f}")
            print(f"{'='*78}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. ACCURACY — Omega/Phi Formula Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestAccuracyOmegaPhi:
    def test_03_omega_phi_formulas(self, capsys):
        rows_omega = []
        rows_phi = []

        # Omega = 1/(1+max(lambda,0))
        for lam in [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 0.91, 1.0, 2.0, 5.0, 10.0]:
            expected = 1.0 / (1.0 + max(lam, 0.0))
            actual = omega_from_lyapunov(lam)
            err = abs(actual - expected)
            rows_omega.append((lam, expected, actual, err))

        # Phi = 1/(1+D/D_max)
        for (d, d_max) in [(0.0, 4), (1.0, 4), (2.0, 4), (4.0, 4), (2.06, 3), (0.0, 1), (10.0, 10)]:
            expected = 1.0 / (1.0 + d / d_max) if d_max > 0 else 1.0
            actual = phi_from_fractal(d, d_max)
            err = abs(actual - expected)
            rows_phi.append((d, d_max, expected, actual, err))

        # Phi from entropy
        rows_entropy = []
        test_cases = [
            ("Delta [0,0,1,0]", FloatVec.from_list([0.0, 0.0, 1.0, 0.0]), 1.0),
            ("Uniform [.25]*4", FloatVec.from_list([0.25, 0.25, 0.25, 0.25]), 0.0),
            ("Peaked [.01,.01,.97,.01]", FloatVec.from_list([0.01, 0.01, 0.97, 0.01]), None),
            ("Binary [.5,.5,0,0]", FloatVec.from_list([0.5, 0.5, 0.0, 0.0]), None),
            ("Uniform [.125]*8", FloatVec.from_list([0.125]*8), 0.0),
            ("Delta 1-of-8", FloatVec.from_list([0,0,0,0,0,1,0,0]), 1.0),
        ]
        for label, probs, expected_phi in test_cases:
            phi = phi_from_entropy(probs)
            rows_entropy.append((label, expected_phi, phi))

        _results["omega_formulas"] = rows_omega
        _results["phi_formulas"] = rows_phi
        _results["entropy_phi"] = rows_entropy

        with capsys.disabled():
            print(f"\n{'='*78}")
            print("  1.3 Omega Formula: Omega = 1/(1+max(lambda,0))")
            print(f"{'='*78}")
            print(f"  {'lambda':>10} {'Expected':>12} {'Actual':>12} {'Error':>12}")
            print(f"{'-'*78}")
            for lam, exp, act, err in rows_omega:
                check = "OK" if err < 1e-10 else "FAIL"
                print(f"  {lam:>10.2f} {exp:>12.6f} {act:>12.6f} {err:>12.2e}  {check}")

            print(f"\n{'='*78}")
            print("  1.4 Phi Formula: Phi = 1/(1+D/D_max)")
            print(f"{'='*78}")
            print(f"  {'D':>8} {'D_max':>8} {'Expected':>12} {'Actual':>12} {'Error':>12}")
            print(f"{'-'*78}")
            for d, dm, exp, act, err in rows_phi:
                check = "OK" if err < 1e-10 else "FAIL"
                print(f"  {d:>8.2f} {dm:>8} {exp:>12.6f} {act:>12.6f} {err:>12.2e}  {check}")

            print(f"\n{'='*78}")
            print("  1.5 Phi from Entropy")
            print(f"{'='*78}")
            print(f"  {'Distribution':<28} {'Expected':>10} {'Actual':>10}")
            print(f"{'-'*78}")
            for label, exp, act in rows_entropy:
                exp_str = f"{exp:.4f}" if exp is not None else "~"
                print(f"  {label:<28} {exp_str:>10} {act:>10.4f}")
            print(f"{'='*78}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. ACCURACY — Composition Rules
# ═══════════════════════════════════════════════════════════════════════════

class TestAccuracyComposition:
    def test_04_composition_rules(self, capsys):
        rows = []

        # Serial: lambda accumulates
        o, p, l, d = compose_serial(1.0, 0.9, -1.0, 0.5, 1.0, 0.8, -0.5, 0.3)
        rows.append(("Serial: 2 convergent",
                      f"lambda={l:.2f}", f"Omega={o:.4f}", f"Phi={p:.4f}", f"D={d:.2f}",
                      l == -1.5 and o == 1.0 and abs(p - 0.72) < 0.001))

        o, p, l, d = compose_serial(0.8, 0.7, 0.5, 1.0, 0.7, 0.6, 0.3, 0.5)
        expected_o = omega_from_lyapunov(0.8)
        rows.append(("Serial: 2 chaotic",
                      f"lambda={l:.2f}", f"Omega={o:.4f}", f"Phi={p:.4f}", f"D={d:.2f}",
                      abs(l - 0.8) < 0.001 and abs(o - expected_o) < 0.001))

        # Parallel: min/max
        o, p, l, d = compose_parallel(0.9, 0.8, -0.5, 0.5, 0.5, 0.3, 1.0, 2.0)
        rows.append(("Parallel: weak link",
                      f"lambda={l:.2f}", f"Omega={o:.4f}", f"Phi={p:.4f}", f"D={d:.2f}",
                      o == 0.5 and p == 0.3 and l == 1.0 and d == 2.0))

        # Multi-stage serial degradation
        o, p, l, d = 1.0, 1.0, 0.0, 0.0
        for i in range(5):
            o, p, l, d = compose_serial(o, p, l, d, 0.95, 0.9, 0.1, 0.3)
        rows.append(("Serial 5-stage (lambda=0.1 each)",
                      f"lambda={l:.2f}", f"Omega={o:.4f}", f"Phi={p:.4f}", f"D={d:.2f}",
                      abs(l - 0.5) < 0.001))

        _results["composition"] = rows

        with capsys.disabled():
            print(f"\n{'='*78}")
            print("  1.6 Composition Rules Verification")
            print(f"{'='*78}")
            for name, *metrics, ok in rows:
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] {name}")
                print(f"         {', '.join(metrics)}")
            print(f"{'='*78}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. ACCURACY — Observation Consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestAccuracyObservation:
    def test_05_observation_consistency(self, capsys):
        rows = []

        # Build a simple convergent tapestry
        decl = (
            DeclarationBuilder("consistency")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .quality(0.9, 0.8)
            .build()
        )
        tapestry = weave(decl, seed=42)
        inputs = {"x": FloatVec.from_list([1, 0, 0, 0, 0, 0, 0, 0])}

        # Single observation repeated N times — check argmax stability
        obs_list = [observe(tapestry, inputs) for _ in range(50)]
        indices = [o.value_index for o in obs_list]
        unique_indices = set(indices)
        mode_count = max(indices.count(i) for i in unique_indices)
        stability = mode_count / len(indices)
        rows.append(("Single observe x50 (argmax stability)", stability, stability >= 0.9))

        # Reobserve should converge
        reobs = reobserve(tapestry, inputs, count=20, seed=42)
        rows.append(("Reobserve x20 (Omega)", reobs.omega, reobs.omega > 0.0))
        rows.append(("Reobserve x20 (Phi)", reobs.phi, reobs.phi > 0.0))

        # Probabilities should sum to 1
        prob_sum = float(np.sum(reobs.probabilities.data))
        rows.append(("Probability normalisation", prob_sum, abs(prob_sum - 1.0) < 0.01))

        # Different inputs should produce different outputs
        inputs2 = {"x": FloatVec.from_list([0, 0, 0, 0, 0, 0, 0, 1])}
        obs_a = observe(tapestry, inputs)
        obs_b = observe(tapestry, inputs2)
        diff = float(np.sum(np.abs(obs_a.probabilities.data - obs_b.probabilities.data)))
        rows.append(("Different inputs -> different outputs", diff, diff > 0.001))

        _results["observation_consistency"] = rows

        with capsys.disabled():
            print(f"\n{'='*78}")
            print("  1.7 Observation Consistency")
            print(f"{'='*78}")
            for name, value, ok in rows:
                status = "PASS" if ok else "FAIL"
                if isinstance(value, float):
                    print(f"  [{status}] {name}: {value:.4f}")
                else:
                    print(f"  [{status}] {name}: {value}")
            print(f"{'='*78}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. SPEED — Lyapunov Estimation
# ═══════════════════════════════════════════════════════════════════════════

class TestSpeedLyapunov:
    def test_06_lyapunov_speed(self, capsys):
        rows = []
        for dim in [4, 8, 16, 32, 64, 128]:
            rng = np.random.default_rng(42)
            M_data = rng.standard_normal((dim, dim)).astype(np.float32) * 0.3
            M_data += np.eye(dim, dtype=np.float32) * 0.8
            M = TransMatrix(data=M_data)

            for steps in [50, 200]:
                t = _time_fn(lambda: estimate_lyapunov(M, steps=steps), n=10)
                rows.append((dim, steps, t * 1e6))

        _results["lyapunov_speed"] = rows

        with capsys.disabled():
            print(f"\n{'='*78}")
            print("  2.1 Lyapunov Estimation Speed")
            print(f"{'='*78}")
            print(f"  {'Dim':>6} {'Steps':>8} {'Avg Time':>14}")
            print(f"{'-'*78}")
            for dim, steps, us in rows:
                print(f"  {dim:>6} {steps:>8} {us:>11.1f} us")
            print(f"{'='*78}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. SPEED — Fractal Dimension Estimation
# ═══════════════════════════════════════════════════════════════════════════

class TestSpeedFractal:
    def test_07_fractal_speed(self, capsys):
        rows = []
        rng = np.random.default_rng(42)

        for n_points in [50, 200, 500, 1000, 2000]:
            for phase_dim in [2, 4]:
                pts = rng.uniform(0, 1, (n_points, phase_dim)).flatten()
                fv = FloatVec.from_list(pts.tolist())

                t_box = _time_fn(
                    lambda: estimate_fractal_dim(fv, method="box_counting", phase_space_dim=phase_dim),
                    n=10,
                )
                t_corr = _time_fn(
                    lambda: estimate_fractal_dim(fv, method="correlation", phase_space_dim=phase_dim),
                    n=5,
                )
                rows.append((n_points, phase_dim, t_box * 1e6, t_corr * 1e6))

        _results["fractal_speed"] = rows

        with capsys.disabled():
            print(f"\n{'='*78}")
            print("  2.2 Fractal Dimension Estimation Speed")
            print(f"{'='*78}")
            print(f"  {'Points':>8} {'Dim':>6} {'Box-count (us)':>16} {'Correlation (us)':>18}")
            print(f"{'-'*78}")
            for np_, pd, tb, tc in rows:
                print(f"  {np_:>8} {pd:>6} {tb:>16.1f} {tc:>18.1f}")
            print(f"{'='*78}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. SPEED — Weave
# ═══════════════════════════════════════════════════════════════════════════

class TestSpeedWeave:
    def test_08_weave_speed(self, capsys):
        rows = []

        # Vary number of nodes in pipeline
        for n_nodes in [1, 2, 4, 8, 16]:
            builder = DeclarationBuilder(f"pipe_{n_nodes}").input("x", 8)
            prev = "x"
            for i in range(n_nodes):
                name = f"n{i}"
                builder.relate(name, [prev], RelationKind.PROPORTIONAL)
                prev = name
            builder.output(prev).quality(0.8, 0.7)
            decl = builder.build()
            t = _time_fn(lambda: weave(decl, seed=42), n=5)
            rows.append((n_nodes, 8, t * 1e3))

        # Vary dimension
        for dim in [4, 8, 16, 32, 64]:
            decl = (
                DeclarationBuilder(f"dim_{dim}")
                .input("x", dim)
                .relate("y", ["x"], RelationKind.PROPORTIONAL)
                .relate("z", ["y"], RelationKind.ADDITIVE)
                .output("z")
                .quality(0.8, 0.7)
                .build()
            )
            t = _time_fn(lambda: weave(decl, seed=42), n=5)
            rows.append((2, dim, t * 1e3))

        _results["weave_speed"] = rows

        with capsys.disabled():
            print(f"\n{'='*78}")
            print("  2.3 Weave Speed")
            print(f"{'='*78}")
            print(f"  {'Nodes':>8} {'Dim':>8} {'Avg Time (ms)':>16}")
            print(f"{'-'*78}")
            for nodes, dim, ms in rows:
                print(f"  {nodes:>8} {dim:>8} {ms:>16.2f}")
            print(f"{'='*78}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. SPEED — Observe
# ═══════════════════════════════════════════════════════════════════════════

class TestSpeedObserve:
    def test_09_observe_speed(self, capsys):
        rows = []

        for dim in [4, 8, 16, 32, 64]:
            decl = (
                DeclarationBuilder(f"obs_{dim}")
                .input("x", dim)
                .relate("y", ["x"], RelationKind.PROPORTIONAL)
                .output("y")
                .quality(0.8, 0.7)
                .build()
            )
            tapestry = weave(decl, seed=42)
            inp = {"x": FloatVec(data=np.ones(dim, dtype=np.float32) / math.sqrt(dim))}

            # Single observe
            t_single = _time_fn(lambda: observe(tapestry, inp), n=20)

            # Reobserve x10
            t_re10 = _time_fn(lambda: reobserve(tapestry, inp, count=10, seed=42), n=5)

            rows.append((dim, t_single * 1e6, t_re10 * 1e6))

        _results["observe_speed"] = rows

        with capsys.disabled():
            print(f"\n{'='*78}")
            print("  2.4 Observe Speed")
            print(f"{'='*78}")
            print(f"  {'Dim':>8} {'Single (us)':>14} {'Reobserve x10 (us)':>22}")
            print(f"{'-'*78}")
            for dim, ts, tr in rows:
                print(f"  {dim:>8} {ts:>14.1f} {tr:>22.1f}")
            print(f"{'='*78}")


# ═══════════════════════════════════════════════════════════════════════════
# 10. SPEED — DSL Parse
# ═══════════════════════════════════════════════════════════════════════════

class TestSpeedDSL:
    def test_10_dsl_parse_speed(self, capsys):
        rows = []

        sources = {
            "simple (1 relation)": textwrap.dedent("""\
                entangle test(x: float[8]) @ Omega(0.9) Phi(0.7) {
                    y <~> x
                }
            """),
            "medium (3 relations)": textwrap.dedent("""\
                entangle test(a: float[8], b: float[8]) @ Omega(0.9) Phi(0.7) {
                    c <~> combine(a, b)
                    d <+> c
                    e <*> d
                }
            """),
            "full program": textwrap.dedent("""\
                entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
                    relevance <~> similarity(query, db)
                    ranking <~> relevance
                }

                result = observe search(query_vec, db_vec)

                if result.Omega < 0.95 {
                    result = reobserve search(query_vec, db_vec) x 10
                }
            """),
        }

        for label, src in sources.items():
            t = _time_fn(lambda: parse_quantum(src), n=50)
            rows.append((label, len(src), t * 1e6))

        _results["dsl_speed"] = rows

        with capsys.disabled():
            print(f"\n{'='*78}")
            print("  2.5 DSL Parse Speed")
            print(f"{'='*78}")
            print(f"  {'Source':<30} {'Chars':>8} {'Avg Time (us)':>16}")
            print(f"{'-'*78}")
            for label, chars, us in rows:
                print(f"  {label:<30} {chars:>8} {us:>16.1f}")
            print(f"{'='*78}")


# ═══════════════════════════════════════════════════════════════════════════
# 11. SPEED — End-to-End Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class TestSpeedEndToEnd:
    def test_11_end_to_end(self, capsys):
        rows = []

        source = textwrap.dedent("""\
            entangle search(query: float[16], db: float[16]) @ Omega(0.9) Phi(0.7) {
                relevance <~> similarity(query, db)
                ranking <~> relevance
            }
        """)

        inputs = {
            "query": FloatVec(data=np.ones(16, dtype=np.float32) / 4.0),
            "db": FloatVec(data=np.zeros(16, dtype=np.float32)),
        }

        # Measure each phase
        t_parse = _time_fn(lambda: parse_quantum(source), n=20)

        prog = parse_quantum(source)
        decl = prog.declarations[0]

        t_cost = _time_fn(lambda: estimate_cost(decl), n=20)
        t_weave = _time_fn(lambda: weave(decl, seed=42), n=10)

        tapestry = weave(decl, seed=42)

        t_observe = _time_fn(lambda: observe(tapestry, inputs), n=20)
        t_reobs = _time_fn(lambda: reobserve(tapestry, inputs, count=10, seed=42), n=5)

        # Total pipeline
        def full_pipeline():
            p = parse_quantum(source)
            d = p.declarations[0]
            t = weave(d, seed=42)
            return observe(t, inputs)

        t_total = _time_fn(full_pipeline, n=5)

        rows = [
            ("Parse DSL", t_parse * 1e6),
            ("Estimate cost", t_cost * 1e6),
            ("Weave (build tapestry)", t_weave * 1e3 * 1000),  # in us
            ("Observe (single)", t_observe * 1e6),
            ("Reobserve x10", t_reobs * 1e6),
            ("Full pipeline (parse->weave->observe)", t_total * 1e6),
        ]

        _results["end_to_end"] = rows

        with capsys.disabled():
            print(f"\n{'='*78}")
            print("  2.6 End-to-End Pipeline (dim=16, 2 relations)")
            print(f"{'='*78}")
            print(f"  {'Phase':<45} {'Avg Time':>14}")
            print(f"{'-'*78}")
            for label, us in rows:
                if us > 1_000_000:
                    print(f"  {label:<45} {us/1e6:>11.2f} s")
                elif us > 1000:
                    print(f"  {label:<45} {us/1000:>11.2f} ms")
                else:
                    print(f"  {label:<45} {us:>11.1f} us")
            print(f"{'='*78}")


# ═══════════════════════════════════════════════════════════════════════════
# 12. TOKEN EFFICIENCY — Quantum DSL vs Python
# ═══════════════════════════════════════════════════════════════════════════

QUANTUM_PROGRAMS = {
    "search": {
        "python": textwrap.dedent("""\
            import numpy as np
            from scipy.linalg import qr

            def build_search(query_dim, db_dim, omega_target=0.9, phi_target=0.7):
                M = np.random.randn(query_dim, query_dim) * 0.3 + np.eye(query_dim) * 0.8
                Q, R = qr(M)
                trajectory = Q * np.sign(np.diag(R))
                lyapunov = np.log(np.max(np.abs(np.linalg.eigvals(trajectory))))
                omega = 1.0 / (1.0 + max(lyapunov, 0))
                return trajectory, omega

            def observe_search(trajectory, query, db):
                state = query @ trajectory
                probs = state ** 2
                probs = probs / probs.sum()
                return np.argmax(probs), probs
        """),
        "axol_dsl": textwrap.dedent("""\
            entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
                relevance <~> similarity(query, db)
                ranking <~> relevance
            }
            result = observe search(query_vec, db_vec)
        """),
    },
    "classify": {
        "python": textwrap.dedent("""\
            import numpy as np

            def build_classifier(input_dim, n_classes, omega=0.9, phi=0.6):
                W = np.random.randn(input_dim, input_dim) * 0.1 + np.eye(input_dim) * 0.8
                labels = {i: f"class_{i}" for i in range(n_classes)}
                return W, labels

            def classify(W, labels, image):
                state = image @ W
                probs = state ** 2 / (state ** 2).sum()
                idx = np.argmax(probs)
                return labels.get(idx, "unknown"), probs
        """),
        "axol_dsl": textwrap.dedent("""\
            entangle classify(image: float[16]) @ Omega(0.9) Phi(0.6) {
                category <~> features(image)
            }
            result = observe classify(photo_vec)
        """),
    },
    "pipeline": {
        "python": textwrap.dedent("""\
            import numpy as np

            def build_pipeline(dim, stages=3, omega=0.8, phi=0.7):
                matrices = []
                for _ in range(stages):
                    M = np.random.randn(dim, dim) * 0.3 + np.eye(dim) * 0.8
                    Q, R = np.linalg.qr(M)
                    matrices.append(Q * np.sign(np.diag(R)))
                return matrices

            def run_pipeline(matrices, x):
                state = x
                for M in matrices:
                    state = state @ M
                probs = state ** 2 / (state ** 2).sum()
                return np.argmax(probs), probs

            def run_with_reobserve(matrices, x, count=10):
                results = [run_pipeline(matrices, x) for _ in range(count)]
                avg_probs = np.mean([r[1] for r in results], axis=0)
                avg_probs /= avg_probs.sum()
                return np.argmax(avg_probs), avg_probs
        """),
        "axol_dsl": textwrap.dedent("""\
            entangle pipeline(x: float[8]) @ Omega(0.8) Phi(0.7) {
                a <~> x
                b <+> a
                c <*> b
            }
            result = observe pipeline(input_vec)
            if result.Omega < 0.9 {
                result = reobserve pipeline(input_vec) x 10
            }
        """),
    },
    "multi_input": {
        "python": textwrap.dedent("""\
            import numpy as np

            def build_fusion(dim_a, dim_b, dim_c, omega=0.7, phi=0.6):
                dim = max(dim_a, dim_b, dim_c)
                M_inv = np.linalg.inv(np.random.randn(dim, dim) * 0.3 + np.eye(dim))
                M_cond = np.zeros((dim, dim))
                block = dim // 2
                M_cond[:block, :block] = np.eye(block) * 0.7
                M_cond[block:, block:] = np.eye(dim - block) * 0.7
                return M_inv, M_cond

            def fuse(M_inv, M_cond, a, b, c):
                combined = a @ M_inv + b @ M_inv + c @ M_cond
                probs = combined ** 2 / (combined ** 2).sum()
                return np.argmax(probs), probs
        """),
        "axol_dsl": textwrap.dedent("""\
            entangle fuse(a: float[8], b: float[8], c: float[8]) @ Omega(0.7) Phi(0.6) {
                d <!> combine(a, b, c)
                e <*> d
                f <?> merge(e, d)
            }
            result = observe fuse(vec_a, vec_b, vec_c)
        """),
    },
    "reobserve_pattern": {
        "python": textwrap.dedent("""\
            import numpy as np

            def compute_with_quality_check(M, x, omega_threshold=0.95, max_retries=10):
                results = []
                for i in range(max_retries):
                    state = x @ M
                    probs = state ** 2 / (state ** 2).sum()
                    results.append(np.argmax(probs))
                    indices, counts = np.unique(results, return_counts=True)
                    omega = counts.max() / len(results)
                    if omega >= omega_threshold:
                        avg_probs = probs
                        break
                else:
                    avg_probs = probs
                return np.argmax(avg_probs), avg_probs, omega
        """),
        "axol_dsl": textwrap.dedent("""\
            entangle compute(x: float[8]) @ Omega(0.95) Phi(0.8) {
                y <~> x
            }
            result = observe compute(input_vec)
            if result.Omega < 0.95 {
                result = reobserve compute(input_vec) x 10
            }
        """),
    },
}


class TestTokenEfficiency:
    def test_12_token_efficiency(self, capsys):
        rows = []

        for name, srcs in QUANTUM_PROGRAMS.items():
            py_tokens = _count_tokens(srcs["python"])
            dsl_tokens = _count_tokens(srcs["axol_dsl"])
            saving = (1 - dsl_tokens / py_tokens) * 100
            py_lines = len(srcs["python"].strip().split("\n"))
            dsl_lines = len(srcs["axol_dsl"].strip().split("\n"))
            rows.append((name, py_tokens, py_lines, dsl_tokens, dsl_lines, saving))

        _results["token_efficiency"] = rows

        with capsys.disabled():
            print(f"\n{'='*90}")
            print("  3. Token Efficiency: Quantum DSL vs Python (tiktoken cl100k_base)")
            print(f"{'='*90}")
            print(f"  {'Program':<20} {'Py Tokens':>10} {'Py Lines':>9} {'DSL Tokens':>11} {'DSL Lines':>10} {'Saving':>8}")
            print(f"{'-'*90}")
            total_py = 0
            total_dsl = 0
            for name, pt, pl, dt, dl, sv in rows:
                total_py += pt
                total_dsl += dt
                print(f"  {name:<20} {pt:>10} {pl:>9} {dt:>11} {dl:>10} {sv:>7.0f}%")
            total_saving = (1 - total_dsl / total_py) * 100
            print(f"{'-'*90}")
            print(f"  {'TOTAL':<20} {total_py:>10} {'':>9} {total_dsl:>11} {'':>10} {total_saving:>7.0f}%")
            print(f"{'='*90}")


# ═══════════════════════════════════════════════════════════════════════════
# 13. REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateReport:
    def test_99_generate_report(self, capsys):
        """Generate QUANTUM_PERFORMANCE_REPORT.md from accumulated results."""
        lines = [
            "# AXOL Quantum Module Performance Report",
            "",
            "Auto-generated benchmark results for the chaos-theory-based quantum module.",
            "",
        ]

        # ── 1. Accuracy ──
        lines.append("## 1. Accuracy")
        lines.append("")

        # 1.1 Lyapunov
        if "lyapunov_accuracy" in _results:
            lines.append("### 1.1 Lyapunov Estimation Accuracy")
            lines.append("")
            lines.append("| System | Expected | Measured | Abs Error |")
            lines.append("|--------|----------|----------|-----------|")
            total_err = 0.0
            for name, exp, meas, err in _results["lyapunov_accuracy"]:
                lines.append(f"| {name} | {exp:.4f} | {meas:.4f} | {err:.4f} |")
                total_err += err
            avg = total_err / len(_results["lyapunov_accuracy"])
            lines.append(f"| **Average** | | | **{avg:.4f}** |")
            lines.append("")

        # 1.2 Fractal
        if "fractal_accuracy" in _results:
            lines.append("### 1.2 Fractal Dimension Estimation Accuracy")
            lines.append("")
            lines.append("| Geometry | Expected | Measured | Abs Error |")
            lines.append("|----------|----------|----------|-----------|")
            total_err = 0.0
            for name, exp, meas, err in _results["fractal_accuracy"]:
                lines.append(f"| {name} | {exp:.3f} | {meas:.3f} | {err:.3f} |")
                total_err += err
            avg = total_err / len(_results["fractal_accuracy"])
            lines.append(f"| **Average** | | | **{avg:.3f}** |")
            lines.append("")

        # 1.3 Omega formula
        if "omega_formulas" in _results:
            lines.append("### 1.3 Omega Formula Verification")
            lines.append("")
            lines.append("`Omega = 1/(1+max(lambda,0))`")
            lines.append("")
            lines.append("| lambda | Expected | Actual | Error |")
            lines.append("|--------|----------|--------|-------|")
            for lam, exp, act, err in _results["omega_formulas"]:
                lines.append(f"| {lam:.2f} | {exp:.6f} | {act:.6f} | {err:.2e} |")
            lines.append("")

        # 1.4 Phi formula
        if "phi_formulas" in _results:
            lines.append("### 1.4 Phi Formula Verification")
            lines.append("")
            lines.append("`Phi = 1/(1+D/D_max)`")
            lines.append("")
            lines.append("| D | D_max | Expected | Actual | Error |")
            lines.append("|---|-------|----------|--------|-------|")
            for d, dm, exp, act, err in _results["phi_formulas"]:
                lines.append(f"| {d:.2f} | {dm} | {exp:.6f} | {act:.6f} | {err:.2e} |")
            lines.append("")

        # 1.5 Entropy Phi
        if "entropy_phi" in _results:
            lines.append("### 1.5 Phi from Entropy")
            lines.append("")
            lines.append("| Distribution | Expected | Actual |")
            lines.append("|-------------|----------|--------|")
            for label, exp, act in _results["entropy_phi"]:
                exp_str = f"{exp:.4f}" if exp is not None else "~"
                lines.append(f"| {label} | {exp_str} | {act:.4f} |")
            lines.append("")

        # 1.6 Composition
        if "composition" in _results:
            lines.append("### 1.6 Composition Rules Verification")
            lines.append("")
            lines.append("| Test | Metrics | Result |")
            lines.append("|------|---------|--------|")
            for name, *metrics, ok in _results["composition"]:
                status = "PASS" if ok else "FAIL"
                lines.append(f"| {name} | {', '.join(metrics)} | {status} |")
            lines.append("")

        # 1.7 Observation
        if "observation_consistency" in _results:
            lines.append("### 1.7 Observation Consistency")
            lines.append("")
            lines.append("| Test | Value | Result |")
            lines.append("|------|-------|--------|")
            for name, value, ok in _results["observation_consistency"]:
                status = "PASS" if ok else "FAIL"
                v_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                lines.append(f"| {name} | {v_str} | {status} |")
            lines.append("")

        # ── 2. Speed ──
        lines.append("## 2. Speed Benchmarks")
        lines.append("")

        # 2.1 Lyapunov speed
        if "lyapunov_speed" in _results:
            lines.append("### 2.1 Lyapunov Estimation Speed")
            lines.append("")
            lines.append("| Dimension | Steps | Avg Time (us) |")
            lines.append("|-----------|-------|---------------|")
            for dim, steps, us in _results["lyapunov_speed"]:
                lines.append(f"| {dim} | {steps} | {us:.1f} |")
            lines.append("")

        # 2.2 Fractal speed
        if "fractal_speed" in _results:
            lines.append("### 2.2 Fractal Dimension Estimation Speed")
            lines.append("")
            lines.append("| Points | Phase Dim | Box-counting (us) | Correlation (us) |")
            lines.append("|--------|-----------|-------------------|------------------|")
            for np_, pd, tb, tc in _results["fractal_speed"]:
                lines.append(f"| {np_} | {pd} | {tb:.1f} | {tc:.1f} |")
            lines.append("")

        # 2.3 Weave speed
        if "weave_speed" in _results:
            lines.append("### 2.3 Weave Speed")
            lines.append("")
            lines.append("| Nodes | Dimension | Avg Time (ms) |")
            lines.append("|-------|-----------|---------------|")
            for nodes, dim, ms in _results["weave_speed"]:
                lines.append(f"| {nodes} | {dim} | {ms:.2f} |")
            lines.append("")

        # 2.4 Observe speed
        if "observe_speed" in _results:
            lines.append("### 2.4 Observe Speed")
            lines.append("")
            lines.append("| Dimension | Single Observe (us) | Reobserve x10 (us) |")
            lines.append("|-----------|--------------------|--------------------|")
            for dim, ts, tr in _results["observe_speed"]:
                lines.append(f"| {dim} | {ts:.1f} | {tr:.1f} |")
            lines.append("")

        # 2.5 DSL speed
        if "dsl_speed" in _results:
            lines.append("### 2.5 DSL Parse Speed")
            lines.append("")
            lines.append("| Source | Characters | Avg Time (us) |")
            lines.append("|--------|------------|---------------|")
            for label, chars, us in _results["dsl_speed"]:
                lines.append(f"| {label} | {chars} | {us:.1f} |")
            lines.append("")

        # 2.6 End-to-end
        if "end_to_end" in _results:
            lines.append("### 2.6 End-to-End Pipeline (dim=16, 2 relations)")
            lines.append("")
            lines.append("| Phase | Avg Time |")
            lines.append("|-------|----------|")
            for label, us in _results["end_to_end"]:
                if us > 1_000_000:
                    lines.append(f"| {label} | {us/1e6:.2f} s |")
                elif us > 1000:
                    lines.append(f"| {label} | {us/1000:.2f} ms |")
                else:
                    lines.append(f"| {label} | {us:.1f} us |")
            lines.append("")

        # ── 3. Token Efficiency ──
        if "token_efficiency" in _results:
            lines.append("## 3. Token Efficiency")
            lines.append("")
            lines.append("Quantum DSL vs equivalent Python (tiktoken cl100k_base)")
            lines.append("")
            lines.append("| Program | Python Tokens | Python Lines | DSL Tokens | DSL Lines | Saving |")
            lines.append("|---------|--------------|-------------|------------|-----------|--------|")
            total_py = 0
            total_dsl = 0
            for name, pt, pl, dt, dl, sv in _results["token_efficiency"]:
                total_py += pt
                total_dsl += dt
                lines.append(f"| {name} | {pt} | {pl} | {dt} | {dl} | {sv:.0f}% |")
            total_saving = (1 - total_dsl / total_py) * 100
            lines.append(f"| **TOTAL** | **{total_py}** | | **{total_dsl}** | | **{total_saving:.0f}%** |")
            lines.append("")

        # Write report
        report_path = os.path.join(PROJECT_ROOT, "QUANTUM_PERFORMANCE_REPORT.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        with capsys.disabled():
            print(f"\n{'='*78}")
            print(f"  Report written to: {report_path}")
            print(f"{'='*78}")
