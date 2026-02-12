"""Koopman-AXOL validation experiments (7 experiments).

Experiment 1: Approximation Fidelity -- per-function-class accuracy
Experiment 2: Depth Scaling -- error propagation characteristics
Experiment 3: Speed Benchmark -- Koopman vs Fallback timing
Experiment 4: Amortization -- weave cost vs repeated observation
Experiment 5: Degree Tradeoff -- degree vs accuracy vs cost
Experiment 6: Omega/Phi Correlation -- metric validity under Koopman
Experiment 7: Realistic Pipelines -- practical scenario validation
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from axol.core.types import FloatVec, TransMatrix
from axol.core import operations as ops
from axol.quantum.koopman import (
    lifted_dim, lift, unlift, estimate_koopman_matrix, compose_koopman_chain,
)
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe, reobserve


# ===================================================================
# Utilities
# ===================================================================

def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two probability distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.maximum(p, 0.0)
    q = np.maximum(q, 0.0)
    sp, sq = np.sum(p), np.sum(q)
    if sp > 0:
        p = p / sp
    if sq > 0:
        q = q / sq
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _tanh(x):
    return np.tanh(x)


def _relu(x):
    return np.maximum(x, 0.0)


def _leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def _abs_fn(x):
    return np.abs(x)


def _hard_threshold(x):
    return np.where(x > 0, 1.0, 0.0)


def _koopman_predict(step_fn, K, x, dim, degree=2):
    """Predict output via Koopman: lift -> multiply -> unlift."""
    psi_x = lift(x.astype(np.float64), degree=degree)
    psi_y = psi_x @ K.data.astype(np.float64)
    return unlift(psi_y, dim, degree=degree)


def _build_chain_declaration(depth, dim=4, transform_fn=None):
    """Build a sequential chain declaration with input 'x'."""
    builder = DeclarationBuilder(f"chain_d{depth}")
    builder.input("x", dim)
    prev = "x"
    for i in range(depth):
        node = f"h{i}"
        builder.relate(node, [prev], RelationKind.PROPORTIONAL, transform_fn=transform_fn)
        prev = node
    builder.output(prev)
    return builder.build()


def _evaluate_pipeline(decl, dim, n_test=30):
    """Evaluate a pipeline: Koopman vs fallback. Uses the first input name."""
    tap_k = weave(decl, seed=42, optimize=True, nonlinear_method="koopman")
    tap_f = weave(decl, seed=42, optimize=False)

    input_name = decl.input_names[0]
    rng = np.random.default_rng(42)
    h_sum, match = 0.0, 0

    for _ in range(n_test):
        x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
        obs_k = observe(tap_k, {input_name: x})
        obs_f = observe(tap_f, {input_name: x})
        h_sum += hellinger_distance(obs_k.probabilities.data, obs_f.probabilities.data)
        if obs_k.value_index == obs_f.value_index:
            match += 1

    # Time
    x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
    observe(tap_k, {input_name: x})
    observe(tap_f, {input_name: x})
    t0 = time.perf_counter()
    for _ in range(50):
        observe(tap_k, {input_name: x})
    t_k = (time.perf_counter() - t0) / 50
    t0 = time.perf_counter()
    for _ in range(50):
        observe(tap_f, {input_name: x})
    t_f = (time.perf_counter() - t0) / 50

    return {
        "hellinger": h_sum / n_test,
        "argmax_rate": match / n_test,
        "koopman_us": t_k * 1e6,
        "fallback_us": t_f * 1e6,
        "has_koopman": tap_k._koopman_matrix is not None,
    }


# ===================================================================
# Experiment 1: Approximation Fidelity
# ===================================================================

class TestExp1_ApproximationFidelity:
    """Test Koopman approximation quality across nonlinear function tiers."""

    DIM = 4
    N_TEST = 50
    N_SAMPLES = 500

    def _evaluate_fidelity(self, fn, fn_name, dim, degree, seed=42):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((dim, dim)) * 0.3 + np.eye(dim) * 0.5

        def step_fn(x):
            return fn(x @ M)

        K = estimate_koopman_matrix(step_fn, dim, degree=degree, n_samples=self.N_SAMPLES, seed=seed)

        argmax_matches = 0
        hellinger_sum = 0.0
        test_rng = np.random.default_rng(seed + 1000)

        for _ in range(self.N_TEST):
            x = test_rng.standard_normal(dim) * 0.3
            actual = step_fn(x)
            predicted = _koopman_predict(step_fn, K, x, dim, degree=degree)

            actual_p = ops.measure(FloatVec(data=actual.astype(np.float32))).data
            pred_p = ops.measure(FloatVec(data=predicted.astype(np.float32))).data

            if np.argmax(actual_p) == np.argmax(pred_p):
                argmax_matches += 1
            hellinger_sum += hellinger_distance(actual_p, pred_p)

        return {
            "argmax_rate": argmax_matches / self.N_TEST,
            "hellinger_mean": hellinger_sum / self.N_TEST,
        }

    def test_fidelity_table(self):
        """Comprehensive fidelity table across tiers and degrees."""
        functions = [
            ("Tier1-ReLU", _relu),
            ("Tier1-LeakyReLU", _leaky_relu),
            ("Tier2-sigmoid", _sigmoid),
            ("Tier2-tanh", _tanh),
            ("Tier3-abs", _abs_fn),
            ("Tier3-threshold", _hard_threshold),
        ]
        degrees = [2, 3]

        print()
        print("=" * 78)
        print("  Exp 1: Koopman Approximation Fidelity (dim=4)")
        print("=" * 78)
        header = f"  {'Function':<20}"
        for d in degrees:
            header += f"  deg={d} argmax  deg={d} H-dist"
        print(header)
        print("-" * 78)

        for fn_name, fn in functions:
            row = f"  {fn_name:<20}"
            for deg in degrees:
                result = self._evaluate_fidelity(fn, fn_name, self.DIM, deg)
                row += f"  {result['argmax_rate']:>10.1%}  {result['hellinger_mean']:>10.4f}"
            print(row)

        print("=" * 78)

        # Assert: Tier 1 and 2 should have reasonable argmax rates at degree=2
        for fn_name, fn in functions[:4]:
            result = self._evaluate_fidelity(fn, fn_name, self.DIM, degree=2)
            assert result["argmax_rate"] >= 0.5, f"{fn_name}: argmax rate too low"

    def test_tier4_composite(self):
        """Tier 4: composite sigmoid(x @ M1) @ M2."""
        rng = np.random.default_rng(77)
        dim = self.DIM
        M1 = rng.standard_normal((dim, dim)) * 0.3
        M2 = rng.standard_normal((dim, dim)) * 0.3 + np.eye(dim) * 0.5

        def composite_fn(x):
            return _sigmoid(x @ M1) @ M2

        K = estimate_koopman_matrix(composite_fn, dim, degree=2, n_samples=self.N_SAMPLES, seed=42)

        argmax_ok = 0
        test_rng = np.random.default_rng(999)
        for _ in range(self.N_TEST):
            x = test_rng.standard_normal(dim) * 0.3
            actual = composite_fn(x)
            predicted = _koopman_predict(composite_fn, K, x, dim, degree=2)
            actual_p = ops.measure(FloatVec(data=actual.astype(np.float32))).data
            pred_p = ops.measure(FloatVec(data=predicted.astype(np.float32))).data
            if np.argmax(actual_p) == np.argmax(pred_p):
                argmax_ok += 1

        rate = argmax_ok / self.N_TEST
        print(f"\n  Tier4-composite: argmax={rate:.1%}")
        assert rate >= 0.4, f"Tier4 composite rate too low: {rate}"


# ===================================================================
# Experiment 2: Depth Scaling
# ===================================================================

class TestExp2_DepthScaling:
    """Test how Koopman approximation error evolves with pipeline depth."""

    def _measure_depth_error(self, dim, depth, degree=2):
        decl = _build_chain_declaration(depth, dim=dim, transform_fn=_sigmoid)
        tap_k = weave(decl, seed=42, optimize=True, nonlinear_method="koopman", koopman_degree=degree)
        tap_f = weave(decl, seed=42, optimize=False)

        rng = np.random.default_rng(123)
        n_test = 30
        hellinger_sum = 0.0
        argmax_match = 0

        for _ in range(n_test):
            x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
            obs_k = observe(tap_k, {"x": x})
            obs_f = observe(tap_f, {"x": x})
            hd = hellinger_distance(obs_k.probabilities.data, obs_f.probabilities.data)
            hellinger_sum += hd
            if obs_k.value_index == obs_f.value_index:
                argmax_match += 1

        return {
            "hellinger": hellinger_sum / n_test,
            "argmax_rate": argmax_match / n_test,
        }

    def test_depth_scaling(self):
        """Main depth scaling experiment: sigmoid, dim=4 and dim=8."""
        depths = [1, 2, 5, 10, 20, 50]

        print()
        print("=" * 78)
        print("  Exp 2: Depth Scaling -- Koopman vs Fallback (sigmoid, degree=2)")
        print("=" * 78)
        print(f"  {'Depth':>6}  {'dim=4 H-dist':>12}  {'dim=4 argmax':>12}"
              f"  {'dim=8 H-dist':>12}  {'dim=8 argmax':>12}")
        print("-" * 78)

        for depth in depths:
            r4 = self._measure_depth_error(4, depth)
            r8 = self._measure_depth_error(8, depth)
            print(f"  {depth:>6}  {r4['hellinger']:>12.4f}  {r4['argmax_rate']:>11.1%}"
                  f"  {r8['hellinger']:>12.4f}  {r8['argmax_rate']:>11.1%}")

        print("=" * 78)

        # Shallow depths should have good agreement
        r_shallow = self._measure_depth_error(4, 2)
        assert r_shallow["argmax_rate"] >= 0.5


# ===================================================================
# Experiment 3: Speed Benchmark
# ===================================================================

class TestExp3_SpeedBenchmark:
    """Compare observation timing: Koopman fast path vs sequential fallback."""

    N_ITER = 50

    def _time_observe(self, tapestry, inputs, n_iter):
        observe(tapestry, inputs)  # warmup
        t0 = time.perf_counter()
        for _ in range(n_iter):
            observe(tapestry, inputs)
        return (time.perf_counter() - t0) / n_iter

    def test_speed_comparison(self):
        """Speed: Koopman vs Fallback across depths."""
        dims = [4, 8]
        depths = [5, 10, 20, 50, 100]

        print()
        print("=" * 88)
        print("  Exp 3: Speed Benchmark -- Koopman vs Fallback (sigmoid, degree=2)")
        print("=" * 88)
        print(f"  {'Dim':>4}  {'Depth':>6}  {'Koopman (us)':>13}"
              f"  {'Fallback (us)':>14}  {'Speedup':>8}")
        print("-" * 88)

        for dim in dims:
            rng = np.random.default_rng(42)
            x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)

            for depth in depths:
                decl = _build_chain_declaration(depth, dim=dim, transform_fn=_sigmoid)
                tap_k = weave(decl, seed=42, optimize=True, nonlinear_method="koopman")
                tap_f = weave(decl, seed=42, optimize=False)

                t_k = self._time_observe(tap_k, {"x": x}, self.N_ITER)
                t_f = self._time_observe(tap_f, {"x": x}, self.N_ITER)
                speedup = t_f / t_k if t_k > 0 else float("inf")

                print(f"  {dim:>4}  {depth:>6}  {t_k*1e6:>13.1f}"
                      f"  {t_f*1e6:>14.1f}  {speedup:>7.1f}x")

        print("=" * 88)

        # Koopman should be faster than fallback at depth>=50
        decl = _build_chain_declaration(50, dim=4, transform_fn=_sigmoid)
        tap_k = weave(decl, seed=42, optimize=True, nonlinear_method="koopman")
        tap_f = weave(decl, seed=42, optimize=False)
        x = FloatVec.from_list([0.3, -0.1, 0.5, 0.2])
        t_k = self._time_observe(tap_k, {"x": x}, 30)
        t_f = self._time_observe(tap_f, {"x": x}, 30)
        assert t_k < t_f * 2, "Koopman should not be drastically slower than fallback"


# ===================================================================
# Experiment 4: Amortization
# ===================================================================

class TestExp4_Amortization:
    """Weave (EDMD) cost amortization over repeated observations."""

    def test_amortization(self):
        dims = [4, 8]

        print()
        print("=" * 78)
        print("  Exp 4: Koopman Weave Cost Amortization")
        print("=" * 78)
        print(f"  {'Dim':>4}  {'Weave (ms)':>11}  {'Obs (us)':>9}"
              f"  {'@100':>10}  {'@10K':>10}  {'@1M':>10}")
        print("-" * 78)

        for dim in dims:
            decl = _build_chain_declaration(5, dim=dim, transform_fn=_sigmoid)
            x = FloatVec(data=np.random.default_rng(42).standard_normal(dim).astype(np.float32) * 0.3)

            t0 = time.perf_counter()
            tap = weave(decl, seed=42, optimize=True, nonlinear_method="koopman")
            weave_time = time.perf_counter() - t0

            observe(tap, {"x": x})
            t0 = time.perf_counter()
            for _ in range(100):
                observe(tap, {"x": x})
            obs_time = (time.perf_counter() - t0) / 100

            amort_100 = weave_time / 100 + obs_time
            amort_10k = weave_time / 10_000 + obs_time
            amort_1m = weave_time / 1_000_000 + obs_time

            print(
                f"  {dim:>4}  {weave_time*1e3:>11.1f}  {obs_time*1e6:>9.1f}"
                f"  {amort_100*1e6:>9.1f}u  {amort_10k*1e6:>9.1f}u  {amort_1m*1e6:>9.1f}u"
            )

        print("=" * 78)


# ===================================================================
# Experiment 5: Degree Tradeoff
# ===================================================================

class TestExp5_DegreeTradeoff:
    """Polynomial degree vs accuracy vs cost tradeoff."""

    def test_degree_tradeoff(self):
        dim = 4
        depth = 20
        degrees = [2, 3, 4]

        print()
        print("=" * 78)
        print("  Exp 5: Degree Tradeoff (dim=4, sigmoid, depth=20)")
        print("=" * 78)
        print(f"  {'Degree':>7}  {'Lifted':>7}  {'H-dist':>8}"
              f"  {'Argmax':>8}  {'Weave(ms)':>10}  {'Obs(us)':>8}")
        print("-" * 78)

        for deg in degrees:
            ld = lifted_dim(dim, deg)
            decl = _build_chain_declaration(depth, dim=dim, transform_fn=_sigmoid)

            t0 = time.perf_counter()
            tap_k = weave(decl, seed=42, optimize=True, nonlinear_method="koopman", koopman_degree=deg, koopman_samples=500)
            weave_ms = (time.perf_counter() - t0) * 1e3

            tap_f = weave(decl, seed=42, optimize=False)

            rng = np.random.default_rng(77)
            h_sum = 0.0
            match = 0
            n_test = 30
            x_test = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)

            for _ in range(n_test):
                x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
                obs_k = observe(tap_k, {"x": x})
                obs_f = observe(tap_f, {"x": x})
                h_sum += hellinger_distance(obs_k.probabilities.data, obs_f.probabilities.data)
                if obs_k.value_index == obs_f.value_index:
                    match += 1

            observe(tap_k, {"x": x_test})
            t0 = time.perf_counter()
            for _ in range(50):
                observe(tap_k, {"x": x_test})
            obs_us = (time.perf_counter() - t0) / 50 * 1e6

            print(
                f"  {deg:>7}  {ld:>7}  {h_sum/n_test:>8.4f}  {match/n_test:>7.1%}"
                f"  {weave_ms:>10.1f}  {obs_us:>8.1f}"
            )

        print("=" * 78)


# ===================================================================
# Experiment 6: Omega/Phi Correlation
# ===================================================================

class TestExp6_OmegaPhiCorrelation:
    """Test whether Omega/Phi metrics correlate with actual Koopman accuracy."""

    def test_omega_phi_vs_accuracy(self):
        configs = [
            (4, 2, _sigmoid, "sigmoid-d2"),
            (4, 5, _sigmoid, "sigmoid-d5"),
            (4, 10, _sigmoid, "sigmoid-d10"),
            (4, 20, _sigmoid, "sigmoid-d20"),
            (8, 5, _sigmoid, "sigmoid-d5-dim8"),
            (4, 5, _relu, "relu-d5"),
            (4, 5, _tanh, "tanh-d5"),
            (4, 10, _tanh, "tanh-d10"),
        ]

        print()
        print("=" * 78)
        print("  Exp 6: Omega/Phi vs Koopman Accuracy Correlation")
        print("=" * 78)
        print(f"  {'Config':<20}  {'Omega':>6}  {'Phi':>6}  {'H-dist':>8}  {'Argmax':>7}")
        print("-" * 78)

        omegas, phis, hellingers = [], [], []

        for dim, depth, fn, label in configs:
            decl = _build_chain_declaration(depth, dim=dim, transform_fn=fn)
            tap_k = weave(decl, seed=42, optimize=True, nonlinear_method="koopman")
            tap_f = weave(decl, seed=42, optimize=False)

            rng = np.random.default_rng(55)
            h_sum, match_sum = 0.0, 0
            n_test = 20

            for _ in range(n_test):
                x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
                obs_k = observe(tap_k, {"x": x})
                obs_f = observe(tap_f, {"x": x})
                h_sum += hellinger_distance(obs_k.probabilities.data, obs_f.probabilities.data)
                if obs_k.value_index == obs_f.value_index:
                    match_sum += 1

            omega = tap_k.weaver_report.estimated_omega
            phi = tap_k.weaver_report.estimated_phi
            h_mean = h_sum / n_test

            omegas.append(omega)
            phis.append(phi)
            hellingers.append(h_mean)

            print(
                f"  {label:<20}  {omega:>6.3f}  {phi:>6.3f}"
                f"  {h_mean:>8.4f}  {match_sum/n_test:>6.1%}"
            )

        print("=" * 78)

        # Rank correlation (skip scipy if unavailable)
        try:
            from scipy.stats import spearmanr
            # Only compute if values are non-constant
            if len(set(omegas)) > 1:
                corr_omega, _ = spearmanr(omegas, hellingers)
                print(f"  Spearman(Omega, H-dist) = {corr_omega:+.3f}")
            if len(set(phis)) > 1:
                corr_phi, _ = spearmanr(phis, hellingers)
                print(f"  Spearman(Phi, H-dist)   = {corr_phi:+.3f}")
        except ImportError:
            print("  (scipy not available -- skipping correlation)")


# ===================================================================
# Experiment 7: Realistic Pipelines
# ===================================================================

class TestExp7_RealisticPipelines:
    """Validate Koopman on realistic pipeline scenarios."""

    def test_game_ai_decision(self):
        """Game AI: input(HP,dist,ammo,shield) -> sigmoid layers -> action."""
        dim = 4
        builder = DeclarationBuilder("game_ai")
        builder.input("state", dim)
        builder.relate("hidden1", ["state"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.relate("hidden2", ["hidden1"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.relate("hidden3", ["hidden2"], RelationKind.PROPORTIONAL, transform_fn=_relu)
        builder.relate("action", ["hidden3"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.output("action")

        result = _evaluate_pipeline(builder.build(), dim)
        print(f"\n  Game AI: argmax={result['argmax_rate']:.1%} H={result['hellinger']:.4f}"
              f"  K={result['koopman_us']:.0f}us  F={result['fallback_us']:.0f}us")
        assert result["has_koopman"], "Game AI should use Koopman"

    def test_recommendation_scoring(self):
        """Recommender: user_vec -> tanh embedding -> projection -> scores."""
        dim = 8
        builder = DeclarationBuilder("recommender")
        builder.input("user", dim)
        builder.relate("embed", ["user"], RelationKind.PROPORTIONAL, transform_fn=_tanh)
        builder.relate("project", ["embed"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.relate("score", ["project"], RelationKind.PROPORTIONAL)
        builder.output("score")

        result = _evaluate_pipeline(builder.build(), dim)
        print(f"\n  Recommender: argmax={result['argmax_rate']:.1%} H={result['hellinger']:.4f}"
              f"  K={result['koopman_us']:.0f}us  F={result['fallback_us']:.0f}us")
        assert result["has_koopman"], "Recommender should use Koopman"

    def test_physics_simulation(self):
        """Physics: position/velocity -> nonlinear force -> state update chain."""
        dim = 4

        def force_fn(x):
            return _tanh(x * 0.5) * 0.8

        builder = DeclarationBuilder("physics")
        builder.input("state", dim)
        builder.relate("force1", ["state"], RelationKind.PROPORTIONAL, transform_fn=force_fn)
        builder.relate("update1", ["force1"], RelationKind.PROPORTIONAL, transform_fn=force_fn)
        builder.relate("force2", ["update1"], RelationKind.PROPORTIONAL, transform_fn=force_fn)
        builder.relate("update2", ["force2"], RelationKind.PROPORTIONAL, transform_fn=force_fn)
        builder.relate("final", ["update2"], RelationKind.PROPORTIONAL)
        builder.output("final")

        result = _evaluate_pipeline(builder.build(), dim)
        print(f"\n  Physics Sim: argmax={result['argmax_rate']:.1%} H={result['hellinger']:.4f}"
              f"  K={result['koopman_us']:.0f}us  F={result['fallback_us']:.0f}us")
        assert result["has_koopman"], "Physics sim should use Koopman"

    def test_summary_table(self):
        """Print summary table for all realistic pipelines."""
        scenarios = []

        # Game AI
        dim = 4
        builder = DeclarationBuilder("game_ai")
        builder.input("state", dim)
        builder.relate("h1", ["state"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.relate("h2", ["h1"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.relate("h3", ["h2"], RelationKind.PROPORTIONAL, transform_fn=_relu)
        builder.relate("action", ["h3"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.output("action")
        scenarios.append(("Game AI (dim=4, d=4)", builder.build(), dim))

        # Recommender
        dim = 8
        builder = DeclarationBuilder("recsys")
        builder.input("user", dim)
        builder.relate("embed", ["user"], RelationKind.PROPORTIONAL, transform_fn=_tanh)
        builder.relate("proj", ["embed"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.relate("score", ["proj"], RelationKind.PROPORTIONAL)
        builder.output("score")
        scenarios.append(("Recommender (dim=8, d=3)", builder.build(), dim))

        # Physics
        dim = 4

        def force_fn(x):
            return _tanh(x * 0.5) * 0.8

        builder = DeclarationBuilder("physics")
        builder.input("s", dim)
        builder.relate("f1", ["s"], RelationKind.PROPORTIONAL, transform_fn=force_fn)
        builder.relate("u1", ["f1"], RelationKind.PROPORTIONAL, transform_fn=force_fn)
        builder.relate("f2", ["u1"], RelationKind.PROPORTIONAL, transform_fn=force_fn)
        builder.relate("u2", ["f2"], RelationKind.PROPORTIONAL, transform_fn=force_fn)
        builder.relate("out", ["u2"], RelationKind.PROPORTIONAL)
        builder.output("out")
        scenarios.append(("Physics (dim=4, d=5)", builder.build(), dim))

        print()
        print("=" * 88)
        print("  Exp 7: Realistic Pipeline Summary")
        print("=" * 88)
        print(f"  {'Scenario':<28}  {'H-dist':>7}  {'Argmax':>7}"
              f"  {'K(us)':>7}  {'F(us)':>7}  {'Speedup':>8}")
        print("-" * 88)

        for name, decl, dim in scenarios:
            r = _evaluate_pipeline(decl, dim)
            sp = r['fallback_us'] / r['koopman_us'] if r['koopman_us'] > 0 else 0
            print(
                f"  {name:<28}  {r['hellinger']:>7.4f}  {r['argmax_rate']:>6.1%}"
                f"  {r['koopman_us']:>7.0f}  {r['fallback_us']:>7.0f}  {sp:>7.1f}x"
            )

        print("=" * 88)
