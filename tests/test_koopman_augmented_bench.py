"""Benchmark: Augmented Dictionary vs Polynomial basis for Koopman operator.

Compares poly vs augmented basis on:
  1. Fidelity per function class (H-dist, argmax match)
  2. Speed (weave time, observe time)
  3. Depth scaling with discontinuous functions
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from axol.core.types import FloatVec
from axol.core import operations as ops
from axol.quantum.koopman import lifted_dim, lift, unlift, estimate_koopman_matrix
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe


# ===================================================================
# Utilities
# ===================================================================

def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
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

def _abs_fn(x):
    return np.abs(x)

def _hard_threshold(x):
    return np.where(x > 0, 1.0, 0.0)

def _leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


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


# ===================================================================
# Experiment A: Fidelity Table -- poly vs augmented per function class
# ===================================================================

class TestAugmentedFidelity:
    """Compare poly vs augmented basis fidelity for each function class."""

    def test_fidelity_comparison(self):
        dim = 4
        depth = 3
        n_test = 30
        funcs = [
            ("sigmoid",        _sigmoid),
            ("tanh",           _tanh),
            ("relu",           _relu),
            ("leaky_relu",     _leaky_relu),
            ("abs",            _abs_fn),
            ("hard_threshold", _hard_threshold),
        ]

        print()
        print("=" * 100)
        print("  Augmented Dictionary Fidelity: poly vs augmented (depth=%d, dim=%d)" % (depth, dim))
        print("=" * 100)
        header = "  %-16s | %10s %10s %10s | %10s %10s %10s | %s" % (
            "Function", "H-dist(P)", "Match%(P)", "LdDim(P)",
            "H-dist(A)", "Match%(A)", "LdDim(A)", "Winner",
        )
        print(header)
        print("  " + "-" * 96)

        aug_wins = 0
        for fname, fn in funcs:
            decl = _build_chain(depth, dim, fn)

            # Poly
            tap_p = weave(decl, seed=42, nonlinear_method="koopman", koopman_basis="poly")
            tap_fb = weave(decl, seed=42, optimize=False)
            # Augmented
            tap_a = weave(decl, seed=42, nonlinear_method="koopman", koopman_basis="augmented")

            rng = np.random.default_rng(42)
            h_poly, h_aug = 0.0, 0.0
            match_poly, match_aug = 0, 0

            for _ in range(n_test):
                x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
                obs_fb = observe(tap_fb, {"x": x})
                obs_p = observe(tap_p, {"x": x})
                obs_a = observe(tap_a, {"x": x})

                h_poly += hellinger_distance(obs_p.probabilities.data, obs_fb.probabilities.data)
                h_aug += hellinger_distance(obs_a.probabilities.data, obs_fb.probabilities.data)
                if obs_p.value_index == obs_fb.value_index:
                    match_poly += 1
                if obs_a.value_index == obs_fb.value_index:
                    match_aug += 1

            h_poly /= n_test
            h_aug /= n_test
            ld_p = lifted_dim(dim, basis="poly")
            ld_a = lifted_dim(dim, basis="augmented")

            winner = "AUG" if h_aug < h_poly else ("POLY" if h_poly < h_aug else "TIE")
            if winner == "AUG":
                aug_wins += 1

            print("  %-16s | %10.4f %9.1f%% %10d | %10.4f %9.1f%% %10d | %s" % (
                fname,
                h_poly, match_poly / n_test * 100, ld_p,
                h_aug, match_aug / n_test * 100, ld_a,
                winner,
            ))

        print("  " + "-" * 96)
        print("  Augmented wins: %d / %d function classes" % (aug_wins, len(funcs)))
        print("=" * 100)

        # Augmented should win on at least the PWA functions (relu, abs, hard_threshold)
        assert aug_wins >= 2, f"Augmented only won {aug_wins}/6"


# ===================================================================
# Experiment B: Speed -- weave + observe timing poly vs augmented
# ===================================================================

class TestAugmentedSpeed:
    """Compare weave and observe speed between poly and augmented."""

    def test_speed_comparison(self):
        dim = 4
        depth = 3
        n_obs = 100

        print()
        print("=" * 90)
        print("  Speed Comparison: poly vs augmented (depth=%d, dim=%d)" % (depth, dim))
        print("=" * 90)
        header = "  %-16s | %12s %12s | %12s %12s" % (
            "Function", "Weave(P)", "Weave(A)", "Observe(P)", "Observe(A)",
        )
        print(header)
        print("  " + "-" * 86)

        funcs = [
            ("sigmoid",        _sigmoid),
            ("relu",           _relu),
            ("hard_threshold", _hard_threshold),
        ]

        for fname, fn in funcs:
            decl = _build_chain(depth, dim, fn)

            # Weave timing
            t0 = time.perf_counter()
            tap_p = weave(decl, seed=42, nonlinear_method="koopman", koopman_basis="poly")
            t_weave_p = time.perf_counter() - t0

            t0 = time.perf_counter()
            tap_a = weave(decl, seed=42, nonlinear_method="koopman", koopman_basis="augmented")
            t_weave_a = time.perf_counter() - t0

            # Observe timing (warmup + measure)
            x = FloatVec(data=np.random.default_rng(42).standard_normal(dim).astype(np.float32) * 0.3)
            observe(tap_p, {"x": x})
            observe(tap_a, {"x": x})

            t0 = time.perf_counter()
            for _ in range(n_obs):
                observe(tap_p, {"x": x})
            t_obs_p = (time.perf_counter() - t0) / n_obs

            t0 = time.perf_counter()
            for _ in range(n_obs):
                observe(tap_a, {"x": x})
            t_obs_a = (time.perf_counter() - t0) / n_obs

            print("  %-16s | %10.1f ms %10.1f ms | %10.1f us %10.1f us" % (
                fname,
                t_weave_p * 1000, t_weave_a * 1000,
                t_obs_p * 1e6, t_obs_a * 1e6,
            ))

        print("=" * 90)


# ===================================================================
# Experiment C: Depth scaling for discontinuous (augmented vs poly)
# ===================================================================

class TestAugmentedDepthScaling:
    """Depth scaling for hard_threshold and relu with augmented basis."""

    def test_depth_scaling(self):
        dim = 4
        depths = [1, 3, 5, 10, 20]
        n_test = 20

        print()
        print("=" * 90)
        print("  Depth Scaling: augmented vs poly for PWA functions (dim=%d)" % dim)
        print("=" * 90)

        for fname, fn in [("relu", _relu), ("hard_threshold", _hard_threshold)]:
            print()
            print("  --- %s ---" % fname)
            print("  %8s | %10s %10s | %10s %10s" % (
                "Depth", "H-dist(P)", "H-dist(A)", "Match%(P)", "Match%(A)",
            ))
            print("  " + "-" * 60)

            for depth in depths:
                decl = _build_chain(depth, dim, fn)
                tap_p = weave(decl, seed=42, nonlinear_method="koopman", koopman_basis="poly")
                tap_a = weave(decl, seed=42, nonlinear_method="koopman", koopman_basis="augmented")
                tap_fb = weave(decl, seed=42, optimize=False)

                rng = np.random.default_rng(42)
                h_p, h_a = 0.0, 0.0
                m_p, m_a = 0, 0

                for _ in range(n_test):
                    x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
                    obs_fb = observe(tap_fb, {"x": x})
                    obs_p = observe(tap_p, {"x": x})
                    obs_a = observe(tap_a, {"x": x})

                    h_p += hellinger_distance(obs_p.probabilities.data, obs_fb.probabilities.data)
                    h_a += hellinger_distance(obs_a.probabilities.data, obs_fb.probabilities.data)
                    if obs_p.value_index == obs_fb.value_index:
                        m_p += 1
                    if obs_a.value_index == obs_fb.value_index:
                        m_a += 1

                print("  %8d | %10.4f %10.4f | %9.1f%% %9.1f%%" % (
                    depth,
                    h_p / n_test, h_a / n_test,
                    m_p / n_test * 100, m_a / n_test * 100,
                ))

            print("  " + "-" * 60)

        print("=" * 90)


# ===================================================================
# Experiment D: Raw EDMD accuracy -- poly vs augmented (no pipeline)
# ===================================================================

class TestAugmentedEDMD:
    """Direct EDMD accuracy comparison without pipeline overhead."""

    def test_edmd_comparison(self):
        dim = 4
        n_test = 50
        rng = np.random.default_rng(42)
        A = rng.standard_normal((dim, dim)) * 0.3

        funcs = [
            ("sigmoid",        lambda x: _sigmoid(x @ A)),
            ("relu",           lambda x: _relu(x @ A)),
            ("abs",            lambda x: _abs_fn(x @ A)),
            ("hard_threshold", lambda x: _hard_threshold(x @ A)),
        ]

        print()
        print("=" * 80)
        print("  Raw EDMD Accuracy: poly vs augmented (dim=%d, 1000 samples)" % dim)
        print("=" * 80)
        print("  %-16s | %12s %12s | %12s" % (
            "Function", "MSE(poly)", "MSE(aug)", "Improvement",
        ))
        print("  " + "-" * 66)

        for fname, fn in funcs:
            K_p = estimate_koopman_matrix(fn, dim, n_samples=1000, seed=42, basis="poly")
            K_a = estimate_koopman_matrix(fn, dim, n_samples=1000, seed=42, basis="augmented")

            mse_p, mse_a = 0.0, 0.0
            for _ in range(n_test):
                x = rng.standard_normal(dim) * 0.3
                expected = fn(x)

                pred_p = unlift(lift(x) @ K_p.data.astype(np.float64), dim)
                pred_a = unlift(
                    lift(x, basis="augmented") @ K_a.data.astype(np.float64),
                    dim, basis="augmented",
                )

                mse_p += np.mean((pred_p - expected) ** 2)
                mse_a += np.mean((pred_a - expected) ** 2)

            mse_p /= n_test
            mse_a /= n_test
            improvement = (mse_p - mse_a) / max(mse_p, 1e-10) * 100

            print("  %-16s | %12.6f %12.6f | %+10.1f%%" % (
                fname, mse_p, mse_a, improvement,
            ))

        print("=" * 80)
