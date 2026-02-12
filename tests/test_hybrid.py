"""Tests for Hybrid (unitary rotation + singular-value scales) composition."""

from __future__ import annotations

import numpy as np
import pytest

from axol.core.types import FloatVec, TransMatrix
from axol.quantum.unitary import (
    estimate_hybrid_step,
    compose_hybrid_chain,
    nearest_unitary,
)
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe, reobserve


# ===================================================================
# Unit Tests -- hybrid math in unitary.py
# ===================================================================


class TestEstimateHybridStep:
    def test_identity_function(self):
        """Identity function should produce near-identity raw matrix."""
        dim = 4
        A = estimate_hybrid_step(lambda x: x, dim, n_samples=1000, seed=42)
        assert A.shape == (dim, dim)
        # NOT projected to unitary -- should be close to identity
        np.testing.assert_allclose(A.data, np.eye(dim, dtype=np.float32), atol=0.15)

    def test_shape_matches_dim(self):
        for dim in [1, 4, 8]:
            A = estimate_hybrid_step(lambda x: x, dim, n_samples=100, seed=42)
            assert A.shape == (dim, dim)

    def test_not_unitary(self):
        """Hybrid step should NOT be projected to unitary (preserves magnitude)."""
        dim = 4

        def sigmoid_fn(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

        A = estimate_hybrid_step(sigmoid_fn, dim, n_samples=500, seed=42)
        UUT = A.data.astype(np.float64) @ A.data.astype(np.float64).T
        # If it were unitary, UUT would equal I. It should NOT.
        diff = np.max(np.abs(UUT - np.eye(dim)))
        assert diff > 0.01, "Hybrid step should not be unitary"


class TestComposeHybridChain:
    def test_single_matrix(self):
        """Single matrix -> SVD decomposition returns composed + rotation + scales."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 4)).astype(np.float32) * 0.5 + np.eye(4, dtype=np.float32)
        composed, rotation, scales = compose_hybrid_chain([TransMatrix(data=A)])
        assert composed.shape == (4, 4)
        assert rotation.shape == (4, 4)
        assert scales.shape == (4,)
        # composed should be close to original
        np.testing.assert_allclose(composed.data, A, atol=1e-5)
        # rotation should be unitary
        R = rotation.data.astype(np.float64)
        np.testing.assert_allclose(R @ R.T, np.eye(4), atol=1e-5)
        # scales should be non-negative
        assert np.all(scales >= 0)

    def test_two_matrices(self):
        """Composing two matrices should produce valid rotation + scales."""
        rng = np.random.default_rng(1)
        A1 = rng.standard_normal((5, 5)).astype(np.float32) * 0.3
        A2 = rng.standard_normal((5, 5)).astype(np.float32) * 0.3
        composed, rotation, scales = compose_hybrid_chain(
            [TransMatrix(data=A1), TransMatrix(data=A2)]
        )
        assert composed.shape == (5, 5)
        assert rotation.shape == (5, 5)
        assert scales.shape == (5,)
        R = rotation.data.astype(np.float64)
        np.testing.assert_allclose(R @ R.T, np.eye(5), atol=1e-5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compose_hybrid_chain([])

    def test_scales_preserve_magnitude_info(self):
        """Scales from a scaled identity should reflect the scale factor."""
        scale = 3.0
        A = np.eye(4, dtype=np.float32) * scale
        _, rotation, scales = compose_hybrid_chain([TransMatrix(data=A)])
        # All singular values should be close to 3.0
        np.testing.assert_allclose(scales, scale, atol=0.01)

    def test_rotation_is_direction_only(self):
        """Rotation from SVD should be unitary (direction, no magnitude)."""
        rng = np.random.default_rng(99)
        matrices = [
            TransMatrix(data=rng.standard_normal((4, 4)).astype(np.float32))
            for _ in range(5)
        ]
        _, rotation, scales = compose_hybrid_chain(matrices)
        R = rotation.data.astype(np.float64)
        np.testing.assert_allclose(R @ R.T, np.eye(4), atol=1e-5)


# ===================================================================
# Integration Tests -- weave + observe with Hybrid
# ===================================================================


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _relu(x):
    return np.maximum(x, 0.0)


def _hard_threshold(x):
    return (x > 0).astype(np.float64)


def _build_nonlinear_chain(depth: int = 3, dim: int = 4, transform_fn=None):
    if transform_fn is None:
        transform_fn = _sigmoid
    builder = DeclarationBuilder("nonlinear_chain")
    builder.input("x", dim)
    prev = "x"
    for i in range(depth):
        node = f"h{i}"
        builder.relate(target=node, sources=[prev],
                       kind=RelationKind.PROPORTIONAL, transform_fn=transform_fn)
        prev = node
    builder.output(prev)
    return builder.build()


def _build_linear_chain(depth: int = 3, dim: int = 4):
    builder = DeclarationBuilder("linear_chain")
    builder.input("x", dim)
    prev = "x"
    for i in range(depth):
        node = f"h{i}"
        builder.relate(target=node, sources=[prev], kind=RelationKind.PROPORTIONAL)
        prev = node
    builder.output(prev)
    return builder.build()


class TestWeaveHybrid:
    def test_nonlinear_produces_hybrid(self):
        """Explicit nonlinear_method='hybrid' should set hybrid fields."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")
        assert tapestry._hybrid_matrix is not None
        assert tapestry._hybrid_rotation is not None
        assert tapestry._hybrid_scales is not None
        assert tapestry._hybrid_chain_info is not None
        assert tapestry._unitary_matrix is None
        assert tapestry._koopman_matrix is None
        assert tapestry._composed_matrix is None

    def test_linear_uses_composed_not_hybrid(self):
        """Pure linear chain should use composed matrix regardless of default."""
        decl = _build_linear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")
        assert tapestry._composed_matrix is not None
        assert tapestry._hybrid_rotation is None

    def test_hybrid_chain_info_fields(self):
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")
        info = tapestry._hybrid_chain_info
        assert info is not None
        assert "input_key" in info
        assert "output_key" in info
        assert "num_composed" in info
        assert "dim" in info
        assert info["dim"] == 4
        assert info["num_composed"] == 2

    def test_hybrid_rotation_shape(self):
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")
        assert tapestry._hybrid_rotation.shape == (4, 4)

    def test_hybrid_scales_shape(self):
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")
        assert tapestry._hybrid_scales.shape == (4,)

    def test_hybrid_rotation_is_unitary(self):
        """The rotation component should be unitary."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")
        R = tapestry._hybrid_rotation.data.astype(np.float64)
        np.testing.assert_allclose(R @ R.T, np.eye(4), atol=1e-5)

    def test_hybrid_scales_nonnegative(self):
        """Singular values should be non-negative."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")
        assert np.all(tapestry._hybrid_scales >= 0)

    def test_optimize_false_skips_hybrid(self):
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, optimize=False)
        assert tapestry._hybrid_matrix is None
        assert tapestry._hybrid_rotation is None
        assert tapestry._hybrid_scales is None
        assert tapestry._hybrid_chain_info is None

    def test_explicit_hybrid_method(self):
        """Explicit nonlinear_method='hybrid' should produce hybrid fields."""
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tap_explicit = weave(decl, seed=42, nonlinear_method="hybrid")
        assert tap_explicit._hybrid_rotation is not None
        assert tap_explicit._hybrid_matrix is not None


class TestObserveHybrid:
    def test_hybrid_observe_runs(self):
        """Hybrid observe should return a valid Observation."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")

        x = FloatVec.from_list([0.3, -0.1, 0.5, 0.2])
        obs = observe(tapestry, {"x": x})
        assert obs is not None
        assert obs.value.size == 4
        assert 0 <= obs.value_index < 4
        assert not np.any(np.isnan(obs.value.data))
        assert not np.any(np.isnan(obs.probabilities.data))

    def test_probabilities_sum_to_one(self):
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")

        rng = np.random.default_rng(55)
        for _ in range(10):
            x = FloatVec(data=rng.standard_normal(4).astype(np.float32) * 0.5)
            obs = observe(tapestry, {"x": x})
            np.testing.assert_allclose(np.sum(obs.probabilities.data), 1.0, atol=1e-5)

    def test_reobserve_with_hybrid(self):
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")
        x = FloatVec.from_list([0.3, -0.1, 0.5, 0.2])
        obs = reobserve(tapestry, {"x": x}, count=5, seed=42)
        assert obs.observation_count == 5

    def test_hybrid_vs_fallback_accuracy(self):
        """Hybrid should have better accuracy than pure unitary (lower H-dist)."""
        decl = _build_nonlinear_chain(depth=3, dim=4, transform_fn=_sigmoid)

        tap_h = weave(decl, seed=42, nonlinear_method="hybrid")
        tap_u = weave(decl, seed=42, nonlinear_method="unitary")
        tap_f = weave(decl, seed=42, optimize=False)

        rng = np.random.default_rng(77)
        h_hybrid, h_unitary = 0.0, 0.0
        N = 30
        for _ in range(N):
            x = FloatVec(data=rng.standard_normal(4).astype(np.float32) * 0.3)
            obs_h = observe(tap_h, {"x": x})
            obs_u = observe(tap_u, {"x": x})
            obs_f = observe(tap_f, {"x": x})
            p_f = obs_f.probabilities.data.astype(np.float64)
            p_h = obs_h.probabilities.data.astype(np.float64)
            p_u = obs_u.probabilities.data.astype(np.float64)
            # Hellinger distance
            p_f = p_f / max(p_f.sum(), 1e-12)
            p_h = p_h / max(p_h.sum(), 1e-12)
            p_u = p_u / max(p_u.sum(), 1e-12)
            h_hybrid += float(np.sqrt(0.5 * np.sum((np.sqrt(p_h) - np.sqrt(p_f)) ** 2)))
            h_unitary += float(np.sqrt(0.5 * np.sum((np.sqrt(p_u) - np.sqrt(p_f)) ** 2)))

        h_hybrid /= N
        h_unitary /= N
        # Hybrid should be better (lower H-dist) than pure unitary
        assert h_hybrid <= h_unitary + 0.05, (
            f"Hybrid H-dist {h_hybrid:.4f} should be <= Unitary H-dist {h_unitary:.4f}"
        )


# ===================================================================
# Edge Cases
# ===================================================================


class TestHybridEdgeCases:
    def test_dim_1(self):
        builder = DeclarationBuilder("dim1")
        builder.input("x", 1)
        builder.relate("y", ["x"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.output("y")
        decl = builder.build()

        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")
        assert tapestry._hybrid_rotation is not None
        assert tapestry._hybrid_rotation.shape == (1, 1)
        assert tapestry._hybrid_scales.shape == (1,)

        x = FloatVec.from_list([0.5])
        obs = observe(tapestry, {"x": x})
        assert obs is not None

    def test_dim_16(self):
        builder = DeclarationBuilder("dim16")
        builder.input("x", 16)
        builder.relate("y", ["x"], RelationKind.PROPORTIONAL, transform_fn=_relu)
        builder.output("y")
        decl = builder.build()

        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")
        assert tapestry._hybrid_rotation is not None
        assert tapestry._hybrid_rotation.shape == (16, 16)
        assert tapestry._hybrid_scales.shape == (16,)

    def test_nan_producing_transform_fn(self):
        def bad_fn(x):
            return np.exp(x * 100)

        builder = DeclarationBuilder("bad_fn")
        builder.input("x", 4)
        builder.relate("y", ["x"], RelationKind.PROPORTIONAL, transform_fn=bad_fn)
        builder.output("y")
        decl = builder.build()

        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")
        assert tapestry._hybrid_rotation is not None

        x = FloatVec.from_list([0.1, -0.1, 0.05, -0.05])
        obs = observe(tapestry, {"x": x})
        assert not np.any(np.isnan(obs.value.data))
        assert not np.any(np.isnan(obs.probabilities.data))

    def test_multi_source_no_hybrid(self):
        """Multi-source should fall back, not use hybrid."""
        builder = DeclarationBuilder("multi_source")
        builder.input("x", 4)
        builder.input("y", 4)
        builder.relate("z", ["x", "y"], RelationKind.ADDITIVE, transform_fn=_sigmoid)
        builder.output("z")
        decl = builder.build()

        tapestry = weave(decl, seed=42, nonlinear_method="hybrid")
        assert tapestry._hybrid_matrix is None
        assert tapestry._hybrid_rotation is None
        assert tapestry._unitary_matrix is None
        assert tapestry._koopman_matrix is None
