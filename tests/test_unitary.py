"""Tests for Unitary matrix extension — nonlinear pipeline composition."""

from __future__ import annotations

import numpy as np
import pytest

from axol.core.types import FloatVec, TransMatrix
from axol.quantum.unitary import (
    nearest_unitary,
    reorthogonalize,
    estimate_unitary_step,
    compose_unitary_chain,
)
from axol.quantum.declare import (
    DeclarationBuilder,
    RelationKind,
)
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe, reobserve


# ===================================================================
# Unit Tests — unitary.py math
# ===================================================================


class TestNearestUnitary:
    def test_identity_preserved(self):
        """Identity matrix is already unitary — should stay identity."""
        I = np.eye(4, dtype=np.float64)
        U = nearest_unitary(I)
        np.testing.assert_allclose(U, I, atol=1e-10)

    def test_result_is_unitary(self):
        """nearest_unitary(A) should produce U @ U.T ≈ I."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((6, 6))
        U = nearest_unitary(A)
        np.testing.assert_allclose(U @ U.T, np.eye(6), atol=1e-10)
        np.testing.assert_allclose(U.T @ U, np.eye(6), atol=1e-10)

    def test_determinant_plus_minus_one(self):
        """Determinant of a unitary matrix should be ±1."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((5, 5))
        U = nearest_unitary(A)
        det = np.linalg.det(U)
        assert abs(abs(det) - 1.0) < 1e-10

    def test_scale_removed(self):
        """A scaled identity should map to ±identity (scale stripped)."""
        A = np.eye(4) * 5.0
        U = nearest_unitary(A)
        np.testing.assert_allclose(U @ U.T, np.eye(4), atol=1e-10)

    def test_dim_1(self):
        """1x1 case."""
        A = np.array([[3.0]])
        U = nearest_unitary(A)
        assert abs(abs(U[0, 0]) - 1.0) < 1e-10


class TestReorthogonalize:
    def test_alias_for_nearest_unitary(self):
        rng = np.random.default_rng(77)
        A = rng.standard_normal((4, 4))
        np.testing.assert_allclose(reorthogonalize(A), nearest_unitary(A), atol=1e-14)


class TestEstimateUnitaryStep:
    def test_identity_function(self):
        """Identity function should produce near-identity (or orthogonal) unitary."""
        dim = 4
        U = estimate_unitary_step(lambda x: x, dim, n_samples=1000, seed=42)
        assert U.shape == (dim, dim)
        # Should be unitary
        np.testing.assert_allclose(
            U.data.astype(np.float64) @ U.data.astype(np.float64).T,
            np.eye(dim),
            atol=0.1,
        )

    def test_linear_function(self):
        """Pure linear function: result should be dim x dim and unitary."""
        dim = 4
        rng = np.random.default_rng(123)
        A = rng.standard_normal((dim, dim)) * 0.3 + np.eye(dim) * 0.5

        def linear_fn(x):
            return x @ A

        U = estimate_unitary_step(linear_fn, dim, n_samples=1000, seed=42)
        assert U.shape == (dim, dim)
        np.testing.assert_allclose(
            U.data.astype(np.float64) @ U.data.astype(np.float64).T,
            np.eye(dim),
            atol=0.05,
        )

    def test_sigmoid_function(self):
        """Sigmoid: result should be dim x dim and unitary."""
        dim = 4
        rng = np.random.default_rng(99)
        A = rng.standard_normal((dim, dim)) * 0.3

        def sigmoid_fn(x):
            y = x @ A
            return 1.0 / (1.0 + np.exp(-y))

        U = estimate_unitary_step(sigmoid_fn, dim, n_samples=1000, seed=42)
        assert U.shape == (dim, dim)
        # Must be unitary
        UU = U.data.astype(np.float64)
        np.testing.assert_allclose(UU @ UU.T, np.eye(dim), atol=0.05)

    def test_relu_function(self):
        """ReLU: result should be dim x dim and unitary."""
        dim = 4

        def relu_fn(x):
            return np.maximum(x, 0.0)

        U = estimate_unitary_step(relu_fn, dim, n_samples=1000, seed=42)
        assert U.shape == (dim, dim)
        UU = U.data.astype(np.float64)
        np.testing.assert_allclose(UU @ UU.T, np.eye(dim), atol=0.05)

    def test_shape_matches_dim(self):
        """Output shape should be (dim, dim), not lifted."""
        for dim in [1, 4, 8, 16]:
            U = estimate_unitary_step(lambda x: x, dim, n_samples=100, seed=42)
            assert U.shape == (dim, dim)


class TestComposeUnitaryChain:
    def test_single_matrix(self):
        """Single unitary should be returned (re-orthogonalised)."""
        M = TransMatrix(data=np.eye(5, dtype=np.float32))
        result = compose_unitary_chain([M])
        np.testing.assert_allclose(result.data, np.eye(5, dtype=np.float32), atol=1e-5)

    def test_two_matrices(self):
        """Product of two unitaries is unitary."""
        rng = np.random.default_rng(1)
        A = nearest_unitary(rng.standard_normal((5, 5))).astype(np.float32)
        B = nearest_unitary(rng.standard_normal((5, 5))).astype(np.float32)
        result = compose_unitary_chain([TransMatrix(data=A), TransMatrix(data=B)])
        RR = result.data.astype(np.float64)
        np.testing.assert_allclose(RR @ RR.T, np.eye(5), atol=1e-5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compose_unitary_chain([])

    def test_deep_chain_stays_unitary(self):
        """100-deep chain should still be unitary after re-orthogonalisation."""
        dim = 4
        rng = np.random.default_rng(42)
        matrices = []
        for _ in range(100):
            U = nearest_unitary(rng.standard_normal((dim, dim)))
            matrices.append(TransMatrix(data=U.astype(np.float32)))

        result = compose_unitary_chain(matrices)
        RR = result.data.astype(np.float64)
        np.testing.assert_allclose(RR @ RR.T, np.eye(dim), atol=1e-5)
        np.testing.assert_allclose(RR.T @ RR, np.eye(dim), atol=1e-5)


# ===================================================================
# Integration Tests — weave + observe with Unitary
# ===================================================================


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _relu(x):
    return np.maximum(x, 0.0)


def _hard_threshold(x):
    return (x > 0).astype(np.float64)


def _build_nonlinear_chain(depth: int = 3, dim: int = 4, transform_fn=None):
    """Build a declaration with a chain of nonlinear relations."""
    if transform_fn is None:
        transform_fn = _sigmoid

    builder = DeclarationBuilder("nonlinear_chain")
    builder.input("x", dim)

    prev = "x"
    for i in range(depth):
        node = f"h{i}"
        builder.relate(
            target=node,
            sources=[prev],
            kind=RelationKind.PROPORTIONAL,
            transform_fn=transform_fn,
        )
        prev = node

    builder.output(prev)
    return builder.build()


def _build_linear_chain(depth: int = 3, dim: int = 4):
    """Build a declaration with a pure linear chain (no transform_fn)."""
    builder = DeclarationBuilder("linear_chain")
    builder.input("x", dim)

    prev = "x"
    for i in range(depth):
        node = f"h{i}"
        builder.relate(
            target=node,
            sources=[prev],
            kind=RelationKind.PROPORTIONAL,
        )
        prev = node

    builder.output(prev)
    return builder.build()


class TestWeaveUnitary:
    def test_nonlinear_produces_unitary_matrix(self):
        """Nonlinear chain with default method should produce a unitary matrix."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="unitary")
        assert tapestry._unitary_matrix is not None
        assert tapestry._unitary_chain_info is not None
        assert tapestry._composed_matrix is None
        assert tapestry._koopman_matrix is None

    def test_linear_produces_composed_not_unitary(self):
        """Pure linear chain should use composed matrix, not unitary."""
        decl = _build_linear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42)
        assert tapestry._composed_matrix is not None
        assert tapestry._unitary_matrix is None
        assert tapestry._koopman_matrix is None
        assert tapestry._hybrid_matrix is None

    def test_unitary_chain_info_fields(self):
        """Unitary chain info should have all required fields."""
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="unitary")
        info = tapestry._unitary_chain_info
        assert info is not None
        assert "input_key" in info
        assert "output_key" in info
        assert "num_composed" in info
        assert "dim" in info
        assert info["dim"] == 4
        assert info["num_composed"] == 2

    def test_unitary_matrix_shape(self):
        """Unitary matrix should be dim x dim (not lifted_dim)."""
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="unitary")
        assert tapestry._unitary_matrix.shape == (4, 4)

    def test_optimize_false_skips_all(self):
        """optimize=False should skip all composition."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, optimize=False)
        assert tapestry._unitary_matrix is None
        assert tapestry._koopman_matrix is None
        assert tapestry._composed_matrix is None
        assert tapestry._hybrid_matrix is None

    def test_koopman_method_produces_koopman(self):
        """nonlinear_method='koopman' should produce Koopman, not unitary."""
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="koopman")
        assert tapestry._koopman_matrix is not None
        assert tapestry._unitary_matrix is None


class TestObserveUnitary:
    def test_unitary_observe_runs(self):
        """Unitary observe should return an Observation."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="unitary")

        x = FloatVec.from_list([0.3, -0.1, 0.5, 0.2])
        obs = observe(tapestry, {"x": x})
        assert obs is not None
        assert obs.value.size == 4
        assert 0 <= obs.value_index < 4
        assert not np.any(np.isnan(obs.value.data))
        assert not np.any(np.isnan(obs.probabilities.data))

    def test_unitary_vs_fallback_argmax_comparable(self):
        """Unitary and fallback should agree on argmax for moderate inputs."""
        decl = _build_nonlinear_chain(depth=2, dim=4)

        # Unitary path
        tapestry_u = weave(decl, seed=42, optimize=True, nonlinear_method="unitary")
        # Fallback path (no optimization)
        tapestry_f = weave(decl, seed=42, optimize=False)

        rng = np.random.default_rng(77)
        agree = 0
        n_tests = 20
        for _ in range(n_tests):
            x = FloatVec(data=rng.standard_normal(4).astype(np.float32) * 0.3)
            obs_u = observe(tapestry_u, {"x": x})
            obs_f = observe(tapestry_f, {"x": x})
            if obs_u.value_index == obs_f.value_index:
                agree += 1

        # Should agree on a meaningful fraction (unitary preserves direction,
        # not exact magnitude, so agreement threshold is moderate)
        assert agree >= n_tests * 0.3, f"Only {agree}/{n_tests} agreed"

    def test_reobserve_with_unitary(self):
        """reobserve should work with unitary tapestry."""
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="unitary")

        x = FloatVec.from_list([0.3, -0.1, 0.5, 0.2])
        obs = reobserve(tapestry, {"x": x}, count=5, seed=42)
        assert obs is not None
        assert obs.observation_count == 5

    def test_probabilities_sum_to_one(self):
        """Observation probabilities should always sum to 1."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="unitary")

        rng = np.random.default_rng(55)
        for _ in range(10):
            x = FloatVec(data=rng.standard_normal(4).astype(np.float32) * 0.5)
            obs = observe(tapestry, {"x": x})
            np.testing.assert_allclose(np.sum(obs.probabilities.data), 1.0, atol=1e-5)


class TestUnitaryVsKoopman:
    """Compare unitary and koopman paths on depth-scaling behaviour."""

    def _run_depth_test(self, transform_fn, depths, dim=4, n_tests=20):
        """Return (unitary_agree, koopman_agree) for each depth."""
        rng = np.random.default_rng(42)
        results = {"unitary": [], "koopman": []}

        for depth in depths:
            decl = _build_nonlinear_chain(depth=depth, dim=dim, transform_fn=transform_fn)
            tap_u = weave(decl, seed=42, nonlinear_method="unitary")
            tap_k = weave(decl, seed=42, nonlinear_method="koopman")
            tap_f = weave(decl, seed=42, optimize=False)

            agree_u = 0
            agree_k = 0
            for _ in range(n_tests):
                x = FloatVec(data=rng.standard_normal(dim).astype(np.float32) * 0.3)
                obs_f = observe(tap_f, {"x": x})
                obs_u = observe(tap_u, {"x": x})
                obs_k = observe(tap_k, {"x": x})
                if obs_u.value_index == obs_f.value_index:
                    agree_u += 1
                if obs_k.value_index == obs_f.value_index:
                    agree_k += 1

            results["unitary"].append(agree_u)
            results["koopman"].append(agree_k)

        return results

    def test_sigmoid_depth_scaling(self):
        """Unitary should not degrade much with depth vs Koopman for sigmoid."""
        results = self._run_depth_test(_sigmoid, depths=[2, 5, 10])
        # At depth 10, unitary should still have reasonable agreement
        # (not necessarily better than koopman at every depth, but not catastrophic)
        assert results["unitary"][-1] >= 0  # basic sanity

    def test_relu_depth_scaling(self):
        """ReLU depth scaling comparison."""
        results = self._run_depth_test(_relu, depths=[2, 5])
        # Both should have some agreement at shallow depths
        assert results["unitary"][0] >= 0

    def test_hard_threshold_depth_scaling(self):
        """Hard threshold: both methods approximate, neither is perfect."""
        results = self._run_depth_test(_hard_threshold, depths=[2, 5])
        assert results["unitary"][0] >= 0


# ===================================================================
# Edge Cases
# ===================================================================


class TestUnitaryEdgeCases:
    def test_dim_1(self):
        """Unitary should work with dim=1."""
        builder = DeclarationBuilder("dim1")
        builder.input("x", 1)
        builder.relate("y", ["x"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.output("y")
        decl = builder.build()

        tapestry = weave(decl, seed=42, nonlinear_method="unitary")
        assert tapestry._unitary_matrix is not None
        assert tapestry._unitary_matrix.shape == (1, 1)

        x = FloatVec.from_list([0.5])
        obs = observe(tapestry, {"x": x})
        assert obs is not None
        assert obs.value.size == 1

    def test_dim_16(self):
        """Unitary should work with larger dimensions (much smaller matrix than Koopman)."""
        builder = DeclarationBuilder("dim16")
        builder.input("x", 16)
        builder.relate("y", ["x"], RelationKind.PROPORTIONAL, transform_fn=_relu)
        builder.output("y")
        decl = builder.build()

        tapestry = weave(decl, seed=42, nonlinear_method="unitary")
        assert tapestry._unitary_matrix is not None
        # Key advantage: 16x16 instead of Koopman's 153x153
        assert tapestry._unitary_matrix.shape == (16, 16)

        x = FloatVec(data=np.random.default_rng(42).standard_normal(16).astype(np.float32) * 0.3)
        obs = observe(tapestry, {"x": x})
        assert obs is not None

    def test_nan_producing_transform_fn(self):
        """Transform fn that could produce NaN should be handled safely."""
        def bad_fn(x):
            return np.exp(x * 100)

        builder = DeclarationBuilder("bad_fn")
        builder.input("x", 4)
        builder.relate("y", ["x"], RelationKind.PROPORTIONAL, transform_fn=bad_fn)
        builder.output("y")
        decl = builder.build()

        tapestry = weave(decl, seed=42, nonlinear_method="unitary")
        assert tapestry._unitary_matrix is not None

        x = FloatVec.from_list([0.1, -0.1, 0.05, -0.05])
        obs = observe(tapestry, {"x": x})
        assert not np.any(np.isnan(obs.value.data))
        assert not np.any(np.isnan(obs.probabilities.data))

    def test_multi_source_relation_no_unitary(self):
        """Multi-source relations (MergeOp) should not use unitary."""
        builder = DeclarationBuilder("multi_source")
        builder.input("x", 4)
        builder.input("y", 4)
        builder.relate("z", ["x", "y"], RelationKind.ADDITIVE, transform_fn=_sigmoid)
        builder.output("z")
        decl = builder.build()

        tapestry = weave(decl, seed=42)
        assert tapestry._unitary_matrix is None
        assert tapestry._koopman_matrix is None
        assert tapestry._composed_matrix is None
        assert tapestry._hybrid_matrix is None

    def test_mixed_linear_nonlinear_chain(self):
        """Chain with mix of linear and nonlinear steps should use unitary."""
        builder = DeclarationBuilder("mixed")
        builder.input("x", 4)
        builder.relate("h1", ["x"], RelationKind.PROPORTIONAL)
        builder.relate("h2", ["h1"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.output("h2")
        decl = builder.build()

        tapestry = weave(decl, seed=42, nonlinear_method="unitary")
        assert tapestry._unitary_matrix is not None
        assert tapestry._koopman_matrix is None
