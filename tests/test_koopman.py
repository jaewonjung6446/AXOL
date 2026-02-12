"""Tests for Koopman operator extension — nonlinear pipeline composition."""

from __future__ import annotations

import numpy as np
import pytest

from axol.core.types import FloatVec, TransMatrix
from axol.quantum.koopman import (
    lifted_dim,
    lift,
    unlift,
    estimate_koopman_matrix,
    compose_koopman_chain,
)
from axol.quantum.declare import (
    DeclarationBuilder,
    RelationKind,
)
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe, reobserve


# ===================================================================
# Unit Tests — koopman.py math
# ===================================================================


class TestLiftedDim:
    def test_dim_1(self):
        # 1 + 1 + 1*(1+1)//2 = 1 + 1 + 1 = 3
        assert lifted_dim(1) == 3

    def test_dim_4(self):
        # 1 + 4 + 4*5//2 = 1 + 4 + 10 = 15
        assert lifted_dim(4) == 15

    def test_dim_8(self):
        # 1 + 8 + 8*9//2 = 1 + 8 + 36 = 45
        assert lifted_dim(8) == 45

    def test_dim_16(self):
        # 1 + 16 + 16*17//2 = 1 + 16 + 136 = 153
        assert lifted_dim(16) == 153

    def test_degree_3(self):
        # 1 + 4 + 10 + C(6,3) = 1 + 4 + 10 + 20 = 35
        assert lifted_dim(4, degree=3) == 35

    def test_degree_4(self):
        # 1 + 4 + 10 + 20 + C(7,4) = 35 + 35 = 70
        assert lifted_dim(4, degree=4) == 70


class TestLift:
    def test_1d_structure(self):
        """Check that lift produces [1, x1, x2, x1^2, x1*x2, x2^2]."""
        x = np.array([2.0, 3.0])
        psi = lift(x)
        expected = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 9.0])
        np.testing.assert_allclose(psi, expected, atol=1e-10)

    def test_constant_term(self):
        x = np.array([0.0, 0.0, 0.0])
        psi = lift(x)
        assert psi[0] == 1.0

    def test_linear_terms(self):
        x = np.array([1.5, -2.0, 3.0])
        psi = lift(x)
        np.testing.assert_allclose(psi[1:4], x, atol=1e-10)

    def test_quadratic_terms(self):
        x = np.array([2.0, 3.0])
        psi = lift(x)
        # Quadratic: x1^2, x1*x2, x2^2
        np.testing.assert_allclose(psi[3:], [4.0, 6.0, 9.0], atol=1e-10)

    def test_batch(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Psi = lift(X)
        assert Psi.shape == (2, lifted_dim(2))
        # First row
        np.testing.assert_allclose(Psi[0], [1, 1, 2, 1, 2, 4], atol=1e-10)
        # Second row
        np.testing.assert_allclose(Psi[1], [1, 3, 4, 9, 12, 16], atol=1e-10)

    def test_dim_1(self):
        x = np.array([5.0])
        psi = lift(x)
        np.testing.assert_allclose(psi, [1.0, 5.0, 25.0], atol=1e-10)


class TestUnlift:
    def test_extracts_linear(self):
        """Unlift should extract indices [1:dim+1]."""
        psi = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 9.0])
        result = unlift(psi, dim=2)
        np.testing.assert_allclose(result, [2.0, 3.0], atol=1e-10)

    def test_roundtrip(self):
        """unlift(lift(x)) == x for the linear terms."""
        x = np.array([1.5, -2.0, 3.0, 0.5])
        psi = lift(x)
        recovered = unlift(psi, dim=4)
        np.testing.assert_allclose(recovered, x, atol=1e-10)

    def test_batch_roundtrip(self):
        X = np.random.default_rng(42).standard_normal((10, 4))
        Psi = lift(X)
        recovered = unlift(Psi, dim=4)
        np.testing.assert_allclose(recovered, X, atol=1e-10)


class TestEstimateKoopmanMatrix:
    def test_identity_function(self):
        """Identity function should produce near-identity Koopman matrix."""
        dim = 4
        K = estimate_koopman_matrix(lambda x: x, dim, n_samples=1000, seed=42)
        ld = lifted_dim(dim)
        assert K.shape == (ld, ld)
        # The Koopman matrix of identity is the identity
        np.testing.assert_allclose(K.data, np.eye(ld, dtype=np.float32), atol=0.1)

    def test_linear_function(self):
        """Pure linear function: Koopman should reproduce it accurately."""
        dim = 4
        rng = np.random.default_rng(123)
        A = rng.standard_normal((dim, dim)) * 0.3 + np.eye(dim) * 0.5
        A = A.astype(np.float64)

        def linear_fn(x):
            return x @ A

        K = estimate_koopman_matrix(linear_fn, dim, n_samples=1000, seed=42)

        # Test prediction accuracy
        test_x = rng.standard_normal(dim) * 0.5
        expected = linear_fn(test_x)

        psi_x = lift(test_x)
        psi_y = psi_x @ K.data.astype(np.float64)
        predicted = unlift(psi_y, dim)

        np.testing.assert_allclose(predicted, expected, atol=0.05)

    def test_sigmoid_approximation(self):
        """Sigmoid applied after linear: reasonable approximation."""
        dim = 4
        rng = np.random.default_rng(99)
        A = rng.standard_normal((dim, dim)) * 0.3

        def sigmoid_fn(x):
            y = x @ A
            return 1.0 / (1.0 + np.exp(-y))

        K = estimate_koopman_matrix(sigmoid_fn, dim, n_samples=1000, seed=42)

        # Test on a few points
        for i in range(5):
            test_x = rng.standard_normal(dim) * 0.3
            expected = sigmoid_fn(test_x)
            psi_x = lift(test_x)
            psi_y = psi_x @ K.data.astype(np.float64)
            predicted = unlift(psi_y, dim)
            np.testing.assert_allclose(predicted, expected, atol=0.15)

    def test_output_shape(self):
        dim = 3
        K = estimate_koopman_matrix(lambda x: x, dim)
        ld = lifted_dim(dim)
        assert K.shape == (ld, ld)


class TestComposeKoopmanChain:
    def test_single_matrix(self):
        M = TransMatrix(data=np.eye(5, dtype=np.float32))
        result = compose_koopman_chain([M])
        np.testing.assert_allclose(result.data, np.eye(5, dtype=np.float32), atol=1e-5)

    def test_two_matrices(self):
        A = np.random.default_rng(1).standard_normal((5, 5)).astype(np.float32)
        B = np.random.default_rng(2).standard_normal((5, 5)).astype(np.float32)
        result = compose_koopman_chain([TransMatrix(data=A), TransMatrix(data=B)])
        expected = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
        np.testing.assert_allclose(result.data, expected, atol=1e-4)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compose_koopman_chain([])


# ===================================================================
# Integration Tests — weave + observe with Koopman
# ===================================================================


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _relu(x):
    return np.maximum(x, 0.0)


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


class TestWeaveKoopman:
    def test_nonlinear_produces_koopman_matrix(self):
        """Nonlinear chain should produce a Koopman matrix."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="koopman")
        assert tapestry._koopman_matrix is not None
        assert tapestry._koopman_chain_info is not None
        assert tapestry._composed_matrix is None  # Not linear

    def test_linear_produces_composed_not_koopman(self):
        """Pure linear chain should use composed matrix, not Koopman."""
        decl = _build_linear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42)
        assert tapestry._composed_matrix is not None
        assert tapestry._koopman_matrix is None

    def test_koopman_chain_info_fields(self):
        """Koopman chain info should have all required fields."""
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="koopman")
        info = tapestry._koopman_chain_info
        assert info is not None
        assert "input_key" in info
        assert "output_key" in info
        assert "num_composed" in info
        assert "original_dim" in info
        assert "lifted_dim" in info
        assert "degree" in info
        assert info["original_dim"] == 4
        assert info["degree"] == 2
        assert info["num_composed"] == 2

    def test_optimize_false_skips_koopman(self):
        """optimize=False should skip Koopman composition."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, optimize=False)
        assert tapestry._koopman_matrix is None
        assert tapestry._composed_matrix is None


class TestObserveKoopman:
    def test_koopman_observe_runs(self):
        """Koopman observe should return an Observation."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="koopman")

        x = FloatVec.from_list([0.3, -0.1, 0.5, 0.2])
        obs = observe(tapestry, {"x": x})
        assert obs is not None
        assert obs.value.size == 4
        assert 0 <= obs.value_index < 4

    def test_koopman_vs_fallback_argmax_comparable(self):
        """Koopman and fallback should agree on argmax for moderate inputs."""
        decl = _build_nonlinear_chain(depth=2, dim=4)

        # Koopman path
        tapestry_k = weave(decl, seed=42, optimize=True, nonlinear_method="koopman")
        # Fallback path (no optimization)
        tapestry_f = weave(decl, seed=42, optimize=False)

        rng = np.random.default_rng(77)
        agree = 0
        n_tests = 20
        for _ in range(n_tests):
            x = FloatVec(data=rng.standard_normal(4).astype(np.float32) * 0.3)
            obs_k = observe(tapestry_k, {"x": x})
            obs_f = observe(tapestry_f, {"x": x})
            if obs_k.value_index == obs_f.value_index:
                agree += 1

        # Should agree on most inputs (at least 50%)
        assert agree >= n_tests * 0.5, f"Only {agree}/{n_tests} agreed"

    def test_reobserve_with_koopman(self):
        """reobserve should work with Koopman tapestry."""
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="koopman")

        x = FloatVec.from_list([0.3, -0.1, 0.5, 0.2])
        obs = reobserve(tapestry, {"x": x}, count=5, seed=42)
        assert obs is not None
        assert obs.observation_count == 5


class TestKoopmanEdgeCases:
    def test_dim_1(self):
        """Koopman should work with dim=1."""
        builder = DeclarationBuilder("dim1")
        builder.input("x", 1)
        builder.relate("y", ["x"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.output("y")
        decl = builder.build()

        tapestry = weave(decl, seed=42, nonlinear_method="koopman")
        assert tapestry._koopman_matrix is not None

        x = FloatVec.from_list([0.5])
        obs = observe(tapestry, {"x": x})
        assert obs is not None
        assert obs.value.size == 1

    def test_dim_16(self):
        """Koopman should work with larger dimensions."""
        builder = DeclarationBuilder("dim16")
        builder.input("x", 16)
        builder.relate("y", ["x"], RelationKind.PROPORTIONAL, transform_fn=_relu)
        builder.output("y")
        decl = builder.build()

        tapestry = weave(decl, seed=42, nonlinear_method="koopman")
        assert tapestry._koopman_matrix is not None

        x = FloatVec(data=np.random.default_rng(42).standard_normal(16).astype(np.float32) * 0.3)
        obs = observe(tapestry, {"x": x})
        assert obs is not None

    def test_nan_producing_transform_fn(self):
        """Transform fn that could produce NaN should be handled safely."""
        def bad_fn(x):
            # Could produce very large values or NaN
            return np.exp(x * 100)

        builder = DeclarationBuilder("bad_fn")
        builder.input("x", 4)
        builder.relate("y", ["x"], RelationKind.PROPORTIONAL, transform_fn=bad_fn)
        builder.output("y")
        decl = builder.build()

        # Should not raise
        tapestry = weave(decl, seed=42, nonlinear_method="koopman")
        assert tapestry._koopman_matrix is not None

        x = FloatVec.from_list([0.1, -0.1, 0.05, -0.05])
        obs = observe(tapestry, {"x": x})
        # Should not contain NaN
        assert not np.any(np.isnan(obs.value.data))
        assert not np.any(np.isnan(obs.probabilities.data))

    def test_multi_source_relation_no_koopman(self):
        """Multi-source relations (MergeOp) should not use Koopman."""
        builder = DeclarationBuilder("multi_source")
        builder.input("x", 4)
        builder.input("y", 4)
        builder.relate("z", ["x", "y"], RelationKind.ADDITIVE, transform_fn=_sigmoid)
        builder.output("z")
        decl = builder.build()

        tapestry = weave(decl, seed=42, nonlinear_method="koopman")
        # MergeOp breaks the TransformOp-only chain requirement
        assert tapestry._koopman_matrix is None
        assert tapestry._composed_matrix is None

    def test_mixed_linear_nonlinear_chain(self):
        """Chain with mix of linear and nonlinear steps should use Koopman."""
        builder = DeclarationBuilder("mixed")
        builder.input("x", 4)
        # First relation: linear (no transform_fn)
        builder.relate("h1", ["x"], RelationKind.PROPORTIONAL)
        # Second relation: nonlinear
        builder.relate("h2", ["h1"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.output("h2")
        decl = builder.build()

        tapestry = weave(decl, seed=42, nonlinear_method="koopman")
        # Has nonlinear -> should try Koopman
        assert tapestry._koopman_matrix is not None


# ===================================================================
# Augmented Dictionary Tests
# ===================================================================


class TestAugmentedLiftedDim:
    def test_augmented_dim_4(self):
        # poly(4, 2) = 15, augmented adds 2*4 = 8 -> 23
        assert lifted_dim(4, basis="augmented") == 23

    def test_augmented_dim_1(self):
        # poly(1, 2) = 3, augmented adds 2*1 = 2 -> 5
        assert lifted_dim(1, basis="augmented") == 5

    def test_augmented_dim_8(self):
        # poly(8, 2) = 45, augmented adds 2*8 = 16 -> 61
        assert lifted_dim(8, basis="augmented") == 61

    def test_augmented_degree_3(self):
        # poly(4, 3) = 35, augmented adds 2*4 = 8 -> 43
        assert lifted_dim(4, degree=3, basis="augmented") == 43

    def test_poly_unchanged(self):
        assert lifted_dim(4) == 15
        assert lifted_dim(4, basis="poly") == 15


class TestAugmentedLift:
    def test_structure_positive(self):
        """Augmented lift of positive vector: indicators=1, relu_cross=x."""
        x = np.array([2.0, 3.0])
        psi = lift(x, basis="augmented")
        # poly part: [1, 2, 3, 4, 6, 9] (6 terms)
        # indicators: [1, 1] (both positive)
        # relu_cross: [2, 3] (x * 1)
        assert psi.shape == (lifted_dim(2, basis="augmented"),)
        np.testing.assert_allclose(psi[:6], [1.0, 2.0, 3.0, 4.0, 6.0, 9.0], atol=1e-10)
        np.testing.assert_allclose(psi[6:8], [1.0, 1.0], atol=1e-10)  # indicators
        np.testing.assert_allclose(psi[8:10], [2.0, 3.0], atol=1e-10)  # relu_cross

    def test_structure_negative(self):
        """Augmented lift of negative vector: indicators=0, relu_cross=0."""
        x = np.array([-2.0, -3.0])
        psi = lift(x, basis="augmented")
        np.testing.assert_allclose(psi[6:8], [0.0, 0.0], atol=1e-10)  # indicators
        np.testing.assert_allclose(psi[8:10], [0.0, 0.0], atol=1e-10)  # relu_cross

    def test_structure_mixed(self):
        """Mixed signs: only positive components have indicator=1."""
        x = np.array([2.0, -3.0])
        psi = lift(x, basis="augmented")
        np.testing.assert_allclose(psi[6:8], [1.0, 0.0], atol=1e-10)  # indicators
        np.testing.assert_allclose(psi[8:10], [2.0, 0.0], atol=1e-10)  # relu_cross

    def test_relu_exactly_representable(self):
        """ReLU(x_i) = x_i * 1_{x_i>0} is in the augmented basis."""
        x = np.array([2.0, -3.0, 0.0, 1.5])
        psi = lift(x, basis="augmented")
        poly_ld = lifted_dim(4, basis="poly")
        relu_cross = psi[poly_ld + 4:poly_ld + 8]
        expected_relu = np.maximum(x, 0.0)
        np.testing.assert_allclose(relu_cross, expected_relu, atol=1e-10)

    def test_batch(self):
        X = np.array([[1.0, -2.0], [3.0, 4.0]])
        Psi = lift(X, basis="augmented")
        assert Psi.shape == (2, lifted_dim(2, basis="augmented"))
        # Row 0: x=[1,-2] -> indicators=[1,0], relu_cross=[1,0]
        np.testing.assert_allclose(Psi[0, 6:8], [1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(Psi[0, 8:10], [1.0, 0.0], atol=1e-10)

    def test_unlift_augmented_roundtrip(self):
        """unlift(lift(x, augmented), augmented) == x."""
        x = np.array([1.5, -2.0, 3.0, 0.5])
        psi = lift(x, basis="augmented")
        recovered = unlift(psi, dim=4, basis="augmented")
        np.testing.assert_allclose(recovered, x, atol=1e-10)


class TestAugmentedKoopmanEstimation:
    def test_relu_augmented_accuracy(self):
        """Augmented basis should approximate ReLU much better than poly."""
        dim = 4
        rng = np.random.default_rng(99)
        A = rng.standard_normal((dim, dim)) * 0.3

        def relu_fn(x):
            return np.maximum(x @ A, 0.0)

        K_poly = estimate_koopman_matrix(relu_fn, dim, n_samples=1000, seed=42, basis="poly")
        K_aug = estimate_koopman_matrix(relu_fn, dim, n_samples=1000, seed=42, basis="augmented")

        # Compare prediction errors
        errors_poly = []
        errors_aug = []
        for i in range(20):
            test_x = rng.standard_normal(dim) * 0.3
            expected = relu_fn(test_x)

            # Poly prediction
            psi_p = lift(test_x)
            pred_p = unlift(psi_p @ K_poly.data.astype(np.float64), dim)
            errors_poly.append(np.linalg.norm(pred_p - expected))

            # Augmented prediction
            psi_a = lift(test_x, basis="augmented")
            pred_a = unlift(psi_a @ K_aug.data.astype(np.float64), dim, basis="augmented")
            errors_aug.append(np.linalg.norm(pred_a - expected))

        # Augmented should have lower mean error
        assert np.mean(errors_aug) < np.mean(errors_poly), (
            f"Augmented error {np.mean(errors_aug):.4f} >= poly error {np.mean(errors_poly):.4f}"
        )

    def test_step_function_augmented(self):
        """Hard threshold: augmented should improve over poly."""
        dim = 4
        rng = np.random.default_rng(77)
        A = rng.standard_normal((dim, dim)) * 0.3

        def step_fn(x):
            return (x @ A > 0).astype(np.float64)

        K_aug = estimate_koopman_matrix(step_fn, dim, n_samples=1000, seed=42, basis="augmented")

        # Should not produce NaN
        assert not np.any(np.isnan(K_aug.data))

        # Should have reasonable predictions
        test_x = rng.standard_normal(dim) * 0.3
        psi = lift(test_x, basis="augmented")
        pred = unlift(psi @ K_aug.data.astype(np.float64), dim, basis="augmented")
        assert not np.any(np.isnan(pred))


class TestAugmentedWeaveObserve:
    def test_weave_with_augmented_basis(self):
        """weave with koopman_basis='augmented' should produce Koopman matrix."""
        decl = _build_nonlinear_chain(depth=2, dim=4, transform_fn=_relu)
        tapestry = weave(decl, seed=42, nonlinear_method="koopman", koopman_basis="augmented")
        assert tapestry._koopman_matrix is not None
        info = tapestry._koopman_chain_info
        assert info["basis"] == "augmented"
        assert info["lifted_dim"] == lifted_dim(4, basis="augmented")

    def test_augmented_observe_runs(self):
        """Observe with augmented Koopman tapestry should work."""
        decl = _build_nonlinear_chain(depth=2, dim=4, transform_fn=_relu)
        tapestry = weave(decl, seed=42, nonlinear_method="koopman", koopman_basis="augmented")

        x = FloatVec.from_list([0.3, -0.1, 0.5, 0.2])
        obs = observe(tapestry, {"x": x})
        assert obs is not None
        assert obs.value.size == 4
        assert not np.any(np.isnan(obs.value.data))

    def test_augmented_vs_poly_relu_accuracy(self):
        """Augmented should agree more with fallback than poly for ReLU chains."""
        decl = _build_nonlinear_chain(depth=2, dim=4, transform_fn=_relu)

        tap_aug = weave(decl, seed=42, nonlinear_method="koopman", koopman_basis="augmented")
        tap_poly = weave(decl, seed=42, nonlinear_method="koopman", koopman_basis="poly")
        tap_fallback = weave(decl, seed=42, optimize=False)

        rng = np.random.default_rng(88)
        agree_aug = 0
        agree_poly = 0
        n_tests = 30
        for _ in range(n_tests):
            x = FloatVec(data=rng.standard_normal(4).astype(np.float32) * 0.3)
            obs_fb = observe(tap_fallback, {"x": x})
            obs_aug = observe(tap_aug, {"x": x})
            obs_poly = observe(tap_poly, {"x": x})
            if obs_aug.value_index == obs_fb.value_index:
                agree_aug += 1
            if obs_poly.value_index == obs_fb.value_index:
                agree_poly += 1

        # Augmented should agree at least as much as poly (usually more)
        assert agree_aug >= agree_poly - 2, (
            f"Augmented {agree_aug}/{n_tests} vs Poly {agree_poly}/{n_tests}"
        )
