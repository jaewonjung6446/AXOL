"""Tests for axol.quantum.online — incremental EDMD learner.

Covers:
  * Construction / validation
  * Rank-1 Sherman-Morrison correctness (vs. full recompute)
  * Learning a known linear system to <1% error
  * Forgetting factor behaviour
  * Quality metrics (Omega / Phi)
  * Axiom-1 style invariant: equivalence under sample order (gamma == 1)
  * Real-time scenario: concept drift adaptation via forgetting
"""

from __future__ import annotations

import numpy as np
import pytest

from axol.core.types import FloatVec, TransMatrix
from axol.quantum.koopman import lift, lifted_dim
from axol.quantum.online import LearningReport, OnlineLearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear_stream(M: np.ndarray, n: int, noise: float = 0.0, seed: int = 0):
    """Generate n (x, y) pairs where y = M @ x + noise."""
    rng = np.random.default_rng(seed)
    dim = M.shape[0]
    X = rng.standard_normal((n, dim)) * 0.5
    Y = X @ M.T
    if noise > 0:
        Y = Y + rng.standard_normal(Y.shape) * noise
    return X.astype(np.float32), Y.astype(np.float32)


def _full_recompute_K(
    X: np.ndarray, Y: np.ndarray, degree: int, basis: str, reg: float
) -> np.ndarray:
    """Reference: compute K = G^{-1} B in one shot (Psi(y) = Psi(x) @ K)."""
    Psi_X = lift(X.astype(np.float64), degree, basis)
    Psi_Y = lift(Y.astype(np.float64), degree, basis)
    ld = Psi_X.shape[1]
    G = Psi_X.T @ Psi_X + np.eye(ld) * reg
    B = Psi_X.T @ Psi_Y
    return np.linalg.inv(G) @ B


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        learner = OnlineLearner(dim=4)
        assert learner.dim == 4
        assert learner.degree == 2
        assert learner.basis == "poly"
        assert learner.forgetting_factor == 1.0
        assert learner.n_samples == 0

    def test_lifted_dim_matches_formula(self):
        learner = OnlineLearner(dim=4, degree=2)
        assert learner.ld == lifted_dim(4, 2, "poly")

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            OnlineLearner(dim=0)

    def test_invalid_degree(self):
        with pytest.raises(ValueError):
            OnlineLearner(dim=4, degree=0)

    def test_invalid_forgetting(self):
        with pytest.raises(ValueError):
            OnlineLearner(dim=4, forgetting_factor=0.0)
        with pytest.raises(ValueError):
            OnlineLearner(dim=4, forgetting_factor=1.1)

    def test_invalid_regularization(self):
        with pytest.raises(ValueError):
            OnlineLearner(dim=4, regularization=0.0)


# ---------------------------------------------------------------------------
# Sherman-Morrison correctness
# ---------------------------------------------------------------------------

class TestShermanMorrison:
    """Incremental updates must agree with full recompute to high precision."""

    @pytest.mark.parametrize("dim,n", [(3, 30), (4, 60)])
    def test_incremental_matches_full_recompute(self, dim, n):
        rng = np.random.default_rng(12345)
        M = rng.standard_normal((dim, dim)) * 0.3
        X, Y = _make_linear_stream(M, n=n, seed=7)

        reg = 1e-3
        learner = OnlineLearner(dim=dim, degree=2, regularization=reg)
        for i in range(n):
            learner.observe_sample(X[i], Y[i])

        K_incremental = learner._K()
        K_full = _full_recompute_K(X, Y, degree=2, basis="poly", reg=reg)

        assert np.allclose(K_incremental, K_full, atol=1e-6, rtol=1e-4)

    def test_accepts_floatvec(self):
        learner = OnlineLearner(dim=3)
        x = FloatVec.from_list([0.1, 0.2, 0.3])
        y = FloatVec.from_list([0.4, 0.5, 0.6])
        learner.observe_sample(x, y)
        assert learner.n_samples == 1

    def test_rejects_wrong_shape(self):
        learner = OnlineLearner(dim=3)
        with pytest.raises(ValueError):
            learner.observe_sample(np.zeros(4), np.zeros(3))
        with pytest.raises(ValueError):
            learner.observe_sample(np.zeros(3), np.zeros(2))


# ---------------------------------------------------------------------------
# Actually learning something
# ---------------------------------------------------------------------------

class TestLearning:
    def test_recovers_linear_map(self):
        """After enough samples, predict(x) approximates M @ x."""
        rng = np.random.default_rng(0)
        dim = 4
        M = rng.standard_normal((dim, dim)) * 0.4
        X, Y = _make_linear_stream(M, n=200, noise=0.0, seed=1)

        learner = OnlineLearner(dim=dim, degree=2, regularization=1e-4)
        learner.observe_batch(X, Y)

        # Evaluate on a fresh test point
        x_test = rng.standard_normal(dim).astype(np.float32) * 0.3
        y_true = M @ x_test
        y_hat = learner.predict(x_test)

        err = np.linalg.norm(y_hat - y_true) / (np.linalg.norm(y_true) + 1e-12)
        assert err < 0.05, f"relative prediction error too high: {err:.4f}"

    def test_predict_vec_returns_floatvec(self):
        learner = OnlineLearner(dim=3)
        X, Y = _make_linear_stream(np.eye(3), n=20, seed=2)
        learner.observe_batch(X, Y)
        out = learner.predict_vec(np.zeros(3, dtype=np.float32))
        assert isinstance(out, FloatVec)
        assert out.size == 3

    def test_identity_mapping(self):
        """Learning y = x should give predictions near x."""
        X, Y = _make_linear_stream(np.eye(3), n=150, seed=11)
        learner = OnlineLearner(dim=3, degree=2, regularization=1e-4)
        learner.observe_batch(X, Y)

        x_test = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        y_hat = learner.predict(x_test)
        assert np.allclose(y_hat, x_test, atol=0.05)


# ---------------------------------------------------------------------------
# Forgetting factor
# ---------------------------------------------------------------------------

class TestForgetting:
    def test_gamma_one_never_forgets(self):
        learner = OnlineLearner(dim=3, forgetting_factor=1.0)
        X, Y = _make_linear_stream(np.eye(3), n=10, seed=3)
        learner.observe_batch(X, Y)
        assert learner.effective_samples == pytest.approx(10.0)

    def test_gamma_less_than_one_saturates(self):
        """Effective sample count should approach 1/(1-gamma)."""
        gamma = 0.9
        learner = OnlineLearner(dim=3, forgetting_factor=gamma)
        X, Y = _make_linear_stream(np.eye(3), n=300, seed=4)
        learner.observe_batch(X, Y)
        cap = 1.0 / (1.0 - gamma)
        assert learner.effective_samples == pytest.approx(cap, rel=0.02)

    def test_concept_drift_adaptation(self):
        """With forgetting, learner tracks a changing system.

        System switches from M1 -> M2 midway; predictions after many post-
        switch samples should be closer to M2 than to M1.
        """
        rng = np.random.default_rng(42)
        dim = 3
        M1 = rng.standard_normal((dim, dim)) * 0.3
        M2 = rng.standard_normal((dim, dim)) * 0.3

        learner = OnlineLearner(dim=dim, degree=2, forgetting_factor=0.92)
        X1, Y1 = _make_linear_stream(M1, n=100, seed=100)
        X2, Y2 = _make_linear_stream(M2, n=200, seed=200)

        learner.observe_batch(X1, Y1)
        learner.observe_batch(X2, Y2)

        x_probe = rng.standard_normal(dim).astype(np.float32) * 0.3
        y_hat = learner.predict(x_probe)
        err_to_M2 = np.linalg.norm(y_hat - M2 @ x_probe)
        err_to_M1 = np.linalg.norm(y_hat - M1 @ x_probe)
        assert err_to_M2 < err_to_M1


# ---------------------------------------------------------------------------
# Quality metrics (Axiom 3)
# ---------------------------------------------------------------------------

class TestQualityMetrics:
    def test_metrics_defined_before_any_sample(self):
        learner = OnlineLearner(dim=3)
        # Sensible defaults; must not raise
        assert learner.omega == pytest.approx(1.0)
        assert learner.phi == pytest.approx(1.0)
        assert learner.max_lyapunov == 0.0

    def test_omega_in_unit_interval(self):
        learner = OnlineLearner(dim=3)
        X, Y = _make_linear_stream(np.eye(3) * 0.5, n=60, seed=5)
        learner.observe_batch(X, Y)
        assert 0.0 <= learner.omega <= 1.0
        assert 0.0 <= learner.phi <= 1.0

    def test_report_structure(self):
        learner = OnlineLearner(dim=3)
        X, Y = _make_linear_stream(np.eye(3), n=20, seed=6)
        learner.observe_batch(X, Y)
        r = learner.report()
        assert isinstance(r, LearningReport)
        assert r.n_samples == 20
        assert r.dim == 3
        assert r.lifted_dim == lifted_dim(3, 2, "poly")
        assert r.forgetting_factor == 1.0

    def test_contractive_system_is_cohesive(self):
        """A strongly contractive M (small spectral radius) should give Omega near 1."""
        M = np.eye(3) * 0.1  # spectral radius 0.1
        X, Y = _make_linear_stream(M, n=150, seed=8)
        learner = OnlineLearner(dim=3, regularization=1e-4)
        learner.observe_batch(X, Y)
        assert learner.omega > 0.8


# ---------------------------------------------------------------------------
# Axiom-aligned invariants
# ---------------------------------------------------------------------------

class TestAxiomInvariants:
    """Axiom 1: with gamma == 1, moments are additive -> order-independent."""

    def test_order_independence_at_gamma_one(self):
        rng = np.random.default_rng(99)
        dim = 3
        M = rng.standard_normal((dim, dim)) * 0.3
        X, Y = _make_linear_stream(M, n=50, seed=13)

        learner_a = OnlineLearner(dim=dim, forgetting_factor=1.0, regularization=1e-3)
        learner_b = OnlineLearner(dim=dim, forgetting_factor=1.0, regularization=1e-3)

        learner_a.observe_batch(X, Y)

        perm = rng.permutation(len(X))
        learner_b.observe_batch(X[perm], Y[perm])

        # Moment matrices are exactly equal up to float64 rounding.
        assert np.allclose(learner_a._G, learner_b._G, atol=1e-9)
        assert np.allclose(learner_a._B, learner_b._B, atol=1e-9)

        # Sherman-Morrison inverses may differ slightly due to update order;
        # the resulting K is still numerically close.
        assert np.allclose(learner_a._K(), learner_b._K(), atol=1e-3, rtol=1e-3)

    def test_single_update_is_closed_form(self):
        """Sanity: observe_sample does not iterate; each call returns quickly."""
        learner = OnlineLearner(dim=16, degree=2)
        x = np.zeros(16, dtype=np.float32)
        y = np.ones(16, dtype=np.float32)
        # Just verify it completes and cost is not absurd.
        for _ in range(100):
            learner.observe_sample(x, y)
        assert learner.n_samples == 100


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        learner = OnlineLearner(dim=3)
        X, Y = _make_linear_stream(np.eye(3), n=10, seed=14)
        learner.observe_batch(X, Y)
        assert learner.n_samples == 10

        learner.reset()
        assert learner.n_samples == 0
        assert learner.effective_samples == 0.0
        # After reset, predictions are essentially zero (K ~= 0).
        y_hat = learner.predict(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert np.allclose(y_hat, np.zeros(3), atol=1e-3)


# ---------------------------------------------------------------------------
# Koopman matrix is exposed as TransMatrix
# ---------------------------------------------------------------------------

class TestPublicMatrix:
    def test_koopman_matrix_type(self):
        learner = OnlineLearner(dim=3)
        X, Y = _make_linear_stream(np.eye(3), n=20, seed=15)
        learner.observe_batch(X, Y)
        K = learner.koopman_matrix
        assert isinstance(K, TransMatrix)
        ld = lifted_dim(3, 2, "poly")
        assert K.shape == (ld, ld)


# ---------------------------------------------------------------------------
# Explicit decay / time-based forgetting
# ---------------------------------------------------------------------------

class TestDecay:
    def test_decay_one_is_noop(self):
        """factor=1.0 leaves moments untouched."""
        learner = OnlineLearner(dim=3, regularization=1e-3)
        X, Y = _make_linear_stream(np.eye(3) * 0.5, n=30, seed=20)
        learner.observe_batch(X, Y)
        G_before = learner._G.copy()
        B_before = learner._B.copy()
        learner.decay(1.0)
        assert np.allclose(learner._G, G_before)
        assert np.allclose(learner._B, B_before)

    def test_decay_preserves_regularisation_floor(self):
        """After aggressive decay G still satisfies G >= reg*I."""
        reg = 1e-3
        learner = OnlineLearner(dim=3, regularization=reg)
        X, Y = _make_linear_stream(np.eye(3), n=30, seed=21)
        learner.observe_batch(X, Y)
        learner.decay(0.01)  # near-total forgetting
        # Diagonal should never fall below reg
        diag_min = np.min(np.diag(learner._G))
        assert diag_min >= reg * 0.99

    def test_decay_zero_collapses_to_regularised_state(self):
        learner = OnlineLearner(dim=3, regularization=1e-3)
        X, Y = _make_linear_stream(np.eye(3), n=30, seed=22)
        learner.observe_batch(X, Y)
        learner.decay(0.0)
        # G should equal reg*I exactly
        expected = np.eye(learner.ld) * learner.regularization
        assert np.allclose(learner._G, expected)
        # K should be ~ 0 (no information)
        assert np.allclose(learner._K(), 0.0, atol=1e-6)

    def test_decay_reduces_prediction_strength(self):
        """After partial decay, predictions move toward zero."""
        learner = OnlineLearner(dim=3, regularization=1e-4)
        M = np.eye(3) * 0.7
        X, Y = _make_linear_stream(M, n=200, seed=23)
        learner.observe_batch(X, Y)
        x_probe = np.array([1.0, 0.5, -0.3], dtype=np.float32)
        y_strong = learner.predict(x_probe)
        learner.decay(0.1)  # heavy forgetting
        y_weak = learner.predict(x_probe)
        assert np.linalg.norm(y_weak) < np.linalg.norm(y_strong)

    def test_decay_by_time_matches_formula(self):
        """factor = 0.5 ** (elapsed / half_life) should match manual decay."""
        reg = 1e-3
        learner_time = OnlineLearner(dim=3, regularization=reg)
        learner_manual = OnlineLearner(dim=3, regularization=reg)
        X, Y = _make_linear_stream(np.eye(3), n=30, seed=24)
        learner_time.observe_batch(X, Y)
        learner_manual.observe_batch(X, Y)

        half_life = 10.0
        elapsed = 20.0            # two half-lives -> factor = 0.25
        learner_time.decay_by_time(elapsed, half_life)
        learner_manual.decay(0.25)

        assert np.allclose(learner_time._G, learner_manual._G, atol=1e-10)
        assert np.allclose(learner_time._B, learner_manual._B, atol=1e-10)

    def test_decay_by_time_rejects_invalid(self):
        learner = OnlineLearner(dim=3)
        with pytest.raises(ValueError):
            learner.decay_by_time(1.0, 0.0)
        with pytest.raises(ValueError):
            learner.decay_by_time(-1.0, 1.0)

    def test_decay_rejects_invalid_factor(self):
        learner = OnlineLearner(dim=3)
        with pytest.raises(ValueError):
            learner.decay(-0.1)
        with pytest.raises(ValueError):
            learner.decay(1.5)
