"""Tests for end-to-end pipeline distillation."""

from __future__ import annotations

import numpy as np
import pytest

from axol.core.types import FloatVec, TransMatrix
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe, reobserve


# ===================================================================
# Helpers
# ===================================================================


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _relu(x):
    return np.maximum(x, 0.0)


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


def hellinger_distance(p, q):
    p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    q = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
    sp, sq = p.sum(), q.sum()
    if sp > 0: p = p / sp
    if sq > 0: q = q / sq
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


# ===================================================================
# Weave Tests
# ===================================================================


class TestWeaveDistill:
    def test_nonlinear_produces_distilled(self):
        """Default nonlinear_method='distill' should set distilled fields."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42)  # default = "distill"
        assert tapestry._distilled_matrix is not None
        assert tapestry._distilled_chain_info is not None
        # Should NOT set other fast-path matrices
        assert tapestry._hybrid_matrix is None
        assert tapestry._unitary_matrix is None
        assert tapestry._koopman_matrix is None

    def test_linear_uses_composed_not_distilled(self):
        """Pure linear chain should use composed matrix regardless of default."""
        decl = _build_linear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42)
        assert tapestry._composed_matrix is not None
        assert tapestry._distilled_matrix is None

    def test_distilled_chain_info_fields(self):
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42)
        info = tapestry._distilled_chain_info
        assert info is not None
        assert "input_key" in info
        assert "output_key" in info
        assert "dim" in info
        assert info["dim"] == 4

    def test_distilled_matrix_shape(self):
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42)
        assert tapestry._distilled_matrix.shape == (4, 4)

    def test_optimize_false_skips_distill(self):
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42, optimize=False)
        assert tapestry._distilled_matrix is None
        assert tapestry._distilled_chain_info is None

    def test_explicit_distill_method(self):
        """Explicit nonlinear_method='distill' should produce distilled fields."""
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42, nonlinear_method="distill")
        assert tapestry._distilled_matrix is not None
        assert tapestry._distilled_chain_info is not None


# ===================================================================
# Observe Tests
# ===================================================================


class TestObserveDistill:
    def test_distill_observe_runs(self):
        """Distilled observe should return a valid Observation."""
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42)

        x = FloatVec.from_list([0.3, -0.1, 0.5, 0.2])
        obs = observe(tapestry, {"x": x})
        assert obs is not None
        assert obs.value.size == 4
        assert 0 <= obs.value_index < 4
        assert not np.any(np.isnan(obs.value.data))
        assert not np.any(np.isnan(obs.probabilities.data))

    def test_probabilities_sum_to_one(self):
        decl = _build_nonlinear_chain(depth=3, dim=4)
        tapestry = weave(decl, seed=42)

        rng = np.random.default_rng(55)
        for _ in range(10):
            x = FloatVec(data=rng.standard_normal(4).astype(np.float32) * 0.5)
            obs = observe(tapestry, {"x": x})
            np.testing.assert_allclose(np.sum(obs.probabilities.data), 1.0, atol=1e-5)

    def test_reobserve_with_distill(self):
        decl = _build_nonlinear_chain(depth=2, dim=4)
        tapestry = weave(decl, seed=42)
        x = FloatVec.from_list([0.3, -0.1, 0.5, 0.2])
        obs = reobserve(tapestry, {"x": x}, count=5, seed=42)
        assert obs.observation_count == 5


# ===================================================================
# Accuracy Tests
# ===================================================================


class TestDistillAccuracy:
    def test_sigmoid_depth5_argmax(self):
        """Distill on sigmoid depth=5 should achieve >= 80% argmax match."""
        decl = _build_nonlinear_chain(depth=5, dim=4, transform_fn=_sigmoid)
        tap_d = weave(decl, seed=42, nonlinear_method="distill", distill_samples=200)
        tap_f = weave(decl, seed=42, optimize=False)

        rng = np.random.default_rng(77)
        match = 0
        N = 50
        for _ in range(N):
            x = FloatVec(data=rng.standard_normal(4).astype(np.float32) * 0.3)
            obs_d = observe(tap_d, {"x": x})
            obs_f = observe(tap_f, {"x": x})
            if obs_d.value_index == obs_f.value_index:
                match += 1
        accuracy = match / N
        assert accuracy >= 0.80, f"Distill argmax accuracy {accuracy:.0%} < 80%"

    def test_distill_beats_hybrid_at_depth(self):
        """Distill should have lower H-dist than hybrid at depth >= 5."""
        decl = _build_nonlinear_chain(depth=5, dim=4, transform_fn=_sigmoid)
        tap_d = weave(decl, seed=42, nonlinear_method="distill", distill_samples=200)
        tap_h = weave(decl, seed=42, nonlinear_method="hybrid")
        tap_f = weave(decl, seed=42, optimize=False)

        rng = np.random.default_rng(88)
        hd, hh = 0.0, 0.0
        N = 30
        for _ in range(N):
            x = FloatVec(data=rng.standard_normal(4).astype(np.float32) * 0.3)
            obs_d = observe(tap_d, {"x": x})
            obs_h = observe(tap_h, {"x": x})
            obs_f = observe(tap_f, {"x": x})
            hd += hellinger_distance(obs_d.probabilities.data, obs_f.probabilities.data)
            hh += hellinger_distance(obs_h.probabilities.data, obs_f.probabilities.data)
        hd /= N
        hh /= N
        assert hd <= hh + 0.05, (
            f"Distill H-dist {hd:.4f} should be <= Hybrid H-dist {hh:.4f} + 0.05"
        )


# ===================================================================
# Edge Cases
# ===================================================================


class TestDistillEdgeCases:
    def test_dim_1(self):
        builder = DeclarationBuilder("dim1")
        builder.input("x", 1)
        builder.relate("y", ["x"], RelationKind.PROPORTIONAL, transform_fn=_sigmoid)
        builder.output("y")
        decl = builder.build()

        tapestry = weave(decl, seed=42)
        assert tapestry._distilled_matrix is not None
        assert tapestry._distilled_matrix.shape == (1, 1)

        x = FloatVec.from_list([0.5])
        obs = observe(tapestry, {"x": x})
        assert obs is not None

    def test_multi_source_no_distill(self):
        """Multi-source should fall back, not use distill."""
        builder = DeclarationBuilder("multi_source")
        builder.input("x", 4)
        builder.input("y", 4)
        builder.relate("z", ["x", "y"], RelationKind.ADDITIVE, transform_fn=_sigmoid)
        builder.output("z")
        decl = builder.build()

        tapestry = weave(decl, seed=42)
        assert tapestry._distilled_matrix is None
        assert tapestry._hybrid_matrix is None
        assert tapestry._unitary_matrix is None
        assert tapestry._koopman_matrix is None

    def test_nan_producing_transform_fn(self):
        def bad_fn(x):
            return np.exp(x * 100)

        builder = DeclarationBuilder("bad_fn")
        builder.input("x", 4)
        builder.relate("y", ["x"], RelationKind.PROPORTIONAL, transform_fn=bad_fn)
        builder.output("y")
        decl = builder.build()

        tapestry = weave(decl, seed=42)
        assert tapestry._distilled_matrix is not None

        x = FloatVec.from_list([0.1, -0.1, 0.05, -0.05])
        obs = observe(tapestry, {"x": x})
        assert not np.any(np.isnan(obs.value.data))
        assert not np.any(np.isnan(obs.probabilities.data))
