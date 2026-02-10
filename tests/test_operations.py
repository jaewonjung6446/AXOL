"""Tests for axol.core.operations."""

import numpy as np
import pytest

from axol.core.types import (
    FloatVec,
    IntVec,
    GateVec,
    TransMatrix,
    StateBundle,
)
from axol.core.operations import (
    transform,
    gate,
    merge,
    distance,
    route,
    transform_bundle,
    gate_bundle,
)


# ── transform ──────────────────────────────────────────────────────────────

class TestTransform:
    def test_identity(self):
        v = FloatVec.from_list([1.0, 2.0, 3.0])
        m = TransMatrix.identity(3)
        result = transform(v, m)
        assert result.to_list() == pytest.approx([1, 2, 3])

    def test_dimension_change(self):
        v = FloatVec.from_list([1.0, 0.0])
        m = TransMatrix.from_list([[1, 0, 0], [0, 1, 0]])  # 2→3
        result = transform(v, m)
        assert result.size == 3
        assert result.to_list() == pytest.approx([1, 0, 0])

    def test_scaling(self):
        v = FloatVec.from_list([2.0, 3.0])
        m = TransMatrix.from_list([[0.5, 0], [0, 2.0]])
        result = transform(v, m)
        assert result.to_list() == pytest.approx([1.0, 6.0])

    def test_dimension_mismatch(self):
        v = FloatVec.from_list([1.0, 2.0])
        m = TransMatrix.identity(3)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            transform(v, m)


# ── gate ───────────────────────────────────────────────────────────────────

class TestGate:
    def test_pass_all(self):
        v = FloatVec.from_list([1.0, 2.0, 3.0])
        g = GateVec.ones(3)
        assert gate(v, g).to_list() == pytest.approx([1, 2, 3])

    def test_block_all(self):
        v = FloatVec.from_list([1.0, 2.0, 3.0])
        g = GateVec.zeros(3)
        assert gate(v, g).to_list() == pytest.approx([0, 0, 0])

    def test_selective(self):
        v = FloatVec.from_list([10.0, 20.0, 30.0])
        g = GateVec.from_list([1.0, 0.0, 1.0])
        assert gate(v, g).to_list() == pytest.approx([10, 0, 30])

    def test_works_with_intvec(self):
        v = IntVec.from_list([5, 10])
        g = GateVec.from_list([0.0, 1.0])
        assert gate(v, g).to_list() == pytest.approx([0, 10])

    def test_dimension_mismatch(self):
        v = FloatVec.from_list([1.0])
        g = GateVec.from_list([1.0, 1.0])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            gate(v, g)


# ── merge ──────────────────────────────────────────────────────────────────

class TestMerge:
    def test_equal_weights(self):
        a = FloatVec.from_list([2.0, 0.0])
        b = FloatVec.from_list([0.0, 4.0])
        w = FloatVec.from_list([0.5, 0.5])
        result = merge([a, b], w)
        assert result.to_list() == pytest.approx([1.0, 2.0])

    def test_single_vector(self):
        a = FloatVec.from_list([3.0])
        w = FloatVec.from_list([2.0])
        assert merge([a], w).to_list() == pytest.approx([6.0])

    def test_count_mismatch(self):
        a = FloatVec.from_list([1.0])
        w = FloatVec.from_list([1.0, 1.0])
        with pytest.raises(ValueError, match="Count mismatch"):
            merge([a], w)

    def test_empty(self):
        w = FloatVec.from_list([])
        with pytest.raises(ValueError, match="at least one"):
            merge([], w)


# ── distance ───────────────────────────────────────────────────────────────

class TestDistance:
    def test_euclidean_identical(self):
        v = FloatVec.from_list([1.0, 2.0])
        assert distance(v, v, "euclidean") == pytest.approx(0.0)

    def test_euclidean_known(self):
        a = FloatVec.from_list([0.0, 0.0])
        b = FloatVec.from_list([3.0, 4.0])
        assert distance(a, b, "euclidean") == pytest.approx(5.0)

    def test_cosine_identical(self):
        v = FloatVec.from_list([1.0, 0.0])
        assert distance(v, v, "cosine") == pytest.approx(0.0)

    def test_cosine_orthogonal(self):
        a = FloatVec.from_list([1.0, 0.0])
        b = FloatVec.from_list([0.0, 1.0])
        assert distance(a, b, "cosine") == pytest.approx(1.0)

    def test_dot(self):
        a = FloatVec.from_list([2.0, 3.0])
        b = FloatVec.from_list([4.0, 5.0])
        assert distance(a, b, "dot") == pytest.approx(23.0)

    def test_shape_mismatch(self):
        a = FloatVec.from_list([1.0])
        b = FloatVec.from_list([1.0, 2.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            distance(a, b)


# ── route ──────────────────────────────────────────────────────────────────

class TestRoute:
    def test_simple_routing(self):
        v = FloatVec.from_list([1.0, 0.0])
        # Router: column 0 has high score when first element is high
        router = TransMatrix.from_list([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        assert route(v, router) == 0

    def test_second_route(self):
        v = FloatVec.from_list([0.0, 1.0])
        router = TransMatrix.from_list([[10.0, 0.0], [0.0, 10.0]])
        assert route(v, router) == 1


# ── bundle helpers ─────────────────────────────────────────────────────────

class TestBundleHelpers:
    def test_transform_bundle(self):
        sb = StateBundle()
        sb["pos"] = FloatVec.from_list([1.0, 0.0])
        m = TransMatrix.from_list([[0.0, 1.0], [1.0, 0.0]])  # swap
        result = transform_bundle(sb, "pos", m)
        assert result["pos"].to_list() == pytest.approx([0.0, 1.0])
        # original unchanged
        assert sb["pos"].to_list() == pytest.approx([1.0, 0.0])

    def test_gate_bundle(self):
        sb = StateBundle()
        sb["v"] = FloatVec.from_list([5.0, 10.0])
        sb["g"] = GateVec.from_list([1.0, 0.0])
        result = gate_bundle(sb, "v", sb["g"])
        assert result["v"].to_list() == pytest.approx([5.0, 0.0])
