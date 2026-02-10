"""Tests for axol.core.types."""

import numpy as np
import pytest

from axol.core.types import (
    BinaryVec,
    IntVec,
    FloatVec,
    OneHotVec,
    GateVec,
    TransMatrix,
    StateBundle,
)


# ── BinaryVec ──────────────────────────────────────────────────────────────

class TestBinaryVec:
    def test_from_list(self):
        v = BinaryVec.from_list([1, 0, 1])
        assert v.size == 3
        assert v.to_list() == [1, 0, 1]

    def test_zeros_ones(self):
        assert BinaryVec.zeros(4).to_list() == [0, 0, 0, 0]
        assert BinaryVec.ones(3).to_list() == [1, 1, 1]

    def test_rejects_invalid(self):
        with pytest.raises(ValueError, match="0 or 1"):
            BinaryVec.from_list([0, 2, 1])

    def test_immutable(self):
        v = BinaryVec.from_list([1, 0])
        with pytest.raises(Exception):
            v.data = np.array([0, 0], dtype=np.int8)

    def test_equality(self):
        a = BinaryVec.from_list([1, 0])
        b = BinaryVec.from_list([1, 0])
        c = BinaryVec.from_list([0, 1])
        assert a == b
        assert a != c


# ── IntVec ─────────────────────────────────────────────────────────────────

class TestIntVec:
    def test_from_list(self):
        v = IntVec.from_list([10, -3, 42])
        assert v.size == 3
        assert v.to_list() == [10, -3, 42]

    def test_zeros(self):
        assert IntVec.zeros(2).to_list() == [0, 0]


# ── FloatVec ───────────────────────────────────────────────────────────────

class TestFloatVec:
    def test_from_list(self):
        v = FloatVec.from_list([1.5, 2.0])
        assert v.size == 2
        assert v.to_list() == pytest.approx([1.5, 2.0])

    def test_zeros_ones(self):
        assert FloatVec.zeros(3).to_list() == pytest.approx([0, 0, 0])
        assert FloatVec.ones(2).to_list() == pytest.approx([1, 1])


# ── OneHotVec ──────────────────────────────────────────────────────────────

class TestOneHotVec:
    def test_from_index(self):
        v = OneHotVec.from_index(1, 3)
        assert v.to_list() == pytest.approx([0, 1, 0])
        assert v.active_index == 1

    def test_rejects_not_one_hot(self):
        with pytest.raises(ValueError, match="exactly one"):
            OneHotVec.from_list([1, 1, 0])

    def test_rejects_all_zero(self):
        with pytest.raises(ValueError, match="exactly one"):
            OneHotVec.from_list([0, 0, 0])


# ── GateVec ────────────────────────────────────────────────────────────────

class TestGateVec:
    def test_from_list(self):
        g = GateVec.from_list([1.0, 0.0, 1.0])
        assert g.to_list() == pytest.approx([1, 0, 1])

    def test_all_open(self):
        assert GateVec.ones(3).all_open is True
        assert GateVec.zeros(3).all_open is False

    def test_rejects_invalid(self):
        with pytest.raises(ValueError, match="0.0 or 1.0"):
            GateVec.from_list([0.5, 1.0])


# ── TransMatrix ────────────────────────────────────────────────────────────

class TestTransMatrix:
    def test_identity(self):
        m = TransMatrix.identity(3)
        assert m.shape == (3, 3)
        np.testing.assert_array_almost_equal(m.data, np.eye(3, dtype=np.float32))

    def test_from_list(self):
        m = TransMatrix.from_list([[1, 2], [3, 4], [5, 6]])
        assert m.shape == (3, 2)

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            TransMatrix(data=np.array([1, 2, 3], dtype=np.float32))

    def test_zeros(self):
        m = TransMatrix.zeros(2, 4)
        assert m.shape == (2, 4)


# ── StateBundle ────────────────────────────────────────────────────────────

class TestStateBundle:
    def test_get_set(self):
        sb = StateBundle()
        sb["hp"] = FloatVec.from_list([100.0])
        assert "hp" in sb
        assert sb["hp"] == FloatVec.from_list([100.0])

    def test_copy_is_deep(self):
        sb = StateBundle()
        sb["v"] = FloatVec.from_list([1.0, 2.0])
        clone = sb.copy()
        clone["v"] = FloatVec.from_list([9.0, 9.0])
        assert sb["v"].to_list() == pytest.approx([1.0, 2.0])

    def test_get_flat_array(self):
        sb = StateBundle()
        sb["a"] = FloatVec.from_list([1.0, 2.0])
        sb["b"] = IntVec.from_list([3, 4])
        flat = sb.get_flat_array()
        assert flat.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_equality(self):
        a = StateBundle(vectors={"x": FloatVec.from_list([1.0])})
        b = StateBundle(vectors={"x": FloatVec.from_list([1.0])})
        c = StateBundle(vectors={"x": FloatVec.from_list([2.0])})
        assert a == b
        assert a != c
