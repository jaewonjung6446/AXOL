"""Tests for axol.core.verify."""

import pytest

from axol.core.types import FloatVec, IntVec, StateBundle
from axol.core.verify import VerifySpec, MatchMode, verify_states


class TestVerifyExact:
    def test_identical_bundles(self):
        a = StateBundle(vectors={"x": FloatVec.from_list([1.0, 2.0])})
        b = StateBundle(vectors={"x": FloatVec.from_list([1.0, 2.0])})
        result = verify_states(a, b)
        assert result.passed is True
        assert len(result.vector_results) == 1
        assert result.vector_results[0].passed is True

    def test_different_values(self):
        a = StateBundle(vectors={"x": FloatVec.from_list([1.0])})
        b = StateBundle(vectors={"x": FloatVec.from_list([2.0])})
        result = verify_states(a, b)
        assert result.passed is False

    def test_within_tolerance(self):
        a = StateBundle(vectors={"x": FloatVec.from_list([1.0])})
        b = StateBundle(vectors={"x": FloatVec.from_list([1.00001])})
        result = verify_states(a, b, default_spec=VerifySpec.exact(tolerance=0.001))
        assert result.passed is True


class TestVerifyMissingExtra:
    def test_missing_key(self):
        a = StateBundle(vectors={"x": FloatVec.from_list([1.0]), "y": FloatVec.from_list([2.0])})
        b = StateBundle(vectors={"x": FloatVec.from_list([1.0])})
        result = verify_states(a, b)
        assert result.passed is False
        assert "y" in result.missing_keys

    def test_extra_key_strict(self):
        a = StateBundle(vectors={"x": FloatVec.from_list([1.0])})
        b = StateBundle(vectors={"x": FloatVec.from_list([1.0]), "z": FloatVec.from_list([3.0])})
        result = verify_states(a, b, strict_keys=True)
        assert result.passed is False
        assert "z" in result.extra_keys

    def test_extra_key_lenient(self):
        a = StateBundle(vectors={"x": FloatVec.from_list([1.0])})
        b = StateBundle(vectors={"x": FloatVec.from_list([1.0]), "z": FloatVec.from_list([3.0])})
        result = verify_states(a, b)
        assert result.passed is True
        assert "z" in result.extra_keys


class TestVerifyCosine:
    def test_same_direction(self):
        a = StateBundle(vectors={"v": FloatVec.from_list([1.0, 0.0])})
        b = StateBundle(vectors={"v": FloatVec.from_list([2.0, 0.0])})
        result = verify_states(a, b, default_spec=VerifySpec.cosine(tolerance=0.01))
        assert result.passed is True

    def test_orthogonal(self):
        a = StateBundle(vectors={"v": FloatVec.from_list([1.0, 0.0])})
        b = StateBundle(vectors={"v": FloatVec.from_list([0.0, 1.0])})
        result = verify_states(a, b, default_spec=VerifySpec.cosine(tolerance=0.01))
        assert result.passed is False


class TestVerifyEuclidean:
    def test_close_enough(self):
        a = StateBundle(vectors={"v": FloatVec.from_list([1.0, 2.0])})
        b = StateBundle(vectors={"v": FloatVec.from_list([1.05, 2.05])})
        result = verify_states(a, b, default_spec=VerifySpec.euclidean(tolerance=0.1))
        assert result.passed is True

    def test_too_far(self):
        a = StateBundle(vectors={"v": FloatVec.from_list([0.0])})
        b = StateBundle(vectors={"v": FloatVec.from_list([5.0])})
        result = verify_states(a, b, default_spec=VerifySpec.euclidean(tolerance=0.1))
        assert result.passed is False


class TestVerifyPerKeySpec:
    def test_mixed_specs(self):
        expected = StateBundle(vectors={
            "pos": FloatVec.from_list([1.0, 0.0]),
            "hp": FloatVec.from_list([100.0]),
        })
        actual = StateBundle(vectors={
            "pos": FloatVec.from_list([2.0, 0.0]),   # same direction
            "hp": FloatVec.from_list([100.0]),
        })
        specs = {
            "pos": VerifySpec.cosine(tolerance=0.01),
            "hp": VerifySpec.exact(),
        }
        result = verify_states(expected, actual, specs=specs)
        assert result.passed is True


class TestVerifySummary:
    def test_summary_contains_status(self):
        a = StateBundle(vectors={"x": FloatVec.from_list([1.0])})
        result = verify_states(a, a)
        s = result.summary()
        assert "PASS" in s
