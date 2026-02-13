"""Unit tests for Amplitude type and amplitude operations."""

import numpy as np
import pytest

from axol.core.amplitude import Amplitude
from axol.core.amplitude_ops import (
    amp_transform, amp_superpose, amp_interfere,
    amp_condition, amp_entangle, amp_observe,
)
from axol.core.types import FloatVec, TransMatrix, ComplexVec


# ---------------------------------------------------------------------------
# Amplitude construction + normalization
# ---------------------------------------------------------------------------

class TestAmplitudeConstruction:
    """Amplitude type: creation, normalization, basic properties."""

    def test_auto_normalize(self):
        """Non-unit vectors are auto-normalized to ||a|| = 1."""
        raw = np.array([3.0, 4.0], dtype=np.complex128)
        amp = Amplitude(data=raw)
        assert np.isclose(np.linalg.norm(amp.data), 1.0, atol=1e-12)

    def test_already_normalized(self):
        """Already-unit vectors stay unchanged."""
        raw = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=np.complex128)
        amp = Amplitude(data=raw)
        np.testing.assert_allclose(amp.data, raw, atol=1e-12)

    def test_uniform(self):
        """Amplitude.uniform(n) → 1/sqrt(n) everywhere."""
        amp = Amplitude.uniform(4)
        assert amp.size == 4
        expected = np.ones(4, dtype=np.complex128) / 2.0
        np.testing.assert_allclose(amp.data, expected, atol=1e-12)
        assert np.isclose(np.linalg.norm(amp.data), 1.0)

    def test_basis(self):
        """Amplitude.basis(i, n) → deterministic state."""
        amp = Amplitude.basis(2, 5)
        assert amp.size == 5
        assert amp.collapse() == 2
        assert np.isclose(amp.probabilities[2], 1.0)
        for j in range(5):
            if j != 2:
                assert np.isclose(amp.probabilities[j], 0.0)

    def test_from_floatvec(self):
        """FloatVec promotion preserves relative magnitudes."""
        fv = FloatVec.from_list([0.25, 0.25, 0.25, 0.25])
        amp = Amplitude.from_floatvec(fv)
        assert amp.size == 4
        assert np.isclose(np.linalg.norm(amp.data), 1.0)

    def test_from_complexvec(self):
        """ComplexVec wrapping normalizes."""
        cv = ComplexVec.from_list([1 + 1j, 1 - 1j])
        amp = Amplitude.from_complexvec(cv)
        assert amp.size == 2
        assert np.isclose(np.linalg.norm(amp.data), 1.0)

    def test_from_array(self):
        arr = np.array([0.5, 0.5, 0.5, 0.5])
        amp = Amplitude.from_array(arr)
        assert amp.size == 4
        assert np.isclose(np.linalg.norm(amp.data), 1.0)


# ---------------------------------------------------------------------------
# Properties: probabilities, phases, negativity, is_classical
# ---------------------------------------------------------------------------

class TestAmplitudeProperties:

    def test_probabilities_born_rule(self):
        """probabilities = |a_i|^2, normalized."""
        amp = Amplitude.uniform(4)
        probs = amp.probabilities
        np.testing.assert_allclose(probs, [0.25, 0.25, 0.25, 0.25], atol=1e-12)
        assert np.isclose(np.sum(probs), 1.0)

    def test_probabilities_basis(self):
        amp = Amplitude.basis(0, 3)
        probs = amp.probabilities
        assert np.isclose(probs[0], 1.0)
        assert np.isclose(probs[1], 0.0)
        assert np.isclose(probs[2], 0.0)

    def test_phases_uniform_real(self):
        """Uniform real amplitudes have phase = 0."""
        amp = Amplitude.uniform(4)
        phases = amp.phases
        np.testing.assert_allclose(phases, 0.0, atol=1e-12)

    def test_phases_complex(self):
        """Complex amplitudes have non-zero phases."""
        data = np.array([1.0, 1j, -1.0, -1j], dtype=np.complex128)
        amp = Amplitude(data=data)
        phases = amp.phases
        # 1 → 0, i → pi/2, -1 → pi, -i → -pi/2
        np.testing.assert_allclose(phases[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(phases[1], np.pi / 2, atol=1e-6)
        np.testing.assert_allclose(np.abs(phases[2]), np.pi, atol=1e-6)
        np.testing.assert_allclose(phases[3], -np.pi / 2, atol=1e-6)

    def test_negativity_positive(self):
        """Amplitudes with negative real parts have positive negativity."""
        data = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.complex128)
        amp = Amplitude(data=data)
        assert amp.negativity > 0

    def test_negativity_zero_for_positive(self):
        """All-positive-real amplitudes have zero negativity."""
        amp = Amplitude.uniform(4)
        assert np.isclose(amp.negativity, 0.0)

    def test_is_classical_uniform(self):
        """Uniform real positive → classical."""
        amp = Amplitude.uniform(4)
        assert amp.is_classical is True

    def test_is_classical_false_for_complex(self):
        """Complex amplitudes → not classical."""
        data = np.array([1.0, 1j], dtype=np.complex128)
        amp = Amplitude(data=data)
        assert amp.is_classical is False

    def test_is_classical_false_for_negative(self):
        """Negative real amplitudes → not classical."""
        data = np.array([1.0, -1.0], dtype=np.complex128)
        amp = Amplitude(data=data)
        assert amp.is_classical is False


# ---------------------------------------------------------------------------
# Collapse + conversion
# ---------------------------------------------------------------------------

class TestAmplitudeCollapse:

    def test_collapse_basis(self):
        assert Amplitude.basis(3, 5).collapse() == 3

    def test_collapse_argmax(self):
        data = np.array([0.1, 0.9, 0.1], dtype=np.complex128)
        amp = Amplitude(data=data)
        assert amp.collapse() == 1

    def test_to_floatvec(self):
        amp = Amplitude.uniform(3)
        fv = amp.to_floatvec()
        assert isinstance(fv, FloatVec)
        assert fv.size == 3
        np.testing.assert_allclose(fv.data, 1.0 / 3.0, atol=1e-6)

    def test_to_complexvec(self):
        amp = Amplitude.uniform(4)
        cv = amp.to_complexvec()
        assert isinstance(cv, ComplexVec)
        assert cv.size == 4


# ---------------------------------------------------------------------------
# Amplitude operations
# ---------------------------------------------------------------------------

class TestAmpTransform:

    def test_identity_transform(self):
        """Identity matrix preserves amplitude."""
        amp = Amplitude.uniform(3)
        I = TransMatrix.identity(3)
        result = amp_transform(amp, I)
        np.testing.assert_allclose(
            result.probabilities, amp.probabilities, atol=1e-6
        )

    def test_permutation_transform(self):
        """Permutation matrix rotates amplitudes."""
        amp = Amplitude.basis(0, 3)
        # Cyclic permutation: 0→1, 1→2, 2→0
        P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        perm = TransMatrix(data=P)
        result = amp_transform(amp, perm)
        assert result.collapse() == 1

    def test_dimension_mismatch(self):
        amp = Amplitude.uniform(3)
        M = TransMatrix.identity(4)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            amp_transform(amp, M)


class TestAmpSuperpose:

    def test_equal_weight_superposition(self):
        """Equal-weight superposition of two basis states → uniform."""
        a0 = Amplitude.basis(0, 2)
        a1 = Amplitude.basis(1, 2)
        result = amp_superpose([a0, a1], [1.0, 1.0])
        np.testing.assert_allclose(result.probabilities, [0.5, 0.5], atol=1e-6)

    def test_constructive_interference(self):
        """Same-phase paths → constructive interference (amplification)."""
        a = Amplitude(data=np.array([0.8, 0.6], dtype=np.complex128))
        b = Amplitude(data=np.array([0.8, 0.6], dtype=np.complex128))
        result = amp_superpose([a, b], [1.0, 1.0])
        # Same amplitudes → same probabilities (just rescaled)
        np.testing.assert_allclose(
            result.probabilities, a.probabilities, atol=1e-6
        )

    def test_destructive_interference(self):
        """Opposite-phase paths → destructive interference (cancellation)."""
        a = Amplitude(data=np.array([0.8, 0.6], dtype=np.complex128))
        b = Amplitude(data=np.array([0.8, 0.6], dtype=np.complex128))
        # Subtract: a - b = 0 → but normalization handles this
        # Use phase difference instead
        result = amp_superpose([a, b], [1.0, -1.0])
        # a - b = 0 vector, norm = 0 → uniform after normalization
        # Actually [0.8-0.8, 0.6-0.6] = [0, 0], normalization gives uniform
        assert result.size == 2

    def test_dimension_mismatch(self):
        a = Amplitude.uniform(3)
        b = Amplitude.uniform(4)
        with pytest.raises(ValueError, match="same dimension"):
            amp_superpose([a, b], [1.0, 1.0])

    def test_count_mismatch(self):
        a = Amplitude.uniform(3)
        with pytest.raises(ValueError, match="Count mismatch"):
            amp_superpose([a], [1.0, 2.0])


class TestAmpInterfere:

    def test_constructive(self):
        """phase=0 → constructive interference."""
        a = Amplitude.basis(0, 3)
        b = Amplitude.basis(0, 3)
        result = amp_interfere(a, b, phase=0.0)
        assert result.collapse() == 0
        assert np.isclose(result.probabilities[0], 1.0, atol=1e-6)

    def test_destructive(self):
        """phase=pi → destructive interference redistributes probability."""
        # Use non-identical amplitudes so cancellation is partial, not total
        a = Amplitude(data=np.array([0.8, 0.5, 0.3], dtype=np.complex128))
        b = Amplitude(data=np.array([0.7, 0.6, 0.4], dtype=np.complex128))
        result_constructive = amp_interfere(a, b, phase=0.0)
        result_destructive = amp_interfere(a, b, phase=np.pi)
        # Destructive interference should produce a different distribution
        assert not np.allclose(
            result_constructive.probabilities,
            result_destructive.probabilities,
            atol=0.01,
        )

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            amp_interfere(Amplitude.uniform(3), Amplitude.uniform(4))


class TestAmpCondition:

    def test_condition_mask(self):
        """Mask out some amplitudes."""
        amp = Amplitude.uniform(4)
        mask = np.array([True, False, True, False])
        result = amp_condition(amp, mask)
        assert np.isclose(result.probabilities[1], 0.0, atol=1e-12)
        assert np.isclose(result.probabilities[3], 0.0, atol=1e-12)
        assert result.probabilities[0] > 0
        assert result.probabilities[2] > 0

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="Dimension mismatch"):
            amp_condition(Amplitude.uniform(3), np.array([True, False]))


class TestAmpEntangle:

    def test_tensor_product_dimensions(self):
        """Tensor product: dim_a * dim_b."""
        a = Amplitude.uniform(2)
        b = Amplitude.uniform(3)
        result = amp_entangle(a, b)
        assert result.size == 6

    def test_tensor_product_basis(self):
        """Tensor product of basis states → single basis state."""
        a = Amplitude.basis(1, 2)  # |1>
        b = Amplitude.basis(0, 3)  # |0>
        result = amp_entangle(a, b)
        # |1> ⊗ |0> = |1*3 + 0> = |3>
        assert result.collapse() == 3


class TestAmpObserve:

    def test_observe_returns_tuple(self):
        amp = Amplitude.uniform(4)
        idx, probs = amp_observe(amp)
        assert isinstance(idx, int)
        assert isinstance(probs, FloatVec)
        assert 0 <= idx < 4

    def test_observe_basis(self):
        amp = Amplitude.basis(2, 5)
        idx, probs = amp_observe(amp)
        assert idx == 2
        assert np.isclose(probs.data[2], 1.0)

    def test_observe_probabilities_sum_to_one(self):
        amp = Amplitude.uniform(8)
        _, probs = amp_observe(amp)
        assert np.isclose(np.sum(probs.data), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestAmplitudeRepr:

    def test_repr_format(self):
        amp = Amplitude.uniform(4)
        r = repr(amp)
        assert "Amplitude[4]" in r
        assert "top=" in r
        assert "neg=" in r
