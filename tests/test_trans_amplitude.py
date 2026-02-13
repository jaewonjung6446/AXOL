"""Unit tests for TransAmplitude — open relation type + operations."""

import numpy as np
import pytest

from axol.core.types import TransMatrix
from axol.core.amplitude import Amplitude
from axol.core.amplitude_ops import amp_transform, amp_observe
from axol.core.trans_amplitude import (
    TransAmplitude,
    trans_amp_apply,
    trans_amp_compose,
    trans_amp_superpose,
    trans_amp_collapse,
    trans_amp_effective,
)


# ---------------------------------------------------------------------------
# 1. Construction + properties
# ---------------------------------------------------------------------------

class TestTransAmplitudeConstruction:

    def test_from_matrix_single_path(self):
        """from_matrix wraps a single TransMatrix as 1-path TransAmplitude."""
        M = TransMatrix.identity(3)
        ta = TransAmplitude.from_matrix(M)
        assert ta.n_paths == 1
        assert ta.input_dim == 3
        assert ta.output_dim == 3

    def test_from_matrices_multiple(self):
        """from_matrices creates a multi-path TransAmplitude."""
        I = TransMatrix.identity(3)
        P = TransMatrix(data=np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32))
        ta = TransAmplitude.from_matrices([I, P], [1.0, 1j])
        assert ta.n_paths == 2
        assert ta.input_dim == 3
        assert ta.output_dim == 3

    def test_uniform_constructor(self):
        """uniform() creates equal-weight superposition."""
        I = TransMatrix.identity(2)
        Z = TransMatrix(data=np.array([[1, 0], [0, -1]], dtype=np.float32))
        ta = TransAmplitude.uniform([I, Z])
        assert ta.n_paths == 2
        np.testing.assert_allclose(
            np.abs(ta.weights), 1.0 / np.sqrt(2), atol=1e-12
        )

    def test_from_kraus(self):
        """from_kraus creates from raw numpy arrays."""
        ops = [np.eye(2), np.array([[0, 1], [1, 0]])]
        ta = TransAmplitude.from_kraus(ops)
        assert ta.n_paths == 2
        assert ta.input_dim == 2

    def test_auto_normalize_weights(self):
        """Weights are auto-normalized to unit norm."""
        M = TransMatrix.identity(2)
        ta = TransAmplitude.from_matrices([M, M], [3.0, 4.0])
        assert np.isclose(np.linalg.norm(ta.weights), 1.0, atol=1e-12)

    def test_shape_validation_not_3d(self):
        """matrices must be 3D."""
        with pytest.raises(ValueError, match="3-dimensional"):
            TransAmplitude(
                matrices=np.eye(3, dtype=np.float64),
                weights=np.array([1.0 + 0j]),
            )

    def test_shape_validation_mismatch(self):
        """matrices K must match weights K."""
        with pytest.raises(ValueError, match="paths"):
            TransAmplitude(
                matrices=np.zeros((2, 3, 3), dtype=np.float64),
                weights=np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j]),
            )


class TestTransAmplitudeProperties:

    def test_negativity_positive(self):
        """Negative real weights produce positive negativity."""
        I = TransMatrix.identity(2)
        ta = TransAmplitude.from_matrices([I, I], [1.0, -1.0])
        assert ta.negativity > 0

    def test_negativity_zero_classical(self):
        """All-positive weights have zero negativity."""
        I = TransMatrix.identity(2)
        ta = TransAmplitude.from_matrices([I, I], [0.6, 0.8])
        assert np.isclose(ta.negativity, 0.0)

    def test_is_classical_true(self):
        """Positive real weights → classical."""
        I = TransMatrix.identity(2)
        ta = TransAmplitude.from_matrices([I, I], [0.6, 0.8])
        assert ta.is_classical is True

    def test_is_classical_false_complex(self):
        """Complex weights → not classical."""
        I = TransMatrix.identity(2)
        ta = TransAmplitude.from_matrices([I, I], [1.0, 1j])
        assert ta.is_classical is False

    def test_is_classical_false_negative(self):
        """Negative weights → not classical."""
        I = TransMatrix.identity(2)
        ta = TransAmplitude.from_matrices([I, I], [1.0, -1.0])
        assert ta.is_classical is False

    def test_path_probabilities(self):
        """|w_k|^2 sums to 1."""
        I = TransMatrix.identity(2)
        ta = TransAmplitude.from_matrices([I, I, I], [1.0, 1j, -1.0])
        probs = ta.path_probabilities
        assert np.isclose(np.sum(probs), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. trans_amp_apply
# ---------------------------------------------------------------------------

class TestTransAmpApply:

    def test_identity_single_path(self):
        """Single identity path preserves the amplitude."""
        amp = Amplitude.uniform(3)
        ta = TransAmplitude.from_matrix(TransMatrix.identity(3))
        result = trans_amp_apply(amp, ta)
        np.testing.assert_allclose(
            result.probabilities, amp.probabilities, atol=1e-6
        )

    def test_permutation_single_path(self):
        """Single permutation path rotates amplitudes."""
        amp = Amplitude.basis(0, 3)
        P = TransMatrix(data=np.array(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32
        ))
        ta = TransAmplitude.from_matrix(P)
        result = trans_amp_apply(amp, ta)
        assert result.collapse() == 1

    def test_dimension_mismatch(self):
        """Mismatched dimensions raise ValueError."""
        amp = Amplitude.uniform(3)
        ta = TransAmplitude.from_matrix(TransMatrix.identity(4))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            trans_amp_apply(amp, ta)

    def test_two_path_superposition(self):
        """Two-path superposition produces a valid amplitude."""
        amp = Amplitude.basis(0, 2)
        I = TransMatrix.identity(2)
        X = TransMatrix(data=np.array([[0, 1], [1, 0]], dtype=np.float32))
        ta = TransAmplitude.from_matrices([I, X], [1.0, 1.0])
        result = trans_amp_apply(amp, ta)
        assert result.size == 2
        assert np.isclose(np.linalg.norm(result.data), 1.0, atol=1e-12)
        # Equal superposition of |0> and |1>
        np.testing.assert_allclose(result.probabilities, [0.5, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Interference proof
# ---------------------------------------------------------------------------

class TestInterference:

    def test_constructive_interference(self):
        """Same-phase weights → constructive interference (amplification)."""
        amp = Amplitude.basis(0, 2)
        # Two paths that both map |0> → |0>
        I = TransMatrix.identity(2)
        ta_constructive = TransAmplitude.from_matrices([I, I], [1.0, 1.0])
        result = trans_amp_apply(amp, ta_constructive)
        # Both paths reinforce |0> → high probability at index 0
        assert result.collapse() == 0
        assert result.probabilities[0] > 0.99

    def test_destructive_interference(self):
        """Opposite-phase weights → destructive interference (cancellation)."""
        amp = Amplitude.uniform(2)
        # Identity and bit-flip with opposite phases
        I = TransMatrix.identity(2)
        X = TransMatrix(data=np.array([[0, 1], [1, 0]], dtype=np.float32))
        # Same phase → specific distribution
        ta_same = TransAmplitude.from_matrices([I, X], [1.0, 1.0])
        result_same = trans_amp_apply(amp, ta_same)
        # Opposite phase → different distribution (destructive)
        ta_opp = TransAmplitude.from_matrices([I, X], [1.0, -1.0])
        result_opp = trans_amp_apply(amp, ta_opp)
        # The two results should differ due to interference
        assert not np.allclose(
            result_same.probabilities, result_opp.probabilities, atol=0.01
        )

    def test_phase_interference_changes_distribution(self):
        """Complex phase weights produce measurably different distributions."""
        dim = 3
        # Asymmetric input
        amp = Amplitude(data=np.array([0.9, 0.3, 0.1], dtype=np.complex128))
        M1 = TransMatrix.identity(dim)
        # Cyclic permutation: a very different transformation from identity
        M2 = TransMatrix(data=np.array(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32
        ))

        # Weight with phase 0: constructive in some directions
        ta_0 = TransAmplitude.from_matrices([M1, M2], [1.0, 1.0])
        r0 = trans_amp_apply(amp, ta_0)

        # Weight with phase pi: destructive interference
        ta_pi = TransAmplitude.from_matrices([M1, M2], [1.0, -1.0])
        r_pi = trans_amp_apply(amp, ta_pi)

        # Distributions should differ significantly
        assert not np.allclose(r0.probabilities, r_pi.probabilities, atol=0.01)


# ---------------------------------------------------------------------------
# 4. Composition
# ---------------------------------------------------------------------------

class TestTransAmpCompose:

    def test_compose_path_count(self):
        """Compose K1=2, K2=3 → K=6 paths."""
        I2 = TransMatrix.identity(2)
        ta1 = TransAmplitude.from_matrices([I2, I2], [1.0, 1j])
        ta2 = TransAmplitude.from_matrices([I2, I2, I2], [1.0, 0.5, -0.5])
        composed = trans_amp_compose(ta1, ta2)
        assert composed.n_paths == 6
        assert composed.input_dim == 2
        assert composed.output_dim == 2

    def test_compose_dimension_check(self):
        """Composition requires ta1.output_dim == ta2.input_dim."""
        M23 = TransMatrix(data=np.zeros((2, 3), dtype=np.float32))
        M34 = TransMatrix(data=np.zeros((3, 4), dtype=np.float32))
        M55 = TransMatrix(data=np.zeros((5, 5), dtype=np.float32))
        ta1 = TransAmplitude.from_matrix(M23)
        ta2 = TransAmplitude.from_matrix(M34)
        ta_bad = TransAmplitude.from_matrix(M55)

        composed = trans_amp_compose(ta1, ta2)
        assert composed.input_dim == 2
        assert composed.output_dim == 4

        with pytest.raises(ValueError, match="Dimension mismatch"):
            trans_amp_compose(ta1, ta_bad)

    def test_compose_identity(self):
        """Composing with identity preserves behavior."""
        I3 = TransMatrix.identity(3)
        P = TransMatrix(data=np.array(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32
        ))
        ta = TransAmplitude.from_matrices([I3, P], [0.8, 0.6])
        ta_id = TransAmplitude.from_matrix(I3)

        composed = trans_amp_compose(ta, ta_id)
        assert composed.n_paths == 2  # 2 * 1

        amp = Amplitude.basis(0, 3)
        r_orig = trans_amp_apply(amp, ta)
        r_composed = trans_amp_apply(amp, composed)
        np.testing.assert_allclose(
            r_orig.probabilities, r_composed.probabilities, atol=1e-6
        )


# ---------------------------------------------------------------------------
# 5. Classical degeneration
# ---------------------------------------------------------------------------

class TestClassicalDegeneration:

    def test_from_matrix_matches_amp_transform(self):
        """TransAmplitude.from_matrix(M) should match amp_transform(amp, M)."""
        M = TransMatrix(data=np.array(
            [[0.5, 0.5, 0.0],
             [0.0, 0.5, 0.5],
             [0.5, 0.0, 0.5]], dtype=np.float32
        ))
        amp = Amplitude(data=np.array([0.8, 0.5, 0.3], dtype=np.complex128))

        classical_result = amp_transform(amp, M)
        ta = TransAmplitude.from_matrix(M)
        ta_result = trans_amp_apply(amp, ta)

        np.testing.assert_allclose(
            ta_result.probabilities, classical_result.probabilities, atol=1e-6
        )

    def test_single_path_is_classical(self):
        """A single-path TransAmplitude is classical."""
        ta = TransAmplitude.from_matrix(TransMatrix.identity(4))
        assert ta.is_classical is True
        assert np.isclose(ta.negativity, 0.0)


# ---------------------------------------------------------------------------
# 6. Effective matrix
# ---------------------------------------------------------------------------

class TestTransAmpEffective:

    def test_single_path_effective_equals_original(self):
        """Effective matrix of single-path = original matrix."""
        M_data = np.array(
            [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
        )
        M = TransMatrix(data=M_data)
        ta = TransAmplitude.from_matrix(M)
        eff = trans_amp_effective(ta)
        np.testing.assert_allclose(eff.data, M.data, atol=1e-5)

    def test_effective_includes_interference(self):
        """Effective matrix is the weighted sum (with interference)."""
        I = TransMatrix.identity(2)
        Z = TransMatrix(data=np.array([[1, 0], [0, -1]], dtype=np.float32))
        # Equal superposition: M_eff = 0.5*I + 0.5*Z = [[1,0],[0,0]]
        ta = TransAmplitude.from_matrices([I, Z], [1.0, 1.0])
        eff = trans_amp_effective(ta)
        # After normalization of weights: w = [1/sqrt(2), 1/sqrt(2)]
        # M_eff = 1/sqrt(2) * I + 1/sqrt(2) * Z
        #       = 1/sqrt(2) * [[1,0],[0,1]] + 1/sqrt(2) * [[1,0],[0,-1]]
        #       = [[sqrt(2), 0], [0, 0]]
        s = 1.0 / np.sqrt(2)
        expected = np.array([[s + s, 0], [0, s - s]], dtype=np.float32)
        np.testing.assert_allclose(eff.data, expected, atol=1e-5)

    def test_destructive_effective_cancels(self):
        """Opposite-weight paths cancel in effective matrix."""
        M = TransMatrix(data=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        # +M and -M → M_eff = 0
        ta = TransAmplitude.from_matrices([M, M], [1.0, -1.0])
        eff = trans_amp_effective(ta)
        np.testing.assert_allclose(eff.data, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 7. Collapse
# ---------------------------------------------------------------------------

class TestTransAmpCollapse:

    def test_collapse_returns_most_probable(self):
        """Collapse picks the path with highest |w_k|^2."""
        I = TransMatrix.identity(2)
        X = TransMatrix(data=np.array([[0, 1], [1, 0]], dtype=np.float32))
        # Weight 0.9 for I, 0.1 for X → I is most probable
        ta = TransAmplitude.from_matrices([I, X], [0.9, 0.1])
        matrix, idx = trans_amp_collapse(ta)
        assert idx == 0
        np.testing.assert_allclose(matrix.data, np.eye(2, dtype=np.float32), atol=1e-5)

    def test_collapse_returns_transmatrix(self):
        """Collapsed result is a proper TransMatrix."""
        M = TransMatrix(data=np.array([[1, 2], [3, 4]], dtype=np.float32))
        ta = TransAmplitude.from_matrix(M)
        matrix, idx = trans_amp_collapse(ta)
        assert isinstance(matrix, TransMatrix)
        assert idx == 0


# ---------------------------------------------------------------------------
# 8. Superpose
# ---------------------------------------------------------------------------

class TestTransAmpSuperpose:

    def test_superpose_concatenates_paths(self):
        """Superpose concatenates all paths."""
        I = TransMatrix.identity(2)
        ta1 = TransAmplitude.from_matrices([I, I], [1.0, 0.5])
        ta2 = TransAmplitude.from_matrices([I], [1.0])
        combined = trans_amp_superpose([ta1, ta2], [1.0, 1.0])
        assert combined.n_paths == 3

    def test_superpose_dimension_mismatch(self):
        """Superposing incompatible dimensions raises ValueError."""
        ta1 = TransAmplitude.from_matrix(TransMatrix.identity(2))
        ta2 = TransAmplitude.from_matrix(TransMatrix.identity(3))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            trans_amp_superpose([ta1, ta2], [1.0, 1.0])

    def test_superpose_preserves_weights(self):
        """Outer weights scale inner weights correctly."""
        I = TransMatrix.identity(2)
        ta = TransAmplitude.from_matrices([I], [1.0])
        # Scale by 2.0 — after auto-normalization, the result is still valid
        combined = trans_amp_superpose([ta, ta], [1.0, 1j])
        assert combined.n_paths == 2
        assert np.isclose(np.linalg.norm(combined.weights), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. Repr
# ---------------------------------------------------------------------------

class TestTransAmplitudeRepr:

    def test_repr_format(self):
        ta = TransAmplitude.from_matrix(TransMatrix.identity(3))
        r = repr(ta)
        assert "TransAmplitude" in r
        assert "1 paths" in r
        assert "3→3" in r
        assert "neg=" in r
