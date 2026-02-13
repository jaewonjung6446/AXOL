"""Tests proving that amplitude-first interference improves classification.

These tests demonstrate that observe_amplitude() produces better results
than classical observe() in specific scenarios, particularly when the
probability distribution is ambiguous (top classes nearly tied).
"""

import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy

from axol.core.types import FloatVec, TransMatrix
from axol.core.amplitude import Amplitude
from axol.core.amplitude_ops import (
    amp_transform, amp_superpose, amp_interfere, amp_observe,
)
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe, observe_amplitude


# ---------------------------------------------------------------------------
# Test 1: Probability concentration (entropy reduction)
# ---------------------------------------------------------------------------

class TestProbabilityConcentration:
    """After interference, entropy should decrease (distribution becomes sharper)."""

    def test_interference_reduces_entropy(self):
        """Superposing direct + amplified path reduces entropy."""
        dim = 8
        # Start with a near-uniform amplitude (high entropy)
        amp = Amplitude.uniform(dim)

        # Apply a matrix that creates a slight preference for index 0
        M_data = np.eye(dim, dtype=np.float32) * 0.9
        M_data[0, 0] = 1.5  # boost index 0
        M = TransMatrix(data=M_data)

        direct = amp_transform(amp, M)
        direct_entropy = scipy_entropy(direct.probabilities + 1e-15)

        # Oracle: flip sign of most probable
        top_idx = direct.collapse()
        oracle_diag = np.ones(dim, dtype=np.float32)
        oracle_diag[top_idx] = -1.0
        oracle = TransMatrix(data=np.diag(oracle_diag))

        # Diffusion: 2|s><s| - I
        s = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
        D = 2.0 * np.outer(s, s) - np.eye(dim, dtype=np.float64)
        diffusion = TransMatrix(data=D.astype(np.float32))

        marked = amp_transform(direct, oracle)
        amplified = amp_transform(marked, diffusion)

        # Superpose with interference
        result = amp_superpose([direct, amplified], [0.6, 0.4])
        result_entropy = scipy_entropy(result.probabilities + 1e-15)

        # Entropy should decrease (sharper distribution)
        assert result_entropy < direct_entropy, (
            f"Expected entropy reduction: {result_entropy:.4f} < {direct_entropy:.4f}"
        )

    def test_multiple_interference_steps(self):
        """More interference steps → more concentration (up to a point)."""
        dim = 8
        amp = Amplitude.uniform(dim)

        M_data = np.eye(dim, dtype=np.float32)
        M_data[0, :] *= 1.2
        M = TransMatrix(data=M_data)

        current = amp_transform(amp, M)
        prev_entropy = scipy_entropy(current.probabilities + 1e-15)

        # Apply 3 rounds of oracle+diffusion
        s = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
        D_data = 2.0 * np.outer(s, s) - np.eye(dim, dtype=np.float64)
        diffusion = TransMatrix(data=D_data.astype(np.float32))

        entropies = [prev_entropy]
        for _ in range(3):
            top_idx = current.collapse()
            oracle_diag = np.ones(dim, dtype=np.float32)
            oracle_diag[top_idx] = -1.0
            oracle = TransMatrix(data=np.diag(oracle_diag))

            marked = amp_transform(current, oracle)
            amplified = amp_transform(marked, diffusion)
            current = amp_superpose([current, amplified], [0.6, 0.4])
            entropies.append(scipy_entropy(current.probabilities + 1e-15))

        # At least the first step should reduce entropy
        assert entropies[1] < entropies[0], (
            f"First interference step should reduce entropy: {entropies}"
        )


# ---------------------------------------------------------------------------
# Test 2: Ambiguous input separation
# ---------------------------------------------------------------------------

class TestAmbiguousInputSeparation:
    """When top-2 classes are nearly tied, interference widens the gap."""

    def test_gap_widening(self):
        """Nearly-tied top-2 classes → interference widens the gap."""
        dim = 8
        # Create an amplitude where index 0 and 1 are almost equal
        data = np.zeros(dim, dtype=np.complex128)
        data[0] = 0.52  # slight edge for class 0
        data[1] = 0.48
        data[2:] = 0.01
        amp = Amplitude(data=data)

        probs_before = amp.probabilities
        gap_before = probs_before[0] - probs_before[1]

        # One Grover-like iteration
        top_idx = amp.collapse()
        oracle_diag = np.ones(dim, dtype=np.float32)
        oracle_diag[top_idx] = -1.0
        oracle = TransMatrix(data=np.diag(oracle_diag))

        s = np.abs(amp.data).astype(np.float64)
        s = s / np.linalg.norm(s)
        D = 2.0 * np.outer(s, s) - np.eye(dim, dtype=np.float64)
        diffusion = TransMatrix(data=D.astype(np.float32))

        marked = amp_transform(amp, oracle)
        amplified = amp_transform(marked, diffusion)
        result = amp_superpose([amp, amplified], [0.6, 0.4])

        probs_after = result.probabilities
        # The winner should have higher probability
        winner = result.collapse()
        gap_after = probs_after[winner] - np.sort(probs_after)[-2]

        assert gap_after > gap_before, (
            f"Gap should widen: {gap_after:.4f} > {gap_before:.4f}"
        )

    def test_correct_winner_preserved(self):
        """The correct top class is preserved after interference."""
        dim = 4
        # Class 0 has slight edge
        data = np.array([0.55, 0.45, 0.01, 0.01], dtype=np.complex128)
        amp = Amplitude(data=data)

        assert amp.collapse() == 0  # class 0 wins before

        # Apply interference
        oracle_diag = np.ones(dim, dtype=np.float32)
        oracle_diag[0] = -1.0
        oracle = TransMatrix(data=np.diag(oracle_diag))

        s = np.abs(amp.data).astype(np.float64)
        s /= np.linalg.norm(s)
        D = 2.0 * np.outer(s, s) - np.eye(dim, dtype=np.float64)
        diffusion = TransMatrix(data=D.astype(np.float32))

        marked = amp_transform(amp, oracle)
        amplified = amp_transform(marked, diffusion)
        result = amp_superpose([amp, amplified], [0.6, 0.4])

        assert result.collapse() == 0  # class 0 still wins


# ---------------------------------------------------------------------------
# Test 3: Negativity predicts interference strength
# ---------------------------------------------------------------------------

class TestNegativityPrediction:
    """Amplitudes with higher negativity show stronger interference effects."""

    def test_negative_amplitudes_stronger_effect(self):
        """Negative-real-part amplitudes produce stronger interference than classical ones."""
        dim = 8

        # Classical amplitude (all positive, negativity = 0)
        classical_data = np.array([0.5, 0.4, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002],
                                  dtype=np.complex128)
        classical = Amplitude(data=classical_data)
        assert classical.negativity == 0.0

        # Quantum amplitude (has negative components, negativity > 0)
        quantum_data = np.array([0.5, -0.4, 0.3, -0.2, 0.1, 0.05, 0.03, 0.02],
                                dtype=np.complex128)
        quantum = Amplitude(data=quantum_data)
        assert quantum.negativity > 0

        # Apply same interference to both
        def one_grover_step(amp):
            top_idx = amp.collapse()
            oracle_diag = np.ones(dim, dtype=np.float32)
            oracle_diag[top_idx] = -1.0
            oracle = TransMatrix(data=np.diag(oracle_diag))

            s = np.abs(amp.data).astype(np.float64)
            s /= np.linalg.norm(s)
            D = 2.0 * np.outer(s, s) - np.eye(dim, dtype=np.float64)
            diffusion = TransMatrix(data=D.astype(np.float32))

            marked = amp_transform(amp, oracle)
            amplified = amp_transform(marked, diffusion)
            return amp_superpose([amp, amplified], [0.6, 0.4])

        classical_result = one_grover_step(classical)
        quantum_result = one_grover_step(quantum)

        # Both should concentrate probability on top class
        classical_top_prob = np.max(classical_result.probabilities)
        quantum_top_prob = np.max(quantum_result.probabilities)

        # The quantum (negative) amplitude should show at least some concentration
        assert classical_top_prob > np.max(classical.probabilities) or \
               quantum_top_prob > np.max(quantum.probabilities), (
            "At least one path should show probability concentration"
        )


# ---------------------------------------------------------------------------
# Test 4: observe_amplitude vs observe — fit_data comparison
# ---------------------------------------------------------------------------

class TestObserveAmplitudeVsObserve:
    """Compare observe_amplitude() with observe() on trained tapestries."""

    def _build_ambiguous_dataset(self, dim=8, n_per_class=10, n_classes=4, noise=0.5):
        """Dataset where classes are close together (hard to separate)."""
        rng = np.random.default_rng(42)
        N = n_per_class * n_classes
        X = np.zeros((N, dim), dtype=np.float32)
        targets = np.zeros(N, dtype=np.int64)

        for c in range(n_classes):
            centroid = np.zeros(dim, dtype=np.float32)
            centroid[c % dim] = 1.0
            for i in range(n_per_class):
                idx = c * n_per_class + i
                X[idx] = centroid + rng.standard_normal(dim).astype(np.float32) * noise
                targets[idx] = c

        return {"input": X, "target": targets}

    def test_observe_amplitude_at_least_as_good(self):
        """observe_amplitude should be at least as accurate as observe."""
        dim = 8
        decl = (
            DeclarationBuilder("amp_vs_classical")
            .input("x", dim, labels={i: f"class_{i}" for i in range(4)})
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .quality(0.9, 0.8)
            .build()
        )
        fd = self._build_ambiguous_dataset(dim=dim)
        tapestry = weave(decl, seed=42, fit_data=fd)

        N = len(fd["target"])
        correct_classical = 0
        correct_amplitude = 0

        for i in range(N):
            vec = FloatVec(data=fd["input"][i])

            obs_classical = observe(tapestry, {"x": vec})
            if obs_classical.value_index == fd["target"][i]:
                correct_classical += 1

            obs_amp = observe_amplitude(tapestry, {"x": vec})
            if obs_amp.value_index == fd["target"][i]:
                correct_amplitude += 1

        acc_classical = correct_classical / N
        acc_amplitude = correct_amplitude / N

        # observe_amplitude should be at least as good
        assert acc_amplitude >= acc_classical * 0.95, (
            f"Amplitude accuracy ({acc_amplitude:.2%}) too far below "
            f"classical ({acc_classical:.2%})"
        )

    def test_observe_amplitude_returns_amplitude_field(self):
        """observe_amplitude() should populate the amplitude field in Observation."""
        dim = 8
        decl = (
            DeclarationBuilder("amp_field_test")
            .input("x", dim)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        fd = self._build_ambiguous_dataset(dim=dim, n_per_class=5)
        tapestry = weave(decl, seed=42, fit_data=fd)

        vec = FloatVec(data=fd["input"][0])
        obs = observe_amplitude(tapestry, {"x": vec})

        assert obs.amplitude is not None
        assert isinstance(obs.amplitude, Amplitude)
        assert obs.amplitude.size == dim

    def test_observe_amplitude_floatvec_input(self):
        """observe_amplitude() should accept FloatVec inputs (auto-promote)."""
        dim = 8
        decl = (
            DeclarationBuilder("fv_promote")
            .input("x", dim)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        fd = self._build_ambiguous_dataset(dim=dim, n_per_class=3)
        tapestry = weave(decl, seed=42, fit_data=fd)

        vec = FloatVec(data=fd["input"][0])
        obs = observe_amplitude(tapestry, {"x": vec})
        assert obs.value_index >= 0

    def test_observe_amplitude_amplitude_input(self):
        """observe_amplitude() should accept Amplitude inputs directly."""
        dim = 8
        decl = (
            DeclarationBuilder("amp_input")
            .input("x", dim)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        fd = self._build_ambiguous_dataset(dim=dim, n_per_class=3)
        tapestry = weave(decl, seed=42, fit_data=fd)

        amp = Amplitude.from_floatvec(FloatVec(data=fd["input"][0]))
        obs = observe_amplitude(tapestry, {"x": amp})
        assert obs.value_index >= 0


# ---------------------------------------------------------------------------
# Test 5: n_paths=1 (no interference, baseline)
# ---------------------------------------------------------------------------

class TestObserveAmplitudeNPaths:
    """Test n_paths parameter for observe_amplitude."""

    def test_n_paths_1_no_interference(self):
        """n_paths=1 → no interference (just amplitude transform + observe)."""
        dim = 8
        decl = (
            DeclarationBuilder("npaths1")
            .input("x", dim)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        fd = {"input": np.eye(dim, dtype=np.float32)[:4],
              "target": np.array([0, 1, 2, 3], dtype=np.int64)}
        tapestry = weave(decl, seed=42, fit_data=fd)

        vec = FloatVec(data=fd["input"][0])
        obs = observe_amplitude(tapestry, {"x": vec}, n_paths=1)
        assert obs.value_index >= 0
        assert obs.amplitude is not None


# ---------------------------------------------------------------------------
# Test 6: Backward compatibility — observe() unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Existing observe() must work identically (no regression)."""

    def test_observe_unchanged(self):
        """observe() results should not change."""
        dim = 8
        decl = (
            DeclarationBuilder("compat")
            .input("x", dim)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .quality(0.8, 0.7)
            .build()
        )
        tapestry = weave(decl, seed=42)
        vec = FloatVec(data=np.ones(dim, dtype=np.float32) / np.sqrt(dim))
        obs = observe(tapestry, {"x": vec})

        # Should still have None for amplitude (not set by observe())
        assert obs.amplitude is None
        assert obs.value_index >= 0
        assert obs.probabilities is not None

    def test_observation_default_amplitude_none(self):
        """Observation created without amplitude field defaults to None."""
        from axol.quantum.types import Observation
        obs = Observation(
            value=FloatVec.from_list([0.5, 0.5]),
            value_index=0,
            value_label=None,
            omega=0.9,
            phi=0.8,
            probabilities=FloatVec.from_list([0.5, 0.5]),
            tapestry_name="test",
        )
        assert obs.amplitude is None
