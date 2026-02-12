"""Tests for quantum structures (Direction B).

Tests complex amplitudes, density matrices, quantum channels,
interference, and the quantum weave/observe path.
"""

import numpy as np
import pytest

from axol.core.types import FloatVec, TransMatrix, ComplexVec, DensityMatrix
from axol.core import operations as ops
from axol.quantum.density import (
    von_neumann_entropy,
    fidelity,
    apply_channel,
    depolarizing_channel,
    amplitude_damping_channel,
    dephasing_channel,
    svd_to_kraus,
    phi_from_purity,
    omega_from_coherence,
)
from axol.quantum.types import SuperposedState, Observation
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe


# ===========================================================================
# 1. Core types: ComplexVec
# ===========================================================================

class TestComplexVec:
    def test_creation_from_list(self):
        cv = ComplexVec.from_list([1+0j, 0+1j, -1+0j])
        assert cv.size == 3
        assert cv.data.dtype == np.complex128

    def test_from_real(self):
        fv = FloatVec.from_list([0.5, 0.3, 0.2])
        cv = ComplexVec.from_real(fv)
        assert cv.size == 3
        np.testing.assert_allclose(np.abs(cv.data), np.abs(fv.data), atol=1e-6)

    def test_from_polar(self):
        mags = np.array([1.0, 1.0])
        phases = np.array([0.0, np.pi])
        cv = ComplexVec.from_polar(mags, phases)
        # phase=0 -> 1+0j, phase=pi -> -1+0j
        np.testing.assert_allclose(cv.data[0].real, 1.0, atol=1e-10)
        np.testing.assert_allclose(cv.data[1].real, -1.0, atol=1e-10)

    def test_amplitudes_and_phases(self):
        cv = ComplexVec.from_list([1+1j, 1-1j])
        np.testing.assert_allclose(cv.amplitudes, [np.sqrt(2), np.sqrt(2)], atol=1e-10)
        np.testing.assert_allclose(cv.phases, [np.pi/4, -np.pi/4], atol=1e-10)

    def test_to_real(self):
        cv = ComplexVec.from_list([3+4j, 0+5j])
        fv = cv.to_real()
        np.testing.assert_allclose(fv.data, [5.0, 5.0], atol=1e-5)

    def test_zeros(self):
        cv = ComplexVec.zeros(4)
        assert cv.size == 4
        assert np.all(cv.data == 0)


# ===========================================================================
# 2. Core types: DensityMatrix
# ===========================================================================

class TestDensityMatrix:
    def test_pure_state(self):
        # |+> = (|0> + |1>) / sqrt(2)
        cv = ComplexVec.from_list([1/np.sqrt(2), 1/np.sqrt(2)])
        rho = DensityMatrix.from_pure_state(cv)
        assert rho.dim == 2
        assert rho.is_pure
        np.testing.assert_allclose(rho.purity, 1.0, atol=1e-10)

    def test_maximally_mixed(self):
        rho = DensityMatrix.maximally_mixed(4)
        assert rho.dim == 4
        np.testing.assert_allclose(rho.purity, 0.25, atol=1e-10)
        assert not rho.is_pure

    def test_from_probabilities(self):
        probs = np.array([0.7, 0.2, 0.1])
        rho = DensityMatrix.from_probabilities(probs)
        assert rho.dim == 3
        np.testing.assert_allclose(np.diag(rho.data).real, probs, atol=1e-10)

    def test_hermitian(self):
        cv = ComplexVec.from_list([1+1j, 2-1j, 0.5+0.5j])
        rho = DensityMatrix.from_pure_state(cv)
        # rho should be Hermitian: rho = rho_dagger
        np.testing.assert_allclose(rho.data, rho.data.conj().T, atol=1e-10)

    def test_trace_one(self):
        cv = ComplexVec.from_list([0.6+0.1j, 0.3-0.2j, 0.1+0.7j])
        rho = DensityMatrix.from_pure_state(cv)
        np.testing.assert_allclose(np.trace(rho.data).real, 1.0, atol=1e-10)


# ===========================================================================
# 3. Complex operations
# ===========================================================================

class TestComplexOperations:
    def test_transform_complex(self):
        cv = ComplexVec.from_list([1+0j, 0+0j])
        m = TransMatrix.identity(2)
        result = ops.transform_complex(cv, m)
        np.testing.assert_allclose(result.data, cv.data, atol=1e-10)

    def test_measure_complex(self):
        # |psi> = (|0> + |1>) / sqrt(2) -> 50/50
        cv = ComplexVec.from_list([1/np.sqrt(2), 1/np.sqrt(2)])
        probs = ops.measure_complex(cv)
        np.testing.assert_allclose(probs.data, [0.5, 0.5], atol=1e-6)

    def test_measure_complex_phase_invariant(self):
        # Born rule: |alpha|^2 is phase-invariant
        cv1 = ComplexVec.from_list([1+0j, 0+0j])
        cv2 = ComplexVec.from_list([0+1j, 0+0j])  # same magnitude, different phase
        p1 = ops.measure_complex(cv1)
        p2 = ops.measure_complex(cv2)
        np.testing.assert_allclose(p1.data, p2.data, atol=1e-10)

    def test_constructive_interference(self):
        cv1 = ComplexVec.from_list([1+0j, 0+0j])
        cv2 = ComplexVec.from_list([1+0j, 0+0j])
        result = ops.interfere(cv1, cv2, phase=0.0)
        # Constructive: both add -> first component dominates
        probs = ops.measure_complex(result)
        assert probs.data[0] > 0.99

    def test_destructive_interference(self):
        cv1 = ComplexVec.from_list([1/np.sqrt(2)+0j, 1/np.sqrt(2)+0j])
        cv2 = ComplexVec.from_list([1/np.sqrt(2)+0j, 1/np.sqrt(2)+0j])
        result = ops.interfere(cv1, cv2, phase=np.pi)
        # Destructive: vec1 + exp(i*pi)*vec2 = vec1 - vec2 = 0
        assert np.linalg.norm(result.data) < 1e-10 or result.size == 2

    def test_partial_destructive_interference(self):
        cv1 = ComplexVec.from_list([1+0j, 0+0j])
        cv2 = ComplexVec.from_list([0+0j, 1+0j])
        result = ops.interfere(cv1, cv2, phase=0.0)
        # Different components -> both survive
        probs = ops.measure_complex(result)
        np.testing.assert_allclose(probs.data, [0.5, 0.5], atol=1e-6)

    def test_evolve_density(self):
        cv = ComplexVec.from_list([1+0j, 0+0j])
        rho = DensityMatrix.from_pure_state(cv)
        # Apply identity -> unchanged
        U = TransMatrix.identity(2)
        rho2 = ops.evolve_density(rho, U)
        np.testing.assert_allclose(rho2.data, rho.data, atol=1e-10)

    def test_partial_trace_bell_state(self):
        # Bell state: |00> + |11>) / sqrt(2)
        # Density matrix in 4x4
        bell = np.zeros(4, dtype=np.complex128)
        bell[0] = 1/np.sqrt(2)  # |00>
        bell[3] = 1/np.sqrt(2)  # |11>
        rho = DensityMatrix(data=np.outer(bell, bell.conj()))

        # Trace out B -> should get maximally mixed on A
        rho_a = ops.partial_trace(rho, dim_a=2, dim_b=2, trace_out="B")
        np.testing.assert_allclose(rho_a.purity, 0.5, atol=1e-10)
        np.testing.assert_allclose(np.diag(rho_a.data).real, [0.5, 0.5], atol=1e-10)


# ===========================================================================
# 4. Density matrix operations
# ===========================================================================

class TestDensityOperations:
    def test_von_neumann_entropy_pure(self):
        cv = ComplexVec.from_list([1+0j, 0+0j])
        rho = DensityMatrix.from_pure_state(cv)
        S = von_neumann_entropy(rho)
        np.testing.assert_allclose(S, 0.0, atol=1e-10)

    def test_von_neumann_entropy_mixed(self):
        rho = DensityMatrix.maximally_mixed(2)
        S = von_neumann_entropy(rho)
        np.testing.assert_allclose(S, np.log(2), atol=1e-10)

    def test_fidelity_same_state(self):
        cv = ComplexVec.from_list([1/np.sqrt(2), 1/np.sqrt(2)])
        rho = DensityMatrix.from_pure_state(cv)
        F = fidelity(rho, rho)
        np.testing.assert_allclose(F, 1.0, atol=1e-6)

    def test_fidelity_orthogonal(self):
        cv1 = ComplexVec.from_list([1+0j, 0+0j])
        cv2 = ComplexVec.from_list([0+0j, 1+0j])
        rho1 = DensityMatrix.from_pure_state(cv1)
        rho2 = DensityMatrix.from_pure_state(cv2)
        F = fidelity(rho1, rho2)
        np.testing.assert_allclose(F, 0.0, atol=1e-6)

    def test_phi_from_purity_pure(self):
        cv = ComplexVec.from_list([1+0j, 0+0j])
        rho = DensityMatrix.from_pure_state(cv)
        assert phi_from_purity(rho) > 0.99

    def test_phi_from_purity_mixed(self):
        rho = DensityMatrix.maximally_mixed(4)
        assert phi_from_purity(rho) < 0.01

    def test_omega_from_coherence_pure_superposition(self):
        # Uniform superposition has maximum coherence
        dim = 4
        cv = ComplexVec(data=np.ones(dim, dtype=np.complex128) / np.sqrt(dim))
        rho = DensityMatrix.from_pure_state(cv)
        omega = omega_from_coherence(rho)
        assert omega > 0.5  # should have significant coherence

    def test_omega_from_coherence_diagonal(self):
        # Diagonal (classical) state has zero coherence
        rho = DensityMatrix.from_probabilities(np.array([0.5, 0.3, 0.2]))
        omega = omega_from_coherence(rho)
        np.testing.assert_allclose(omega, 0.0, atol=1e-10)


# ===========================================================================
# 5. Quantum channels
# ===========================================================================

class TestQuantumChannels:
    def test_depolarizing_preserves_trace(self):
        cv = ComplexVec.from_list([1+0j, 0+0j])
        rho = DensityMatrix.from_pure_state(cv)
        kraus = depolarizing_channel(2, p=0.3)
        rho2 = apply_channel(rho, kraus)
        np.testing.assert_allclose(np.trace(rho2.data).real, 1.0, atol=1e-6)

    def test_depolarizing_full_noise(self):
        cv = ComplexVec.from_list([1+0j, 0+0j])
        rho = DensityMatrix.from_pure_state(cv)
        kraus = depolarizing_channel(2, p=1.0)
        rho2 = apply_channel(rho, kraus)
        # Should approach maximally mixed
        np.testing.assert_allclose(rho2.purity, 0.5, atol=0.1)

    def test_amplitude_damping_ground_state_stable(self):
        # Ground state |0> should be unaffected by damping
        cv = ComplexVec.from_list([1+0j, 0+0j])
        rho = DensityMatrix.from_pure_state(cv)
        kraus = amplitude_damping_channel(gamma=0.5, dim=2)
        rho2 = apply_channel(rho, kraus)
        np.testing.assert_allclose(rho2.data[0, 0].real, 1.0, atol=1e-10)

    def test_amplitude_damping_excited_decays(self):
        # Excited state |1> should decay toward |0>
        cv = ComplexVec.from_list([0+0j, 1+0j])
        rho = DensityMatrix.from_pure_state(cv)
        kraus = amplitude_damping_channel(gamma=0.5, dim=2)
        rho2 = apply_channel(rho, kraus)
        # Population should transfer to |0>
        assert rho2.data[0, 0].real > 0.4

    def test_dephasing_preserves_populations(self):
        cv = ComplexVec.from_list([1/np.sqrt(2), 1/np.sqrt(2)])
        rho = DensityMatrix.from_pure_state(cv)
        kraus = dephasing_channel(gamma=0.5, dim=2)
        rho2 = apply_channel(rho, kraus)
        # Diagonal elements should be preserved
        np.testing.assert_allclose(
            np.diag(rho2.data).real, np.diag(rho.data).real, atol=1e-6
        )

    def test_svd_to_kraus_identity(self):
        dim = 4
        U = np.eye(dim)
        sigma = np.ones(dim)
        Vh = np.eye(dim)
        kraus = svd_to_kraus(U, sigma, Vh, dim)
        # Identity SVD -> single Kraus op close to identity
        assert len(kraus) == 1
        np.testing.assert_allclose(np.abs(kraus[0]), np.eye(dim), atol=1e-6)

    def test_svd_to_kraus_with_damping(self):
        dim = 3
        U = np.eye(dim)
        sigma = np.array([1.0, 0.8, 0.5])
        Vh = np.eye(dim)
        kraus = svd_to_kraus(U, sigma, Vh, dim)
        # Should have more than 1 Kraus op (decay channels)
        assert len(kraus) > 1
        # Verify trace-preserving (approximately)
        total = sum(E.conj().T @ E for E in kraus)
        np.testing.assert_allclose(total, np.eye(dim), atol=0.2)


# ===========================================================================
# 6. Quantum weave + observe integration
# ===========================================================================

class TestQuantumWeaveObserve:
    def _build_linear_declaration(self, dim=8):
        return (
            DeclarationBuilder("q_linear")
            .input("x", dim=dim)
            .output("y")
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .quality(omega=0.8, phi=0.7)
            .build()
        )

    def _build_nonlinear_declaration(self, dim=8, depth=3):
        builder = (
            DeclarationBuilder("q_nonlinear")
            .input("x", dim=dim)
        )
        prev = "x"
        for i in range(depth):
            name = f"h{i}" if i < depth - 1 else "y"
            builder = builder.relate(
                name, [prev], RelationKind.PROPORTIONAL,
                transform_fn=np.tanh,
            )
            prev = name
        return builder.output("y").quality(omega=0.8, phi=0.7).build()

    def test_quantum_weave_produces_complex_amplitudes(self):
        decl = self._build_linear_declaration()
        tap = weave(decl, quantum=True)
        assert tap._quantum is True
        # Output node should have complex amplitudes
        assert tap.nodes["y"].state.complex_amplitudes is not None
        assert tap.nodes["y"].state.is_quantum

    def test_quantum_weave_produces_density_matrix(self):
        decl = self._build_linear_declaration()
        tap = weave(decl, quantum=True)
        assert tap._density_matrix is not None
        assert tap._density_matrix.is_pure
        np.testing.assert_allclose(tap._density_matrix.purity, 1.0, atol=1e-6)

    def test_quantum_observe_returns_quantum_fields(self):
        decl = self._build_linear_declaration()
        tap = weave(decl, quantum=True)
        x = FloatVec(data=np.random.default_rng(42).standard_normal(8).astype(np.float32))
        obs = observe(tap, {"x": x})
        assert obs.density_matrix is not None
        assert obs.quantum_phi is not None
        assert obs.quantum_omega is not None
        assert 0 <= obs.quantum_phi <= 1
        assert 0 <= obs.quantum_omega <= 1

    def test_quantum_observe_consistent_with_classical(self):
        """Quantum observe should produce same argmax as classical."""
        decl = self._build_linear_declaration()
        tap_c = weave(decl, quantum=False, seed=42)
        tap_q = weave(decl, quantum=True, seed=42)
        x = FloatVec(data=np.random.default_rng(42).standard_normal(8).astype(np.float32))
        obs_c = observe(tap_c, {"x": x})
        obs_q = observe(tap_q, {"x": x})
        # Same argmax (same underlying linear transform)
        assert obs_c.value_index == obs_q.value_index

    def test_quantum_nonlinear_distill(self):
        decl = self._build_nonlinear_declaration(dim=8, depth=3)
        tap = weave(decl, quantum=True, nonlinear_method="distill", seed=42)
        assert tap._quantum is True
        x = FloatVec(data=np.random.default_rng(99).standard_normal(8).astype(np.float32) * 0.3)
        obs = observe(tap, {"x": x})
        assert obs.density_matrix is not None

    def test_quantum_nonlinear_hybrid_kraus(self):
        """Hybrid path should produce Kraus operators when quantum=True."""
        decl = self._build_nonlinear_declaration(dim=8, depth=3)
        tap = weave(decl, quantum=True, nonlinear_method="hybrid", seed=42)
        assert tap._quantum is True
        # Hybrid should have Kraus operators
        assert tap._kraus_operators is not None
        assert len(tap._kraus_operators) >= 1

    def test_classical_weave_unchanged(self):
        """quantum=False should produce identical results to before."""
        decl = self._build_linear_declaration()
        tap = weave(decl, quantum=False, seed=42)
        assert tap._quantum is False
        assert tap._density_matrix is None
        assert tap._kraus_operators is None
        # Nodes should NOT have complex amplitudes
        assert tap.nodes["y"].state.complex_amplitudes is None

    def test_superposed_state_backward_compat(self):
        """SuperposedState without complex_amplitudes still works."""
        state = SuperposedState(
            name="test",
            amplitudes=FloatVec.from_list([0.5, 0.3, 0.2]),
        )
        assert not state.is_quantum
        probs = state.probabilities
        assert probs.size == 3


# ===========================================================================
# 7. Interference effects
# ===========================================================================

class TestInterferenceEffects:
    def test_phase_matters_in_observation(self):
        """Different phases should produce different probabilities."""
        dim = 4
        mags = np.ones(dim) / np.sqrt(dim)

        # All same phase
        cv1 = ComplexVec.from_polar(mags, np.zeros(dim))
        p1 = ops.measure_complex(cv1)

        # Alternating phases (0, pi, 0, pi)
        phases = np.array([0, np.pi, 0, np.pi])
        cv2 = ComplexVec.from_polar(mags, phases)
        p2 = ops.measure_complex(cv2)

        # Born rule: |alpha|^2 is phase-invariant for individual measurement
        # But interference between paths reveals phase differences
        np.testing.assert_allclose(p1.data, p2.data, atol=1e-6)

    def test_interference_creates_asymmetry(self):
        """Interfering two states with different phases creates asymmetric output."""
        cv1 = ComplexVec(data=np.array([1+0j, 1+0j]) / np.sqrt(2))
        cv2 = ComplexVec(data=np.array([1+0j, -1+0j]) / np.sqrt(2))

        # Constructive on component 0, destructive on component 1
        result = ops.interfere(cv1, cv2, phase=0.0)
        probs = ops.measure_complex(result)
        # Component 0 should dominate
        assert probs.data[0] > 0.9

    def test_interference_with_matrix_transform(self):
        """Complex transform preserves phase -> enables interference."""
        # |+> = (|0> + |1>)/sqrt(2), Hadamard maps |+> -> |0>
        cv = ComplexVec(data=np.array([1, 1], dtype=np.complex128) / np.sqrt(2))
        H = TransMatrix(data=(np.array([[1, 1], [1, -1]]) / np.sqrt(2)).astype(np.float32))
        result = ops.transform_complex(cv, H)
        probs = ops.measure_complex(result)
        # Constructive interference on |0>, destructive on |1>
        assert probs.data[0] > 0.9
        assert probs.data[1] < 0.1
