"""Quantum density matrix operations — bridges chaos theory and quantum formalism.

Physical interpretation:
  - DensityMatrix rho: quantum state (pure |psi><psi| or mixed)
  - Purity tr(rho^2): 1.0 = pure, 1/dim = maximally mixed
  - Von Neumann entropy: -tr(rho log rho), quantum Shannon entropy
  - Quantum channels: Kraus operators E_k, epsilon(rho) = sum E_k rho E_k_dagger
  - SVD -> Kraus: connects Hybrid SVD to quantum channel theory

Quality metrics from quantum state:
  - Phi from purity:     pure state -> 1.0, mixed -> 0.0
  - Omega from coherence: off-diagonal elements measure quantum correlations
"""

from __future__ import annotations

import numpy as np

from axol.core.types import ComplexVec, DensityMatrix, FloatVec, TransMatrix


# ---------------------------------------------------------------------------
# Entropy and information measures
# ---------------------------------------------------------------------------

def von_neumann_entropy(rho: DensityMatrix) -> float:
    """Von Neumann entropy: S(rho) = -tr(rho log rho).

    Returns 0 for pure states, log(dim) for maximally mixed.
    """
    eigenvalues = np.linalg.eigvalsh(rho.data)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def fidelity(rho1: DensityMatrix, rho2: DensityMatrix) -> float:
    """Quantum fidelity: F(rho1, rho2) = (tr sqrt(sqrt(rho1) rho2 sqrt(rho1)))^2.

    F = 1 for identical states, F = 0 for orthogonal states.
    """
    sqrt_rho1 = _matrix_sqrt(rho1.data)
    product = sqrt_rho1 @ rho2.data @ sqrt_rho1
    eigenvalues = np.linalg.eigvalsh(product)
    eigenvalues = np.real(np.maximum(eigenvalues, 0))
    return float(np.sum(np.sqrt(eigenvalues)) ** 2)


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Matrix square root via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.real(np.maximum(eigenvalues, 0))
    sqrt_eig = np.sqrt(eigenvalues)
    return eigenvectors @ np.diag(sqrt_eig) @ eigenvectors.conj().T


# ---------------------------------------------------------------------------
# Quantum channels (Kraus operator formalism)
# ---------------------------------------------------------------------------

def apply_channel(
    rho: DensityMatrix,
    kraus_ops: list[np.ndarray],
) -> DensityMatrix:
    """Apply quantum channel: epsilon(rho) = sum_k E_k rho E_k_dagger.

    Kraus operators must satisfy: sum_k E_k_dagger E_k = I (trace-preserving).
    """
    dim = rho.dim
    result = np.zeros((dim, dim), dtype=np.complex128)
    for E in kraus_ops:
        result += E @ rho.data @ E.conj().T
    result = (result + result.conj().T) / 2
    return DensityMatrix(data=result)


def depolarizing_channel(dim: int, p: float) -> list[np.ndarray]:
    """Depolarizing channel: epsilon(rho) = (1-p) rho + (p/dim) I.

    Args:
        dim: Hilbert space dimension
        p: depolarizing probability (0 = no noise, 1 = fully depolarized)

    Returns:
        List of Kraus operators [E_0, E_ij, ...].
    """
    E0 = np.sqrt(1 - p) * np.eye(dim, dtype=np.complex128)
    kraus = [E0]

    scale = np.sqrt(p / dim)
    for i in range(dim):
        for j in range(dim):
            E = np.zeros((dim, dim), dtype=np.complex128)
            E[i, j] = scale
            kraus.append(E)

    return kraus


def amplitude_damping_channel(gamma: float, dim: int = 2) -> list[np.ndarray]:
    """Amplitude damping channel (energy dissipation).

    gamma: damping rate (0 = no damping, 1 = full decay to ground state)
    dim: Hilbert space dimension

    For qubit (dim=2):
      E_0 = [[1, 0], [0, sqrt(1-gamma)]]
      E_1 = [[0, sqrt(gamma)], [0, 0]]
    """
    E0 = np.eye(dim, dtype=np.complex128)
    for k in range(1, dim):
        E0[k, k] = np.sqrt(1 - gamma)
    kraus = [E0]

    for k in range(1, dim):
        Ek = np.zeros((dim, dim), dtype=np.complex128)
        Ek[0, k] = np.sqrt(gamma)
        kraus.append(Ek)

    return kraus


def dephasing_channel(gamma: float, dim: int = 2) -> list[np.ndarray]:
    """Dephasing channel — destroys off-diagonal coherence.

    Preserves populations (diagonal), damps coherences by (1-gamma).

    Returns Kraus operators.
    """
    E0 = np.sqrt(1 - gamma) * np.eye(dim, dtype=np.complex128)
    kraus = [E0]

    scale = np.sqrt(gamma)
    for k in range(dim):
        Ek = np.zeros((dim, dim), dtype=np.complex128)
        Ek[k, k] = scale
        kraus.append(Ek)

    return kraus


# ---------------------------------------------------------------------------
# SVD -> Kraus: bridge Hybrid composition to quantum channels
# ---------------------------------------------------------------------------

def svd_to_kraus(
    U: np.ndarray,
    sigma: np.ndarray,
    Vh: np.ndarray,
    dim: int,
) -> list[np.ndarray]:
    """Convert SVD decomposition to Kraus operators.

    Maps Hybrid's SVD (A = U Sigma Vh) to quantum channel:
      - U @ Vh = coherent rotation (quantum gate)
      - sigma = decoherence strengths per mode

    sigma_i = 1: no decoherence on mode i
    sigma_i < 1: amplitude damping on mode i
    sigma_i > 1: amplification (clamped to 1 for physicality)
    """
    sigma_clamped = np.clip(sigma[:dim], 0, 1).astype(np.float64)

    rotation = (U[:dim, :dim] @ Vh[:dim, :dim]).astype(np.complex128)

    # E_0: damped rotation
    E0 = rotation @ np.diag(sigma_clamped.astype(np.complex128))
    kraus = [E0]

    # Decay operators for each damped mode
    for i in range(min(dim, len(sigma_clamped))):
        if sigma_clamped[i] < 1.0 - 1e-10:
            gamma_i = 1.0 - sigma_clamped[i] ** 2
            Ek = np.zeros((dim, dim), dtype=np.complex128)
            Ek[i, i] = np.sqrt(gamma_i)
            Ek = rotation @ Ek
            kraus.append(Ek)

    return kraus


# ---------------------------------------------------------------------------
# Quality metrics from quantum state
# ---------------------------------------------------------------------------

def phi_from_purity(rho: DensityMatrix) -> float:
    """Derive Phi (clarity) from quantum purity.

    Maps purity tr(rho^2) in [1/dim, 1] to Phi in [0, 1].
    Pure state -> 1.0, maximally mixed -> 0.0.
    """
    p = rho.purity
    dim = rho.dim
    min_purity = 1.0 / dim
    if p <= min_purity:
        return 0.0
    return min((p - min_purity) / (1.0 - min_purity), 1.0)


def omega_from_coherence(rho: DensityMatrix) -> float:
    """Derive Omega (cohesion) from off-diagonal coherence.

    Coherence = sum |rho_ij| for i != j, normalized by max possible.
    High coherence -> stable quantum correlations -> high Omega.
    Decoherence destroys off-diagonal -> low Omega.
    """
    dim = rho.dim
    off_diag = rho.data.copy()
    np.fill_diagonal(off_diag, 0)
    coherence = float(np.sum(np.abs(off_diag)))

    # Max coherence: uniform pure state |+> = (1/sqrt(dim)) sum|i>
    # All off-diagonal = 1/dim, so total = dim*(dim-1)/dim = dim - 1
    max_coherence = max(dim - 1, 1)

    return min(coherence / max_coherence, 1.0)
