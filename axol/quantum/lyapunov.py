"""Lyapunov exponent estimation and Omega (cohesion) calculation.

Mathematical basis:
  lambda = lim_{k->inf} (1/k) * ln(||delta x_k|| / ||delta x_0||)
  Omega = 1 / (1 + max(lambda, 0))
"""

from __future__ import annotations

import numpy as np

from axol.core.types import FloatVec, TransMatrix


def estimate_lyapunov(trajectory_matrix: TransMatrix, steps: int = 100) -> float:
    """Estimate the maximum Lyapunov exponent from a trajectory matrix.

    Uses the Benettin QR decomposition method:
    1. Start with a unit perturbation vector
    2. Multiply by the trajectory matrix at each step
    3. Track the logarithmic growth rate via QR decomposition

    Args:
        trajectory_matrix: The n x n matrix defining the system dynamics.
        steps: Number of iterations for the estimate.

    Returns:
        Estimated maximum Lyapunov exponent (lambda).
    """
    M = trajectory_matrix.data.astype(np.float64)
    n = M.shape[0]

    if n == 0:
        return 0.0

    # Start with identity-like perturbation (first column)
    v = np.random.default_rng(42).standard_normal(n)
    v = v / np.linalg.norm(v)

    lyap_sum = 0.0
    valid_steps = 0

    for _ in range(steps):
        v_new = M @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-15:
            # System collapses — strongly convergent
            lyap_sum += np.log(1e-15)
            valid_steps += 1
            break
        lyap_sum += np.log(norm)
        valid_steps += 1
        v = v_new / norm

    if valid_steps == 0:
        return 0.0
    return float(lyap_sum / valid_steps)


def lyapunov_spectrum(trajectory_matrix: TransMatrix, dim: int | None = None, steps: int = 100) -> list[float]:
    """Compute the full Lyapunov spectrum using QR decomposition (Benettin method).

    Args:
        trajectory_matrix: The n x n matrix defining the system dynamics.
        dim: Number of exponents to compute (default: n).
        steps: Number of iterations.

    Returns:
        List of Lyapunov exponents in descending order.
    """
    M = trajectory_matrix.data.astype(np.float64)
    n = M.shape[0]
    if dim is None:
        dim = n
    dim = min(dim, n)

    if n == 0:
        return []

    # Initialise orthonormal frame
    Q = np.eye(n, dim, dtype=np.float64)
    lyap_sums = np.zeros(dim, dtype=np.float64)
    valid_steps = 0

    for _ in range(steps):
        # Propagate the frame
        Z = M @ Q
        # QR decomposition to re-orthonormalise
        Q_new, R = np.linalg.qr(Z)
        # Accumulate log of diagonal elements
        diag = np.abs(np.diag(R[:dim, :dim]))
        diag = np.maximum(diag, 1e-15)
        lyap_sums += np.log(diag)
        valid_steps += 1
        Q = Q_new[:, :dim]

    if valid_steps == 0:
        return [0.0] * dim

    spectrum = (lyap_sums / valid_steps).tolist()
    spectrum.sort(reverse=True)
    return spectrum


def omega_from_lyapunov(lyapunov: float) -> float:
    """Compute Omega (cohesion) from the maximum Lyapunov exponent.

    Omega = 1 / (1 + max(lambda, 0))

    - lambda << 0  =>  Omega -> 1.0  (strong convergence)
    - lambda = 0   =>  Omega = 1.0   (marginally stable)
    - lambda > 0   =>  Omega < 1.0   (chaotic)
    """
    return 1.0 / (1.0 + max(lyapunov, 0.0))


def omega_from_observations(observations: list[FloatVec]) -> float:
    """Compute empirical Omega from multiple observations.

    Measures the stability of the argmax across observations:
    Omega = (count of modal argmax) / total_observations

    Args:
        observations: List of observation probability vectors.

    Returns:
        Empirical Omega in [0, 1].
    """
    if not observations:
        return 0.0

    indices = [int(np.argmax(obs.data)) for obs in observations]
    if not indices:
        return 0.0

    # Find mode
    unique, counts = np.unique(indices, return_counts=True)
    max_count = int(np.max(counts))
    return max_count / len(indices)
