"""Composition rules for serial and parallel attractor systems.

Serial:
  lambda_total = lambda_A + lambda_B
  Omega_total = 1 / (1 + max(lambda_A + lambda_B, 0))
  D_total <= D_A + D_B
  Phi_total >= Phi_A * Phi_B

Parallel:
  lambda_total = max(lambda_A, lambda_B)
  Omega_total = min(Omega_A, Omega_B)
  D_total = max(D_A, D_B)
  Phi_total = min(Phi_A, Phi_B)
"""

from __future__ import annotations

from axol.quantum.lyapunov import omega_from_lyapunov
from axol.quantum.fractal import phi_from_fractal


def compose_serial(
    omega_a: float, phi_a: float, lambda_a: float, d_a: float,
    omega_b: float, phi_b: float, lambda_b: float, d_b: float,
    phase_space_dim: int = 8,
) -> tuple[float, float, float, float]:
    """Compose two stages in series.

    Returns:
        (omega_total, phi_total, lambda_total, d_total)
    """
    lambda_total = lambda_a + lambda_b
    omega_total = omega_from_lyapunov(lambda_total)
    d_total = d_a + d_b
    phi_total = phi_a * phi_b

    return (omega_total, phi_total, lambda_total, d_total)


def compose_parallel(
    omega_a: float, phi_a: float, lambda_a: float, d_a: float,
    omega_b: float, phi_b: float, lambda_b: float, d_b: float,
    phase_space_dim: int = 8,
) -> tuple[float, float, float, float]:
    """Compose two stages in parallel.

    Returns:
        (omega_total, phi_total, lambda_total, d_total)
    """
    lambda_total = max(lambda_a, lambda_b)
    omega_total = min(omega_a, omega_b)
    d_total = max(d_a, d_b)
    phi_total = min(phi_a, phi_b)

    return (omega_total, phi_total, lambda_total, d_total)


def can_reuse_after_observe(lyapunov: float) -> bool:
    """Whether a tapestry can be reused after observation.

    lambda < 0: convergent system, observation doesn't destroy structure -> reusable
    lambda >= 0: chaotic system, observation perturbs state -> must re-weave
    """
    return lyapunov < 0.0
