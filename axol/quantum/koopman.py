"""Koopman operator utilities for nonlinear pipeline composition.

Lifts nonlinear dynamics into a polynomial observable space where they
become linear, enabling depth-independent composed observation via EDMD
(Extended Dynamic Mode Decomposition).

Core mapping:
  - lift(x)   : x -> Ψ(x) = [1, x₁, ..., xₐ, x₁², x₁x₂, ..., xₐ²]
  - unlift(y) : Ψ(y) -> y  (extract linear terms)
  - EDMD      : estimate Koopman matrix K such that Ψ(f(x)) ≈ Ψ(x) @ K
  - compose   : K_chain = K₁ @ K₂ @ ... @ Kₙ
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb
from typing import Callable

import numpy as np

from axol.core.types import TransMatrix


# ---------------------------------------------------------------------------
# Lifted dimension
# ---------------------------------------------------------------------------

def lifted_dim(dim: int, degree: int = 2) -> int:
    """Dimension of the polynomial observable space.

    Sum of C(dim+k-1, k) for k = 0 .. degree.
    degree=2, dim=4 → 15;  degree=3, dim=4 → 35.
    """
    total = 0
    for k in range(degree + 1):
        total += comb(dim + k - 1, k)
    return total


# ---------------------------------------------------------------------------
# Lift / Unlift
# ---------------------------------------------------------------------------

def lift(x: np.ndarray, degree: int = 2) -> np.ndarray:
    """Lift vector(s) into polynomial observable space.

    Args:
        x: shape (dim,) or (n, dim)
        degree: polynomial degree (≥1)

    Returns:
        Lifted array of shape (lifted_dim,) or (n, lifted_dim).
    """
    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)

    x = x.astype(np.float64)
    n, dim = x.shape
    ld = lifted_dim(dim, degree)
    out = np.empty((n, ld), dtype=np.float64)

    if degree == 2:
        # Optimised fast path for the common case
        out[:, 0] = 1.0
        out[:, 1:dim + 1] = x
        idx = dim + 1
        for i in range(dim):
            for j in range(i, dim):
                out[:, idx] = x[:, i] * x[:, j]
                idx += 1
    else:
        # General path for any degree
        idx = 0
        for k in range(degree + 1):
            for combo in combinations_with_replacement(range(dim), k):
                if k == 0:
                    out[:, idx] = 1.0
                else:
                    col = np.ones(n, dtype=np.float64)
                    for c in combo:
                        col *= x[:, c]
                    out[:, idx] = col
                idx += 1

    if single:
        return out[0]
    return out


def unlift(y_lifted: np.ndarray, dim: int, degree: int = 2) -> np.ndarray:
    """Extract original-space vector from lifted representation.

    Takes the linear terms (indices 1:dim+1) from the lifted vector.

    Args:
        y_lifted: shape (lifted_dim,) or (n, lifted_dim)
        dim: original space dimension
        degree: polynomial degree

    Returns:
        Array of shape (dim,) or (n, dim).
    """
    single = y_lifted.ndim == 1
    if single:
        y_lifted = y_lifted.reshape(1, -1)

    result = y_lifted[:, 1:dim + 1].astype(np.float64)

    if single:
        return result[0]
    return result


# ---------------------------------------------------------------------------
# EDMD estimation
# ---------------------------------------------------------------------------

def estimate_koopman_matrix(
    step_fn: Callable[[np.ndarray], np.ndarray],
    dim: int,
    degree: int = 2,
    n_samples: int = 500,
    seed: int = 42,
) -> TransMatrix:
    """Estimate the Koopman matrix via Extended Dynamic Mode Decomposition.

    1. Generate n_samples random input vectors
    2. Apply step_fn to each
    3. Lift both inputs and outputs
    4. Solve least-squares: Ψ_X @ K ≈ Ψ_Y

    Args:
        step_fn: function mapping (dim,) -> (dim,)
        dim: state space dimension
        degree: polynomial degree for lifting
        n_samples: number of samples for EDMD
        seed: random seed

    Returns:
        TransMatrix of shape (lifted_dim, lifted_dim).
    """
    rng = np.random.default_rng(seed)
    ld = lifted_dim(dim, degree)

    # Ensure enough samples for the lifted dimension
    n_samples = max(n_samples, ld + 10)

    # Generate samples from a moderate distribution
    X = rng.standard_normal((n_samples, dim)) * 0.5

    # Apply step function to each sample
    Y = np.empty_like(X)
    for i in range(n_samples):
        yi = step_fn(X[i])
        # Safety: clip NaN/Inf
        yi = np.nan_to_num(yi, nan=0.0, posinf=10.0, neginf=-10.0)
        Y[i] = yi

    # Lift inputs and outputs
    Psi_X = lift(X, degree)  # (n_samples, ld)
    Psi_Y = lift(Y, degree)  # (n_samples, ld)

    # Least-squares solve: Psi_X @ K ≈ Psi_Y
    # K = pinv(Psi_X) @ Psi_Y
    K, _, _, _ = np.linalg.lstsq(Psi_X, Psi_Y, rcond=None)

    # Safety: clip extreme values
    K = np.nan_to_num(K, nan=0.0, posinf=100.0, neginf=-100.0)
    K = np.clip(K, -100.0, 100.0)

    return TransMatrix(data=K.astype(np.float32))


# ---------------------------------------------------------------------------
# Koopman chain composition
# ---------------------------------------------------------------------------

def compose_koopman_chain(matrices: list[TransMatrix]) -> TransMatrix:
    """Compose a chain of Koopman matrices: K_chain = K₁ @ K₂ @ ... @ Kₙ.

    Uses float64 for intermediate precision.
    """
    if not matrices:
        raise ValueError("Cannot compose empty chain")

    composed = matrices[0].data.astype(np.float64)
    for m in matrices[1:]:
        composed = composed @ m.data.astype(np.float64)

    # Safety: clip extreme values
    composed = np.nan_to_num(composed, nan=0.0, posinf=100.0, neginf=-100.0)
    composed = np.clip(composed, -100.0, 100.0)

    return TransMatrix(data=composed.astype(np.float32))
