"""Koopman operator utilities for nonlinear pipeline composition.

Lifts nonlinear dynamics into a polynomial observable space where they
become linear, enabling depth-independent composed observation via EDMD
(Extended Dynamic Mode Decomposition).

Core mapping:
  - lift(x)   : x -> Ψ(x) = [1, x₁, ..., xₐ, x₁², x₁x₂, ..., xₐ²]
  - unlift(y) : Ψ(y) -> y  (extract linear terms)
  - EDMD      : estimate Koopman matrix K such that Ψ(f(x)) ≈ Ψ(x) @ K
  - compose   : K_chain = K₁ @ K₂ @ ... @ Kₙ

Basis options:
  - "poly"      : polynomial observables only (default)
  - "augmented" : polynomial + indicator functions 1_{x_i>0} + ReLU cross
                  terms x_i*1_{x_i>0}.  Exactly captures PWA functions
                  (ReLU, abs, step) per Mauroy & Goncalves (2020).
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

def lifted_dim(dim: int, degree: int = 2, basis: str = "poly") -> int:
    """Dimension of the observable space.

    Sum of C(dim+k-1, k) for k = 0 .. degree.
    degree=2, dim=4 → 15;  degree=3, dim=4 → 35.

    With basis="augmented", adds 2*dim indicator/cross terms:
      dim indicators 1_{x_i>0} + dim ReLU cross terms x_i*1_{x_i>0}.
    """
    poly = 0
    for k in range(degree + 1):
        poly += comb(dim + k - 1, k)
    if basis == "augmented":
        return poly + 2 * dim
    return poly


# ---------------------------------------------------------------------------
# Lift / Unlift
# ---------------------------------------------------------------------------

def lift(x: np.ndarray, degree: int = 2, basis: str = "poly") -> np.ndarray:
    """Lift vector(s) into observable space.

    Args:
        x: shape (dim,) or (n, dim)
        degree: polynomial degree (>=1)
        basis: "poly" for polynomial only, "augmented" to add indicator/cross terms

    Returns:
        Lifted array of shape (lifted_dim,) or (n, lifted_dim).
    """
    single = x.ndim == 1
    if single:
        x = x.reshape(1, -1)

    x = x.astype(np.float64)
    n, dim = x.shape
    ld = lifted_dim(dim, degree, basis)
    out = np.empty((n, ld), dtype=np.float64)

    # --- polynomial terms ---
    poly_ld = lifted_dim(dim, degree, "poly")

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

    # --- augmented indicator / cross terms ---
    if basis == "augmented":
        indicators = (x > 0).astype(np.float64)          # 1_{x_i > 0}
        relu_cross = x * indicators                       # x_i * 1_{x_i > 0}
        out[:, poly_ld:poly_ld + dim] = indicators
        out[:, poly_ld + dim:poly_ld + 2 * dim] = relu_cross

    if single:
        return out[0]
    return out


def unlift(y_lifted: np.ndarray, dim: int, degree: int = 2, basis: str = "poly") -> np.ndarray:
    """Extract original-space vector from lifted representation.

    Takes the linear terms (indices 1:dim+1) from the lifted vector.
    The basis parameter is accepted for API symmetry but does not change
    the extraction (linear terms are always at the same position).

    Args:
        y_lifted: shape (lifted_dim,) or (n, lifted_dim)
        dim: original space dimension
        degree: polynomial degree
        basis: "poly" or "augmented" (accepted for symmetry)

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
    basis: str = "poly",
) -> TransMatrix:
    """Estimate the Koopman matrix via Extended Dynamic Mode Decomposition.

    1. Generate n_samples random input vectors
    2. Apply step_fn to each
    3. Lift both inputs and outputs
    4. Solve least-squares: Psi_X @ K ~ Psi_Y

    Args:
        step_fn: function mapping (dim,) -> (dim,)
        dim: state space dimension
        degree: polynomial degree for lifting
        n_samples: number of samples for EDMD
        seed: random seed
        basis: "poly" or "augmented"

    Returns:
        TransMatrix of shape (lifted_dim, lifted_dim).
    """
    rng = np.random.default_rng(seed)
    ld = lifted_dim(dim, degree, basis)

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
    Psi_X = lift(X, degree, basis)  # (n_samples, ld)
    Psi_Y = lift(Y, degree, basis)  # (n_samples, ld)

    # Least-squares solve: Psi_X @ K ~ Psi_Y
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
