"""Unitary and hybrid matrix utilities for nonlinear pipeline composition.

SVD decomposes any matrix A = U @ Sigma @ Vh:
  - U @ Vh  = unitary part (direction, quantum gate)
  - Sigma   = scale part   (magnitude, decoherence)

Three approaches to nonlinear composition:
  1. Pure unitary:  keep U@Vh, discard Sigma  -> fast, direction-only
  2. Hybrid:        keep U@Vh AND Sigma       -> fast + accurate
  3. Koopman:       lift to high-dim space     -> accurate, slow

Hybrid maps naturally to open quantum systems:
  unitary part = isolated quantum evolution
  scale part   = environment interaction (decoherence)

Core functions:
  - nearest_unitary(A)            : SVD polar decomposition -> U @ Vh
  - estimate_unitary_step(...)    : lstsq -> nearest_unitary (pure unitary)
  - compose_unitary_chain(...)    : U1 @ U2 @ ... @ Un + reorthogonalise
  - estimate_hybrid_step(...)     : lstsq -> SVD -> (rotation, scales)
  - compose_hybrid_chain(...)     : compose raw A's -> SVD -> (rotation, scales)
  - reorthogonalize(U)            : correct floating-point drift
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from axol.core.types import TransMatrix


# ---------------------------------------------------------------------------
# Shared: least-squares estimation of dim x dim transition matrix
# ---------------------------------------------------------------------------

def _estimate_raw_matrix(
    step_fn: Callable[[np.ndarray], np.ndarray],
    dim: int,
    n_samples: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Estimate dim x dim transition matrix via least-squares.

    Shared by both pure-unitary and hybrid paths.
    Returns raw float64 matrix (not projected to unitary).
    """
    rng = np.random.default_rng(seed)
    n_samples = max(n_samples, dim + 10)

    X = rng.standard_normal((n_samples, dim)) * 0.5
    Y = np.empty_like(X)
    for i in range(n_samples):
        yi = step_fn(X[i])
        yi = np.nan_to_num(yi, nan=0.0, posinf=10.0, neginf=-10.0)
        Y[i] = yi

    A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return A.astype(np.float64)


# ---------------------------------------------------------------------------
# Nearest unitary (polar factor)
# ---------------------------------------------------------------------------

def nearest_unitary(A: np.ndarray) -> np.ndarray:
    """Return the closest unitary matrix to *A* in Frobenius norm.

    Uses the SVD polar decomposition: A = U @ S @ Vh  ->  nearest unitary = U @ Vh.
    """
    U, _S, Vh = np.linalg.svd(A, full_matrices=False)
    return U @ Vh


# ---------------------------------------------------------------------------
# Re-orthogonalise (= nearest_unitary, explicit alias)
# ---------------------------------------------------------------------------

def reorthogonalize(U: np.ndarray) -> np.ndarray:
    """Correct floating-point drift so that U stays unitary."""
    return nearest_unitary(U)


# ---------------------------------------------------------------------------
# Pure unitary: estimate + compose (direction only, Sigma discarded)
# ---------------------------------------------------------------------------

def estimate_unitary_step(
    step_fn: Callable[[np.ndarray], np.ndarray],
    dim: int,
    n_samples: int = 500,
    seed: int = 42,
) -> TransMatrix:
    """Estimate a dim x dim unitary matrix that best approximates *step_fn*.

    lstsq -> project onto nearest unitary (discards singular values).
    Returns TransMatrix of shape (dim, dim).
    """
    A = _estimate_raw_matrix(step_fn, dim, n_samples, seed)
    U = nearest_unitary(A)
    return TransMatrix(data=U.astype(np.float32))


def compose_unitary_chain(matrices: list[TransMatrix]) -> TransMatrix:
    """Compose U1 @ U2 @ ... @ Un and re-orthogonalise the result."""
    if not matrices:
        raise ValueError("Cannot compose empty chain")

    composed = matrices[0].data.astype(np.float64)
    for m in matrices[1:]:
        composed = composed @ m.data.astype(np.float64)

    composed = reorthogonalize(composed)
    return TransMatrix(data=composed.astype(np.float32))


# ---------------------------------------------------------------------------
# Hybrid: estimate + compose (unitary direction + singular-value scales)
# ---------------------------------------------------------------------------

def estimate_hybrid_step(
    step_fn: Callable[[np.ndarray], np.ndarray],
    dim: int,
    n_samples: int = 500,
    seed: int = 42,
) -> TransMatrix:
    """Estimate the raw dim x dim transition matrix (NOT projected to unitary).

    Returns a TransMatrix that preserves both direction and magnitude.
    """
    A = _estimate_raw_matrix(step_fn, dim, n_samples, seed)
    A = np.nan_to_num(A, nan=0.0, posinf=100.0, neginf=-100.0)
    return TransMatrix(data=A.astype(np.float32))


def compose_hybrid_chain(
    matrices: list[TransMatrix],
) -> tuple[TransMatrix, TransMatrix, np.ndarray]:
    """Compose raw A matrices then SVD-decompose the result.

    A_total = A1 @ A2 @ ... @ An
    SVD:  A_total = U @ diag(sigma) @ Vh
    rotation = U @ Vh   (dim x dim, unitary -- quantum gate)
    scales   = sigma    (dim, non-negative -- decoherence weights)

    Returns (composed_matrix, rotation, scales).
      - composed_matrix: the raw A_total for direct observation
      - rotation: unitary part (direction, quantum gate)
      - scales: singular values (magnitude, decoherence)
    """
    if not matrices:
        raise ValueError("Cannot compose empty chain")

    composed = matrices[0].data.astype(np.float64)
    for m in matrices[1:]:
        composed = composed @ m.data.astype(np.float64)

    composed = np.nan_to_num(composed, nan=0.0, posinf=100.0, neginf=-100.0)

    U, sigma, Vh = np.linalg.svd(composed, full_matrices=False)
    rotation = U @ Vh

    return (
        TransMatrix(data=composed.astype(np.float32)),
        TransMatrix(data=rotation.astype(np.float32)),
        sigma.astype(np.float32),
    )
