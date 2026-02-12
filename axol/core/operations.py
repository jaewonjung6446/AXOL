"""Axol 9 primitive operations — pure functions over vector types."""

from __future__ import annotations

import numpy as np

from axol.core.types import (
    _VecBase,
    FloatVec,
    GateVec,
    TransMatrix,
    StateBundle,
    ComplexVec,
    DensityMatrix,
)


# ---------------------------------------------------------------------------
# transform: state @ matrix  →  new state
# ---------------------------------------------------------------------------

def transform(vec: _VecBase, matrix: TransMatrix) -> FloatVec:
    """Linear transformation: vec @ matrix → FloatVec[N]."""
    v = vec.data.astype(np.float32)
    if v.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"Dimension mismatch: vec({v.shape[0]}) vs matrix({matrix.shape[0]}×{matrix.shape[1]})"
        )
    result = v @ matrix.data
    return FloatVec(data=result)


# ---------------------------------------------------------------------------
# gate: element-wise conditional pass/block
# ---------------------------------------------------------------------------

def gate(vec: _VecBase, g: GateVec) -> FloatVec:
    """Element-wise gating: vec * gate → FloatVec[N]."""
    v = vec.data.astype(np.float32)
    if v.shape[0] != g.size:
        raise ValueError(
            f"Dimension mismatch: vec({v.shape[0]}) vs gate({g.size})"
        )
    result = v * g.data
    return FloatVec(data=result)


# ---------------------------------------------------------------------------
# merge: weighted sum of multiple vectors
# ---------------------------------------------------------------------------

def merge(vectors: list[_VecBase], weights: FloatVec) -> FloatVec:
    """Weighted combination: sum(v_i * w_i) → FloatVec[N]."""
    if len(vectors) != weights.size:
        raise ValueError(
            f"Count mismatch: {len(vectors)} vectors vs {weights.size} weights"
        )
    if len(vectors) == 0:
        raise ValueError("merge requires at least one vector")
    n = vectors[0].data.shape[0]
    result = np.zeros(n, dtype=np.float32)
    for vec, w in zip(vectors, weights.data):
        v = vec.data.astype(np.float32)
        if v.shape[0] != n:
            raise ValueError("All vectors in merge must have the same dimension")
        result += v * float(w)
    return FloatVec(data=result)


# ---------------------------------------------------------------------------
# distance: similarity / dissimilarity between two vectors
# ---------------------------------------------------------------------------

def distance(
    a: _VecBase,
    b: _VecBase,
    metric: str = "euclidean",
) -> float:
    """Compute distance between two vectors.

    Metrics: "euclidean" (L2 norm of diff), "cosine" (1 − cos_sim), "dot".
    """
    va = a.data.astype(np.float32)
    vb = b.data.astype(np.float32)
    if va.shape != vb.shape:
        raise ValueError(
            f"Shape mismatch: {va.shape} vs {vb.shape}"
        )

    if metric == "euclidean":
        return float(np.linalg.norm(va - vb))
    elif metric == "cosine":
        dot = float(np.dot(va, vb))
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 1.0  # maximum distance for zero vectors
        return 1.0 - dot / (na * nb)
    elif metric == "dot":
        return float(np.dot(va, vb))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# route: argmax(state @ router) → int index
# ---------------------------------------------------------------------------

def route(vec: _VecBase, router: TransMatrix) -> int:
    """Route to the index with highest activation: argmax(vec @ router)."""
    v = vec.data.astype(np.float32)
    if v.shape[0] != router.shape[0]:
        raise ValueError(
            f"Dimension mismatch: vec({v.shape[0]}) vs router({router.shape[0]}×{router.shape[1]})"
        )
    scores = v @ router.data
    return int(np.argmax(scores))


# ---------------------------------------------------------------------------
# step: threshold → binary gate vector (Plaintext only)
# ---------------------------------------------------------------------------

def step(vec: _VecBase, threshold: float = 0.0) -> GateVec:
    """Element-wise step function: 1.0 where vec >= threshold, else 0.0."""
    v = vec.data.astype(np.float32)
    result = np.where(v >= threshold, 1.0, 0.0).astype(np.float32)
    return GateVec(data=result)


# ---------------------------------------------------------------------------
# branch: conditional select via gate vector (Plaintext only)
# ---------------------------------------------------------------------------

def branch(gate_vec: GateVec, then_vec: _VecBase, else_vec: _VecBase) -> FloatVec:
    """Element-wise branch: where gate==1 pick then_vec, else else_vec."""
    g = gate_vec.data.astype(np.float32)
    t = then_vec.data.astype(np.float32)
    e = else_vec.data.astype(np.float32)
    if g.shape[0] != t.shape[0] or g.shape[0] != e.shape[0]:
        raise ValueError(
            f"Dimension mismatch: gate({g.shape[0]}) vs then({t.shape[0]}) vs else({e.shape[0]})"
        )
    result = np.where(g == 1.0, t, e).astype(np.float32)
    return FloatVec(data=result)


# ---------------------------------------------------------------------------
# clamp: clip values to [min, max] (Plaintext only)
# ---------------------------------------------------------------------------

def clamp(vec: _VecBase, min_val: float = -np.inf, max_val: float = np.inf) -> FloatVec:
    """Element-wise clamp: clip values to [min_val, max_val]."""
    v = vec.data.astype(np.float32)
    result = np.clip(v, min_val, max_val).astype(np.float32)
    return FloatVec(data=result)


# ---------------------------------------------------------------------------
# map_fn: apply named element-wise function (Plaintext only)
# ---------------------------------------------------------------------------

_MAP_FUNCTIONS: dict[str, callable] = {
    "relu": lambda x: np.maximum(x, 0.0),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "abs": lambda x: np.abs(x),
    "neg": lambda x: -x,
    "square": lambda x: x * x,
    "sqrt": lambda x: np.sqrt(np.maximum(x, 0.0)),
    "abs_sq": lambda x: x * x,
}


def map_fn(vec: _VecBase, fn_name: str) -> FloatVec:
    """Apply a named element-wise function to a vector."""
    if fn_name not in _MAP_FUNCTIONS:
        raise ValueError(f"Unknown map function: '{fn_name}'. Available: {sorted(_MAP_FUNCTIONS)}")
    v = vec.data.astype(np.float32)
    result = _MAP_FUNCTIONS[fn_name](v).astype(np.float32)
    return FloatVec(data=result)


# ---------------------------------------------------------------------------
# Quantum operations
# ---------------------------------------------------------------------------

def measure(vec: _VecBase) -> FloatVec:
    """Born rule measurement: |alpha_i|^2 normalized to probabilities."""
    v = vec.data.astype(np.float32)
    probs = v * v  # |alpha|^2 — negative amplitudes also become positive
    total = np.sum(probs)
    if total > 0:
        probs = probs / total
    return FloatVec(data=probs)


def hadamard_matrix(n: int) -> TransMatrix:
    """N-dim Hadamard matrix (Walsh-Hadamard). N must be power of 2.

    H = 1/sqrt(N) * H_N where H_N is constructed via Kronecker product.
    All elements are real. Negative entries enable interference.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"hadamard_matrix requires power of 2, got {n}")
    h = np.array([[1.0]], dtype=np.float32)
    h2 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    k = int(np.log2(n))
    for _ in range(k):
        h = np.kron(h, h2)
    h = h / np.sqrt(n)
    return TransMatrix(data=h.astype(np.float32))


def oracle_matrix(marked: list[int], n: int) -> TransMatrix:
    """Oracle: flip the sign of marked indices (diagonal matrix).

    O_ii = -1 if i in marked, else 1. Pure real.
    """
    diag = np.ones(n, dtype=np.float32)
    for idx in marked:
        if 0 <= idx < n:
            diag[idx] = -1.0
    return TransMatrix(data=np.diag(diag))


def diffusion_matrix(n: int) -> TransMatrix:
    """Grover diffusion operator: D = 2|s><s| - I, where |s> = 1/sqrt(N) * [1,...,1].

    All elements are real. Negative entries enable interference.
    """
    s = np.ones((n, 1), dtype=np.float32) / np.sqrt(n)
    D = 2.0 * (s @ s.T) - np.eye(n, dtype=np.float32)
    return TransMatrix(data=D.astype(np.float32))


# ---------------------------------------------------------------------------
# Complex quantum operations
# ---------------------------------------------------------------------------

def transform_complex(vec: ComplexVec, matrix: TransMatrix) -> ComplexVec:
    """Complex linear transformation: vec @ matrix -> ComplexVec."""
    v = vec.data.astype(np.complex128)
    m = matrix.data.astype(np.complex128)
    if v.shape[0] != m.shape[0]:
        raise ValueError(
            f"Dimension mismatch: vec({v.shape[0]}) vs matrix({m.shape[0]}x{m.shape[1]})"
        )
    return ComplexVec(data=v @ m)


def measure_complex(vec: ComplexVec) -> FloatVec:
    """Born rule on complex amplitudes: |alpha_i|^2 normalized."""
    probs = np.abs(vec.data) ** 2
    total = np.sum(probs)
    if total > 0:
        probs = probs / total
    return FloatVec(data=probs.astype(np.float32))


def interfere(
    vec1: ComplexVec,
    vec2: ComplexVec,
    phase: float = 0.0,
) -> ComplexVec:
    """Quantum interference: (vec1 + exp(i*phase) * vec2), normalized.

    phase=0:   constructive interference
    phase=pi:  destructive interference
    """
    result = vec1.data.astype(np.complex128) + np.exp(1j * phase) * vec2.data.astype(np.complex128)
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    return ComplexVec(data=result)


def evolve_density(rho: DensityMatrix, U: TransMatrix) -> DensityMatrix:
    """Unitary evolution of density matrix: rho' = U rho U_dagger."""
    u = U.data.astype(np.complex128)
    new_rho = u @ rho.data @ u.conj().T
    # Ensure Hermitian
    new_rho = (new_rho + new_rho.conj().T) / 2
    return DensityMatrix(data=new_rho)


def partial_trace(
    rho: DensityMatrix,
    dim_a: int,
    dim_b: int,
    trace_out: str = "B",
) -> DensityMatrix:
    """Partial trace of bipartite system AB.

    Args:
        rho: density matrix of composite system (dim_a*dim_b x dim_a*dim_b)
        dim_a: dimension of subsystem A
        dim_b: dimension of subsystem B
        trace_out: "A" or "B" -- which subsystem to trace out

    Returns:
        Reduced density matrix of the remaining subsystem.
    """
    rho_tensor = rho.data.reshape(dim_a, dim_b, dim_a, dim_b)

    if trace_out == "B":
        result = np.trace(rho_tensor, axis1=1, axis2=3)
    else:
        result = np.trace(rho_tensor, axis1=0, axis2=2)

    return DensityMatrix(data=result.astype(np.complex128))


# ---------------------------------------------------------------------------
# Bundle-level helpers
# ---------------------------------------------------------------------------

def transform_bundle(
    bundle: StateBundle,
    key: str,
    matrix: TransMatrix,
    *,
    out_key: str | None = None,
) -> StateBundle:
    """Apply transform to a single vector in the bundle, returning a new bundle."""
    result = bundle.copy()
    result[out_key or key] = transform(bundle[key], matrix)
    return result


def gate_bundle(
    bundle: StateBundle,
    key: str,
    g: GateVec,
    *,
    out_key: str | None = None,
) -> StateBundle:
    """Apply gate to a single vector in the bundle, returning a new bundle."""
    result = bundle.copy()
    result[out_key or key] = gate(bundle[key], g)
    return result
