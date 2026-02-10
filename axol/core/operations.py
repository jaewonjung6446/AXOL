"""Axol 5 primitive operations — pure functions over vector types."""

from __future__ import annotations

import numpy as np

from axol.core.types import (
    _VecBase,
    FloatVec,
    GateVec,
    TransMatrix,
    StateBundle,
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
