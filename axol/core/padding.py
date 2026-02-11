"""Axol padding layer — dimension-hiding double encryption.

Embeds all vectors/matrices into a uniform max_dim space,
then encrypts in the padded domain. This hides the original
dimensions from the server.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from axol.core.types import FloatVec, TransMatrix, StateBundle
from axol.core.program import (
    Program, Transition, TransformOp, GateOp, MergeOp,
    SecurityLevel,
    run_program,
)
from axol.core.encryption import (
    KeyFamily, encrypt_vec, encrypt_matrix_rect,
)


@dataclass(frozen=True)
class PaddedProgram:
    """A program padded to uniform dimension and then encrypted."""
    encrypted_program: Program
    max_dim: int
    key_family: KeyFamily
    original_dims: dict[str, int]
    dim_map: dict[str, int]


def _embed_matrix(M: np.ndarray, max_dim: int) -> np.ndarray:
    """Embed an N×M matrix into max_dim × max_dim.

    The top-left N×M block is M, the rest is identity-padded:
    - diagonal elements outside the block are 1.0
    - off-diagonal padding elements are 0.0
    """
    n, m = M.shape
    if n > max_dim or m > max_dim:
        raise ValueError(
            f"Matrix shape ({n}, {m}) exceeds max_dim={max_dim}"
        )
    result = np.eye(max_dim, dtype=np.float32)
    result[:n, :m] = M.astype(np.float32)
    # Zero out the identity parts that overlap with M's region
    # but aren't part of M
    if n < max_dim and m < max_dim:
        # The bottom-right identity block starts at (n, m)
        # We need: top-right block (0:n, m:max_dim) = 0
        result[:n, m:max_dim] = 0.0
        # bottom-left block (n:max_dim, 0:m) = 0
        result[n:max_dim, :m] = 0.0
    elif n < max_dim:
        result[n:max_dim, :m] = 0.0
    elif m < max_dim:
        result[:n, m:max_dim] = 0.0
    return result


def _embed_vec(v: np.ndarray, max_dim: int) -> np.ndarray:
    """Embed a vector of size N into max_dim (zero-padded)."""
    n = v.shape[0]
    if n > max_dim:
        raise ValueError(f"Vector size {n} exceeds max_dim={max_dim}")
    result = np.zeros(max_dim, dtype=np.float32)
    result[:n] = v.astype(np.float32)
    return result


def pad_and_encrypt(
    program: Program, key_family: KeyFamily, max_dim: int
) -> PaddedProgram:
    """Pad all vectors/matrices to max_dim, then encrypt.

    Steps:
      1. Record original dimensions.
      2. Embed all vectors to max_dim (zero-pad).
      3. Embed all matrices to max_dim×max_dim.
      4. Encrypt everything with K_{max_dim}.

    Returns:
        PaddedProgram with encrypted uniform-dimension program.
    """
    original_dims: dict[str, int] = {}
    K = key_family.key(max_dim)

    # Pad and encrypt initial state
    new_vectors = {}
    for key, vec in program.initial_state.items():
        if vec.data.ndim == 1:
            original_dims[key] = vec.data.shape[0]
            padded = _embed_vec(vec.data, max_dim)
            new_vectors[key] = FloatVec(data=encrypt_vec(padded, K))
        else:
            new_vectors[key] = vec
    new_state = StateBundle(vectors=new_vectors)

    # dim_map: all keys are now max_dim
    dim_map = {k: max_dim for k in new_vectors}

    # Pad and encrypt transitions
    new_transitions = []
    for t in program.transitions:
        op = t.operation
        security = getattr(op, "security", SecurityLevel.PLAINTEXT)

        if security == SecurityLevel.PLAINTEXT:
            new_transitions.append(t)
            continue

        if isinstance(op, TransformOp):
            m = op.matrix.data
            padded_m = _embed_matrix(m, max_dim)
            # Square matrix encryption: M' = K^T @ M @ K
            K_inv = key_family.inv(max_dim)
            m_enc = (K_inv @ padded_m @ K).astype(np.float32)
            new_op = TransformOp(
                key=op.key,
                matrix=TransMatrix(data=m_enc),
                out_key=op.out_key,
            )
            out_key_name = op.out_key or op.key
            dim_map[out_key_name] = max_dim
            new_transitions.append(Transition(name=t.name, operation=new_op, metadata=t.metadata))
        else:
            new_transitions.append(t)

    encrypted_program = Program(
        name=f"{program.name}_padded",
        initial_state=new_state,
        transitions=new_transitions,
        terminal_key=program.terminal_key,
        max_iterations=program.max_iterations,
    )

    return PaddedProgram(
        encrypted_program=encrypted_program,
        max_dim=max_dim,
        key_family=key_family,
        original_dims=original_dims,
        dim_map=dim_map,
    )


def unpad_and_decrypt(
    result: StateBundle, padded_program: PaddedProgram
) -> StateBundle:
    """Decrypt and unpad the result back to original dimensions.

    Steps:
      1. Decrypt all max_dim vectors with K_{max_dim}.
      2. Trim each vector to its original dimension.
    """
    kf = padded_program.key_family
    max_dim = padded_program.max_dim
    K_inv = kf.inv(max_dim)

    new_vectors = {}
    for key, vec in result.items():
        if vec.data.ndim == 1 and vec.data.shape[0] == max_dim:
            dec_data = (vec.data.astype(np.float32) @ K_inv).astype(np.float32)
            orig_dim = padded_program.original_dims.get(key)
            if orig_dim is not None:
                new_vectors[key] = FloatVec(data=dec_data[:orig_dim])
            else:
                new_vectors[key] = FloatVec(data=dec_data)
        else:
            new_vectors[key] = vec
    return StateBundle(vectors=new_vectors)
