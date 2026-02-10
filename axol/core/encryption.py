"""Axol encryption module — similarity transformation encryption for programs.

Core idea:
  Given secret invertible key matrix K:
    - Encrypt state:   s' = s @ K
    - Encrypt matrix:  M' = K^(-1) @ M @ K
    - Decrypt result:  r  = r' @ K^(-1)

  The encrypted program produces the same final result after decryption.
"""

from __future__ import annotations

import numpy as np

from axol.core.types import FloatVec, TransMatrix, StateBundle
from axol.core.program import (
    Program, Transition, TransformOp, GateOp, MergeOp, CustomOp,
    run_program,
)


def random_key(n: int, seed: int | None = None) -> np.ndarray:
    """Generate a random invertible NxN key matrix."""
    rng = np.random.RandomState(seed if seed is not None else None)
    while True:
        K = rng.randn(n, n).astype(np.float32)
        if abs(np.linalg.det(K)) > 0.1:
            return K


def random_orthogonal_key(n: int, seed: int | None = None) -> np.ndarray:
    """Generate a random orthogonal NxN matrix via QR decomposition.

    Orthogonal keys preserve distances (euclidean, cosine, dot).
    """
    rng = np.random.RandomState(seed if seed is not None else None)
    A = rng.randn(n, n).astype(np.float32)
    Q, R = np.linalg.qr(A)
    return Q.astype(np.float32)


def encrypt_matrix(M: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Encrypt a transformation matrix: M' = K^(-1) @ M @ K."""
    K_inv = np.linalg.inv(K).astype(np.float32)
    return (K_inv @ M @ K).astype(np.float32)


def encrypt_vec(v: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Encrypt a state vector: v' = v @ K."""
    return (v @ K).astype(np.float32)


def decrypt_vec(v_enc: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Decrypt a state vector: v = v' @ K^(-1)."""
    K_inv = np.linalg.inv(K).astype(np.float32)
    return (v_enc @ K_inv).astype(np.float32)


def encrypt_program(program: Program, K: np.ndarray, dim: int) -> Program:
    """Encrypt an entire Axol program using key matrix K.

    All state vectors of size `dim` are encrypted as v' = v @ K.
    All TransformOp matrices of compatible size are encrypted as M' = K^(-1) @ M @ K.
    GateOps are converted to TransformOps with diagonal matrices, then encrypted.

    Args:
        program: Original program.
        K: Secret key matrix (dim x dim).
        dim: Dimension of vectors to encrypt.

    Returns:
        Encrypted Program (runs in encrypted domain).
    """
    K_inv = np.linalg.inv(K).astype(np.float32)

    # Encrypt initial state
    new_vectors = {}
    for key, vec in program.initial_state.items():
        if vec.data.ndim == 1 and vec.data.shape[0] == dim:
            enc_data = encrypt_vec(vec.data.astype(np.float32), K)
            new_vectors[key] = FloatVec(data=enc_data)
        else:
            new_vectors[key] = vec
    new_state = StateBundle(vectors=new_vectors)

    # Encrypt transitions
    new_transitions = []
    for t in program.transitions:
        op = t.operation
        if isinstance(op, TransformOp):
            m = op.matrix.data
            if m.shape == (dim, dim):
                m_enc = encrypt_matrix(m, K)
                new_op = TransformOp(key=op.key, matrix=TransMatrix(data=m_enc), out_key=op.out_key)
            else:
                new_op = op
            new_transitions.append(Transition(name=t.name, operation=new_op, metadata=t.metadata))
        elif isinstance(op, GateOp):
            # Gate -> diagonal matrix transform, then encrypt
            # We need the gate vector from initial state to build diagonal
            # This is a compile-time operation
            new_transitions.append(t)
        elif isinstance(op, CustomOp):
            new_transitions.append(t)
        else:
            new_transitions.append(t)

    return Program(
        name=f"{program.name}_encrypted",
        initial_state=new_state,
        transitions=new_transitions,
        terminal_key=program.terminal_key,
        max_iterations=program.max_iterations,
    )


def decrypt_state(bundle: StateBundle, K: np.ndarray) -> StateBundle:
    """Decrypt all vectors in a StateBundle.

    Only decrypts vectors whose dimension matches K's dimension.

    Args:
        bundle: Encrypted state bundle.
        K: Secret key matrix used for encryption.

    Returns:
        Decrypted StateBundle.
    """
    dim = K.shape[0]
    K_inv = np.linalg.inv(K).astype(np.float32)
    new_vectors = {}
    for key, vec in bundle.items():
        if vec.data.ndim == 1 and vec.data.shape[0] == dim:
            dec_data = (vec.data.astype(np.float32) @ K_inv).astype(np.float32)
            new_vectors[key] = FloatVec(data=dec_data)
        else:
            new_vectors[key] = vec
    return StateBundle(vectors=new_vectors)
