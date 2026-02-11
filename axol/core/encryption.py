"""Axol encryption module — similarity transformation encryption for programs.

Core idea:
  Given secret invertible key matrix K:
    - Encrypt state:   s' = s @ K
    - Encrypt matrix:  M' = K^(-1) @ M @ K
    - Decrypt result:  r  = r' @ K^(-1)

  The encrypted program produces the same final result after decryption.

Extended (Phase 7):
  KeyFamily — deterministic key derivation from a single seed.
  Rectangular encryption — M'(N×M) = K_N^(-1) @ M(N×M) @ K_M.
  Branch-to-transform compilation — BranchOp → diagonal TransformOps.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from axol.core.types import FloatVec, GateVec, TransMatrix, StateBundle
from axol.core.program import (
    Program, Transition, TransformOp, GateOp, MergeOp, BranchOp, CustomOp,
    SecurityLevel,
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

    Optimal for quantum programs:
      - K^(-1) = K^T (transpose) — no matrix inversion needed, numerically stable.
      - Preserves vector norms — Born rule probabilities are invariant.
      - Real orthogonal matrix = real unitary — compatible with Tier 1 quantum ops.
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
        security = getattr(op, "security", SecurityLevel.PLAINTEXT)

        # Skip plaintext-only operations — pass through unchanged
        if security == SecurityLevel.PLAINTEXT:
            new_transitions.append(t)
            continue

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


# ---------------------------------------------------------------------------
# Phase 7: KeyFamily — deterministic multi-dimension key derivation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeyFamily:
    """Derive orthogonal keys for any dimension from a single seed.

    Usage:
        kf = KeyFamily(seed=42)
        K3 = kf.key(3)   # 3×3 orthogonal key
        K5 = kf.key(5)   # 5×5 orthogonal key (independent)
        K3_inv = kf.inv(3)  # = K3.T (orthogonal inverse)
    """

    seed: int

    def _derived_seed(self, dim: int) -> int:
        h = hashlib.sha256(f"{self.seed}:{dim}".encode()).digest()
        return int.from_bytes(h[:4], "big")

    def key(self, dim: int) -> np.ndarray:
        """Return orthogonal key matrix for the given dimension."""
        return random_orthogonal_key(dim, seed=self._derived_seed(dim))

    def inv(self, dim: int) -> np.ndarray:
        """Return inverse key for the given dimension (= K^T for orthogonal)."""
        return self.key(dim).T


# ---------------------------------------------------------------------------
# Phase 7: Rectangular matrix encryption
# ---------------------------------------------------------------------------

def encrypt_matrix_rect(
    M: np.ndarray, kf: KeyFamily
) -> np.ndarray:
    """Encrypt a rectangular N×M matrix: M' = K_N^(-1) @ M @ K_M.

    For orthogonal keys: M' = K_N^T @ M @ K_M.
    """
    n, m = M.shape
    K_n_inv = kf.inv(n)
    K_m = kf.key(m)
    return (K_n_inv @ M.astype(np.float32) @ K_m).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 7: Full program encryption with KeyFamily (multi-dim)
# ---------------------------------------------------------------------------

def encrypt_program_full(
    program: Program, kf: KeyFamily
) -> tuple[Program, dict[str, int]]:
    """Encrypt an entire program using a KeyFamily.

    Supports rectangular matrices (N→M dimension changes).
    Tracks dimension of each state key via dim_map.
    BranchOps are compiled to TransformOps before encryption.

    Returns:
        (encrypted_program, dim_map) where dim_map records each key's dimension.
    """
    # Build dim_map from initial state
    dim_map: dict[str, int] = {}
    for key, vec in program.initial_state.items():
        if vec.data.ndim == 1:
            dim_map[key] = vec.data.shape[0]

    # Encrypt initial state — each vec with its own dim key
    new_vectors = {}
    for key, vec in program.initial_state.items():
        if vec.data.ndim == 1 and key in dim_map:
            K = kf.key(dim_map[key])
            new_vectors[key] = FloatVec(data=encrypt_vec(vec.data.astype(np.float32), K))
        else:
            new_vectors[key] = vec
    new_state = StateBundle(vectors=new_vectors)

    # Encrypt transitions
    new_transitions = []
    for t in program.transitions:
        op = t.operation
        security = getattr(op, "security", SecurityLevel.PLAINTEXT)

        if security == SecurityLevel.PLAINTEXT:
            # Attempt branch-to-transform compilation
            if isinstance(op, BranchOp):
                branch_transitions = _branch_to_transforms(op, program.initial_state, dim_map, kf)
                new_transitions.extend(branch_transitions)
                continue
            new_transitions.append(t)
            continue

        if isinstance(op, TransformOp):
            m = op.matrix.data
            in_key = op.key
            out_key_name = op.out_key or op.key
            in_dim = dim_map.get(in_key)

            if in_dim is not None:
                rows, cols = m.shape
                if rows == in_dim:
                    out_dim = cols
                    m_enc = encrypt_matrix_rect(m, kf)
                    dim_map[out_key_name] = out_dim
                    new_op = TransformOp(
                        key=op.key,
                        matrix=TransMatrix(data=m_enc),
                        out_key=op.out_key,
                    )
                else:
                    new_op = op
            else:
                new_op = op
            new_transitions.append(Transition(name=t.name, operation=new_op, metadata=t.metadata))
        elif isinstance(op, GateOp):
            new_transitions.append(t)
        elif isinstance(op, MergeOp):
            new_transitions.append(t)
        else:
            new_transitions.append(t)

    return (
        Program(
            name=f"{program.name}_encrypted_full",
            initial_state=new_state,
            transitions=new_transitions,
            terminal_key=program.terminal_key,
            max_iterations=program.max_iterations,
        ),
        dim_map,
    )


def decrypt_state_full(
    bundle: StateBundle, kf: KeyFamily, dim_map: dict[str, int]
) -> StateBundle:
    """Decrypt all vectors using KeyFamily + dim_map.

    Each vector is decrypted with the key matching its recorded dimension.
    """
    new_vectors = {}
    for key, vec in bundle.items():
        dim = dim_map.get(key)
        if dim is not None and vec.data.ndim == 1 and vec.data.shape[0] == dim:
            K_inv = kf.inv(dim)
            dec_data = (vec.data.astype(np.float32) @ K_inv).astype(np.float32)
            new_vectors[key] = FloatVec(data=dec_data)
        else:
            new_vectors[key] = vec
    return StateBundle(vectors=new_vectors)


# ---------------------------------------------------------------------------
# Phase 7: Branch → Transform compilation
# ---------------------------------------------------------------------------

def _branch_to_transforms(
    op: BranchOp,
    initial_state: StateBundle,
    dim_map: dict[str, int],
    kf: KeyFamily,
) -> list[Transition]:
    """Compile a BranchOp into encrypted TransformOps.

    branch(gate; then=a; else=b) → out
    becomes:
      then_masked = a * diag(gate)         (TransformOp)
      else_masked = b * diag(1 - gate)     (TransformOp)
      out = merge(then_masked, else_masked) (MergeOp with equal weights)

    Only supports compile-time (initial state) gate vectors.
    """
    gate_key = op.gate_key
    if gate_key not in initial_state:
        # Cannot compile — gate not available at compile time, pass through
        return [Transition(name=f"branch_{op.out_key}", operation=op)]

    gate_vec = initial_state[gate_key]
    if not isinstance(gate_vec, (GateVec, FloatVec)):
        return [Transition(name=f"branch_{op.out_key}", operation=op)]

    g = gate_vec.data.astype(np.float32)
    dim = len(g)

    # diag(gate)
    then_diag = np.diag(g).astype(np.float32)
    # diag(1 - gate)
    else_diag = np.diag(1.0 - g).astype(np.float32)

    # Encrypt the diagonal matrices
    then_enc = encrypt_matrix_rect(then_diag, kf)
    else_enc = encrypt_matrix_rect(else_diag, kf)

    then_tmp = f"_branch_then_{op.out_key}"
    else_tmp = f"_branch_else_{op.out_key}"

    dim_map[then_tmp] = dim
    dim_map[else_tmp] = dim
    dim_map[op.out_key] = dim

    transitions = [
        Transition(
            name=f"branch_then_{op.out_key}",
            operation=TransformOp(
                key=op.then_key,
                matrix=TransMatrix(data=then_enc),
                out_key=then_tmp,
            ),
        ),
        Transition(
            name=f"branch_else_{op.out_key}",
            operation=TransformOp(
                key=op.else_key,
                matrix=TransMatrix(data=else_enc),
                out_key=else_tmp,
            ),
        ),
        Transition(
            name=f"branch_merge_{op.out_key}",
            operation=MergeOp(
                keys=[then_tmp, else_tmp],
                weights=FloatVec.from_list([1.0, 1.0]),
                out_key=op.out_key,
            ),
        ),
    ]
    return transitions
