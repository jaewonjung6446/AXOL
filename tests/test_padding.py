"""Tests for axol.core.padding — dimension-hiding double encryption.

Verifies embedding helpers, pad/encrypt/decrypt roundtrip, dimension hiding,
and error handling for the padding layer.
"""

import pytest
import numpy as np

from axol.core.types import FloatVec, GateVec, TransMatrix, StateBundle
from axol.core.program import (
    TransformOp, CustomOp, Transition, Program, run_program,
)
from axol.core.encryption import KeyFamily
from axol.core.padding import (
    PaddedProgram, pad_and_encrypt, unpad_and_decrypt,
    _embed_vec, _embed_matrix,
)


# ───────────────────────────────────────────────────────────────────────────
# 1. _embed_vec
# ───────────────────────────────────────────────────────────────────────────

def test_embed_vec():
    """_embed_vec zero-pads a short vector to max_dim."""
    v = np.array([1, 2, 3], dtype=np.float32)
    result = _embed_vec(v, 5)

    assert result.shape == (5,)
    np.testing.assert_allclose(result, [1, 2, 3, 0, 0], atol=1e-4)


# ───────────────────────────────────────────────────────────────────────────
# 2. _embed_matrix
# ───────────────────────────────────────────────────────────────────────────

def test_embed_matrix():
    """_embed_matrix pads a 2x2 matrix into 4x4 with identity in the remainder."""
    M = np.array([[2, 3], [4, 5]], dtype=np.float32)
    result = _embed_matrix(M, 4)

    assert result.shape == (4, 4)
    # Top-left 2x2 block is M
    np.testing.assert_allclose(result[:2, :2], M, atol=1e-4)
    # Bottom-right 2x2 block is identity
    np.testing.assert_allclose(result[2:, 2:], np.eye(2), atol=1e-4)
    # Off-diagonal padding blocks are zero
    np.testing.assert_allclose(result[:2, 2:], 0.0, atol=1e-4)
    np.testing.assert_allclose(result[2:, :2], 0.0, atol=1e-4)


# ───────────────────────────────────────────────────────────────────────────
# 3. pad -> encrypt -> run -> decrypt -> unpad roundtrip
# ───────────────────────────────────────────────────────────────────────────

def test_pad_encrypt_decrypt_roundtrip():
    """Padded+encrypted program gives the same result as the plain program."""
    # Simple 2-state transform: [1, 0] @ [[0, 1], [1, 0]] = [0, 1]
    M = TransMatrix.from_list([[0.0, 1.0], [1.0, 0.0]])
    initial = StateBundle(vectors={
        "s": FloatVec.from_list([1.0, 0.0]),
    })
    plain_prog = Program(
        name="swap_test",
        initial_state=initial,
        transitions=[
            Transition(name="swap", operation=TransformOp(key="s", matrix=M)),
        ],
    )

    # Run plain
    plain_result = run_program(plain_prog)
    plain_s = plain_result.final_state["s"].data

    # Pad, encrypt, run, unpad, decrypt
    kf = KeyFamily(seed=42)
    max_dim = 8
    padded = pad_and_encrypt(plain_prog, kf, max_dim)
    enc_result = run_program(padded.encrypted_program)
    dec_result = unpad_and_decrypt(enc_result.final_state, padded)

    np.testing.assert_allclose(dec_result["s"].data, plain_s, atol=1e-4)


# ───────────────────────────────────────────────────────────────────────────
# 4. Dimension hiding — all vectors in encrypted program have max_dim
# ───────────────────────────────────────────────────────────────────────────

def test_dimension_hidden():
    """All 1-D vectors in the encrypted program must have size == max_dim."""
    initial = StateBundle(vectors={
        "a": FloatVec.from_list([1.0, 2.0, 3.0]),
        "b": FloatVec.from_list([4.0, 5.0]),
    })
    # Identity transform on 'a' (no shape change)
    M_a = TransMatrix(data=np.eye(3, dtype=np.float32))
    prog = Program(
        name="dim_test",
        initial_state=initial,
        transitions=[
            Transition(name="id_a", operation=TransformOp(key="a", matrix=M_a)),
        ],
    )

    kf = KeyFamily(seed=99)
    max_dim = 8
    padded = pad_and_encrypt(prog, kf, max_dim)

    enc_state = padded.encrypted_program.initial_state
    for key, vec in enc_state.items():
        if vec.data.ndim == 1:
            assert vec.data.shape[0] == max_dim, (
                f"Vector '{key}' has dim {vec.data.shape[0]}, expected {max_dim}"
            )


# ───────────────────────────────────────────────────────────────────────────
# 5. original_dims preserved
# ───────────────────────────────────────────────────────────────────────────

def test_original_dims_preserved():
    """PaddedProgram.original_dims records the correct pre-padding sizes."""
    initial = StateBundle(vectors={
        "hp": FloatVec.from_list([100.0]),
        "pos": FloatVec.from_list([1.0, 2.0, 3.0]),
    })
    prog = Program(
        name="dims_test",
        initial_state=initial,
        transitions=[],
    )

    kf = KeyFamily(seed=7)
    padded = pad_and_encrypt(prog, kf, max_dim=16)

    assert padded.original_dims["hp"] == 1
    assert padded.original_dims["pos"] == 3


# ───────────────────────────────────────────────────────────────────────────
# 6. Different max_dims both produce correct results
# ───────────────────────────────────────────────────────────────────────────

def test_different_max_dims():
    """The same program padded to max_dim=8 vs max_dim=16 both decrypt correctly."""
    M = TransMatrix.from_list([[0.8, 0.0], [0.0, 0.5]])
    initial = StateBundle(vectors={
        "v": FloatVec.from_list([10.0, 20.0]),
    })
    prog = Program(
        name="multi_dim",
        initial_state=initial,
        transitions=[
            Transition(name="scale", operation=TransformOp(key="v", matrix=M)),
        ],
    )
    plain_result = run_program(prog).final_state["v"].data

    kf = KeyFamily(seed=123)

    for md in (8, 16):
        padded = pad_and_encrypt(prog, kf, md)
        enc_result = run_program(padded.encrypted_program)
        dec_result = unpad_and_decrypt(enc_result.final_state, padded)
        np.testing.assert_allclose(
            dec_result["v"].data, plain_result, atol=1e-4,
            err_msg=f"Failed for max_dim={md}",
        )


# ───────────────────────────────────────────────────────────────────────────
# 7. 3-state machine padded and encrypted
# ───────────────────────────────────────────────────────────────────────────

def test_3state_machine_padded():
    """Encrypt a 3-state IDLE->RUNNING->DONE machine with max_dim=8, verify result."""
    state_vec = FloatVec.from_list([1.0, 0.0, 0.0])  # IDLE
    shift = TransMatrix.from_list([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])

    # Two transitions to go IDLE -> RUNNING -> DONE
    prog = Program(
        name="sm3",
        initial_state=StateBundle(vectors={"state": state_vec}),
        transitions=[
            Transition(name="step1", operation=TransformOp(key="state", matrix=shift)),
            Transition(name="step2", operation=TransformOp(key="state", matrix=shift)),
        ],
    )

    # Plain run
    plain_final = run_program(prog).final_state["state"].data
    np.testing.assert_allclose(plain_final, [0.0, 0.0, 1.0], atol=1e-4)

    # Padded + encrypted run
    kf = KeyFamily(seed=55)
    padded = pad_and_encrypt(prog, kf, max_dim=8)
    enc_result = run_program(padded.encrypted_program)
    dec_result = unpad_and_decrypt(enc_result.final_state, padded)

    np.testing.assert_allclose(dec_result["state"].data, [0.0, 0.0, 1.0], atol=1e-4)


# ───────────────────────────────────────────────────────────────────────────
# 8. Error when max_dim is too small
# ───────────────────────────────────────────────────────────────────────────

def test_error_on_too_small_max_dim():
    """max_dim smaller than a vector dimension must raise ValueError."""
    initial = StateBundle(vectors={
        "big": FloatVec.from_list([1.0, 2.0, 3.0, 4.0, 5.0]),
    })
    prog = Program(
        name="too_small",
        initial_state=initial,
        transitions=[],
    )

    kf = KeyFamily(seed=0)
    with pytest.raises(ValueError, match="exceeds max_dim"):
        pad_and_encrypt(prog, kf, max_dim=3)
