"""Tests for branch-to-transform compilation in the encryption module.

Verifies that _branch_to_transforms correctly compiles BranchOp into
encrypted TransformOps + MergeOp, and that the encrypted branch produces
the same result as the plaintext branch after decryption.

branch(gate; then=a; else=b) compiles to:
    then_masked = a @ diag(gate)         (encrypted TransformOp)
    else_masked = b @ diag(1 - gate)     (encrypted TransformOp)
    out = merge(then_masked, else_masked, weights=[1,1])  (MergeOp)
"""

import pytest
import numpy as np

from axol.core.encryption import (
    KeyFamily,
    _branch_to_transforms,
    encrypt_program_full,
    decrypt_state_full,
)
from axol.core.types import FloatVec, GateVec, StateBundle, TransMatrix
from axol.core.program import (
    Program,
    Transition,
    BranchOp,
    TransformOp,
    MergeOp,
    run_program,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_compiled_branch(
    gate: list[float],
    then_vec: list[float],
    else_vec: list[float],
    seed: int = 42,
) -> np.ndarray:
    """Compile a BranchOp via _branch_to_transforms, execute, and decrypt.

    Returns the decrypted result vector.
    """
    kf = KeyFamily(seed=seed)
    dim = len(gate)

    initial_state = StateBundle(vectors={
        "gate": GateVec.from_list(gate),
        "then": FloatVec.from_list(then_vec),
        "else": FloatVec.from_list(else_vec),
    })
    dim_map: dict[str, int] = {
        "gate": dim,
        "then": dim,
        "else": dim,
    }

    branch_op = BranchOp(
        gate_key="gate",
        then_key="then",
        else_key="else",
        out_key="result",
    )

    transitions = _branch_to_transforms(branch_op, initial_state, dim_map, kf)

    # Encrypt the initial state vectors
    enc_vectors: dict[str, FloatVec] = {}
    for key, vec in initial_state.items():
        K = kf.key(dim)
        enc_data = (vec.data.astype(np.float32) @ K).astype(np.float32)
        enc_vectors[key] = FloatVec(data=enc_data)
    enc_state = StateBundle(vectors=enc_vectors)

    program = Program(
        name="branch_test",
        initial_state=enc_state,
        transitions=transitions,
    )
    result = run_program(program)

    # Decrypt the result
    decrypted = decrypt_state_full(result.final_state, kf, dim_map)
    return decrypted["result"].data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBranchToTransforms:

    def test_branch_to_transforms_basic(self):
        """gate=[1,0,1]: compilation must produce exactly 3 transitions
        (then_masked, else_masked, merge).
        """
        kf = KeyFamily(seed=7)
        dim = 3

        initial_state = StateBundle(vectors={
            "gate": GateVec.from_list([1.0, 0.0, 1.0]),
            "a": FloatVec.from_list([10.0, 20.0, 30.0]),
            "b": FloatVec.from_list([40.0, 50.0, 60.0]),
        })
        dim_map: dict[str, int] = {"gate": dim, "a": dim, "b": dim}

        branch_op = BranchOp(
            gate_key="gate",
            then_key="a",
            else_key="b",
            out_key="out",
        )

        transitions = _branch_to_transforms(branch_op, initial_state, dim_map, kf)

        assert len(transitions) == 3, (
            f"Expected 3 transitions (then, else, merge), got {len(transitions)}"
        )
        # First two should be TransformOps, last should be MergeOp
        assert isinstance(transitions[0].operation, TransformOp)
        assert isinstance(transitions[1].operation, TransformOp)
        assert isinstance(transitions[2].operation, MergeOp)

    def test_branch_all_then(self):
        """gate=[1,1,1]: result must equal then_vec exactly."""
        then_vec = [5.0, 15.0, 25.0]
        else_vec = [100.0, 200.0, 300.0]
        gate = [1.0, 1.0, 1.0]

        result = _run_compiled_branch(gate, then_vec, else_vec)
        np.testing.assert_allclose(result, then_vec, atol=1e-3)

    def test_branch_all_else(self):
        """gate=[0,0,0]: result must equal else_vec exactly."""
        then_vec = [100.0, 200.0, 300.0]
        else_vec = [7.0, 14.0, 21.0]
        gate = [0.0, 0.0, 0.0]

        result = _run_compiled_branch(gate, then_vec, else_vec)
        np.testing.assert_allclose(result, else_vec, atol=1e-3)

    def test_branch_mixed_gate(self):
        """gate=[1,0], then=[10,20], else=[30,40] must produce [10,40]."""
        gate = [1.0, 0.0]
        then_vec = [10.0, 20.0]
        else_vec = [30.0, 40.0]

        result = _run_compiled_branch(gate, then_vec, else_vec)
        np.testing.assert_allclose(result, [10.0, 40.0], atol=1e-3)

    def test_encrypt_program_full_with_branch(self):
        """Full program containing a BranchOp: encrypt, run, decrypt must
        match the plaintext execution.
        """
        dim = 3
        gate = [1.0, 0.0, 1.0]
        then_vec = [10.0, 20.0, 30.0]
        else_vec = [40.0, 50.0, 60.0]
        # Expected plain result: [10, 50, 30]

        initial_state = StateBundle(vectors={
            "gate": GateVec.from_list(gate),
            "a": FloatVec.from_list(then_vec),
            "b": FloatVec.from_list(else_vec),
        })

        branch_op = BranchOp(
            gate_key="gate",
            then_key="a",
            else_key="b",
            out_key="result",
        )

        plain_program = Program(
            name="branch_plain",
            initial_state=initial_state,
            transitions=[
                Transition(name="branch", operation=branch_op),
            ],
        )

        # Run plaintext
        plain_result = run_program(plain_program)
        plain_out = plain_result.final_state["result"].data

        # Encrypt, run, decrypt
        kf = KeyFamily(seed=123)
        enc_program, dim_map = encrypt_program_full(plain_program, kf)
        enc_result = run_program(enc_program)
        dec_state = decrypt_state_full(enc_result.final_state, kf, dim_map)
        dec_out = dec_state["result"].data

        np.testing.assert_allclose(dec_out, plain_out, atol=1e-3)
        np.testing.assert_allclose(dec_out, [10.0, 50.0, 30.0], atol=1e-3)

    def test_branch_preserves_semantics(self):
        """Compare encrypted branch result with plain branch result across
        several gate patterns to verify semantic equivalence.
        """
        kf = KeyFamily(seed=77)
        dim = 4
        then_vec = [1.0, 2.0, 3.0, 4.0]
        else_vec = [10.0, 20.0, 30.0, 40.0]

        gate_patterns = [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]

        for gate in gate_patterns:
            # Plain branch: where gate==1 pick then, else pick else
            g = np.array(gate, dtype=np.float32)
            t = np.array(then_vec, dtype=np.float32)
            e = np.array(else_vec, dtype=np.float32)
            expected = np.where(g == 1.0, t, e).astype(np.float32)

            # Encrypted branch via compilation
            result = _run_compiled_branch(gate, then_vec, else_vec, seed=77)

            np.testing.assert_allclose(
                result,
                expected,
                atol=1e-3,
                err_msg=f"Semantic mismatch for gate={gate}",
            )
