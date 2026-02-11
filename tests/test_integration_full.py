"""End-to-end integration tests for the full Axol pipeline.

Combines: fn_to_matrix -> pad_and_encrypt -> run_program -> unpad_and_decrypt
Tests the complete flow from function compilation through encrypted execution
and back to plaintext verification.
"""

import pytest
import numpy as np

from axol.core.compiler import fn_to_matrix, truth_table_to_matrix
from axol.core.padding import pad_and_encrypt, unpad_and_decrypt
from axol.core.encryption import KeyFamily, encrypt_program_full, decrypt_state_full
from axol.core.types import FloatVec, OneHotVec, TransMatrix, StateBundle
from axol.core.program import Program, Transition, TransformOp, run_program
from axol.api.client import AxolClient
from axol.api.dispatch import dispatch


class TestE2EFnToMatrixEncryptRun:
    """Test 1: fn_to_matrix -> pad_and_encrypt -> run -> unpad_and_decrypt."""

    def test_e2e_fn_to_matrix_encrypt_run(self):
        # Compile lambda x: (x+1)%4 into a 4x4 permutation matrix
        M = fn_to_matrix(lambda x: (x + 1) % 4, 4, 4)
        assert M.shape == (4, 4)

        # Build a program: apply the matrix to OneHot(0, 4)
        state = StateBundle(vectors={
            "v": OneHotVec.from_index(0, 4),
        })
        prog = Program(
            name="shift4",
            initial_state=state,
            transitions=[
                Transition("shift", TransformOp(key="v", matrix=M)),
            ],
        )

        # Pad and encrypt with max_dim=8
        kf = KeyFamily(seed=42)
        padded = pad_and_encrypt(prog, kf, max_dim=8)

        # Run the encrypted, padded program
        result = run_program(padded.encrypted_program)

        # Unpad and decrypt
        decrypted = unpad_and_decrypt(result.final_state, padded)

        # Verify: OneHot(0,4) @ shift_matrix = OneHot(1,4)
        expected = OneHotVec.from_index(1, 4).data
        np.testing.assert_allclose(decrypted["v"].data, expected, atol=1e-3)


class TestE2ETruthTableEncryptRun:
    """Test 2: truth_table_to_matrix -> pad_and_encrypt -> run -> decrypt -> verify."""

    def test_e2e_truth_table_encrypt_run(self):
        # Build truth table: 0->2, 1->0, 2->1
        table = {0: 2, 1: 0, 2: 1}
        M = truth_table_to_matrix(table, 3, 3)
        assert M.shape == (3, 3)

        # Build program: apply truth table matrix to OneHot(0, 3)
        state = StateBundle(vectors={
            "s": OneHotVec.from_index(0, 3),
        })
        prog = Program(
            name="truth_table_test",
            initial_state=state,
            transitions=[
                Transition("apply", TransformOp(key="s", matrix=M)),
            ],
        )

        # Pad and encrypt
        kf = KeyFamily(seed=77)
        padded = pad_and_encrypt(prog, kf, max_dim=8)

        # Run encrypted
        result = run_program(padded.encrypted_program)

        # Unpad and decrypt
        decrypted = unpad_and_decrypt(result.final_state, padded)

        # Verify: input 0 maps to output 2 => OneHot(2, 3)
        expected = OneHotVec.from_index(2, 3).data
        np.testing.assert_allclose(decrypted["s"].data, expected, atol=1e-3)


class TestE2ERectFnExpand:
    """Test 3: fn_to_matrix (3->5 rectangular) + KeyFamily encrypt -> run -> decrypt."""

    def test_e2e_rect_fn_expand(self):
        # fn maps 3 inputs to 5 outputs: fn(x) = x * 2
        M = fn_to_matrix(lambda x: x * 2, 3, 5)
        assert M.shape == (3, 5)

        # Build program with OneHot(1, 3) as input
        # fn(1) = 2, so result should be OneHot(2, 5)
        state = StateBundle(vectors={
            "v": OneHotVec.from_index(1, 3),
        })
        prog = Program(
            name="rect_expand",
            initial_state=state,
            transitions=[
                Transition("expand", TransformOp(key="v", matrix=M)),
            ],
        )

        # Encrypt with KeyFamily (no padding, uses encrypt_program_full for rect support)
        kf = KeyFamily(seed=99)
        enc_prog, dim_map = encrypt_program_full(prog, kf)

        # Run encrypted program
        result = run_program(enc_prog)

        # Decrypt
        decrypted = decrypt_state_full(result.final_state, kf, dim_map)

        # Verify: OneHot(1,3) through fn(x)=x*2 => OneHot(2,5)
        expected = OneHotVec.from_index(2, 5).data
        np.testing.assert_allclose(decrypted["v"].data, expected, atol=1e-3)


class TestE2EClientFullPipeline:
    """Test 4: AxolClient with use_padding=True, multi-step chained transforms."""

    def test_e2e_client_full_pipeline(self):
        # Create client with padding enabled
        client = AxolClient(seed=42, max_dim=16, use_padding=True)

        # Build a 2-step chained transform:
        #   Step 1: cyclic shift OneHot(0,4) -> OneHot(1,4)
        #   Step 2: cyclic shift OneHot(1,4) -> OneHot(2,4)
        M_shift = fn_to_matrix(lambda x: (x + 1) % 4, 4, 4)

        state = StateBundle(vectors={
            "v": OneHotVec.from_index(0, 4),
        })
        prog = Program(
            name="double_shift",
            initial_state=state,
            transitions=[
                Transition("shift1", TransformOp(key="v", matrix=M_shift)),
                Transition("shift2", TransformOp(key="v", matrix=M_shift)),
            ],
        )

        # Run the full client pipeline: encrypt -> run -> decrypt
        result = client.run_local(prog)

        # After two cyclic shifts: OneHot(0,4) -> OneHot(1,4) -> OneHot(2,4)
        expected = OneHotVec.from_index(2, 4).data
        np.testing.assert_allclose(result["v"].data, expected, atol=1e-3)


class TestE2EDispatchPaddedRun:
    """Test 5: dispatch padded_run vs plain run — results should be close."""

    def test_e2e_dispatch_padded_run(self):
        # DSL source: simple transform pipeline
        dsl_code = "@pipe\ns v=[1 0 0]\n: t=transform(v;M=[0 1 0;0 0 1;1 0 0])"

        # Run with padding + encryption
        padded_result = dispatch({
            "action": "padded_run",
            "source": dsl_code,
            "max_dim": 8,
            "seed": 42,
        })
        assert "error" not in padded_result, f"padded_run failed: {padded_result.get('error')}"

        # Run plaintext
        plain_result = dispatch({
            "action": "run",
            "source": dsl_code,
        })
        assert "error" not in plain_result, f"run failed: {plain_result.get('error')}"

        # Compare final states — should be close (atol=1e-3)
        for key in plain_result["final_state"]:
            plain_vec = np.array(plain_result["final_state"][key], dtype=np.float32)
            padded_vec = np.array(padded_result["final_state"][key], dtype=np.float32)
            np.testing.assert_allclose(padded_vec, plain_vec, atol=1e-3,
                                       err_msg=f"Mismatch for key '{key}'")
