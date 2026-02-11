"""Tests for rectangular matrix encryption (Phase 7).

Covers:
  - encrypt_matrix_rect: K_N^T @ M(N x M) @ K_M for orthogonal KeyFamily keys
  - encrypt_program_full: full program encryption with multi-dim KeyFamily
  - decrypt_state_full: per-key decryption via dim_map
"""

import pytest
import numpy as np

from axol.core.types import FloatVec, TransMatrix, StateBundle, OneHotVec
from axol.core.program import (
    Program, Transition, TransformOp, run_program,
)
from axol.core.encryption import (
    KeyFamily,
    encrypt_matrix, encrypt_matrix_rect,
    encrypt_vec, decrypt_vec,
    encrypt_program_full, decrypt_state_full,
)


# ============================================================================
# 1. Basic rectangular encrypt/decrypt round-trips
# ============================================================================

class TestRectEncryptDecrypt:
    """Verify that rectangular encryption preserves computation:
    v(1xN) @ M(NxM) in plaintext == decrypt(encrypt(v) @ encrypt(M)).
    """

    def test_rect_3x5_encrypt_decrypt(self):
        """Encrypt a 3x5 matrix, verify decrypt recovers the original result."""
        kf = KeyFamily(seed=42)
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        M = np.array([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 1],
        ], dtype=np.float32)

        # Plaintext computation: v(1x3) @ M(3x5) = result(1x5)
        result_plain = v @ M

        # Encrypted: encrypt v with K3, encrypt M with rect, run, decrypt with K5
        K3 = kf.key(3)
        K5_inv = kf.inv(5)
        v_enc = (v @ K3).astype(np.float32)
        M_enc = encrypt_matrix_rect(M, kf)
        result_enc = v_enc @ M_enc

        # Decrypt result with K5 inverse
        result_dec = (result_enc @ K5_inv).astype(np.float32)
        np.testing.assert_allclose(result_dec, result_plain, atol=1e-4)

    def test_rect_5x3_encrypt_decrypt(self):
        """Encrypt a 5x3 matrix, verify decrypt recovers the original result."""
        kf = KeyFamily(seed=99)
        v = np.array([1.0, 0.0, 0.5, 2.0, -1.0], dtype=np.float32)
        M = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ], dtype=np.float32)

        result_plain = v @ M

        K5 = kf.key(5)
        K3_inv = kf.inv(3)
        v_enc = (v @ K5).astype(np.float32)
        M_enc = encrypt_matrix_rect(M, kf)
        result_enc = v_enc @ M_enc

        result_dec = (result_enc @ K3_inv).astype(np.float32)
        np.testing.assert_allclose(result_dec, result_plain, atol=1e-4)

    def test_square_matches_rect(self):
        """Square encryption via encrypt_matrix_rect should match
        encrypt_matrix when using the same orthogonal key."""
        kf = KeyFamily(seed=77)
        dim = 4
        K = kf.key(dim)

        M = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ], dtype=np.float32)

        # encrypt_matrix_rect uses K_N^T @ M @ K_M; for square N==M so K_N == K_M
        M_rect = encrypt_matrix_rect(M, kf)

        # encrypt_matrix uses K^(-1) @ M @ K; for orthogonal K^(-1) == K^T
        K_inv = np.linalg.inv(K).astype(np.float32)
        M_square = (K_inv @ M @ K).astype(np.float32)

        np.testing.assert_allclose(M_rect, M_square, atol=1e-5)


# ============================================================================
# 2. Full program encryption (encrypt_program_full / decrypt_state_full)
# ============================================================================

class TestEncryptProgramFull:
    """End-to-end tests with encrypt_program_full and decrypt_state_full."""

    def test_encrypt_program_full_square(self):
        """Encrypt a 3-state machine program, run it, decrypt, verify result."""
        # 3-state machine: IDLE -> RUNNING -> DONE
        M = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.float32)

        program = Program(
            name="state_machine",
            initial_state=StateBundle(vectors={
                "state": OneHotVec.from_index(0, 3),
            }),
            transitions=[
                Transition(
                    name="step1",
                    operation=TransformOp(key="state", matrix=TransMatrix(data=M)),
                ),
                Transition(
                    name="step2",
                    operation=TransformOp(key="state", matrix=TransMatrix(data=M)),
                ),
            ],
        )

        # Run plaintext
        plain_result = run_program(program)
        plain_state = plain_result.final_state["state"].data

        # Encrypt, run, decrypt
        kf = KeyFamily(seed=12345)
        enc_program, dim_map = encrypt_program_full(program, kf)
        enc_result = run_program(enc_program)
        dec_state = decrypt_state_full(enc_result.final_state, kf, dim_map)

        np.testing.assert_allclose(
            dec_state["state"].data, plain_state, atol=1e-4,
        )
        # Should be in DONE state [0, 0, 1]
        np.testing.assert_allclose(
            dec_state["state"].data, [0, 0, 1], atol=1e-4,
        )

    def test_encrypt_program_full_pipeline(self):
        """Encrypt a pipeline of two transforms and verify the result."""
        M1 = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]], dtype=np.float32)
        M2 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.float32)

        program = Program(
            name="pipeline",
            initial_state=StateBundle(vectors={
                "v": FloatVec.from_list([1.0, 1.0, 1.0]),
            }),
            transitions=[
                Transition(
                    name="scale",
                    operation=TransformOp(key="v", matrix=TransMatrix(data=M1)),
                ),
                Transition(
                    name="mix",
                    operation=TransformOp(key="v", matrix=TransMatrix(data=M2)),
                ),
            ],
        )

        plain_result = run_program(program)
        plain_v = plain_result.final_state["v"].data

        kf = KeyFamily(seed=555)
        enc_program, dim_map = encrypt_program_full(program, kf)
        enc_result = run_program(enc_program)
        dec_state = decrypt_state_full(enc_result.final_state, kf, dim_map)

        np.testing.assert_allclose(dec_state["v"].data, plain_v, atol=1e-4)


# ============================================================================
# 3. dim_map and decrypt_state_full
# ============================================================================

class TestDimMapAndDecrypt:
    """Verify dim_map tracking and decrypt_state_full correctness."""

    def test_dim_map_tracking(self):
        """Verify dim_map records correct dimensions for all initial state keys."""
        program = Program(
            name="multi_dim",
            initial_state=StateBundle(vectors={
                "a": FloatVec.from_list([1.0, 2.0, 3.0]),
                "b": FloatVec.from_list([4.0, 5.0, 6.0, 7.0, 8.0]),
            }),
            transitions=[],
        )

        kf = KeyFamily(seed=101)
        _, dim_map = encrypt_program_full(program, kf)

        assert dim_map["a"] == 3
        assert dim_map["b"] == 5

    def test_decrypt_state_full_basic(self):
        """Decrypt individual vectors with correct dim_map."""
        kf = KeyFamily(seed=42)
        dim_map = {"x": 3, "y": 5}

        # Manually encrypt vectors
        v_x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v_y = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        v_x_enc = encrypt_vec(v_x, kf.key(3))
        v_y_enc = encrypt_vec(v_y, kf.key(5))

        bundle = StateBundle(vectors={
            "x": FloatVec(data=v_x_enc),
            "y": FloatVec(data=v_y_enc),
        })

        dec = decrypt_state_full(bundle, kf, dim_map)

        np.testing.assert_allclose(dec["x"].data, v_x, atol=1e-4)
        np.testing.assert_allclose(dec["y"].data, v_y, atol=1e-4)


# ============================================================================
# 4. Chained rectangular transforms
# ============================================================================

class TestRectTransformChain:
    """Chain rectangular transforms with dimension changes."""

    def test_rect_transform_chain(self):
        """Chain 3 -> 5 -> 3 transforms, encrypt/run/decrypt."""
        kf = KeyFamily(seed=2024)

        # M1: 3->5,  M2: 5->3
        M1 = np.random.RandomState(10).randn(3, 5).astype(np.float32)
        M2 = np.random.RandomState(20).randn(5, 3).astype(np.float32)

        v = np.array([1.0, 0.0, -1.0], dtype=np.float32)

        # Plaintext chain
        r1 = v @ M1          # shape (5,)
        r2 = r1 @ M2         # shape (3,)

        # Build program with out_key to track dimension changes
        program = Program(
            name="chain_3_5_3",
            initial_state=StateBundle(vectors={
                "v": FloatVec(data=v),
            }),
            transitions=[
                Transition(
                    name="expand",
                    operation=TransformOp(
                        key="v",
                        matrix=TransMatrix(data=M1),
                        out_key="mid",
                    ),
                ),
                Transition(
                    name="contract",
                    operation=TransformOp(
                        key="mid",
                        matrix=TransMatrix(data=M2),
                        out_key="out",
                    ),
                ),
            ],
        )

        plain_result = run_program(program)
        plain_out = plain_result.final_state["out"].data
        np.testing.assert_allclose(plain_out, r2, atol=1e-4)

        # Encrypt, run, decrypt
        enc_program, dim_map = encrypt_program_full(program, kf)

        assert dim_map["v"] == 3
        assert dim_map["mid"] == 5
        assert dim_map["out"] == 3

        enc_result = run_program(enc_program)
        dec_state = decrypt_state_full(enc_result.final_state, kf, dim_map)

        np.testing.assert_allclose(dec_state["out"].data, r2, atol=1e-4)


# ============================================================================
# 5. Random and dense matrices
# ============================================================================

class TestEncryptedDense:
    """Test with random dense matrices."""

    def test_encrypted_dense_random_matrix(self):
        """Random 4x6 dense matrix: encrypt/compute/decrypt round-trip."""
        kf = KeyFamily(seed=314)
        rng = np.random.RandomState(159)

        M = rng.randn(4, 6).astype(np.float32)
        v = rng.randn(4).astype(np.float32)

        result_plain = v @ M

        K4 = kf.key(4)
        K6_inv = kf.inv(6)
        v_enc = (v @ K4).astype(np.float32)
        M_enc = encrypt_matrix_rect(M, kf)
        result_enc = v_enc @ M_enc
        result_dec = (result_enc @ K6_inv).astype(np.float32)

        np.testing.assert_allclose(result_dec, result_plain, atol=1e-4)


# ============================================================================
# 6. Computation preservation
# ============================================================================

class TestComputationPreservation:
    """Verify that encryption preserves the algebraic identity:
    v @ M  ==  decrypt(encrypt(v) @ encrypt(M))
    """

    def test_encrypt_preserves_computation(self):
        """v @ M in plaintext == decrypt(encrypt(v) @ encrypt(M))
        for a rectangular 3x5 case."""
        kf = KeyFamily(seed=7)

        v = np.array([3.0, -1.0, 2.0], dtype=np.float32)
        M = np.array([
            [1.0, 0.5, 0.0, -0.5, 1.0],
            [0.0, 1.0, 0.5, 0.0, -1.0],
            [-1.0, 0.0, 1.0, 0.5, 0.0],
        ], dtype=np.float32)

        # Plaintext
        result_plain = v @ M

        # Encrypted
        v_enc = encrypt_vec(v, kf.key(3))
        M_enc = encrypt_matrix_rect(M, kf)
        result_enc = v_enc @ M_enc
        result_dec = decrypt_vec(result_enc, kf.key(5))

        np.testing.assert_allclose(result_dec, result_plain, atol=1e-4)


# ============================================================================
# 7. Key independence and security properties
# ============================================================================

class TestKeyProperties:
    """Verify security-related properties of KeyFamily keys."""

    def test_different_keys_different_results(self):
        """Different seeds must produce different encrypted matrices."""
        M = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float32)

        kf1 = KeyFamily(seed=100)
        kf2 = KeyFamily(seed=200)

        M_enc1 = encrypt_matrix_rect(M, kf1)
        M_enc2 = encrypt_matrix_rect(M, kf2)

        assert not np.allclose(M_enc1, M_enc2, atol=0.01), (
            "Different seeds must produce different encrypted matrices"
        )

    def test_dim_independence(self):
        """Key for dim=3 must be independent from key for dim=5.
        The correlation between flattened keys should be near zero."""
        kf = KeyFamily(seed=42)
        K3 = kf.key(3)  # 3x3
        K5 = kf.key(5)  # 5x5

        # Flatten and take the first min(9, 25)=9 elements from each
        flat3 = K3.flatten()
        flat5 = K5.flatten()[:len(flat3)]

        # Pearson correlation should be close to 0 (independent draws)
        corr = np.corrcoef(flat3, flat5)[0, 1]
        assert abs(corr) < 0.8, (
            f"Keys for dim=3 and dim=5 should be independent, got correlation={corr:.4f}"
        )

        # Additionally, verify they are derived from different seeds
        assert kf._derived_seed(3) != kf._derived_seed(5), (
            "Derived seeds for different dimensions must differ"
        )
