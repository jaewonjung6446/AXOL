"""Tests for axol.api.client — AxolClient SDK."""

import numpy as np
import pytest

from axol.api.client import AxolClient, serialize_program, deserialize_program
from axol.core.types import FloatVec, OneHotVec, TransMatrix, StateBundle
from axol.core.program import Program, Transition, TransformOp, run_program
from axol.core.encryption import KeyFamily


def _make_3state_program() -> Program:
    """Build a 3-state shift machine: 0 -> 1 -> 2 -> 2 (absorbing)."""
    return Program(
        name="test",
        initial_state=StateBundle(vectors={
            "state": OneHotVec.from_index(0, 3),
        }),
        transitions=[
            Transition(
                name="t",
                operation=TransformOp(
                    key="state",
                    matrix=TransMatrix.from_list([
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 1],
                    ]),
                ),
            ),
        ],
    )


def _plain_result(prog: Program) -> np.ndarray:
    """Run a program in plaintext and return the 'state' vector as numpy array."""
    result = run_program(prog)
    return result.final_state["state"].data


class TestClientPrepare:
    def test_client_prepare_returns_metadata(self):
        """prepare() returns a dict with 'program' and 'metadata' keys."""
        client = AxolClient(seed=42, max_dim=32, use_padding=True)
        prog = _make_3state_program()

        payload = client.prepare(prog)

        assert isinstance(payload, dict)
        assert "program" in payload
        assert "metadata" in payload
        assert isinstance(payload["program"], dict)
        assert isinstance(payload["metadata"], dict)


class TestClientRunLocal:
    def test_client_run_local_simple(self):
        """run_local on a 3-state machine matches plain run."""
        client = AxolClient(seed=99, max_dim=8, use_padding=True)
        prog = _make_3state_program()

        expected = _plain_result(prog)
        decrypted = client.run_local(prog)

        np.testing.assert_allclose(
            decrypted["state"].data, expected, atol=1e-3,
        )

    def test_client_run_local_padded(self):
        """use_padding=True: decrypted result matches plaintext."""
        client = AxolClient(seed=7, max_dim=16, use_padding=True)
        prog = _make_3state_program()

        expected = _plain_result(prog)
        decrypted = client.run_local(prog)

        np.testing.assert_allclose(
            decrypted["state"].data, expected, atol=1e-3,
        )

    def test_client_run_local_no_padding(self):
        """use_padding=False: decrypted result matches plaintext."""
        client = AxolClient(seed=7, max_dim=16, use_padding=False)
        prog = _make_3state_program()

        expected = _plain_result(prog)
        decrypted = client.run_local(prog)

        np.testing.assert_allclose(
            decrypted["state"].data, expected, atol=1e-3,
        )


class TestSerializeDeserialize:
    def test_serialize_deserialize_roundtrip(self):
        """serialize then deserialize produces an equivalent program."""
        prog = _make_3state_program()

        data = serialize_program(prog)
        restored = deserialize_program(data)

        # Program metadata
        assert restored.name == prog.name
        assert len(restored.transitions) == len(prog.transitions)

        # Initial state vectors
        for key in prog.initial_state.keys():
            np.testing.assert_allclose(
                restored.initial_state[key].data,
                prog.initial_state[key].data,
                atol=1e-6,
            )

        # Transition matrix
        orig_matrix = prog.transitions[0].operation.matrix.data
        rest_matrix = restored.transitions[0].operation.matrix.data
        np.testing.assert_allclose(rest_matrix, orig_matrix, atol=1e-6)

        # Execution produces same result
        expected = _plain_result(prog)
        actual = _plain_result(restored)
        np.testing.assert_allclose(actual, expected, atol=1e-6)


class TestClientDecryptResult:
    def test_client_decrypt_result(self):
        """prepare, run encrypted program, decrypt matches plain result."""
        client = AxolClient(seed=123, max_dim=8, use_padding=True)
        prog = _make_3state_program()

        # Encrypt and get metadata
        payload = client.prepare(prog)
        metadata = payload["metadata"]

        # Run the encrypted program
        enc_prog = deserialize_program(payload["program"])
        enc_result = run_program(enc_prog)

        # Decrypt
        decrypted = client.decrypt_result(enc_result.final_state, metadata)

        # Compare to plain
        expected = _plain_result(prog)
        np.testing.assert_allclose(
            decrypted["state"].data, expected, atol=1e-3,
        )


class TestClientDifferentSeeds:
    def test_client_different_seeds(self):
        """Different seeds produce different encrypted programs."""
        prog = _make_3state_program()

        client_a = AxolClient(seed=1, max_dim=8, use_padding=False)
        client_b = AxolClient(seed=2, max_dim=8, use_padding=False)

        payload_a = client_a.prepare(prog)
        payload_b = client_b.prepare(prog)

        # The encrypted initial state vectors should differ
        state_a = np.array(
            payload_a["program"]["initial_state"]["state"]["data"],
            dtype=np.float32,
        )
        state_b = np.array(
            payload_b["program"]["initial_state"]["state"]["data"],
            dtype=np.float32,
        )

        assert not np.allclose(state_a, state_b, atol=1e-6), (
            "Encrypted states from different seeds should differ"
        )


class TestClientKeyFamily:
    def test_client_key_family(self):
        """client.key_family returns a valid KeyFamily instance."""
        client = AxolClient(seed=42, max_dim=32)

        kf = client.key_family
        assert isinstance(kf, KeyFamily)
        assert kf.seed == 42

        # Derived key is orthogonal: K @ K^T ~ I
        K = kf.key(4)
        assert K.shape == (4, 4)
        np.testing.assert_allclose(
            K @ K.T, np.eye(4, dtype=np.float32), atol=1e-5,
        )
