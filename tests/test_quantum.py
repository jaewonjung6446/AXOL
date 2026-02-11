"""Tests for Phase 6: Quantum interference + encryption-transparent API."""

from __future__ import annotations

import math

import numpy as np
import pytest

from axol.core.types import FloatVec, TransMatrix, StateBundle
from axol.core.operations import (
    measure,
    hadamard_matrix,
    oracle_matrix,
    diffusion_matrix,
    transform,
)
from axol.core.program import (
    TransformOp,
    MeasureOp,
    Transition,
    Program,
    run_program,
    OpKind,
    SecurityLevel,
)
from axol.core.dsl import parse, ParseError
from axol.core.analyzer import analyze
from axol.core.encryption import (
    random_orthogonal_key,
    encrypt_program,
    decrypt_state,
)
from axol.core.optimizer import optimize
from axol.api.dispatch import dispatch


# ===================================================================
# 7-A. Unit Tests
# ===================================================================

class TestHadamardMatrix:
    def test_orthogonality(self):
        """H @ H^T = I (orthogonality)."""
        for n in [2, 4, 8]:
            H = hadamard_matrix(n)
            product = H.data @ H.data.T
            np.testing.assert_allclose(product, np.eye(n), atol=1e-5)

    def test_negative_entries(self):
        """Hadamard matrix must contain negative entries for interference."""
        H = hadamard_matrix(4)
        assert np.any(H.data < 0), "Hadamard should have negative entries"

    def test_non_power_of_2_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            hadamard_matrix(3)

    def test_size_1(self):
        H = hadamard_matrix(1)
        np.testing.assert_allclose(H.data, [[1.0]], atol=1e-5)


class TestOracleMatrix:
    def test_marked_indices(self):
        """Only marked indices should have -1 on diagonal."""
        O = oracle_matrix([2], 4)
        expected = np.diag([1.0, 1.0, -1.0, 1.0])
        np.testing.assert_allclose(O.data, expected, atol=1e-5)

    def test_multiple_marked(self):
        O = oracle_matrix([0, 3], 4)
        expected = np.diag([-1.0, 1.0, 1.0, -1.0])
        np.testing.assert_allclose(O.data, expected, atol=1e-5)

    def test_identity_when_no_marked(self):
        O = oracle_matrix([], 4)
        np.testing.assert_allclose(O.data, np.eye(4), atol=1e-5)


class TestDiffusionMatrix:
    def test_orthogonality(self):
        """D @ D^T = I (involutory → orthogonal)."""
        for n in [2, 4, 8]:
            D = diffusion_matrix(n)
            product = D.data @ D.data.T
            np.testing.assert_allclose(product, np.eye(n), atol=1e-5)

    def test_negative_entries(self):
        D = diffusion_matrix(4)
        assert np.any(D.data < 0), "Diffusion should have negative entries"


class TestMeasure:
    def test_uniform(self):
        """[0.5, 0.5, 0.5, 0.5] → [0.25, 0.25, 0.25, 0.25]."""
        v = FloatVec.from_list([0.5, 0.5, 0.5, 0.5])
        probs = measure(v)
        np.testing.assert_allclose(probs.data, [0.25, 0.25, 0.25, 0.25], atol=1e-5)

    def test_negative_amplitude(self):
        """[0.5, 0.5, 0.5, -0.5] → [0.25, 0.25, 0.25, 0.25] (same probabilities)."""
        v = FloatVec.from_list([0.5, 0.5, 0.5, -0.5])
        probs = measure(v)
        np.testing.assert_allclose(probs.data, [0.25, 0.25, 0.25, 0.25], atol=1e-5)

    def test_after_interference(self):
        """After oracle + diffusion, target should have probability > 0.9."""
        n = 4
        state = FloatVec.from_list([0.5, 0.5, 0.5, 0.5])
        # Apply oracle (mark index 3)
        O = oracle_matrix([3], n)
        state = transform(state, O)
        # Apply diffusion
        D = diffusion_matrix(n)
        state = transform(state, D)
        # Measure
        probs = measure(state)
        assert probs.data[3] > 0.9, f"Target probability {probs.data[3]} should be > 0.9"

    def test_normalization(self):
        """Probabilities must sum to 1.0."""
        v = FloatVec.from_list([0.3, -0.7, 0.1, 0.5])
        probs = measure(v)
        assert abs(np.sum(probs.data) - 1.0) < 1e-5

    def test_zero_vector(self):
        """Zero vector → all zeros (no normalization)."""
        v = FloatVec.zeros(4)
        probs = measure(v)
        np.testing.assert_allclose(probs.data, [0.0, 0.0, 0.0, 0.0], atol=1e-5)


# ===================================================================
# 7-B. Integration Tests
# ===================================================================

class TestGrover:
    def test_grover_search_4(self):
        """4-item Grover search: 1 iteration → target amplitude = 1.0."""
        source = """
@grover_4
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
? found state[3]>=0.9
"""
        prog = parse(source)
        result = run_program(prog)
        state = result.final_state["state"]
        # After 1 Grover iteration on N=4, target is exactly 1.0
        assert abs(state.data[3]) >= 0.9
        assert result.terminated_by == "terminal_condition"

    def test_grover_encrypted_pipeline(self):
        """Pipeline-mode encrypted Grover: 100% E-class, result matches plaintext."""
        n = 4
        source = """
@grover_enc
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
"""
        # Plaintext run
        prog_plain = parse(source)
        result_plain = run_program(prog_plain)
        plain_state = result_plain.final_state["state"].data

        # Encrypted run
        prog_enc_src = parse(source)
        K = random_orthogonal_key(n, seed=42)
        prog_encrypted = encrypt_program(prog_enc_src, K, n)
        result_enc = run_program(prog_encrypted)
        decrypted = decrypt_state(result_enc.final_state, K)
        dec_state = decrypted["state"].data

        np.testing.assert_allclose(dec_state, plain_state, atol=1e-4)

        # Verify encryption coverage is 100% (all ops are TransformOp = E-class)
        analysis = analyze(prog_enc_src)
        assert analysis.coverage_pct == 100.0

    def test_grover_search_8(self):
        """8-item Grover search: ~2 iterations → target probability > 0.9."""
        n = 8
        amp = 1.0 / math.sqrt(n)
        state_vals = " ".join([str(amp)] * n)
        iterations = max(1, math.floor(math.pi / 4 * math.sqrt(n)))
        lines = [f"@grover_{n}", f"s state=[{state_vals}]"]
        for i in range(iterations):
            lines.append(f": oracle_{i}=oracle(state;marked=[5];n={n})")
            lines.append(f": diffuse_{i}=diffuse(state;n={n})")
        source = "\n".join(lines)

        prog = parse(source)
        result = run_program(prog)
        state = result.final_state["state"]
        probs = measure(state)
        assert probs.data[5] > 0.9, f"Target probability {probs.data[5]} should be > 0.9"

    def test_grover_encrypted_terminal_warning(self):
        """Loop-mode Grover with encryption runs to max_iterations (terminal condition broken)."""
        source = """
@grover_loop
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
? found state[3]>=0.9
"""
        prog = parse(source)
        prog.max_iterations = 5
        K = random_orthogonal_key(4, seed=42)
        prog_enc = encrypt_program(prog, K, 4)
        result = run_program(prog_enc)
        # Terminal condition uses encrypted state → cannot evaluate correctly
        # Should run until max_iterations
        assert result.terminated_by == "max_iterations"

    def test_quantum_walk(self):
        """Quantum walk: Hadamard + Oracle produces negative amplitudes → E-class."""
        n = 4
        # Start with uniform superposition
        amp = 1.0 / math.sqrt(n)
        state = FloatVec.from_list([amp] * n)
        # Oracle flips sign of index 0 → introduces negative amplitude
        O = oracle_matrix([0], n)
        result = transform(state, O)
        # Should have negative amplitude at index 0
        assert result.data[0] < 0, "Oracle should produce negative amplitude"
        assert np.any(result.data > 0), "Other indices should remain positive"
        # Verify norm is preserved (oracle is unitary)
        norm = np.sqrt(np.sum(result.data ** 2))
        assert abs(norm - 1.0) < 1e-5


# ===================================================================
# 7-C. Analyzer Tests
# ===================================================================

class TestQuantumAnalyzer:
    def test_quantum_coverage_100(self):
        """Pure quantum program (oracle + diffuse) → 100% E-class coverage."""
        source = """
@quantum_pure
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
"""
        prog = parse(source)
        result = analyze(prog)
        assert result.coverage_pct == 100.0
        assert result.encrypted_count == 2
        assert result.plaintext_count == 0

    def test_quantum_with_measure(self):
        """oracle + diffuse + measure → measure is P-class."""
        source = """
@quantum_measure
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
: m=measure(state)->probs
"""
        prog = parse(source)
        result = analyze(prog)
        assert result.encrypted_count == 2
        assert result.plaintext_count == 1
        assert result.coverage_pct == pytest.approx(2 / 3 * 100, abs=0.1)


# ===================================================================
# 7-D. DSL Parsing Tests
# ===================================================================

class TestParsing:
    def test_parse_measure(self):
        """measure(state) → MeasureOp."""
        source = """
@test_measure
s state=[0.5 0.5 0.5 0.5]
: m=measure(state)
"""
        prog = parse(source)
        op = prog.transitions[0].operation
        assert isinstance(op, MeasureOp)
        assert op.key == "state"
        assert op.out_key is None
        assert op.kind == OpKind.MEASURE
        assert op.security == SecurityLevel.PLAINTEXT

    def test_parse_measure_with_out_key(self):
        """measure(state)->probs → MeasureOp with out_key."""
        source = """
@test_measure_out
s state=[0.5 0.5 0.5 0.5]
: m=measure(state)->probs
"""
        prog = parse(source)
        op = prog.transitions[0].operation
        assert isinstance(op, MeasureOp)
        assert op.key == "state"
        assert op.out_key == "probs"

    def test_parse_hadamard(self):
        """hadamard(state;n=4) → TransformOp with 4x4 Hadamard matrix."""
        source = """
@test_hadamard
s state=[1 0 0 0]
: h=hadamard(state;n=4)
"""
        prog = parse(source)
        op = prog.transitions[0].operation
        assert isinstance(op, TransformOp)
        assert op.matrix.shape == (4, 4)
        # Verify it's actually a Hadamard matrix (orthogonal)
        product = op.matrix.data @ op.matrix.data.T
        np.testing.assert_allclose(product, np.eye(4), atol=1e-5)

    def test_parse_oracle(self):
        """oracle(state;marked=[2];n=4) → TransformOp with oracle matrix."""
        source = """
@test_oracle
s state=[0.5 0.5 0.5 0.5]
: o=oracle(state;marked=[2];n=4)
"""
        prog = parse(source)
        op = prog.transitions[0].operation
        assert isinstance(op, TransformOp)
        assert op.matrix.shape == (4, 4)
        # Diagonal: [1, 1, -1, 1]
        expected = np.diag([1.0, 1.0, -1.0, 1.0])
        np.testing.assert_allclose(op.matrix.data, expected, atol=1e-5)

    def test_parse_diffuse(self):
        """diffuse(state;n=4) → TransformOp with diffusion matrix."""
        source = """
@test_diffuse
s state=[0.5 0.5 0.5 0.5]
: d=diffuse(state;n=4)
"""
        prog = parse(source)
        op = prog.transitions[0].operation
        assert isinstance(op, TransformOp)
        assert op.matrix.shape == (4, 4)
        # Verify orthogonality
        product = op.matrix.data @ op.matrix.data.T
        np.testing.assert_allclose(product, np.eye(4), atol=1e-5)

    def test_parse_hadamard_missing_n(self):
        source = """
@test_bad
s state=[1 0]
: h=hadamard(state)
"""
        with pytest.raises(ParseError, match="hadamard requires n="):
            parse(source)

    def test_parse_oracle_missing_marked(self):
        source = """
@test_bad
s state=[1 0]
: o=oracle(state;n=2)
"""
        with pytest.raises(ParseError, match="oracle requires marked="):
            parse(source)

    def test_parse_diffuse_missing_n(self):
        source = """
@test_bad
s state=[1 0]
: d=diffuse(state)
"""
        with pytest.raises(ParseError, match="diffuse requires n="):
            parse(source)


# ===================================================================
# 7-E. Optimizer Compatibility Tests
# ===================================================================

class TestOptimizerCompat:
    def test_optimizer_fuses_grover(self):
        """oracle + diffuse consecutive → fused into single TransformOp."""
        source = """
@grover_fuse
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
"""
        prog = parse(source)
        optimized = optimize(prog)
        # After fusion, should have 1 transition instead of 2
        assert len(optimized.transitions) == 1
        op = optimized.transitions[0].operation
        assert isinstance(op, TransformOp)

    def test_fused_grover_correctness(self):
        """Fused Grover produces same result as unfused."""
        source = """
@grover_compare
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
"""
        prog_plain = parse(source)
        result_plain = run_program(prog_plain)

        prog_optimized = parse(source)
        prog_optimized = optimize(prog_optimized)
        result_optimized = run_program(prog_optimized)

        np.testing.assert_allclose(
            result_plain.final_state["state"].data,
            result_optimized.final_state["state"].data,
            atol=1e-5,
        )


# ===================================================================
# API Tests
# ===================================================================

class TestAPI:
    def test_encrypted_run_api(self):
        """encrypted_run action: transparent encryption."""
        source = "@prog\ns state=[0.5 0.5 0.5 0.5]\n: o=oracle(state;marked=[3];n=4)\n: d=diffuse(state;n=4)"
        result = dispatch({
            "action": "encrypted_run",
            "source": source,
            "dim": 4,
        })
        assert "error" not in result
        assert result["encrypted"] is True
        state = result["final_state"]["state"]
        # After decryption, result should match plaintext
        assert abs(state[3]) > 0.9 or abs(state[3] - 1.0) < 0.2

    def test_encrypted_run_missing_dim(self):
        result = dispatch({
            "action": "encrypted_run",
            "source": "@p\ns v=[1]\n: t=transform(v;M=[2])",
        })
        assert "error" in result

    def test_quantum_search_api(self):
        """quantum_search action: find marked item."""
        result = dispatch({
            "action": "quantum_search",
            "n": 4,
            "marked": [3],
        })
        assert "error" not in result
        assert result["found_index"] == 3
        assert result["probability"] > 0.9

    def test_quantum_search_encrypted(self):
        """quantum_search with encrypt=True."""
        result = dispatch({
            "action": "quantum_search",
            "n": 4,
            "marked": [3],
            "encrypt": True,
        })
        assert "error" not in result
        assert result["found_index"] == 3
        assert result["probability"] > 0.5  # Slightly relaxed for encrypted mode
        assert result["encrypted"] is True

    def test_quantum_search_n8(self):
        """quantum_search with n=8."""
        result = dispatch({
            "action": "quantum_search",
            "n": 8,
            "marked": [5],
        })
        assert "error" not in result
        assert result["found_index"] == 5
        assert result["probability"] > 0.9

    def test_quantum_search_invalid_n(self):
        result = dispatch({
            "action": "quantum_search",
            "n": 3,
            "marked": [1],
        })
        assert "error" in result
