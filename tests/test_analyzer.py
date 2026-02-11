"""Tests for axol.core.analyzer — encryption coverage analysis."""

import pytest

from axol.core.types import FloatVec, GateVec, TransMatrix, StateBundle
from axol.core.program import (
    Program,
    Transition,
    TransformOp,
    GateOp,
    StepOp,
    BranchOp,
    ClampOp,
    MapOp,
    SecurityLevel,
)
from axol.core.analyzer import analyze, AnalysisResult


def _make_program(name: str, transitions: list[Transition]) -> Program:
    """Helper to build a minimal Program for analysis."""
    state = StateBundle(vectors={"x": FloatVec.from_list([1.0, 2.0])})
    return Program(name=name, initial_state=state, transitions=transitions)


class TestFullEncrypted:
    def test_100_percent_coverage(self):
        m = TransMatrix.identity(2)
        ts = [
            Transition(name="t1", operation=TransformOp(key="x", matrix=m)),
            Transition(name="t2", operation=GateOp(key="x", gate_key="g")),
        ]
        result = analyze(_make_program("all_e", ts))
        assert result.encrypted_count == 2
        assert result.plaintext_count == 0
        assert result.coverage_pct == pytest.approx(100.0)
        assert "g" in result.encryptable_keys
        assert len(result.plaintext_keys) == 0


class TestMixedCoverage:
    def test_partial_coverage(self):
        m = TransMatrix.identity(2)
        ts = [
            Transition(name="t1", operation=TransformOp(key="x", matrix=m)),
            Transition(name="s1", operation=StepOp(key="x")),
            Transition(name="c1", operation=ClampOp(key="y", min_val=0.0)),
        ]
        result = analyze(_make_program("mixed", ts))
        assert result.total_transitions == 3
        assert result.encrypted_count == 1
        assert result.plaintext_count == 2
        assert result.coverage_pct == pytest.approx(100.0 / 3.0)
        # "x" is accessed by both E (TransformOp) and P (StepOp) → plaintext
        assert "x" in result.plaintext_keys
        # "y" is only accessed by P → plaintext
        assert "y" in result.plaintext_keys

    def test_plaintext_keys_identified(self):
        m = TransMatrix.identity(2)
        ts = [
            Transition(name="t1", operation=TransformOp(key="a", matrix=m, out_key="b")),
            Transition(name="m1", operation=MapOp(key="c", fn_name="relu", out_key="d")),
        ]
        result = analyze(_make_program("keys", ts))
        # a, b accessed only by E → encryptable
        assert "a" in result.encryptable_keys
        assert "b" in result.encryptable_keys
        # c, d accessed only by P → plaintext
        assert "c" in result.plaintext_keys
        assert "d" in result.plaintext_keys


class TestFullPlaintext:
    def test_0_percent_coverage(self):
        ts = [
            Transition(name="s1", operation=StepOp(key="x")),
            Transition(name="c1", operation=ClampOp(key="x")),
            Transition(name="m1", operation=MapOp(key="x", fn_name="relu")),
        ]
        result = analyze(_make_program("all_p", ts))
        assert result.encrypted_count == 0
        assert result.plaintext_count == 3
        assert result.coverage_pct == pytest.approx(0.0)
        assert len(result.encryptable_keys) == 0
        assert "x" in result.plaintext_keys


class TestSummary:
    def test_summary_output(self):
        m = TransMatrix.identity(2)
        ts = [
            Transition(name="t1", operation=TransformOp(key="x", matrix=m)),
            Transition(name="s1", operation=StepOp(key="x")),
        ]
        result = analyze(_make_program("demo", ts))
        s = result.summary()
        assert "Program: demo" in s
        assert "2 total" in s
        assert "1 encrypted" in s
        assert "1 plaintext" in s
        assert "50.0%" in s


class TestBranchOpAnalysis:
    def test_branch_keys(self):
        ts = [
            Transition(
                name="b1",
                operation=BranchOp(
                    gate_key="g", then_key="a", else_key="b", out_key="c"
                ),
            ),
        ]
        result = analyze(_make_program("branch_test", ts))
        info = result.transitions[0]
        assert info.read_keys == {"g", "a", "b"}
        assert info.write_keys == {"c"}
        assert info.security == SecurityLevel.PLAINTEXT


class TestEmptyProgram:
    def test_no_transitions(self):
        result = analyze(_make_program("empty", []))
        assert result.total_transitions == 0
        assert result.coverage_pct == pytest.approx(0.0)
        assert len(result.encryptable_keys) == 0
        assert len(result.plaintext_keys) == 0
