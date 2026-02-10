"""Tests for axol.core.dsl — DSL parser unit + integration tests."""

import textwrap

import pytest
import numpy as np

from axol.core.dsl import parse, ParseError
from axol.core.types import FloatVec, GateVec, OneHotVec, TransMatrix, StateBundle
from axol.core.program import (
    TransformOp, GateOp, MergeOp, DistanceOp, RouteOp, CustomOp,
    Transition, Program, run_program,
)


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests — individual grammar elements
# ═══════════════════════════════════════════════════════════════════════════

class TestParseHeader:
    def test_simple_name(self):
        prog = parse("@my_prog\ns v=[1]\n: t=transform(v;M=[1])")
        assert prog.name == "my_prog"

    def test_alphanumeric_name(self):
        prog = parse("@test123\ns v=[1]\n: t=transform(v;M=[1])")
        assert prog.name == "test123"

    def test_missing_header(self):
        with pytest.raises(ParseError, match="must start with"):
            parse("s v=[1]\n: t=transform(v;M=[1])")


class TestParseState:
    def test_single_vector(self):
        prog = parse("@p\ns hp=[100]\n: t=transform(hp;M=[0.8])")
        assert prog.initial_state["hp"].to_list() == pytest.approx([100.0])

    def test_multiple_vectors_one_line(self):
        prog = parse("@p\ns hp=[100] round=[0] one=[1]\n: t=transform(hp;M=[1])")
        assert "hp" in prog.initial_state
        assert "round" in prog.initial_state
        assert "one" in prog.initial_state

    def test_multiple_state_lines(self):
        src = "@p\ns hp=[100]\ns round=[0]\n: t=transform(hp;M=[1])"
        prog = parse(src)
        assert prog.initial_state["hp"].to_list() == pytest.approx([100.0])
        assert prog.initial_state["round"].to_list() == pytest.approx([0.0])

    def test_multi_element_vector(self):
        prog = parse("@p\ns v=[1 2 3 4.5]\n: t=transform(v;M=[1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1])")
        assert prog.initial_state["v"].to_list() == pytest.approx([1.0, 2.0, 3.0, 4.5])

    def test_onehot_value(self):
        prog = parse("@p\ns state=onehot(0,3)\n: t=transform(state;M=[1 0 0;0 1 0;0 0 1])")
        vec = prog.initial_state["state"]
        assert isinstance(vec, OneHotVec)
        assert vec.active_index == 0
        assert vec.size == 3

    def test_zeros_value(self):
        prog = parse("@p\ns v=zeros(5)\n: t=transform(v;M=[1 0 0 0 0;0 1 0 0 0;0 0 1 0 0;0 0 0 1 0;0 0 0 0 1])")
        assert prog.initial_state["v"].to_list() == pytest.approx([0.0] * 5)

    def test_ones_value(self):
        prog = parse("@p\ns v=ones(3)\n: t=transform(v;M=[1 0 0;0 1 0;0 0 1])")
        assert prog.initial_state["v"].to_list() == pytest.approx([1.0, 1.0, 1.0])

    def test_no_state_raises(self):
        with pytest.raises(ParseError, match="at least one state"):
            parse("@p\n: t=transform(v;M=[1])")


class TestParseTransition:
    def test_transform(self):
        prog = parse("@p\ns v=[1 2]\n: scale=transform(v;M=[2 0;0 3])")
        t = prog.transitions[0]
        assert t.name == "scale"
        assert isinstance(t.operation, TransformOp)
        assert t.operation.key == "v"
        assert t.operation.matrix.shape == (2, 2)

    def test_transform_with_out_key(self):
        prog = parse("@p\ns v=[1]\n: t=transform(v;M=[2])->result")
        t = prog.transitions[0]
        assert t.operation.out_key == "result"

    def test_gate(self):
        prog = parse("@p\ns v=[1 2] g=[1 0]\n: mask=gate(v;g=g)")
        t = prog.transitions[0]
        assert isinstance(t.operation, GateOp)
        assert t.operation.key == "v"
        assert t.operation.gate_key == "g"

    def test_merge(self):
        prog = parse("@p\ns a=[1] b=[2]\n: sum=merge(a b;w=[1 1])->out")
        t = prog.transitions[0]
        assert isinstance(t.operation, MergeOp)
        assert t.operation.keys == ["a", "b"]
        assert t.operation.weights.to_list() == pytest.approx([1.0, 1.0])
        assert t.operation.out_key == "out"

    def test_distance(self):
        prog = parse("@p\ns a=[1 0] b=[0 1]\n: d=distance(a b)")
        t = prog.transitions[0]
        assert isinstance(t.operation, DistanceOp)
        assert t.operation.key_a == "a"
        assert t.operation.key_b == "b"
        assert t.operation.metric == "euclidean"

    def test_distance_with_metric(self):
        prog = parse("@p\ns a=[1 0] b=[0 1]\n: d=distance(a b;metric=cosine)")
        t = prog.transitions[0]
        assert t.operation.metric == "cosine"

    def test_route(self):
        prog = parse("@p\ns v=[1 0]\n: r=route(v;R=[1 0;0 1])")
        t = prog.transitions[0]
        assert isinstance(t.operation, RouteOp)
        assert t.operation.key == "v"

    def test_unknown_op_raises(self):
        with pytest.raises(ParseError, match="Unknown operation"):
            parse("@p\ns v=[1]\n: t=foobar(v;M=[1])")

    def test_no_transition_raises(self):
        with pytest.raises(ParseError, match="at least one transition"):
            parse("@p\ns v=[1]")


class TestParseMatrix:
    def test_dense_1x1(self):
        prog = parse("@p\ns v=[1]\n: t=transform(v;M=[0.8])")
        mat = prog.transitions[0].operation.matrix
        assert mat.shape == (1, 1)
        assert float(mat.data[0, 0]) == pytest.approx(0.8)

    def test_dense_multirow(self):
        prog = parse("@p\ns v=[1 0 0]\n: t=transform(v;M=[0 1 0;0 0 1;0 0 1])")
        mat = prog.transitions[0].operation.matrix
        assert mat.shape == (3, 3)
        assert float(mat.data[0, 1]) == pytest.approx(1.0)

    def test_sparse_matrix(self):
        src = "@p\ns v=onehot(0,4)\n: t=transform(v;M=sparse(4x4;0,1=1 1,2=1 2,3=1 3,3=1))"
        prog = parse(src)
        mat = prog.transitions[0].operation.matrix
        assert mat.shape == (4, 4)
        assert float(mat.data[0, 1]) == pytest.approx(1.0)
        assert float(mat.data[3, 3]) == pytest.approx(1.0)
        assert float(mat.data[0, 0]) == pytest.approx(0.0)


class TestParseTerminal:
    def test_simple_condition(self):
        src = "@p\ns count=[0] one=[1]\n: inc=merge(count one;w=[1 1])->count\n? done count>=5"
        prog = parse(src)
        assert prog.terminal_key == "_done"

    def test_indexed_condition(self):
        src = "@p\ns state=[1 0 0]\n: t=transform(state;M=[0 1 0;0 0 1;0 0 1])\n? done state[2]>=1"
        prog = parse(src)
        assert prog.terminal_key == "_done"

    def test_terminal_gate_auto_created(self):
        src = "@p\ns v=[0]\n: t=transform(v;M=[1])\n? end v>=1"
        prog = parse(src)
        assert "_end" in prog.initial_state

    def test_pipeline_mode_when_no_terminal(self):
        prog = parse("@p\ns v=[1]\n: t=transform(v;M=[2])")
        assert prog.terminal_key is None


class TestComments:
    def test_comment_lines_ignored(self):
        src = "# This is a comment\n@p\n# another\ns v=[1]\n: t=transform(v;M=[2])"
        prog = parse(src)
        assert prog.name == "p"


# ═══════════════════════════════════════════════════════════════════════════
# Integration tests — full DSL → parse → run → verify
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrationCounter:
    """Counter: 0 → 5 via merge increment."""

    DSL = textwrap.dedent("""\
        @counter
        s count=[0] one=[1]
        : increment=merge(count one;w=[1 1])->count
        ? done count>=5
    """)

    def test_parse_and_run(self):
        prog = parse(self.DSL)
        result = run_program(prog)
        assert result.terminated_by == "terminal_condition"
        final_count = float(result.final_state["count"].data[0])
        assert final_count == pytest.approx(5.0)


class TestIntegrationStateMachine:
    """State Machine: IDLE(0) → RUNNING(1) → DONE(2) via shift matrix."""

    DSL = textwrap.dedent("""\
        @state_machine
        s state=onehot(0,3)
        : advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
        ? done state[2]>=1
    """)

    def test_parse_and_run(self):
        prog = parse(self.DSL)
        result = run_program(prog)
        assert result.terminated_by == "terminal_condition"
        final = result.final_state["state"].to_list()
        assert final == pytest.approx([0.0, 0.0, 1.0])


class TestIntegrationHpDecay:
    """HP Decay: 100 × 0.8^3 = 51.2."""

    DSL = textwrap.dedent("""\
        @hp_decay
        s hp=[100] round=[0] one=[1]
        : decay=transform(hp;M=[0.8])
        : tick=merge(round one;w=[1 1])->round
        ? done round>=3
    """)

    def test_parse_and_run(self):
        prog = parse(self.DSL)
        result = run_program(prog)
        assert result.terminated_by == "terminal_condition"
        final_hp = float(result.final_state["hp"].data[0])
        assert final_hp == pytest.approx(51.2, abs=0.1)


class TestIntegrationPipeline:
    """Pipeline mode (no terminal): scale + gate + merge."""

    DSL = textwrap.dedent("""\
        @pipeline_test
        s atk=[50] def_val=[20] flag=[1]
        : scale=transform(atk;M=[1.5])->scaled
        : block=gate(def_val;g=flag)
        : combine=merge(scaled def_val;w=[1 -1])->damage
    """)

    def test_parse_and_run(self):
        prog = parse(self.DSL)
        assert prog.terminal_key is None  # pipeline mode
        result = run_program(prog)
        assert result.terminated_by == "pipeline_end"
        # scaled = 50 * 1.5 = 75, def_val gated by [1] = 20, damage = 75 - 20 = 55
        assert "damage" in result.final_state
        dmg = float(result.final_state["damage"].data[0])
        assert dmg == pytest.approx(55.0)


class TestIntegrationLargeAutomaton:
    """100-state automaton using sparse matrix."""

    @staticmethod
    def _gen_dsl(n=100):
        entries = " ".join(f"{i},{i+1}=1" for i in range(n - 1))
        entries += f" {n-1},{n-1}=1"
        return "\n".join([
            f"@auto_{n}",
            f"s s=onehot(0,{n})",
            f": step=transform(s;M=sparse({n}x{n};{entries}))",
            f"? done s[{n-1}]>=1",
        ])

    def test_parse_and_run_small(self):
        """5-state version for fast test."""
        prog = parse(self._gen_dsl(5))
        result = run_program(prog)
        assert result.terminated_by == "terminal_condition"
        final = result.final_state["s"].to_list()
        assert final[4] == pytest.approx(1.0)
        assert sum(final[:4]) == pytest.approx(0.0)

    def test_parse_100_state(self):
        """100-state: just verify it parses and the matrix is correct."""
        prog = parse(self._gen_dsl(100))
        mat = prog.transitions[0].operation.matrix
        assert mat.shape == (100, 100)
        # Check shift structure
        assert float(mat.data[0, 1]) == pytest.approx(1.0)
        assert float(mat.data[98, 99]) == pytest.approx(1.0)
        assert float(mat.data[99, 99]) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Error cases
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorCases:
    def test_empty_program(self):
        with pytest.raises(ParseError, match="Empty program"):
            parse("")

    def test_whitespace_only(self):
        with pytest.raises(ParseError, match="Empty program"):
            parse("   \n  \n  ")

    def test_invalid_header_name(self):
        with pytest.raises(ParseError, match="Invalid program name"):
            parse("@123bad\ns v=[1]\n: t=transform(v;M=[1])")

    def test_unmatched_paren(self):
        with pytest.raises(ParseError, match="Unmatched parenthesis"):
            parse("@p\ns v=[1]\n: t=transform(v;M=[1]")

    def test_missing_matrix_param(self):
        with pytest.raises(ParseError, match="transform requires M="):
            parse("@p\ns v=[1]\n: t=transform(v)")

    def test_missing_gate_param(self):
        with pytest.raises(ParseError, match="gate requires g="):
            parse("@p\ns v=[1]\n: t=gate(v)")

    def test_merge_missing_out_key(self):
        with pytest.raises(ParseError, match="merge requires ->out_key"):
            parse("@p\ns a=[1] b=[1]\n: t=merge(a b;w=[1 1])")

    def test_merge_missing_weights(self):
        with pytest.raises(ParseError, match="merge requires w="):
            parse("@p\ns a=[1] b=[1]\n: t=merge(a b)->out")

    def test_invalid_sparse_entry(self):
        with pytest.raises(ParseError, match="Invalid sparse entry"):
            parse("@p\ns v=onehot(0,3)\n: t=transform(v;M=sparse(3x3;badentry))")

    def test_invalid_condition(self):
        with pytest.raises(ParseError, match="Invalid condition"):
            parse("@p\ns v=[1]\n: t=transform(v;M=[1])\n? done v~=1")
