"""Tests for 4 new plaintext operations: step, branch, clamp, map_fn."""

import math

import numpy as np
import pytest

from axol.core.types import FloatVec, GateVec, StateBundle
from axol.core.operations import step, branch, clamp, map_fn
from axol.core.program import (
    Program,
    Transition,
    StepOp,
    BranchOp,
    ClampOp,
    MapOp,
    TransformOp,
    GateOp,
    MergeOp,
    DistanceOp,
    RouteOp,
    CustomOp,
    SecurityLevel,
    run_program,
)
from axol.core.dsl import parse, ParseError


# ══════════════════════════════════════════════════════════════════════════
# Pure function tests
# ══════════════════════════════════════════════════════════════════════════


class TestStep:
    def test_default_threshold(self):
        v = FloatVec.from_list([-1.0, 0.0, 1.0, 2.0])
        g = step(v)
        assert g.to_list() == pytest.approx([0, 1, 1, 1])

    def test_custom_threshold(self):
        v = FloatVec.from_list([0.3, 0.5, 0.7, 1.0])
        g = step(v, threshold=0.5)
        assert g.to_list() == pytest.approx([0, 1, 1, 1])

    def test_returns_gate_vec(self):
        v = FloatVec.from_list([1.0, -1.0])
        g = step(v)
        assert isinstance(g, GateVec)

    def test_all_below_threshold(self):
        v = FloatVec.from_list([-5.0, -3.0, -1.0])
        g = step(v, threshold=0.0)
        assert g.to_list() == pytest.approx([0, 0, 0])


class TestBranch:
    def test_basic_select(self):
        g = GateVec.from_list([1.0, 0.0, 1.0])
        then = FloatVec.from_list([10.0, 20.0, 30.0])
        els = FloatVec.from_list([1.0, 2.0, 3.0])
        result = branch(g, then, els)
        assert result.to_list() == pytest.approx([10, 2, 30])

    def test_all_then(self):
        g = GateVec.ones(3)
        then = FloatVec.from_list([10.0, 20.0, 30.0])
        els = FloatVec.from_list([1.0, 2.0, 3.0])
        result = branch(g, then, els)
        assert result.to_list() == pytest.approx([10, 20, 30])

    def test_all_else(self):
        g = GateVec.zeros(3)
        then = FloatVec.from_list([10.0, 20.0, 30.0])
        els = FloatVec.from_list([1.0, 2.0, 3.0])
        result = branch(g, then, els)
        assert result.to_list() == pytest.approx([1, 2, 3])

    def test_dimension_mismatch_raises(self):
        g = GateVec.from_list([1.0, 0.0])
        then = FloatVec.from_list([10.0, 20.0, 30.0])
        els = FloatVec.from_list([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            branch(g, then, els)


class TestClamp:
    def test_basic_clamp(self):
        v = FloatVec.from_list([-5.0, 0.0, 5.0, 10.0])
        result = clamp(v, min_val=0.0, max_val=8.0)
        assert result.to_list() == pytest.approx([0, 0, 5, 8])

    def test_min_only(self):
        v = FloatVec.from_list([-5.0, 0.0, 5.0])
        result = clamp(v, min_val=0.0)
        assert result.to_list() == pytest.approx([0, 0, 5])

    def test_max_only(self):
        v = FloatVec.from_list([1.0, 50.0, 100.0])
        result = clamp(v, max_val=10.0)
        assert result.to_list() == pytest.approx([1, 10, 10])

    def test_no_limits(self):
        v = FloatVec.from_list([-100.0, 0.0, 100.0])
        result = clamp(v)
        assert result.to_list() == pytest.approx([-100, 0, 100])


class TestMapFn:
    def test_relu(self):
        v = FloatVec.from_list([-2.0, 0.0, 3.0])
        result = map_fn(v, "relu")
        assert result.to_list() == pytest.approx([0, 0, 3])

    def test_sigmoid_zero(self):
        v = FloatVec.from_list([0.0])
        result = map_fn(v, "sigmoid")
        assert result.to_list() == pytest.approx([0.5])

    def test_sigmoid_large_positive(self):
        v = FloatVec.from_list([10.0])
        result = map_fn(v, "sigmoid")
        assert result.to_list()[0] > 0.999

    def test_abs(self):
        v = FloatVec.from_list([-3.0, 0.0, 5.0])
        result = map_fn(v, "abs")
        assert result.to_list() == pytest.approx([3, 0, 5])

    def test_neg(self):
        v = FloatVec.from_list([1.0, -2.0, 0.0])
        result = map_fn(v, "neg")
        assert result.to_list() == pytest.approx([-1, 2, 0])

    def test_square(self):
        v = FloatVec.from_list([2.0, -3.0, 0.0])
        result = map_fn(v, "square")
        assert result.to_list() == pytest.approx([4, 9, 0])

    def test_sqrt(self):
        v = FloatVec.from_list([4.0, 9.0, 0.0])
        result = map_fn(v, "sqrt")
        assert result.to_list() == pytest.approx([2, 3, 0])

    def test_unknown_fn_raises(self):
        v = FloatVec.from_list([1.0])
        with pytest.raises(ValueError, match="Unknown map function"):
            map_fn(v, "nonexistent")


# ══════════════════════════════════════════════════════════════════════════
# SecurityLevel classification tests
# ══════════════════════════════════════════════════════════════════════════


class TestSecurityLevel:
    def test_encrypted_ops(self):
        """All 5 original ops should be ENCRYPTED."""
        from axol.core.types import TransMatrix
        m = TransMatrix.identity(2)
        w = FloatVec.from_list([0.5, 0.5])

        ops = [
            TransformOp(key="x", matrix=m),
            GateOp(key="x", gate_key="g"),
            MergeOp(keys=["a", "b"], weights=w, out_key="c"),
            DistanceOp(key_a="a", key_b="b"),
            RouteOp(key="x", router=m),
        ]
        for op in ops:
            assert op.security == SecurityLevel.ENCRYPTED, f"{type(op).__name__} should be E"

    def test_plaintext_ops(self):
        """4 new ops + CustomOp should be PLAINTEXT."""
        ops = [
            StepOp(key="x"),
            BranchOp(gate_key="g", then_key="a", else_key="b", out_key="c"),
            ClampOp(key="x"),
            MapOp(key="x", fn_name="relu"),
            CustomOp(fn=lambda s: s),
        ]
        for op in ops:
            assert op.security == SecurityLevel.PLAINTEXT, f"{type(op).__name__} should be P"


# ══════════════════════════════════════════════════════════════════════════
# DSL parsing tests
# ══════════════════════════════════════════════════════════════════════════


class TestDSLStep:
    def test_basic(self):
        src = """
        @test
        s x=[1 -1 0.5]
        :s1=step(x)
        """
        prog = parse(src)
        op = prog.transitions[0].operation
        assert isinstance(op, StepOp)
        assert op.key == "x"
        assert op.threshold == 0.0
        assert op.out_key is None

    def test_with_threshold_and_out(self):
        src = """
        @test
        s x=[0.3 0.5 0.7]
        :s1=step(x;t=0.5)->g
        """
        prog = parse(src)
        op = prog.transitions[0].operation
        assert isinstance(op, StepOp)
        assert op.threshold == 0.5
        assert op.out_key == "g"


class TestDSLBranch:
    def test_basic(self):
        src = """
        @test
        s g=[1 0 1] a=[10 20 30] b=[1 2 3]
        :b1=branch(g;then=a,else=b)->out
        """
        prog = parse(src)
        op = prog.transitions[0].operation
        assert isinstance(op, BranchOp)
        assert op.gate_key == "g"
        assert op.then_key == "a"
        assert op.else_key == "b"
        assert op.out_key == "out"

    def test_missing_out_key_raises(self):
        src = """
        @test
        s g=[1 0] a=[1 2] b=[3 4]
        :b1=branch(g;then=a,else=b)
        """
        with pytest.raises(ParseError, match="branch requires ->out_key"):
            parse(src)

    def test_auto_coerce_gate(self):
        """FloatVec with 0/1 values should auto-coerce to GateVec."""
        src = """
        @test
        s mask=[1 0 1] hi=[10 20 30] lo=[1 2 3]
        :b1=branch(mask;then=hi,else=lo)->result
        """
        prog = parse(src)
        assert isinstance(prog.initial_state["mask"], GateVec)


class TestDSLClamp:
    def test_basic(self):
        src = """
        @test
        s x=[-5 0 5 10]
        :c1=clamp(x;min=0,max=8)
        """
        prog = parse(src)
        op = prog.transitions[0].operation
        assert isinstance(op, ClampOp)
        assert op.min_val == 0.0
        assert op.max_val == 8.0

    def test_semicolon_separated(self):
        src = """
        @test
        s x=[1 2 3]
        :c1=clamp(x;min=0;max=2)->y
        """
        prog = parse(src)
        op = prog.transitions[0].operation
        assert isinstance(op, ClampOp)
        assert op.min_val == 0.0
        assert op.max_val == 2.0
        assert op.out_key == "y"


class TestDSLMap:
    def test_basic(self):
        src = """
        @test
        s x=[-2 0 3]
        :m1=map(x;fn=relu)
        """
        prog = parse(src)
        op = prog.transitions[0].operation
        assert isinstance(op, MapOp)
        assert op.fn_name == "relu"

    def test_with_out_key(self):
        src = """
        @test
        s x=[1 2 3]
        :m1=map(x;fn=square)->y
        """
        prog = parse(src)
        op = prog.transitions[0].operation
        assert op.fn_name == "square"
        assert op.out_key == "y"

    def test_missing_fn_raises(self):
        src = """
        @test
        s x=[1 2]
        :m1=map(x)
        """
        with pytest.raises(ParseError, match="map requires fn= parameter"):
            parse(src)


# ══════════════════════════════════════════════════════════════════════════
# Integration tests: DSL parse → run → result
# ══════════════════════════════════════════════════════════════════════════


class TestIntegrationStepBranch:
    def test_step_then_branch_pipeline(self):
        """step + branch: threshold select between two vectors."""
        src = """
        @step_branch
        s scores=[0.3 0.8 0.1] high=[100 200 300] low=[1 2 3]
        :s1=step(scores;t=0.5)->mask
        :b1=branch(mask;then=high,else=low)->result
        """
        prog = parse(src)
        result = run_program(prog)
        out = result.final_state["result"].to_list()
        # mask = [0, 1, 0] since 0.3<0.5, 0.8>=0.5, 0.1<0.5
        assert out == pytest.approx([1, 200, 3])


class TestIntegrationClamp:
    def test_clamp_run(self):
        src = """
        @clamp_test
        s x=[-10 5 20]
        :c1=clamp(x;min=0,max=10)
        """
        prog = parse(src)
        result = run_program(prog)
        out = result.final_state["x"].to_list()
        assert out == pytest.approx([0, 5, 10])


class TestIntegrationMap:
    def test_relu_run(self):
        src = """
        @map_test
        s x=[-2 0 3]
        :m1=map(x;fn=relu)
        """
        prog = parse(src)
        result = run_program(prog)
        out = result.final_state["x"].to_list()
        assert out == pytest.approx([0, 0, 3])

    def test_sigmoid_run(self):
        src = """
        @sig_test
        s x=[0 0 0]
        :m1=map(x;fn=sigmoid)->y
        """
        prog = parse(src)
        result = run_program(prog)
        out = result.final_state["y"].to_list()
        assert out == pytest.approx([0.5, 0.5, 0.5])

    def test_map_with_out_key(self):
        src = """
        @map_out
        s x=[4 9 16]
        :m1=map(x;fn=sqrt)->y
        """
        prog = parse(src)
        result = run_program(prog)
        # Original x should be unchanged
        assert result.final_state["x"].to_list() == pytest.approx([4, 9, 16])
        # y should have sqrt values
        assert result.final_state["y"].to_list() == pytest.approx([2, 3, 4])
