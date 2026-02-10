"""Tests for the Axol compiler optimizer."""

import copy

import numpy as np
import pytest

from axol.core.types import FloatVec, GateVec, TransMatrix, StateBundle, OneHotVec
from axol.core.program import (
    Program, Transition, TransformOp, GateOp, MergeOp, CustomOp, run_program,
)
from axol.core.optimizer import optimize
from axol.core.dsl import parse


# ═══════════════════════════════════════════════════════════════════════════
# 1. Transform Fusion
# ═══════════════════════════════════════════════════════════════════════════

class TestTransformFusion:
    def test_fuse_two_transforms(self):
        """Two consecutive TransformOps on the same key fuse into one."""
        M1 = TransMatrix.from_list([[2.0]])
        M2 = TransMatrix.from_list([[3.0]])
        prog = Program(
            name="fuse2",
            initial_state=StateBundle(vectors={"v": FloatVec.from_list([1.0])}),
            transitions=[
                Transition("t1", TransformOp(key="v", matrix=M1)),
                Transition("t2", TransformOp(key="v", matrix=M2)),
            ],
        )
        opt = optimize(prog, eliminate_dead=False, fold_constants=False)
        assert len(opt.transitions) == 1
        # Fused: M1@M2 = [[2*3]] = [[6]]
        fused_m = opt.transitions[0].operation.matrix.data
        np.testing.assert_allclose(fused_m, [[6.0]], atol=1e-5)

    def test_fuse_three_transforms(self):
        """Three consecutive TransformOps fuse into one via fixed-point."""
        M1 = TransMatrix.from_list([[2.0]])
        M2 = TransMatrix.from_list([[3.0]])
        M3 = TransMatrix.from_list([[0.5]])
        prog = Program(
            name="fuse3",
            initial_state=StateBundle(vectors={"v": FloatVec.from_list([1.0])}),
            transitions=[
                Transition("t1", TransformOp(key="v", matrix=M1)),
                Transition("t2", TransformOp(key="v", matrix=M2)),
                Transition("t3", TransformOp(key="v", matrix=M3)),
            ],
        )
        opt = optimize(prog, eliminate_dead=False, fold_constants=False)
        assert len(opt.transitions) == 1
        fused_m = opt.transitions[0].operation.matrix.data
        np.testing.assert_allclose(fused_m, [[3.0]], atol=1e-5)  # 2*3*0.5

    def test_different_keys_no_fusion(self):
        """TransformOps on different keys should NOT fuse."""
        M = TransMatrix.from_list([[2.0]])
        prog = Program(
            name="nofuse",
            initial_state=StateBundle(vectors={
                "a": FloatVec.from_list([1.0]),
                "b": FloatVec.from_list([1.0]),
            }),
            transitions=[
                Transition("t1", TransformOp(key="a", matrix=M)),
                Transition("t2", TransformOp(key="b", matrix=M)),
            ],
        )
        opt = optimize(prog, eliminate_dead=False, fold_constants=False)
        assert len(opt.transitions) == 2

    def test_custom_op_boundary(self):
        """CustomOp between TransformOps prevents fusion."""
        M = TransMatrix.from_list([[2.0]])

        def identity(state):
            return state.copy()

        prog = Program(
            name="boundary",
            initial_state=StateBundle(vectors={"v": FloatVec.from_list([1.0])}),
            transitions=[
                Transition("t1", TransformOp(key="v", matrix=M)),
                Transition("custom", CustomOp(fn=identity, label="id")),
                Transition("t2", TransformOp(key="v", matrix=M)),
            ],
        )
        opt = optimize(prog, eliminate_dead=False, fold_constants=False)
        # CustomOp blocks fusion; t1 and t2 remain separate
        assert len(opt.transitions) == 3

    def test_fusion_with_out_key_chain(self):
        """TransformOp with out_key chains properly."""
        M1 = TransMatrix.from_list([[2.0]])
        M2 = TransMatrix.from_list([[3.0]])
        prog = Program(
            name="chain",
            initial_state=StateBundle(vectors={
                "a": FloatVec.from_list([1.0]),
                "b": FloatVec.from_list([0.0]),
            }),
            transitions=[
                Transition("t1", TransformOp(key="a", matrix=M1, out_key="b")),
                Transition("t2", TransformOp(key="b", matrix=M2)),
            ],
        )
        opt = optimize(prog, eliminate_dead=False, fold_constants=False)
        # t1 outputs to "b", t2 reads "b" -> should fuse
        assert len(opt.transitions) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. Dead State Elimination
# ═══════════════════════════════════════════════════════════════════════════

class TestDeadStateElimination:
    def test_remove_unused_state(self):
        """State keys not read by any transition are removed."""
        M = TransMatrix.from_list([[2.0]])
        prog = Program(
            name="dead",
            initial_state=StateBundle(vectors={
                "used": FloatVec.from_list([1.0]),
                "unused": FloatVec.from_list([99.0]),
            }),
            transitions=[
                Transition("t1", TransformOp(key="used", matrix=M)),
            ],
        )
        opt = optimize(prog, fuse=False, fold_constants=False)
        assert "used" in opt.initial_state
        assert "unused" not in opt.initial_state

    def test_keep_used_state(self):
        """All read keys are preserved."""
        M = TransMatrix.from_list([[2.0]])
        prog = Program(
            name="keep",
            initial_state=StateBundle(vectors={
                "a": FloatVec.from_list([1.0]),
                "b": FloatVec.from_list([2.0]),
            }),
            transitions=[
                Transition("t1", TransformOp(key="a", matrix=M)),
                Transition("t2", TransformOp(key="b", matrix=M)),
            ],
        )
        opt = optimize(prog, fuse=False, fold_constants=False)
        assert "a" in opt.initial_state
        assert "b" in opt.initial_state

    def test_custom_op_conservative(self):
        """With CustomOp, no state is eliminated (conservative)."""
        def noop(state):
            return state.copy()

        prog = Program(
            name="conservative",
            initial_state=StateBundle(vectors={
                "a": FloatVec.from_list([1.0]),
                "unused": FloatVec.from_list([99.0]),
            }),
            transitions=[
                Transition("custom", CustomOp(fn=noop, label="noop")),
            ],
        )
        opt = optimize(prog, fuse=False, fold_constants=False)
        assert "unused" in opt.initial_state

    def test_terminal_key_preserved(self):
        """Terminal key is always considered 'read'."""
        M = TransMatrix.from_list([[2.0]])
        prog = Program(
            name="term",
            initial_state=StateBundle(vectors={
                "v": FloatVec.from_list([1.0]),
                "done": GateVec.from_list([0.0]),
            }),
            transitions=[
                Transition("t1", TransformOp(key="v", matrix=M)),
            ],
            terminal_key="done",
            max_iterations=10,
        )
        opt = optimize(prog, fuse=False, fold_constants=False)
        assert "done" in opt.initial_state


# ═══════════════════════════════════════════════════════════════════════════
# 3. Constant Folding
# ═══════════════════════════════════════════════════════════════════════════

class TestConstantFolding:
    def test_fold_immutable_transform(self):
        """TransformOp on a key that's never written should fold at compile time."""
        M = TransMatrix.from_list([[3.0]])
        prog = Program(
            name="fold",
            initial_state=StateBundle(vectors={
                "const": FloatVec.from_list([10.0]),
            }),
            transitions=[
                Transition("t1", TransformOp(key="const", matrix=M)),
            ],
        )
        opt = optimize(prog, fuse=False, eliminate_dead=False)
        # Transition should be removed (folded)
        assert len(opt.transitions) == 0
        # Result pre-computed in initial state
        val = opt.initial_state["const"].to_list()[0]
        assert val == pytest.approx(30.0, abs=1e-3)

    def test_no_fold_mutable_key(self):
        """Keys that are written to by other transitions should NOT fold."""
        M = TransMatrix.from_list([[3.0]])
        prog = Program(
            name="nofold",
            initial_state=StateBundle(vectors={
                "v": FloatVec.from_list([10.0]),
                "one": FloatVec.from_list([1.0]),
            }),
            transitions=[
                Transition("t1", TransformOp(key="v", matrix=M)),
                Transition("t2", MergeOp(
                    keys=["v", "one"],
                    weights=FloatVec.from_list([1.0, 1.0]),
                    out_key="v",
                )),
            ],
        )
        opt = optimize(prog, fuse=False, eliminate_dead=False)
        # "v" is written by t2, so t1 cannot fold
        transform_count = sum(
            1 for t in opt.transitions
            if isinstance(t.operation, TransformOp)
        )
        assert transform_count == 1

    def test_custom_op_prevents_folding(self):
        """CustomOp prevents all constant folding (conservative)."""
        M = TransMatrix.from_list([[3.0]])

        def noop(state):
            return state.copy()

        prog = Program(
            name="nofold_custom",
            initial_state=StateBundle(vectors={
                "const": FloatVec.from_list([10.0]),
            }),
            transitions=[
                Transition("t1", TransformOp(key="const", matrix=M)),
                Transition("custom", CustomOp(fn=noop, label="noop")),
            ],
        )
        opt = optimize(prog, fuse=False, eliminate_dead=False)
        # CustomOp present -> no folding
        assert len(opt.transitions) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. Semantics Preservation: optimize + run == run
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizePreservesSemantics:
    PROGRAMS = [
        # counter
        "@counter\ns count=[0] one=[1]\n: tick=merge(count one;w=[1 1])->count\n? done count>=5",
        # fsm
        "@fsm\ns state=onehot(0,3)\n: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])\n? done state[2]>=1",
        # hp_decay
        "@hp_decay\ns hp=[100] round=[0] one=[1]\n: decay=transform(hp;M=[0.8])\n: tick=merge(round one;w=[1 1])->round\n? done round>=3",
        # pipeline (no terminal)
        "@pipe\ns v=[1 0 0]\n: t1=transform(v;M=[0 1 0;0 0 1;1 0 0])\n: t2=transform(v;M=[2 0 0;0 2 0;0 0 2])",
        # merge pipeline
        "@merge_pipe\ns a=[1 2] b=[3 4]\n: m=merge(a b;w=[0.5 0.5])->c",
    ]

    @pytest.mark.parametrize("source", PROGRAMS,
                             ids=["counter", "fsm", "hp_decay", "pipeline", "merge_pipe"])
    def test_optimized_same_result(self, source):
        prog = parse(source)
        prog_opt = optimize(prog)

        result_orig = run_program(prog)
        result_opt = run_program(prog_opt)

        # Compare all final state keys
        for key in result_orig.final_state.keys():
            orig_vec = result_orig.final_state[key].to_list()
            opt_vec = result_opt.final_state[key].to_list()
            assert orig_vec == pytest.approx(opt_vec, abs=1e-3), \
                f"Key '{key}': orig={orig_vec}, opt={opt_vec}"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Purity: optimize does not mutate original
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizePurity:
    def test_original_unchanged(self):
        M = TransMatrix.from_list([[2.0]])
        prog = Program(
            name="purity",
            initial_state=StateBundle(vectors={
                "v": FloatVec.from_list([1.0]),
                "unused": FloatVec.from_list([0.0]),
            }),
            transitions=[
                Transition("t1", TransformOp(key="v", matrix=M)),
                Transition("t2", TransformOp(key="v", matrix=M)),
            ],
        )
        orig_trans_count = len(prog.transitions)
        orig_keys = set(prog.initial_state.keys())

        _ = optimize(prog)

        # Original unchanged
        assert len(prog.transitions) == orig_trans_count
        assert set(prog.initial_state.keys()) == orig_keys
