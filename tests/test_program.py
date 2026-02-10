"""Tests for axol.core.program."""

import pytest
import numpy as np

from axol.core.types import (
    FloatVec,
    GateVec,
    TransMatrix,
    StateBundle,
)
from axol.core.program import (
    TransformOp,
    GateOp,
    MergeOp,
    CustomOp,
    Transition,
    Program,
    run_program,
)


class TestPipelineMode:
    def test_single_transform(self):
        state = StateBundle(vectors={"v": FloatVec.from_list([1.0, 2.0])})
        scale = TransMatrix.from_list([[2.0, 0.0], [0.0, 3.0]])

        prog = Program(
            name="scale_test",
            initial_state=state,
            transitions=[
                Transition(name="scale", operation=TransformOp(key="v", matrix=scale)),
            ],
        )
        result = run_program(prog)
        assert result.terminated_by == "pipeline_end"
        assert result.steps_executed == 1
        assert result.final_state["v"].to_list() == pytest.approx([2.0, 6.0])

    def test_two_step_pipeline(self):
        state = StateBundle(vectors={
            "v": FloatVec.from_list([4.0, 6.0]),
            "g": GateVec.from_list([1.0, 0.0]),
        })
        prog = Program(
            name="two_step",
            initial_state=state,
            transitions=[
                Transition(name="halve", operation=TransformOp(
                    key="v", matrix=TransMatrix.from_list([[0.5, 0.0], [0.0, 0.5]])
                )),
                Transition(name="gate", operation=GateOp(key="v", gate_key="g")),
            ],
        )
        result = run_program(prog)
        assert result.steps_executed == 2
        assert result.final_state["v"].to_list() == pytest.approx([2.0, 0.0])


class TestLoopMode:
    def test_terminal_condition(self):
        """Loop increments a counter; a CustomOp sets terminal gate when count >= 3."""
        state = StateBundle(vectors={
            "count": FloatVec.from_list([0.0]),
            "done": GateVec.from_list([0.0]),
        })

        def increment(s: StateBundle) -> StateBundle:
            s = s.copy()
            val = float(s["count"].data[0]) + 1.0
            s["count"] = FloatVec.from_list([val])
            if val >= 3.0:
                s["done"] = GateVec.from_list([1.0])
            return s

        prog = Program(
            name="count_to_3",
            initial_state=state,
            transitions=[
                Transition(name="inc", operation=CustomOp(fn=increment, label="increment")),
            ],
            terminal_key="done",
            max_iterations=100,
        )
        result = run_program(prog)
        assert result.terminated_by == "terminal_condition"
        assert result.final_state["count"].to_list() == pytest.approx([3.0])

    def test_max_iterations(self):
        state = StateBundle(vectors={
            "v": FloatVec.from_list([1.0]),
            "done": GateVec.from_list([0.0]),
        })
        prog = Program(
            name="infinite",
            initial_state=state,
            transitions=[
                Transition(name="noop", operation=CustomOp(fn=lambda s: s.copy())),
            ],
            terminal_key="done",
            max_iterations=5,
        )
        result = run_program(prog)
        assert result.terminated_by == "max_iterations"
        assert result.steps_executed == 5


class TestTrace:
    def test_trace_recording(self):
        state = StateBundle(vectors={"v": FloatVec.from_list([1.0])})
        prog = Program(
            name="trace_test",
            initial_state=state,
            transitions=[
                Transition(name="double", operation=TransformOp(
                    key="v", matrix=TransMatrix.from_list([[2.0]])
                )),
                Transition(name="triple", operation=TransformOp(
                    key="v", matrix=TransMatrix.from_list([[3.0]])
                )),
            ],
        )
        result = run_program(prog)
        assert len(result.trace) == 2
        assert result.trace[0].transition_name == "double"
        assert result.trace[1].transition_name == "triple"
        # After double: 2.0, after triple: 6.0
        assert result.final_state["v"].to_list() == pytest.approx([6.0])


class TestVerification:
    def test_passes_when_matching(self):
        state = StateBundle(vectors={"v": FloatVec.from_list([2.0])})
        expected = StateBundle(vectors={"v": FloatVec.from_list([4.0])})
        prog = Program(
            name="verify_test",
            initial_state=state,
            transitions=[
                Transition(name="double", operation=TransformOp(
                    key="v", matrix=TransMatrix.from_list([[2.0]])
                )),
            ],
            expected_state=expected,
        )
        result = run_program(prog)
        assert result.verification is not None
        assert result.verification.passed is True

    def test_fails_when_mismatched(self):
        state = StateBundle(vectors={"v": FloatVec.from_list([2.0])})
        expected = StateBundle(vectors={"v": FloatVec.from_list([999.0])})
        prog = Program(
            name="verify_fail",
            initial_state=state,
            transitions=[
                Transition(name="double", operation=TransformOp(
                    key="v", matrix=TransMatrix.from_list([[2.0]])
                )),
            ],
            expected_state=expected,
        )
        result = run_program(prog)
        assert result.verification is not None
        assert result.verification.passed is False
