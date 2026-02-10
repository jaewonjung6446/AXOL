"""Example: State Machine — OneHotVec[3] with shift matrix (IDLE→RUNNING→DONE)."""

import pytest
import numpy as np

from axol.core.types import FloatVec, OneHotVec, GateVec, TransMatrix, StateBundle
from axol.core.program import (
    TransformOp,
    CustomOp,
    Transition,
    Program,
    run_program,
)
from axol.core.verify import VerifySpec


def test_state_machine_idle_to_done():
    """Three-state machine: IDLE(0) → RUNNING(1) → DONE(2).

    A shift matrix advances the one-hot state by one position each step.
    Terminal condition: state reaches DONE (index 2).
    """
    # States: [IDLE, RUNNING, DONE]
    initial = StateBundle(vectors={
        "state": OneHotVec.from_index(0, 3),  # IDLE
        "done": GateVec.from_list([0.0]),
    })

    expected = StateBundle(vectors={
        "state": FloatVec.from_list([0.0, 0.0, 1.0]),  # DONE
    })

    # Shift matrix: moves probability mass one step right
    # [0,1,0]     IDLE    → RUNNING
    # [0,0,1]     RUNNING → DONE
    # [0,0,1]     DONE    → DONE (absorbing)
    shift = TransMatrix.from_list([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])

    def check_done(state: StateBundle) -> StateBundle:
        s = state.copy()
        # Done when the last element (DONE state) is the highest
        if float(s["state"].data[2]) >= 0.99:
            s["done"] = GateVec.from_list([1.0])
        return s

    prog = Program(
        name="state_machine_3",
        initial_state=initial,
        transitions=[
            Transition(name="advance", operation=TransformOp(key="state", matrix=shift)),
            Transition(name="check", operation=CustomOp(fn=check_done, label="check_done")),
        ],
        terminal_key="done",
        max_iterations=10,
        expected_state=expected,
        verify_specs={"state": VerifySpec.exact(tolerance=1e-3)},
    )

    result = run_program(prog)

    assert result.terminated_by == "terminal_condition"
    # Should take exactly 2 iterations (IDLE→RUNNING, RUNNING→DONE)
    assert result.steps_executed == 4  # 2 transitions × 2 iterations
    assert result.final_state["state"].to_list() == pytest.approx([0.0, 0.0, 1.0])
    assert result.verification is not None
    assert result.verification.passed is True
