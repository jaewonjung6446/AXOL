"""Example: Counter — FloatVec[1] from 0→5 using merge as increment."""

import pytest

from axol.core.types import FloatVec, GateVec, StateBundle
from axol.core.program import (
    MergeOp,
    CustomOp,
    Transition,
    Program,
    run_program,
)
from axol.core.verify import VerifySpec


def test_counter_0_to_5():
    """Count from 0 to 5 by adding 1.0 each iteration.

    State:  count = FloatVec[1]
    Logic:  merge([count, one], weights=[1.0, 1.0]) → count += 1
            CustomOp sets 'done' gate when count >= 5
    """
    initial = StateBundle(vectors={
        "count": FloatVec.from_list([0.0]),
        "one": FloatVec.from_list([1.0]),
        "done": GateVec.from_list([0.0]),
    })

    expected = StateBundle(vectors={
        "count": FloatVec.from_list([5.0]),
    })

    def check_done(state: StateBundle) -> StateBundle:
        s = state.copy()
        if float(s["count"].data[0]) >= 5.0:
            s["done"] = GateVec.from_list([1.0])
        return s

    prog = Program(
        name="counter_0_to_5",
        initial_state=initial,
        transitions=[
            Transition(
                name="increment",
                operation=MergeOp(
                    keys=["count", "one"],
                    weights=FloatVec.from_list([1.0, 1.0]),
                    out_key="count",
                ),
            ),
            Transition(
                name="check_terminal",
                operation=CustomOp(fn=check_done, label="check_done"),
            ),
        ],
        terminal_key="done",
        max_iterations=100,
        expected_state=expected,
        verify_specs={"count": VerifySpec.exact(tolerance=1e-3)},
    )

    result = run_program(prog)

    assert result.terminated_by == "terminal_condition"
    assert result.final_state["count"].to_list() == pytest.approx([5.0])
    assert result.verification is not None
    assert result.verification.passed is True
