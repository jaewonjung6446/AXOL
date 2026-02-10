"""Example: Character HP — decay via transform (100 → 51.2 after 3 rounds of 0.8x)."""

import pytest
import numpy as np

from axol.core.types import FloatVec, GateVec, TransMatrix, StateBundle
from axol.core.program import (
    TransformOp,
    CustomOp,
    Transition,
    Program,
    run_program,
)
from axol.core.verify import VerifySpec


def test_hp_decay():
    """HP starts at 100, multiplied by 0.8 each round for 3 rounds.

    100 × 0.8 = 80
     80 × 0.8 = 64
     64 × 0.8 = 51.2

    Uses a 1×1 TransMatrix as the decay factor.
    Terminal: CustomOp counts rounds and sets done after 3.
    """
    initial = StateBundle(vectors={
        "hp": FloatVec.from_list([100.0]),
        "round": FloatVec.from_list([0.0]),
        "done": GateVec.from_list([0.0]),
    })

    expected = StateBundle(vectors={
        "hp": FloatVec.from_list([51.2]),
    })

    decay = TransMatrix.from_list([[0.8]])

    def count_round(state: StateBundle) -> StateBundle:
        s = state.copy()
        r = float(s["round"].data[0]) + 1.0
        s["round"] = FloatVec.from_list([r])
        if r >= 3.0:
            s["done"] = GateVec.from_list([1.0])
        return s

    prog = Program(
        name="hp_decay",
        initial_state=initial,
        transitions=[
            Transition(name="apply_decay", operation=TransformOp(key="hp", matrix=decay)),
            Transition(name="count", operation=CustomOp(fn=count_round, label="count_round")),
        ],
        terminal_key="done",
        max_iterations=100,
        expected_state=expected,
        verify_specs={"hp": VerifySpec.exact(tolerance=0.1)},
    )

    result = run_program(prog)

    assert result.terminated_by == "terminal_condition"
    assert result.steps_executed == 6  # 2 transitions × 3 iterations
    hp_final = result.final_state["hp"].to_list()[0]
    assert hp_final == pytest.approx(51.2, abs=0.1)
    assert result.verification is not None
    assert result.verification.passed is True
