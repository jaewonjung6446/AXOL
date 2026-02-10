"""DSL parse → run correctness + token cost comparison vs Python.

Verifies that DSL-parsed programs produce identical results to
hand-built Python/Axol programs, then compares token counts.
"""

import textwrap

import pytest
import numpy as np

from axol.core.dsl import parse
from axol.core.program import run_program
from axol.core.types import FloatVec, GateVec, OneHotVec, TransMatrix, StateBundle
from axol.core.program import (
    TransformOp, MergeOp, CustomOp, Transition, Program,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Correctness: DSL-parsed vs hand-built Python produce same results
# ═══════════════════════════════════════════════════════════════════════════

class TestDslVsPython:
    """Run identical logic via DSL parse and via hand-built Python, compare results."""

    def test_counter_same_result(self):
        """Counter 0→5: DSL vs Python hand-built."""
        # DSL version
        dsl_src = textwrap.dedent("""\
            @counter
            s count=[0] one=[1]
            : increment=merge(count one;w=[1 1])->count
            ? done count>=5
        """)
        dsl_prog = parse(dsl_src)
        dsl_result = run_program(dsl_prog)
        dsl_count = float(dsl_result.final_state["count"].data[0])

        # Python version
        count = 0.0
        while count < 5:
            count += 1.0

        assert dsl_count == pytest.approx(count)

    def test_hp_decay_same_result(self):
        """HP Decay 100×0.8^3 = 51.2: DSL vs Python."""
        dsl_src = textwrap.dedent("""\
            @hp_decay
            s hp=[100] round=[0] one=[1]
            : decay=transform(hp;M=[0.8])
            : tick=merge(round one;w=[1 1])->round
            ? done round>=3
        """)
        dsl_prog = parse(dsl_src)
        dsl_result = run_program(dsl_prog)
        dsl_hp = float(dsl_result.final_state["hp"].data[0])

        # Python version
        hp = 100.0
        for _ in range(3):
            hp *= 0.8

        assert dsl_hp == pytest.approx(hp, abs=0.01)

    def test_state_machine_same_result(self):
        """State machine IDLE→DONE: DSL vs Python."""
        dsl_src = textwrap.dedent("""\
            @state_machine
            s state=onehot(0,3)
            : advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
            ? done state[2]>=1
        """)
        dsl_prog = parse(dsl_src)
        dsl_result = run_program(dsl_prog)
        dsl_state = dsl_result.final_state["state"].to_list()

        # Python version
        TRANS = {"IDLE": "RUNNING", "RUNNING": "DONE", "DONE": "DONE"}
        state = "IDLE"
        while state != "DONE":
            state = TRANS[state]
        py_idx = ["IDLE", "RUNNING", "DONE"].index(state)

        assert int(np.argmax(dsl_state)) == py_idx


# ═══════════════════════════════════════════════════════════════════════════
# 2. Token cost: DSL source vs equivalent Python source
# ═══════════════════════════════════════════════════════════════════════════

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


PYTHON_COUNTER = textwrap.dedent("""\
    def counter(target=5):
        count = 0.0
        while count < target:
            count += 1.0
        return count
""")

DSL_COUNTER = textwrap.dedent("""\
    @counter
    s count=[0] one=[1]
    : increment=merge(count one;w=[1 1])->count
    ? done count>=5
""")

PYTHON_HP = textwrap.dedent("""\
    def hp_decay(hp=100.0, factor=0.8, rounds=3):
        history = [hp]
        for _ in range(rounds):
            hp *= factor
            history.append(hp)
        return hp, history
""")

DSL_HP = textwrap.dedent("""\
    @hp_decay
    s hp=[100] round=[0] one=[1]
    : decay=transform(hp;M=[0.8])
    : tick=merge(round one;w=[1 1])->round
    ? done round>=3
""")

PYTHON_SM = textwrap.dedent("""\
    TRANSITIONS = {"IDLE": "RUNNING", "RUNNING": "DONE", "DONE": "DONE"}

    def state_machine():
        state = "IDLE"
        steps = 0
        while state != "DONE":
            state = TRANSITIONS[state]
            steps += 1
        return state, steps
""")

DSL_SM = textwrap.dedent("""\
    @state_machine
    s state=onehot(0,3)
    : advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
    ? done state[2]>=1
""")

CASES = [
    ("Counter 0->5",     PYTHON_COUNTER, DSL_COUNTER),
    ("HP Decay",         PYTHON_HP,      DSL_HP),
    ("State Machine",    PYTHON_SM,      DSL_SM),
]


@pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken not installed")
class TestTokenComparison:
    @pytest.mark.parametrize("name,py_src,dsl_src", CASES, ids=[c[0] for c in CASES])
    def test_dsl_comparable_tokens(self, name, py_src, dsl_src):
        """DSL should use similar or fewer tokens than Python equivalent."""
        py_tokens = len(enc.encode(py_src))
        dsl_tokens = len(enc.encode(dsl_src))
        # Allow up to 10% overhead for simple programs; DSL wins big on complex ones
        assert dsl_tokens <= py_tokens * 1.1, (
            f"{name}: DSL ({dsl_tokens}) uses too many tokens vs Python ({py_tokens})"
        )

    def test_summary_table(self, capsys):
        with capsys.disabled():
            print(f"\n{'='*60}")
            print(f"  DSL vs Python Token Comparison")
            print(f"{'='*60}")
            print(f"  {'Program':<20} {'Python':>7} {'DSL':>7} {'Saving':>8}")
            print(f"{'-'*60}")
            for name, py_src, dsl_src in CASES:
                pt = len(enc.encode(py_src))
                dt = len(enc.encode(dsl_src))
                saving = (1 - dt / pt) * 100
                print(f"  {name:<20} {pt:>7} {dt:>7} {saving:>7.1f}%")
            print(f"{'='*60}")
