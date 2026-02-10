"""Axol vs Pure Python -동일 로직, 결과 비교 + 코드량/실행시간 벤치마크."""

import time
import pytest
import numpy as np

from axol.core.types import (
    FloatVec, GateVec, OneHotVec, TransMatrix, StateBundle,
)
from axol.core.program import (
    TransformOp, MergeOp, CustomOp, Transition, Program, run_program,
)
from axol.core.verify import VerifySpec


# ═══════════════════════════════════════════════════════════════════════════
# 1. Counter: 0 -> 5
# ═══════════════════════════════════════════════════════════════════════════

def python_counter(target: int = 5) -> float:
    """순수 Python: while 루프로 카운터."""
    count = 0.0
    while count < target:
        count += 1.0
    return count


def axol_counter(target: int = 5) -> tuple[float, str]:
    """Axol: merge 연산으로 카운터."""
    initial = StateBundle(vectors={
        "count": FloatVec.from_list([0.0]),
        "one": FloatVec.from_list([1.0]),
        "done": GateVec.from_list([0.0]),
    })

    def check_done(state: StateBundle) -> StateBundle:
        s = state.copy()
        if float(s["count"].data[0]) >= target:
            s["done"] = GateVec.from_list([1.0])
        return s

    prog = Program(
        name="counter",
        initial_state=initial,
        transitions=[
            Transition("increment", MergeOp(
                keys=["count", "one"],
                weights=FloatVec.from_list([1.0, 1.0]),
                out_key="count",
            )),
            Transition("check", CustomOp(fn=check_done, label="check")),
        ],
        terminal_key="done",
        max_iterations=1000,
    )
    result = run_program(prog)
    return float(result.final_state["count"].data[0]), result.terminated_by


class TestCounterComparison:
    def test_same_result(self):
        py_result = python_counter(5)
        axol_result, terminated = axol_counter(5)
        assert py_result == pytest.approx(axol_result)
        assert terminated == "terminal_condition"

    def test_various_targets(self):
        for target in [1, 3, 10, 20]:
            py = python_counter(target)
            ax, _ = axol_counter(target)
            assert py == pytest.approx(ax), f"target={target}: python={py}, axol={ax}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. State Machine: IDLE -> RUNNING -> DONE
# ═══════════════════════════════════════════════════════════════════════════

STATES = ["IDLE", "RUNNING", "DONE"]
TRANSITIONS_MAP = {"IDLE": "RUNNING", "RUNNING": "DONE", "DONE": "DONE"}


def python_state_machine() -> tuple[str, int]:
    """순수 Python: 문자열 상태 + dict 전이."""
    state = "IDLE"
    steps = 0
    while state != "DONE":
        state = TRANSITIONS_MAP[state]
        steps += 1
    return state, steps


def axol_state_machine() -> tuple[list[float], int]:
    """Axol: OneHotVec + TransMatrix 전이."""
    initial = StateBundle(vectors={
        "state": OneHotVec.from_index(0, 3),
        "done": GateVec.from_list([0.0]),
    })
    shift = TransMatrix.from_list([
        [0.0, 1.0, 0.0],  # IDLE -> RUNNING
        [0.0, 0.0, 1.0],  # RUNNING -> DONE
        [0.0, 0.0, 1.0],  # DONE -> DONE (absorbing)
    ])

    def check_done(state: StateBundle) -> StateBundle:
        s = state.copy()
        if float(s["state"].data[2]) >= 0.99:
            s["done"] = GateVec.from_list([1.0])
        return s

    prog = Program(
        name="state_machine",
        initial_state=initial,
        transitions=[
            Transition("advance", TransformOp(key="state", matrix=shift)),
            Transition("check", CustomOp(fn=check_done, label="check")),
        ],
        terminal_key="done",
        max_iterations=10,
    )
    result = run_program(prog)
    state_vec = result.final_state["state"].to_list()
    iterations = result.steps_executed // 2  # 2 transitions per iteration
    return state_vec, iterations


class TestStateMachineComparison:
    def test_same_final_state(self):
        py_state, py_steps = python_state_machine()
        ax_vec, ax_iters = axol_state_machine()

        # Python: "DONE" -> index 2
        assert py_state == "DONE"
        assert STATES.index(py_state) == 2

        # Axol: one-hot [0, 0, 1] -> index 2
        assert ax_vec == pytest.approx([0.0, 0.0, 1.0])
        assert int(np.argmax(ax_vec)) == 2

        # 동일한 스텝 수
        assert py_steps == ax_iters == 2


# ═══════════════════════════════════════════════════════════════════════════
# 3. Character HP Decay: 100 x 0.8^3 = 51.2
# ═══════════════════════════════════════════════════════════════════════════

def python_hp_decay(hp: float = 100.0, factor: float = 0.8, rounds: int = 3) -> tuple[float, list[float]]:
    """순수 Python: for 루프로 HP 감소."""
    history = [hp]
    for _ in range(rounds):
        hp *= factor
        history.append(hp)
    return hp, history


def axol_hp_decay(hp: float = 100.0, factor: float = 0.8, rounds: int = 3) -> tuple[float, list[float]]:
    """Axol: TransMatrix로 HP 감소."""
    initial = StateBundle(vectors={
        "hp": FloatVec.from_list([hp]),
        "round": FloatVec.from_list([0.0]),
        "done": GateVec.from_list([0.0]),
    })
    decay = TransMatrix.from_list([[factor]])

    def count_round(state: StateBundle) -> StateBundle:
        s = state.copy()
        r = float(s["round"].data[0]) + 1.0
        s["round"] = FloatVec.from_list([r])
        if r >= rounds:
            s["done"] = GateVec.from_list([1.0])
        return s

    prog = Program(
        name="hp_decay",
        initial_state=initial,
        transitions=[
            Transition("decay", TransformOp(key="hp", matrix=decay)),
            Transition("count", CustomOp(fn=count_round, label="count")),
        ],
        terminal_key="done",
        max_iterations=100,
    )
    result = run_program(prog)
    final_hp = float(result.final_state["hp"].data[0])
    history = [hp] + [
        float(t.state_after["hp"].data[0])
        for t in result.trace if t.transition_name == "decay"
    ]
    return final_hp, history


class TestHpDecayComparison:
    def test_same_result(self):
        py_hp, py_hist = python_hp_decay()
        ax_hp, ax_hist = axol_hp_decay()

        assert py_hp == pytest.approx(ax_hp, abs=0.01)
        assert py_hp == pytest.approx(51.2, abs=0.01)

        # 매 라운드 이력도 동일
        for i, (p, a) in enumerate(zip(py_hist, ax_hist)):
            assert p == pytest.approx(a, abs=0.01), f"round {i}: py={p}, axol={a}"

    def test_various_params(self):
        cases = [
            (200.0, 0.5, 4),   # 200 x 0.5^4 = 12.5
            (50.0, 0.9, 10),   # 50 x 0.9^10 ~ 17.43
            (100.0, 1.0, 5),   # 변화 없음
        ]
        for hp, factor, rounds in cases:
            py_hp, _ = python_hp_decay(hp, factor, rounds)
            ax_hp, _ = axol_hp_decay(hp, factor, rounds)
            assert py_hp == pytest.approx(ax_hp, abs=0.1), (
                f"hp={hp}, factor={factor}, rounds={rounds}: py={py_hp}, axol={ax_hp}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 4. 벤치마크: 실행 시간 비교 (informational, not asserted)
# ═══════════════════════════════════════════════════════════════════════════

def _time_it(fn, n=100):
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed / n  # 평균 (초)


class TestBenchmark:
    """실행 시간 비교. assert 없이 결과만 출력."""

    def test_benchmark_counter(self, capsys):
        n = 500
        py_avg = _time_it(lambda: python_counter(50), n)
        ax_avg = _time_it(lambda: axol_counter(50), n)

        with capsys.disabled():
            print(f"\n{'='*60}")
            print(f"  Counter (0->50) -{n}회 평균")
            print(f"  Python : {py_avg*1e6:>10.1f} μs")
            print(f"  Axol   : {ax_avg*1e6:>10.1f} μs")
            print(f"  비율   : Axol은 Python 대비 {ax_avg/py_avg:.1f}x")
            print(f"{'='*60}")

    def test_benchmark_state_machine(self, capsys):
        n = 500
        py_avg = _time_it(python_state_machine, n)
        ax_avg = _time_it(axol_state_machine, n)

        with capsys.disabled():
            print(f"\n{'='*60}")
            print(f"  State Machine (IDLE->DONE) -{n}회 평균")
            print(f"  Python : {py_avg*1e6:>10.1f} μs")
            print(f"  Axol   : {ax_avg*1e6:>10.1f} μs")
            print(f"  비율   : Axol은 Python 대비 {ax_avg/py_avg:.1f}x")
            print(f"{'='*60}")

    def test_benchmark_hp_decay(self, capsys):
        n = 500
        py_avg = _time_it(python_hp_decay, n)
        ax_avg = _time_it(axol_hp_decay, n)

        with capsys.disabled():
            print(f"\n{'='*60}")
            print(f"  HP Decay (100->51.2, 3 rounds) -{n}회 평균")
            print(f"  Python : {py_avg*1e6:>10.1f} μs")
            print(f"  Axol   : {ax_avg*1e6:>10.1f} μs")
            print(f"  비율   : Axol은 Python 대비 {ax_avg/py_avg:.1f}x")
            print(f"{'='*60}")

    def test_benchmark_large_vector(self, capsys):
        """큰 벡터에서 Axol(NumPy)의 강점이 드러나는지 확인."""
        dim = 10000
        n = 200

        def python_large_transform():
            state = [float(i) for i in range(dim)]
            matrix = [[0.0]*dim for _ in range(dim)]
            for i in range(dim):
                matrix[i][i] = 2.0
            result = [0.0]*dim
            for i in range(dim):
                for j in range(dim):
                    result[j] += state[i] * matrix[i][j]
            return result

        def axol_large_transform():
            state = FloatVec(data=np.arange(dim, dtype=np.float32))
            matrix = TransMatrix(data=np.eye(dim, dtype=np.float32) * 2.0)
            from axol.core.operations import transform
            return transform(state, matrix)

        py_avg = _time_it(python_large_transform, n=2)  # 순수 Python은 느려서 2회만
        ax_avg = _time_it(axol_large_transform, n)

        with capsys.disabled():
            print(f"\n{'='*60}")
            print(f"  대규모 벡터 변환 (dim={dim}, identityx2)")
            print(f"  Python : {py_avg*1e3:>10.1f} ms  (순수 루프, {2}회 평균)")
            print(f"  Axol   : {ax_avg*1e3:>10.1f} ms  (NumPy, {n}회 평균)")
            print(f"  비율   : Axol은 Python 대비 {py_avg/ax_avg:.0f}x 빠름")
            print(f"{'='*60}")
