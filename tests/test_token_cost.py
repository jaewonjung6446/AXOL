"""Token cost comparison: Python source vs Axol representations.

AI-native = AI가 읽고 쓸 때 더 적은 토큰을 소모해야 한다.

3가지 직렬화 포맷 비교:
  1. Python source code (baseline)
  2. Axol JSON (pretty-printed) — 구조적이지만 장황
  3. Axol DSL (compact text) — AI 최적화 포맷
"""

import json
import textwrap
import base64
import struct

import pytest
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


# ===================================================================
# Axol DSL compact format
# ===================================================================
# 설계 원칙: JSON의 장황한 키/구문 제거, 벡터는 괄호 안 숫자 나열
#
# 문법 예시:
#   @program counter
#   state count = [0]
#   state one = [1]
#   : increment = merge(count one; w=[1 1]) -> count
#   terminal done when count >= 5
#
# 행렬은 세미콜론으로 행 구분:
#   : advance = transform(state; M=[0 1 0; 0 0 1; 0 0 1])

# -------------------------------------------------------------------
# 1. Counter
# -------------------------------------------------------------------

PYTHON_COUNTER = textwrap.dedent("""\
    def counter(target=5):
        count = 0.0
        while count < target:
            count += 1.0
        return count
""")

AXOL_JSON_COUNTER = json.dumps({
    "name": "counter",
    "state": {"count": [0.0], "one": [1.0]},
    "transitions": [
        {"name": "increment", "op": "merge", "keys": ["count", "one"],
         "weights": [1.0, 1.0], "out": "count"}
    ],
    "terminal": {"key": "done", "when": "count >= 5"}
}, separators=(",", ":"))

AXOL_DSL_COUNTER = textwrap.dedent("""\
    @counter
    s count=[0] one=[1]
    : increment=merge(count one;w=[1 1])->count
    ? done count>=5
""")

# -------------------------------------------------------------------
# 2. State Machine
# -------------------------------------------------------------------

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

AXOL_JSON_SM = json.dumps({
    "name": "state_machine",
    "state": {"state": [1, 0, 0]},
    "transitions": [
        {"name": "advance", "op": "transform", "key": "state",
         "matrix": [[0, 1, 0], [0, 0, 1], [0, 0, 1]]}
    ],
    "terminal": {"key": "done", "when": "state[2] >= 1"}
}, separators=(",", ":"))

AXOL_DSL_SM = textwrap.dedent("""\
    @state_machine
    s state=[1 0 0]
    : advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
    ? done state[2]>=1
""")

# -------------------------------------------------------------------
# 3. HP Decay
# -------------------------------------------------------------------

PYTHON_HP = textwrap.dedent("""\
    def hp_decay(hp=100.0, factor=0.8, rounds=3):
        history = [hp]
        for _ in range(rounds):
            hp *= factor
            history.append(hp)
        return hp, history
""")

AXOL_JSON_HP = json.dumps({
    "name": "hp_decay",
    "state": {"hp": [100.0], "round": [0.0]},
    "transitions": [
        {"name": "decay", "op": "transform", "key": "hp", "matrix": [[0.8]]}
    ],
    "terminal": {"key": "done", "when": "round >= 3"}
}, separators=(",", ":"))

AXOL_DSL_HP = textwrap.dedent("""\
    @hp_decay
    s hp=[100] round=[0]
    : decay=transform(hp;M=[0.8])
    ? done round>=3
""")

# -------------------------------------------------------------------
# 4. RPG Damage
# -------------------------------------------------------------------

PYTHON_RPG = textwrap.dedent("""\
    def calc_damage(atk, defense, element_multiplier, is_critical, is_weak):
        base = max(atk - defense, 1)
        damage = base * element_multiplier
        if is_critical:
            damage *= 1.5
        if is_weak:
            damage *= 2.0
        return min(damage, 9999)

    def apply_damage(hp, atk, defense, element_mult, is_crit, is_weak):
        dmg = calc_damage(atk, defense, element_mult, is_crit, is_weak)
        hp = max(hp - dmg, 0)
        return hp, dmg
""")

AXOL_JSON_RPG = json.dumps({
    "name": "rpg_damage",
    "state": {
        "stats": [100, 30, 10, 1.2],
        "flags": [1, 0, 1, 0],
        "modifiers": [1.5, 2.0, 1.0, 1.0]
    },
    "transitions": [
        {"name": "base_dmg", "op": "transform", "key": "stats",
         "matrix": [[1, 0], [0, -1], [0, 0], [0, 0]]},
        {"name": "apply_mod", "op": "gate", "key": "modifiers", "gate": "flags"},
        {"name": "final", "op": "merge", "keys": ["base_dmg", "modifiers"],
         "weights": [1.0, 1.0], "out": "damage"}
    ]
}, separators=(",", ":"))

AXOL_DSL_RPG = textwrap.dedent("""\
    @rpg_damage
    s stats=[100 30 10 1.2] flags=[1 0 1 0] mods=[1.5 2 1 1]
    : base=transform(stats;M=[1 0;0 -1;0 0;0 0])
    : apply=gate(mods;g=flags)
    : final=merge(base mods;w=[1 1])->damage
""")

# -------------------------------------------------------------------
# 5. Large: 100-state automaton
# -------------------------------------------------------------------

def _gen_python_large(n=100):
    lines = [f"STATES = {list(range(n))}"]
    lines.append("TRANSITIONS = {")
    for i in range(n):
        nxt = min(i + 1, n - 1)
        lines.append(f"    {i}: {nxt},")
    lines.append("}")
    lines.append("")
    lines.append("def run():")
    lines.append("    state = 0")
    lines.append(f"    while state != {n-1}:")
    lines.append("        state = TRANSITIONS[state]")
    lines.append("    return state")
    return "\n".join(lines)


def _gen_axol_json_large(n=100):
    matrix = [[0]*n for _ in range(n)]
    for i in range(n - 1):
        matrix[i][i + 1] = 1
    matrix[n - 1][n - 1] = 1
    return json.dumps({
        "name": f"auto_{n}",
        "state": {"s": [1] + [0]*(n-1)},
        "transitions": [
            {"name": "step", "op": "transform", "key": "s", "matrix": matrix}
        ],
        "terminal": {"key": "done", "when": f"s[{n-1}]>=1"}
    }, separators=(",", ":"))


def _gen_axol_dsl_large(n=100):
    # DSL: shift matrix as sparse notation — only non-zero entries
    # sparse(NxN; 0,1=1 1,2=1 ... 98,99=1 99,99=1)
    entries = " ".join(f"{i},{i+1}=1" for i in range(n - 1))
    entries += f" {n-1},{n-1}=1"
    lines = [
        f"@auto_{n}",
        f"s s=onehot(0,{n})",
        f": step=transform(s;M=sparse({n}x{n};{entries}))",
        f"? done s[{n-1}]>=1",
    ]
    return "\n".join(lines)


PYTHON_LARGE = _gen_python_large(100)
AXOL_JSON_LARGE = _gen_axol_json_large(100)
AXOL_DSL_LARGE = _gen_axol_dsl_large(100)


# ===================================================================
# Test data
# ===================================================================

CASES = [
    ("Counter (0->5)",         PYTHON_COUNTER, AXOL_JSON_COUNTER, AXOL_DSL_COUNTER),
    ("State Machine (3st)",    PYTHON_SM,      AXOL_JSON_SM,      AXOL_DSL_SM),
    ("HP Decay (3 rounds)",    PYTHON_HP,      AXOL_JSON_HP,      AXOL_DSL_HP),
    ("RPG Damage Calc",        PYTHON_RPG,     AXOL_JSON_RPG,     AXOL_DSL_RPG),
    ("100-State Automaton",    PYTHON_LARGE,   AXOL_JSON_LARGE,   AXOL_DSL_LARGE),
]


# ===================================================================
# Tests
# ===================================================================

class TestTokenCost:
    @pytest.mark.parametrize(
        "name,py,axj,axd", CASES, ids=[c[0] for c in CASES]
    )
    def test_all_produce_valid_tokens(self, name, py, axj, axd):
        assert count_tokens(py) > 0
        assert count_tokens(axj) > 0
        assert count_tokens(axd) > 0

    def test_summary_table(self, capsys):
        rows = []
        for name, py, axj, axd in CASES:
            pt = count_tokens(py)
            jt = count_tokens(axj)
            dt = count_tokens(axd)
            rows.append((name, pt, jt, dt))

        with capsys.disabled():
            hdr = (
                f"\n{'='*88}\n"
                f"  TOKEN COST COMPARISON  (tokenizer: cl100k_base)\n"
                f"  Python source vs Axol JSON (compact) vs Axol DSL\n"
                f"{'='*88}\n"
                f"  {'Program':<24} {'Python':>7} {'Ax JSON':>8} {'Ax DSL':>7}"
                f" {'JSON/Py':>8} {'DSL/Py':>8} {'DSL save':>9}\n"
                f"{'-'*88}"
            )
            print(hdr)
            for name, pt, jt, dt in rows:
                jr = jt / pt if pt else 0
                dr = dt / pt if pt else 0
                sv = (1 - dr) * 100
                print(
                    f"  {name:<24} {pt:>7} {jt:>8} {dt:>7}"
                    f" {jr:>7.2f}x {dr:>7.2f}x {sv:>8.1f}%"
                )
            print(f"{'-'*88}")

            tot_p = sum(r[1] for r in rows)
            tot_j = sum(r[2] for r in rows)
            tot_d = sum(r[3] for r in rows)
            print(
                f"  {'TOTAL':<24} {tot_p:>7} {tot_j:>8} {tot_d:>7}"
                f" {tot_j/tot_p:>7.2f}x {tot_d/tot_p:>7.2f}x"
                f" {(1-tot_d/tot_p)*100:>8.1f}%"
            )
            print(f"{'='*88}")

    def test_dsl_examples(self, capsys):
        """Show side-by-side: Python vs Axol DSL for each program."""
        with capsys.disabled():
            for name, py, _, axd in CASES[:4]:  # skip large
                pt = count_tokens(py)
                dt = count_tokens(axd)
                saving = (1 - dt / pt) * 100

                print(f"\n{'='*60}")
                print(f"  {name}  (Python {pt} tok -> Axol DSL {dt} tok, {saving:.0f}% saved)")
                print(f"{'='*60}")
                print(f"\n  [Python]")
                for line in py.strip().splitlines():
                    print(f"    {line}")
                print(f"\n  [Axol DSL]")
                for line in axd.strip().splitlines():
                    print(f"    {line}")
            print()

    def test_scaling_analysis(self, capsys):
        """N-state automaton: token count scaling as N grows."""
        sizes = [5, 10, 25, 50, 100, 200]
        rows = []
        for n in sizes:
            pt = count_tokens(_gen_python_large(n))
            jt = count_tokens(_gen_axol_json_large(n))
            dt = count_tokens(_gen_axol_dsl_large(n))
            rows.append((n, pt, jt, dt))

        with capsys.disabled():
            print(f"\n{'='*78}")
            print(f"  SCALING: N-state automaton token cost vs state count")
            print(f"{'='*78}")
            print(f"  {'N':>5} {'Python':>8} {'Ax JSON':>9} {'Ax DSL':>8}"
                  f"  {'JSON/Py':>8} {'DSL/Py':>8}")
            print(f"{'-'*78}")
            for n, pt, jt, dt in rows:
                print(f"  {n:>5} {pt:>8} {jt:>9} {dt:>8}"
                      f"  {jt/pt:>7.2f}x {dt/pt:>7.2f}x")
            print(f"{'-'*78}")
            print(f"  Python/JSON: O(N^2) dense matrix")
            print(f"  Axol DSL:    O(N) sparse notation")
            print(f"{'='*78}")
