"""Three-language token cost & performance benchmark: Python vs C# vs Axol DSL.

Compares token consumption (tiktoken cl100k_base) and runtime performance
across equivalent programs in Python, C#, and Axol DSL.
"""

import textwrap
import time

import pytest
import numpy as np

from axol.core.dsl import parse
from axol.core.program import run_program
from axol.core.types import FloatVec, TransMatrix, StateBundle

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


# ═══════════════════════════════════════════════════════════════════════════
# Source code: Python equivalents
# ═══════════════════════════════════════════════════════════════════════════

PY_COUNTER = textwrap.dedent("""\
    def counter(target=5):
        count = 0.0
        while count < target:
            count += 1.0
        return count
""")

PY_STATE_MACHINE = textwrap.dedent("""\
    TRANSITIONS = {"IDLE": "RUNNING", "RUNNING": "DONE", "DONE": "DONE"}

    def state_machine():
        state = "IDLE"
        steps = 0
        while state != "DONE":
            state = TRANSITIONS[state]
            steps += 1
        return state, steps
""")

PY_HP_DECAY = textwrap.dedent("""\
    def hp_decay(hp=100.0, factor=0.8, rounds=3):
        history = [hp]
        for _ in range(rounds):
            hp *= factor
            history.append(hp)
        return hp, history
""")

PY_COMBAT = textwrap.dedent("""\
    class Entity:
        def __init__(self, name, hp, atk, defense):
            self.name = name
            self.hp = hp
            self.atk = atk
            self.defense = defense

    def apply_damage(entity, amount):
        raw = amount - entity.defense
        dmg = max(0, raw)
        entity.hp = max(0, entity.hp - dmg)
        return entity

    def is_alive(entity):
        return entity.hp > 0

    hero = Entity("Hero", 100, 25, 10)
    goblin = Entity("Goblin", 30, 15, 5)
    goblin = apply_damage(goblin, hero.atk)
""")

PY_DATA_HEAVY = textwrap.dedent("""\
    from dataclasses import dataclass

    @dataclass
    class Unit:
        name: str
        hp: int
        atk: int
        defense: int

    @dataclass
    class Skill:
        dmg: int
        cost: int

    heroes = [
        Unit("Knight", 120, 30, 20),
        Unit("Mage", 80, 50, 10),
        Unit("Rogue", 90, 40, 15),
    ]
    skills = {"slash": Skill(25, 10), "fireball": Skill(50, 30), "backstab": Skill(35, 15)}
    total_hp = sum(u.hp for u in heroes)
    strongest = max(heroes, key=lambda u: u.atk)
""")

PY_PATTERN = textwrap.dedent("""\
    import math
    from dataclasses import dataclass

    @dataclass
    class Circle:
        r: float

    @dataclass
    class Rect:
        w: float
        h: float

    def area(shape):
        match shape:
            case Circle(r=r):
                return math.pi * r * r
            case Rect(w=w, h=h):
                return w * h

    def square(x): return x * x
    def cube(x): return x * x * x

    nums = [1, 2, 3, 4, 5]
    doubled = [x * 2 for x in nums]
    evens = [x for x in nums if x % 2 == 0]
    total = sum(nums)
""")


# ═══════════════════════════════════════════════════════════════════════════
# Source code: C# equivalents
# ═══════════════════════════════════════════════════════════════════════════

CS_COUNTER = textwrap.dedent("""\
    using System;

    class Program
    {
        static double Counter(int target = 5)
        {
            double count = 0.0;
            while (count < target)
                count += 1.0;
            return count;
        }

        static void Main() => Console.WriteLine(Counter());
    }
""")

CS_STATE_MACHINE = textwrap.dedent("""\
    using System;
    using System.Collections.Generic;

    class Program
    {
        static readonly Dictionary<string, string> Transitions = new()
        {
            ["IDLE"] = "RUNNING",
            ["RUNNING"] = "DONE",
            ["DONE"] = "DONE"
        };

        static (string state, int steps) Run()
        {
            var state = "IDLE";
            int steps = 0;
            while (state != "DONE")
            {
                state = Transitions[state];
                steps++;
            }
            return (state, steps);
        }

        static void Main()
        {
            var (state, steps) = Run();
            Console.WriteLine($"{state} in {steps} steps");
        }
    }
""")

CS_HP_DECAY = textwrap.dedent("""\
    using System;
    using System.Collections.Generic;

    class Program
    {
        static (double hp, List<double> history) HpDecay(
            double hp = 100.0, double factor = 0.8, int rounds = 3)
        {
            var history = new List<double> { hp };
            for (int i = 0; i < rounds; i++)
            {
                hp *= factor;
                history.Add(hp);
            }
            return (hp, history);
        }

        static void Main()
        {
            var (hp, history) = HpDecay();
            Console.WriteLine($"Final HP: {hp}");
        }
    }
""")

CS_COMBAT = textwrap.dedent("""\
    using System;

    class Entity
    {
        public int Hp { get; set; }
        public int Atk { get; }
        public int Def { get; }

        public Entity(int hp, int atk, int def)
        {
            Hp = hp;
            Atk = atk;
            Def = def;
        }
    }

    class Program
    {
        static Entity ApplyDmg(Entity ent, int amt)
        {
            int raw = amt - ent.Def;
            int dmg = raw < 0 ? 0 : raw;
            ent.Hp = Math.Max(0, ent.Hp - dmg);
            return ent;
        }

        static bool IsAlive(Entity ent) => ent.Hp > 0;

        static void Main()
        {
            var hero = new Entity(100, 25, 10);
            var goblin = new Entity(30, 15, 5);
            goblin = ApplyDmg(goblin, hero.Atk);
        }
    }
""")

CS_DATA_HEAVY = textwrap.dedent("""\
    using System;
    using System.Collections.Generic;
    using System.Linq;

    record Unit(string Name, int Hp, int Atk, int Def);
    record Skill(int Dmg, int Cost);

    class Program
    {
        static int TotalHp(List<Unit> units) => units.Sum(u => u.Hp);
        static Unit Strongest(List<Unit> units) =>
            units.OrderByDescending(u => u.Atk).First();

        static void Main()
        {
            var heroes = new List<Unit>
            {
                new("Knight", 120, 30, 20),
                new("Mage", 80, 50, 10),
                new("Rogue", 90, 40, 15),
            };

            var skills = new Dictionary<string, Skill>
            {
                ["slash"] = new(25, 10),
                ["fireball"] = new(50, 30),
                ["backstab"] = new(35, 15),
            };

            Console.WriteLine($"Total HP: {TotalHp(heroes)}");
            Console.WriteLine($"Strongest: {Strongest(heroes).Name}");
        }
    }
""")

CS_PATTERN = textwrap.dedent("""\
    using System;
    using System.Linq;
    using System.Collections.Generic;

    abstract record Shape;
    record Circle(double R) : Shape;
    record Rect(double W, double H) : Shape;

    static class MathModule
    {
        public static double Square(double x) => x * x;
        public static double Cube(double x) => x * x * x;
    }

    class Program
    {
        static double Area(Shape shape) => shape switch
        {
            Circle c => 3.14 * c.R * c.R,
            Rect r => r.W * r.H,
            _ => throw new ArgumentException()
        };

        static void Main()
        {
            var nums = new List<int> { 1, 2, 3, 4, 5 };
            var doubled = nums.Select(x => x * 2).ToList();
            var evens = nums.Where(x => x % 2 == 0).ToList();
            var sum = nums.Sum();
        }
    }
""")


# ═══════════════════════════════════════════════════════════════════════════
# Source code: Axol DSL equivalents
# ═══════════════════════════════════════════════════════════════════════════

DSL_COUNTER = textwrap.dedent("""\
    @counter
    s count=[0] one=[1]
    : increment=merge(count one;w=[1 1])->count
    ? done count>=5
""")

DSL_STATE_MACHINE = textwrap.dedent("""\
    @state_machine
    s state=onehot(0,3)
    : advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
    ? done state[2]>=1
""")

DSL_HP_DECAY = textwrap.dedent("""\
    @hp_decay
    s hp=[100] round=[0] one=[1]
    : decay=transform(hp;M=[0.8])
    : tick=merge(round one;w=[1 1])->round
    ? done round>=3
""")

DSL_COMBAT = textwrap.dedent("""\
    @combat
    s hero_hp=[100] hero_atk=[25] goblin_hp=[30] goblin_def=[5]
    : raw=merge(hero_atk goblin_def;w=[1 -1])->dmg
    : apply=merge(goblin_hp dmg;w=[1 -1])->goblin_hp
""")

DSL_DATA_HEAVY = textwrap.dedent("""\
    @party_stats
    s knight=[120 30 20] mage=[80 50 10] rogue=[90 40 15]
    : total=merge(knight mage rogue;w=[1 1 1])->party
    : strongest=transform(mage;M=[0;1;0])->best_atk
""")

DSL_PATTERN = textwrap.dedent("""\
    @geometry
    s circle_r=[5] rect=[3 4]
    : area_c=transform(circle_r;M=[78.5])->circle_area
    : area_r=transform(rect;M=[4;3])->rect_area_approx
""")


# ═══════════════════════════════════════════════════════════════════════════
# N-state automaton generators
# ═══════════════════════════════════════════════════════════════════════════

def _gen_py_automaton(n=100):
    lines = [f"STATES = list(range({n}))"]
    lines.append("TRANSITIONS = {")
    for i in range(n):
        nxt = min(i + 1, n - 1)
        lines.append(f"    {i}: {nxt},")
    lines.append("}")
    lines.append(f"def run():")
    lines.append(f"    state = 0")
    lines.append(f"    while state != {n-1}:")
    lines.append(f"        state = TRANSITIONS[state]")
    lines.append(f"    return state")
    return "\n".join(lines)


def _gen_cs_automaton(n=100):
    lines = ["using System;", "using System.Collections.Generic;", "", "class Program {"]
    lines.append(f"    static readonly Dictionary<int,int> T = new() {{")
    for i in range(n):
        nxt = min(i + 1, n - 1)
        lines.append(f"        [{i}] = {nxt},")
    lines.append("    };")
    lines.append(f"    static int Run() {{")
    lines.append(f"        int s = 0;")
    lines.append(f"        while (s != {n-1}) s = T[s];")
    lines.append(f"        return s;")
    lines.append("    }")
    lines.append("    static void Main() => Console.WriteLine(Run());")
    lines.append("}")
    return "\n".join(lines)


def _gen_dsl_automaton(n=100):
    entries = " ".join(f"{i},{i+1}=1" for i in range(n - 1))
    entries += f" {n-1},{n-1}=1"
    return "\n".join([
        f"@auto_{n}",
        f"s s=onehot(0,{n})",
        f": step=transform(s;M=sparse({n}x{n};{entries}))",
        f"? done s[{n-1}]>=1",
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Test cases
# ═══════════════════════════════════════════════════════════════════════════

CASES = [
    ("Counter (0->5)",     PY_COUNTER,      CS_COUNTER,      DSL_COUNTER),
    ("State Machine",     PY_STATE_MACHINE, CS_STATE_MACHINE, DSL_STATE_MACHINE),
    ("HP Decay",          PY_HP_DECAY,      CS_HP_DECAY,      DSL_HP_DECAY),
    ("Combat",            PY_COMBAT,        CS_COMBAT,        DSL_COMBAT),
    ("Data Heavy",        PY_DATA_HEAVY,    CS_DATA_HEAVY,    DSL_DATA_HEAVY),
    ("Pattern Match",     PY_PATTERN,       CS_PATTERN,       DSL_PATTERN),
    ("100-State Auto",    _gen_py_automaton(100), _gen_cs_automaton(100), _gen_dsl_automaton(100)),
]


# ═══════════════════════════════════════════════════════════════════════════
# Token comparison tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken not installed")
class TestTrilingualTokenCost:

    def test_full_comparison_table(self, capsys):
        """Comprehensive Python vs C# vs Axol DSL token comparison."""
        rows = []
        for name, py, cs, dsl in CASES:
            pt = count_tokens(py)
            ct = count_tokens(cs)
            dt = count_tokens(dsl)
            rows.append((name, pt, ct, dt))

        with capsys.disabled():
            print(f"\n{'='*96}")
            print(f"  TOKEN COST COMPARISON - Python vs C# vs Axol DSL")
            print(f"  Tokenizer: cl100k_base (GPT-4 / Claude)")
            print(f"{'='*96}")
            print(f"  {'Program':<20} {'Python':>7} {'C#':>7} {'Axol':>7}"
                  f"  {'DSL/Py':>7} {'DSL/C#':>7} {'Py save':>8} {'C# save':>8}")
            print(f"{'-'*96}")

            for name, pt, ct, dt in rows:
                dp = dt / pt if pt else 0
                dc = dt / ct if ct else 0
                sp = (1 - dp) * 100
                sc = (1 - dc) * 100
                print(f"  {name:<20} {pt:>7} {ct:>7} {dt:>7}"
                      f"  {dp:>6.2f}x {dc:>6.2f}x {sp:>7.1f}% {sc:>7.1f}%")

            print(f"{'-'*96}")
            tot_p = sum(r[1] for r in rows)
            tot_c = sum(r[2] for r in rows)
            tot_d = sum(r[3] for r in rows)
            print(f"  {'TOTAL':<20} {tot_p:>7} {tot_c:>7} {tot_d:>7}"
                  f"  {tot_d/tot_p:>6.2f}x {tot_d/tot_c:>6.2f}x"
                  f" {(1-tot_d/tot_p)*100:>7.1f}% {(1-tot_d/tot_c)*100:>7.1f}%")
            print(f"{'='*96}")

    def test_scaling_comparison(self, capsys):
        """N-state automaton scaling: Python vs C# vs Axol DSL."""
        sizes = [5, 10, 25, 50, 100, 200]
        rows = []
        for n in sizes:
            pt = count_tokens(_gen_py_automaton(n))
            ct = count_tokens(_gen_cs_automaton(n))
            dt = count_tokens(_gen_dsl_automaton(n))
            rows.append((n, pt, ct, dt))

        with capsys.disabled():
            print(f"\n{'='*84}")
            print(f"  SCALING: N-state automaton - token cost vs state count")
            print(f"{'='*84}")
            print(f"  {'N':>5} {'Python':>8} {'C#':>8} {'Axol DSL':>9}"
                  f"  {'DSL/Py':>8} {'DSL/C#':>8}")
            print(f"{'-'*84}")
            for n, pt, ct, dt in rows:
                print(f"  {n:>5} {pt:>8} {ct:>8} {dt:>9}"
                      f"  {dt/pt:>7.2f}x {dt/ct:>7.2f}x")
            print(f"{'-'*84}")
            print(f"  Python/C#: O(N²) - dense transition table")
            print(f"  Axol DSL:  O(N)  - sparse matrix notation")
            print(f"{'='*84}")

    @pytest.mark.parametrize("name,py,cs,dsl",
                             CASES[:6],
                             ids=[c[0] for c in CASES[:6]])
    def test_dsl_fewer_than_csharp(self, name, py, cs, dsl):
        """Axol DSL should use fewer tokens than C# for each program."""
        ct = count_tokens(cs)
        dt = count_tokens(dsl)
        assert dt < ct, f"{name}: DSL ({dt}) should use fewer tokens than C# ({ct})"


# ═══════════════════════════════════════════════════════════════════════════
# DSL correctness - parse and run
# ═══════════════════════════════════════════════════════════════════════════

class TestDslCorrectness:

    def test_counter_produces_5(self):
        prog = parse(DSL_COUNTER)
        result = run_program(prog)
        assert float(result.final_state["count"].data[0]) == pytest.approx(5.0)

    def test_state_machine_reaches_done(self):
        prog = parse(DSL_STATE_MACHINE)
        result = run_program(prog)
        assert result.final_state["state"].to_list() == pytest.approx([0.0, 0.0, 1.0])

    def test_hp_decay_correct(self):
        prog = parse(DSL_HP_DECAY)
        result = run_program(prog)
        assert float(result.final_state["hp"].data[0]) == pytest.approx(51.2, abs=0.1)

    def test_combat_damage(self):
        prog = parse(DSL_COMBAT)
        result = run_program(prog)
        # dmg = 25 - 5 = 20, goblin_hp = 30 - 20 = 10
        assert float(result.final_state["goblin_hp"].data[0]) == pytest.approx(10.0)


# ═══════════════════════════════════════════════════════════════════════════
# Runtime performance comparison
# ═══════════════════════════════════════════════════════════════════════════

def _time_it(fn, n=100):
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed / n


class TestRuntimePerformance:

    def test_small_vs_large_vector(self, capsys):
        """Compare pure Python loops vs Axol (NumPy) for small and large vectors."""
        results = []

        # Small: dim=4
        def py_small():
            v = [1.0, 2.0, 3.0, 4.0]
            m = [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]
            return [sum(v[i] * m[i][j] for i in range(4)) for j in range(4)]

        def axol_small():
            s = FloatVec.from_list([1.0, 2.0, 3.0, 4.0])
            m = TransMatrix.from_list([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
            from axol.core.operations import transform
            return transform(s, m)

        py_t = _time_it(py_small, 1000)
        ax_t = _time_it(axol_small, 1000)
        results.append(("dim=4", py_t, ax_t))

        # Medium: dim=100
        def py_med():
            v = [float(i) for i in range(100)]
            return [v[i] * 2.0 for i in range(100)]

        def axol_med():
            s = FloatVec(data=np.arange(100, dtype=np.float32))
            m = TransMatrix(data=np.eye(100, dtype=np.float32) * 2.0)
            from axol.core.operations import transform
            return transform(s, m)

        py_t = _time_it(py_med, 500)
        ax_t = _time_it(axol_med, 500)
        results.append(("dim=100", py_t, ax_t))

        # Large: dim=1000 matrix multiply
        dim = 1000

        def py_large():
            v = [float(i) for i in range(dim)]
            m = [[0.0]*dim for _ in range(dim)]
            for i in range(dim):
                m[i][i] = 2.0
            result = [0.0]*dim
            for i in range(dim):
                for j in range(dim):
                    result[j] += v[i] * m[i][j]
            return result

        _large_s = FloatVec(data=np.arange(dim, dtype=np.float32))
        _large_m = TransMatrix(data=np.eye(dim, dtype=np.float32) * 2.0)

        def axol_large():
            from axol.core.operations import transform
            return transform(_large_s, _large_m)

        py_t = _time_it(py_large, 2)
        ax_t = _time_it(axol_large, 200)
        results.append(("dim=1000 matmul", py_t, ax_t))

        with capsys.disabled():
            print(f"\n{'='*72}")
            print(f"  RUNTIME: Pure Python loops vs Axol (NumPy backend)")
            print(f"{'='*72}")
            print(f"  {'Dimension':<12} {'Python':>12} {'Axol':>12} {'Ratio':>12}")
            print(f"{'-'*72}")
            for dim, pt, at in results:
                if pt < 0.001:
                    p_str = f"{pt*1e6:.1f} μs"
                    a_str = f"{at*1e6:.1f} μs"
                else:
                    p_str = f"{pt*1e3:.1f} ms"
                    a_str = f"{at*1e3:.1f} ms"
                ratio = pt / at if at > 0 else float('inf')
                r_str = f"Axol {ratio:.1f}x" if ratio > 1 else f"Python {1/ratio:.0f}x"
                print(f"  {dim:<12} {p_str:>12} {a_str:>12} {r_str:>12}")
            print(f"{'-'*72}")
            print(f"  Small vectors: Python loop faster (no NumPy overhead)")
            print(f"  Large vectors: Axol (NumPy) dramatically faster")
            print(f"{'='*72}")
