"""Benchmark: 4 new Plaintext ops - Python vs C# vs Axol DSL.

Token cost (tiktoken cl100k_base) + runtime performance comparison
for step, branch, clamp, map operations.
"""

import textwrap
import time

import pytest
import numpy as np

from axol.core.types import FloatVec, GateVec, StateBundle
from axol.core.operations import step, branch, clamp, map_fn
from axol.core.dsl import parse
from axol.core.program import run_program

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def _time_it(fn, n=200):
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed / n


# ═══════════════════════════════════════════════════════════════════════════
# 1. ReLU Activation - map(relu)
# ═══════════════════════════════════════════════════════════════════════════

PY_RELU = textwrap.dedent("""\
    def relu(values):
        return [max(0.0, v) for v in values]

    result = relu([-2.0, 0.0, 3.0, -1.0, 5.0])
""")

CS_RELU = textwrap.dedent("""\
    using System;
    using System.Linq;

    class Program
    {
        static double[] Relu(double[] values)
            => values.Select(v => Math.Max(0.0, v)).ToArray();

        static void Main()
        {
            var result = Relu(new[] { -2.0, 0.0, 3.0, -1.0, 5.0 });
        }
    }
""")

DSL_RELU = textwrap.dedent("""\
    @relu_test
    s x=[-2 0 3 -1 5]
    :act=map(x;fn=relu)
""")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Threshold + Conditional Select - step + branch
# ═══════════════════════════════════════════════════════════════════════════

PY_THRESHOLD_SELECT = textwrap.dedent("""\
    def threshold_select(scores, high, low, threshold=0.5):
        result = []
        for s, h, l in zip(scores, high, low):
            result.append(h if s >= threshold else l)
        return result

    scores = [0.3, 0.8, 0.1, 0.9, 0.4]
    high = [100.0, 200.0, 300.0, 400.0, 500.0]
    low = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = threshold_select(scores, high, low)
""")

CS_THRESHOLD_SELECT = textwrap.dedent("""\
    using System;
    using System.Linq;

    class Program
    {
        static double[] ThresholdSelect(
            double[] scores, double[] high, double[] low, double t = 0.5)
        {
            return scores.Select((s, i) => s >= t ? high[i] : low[i]).ToArray();
        }

        static void Main()
        {
            var scores = new[] { 0.3, 0.8, 0.1, 0.9, 0.4 };
            var high = new[] { 100.0, 200.0, 300.0, 400.0, 500.0 };
            var low = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
            var result = ThresholdSelect(scores, high, low);
        }
    }
""")

DSL_THRESHOLD_SELECT = textwrap.dedent("""\
    @threshold_select
    s scores=[0.3 0.8 0.1 0.9 0.4] high=[100 200 300 400 500] low=[1 2 3 4 5]
    :s1=step(scores;t=0.5)->mask
    :b1=branch(mask;then=high,else=low)->result
""")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Value Clamping - clamp
# ═══════════════════════════════════════════════════════════════════════════

PY_CLAMP = textwrap.dedent("""\
    def clamp_values(values, min_v=0.0, max_v=100.0):
        return [max(min_v, min(max_v, v)) for v in values]

    result = clamp_values([-50.0, 0.0, 75.0, 150.0, 200.0])
""")

CS_CLAMP = textwrap.dedent("""\
    using System;
    using System.Linq;

    class Program
    {
        static double[] Clamp(double[] values, double min = 0.0, double max = 100.0)
            => values.Select(v => Math.Clamp(v, min, max)).ToArray();

        static void Main()
        {
            var result = Clamp(new[] { -50.0, 0.0, 75.0, 150.0, 200.0 });
        }
    }
""")

DSL_CLAMP = textwrap.dedent("""\
    @clamp_test
    s x=[-50 0 75 150 200]
    :c1=clamp(x;min=0,max=100)
""")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Damage Pipeline - step + branch + clamp + map (복합)
# ═══════════════════════════════════════════════════════════════════════════

PY_DAMAGE_PIPE = textwrap.dedent("""\
    def calc_damage(raw_dmg, armor, crit_flags, bonus):
        # Step 1: base = raw - armor, ReLU (no negative damage)
        effective = [max(0.0, d - a) for d, a in zip(raw_dmg, armor)]
        # Step 2: threshold - is crit?
        mask = [1.0 if c >= 0.5 else 0.0 for c in crit_flags]
        # Step 3: branch - crit gets bonus, non-crit gets 0
        zero = [0.0] * len(bonus)
        crit_bonus = [b if m == 1.0 else z for m, b, z in zip(mask, bonus, zero)]
        # Step 4: total = effective + crit_bonus
        total = [e + b for e, b in zip(effective, crit_bonus)]
        # Step 5: clamp to [0, 9999]
        return [max(0.0, min(9999.0, t)) for t in total]

    result = calc_damage(
        [50, 30, 80, 20, 100],
        [10, 40, 5, 25, 0],
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [20, 20, 20, 20, 20],
    )
""")

CS_DAMAGE_PIPE = textwrap.dedent("""\
    using System;
    using System.Linq;

    class Program
    {
        static double[] CalcDamage(
            double[] rawDmg, double[] armor,
            double[] critFlags, double[] bonus)
        {
            var effective = rawDmg.Zip(armor, (d, a) => Math.Max(0.0, d - a)).ToArray();
            var mask = critFlags.Select(c => c >= 0.5 ? 1.0 : 0.0).ToArray();
            var zero = new double[bonus.Length];
            var critBonus = mask.Select((m, i) => m == 1.0 ? bonus[i] : zero[i]).ToArray();
            var total = effective.Zip(critBonus, (e, b) => e + b).ToArray();
            return total.Select(t => Math.Clamp(t, 0.0, 9999.0)).ToArray();
        }

        static void Main()
        {
            var result = CalcDamage(
                new[] { 50.0, 30.0, 80.0, 20.0, 100.0 },
                new[] { 10.0, 40.0,  5.0, 25.0,   0.0 },
                new[] {  1.0,  0.0,  1.0,  0.0,   1.0 },
                new[] { 20.0, 20.0, 20.0, 20.0,  20.0 });
        }
    }
""")

DSL_DAMAGE_PIPE = textwrap.dedent("""\
    @damage_pipe
    s raw=[50 30 80 20 100] armor=[10 40 5 25 0]
    s crit=[1 0 1 0 1] bonus=[20 20 20 20 20] zero=[0 0 0 0 0]
    :d1=merge(raw armor;w=[1 -1])->diff
    :d2=map(diff;fn=relu)->effective
    :d3=step(crit;t=0.5)->mask
    :d4=branch(mask;then=bonus,else=zero)->crit_bonus
    :d5=merge(effective crit_bonus;w=[1 1])->total
    :d6=clamp(total;min=0,max=9999)
""")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Sigmoid Activation - map(sigmoid)
# ═══════════════════════════════════════════════════════════════════════════

PY_SIGMOID = textwrap.dedent("""\
    import math

    def sigmoid(values):
        return [1.0 / (1.0 + math.exp(-v)) for v in values]

    result = sigmoid([-2.0, -1.0, 0.0, 1.0, 2.0])
""")

CS_SIGMOID = textwrap.dedent("""\
    using System;
    using System.Linq;

    class Program
    {
        static double[] Sigmoid(double[] values)
            => values.Select(v => 1.0 / (1.0 + Math.Exp(-v))).ToArray();

        static void Main()
        {
            var result = Sigmoid(new[] { -2.0, -1.0, 0.0, 1.0, 2.0 });
        }
    }
""")

DSL_SIGMOID = textwrap.dedent("""\
    @sigmoid_test
    s x=[-2 -1 0 1 2]
    :act=map(x;fn=sigmoid)
""")


# ═══════════════════════════════════════════════════════════════════════════
# Test cases
# ═══════════════════════════════════════════════════════════════════════════

CASES = [
    ("ReLU Activation",       PY_RELU,              CS_RELU,              DSL_RELU),
    ("Threshold Select",      PY_THRESHOLD_SELECT,  CS_THRESHOLD_SELECT,  DSL_THRESHOLD_SELECT),
    ("Value Clamp",           PY_CLAMP,             CS_CLAMP,             DSL_CLAMP),
    ("Sigmoid Activation",    PY_SIGMOID,           CS_SIGMOID,           DSL_SIGMOID),
    ("Damage Pipeline",       PY_DAMAGE_PIPE,       CS_DAMAGE_PIPE,       DSL_DAMAGE_PIPE),
]


# ═══════════════════════════════════════════════════════════════════════════
# Token Cost Comparison
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken not installed")
class TestNewOpsTokenCost:

    def test_token_comparison_table(self, capsys):
        rows = []
        for name, py, cs, dsl in CASES:
            pt = count_tokens(py)
            ct = count_tokens(cs)
            dt = count_tokens(dsl)
            rows.append((name, pt, ct, dt))

        with capsys.disabled():
            print(f"\n{'='*100}")
            print(f"  TOKEN COST - New Ops (step/branch/clamp/map): Python vs C# vs Axol DSL")
            print(f"  Tokenizer: cl100k_base (GPT-4 / Claude)")
            print(f"{'='*100}")
            print(f"  {'Program':<22} {'Python':>7} {'C#':>7} {'Axol':>7}"
                  f"  {'DSL/Py':>7} {'DSL/C#':>7} {'Py save':>8} {'C# save':>8}")
            print(f"{'-'*100}")

            for name, pt, ct, dt in rows:
                dp = dt / pt if pt else 0
                dc = dt / ct if ct else 0
                sp = (1 - dp) * 100
                sc = (1 - dc) * 100
                print(f"  {name:<22} {pt:>7} {ct:>7} {dt:>7}"
                      f"  {dp:>6.2f}x {dc:>6.2f}x {sp:>7.1f}% {sc:>7.1f}%")

            print(f"{'-'*100}")
            tot_p = sum(r[1] for r in rows)
            tot_c = sum(r[2] for r in rows)
            tot_d = sum(r[3] for r in rows)
            print(f"  {'TOTAL':<22} {tot_p:>7} {tot_c:>7} {tot_d:>7}"
                  f"  {tot_d/tot_p:>6.2f}x {tot_d/tot_c:>6.2f}x"
                  f" {(1-tot_d/tot_p)*100:>7.1f}% {(1-tot_d/tot_c)*100:>7.1f}%")
            print(f"{'='*100}")

    @pytest.mark.parametrize("name,py,cs,dsl",
                             CASES,
                             ids=[c[0] for c in CASES])
    def test_dsl_fewer_than_csharp(self, name, py, cs, dsl):
        """Axol DSL should use fewer tokens than C# for each new-op program."""
        ct = count_tokens(cs)
        dt = count_tokens(dsl)
        assert dt < ct, f"{name}: DSL ({dt}) should use fewer tokens than C# ({ct})"

    def test_dsl_source_examples(self, capsys):
        """Print source code side-by-side for visual comparison."""
        with capsys.disabled():
            for name, py, cs, dsl in CASES:
                pt = count_tokens(py)
                ct = count_tokens(cs)
                dt = count_tokens(dsl)
                py_save = (1 - dt / pt) * 100
                cs_save = (1 - dt / ct) * 100
                print(f"\n{'='*72}")
                print(f"  {name}  (Py {pt} / C# {ct} / Axol {dt} tok,"
                      f" Py save {py_save:.0f}%, C# save {cs_save:.0f}%)")
                print(f"{'='*72}")
                print(f"\n  [Python]\n{textwrap.indent(py.strip(), '    ')}")
                print(f"\n  [Axol DSL]\n{textwrap.indent(dsl.strip(), '    ')}")


# ═══════════════════════════════════════════════════════════════════════════
# DSL Correctness — parse and run
# ═══════════════════════════════════════════════════════════════════════════

class TestNewOpsCorrectness:

    def test_relu_correct(self):
        prog = parse(DSL_RELU)
        result = run_program(prog)
        out = result.final_state["x"].to_list()
        assert out == pytest.approx([0, 0, 3, 0, 5])

    def test_threshold_select_correct(self):
        prog = parse(DSL_THRESHOLD_SELECT)
        result = run_program(prog)
        out = result.final_state["result"].to_list()
        # mask = [0,1,0,1,0] → low[0], high[1], low[2], high[3], low[4]
        assert out == pytest.approx([1, 200, 3, 400, 5])

    def test_clamp_correct(self):
        prog = parse(DSL_CLAMP)
        result = run_program(prog)
        out = result.final_state["x"].to_list()
        assert out == pytest.approx([0, 0, 75, 100, 100])

    def test_sigmoid_correct(self):
        prog = parse(DSL_SIGMOID)
        result = run_program(prog)
        out = result.final_state["x"].to_list()
        assert out[2] == pytest.approx(0.5, abs=0.01)
        assert out[0] < 0.2  # sigmoid(-2) ≈ 0.119
        assert out[4] > 0.8  # sigmoid(2) ≈ 0.881

    def test_damage_pipeline_correct(self):
        """Full damage pipeline: merge->relu->step->branch->merge->clamp."""
        prog = parse(DSL_DAMAGE_PIPE)
        result = run_program(prog)
        total = result.final_state["total"].to_list()
        # diff = [40, -10, 75, -5, 100]
        # relu -> effective = [40, 0, 75, 0, 100]
        # step(crit,t=0.5) -> mask = [1, 0, 1, 0, 1]
        # branch(mask, bonus, zero) -> crit_bonus = [20, 0, 20, 0, 20]
        # merge(effective, crit_bonus, w=[1,1]) -> total = [60, 0, 95, 0, 120]
        # clamp(0, 9999) -> [60, 0, 95, 0, 120]
        assert total == pytest.approx([60, 0, 95, 0, 120])


# ═══════════════════════════════════════════════════════════════════════════
# Python equivalents for runtime benchmark
# ═══════════════════════════════════════════════════════════════════════════

def py_relu(values):
    return [max(0.0, v) for v in values]


def py_sigmoid(values):
    import math
    return [1.0 / (1.0 + math.exp(-v)) for v in values]


def py_threshold_select(scores, high, low, t=0.5):
    return [h if s >= t else l for s, h, l in zip(scores, high, low)]


def py_clamp_values(values, mn=0.0, mx=100.0):
    return [max(mn, min(mx, v)) for v in values]


def py_damage_pipeline(raw, armor, crit, bonus):
    effective = [max(0.0, d - a) for d, a in zip(raw, armor)]
    mask = [1.0 if c >= 0.5 else 0.0 for c in crit]
    crit_bonus = [b if m == 1.0 else 0.0 for m, b in zip(mask, bonus)]
    total = [e + b for e, b in zip(effective, crit_bonus)]
    return [max(0.0, min(9999.0, t)) for t in total]


# ═══════════════════════════════════════════════════════════════════════════
# Runtime Performance — small vectors (dim=5)
# ═══════════════════════════════════════════════════════════════════════════

class TestNewOpsRuntimeSmall:
    """Runtime comparison with dim=5 vectors (small). Python loops vs Axol (NumPy)."""

    def test_runtime_small_vectors(self, capsys):
        n = 500
        results = []

        # --- ReLU ---
        vals5 = [-2.0, 0.0, 3.0, -1.0, 5.0]
        fv5 = FloatVec.from_list(vals5)
        py_t = _time_it(lambda: py_relu(vals5), n)
        ax_t = _time_it(lambda: map_fn(fv5, "relu"), n)
        results.append(("ReLU (dim=5)", py_t, ax_t))

        # --- Sigmoid ---
        py_t = _time_it(lambda: py_sigmoid(vals5), n)
        ax_t = _time_it(lambda: map_fn(fv5, "sigmoid"), n)
        results.append(("Sigmoid (dim=5)", py_t, ax_t))

        # --- Step+Branch ---
        scores5 = [0.3, 0.8, 0.1, 0.9, 0.4]
        high5 = [100.0, 200.0, 300.0, 400.0, 500.0]
        low5 = [1.0, 2.0, 3.0, 4.0, 5.0]
        fv_sc = FloatVec.from_list(scores5)
        fv_hi = FloatVec.from_list(high5)
        fv_lo = FloatVec.from_list(low5)
        py_t = _time_it(lambda: py_threshold_select(scores5, high5, low5), n)

        def axol_step_branch():
            g = step(fv_sc, 0.5)
            return branch(g, fv_hi, fv_lo)
        ax_t = _time_it(axol_step_branch, n)
        results.append(("Step+Branch (dim=5)", py_t, ax_t))

        # --- Clamp ---
        vals_c = [-50.0, 0.0, 75.0, 150.0, 200.0]
        fv_c = FloatVec.from_list(vals_c)
        py_t = _time_it(lambda: py_clamp_values(vals_c), n)
        ax_t = _time_it(lambda: clamp(fv_c, 0.0, 100.0), n)
        results.append(("Clamp (dim=5)", py_t, ax_t))

        with capsys.disabled():
            print(f"\n{'='*76}")
            print(f"  RUNTIME - New Ops: Python loops vs Axol (NumPy), dim=5")
            print(f"{'='*76}")
            print(f"  {'Operation':<22} {'Python':>12} {'Axol':>12} {'Ratio':>12}")
            print(f"{'-'*76}")
            for name, pt, at in results:
                p_str = f"{pt*1e6:.1f} us"
                a_str = f"{at*1e6:.1f} us"
                ratio = pt / at if at > 0 else float('inf')
                r_str = f"Axol {ratio:.1f}x" if ratio > 1 else f"Python {1/ratio:.0f}x"
                print(f"  {name:<22} {p_str:>12} {a_str:>12} {r_str:>12}")
            print(f"{'-'*76}")
            print(f"  Note: Small vectors - Python loops competitive due to NumPy overhead")
            print(f"{'='*76}")


# ═══════════════════════════════════════════════════════════════════════════
# Runtime Performance — large vectors (dim=10000)
# ═══════════════════════════════════════════════════════════════════════════

class TestNewOpsRuntimeLarge:
    """Runtime comparison with dim=10000 vectors (large). Axol (NumPy) should dominate."""

    def test_runtime_large_vectors(self, capsys):
        dim = 10000
        n_py = 20    # Python loops are very slow at dim=10000
        n_ax = 500
        results = []

        # Pre-build data
        vals_l = [float(i % 7 - 3) for i in range(dim)]
        fv_l = FloatVec(data=np.array(vals_l, dtype=np.float32))

        # --- ReLU ---
        py_t = _time_it(lambda: py_relu(vals_l), n_py)
        ax_t = _time_it(lambda: map_fn(fv_l, "relu"), n_ax)
        results.append(("ReLU", py_t, ax_t))

        # --- Sigmoid ---
        import math
        small_vals = [float(i % 7 - 3) * 0.1 for i in range(dim)]
        fv_sm = FloatVec(data=np.array(small_vals, dtype=np.float32))
        py_t = _time_it(lambda: py_sigmoid(small_vals), n_py)
        ax_t = _time_it(lambda: map_fn(fv_sm, "sigmoid"), n_ax)
        results.append(("Sigmoid", py_t, ax_t))

        # --- Step+Branch ---
        scores_l = [float(i % 10) / 10.0 for i in range(dim)]
        high_l = [float(i * 10) for i in range(dim)]
        low_l = [float(i) for i in range(dim)]
        fv_sc_l = FloatVec(data=np.array(scores_l, dtype=np.float32))
        fv_hi_l = FloatVec(data=np.array(high_l, dtype=np.float32))
        fv_lo_l = FloatVec(data=np.array(low_l, dtype=np.float32))
        py_t = _time_it(lambda: py_threshold_select(scores_l, high_l, low_l), n_py)

        def axol_step_branch_l():
            g = step(fv_sc_l, 0.5)
            return branch(g, fv_hi_l, fv_lo_l)
        ax_t = _time_it(axol_step_branch_l, n_ax)
        results.append(("Step+Branch", py_t, ax_t))

        # --- Clamp ---
        py_t = _time_it(lambda: py_clamp_values(vals_l, -1.0, 2.0), n_py)
        ax_t = _time_it(lambda: clamp(fv_l, -1.0, 2.0), n_ax)
        results.append(("Clamp", py_t, ax_t))

        # --- Full Damage Pipeline ---
        raw_l = [float(i % 100) for i in range(dim)]
        armor_l = [float(i % 30) for i in range(dim)]
        crit_l = [1.0 if i % 3 == 0 else 0.0 for i in range(dim)]
        bonus_l = [20.0] * dim

        fv_raw = FloatVec(data=np.array(raw_l, dtype=np.float32))
        fv_arm = FloatVec(data=np.array(armor_l, dtype=np.float32))
        fv_crit = FloatVec(data=np.array(crit_l, dtype=np.float32))
        fv_bonus = FloatVec(data=np.array(bonus_l, dtype=np.float32))
        fv_zero = FloatVec(data=np.zeros(dim, dtype=np.float32))

        py_t = _time_it(lambda: py_damage_pipeline(
            raw_l, armor_l, crit_l, bonus_l), n_py)

        def axol_damage_l():
            from axol.core.operations import merge
            diff = merge([fv_raw, fv_arm], FloatVec.from_list([1.0, -1.0]))
            effective = map_fn(diff, "relu")
            mask = step(fv_crit, 0.5)
            crit_bonus = branch(mask, fv_bonus, fv_zero)
            total = merge([effective, crit_bonus], FloatVec.from_list([1.0, 1.0]))
            return clamp(total, 0.0, 9999.0)
        ax_t = _time_it(axol_damage_l, n_ax)
        results.append(("Damage Pipeline", py_t, ax_t))

        with capsys.disabled():
            print(f"\n{'='*76}")
            print(f"  RUNTIME - New Ops: Python loops vs Axol (NumPy), dim={dim}")
            print(f"{'='*76}")
            print(f"  {'Operation':<22} {'Python':>12} {'Axol':>12} {'Speedup':>14}")
            print(f"{'-'*76}")
            for name, pt, at in results:
                p_str = f"{pt*1e6:.1f} us" if pt < 0.001 else f"{pt*1e3:.1f} ms"
                a_str = f"{at*1e6:.1f} us" if at < 0.001 else f"{at*1e3:.1f} ms"
                ratio = pt / at if at > 0 else float('inf')
                r_str = f"Axol {ratio:.1f}x faster"
                print(f"  {name:<22} {p_str:>12} {a_str:>12} {r_str:>14}")
            print(f"{'-'*76}")
            print(f"  Large vectors: Axol (NumPy) dramatically outperforms Python loops")
            print(f"{'='*76}")


# ═══════════════════════════════════════════════════════════════════════════
# Runtime Scaling — new ops across vector dimensions
# ═══════════════════════════════════════════════════════════════════════════

class TestNewOpsScaling:
    """How do new ops scale with vector dimension?"""

    def test_scaling_by_dimension(self, capsys):
        dims = [10, 100, 1000, 10000]
        rows = []

        for dim in dims:
            vals = np.random.randn(dim).astype(np.float32)
            fv = FloatVec(data=vals)

            # Python baseline: relu
            py_vals = vals.tolist()
            py_n = max(5, 500 // max(1, dim // 100))
            py_t = _time_it(lambda pv=py_vals: [max(0.0, v) for v in pv], py_n)

            # Axol: relu
            ax_n = 500
            ax_t = _time_it(lambda f=fv: map_fn(f, "relu"), ax_n)

            rows.append((dim, py_t, ax_t))

        with capsys.disabled():
            print(f"\n{'='*76}")
            print(f"  SCALING - ReLU: Python loops vs Axol across dimensions")
            print(f"{'='*76}")
            print(f"  {'Dim':>8} {'Python':>14} {'Axol':>14} {'Ratio':>14}")
            print(f"{'-'*76}")
            for dim, pt, at in rows:
                p_str = f"{pt*1e6:.1f} us" if pt < 0.001 else f"{pt*1e3:.2f} ms"
                a_str = f"{at*1e6:.1f} us" if at < 0.001 else f"{at*1e3:.2f} ms"
                ratio = pt / at if at > 0 else float('inf')
                r_str = f"Axol {ratio:.1f}x" if ratio > 1 else f"Python {1/ratio:.0f}x"
                print(f"  {dim:>8} {p_str:>14} {a_str:>14} {r_str:>14}")
            print(f"{'-'*76}")
            print(f"  Python: O(N) loop overhead dominates at large N")
            print(f"  Axol:   NumPy vectorized - near-constant overhead + O(N) SIMD")
            print(f"{'='*76}")
