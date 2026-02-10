"""Performance measurement and report generation for Axol.

Measures:
  1. Token cost comparison (Python vs C# vs Axol DSL)
  2. Runtime benchmarks at various dimensions
  3. Optimizer effect on transition count and runtime
  4. Encryption overhead
  5. Scaling analysis (N-state automatons)
  6. GPU backend comparison (numpy vs cupy if available)

Generates PERFORMANCE_REPORT.md at project root.
"""

import os
import time
import textwrap

import pytest
import numpy as np

from axol.core.types import FloatVec, GateVec, OneHotVec, TransMatrix, StateBundle
from axol.core.program import (
    TransformOp, MergeOp, CustomOp, Transition, Program, run_program,
)
from axol.core.dsl import parse
from axol.core.optimizer import optimize
from axol.core.encryption import (
    random_key, encrypt_program, decrypt_state,
)
from axol.core.backend import set_backend, get_backend_name

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Accumulate results across tests
_results: dict[str, object] = {}


def _time_fn(fn, n=100):
    """Time a function call, return average seconds."""
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n


# ═══════════════════════════════════════════════════════════════════════════
# 1. Token Cost Comparison
# ═══════════════════════════════════════════════════════════════════════════

PROGRAMS = {
    "counter": {
        "python": "def counter(t=5):\n  c=0.0\n  while c<t: c+=1.0\n  return c",
        "csharp": "float Counter(int t=5){float c=0;while(c<t)c+=1;return c;}",
        "axol": "@counter\ns count=[0] one=[1]\n: tick=merge(count one;w=[1 1])->count\n? done count>=5",
    },
    "hp_decay": {
        "python": "def hp_decay(hp=100,f=0.8,r=3):\n  for _ in range(r): hp*=f\n  return hp",
        "csharp": "float HpDecay(float hp=100,float f=0.8f,int r=3){for(int i=0;i<r;i++)hp*=f;return hp;}",
        "axol": "@hp_decay\ns hp=[100] round=[0] one=[1]\n: decay=transform(hp;M=[0.8])\n: tick=merge(round one;w=[1 1])->round\n? done round>=3",
    },
    "state_machine": {
        "python": "TRANS={'IDLE':'RUN','RUN':'DONE','DONE':'DONE'}\ndef sm():\n  s='IDLE'\n  while s!='DONE': s=TRANS[s]\n  return s",
        "csharp": "string SM(){var t=new Dictionary<string,string>{{\"IDLE\",\"RUN\"},{\"RUN\",\"DONE\"},{\"DONE\",\"DONE\"}};var s=\"IDLE\";while(s!=\"DONE\")s=t[s];return s;}",
        "axol": "@fsm\ns state=onehot(0,3)\n: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])\n? done state[2]>=1",
    },
}


class TestTokenCost:
    def test_01_token_cost(self, capsys):
        """Measure token counts for Python vs C# vs Axol."""
        rows = []
        for name, srcs in PROGRAMS.items():
            row = {
                "name": name,
                "python_tokens": len(srcs["python"].split()),
                "csharp_tokens": len(srcs["csharp"].split()),
                "axol_tokens": len(srcs["axol"].split()),
            }
            rows.append(row)
        _results["token_cost"] = rows

        with capsys.disabled():
            print(f"\n{'='*70}")
            print("  Token Cost Comparison (word-split estimate)")
            print(f"{'='*70}")
            print(f"  {'Program':<18} {'Python':>8} {'C#':>8} {'Axol':>8} {'Saving':>8}")
            print(f"{'-'*70}")
            for r in rows:
                saving = (1 - r["axol_tokens"] / r["python_tokens"]) * 100
                print(f"  {r['name']:<18} {r['python_tokens']:>8} {r['csharp_tokens']:>8} {r['axol_tokens']:>8} {saving:>7.0f}%")
            print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Runtime Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

class TestRuntimeBenchmarks:
    def test_02_runtime_dimensions(self, capsys):
        """Benchmark transform at various dimensions."""
        dims = [4, 100, 1000]
        rows = []

        for dim in dims:
            def run_transform():
                v = FloatVec(data=np.ones(dim, dtype=np.float32))
                M = TransMatrix(data=np.eye(dim, dtype=np.float32) * 2.0)
                prog = Program(
                    name="bench",
                    initial_state=StateBundle(vectors={"v": v}),
                    transitions=[Transition("t", TransformOp(key="v", matrix=M))],
                )
                run_program(prog)

            n_iter = 200 if dim <= 100 else 50
            avg = _time_fn(run_transform, n=n_iter)
            rows.append({"dim": dim, "avg_us": avg * 1e6, "n": n_iter})

        _results["runtime"] = rows

        with capsys.disabled():
            print(f"\n{'='*60}")
            print("  Runtime Benchmark (transform, single step)")
            print(f"{'='*60}")
            for r in rows:
                print(f"  dim={r['dim']:<6} {r['avg_us']:>10.1f} μs  ({r['n']} runs)")
            print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Optimizer Effect
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizerEffect:
    def test_03_optimizer_effect(self, capsys):
        """Compare transition counts and runtime before/after optimization."""
        source = "@pipe\ns v=[1 0 0]\n: t1=transform(v;M=[0 1 0;0 0 1;1 0 0])\n: t2=transform(v;M=[2 0 0;0 2 0;0 0 2])"
        prog = parse(source)
        prog_opt = optimize(prog)

        t_orig = _time_fn(lambda: run_program(prog), n=500)
        t_opt = _time_fn(lambda: run_program(prog_opt), n=500)

        _results["optimizer"] = {
            "orig_transitions": len(prog.transitions),
            "opt_transitions": len(prog_opt.transitions),
            "orig_us": t_orig * 1e6,
            "opt_us": t_opt * 1e6,
        }

        with capsys.disabled():
            print(f"\n{'='*60}")
            print("  Optimizer Effect (transform chain)")
            print(f"{'='*60}")
            print(f"  Original:  {len(prog.transitions)} transitions, {t_orig*1e6:.1f} μs")
            print(f"  Optimized: {len(prog_opt.transitions)} transitions, {t_opt*1e6:.1f} μs")
            if t_opt > 0:
                print(f"  Speedup:   {t_orig/t_opt:.2f}x")
            print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Encryption Overhead
# ═══════════════════════════════════════════════════════════════════════════

class TestEncryptionOverhead:
    def test_04_encryption_overhead(self, capsys):
        """Compare runtime of plaintext vs encrypted execution."""
        dim = 3
        source = "@fsm\ns state=onehot(0,3)\n: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])\n? done state[2]>=1"
        prog = parse(source)
        K = random_key(dim, seed=42)
        prog_enc = encrypt_program(prog, K, dim)

        # Run both
        result_plain = run_program(prog)
        result_enc = run_program(prog_enc)
        decrypted = decrypt_state(result_enc.final_state, K)

        # Verify correctness: compare only vectors with matching dimension
        for key in result_plain.final_state.keys():
            plain_vec = result_plain.final_state[key].data
            if plain_vec.ndim == 1 and plain_vec.shape[0] == dim and key in decrypted:
                dec_vec = decrypted[key].data
                np.testing.assert_allclose(dec_vec, plain_vec, atol=1e-2)

        # Benchmark
        t_plain = _time_fn(lambda: run_program(prog), n=300)
        t_enc = _time_fn(lambda: run_program(prog_enc), n=300)

        _results["encryption"] = {
            "plain_us": t_plain * 1e6,
            "encrypted_us": t_enc * 1e6,
            "overhead": t_enc / t_plain if t_plain > 0 else 0,
        }

        with capsys.disabled():
            print(f"\n{'='*60}")
            print("  Encryption Overhead (3-state FSM)")
            print(f"{'='*60}")
            print(f"  Plaintext:  {t_plain*1e6:.1f} μs")
            print(f"  Encrypted:  {t_enc*1e6:.1f} μs")
            print(f"  Overhead:   {t_enc/t_plain:.2f}x" if t_plain > 0 else "  N/A")
            print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Scaling Analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestScalingAnalysis:
    def test_05_scaling_analysis(self, capsys):
        """Measure token count and runtime for N-state automatons."""
        sizes = [5, 10, 20, 50, 100]
        rows = []

        for n in sizes:
            # Build N-state automaton programmatically
            shift = np.zeros((n, n), dtype=np.float32)
            for i in range(n - 1):
                shift[i, i + 1] = 1.0
            shift[n - 1, n - 1] = 1.0

            state = StateBundle(vectors={
                "state": OneHotVec.from_index(0, n),
                "done": GateVec.from_list([0.0]),
            })

            def make_check(size):
                def check_done(s):
                    sc = s.copy()
                    if float(sc["state"].data[size - 1]) >= 0.99:
                        sc["done"] = GateVec.from_list([1.0])
                    return sc
                return check_done

            prog = Program(
                name=f"automaton_{n}",
                initial_state=state,
                transitions=[
                    Transition("advance", TransformOp(key="state", matrix=TransMatrix(data=shift))),
                    Transition("check", CustomOp(fn=make_check(n), label="check")),
                ],
                terminal_key="done",
                max_iterations=n + 5,
            )

            n_iter = 50 if n <= 50 else 10
            avg = _time_fn(lambda p=prog: run_program(p), n=n_iter)
            rows.append({"n": n, "avg_us": avg * 1e6})

        _results["scaling"] = rows

        with capsys.disabled():
            print(f"\n{'='*60}")
            print("  Scaling: N-State Automaton")
            print(f"{'='*60}")
            for r in rows:
                print(f"  N={r['n']:<5} {r['avg_us']:>10.1f} μs")
            print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. GPU Backend (informational)
# ═══════════════════════════════════════════════════════════════════════════

class TestGPUBackend:
    def test_06_gpu_backend(self, capsys):
        """Compare numpy vs cupy (if available)."""
        from axol.core.backend import set_backend, get_backend_name

        backends = ["numpy"]
        try:
            import cupy
            backends.append("cupy")
        except ImportError:
            pass

        results = {}
        dim = 100

        for backend_name in backends:
            set_backend(backend_name)

            v = FloatVec(data=np.ones(dim, dtype=np.float32))
            M = TransMatrix(data=np.eye(dim, dtype=np.float32) * 2.0)
            prog = Program(
                name="bench",
                initial_state=StateBundle(vectors={"v": v}),
                transitions=[Transition("t", TransformOp(key="v", matrix=M))],
            )
            avg = _time_fn(lambda: run_program(prog), n=200)
            results[backend_name] = avg * 1e6

        set_backend("numpy")
        _results["gpu"] = results

        with capsys.disabled():
            print(f"\n{'='*60}")
            print("  Backend Comparison (dim=100 transform)")
            print(f"{'='*60}")
            for name, us in results.items():
                print(f"  {name:<10} {us:>10.1f} μs")
            print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_report(results: dict) -> str:
    lines = []
    lines.append("# Axol Performance Report")
    lines.append("")
    lines.append("Auto-generated benchmark results.")
    lines.append("")

    # Token cost
    if "token_cost" in results:
        lines.append("## 1. Token Cost Comparison")
        lines.append("")
        lines.append("| Program | Python | C# | Axol | Saving vs Python |")
        lines.append("|---------|--------|----|------|------------------|")
        for r in results["token_cost"]:
            saving = (1 - r["axol_tokens"] / r["python_tokens"]) * 100
            lines.append(f"| {r['name']} | {r['python_tokens']} | {r['csharp_tokens']} | {r['axol_tokens']} | {saving:.0f}% |")
        lines.append("")

    # Runtime
    if "runtime" in results:
        lines.append("## 2. Runtime Benchmarks")
        lines.append("")
        lines.append("| Dimension | Avg Time (μs) |")
        lines.append("|-----------|---------------|")
        for r in results["runtime"]:
            lines.append(f"| {r['dim']} | {r['avg_us']:.1f} |")
        lines.append("")

    # Optimizer
    if "optimizer" in results:
        o = results["optimizer"]
        lines.append("## 3. Optimizer Effect")
        lines.append("")
        lines.append(f"- Original: {o['orig_transitions']} transitions, {o['orig_us']:.1f} μs")
        lines.append(f"- Optimized: {o['opt_transitions']} transitions, {o['opt_us']:.1f} μs")
        if o['opt_us'] > 0:
            lines.append(f"- Speedup: {o['orig_us']/o['opt_us']:.2f}x")
        lines.append("")

    # Encryption
    if "encryption" in results:
        e = results["encryption"]
        lines.append("## 4. Encryption Overhead")
        lines.append("")
        lines.append(f"- Plaintext: {e['plain_us']:.1f} μs")
        lines.append(f"- Encrypted: {e['encrypted_us']:.1f} μs")
        lines.append(f"- Overhead: {e['overhead']:.2f}x")
        lines.append("")

    # Scaling
    if "scaling" in results:
        lines.append("## 5. Scaling Analysis (N-State Automaton)")
        lines.append("")
        lines.append("| N States | Avg Time (μs) |")
        lines.append("|----------|---------------|")
        for r in results["scaling"]:
            lines.append(f"| {r['n']} | {r['avg_us']:.1f} |")
        lines.append("")

    # GPU
    if "gpu" in results:
        lines.append("## 6. Backend Comparison")
        lines.append("")
        lines.append("| Backend | Avg Time (μs) |")
        lines.append("|---------|---------------|")
        for name, us in results["gpu"].items():
            lines.append(f"| {name} | {us:.1f} |")
        lines.append("")

    return "\n".join(lines)


@pytest.fixture(scope="session", autouse=True)
def write_report(request):
    """Write the performance report at the end of the test session."""
    yield
    if _results:
        report = _generate_report(_results)
        report_path = os.path.join(PROJECT_ROOT, "PERFORMANCE_REPORT.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
