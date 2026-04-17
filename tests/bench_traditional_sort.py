"""Traditional sort baselines for comparison with AXOL."""
import time
import gc
import numpy as np
import sys

SIZES = [100, 300, 1000, 3000, 5000]
if "--big" in sys.argv:
    SIZES.append(10000)
REPEATS = 50


def bench(fn, warmup=3, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    gc.collect(); gc.disable()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    gc.enable()
    ts.sort()
    return ts[len(ts)//2]  # median ns


def fmt(ns):
    if ns < 1_000: return f"{ns:.0f}ns"
    if ns < 1_000_000: return f"{ns/1000:.1f}us"
    if ns < 1_000_000_000: return f"{ns/1e6:.2f}ms"
    return f"{ns/1e9:.2f}s"


print(f"{'n':>6} {'np.sort':>12} {'np.argsort':>12} {'py sorted':>12} {'mergesort':>12} {'heapsort':>12}")
print("-" * 72)

for n in SIZES:
    rng = np.random.default_rng(42)
    X_np = rng.uniform(0, 1, n).astype(np.float32)
    X_py = X_np.tolist()

    t_np_sort   = bench(lambda: np.sort(X_np, kind="quicksort"))
    t_np_argsort = bench(lambda: np.argsort(X_np, kind="quicksort"))
    t_py_sorted = bench(lambda: sorted(X_py))
    t_np_merge   = bench(lambda: np.sort(X_np, kind="mergesort"))
    t_np_heap    = bench(lambda: np.sort(X_np, kind="heapsort"))

    print(f"{n:>6} {fmt(t_np_sort):>12} {fmt(t_np_argsort):>12} {fmt(t_py_sorted):>12} "
          f"{fmt(t_np_merge):>12} {fmt(t_np_heap):>12}")
