import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ns = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]

# ── Compute-only times (us) — upload/download excluded ──

# GPU scatter k=1: hist + prefix + copy + scatter
gpu_scatter_compute = [
    99+104+89+92,       # n=100
    96+310+83+83,       # n=1K
    89+281+77+95,       # n=10K
    104+290+82+137,     # n=100K
    489+708+108+300,    # n=1M
    24438+3014+643+30893, # n=10M
]

# GPU collapse k=3: slit + hist + prefix + copy + scatter
gpu_k3_compute = [
    104+89+111+119+116,       # n=100
    93+78+278+78+80,          # n=1K
    100+79+313+85+95,         # n=10K
    189+90+516+81+120,        # n=100K
    1236+474+977+124+619,     # n=1M
    18532+13665+5394+1630+29435, # n=10M
]

# GPU collapse k=5: slit + hist + prefix + copy + scatter
gpu_k5_compute = [
    99+88+99+116+118,         # n=100
    109+91+321+89+93,         # n=1K
    102+82+281+77+82,         # n=10K
    236+115+632+123+123,      # n=100K
    1723+767+1093+305+931,    # n=1M
    20954+12283+7472+2740+28849, # n=10M
]

# CPU scatter (total, no overhead)
cpu_scatter = [0.4, 1.7, 22.7, 402.7, 17526.6, 335077.9]

# CPU std::sort for reference
cpu_std = [1.2, 8.2, 107.8, 1448.7, 17380.3, 198915.8]

# ── Per-element time (ns/elem) ──
def per_elem_ns(times, ns_list):
    return [t * 1000.0 / n for t, n in zip(times, ns_list)]

scatter_per = per_elem_ns(gpu_scatter_compute, ns)
k3_per = per_elem_ns(gpu_k3_compute, ns)
k5_per = per_elem_ns(gpu_k5_compute, ns)
cpu_scat_per = per_elem_ns(cpu_scatter, ns)
cpu_std_per = per_elem_ns(cpu_std, ns)

print("=" * 75)
print(f"{'n':>10} | {'GPU k=1':>10} | {'GPU k=3':>10} | {'GPU k=5':>10} | {'CPU scat':>10} | {'CPU std':>10}")
print(f"{'':>10} | {'(ns/elem)':>10} | {'(ns/elem)':>10} | {'(ns/elem)':>10} | {'(ns/elem)':>10} | {'(ns/elem)':>10}")
print("-" * 75)
for i, n in enumerate(ns):
    print(f"{n:>10,} | {scatter_per[i]:>10.2f} | {k3_per[i]:>10.2f} | {k5_per[i]:>10.2f} | {cpu_scat_per[i]:>10.2f} | {cpu_std_per[i]:>10.2f}")
print("=" * 75)

# Slope analysis: log(time)/log(n) between consecutive points
print("\n── Time complexity slope (d log T / d log n) ──")
print("   slope=1.0 → O(n) total → O(1)/elem")
print("   slope>1.0 → worse than O(n)")
print()
for label, times in [("GPU scatter k=1", gpu_scatter_compute),
                      ("GPU collapse k=3", gpu_k3_compute),
                      ("GPU collapse k=5", gpu_k5_compute),
                      ("CPU scatter", cpu_scatter),
                      ("CPU std::sort", cpu_std)]:
    slopes = []
    for i in range(1, len(ns)):
        if times[i] > 0 and times[i-1] > 0:
            s = np.log(times[i] / times[i-1]) / np.log(ns[i] / ns[i-1])
            slopes.append(s)
    # Show slopes for n >= 10K (where GPU overhead is amortized)
    print(f"  {label:20s}: slopes = {['%.3f' % s for s in slopes]}")
    large_slopes = slopes[2:]  # n=100K, 1M, 10M transitions
    if large_slopes:
        print(f"  {'':20s}  avg (n>=100K) = {np.mean(large_slopes):.3f}")
    print()

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: per-element time vs n
ax1.plot(ns, scatter_per, 'o-', color='#F0AD4E', label='GPU scatter (k=1)', lw=2, ms=8)
ax1.plot(ns, k3_per, 'P-', color='#5CB85C', label='GPU collapse k=3', lw=2, ms=8)
ax1.plot(ns, k5_per, '*-', color='#8E44AD', label='GPU collapse k=5', lw=2, ms=8)
ax1.plot(ns, cpu_scat_per, 'D-', color='#4A90D9', label='CPU scatter', lw=2, ms=8)
ax1.plot(ns, cpu_std_per, 's-', color='#888888', label='CPU std::sort (O(n log n))', lw=2, ms=8)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('n', fontsize=13)
ax1.set_ylabel('Compute time / n  (ns/elem)', fontsize=13)
ax1.set_title('Per-element compute time\n(O(1) = flat line)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, which='both', alpha=0.3)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Ideal O(1) reference
ax1.axhline(y=scatter_per[-1], color='#F0AD4E', ls=':', alpha=0.4)
ax1.axhline(y=k3_per[-1], color='#5CB85C', ls=':', alpha=0.4)

# Right: total compute time vs n (log-log, slope=1 means O(n))
ax2.plot(ns, gpu_scatter_compute, 'o-', color='#F0AD4E', label='GPU scatter (k=1)', lw=2, ms=8)
ax2.plot(ns, gpu_k3_compute, 'P-', color='#5CB85C', label='GPU collapse k=3', lw=2, ms=8)
ax2.plot(ns, gpu_k5_compute, '*-', color='#8E44AD', label='GPU collapse k=5', lw=2, ms=8)
ax2.plot(ns, cpu_scatter, 'D-', color='#4A90D9', label='CPU scatter', lw=2, ms=8)
ax2.plot(ns, cpu_std, 's-', color='#888888', label='CPU std::sort', lw=2, ms=8)

# O(n) reference line
ref_n = np.array([1e4, 1e7])
ref_t = ref_n * (gpu_scatter_compute[2] / ns[2])
ax2.plot(ref_n, ref_t, 'k--', alpha=0.3, lw=2, label='O(n) reference (slope=1)')

# O(n log n) reference
ref_nlogn = ref_n * np.log2(ref_n) * (cpu_std[2] / (ns[2] * np.log2(ns[2])))
ax2.plot(ref_n, ref_nlogn, 'k:', alpha=0.3, lw=2, label='O(n log n) reference')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('n', fontsize=13)
ax2.set_ylabel('Total compute time (us)', fontsize=13)
ax2.set_title('Total compute time (log-log)\nslope=1 → O(n), slope>1 → worse', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, which='both', alpha=0.3)
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.suptitle('AXOL Scatter/Collapse: Is it O(1) per element?\n(GPU compute only, no upload/download)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('d:/Unity/Izakoza/AXOL/axol-lang/bench_o1_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: bench_o1_analysis.png")
