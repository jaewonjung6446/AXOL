import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ns = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]

# ── TOTAL end-to-end times (us) — real-world, GPU+CPU ──
total = {
    'GPU scatter (k=1)': [479.1, 663.2, 657.2, 1024.1, 45083.5, 117212.1],
    'GPU collapse k=3':  [657.2, 701.7, 836.9, 2085.6, 53272.5, 226062.6],
    'GPU collapse k=5':  [639.8, 818.9, 816.1, 2693.1, 59885.7, 296755.2],
    'GPU radix':         [1616.2, 1767.2, 1645.4, 2149.0, 7962.4, 56449.0],
    'CPU scatter':       [0.4, 1.7, 22.7, 402.7, 17526.6, 335077.9],
    'CPU std::sort':     [1.2, 8.2, 107.8, 1448.7, 17380.3, 198915.8],
    'CPU radix':         [1.3, 8.1, 74.8, 813.4, 12678.8, 155200.7],
}

colors = {
    'GPU scatter (k=1)': '#F0AD4E',
    'GPU collapse k=3':  '#5CB85C',
    'GPU collapse k=5':  '#8E44AD',
    'GPU radix':         '#E74C3C',
    'CPU scatter':       '#4A90D9',
    'CPU std::sort':     '#888888',
    'CPU radix':         '#D9534F',
}
markers = {
    'GPU scatter (k=1)': 'o',
    'GPU collapse k=3':  'P',
    'GPU collapse k=5':  '*',
    'GPU radix':         'X',
    'CPU scatter':       'D',
    'CPU std::sort':     's',
    'CPU radix':         '^',
}

# ── Per-element (ns/elem) ──
per_elem = {}
for label, times in total.items():
    per_elem[label] = [t * 1000.0 / n for t, n in zip(times, ns)]

# ── Slope ──
print("=" * 80)
print(f"{'n':>10} | {'GPU k=1':>9} | {'GPU k=3':>9} | {'GPU k=5':>9} | {'GPU rdx':>9} | {'CPU scat':>9} | {'CPU std':>9}")
print(f"{'':>10} | {'ns/elem':>9} | {'ns/elem':>9} | {'ns/elem':>9} | {'ns/elem':>9} | {'ns/elem':>9} | {'ns/elem':>9}")
print("-" * 80)
for i, n in enumerate(ns):
    print(f"{n:>10,} | {per_elem['GPU scatter (k=1)'][i]:>9.2f} | {per_elem['GPU collapse k=3'][i]:>9.2f} | "
          f"{per_elem['GPU collapse k=5'][i]:>9.2f} | {per_elem['GPU radix'][i]:>9.2f} | "
          f"{per_elem['CPU scatter'][i]:>9.2f} | {per_elem['CPU std::sort'][i]:>9.2f}")
print("=" * 80)

print("\n── d(log T)/d(log n) slope analysis (실전 전체 시간) ──")
print("   slope = 1.0 → O(n) total → O(1)/elem")
for label, times in total.items():
    slopes = []
    for i in range(1, len(ns)):
        s = np.log(times[i] / times[i-1]) / np.log(ns[i] / ns[i-1])
        slopes.append(s)
    large = slopes[2:]  # n=100K→1M, 1M→10M
    avg = np.mean(large) if large else 0
    print(f"  {label:20s}: slopes = {['%.3f' % s for s in slopes]}")
    print(f"  {'':20s}  avg (n>=100K) = {avg:.3f}  {'← O(1)' if avg < 1.1 else '← O(1) 아님'}")
    print()

# ── Plot ──
fig, axes = plt.subplots(1, 2, figsize=(17, 7))

# Left: per-element total time
ax1 = axes[0]
for label in ['CPU std::sort', 'CPU scatter', 'CPU radix',
              'GPU scatter (k=1)', 'GPU collapse k=3', 'GPU collapse k=5', 'GPU radix']:
    ax1.plot(ns, per_elem[label], marker=markers[label], color=colors[label],
             label=label, lw=2, ms=8)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('n', fontsize=13)
ax1.set_ylabel('Total time / n  (ns/elem)', fontsize=13)
ax1.set_title('Per-element TOTAL time (GPU+CPU 실전)\nO(1) = flat line', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, which='both', alpha=0.3)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Highlight the convergence zone
ax1.axvspan(1e5, 1e7, alpha=0.05, color='green')
ax1.text(5e5, 2, 'GPU 유효 구간', fontsize=10, color='green', ha='center')

# Right: total time log-log with O(n) reference
ax2 = axes[1]
for label in ['CPU std::sort', 'CPU scatter', 'GPU scatter (k=1)',
              'GPU collapse k=3', 'GPU collapse k=5', 'GPU radix']:
    ax2.plot(ns, total[label], marker=markers[label], color=colors[label],
             label=label, lw=2, ms=8)

# O(n) and O(n log n) reference
ref_n = np.array([1e4, 1e7])
k1_base = total['GPU scatter (k=1)'][3] / ns[3]
ax2.plot(ref_n, ref_n * k1_base, 'k--', alpha=0.3, lw=2, label='O(n) reference')
std_base = total['CPU std::sort'][3] / (ns[3] * np.log2(ns[3]))
ax2.plot(ref_n, ref_n * np.log2(ref_n) * std_base, 'k:', alpha=0.3, lw=2, label='O(n log n) ref')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('n', fontsize=13)
ax2.set_ylabel('Total time (us)', fontsize=13)
ax2.set_title('Total time (log-log, 실전)\nslope=1 → O(n) = O(1)/elem', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, which='both', alpha=0.3)
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.suptitle('AXOL 시간복잡도 O(1) 검증 — 실전 (GPU upload + compute + download)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('d:/Unity/Izakoza/AXOL/axol-lang/bench_o1_real.png', dpi=150, bbox_inches='tight')
print("\nSaved: bench_o1_real.png")
