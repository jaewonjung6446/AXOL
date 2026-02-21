import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ns = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
ns_arr = np.array(ns)

# ── Total time (us) ──
t = {
    'CPU std::sort':     [1.2, 8.2, 107.8, 1448.7, 17380.3, 198915.8],
    'CPU radix':         [1.3, 8.1, 74.8, 813.4, 12678.8, 155200.7],
    'CPU scatter':       [0.4, 1.7, 22.7, 402.7, 17526.6, 335077.9],
    'GPU scatter (k=1)': [479.1, 663.2, 657.2, 1024.1, 45083.5, 117212.1],
    'GPU collapse k=3':  [657.2, 701.7, 836.9, 2085.6, 53272.5, 226062.6],
    'GPU collapse k=5':  [639.8, 818.9, 816.1, 2693.1, 59885.7, 296755.2],
    'GPU radix':         [1616.2, 1767.2, 1645.4, 2149.0, 7962.4, 56449.0],
}

# ── Compute-only (us) — no upload/download ──
c = {
    'GPU scatter (k=1)': [384, 572, 542, 613, 1605, 58988],
    'GPU collapse k=3':  [539, 607, 672, 996, 3430, 68656],
    'GPU collapse k=5':  [520, 703, 624, 1229, 4819, 72298],
    'GPU radix':         [1507, 1673, 1474, 1944, 6079, 38041],
}

acc = {
    'GPU scatter (k=1)': [64.0, 59.9, 63.1, 63.2, 65.4, 83.2],
    'GPU collapse k=3':  [84.0, 83.2, 84.1, 84.9, 87.7, 100.0],
    'GPU collapse k=5':  [90.0, 90.9, 90.2, 90.7, 93.3, 100.0],
    'GPU radix':         [100.0, 0.4, 0.1, 0.04, 0.06, 0.08],
    'CPU radix':         [100.0]*6,
}

colors_axol = {'GPU scatter (k=1)': '#F0AD4E', 'GPU collapse k=3': '#5CB85C', 'GPU collapse k=5': '#8E44AD'}
colors_radix = {'GPU radix': '#E74C3C', 'CPU radix': '#D9534F'}
markers = {'GPU scatter (k=1)': 'o', 'GPU collapse k=3': 'P', 'GPU collapse k=5': '*',
           'GPU radix': 'X', 'CPU radix': '^', 'CPU std::sort': 's', 'CPU scatter': 'D'}

fig = plt.figure(figsize=(18, 14))

# ═══ Row 1: Total time comparison ═══
ax1 = fig.add_subplot(2, 2, 1)
for label in ['GPU scatter (k=1)', 'GPU collapse k=3', 'GPU collapse k=5']:
    ax1.plot(ns, t[label], marker=markers[label], color=colors_axol[label], lw=2.5, ms=9, label=label)
for label in ['GPU radix', 'CPU radix']:
    ax1.plot(ns, t[label], marker=markers[label], color=colors_radix[label], lw=2.5, ms=9, label=label, ls='--')
ax1.plot(ns, t['CPU std::sort'], marker='s', color='#888', lw=1.5, ms=7, label='CPU std::sort', ls=':')

ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('n', fontsize=12); ax1.set_ylabel('Total time (us)', fontsize=12)
ax1.set_title('Total Time (GPU+CPU, upload/download 포함)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left'); ax1.grid(True, which='both', alpha=0.3)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# ═══ Row 1: Compute-only comparison ═══
ax2 = fig.add_subplot(2, 2, 2)
for label in ['GPU scatter (k=1)', 'GPU collapse k=3', 'GPU collapse k=5']:
    ax2.plot(ns, c[label], marker=markers[label], color=colors_axol[label], lw=2.5, ms=9, label=label)
ax2.plot(ns, c['GPU radix'], marker='X', color='#E74C3C', lw=2.5, ms=9, label='GPU radix', ls='--')

ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('n', fontsize=12); ax2.set_ylabel('Compute time (us)', fontsize=12)
ax2.set_title('GPU Compute Only (upload/download 제외)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left'); ax2.grid(True, which='both', alpha=0.3)
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Annotate crossover
ax2.annotate('AXOL wins\n(n < 1M)', xy=(1e5, 800), fontsize=10, color='#5CB85C',
             fontweight='bold', ha='center')
ax2.annotate('Radix wins\n(n >= 10M)', xy=(1e7, 30000), fontsize=10, color='#E74C3C',
             fontweight='bold', ha='center')

# ═══ Row 2: AXOL / Radix ratio ═══
ax3 = fig.add_subplot(2, 2, 3)

# Compute-only ratio: AXOL / GPU radix (< 1 means AXOL wins)
for label in ['GPU scatter (k=1)', 'GPU collapse k=3', 'GPU collapse k=5']:
    ratio = [c[label][i] / c['GPU radix'][i] for i in range(len(ns))]
    ax3.plot(ns, ratio, marker=markers[label], color=colors_axol[label], lw=2.5, ms=9, label=f'{label} / GPU radix')

ax3.axhline(y=1.0, color='black', ls='--', lw=1.5, alpha=0.5)
ax3.fill_between([50, 2e7], 0, 1, alpha=0.05, color='green')
ax3.fill_between([50, 2e7], 1, 5, alpha=0.05, color='red')
ax3.text(200, 0.15, 'AXOL wins', fontsize=11, color='green', fontweight='bold')
ax3.text(200, 2.5, 'Radix wins', fontsize=11, color='red', fontweight='bold')

ax3.set_xscale('log')
ax3.set_xlabel('n', fontsize=12); ax3.set_ylabel('AXOL time / Radix time', fontsize=12)
ax3.set_title('AXOL vs Radix 속도비 (GPU compute only)\n< 1 = AXOL이 빠름', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9); ax3.grid(True, which='both', alpha=0.3)
ax3.set_ylim(0, 3.5)
ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# ═══ Row 2: Accuracy comparison ═══
ax4 = fig.add_subplot(2, 2, 4)
for label in ['GPU scatter (k=1)', 'GPU collapse k=3', 'GPU collapse k=5']:
    ax4.plot(ns, acc[label], marker=markers[label], color=colors_axol[label], lw=2.5, ms=9, label=label)
ax4.plot(ns, acc['GPU radix'], marker='X', color='#E74C3C', lw=2.5, ms=9, label='GPU radix (bug)', ls='--')
ax4.plot(ns, acc['CPU radix'], marker='^', color='#D9534F', lw=2.5, ms=9, label='CPU radix (exact)', ls=':')

ax4.set_xscale('log')
ax4.set_xlabel('n', fontsize=12); ax4.set_ylabel('Accuracy (%)', fontsize=12)
ax4.set_title('Accuracy (정확도)', fontsize=13, fontweight='bold')
ax4.set_ylim(-5, 108)
ax4.legend(fontsize=9, loc='center left'); ax4.grid(True, which='both', alpha=0.3)
ax4.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Highlight zones
ax4.axhline(y=63.2, color='#F0AD4E', ls=':', alpha=0.4)
ax4.text(150, 58, '1/e (~63%)', fontsize=9, color='#F0AD4E')

plt.suptitle('AXOL Scatter/Collapse vs Radix Sort — 전면 비교\n(RTX 4050, Vulkan)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('d:/Unity/Izakoza/AXOL/axol-lang/bench_vs_radix.png', dpi=150, bbox_inches='tight')
print("Saved: bench_vs_radix.png")

# ── Summary table ──
print()
print("═" * 90)
print(f"{'n':>10} | {'AXOL k=1':>10} | {'AXOL k=3':>10} | {'AXOL k=5':>10} | {'GPU radix':>10} | {'Winner':>12} | {'Factor':>8}")
print(f"{'':>10} | {'compute':>10} | {'compute':>10} | {'compute':>10} | {'compute':>10} | {'':>12} | {'':>8}")
print("-" * 90)
for i, n in enumerate(ns):
    vals = {'k=1': c['GPU scatter (k=1)'][i], 'k=3': c['GPU collapse k=3'][i],
            'k=5': c['GPU collapse k=5'][i], 'radix': c['GPU radix'][i]}
    fastest = min(vals, key=vals.get)
    slowest_axol = max(c['GPU scatter (k=1)'][i], c['GPU collapse k=3'][i])
    factor = c['GPU radix'][i] / c['GPU scatter (k=1)'][i]
    winner_label = f"AXOL {fastest}" if fastest != 'radix' else "Radix"
    print(f"{n:>10,} | {vals['k=1']:>10.0f} | {vals['k=3']:>10.0f} | {vals['k=5']:>10.0f} | {vals['radix']:>10.0f} | {winner_label:>12} | {factor:>7.2f}x")
print("═" * 90)
print("Factor = GPU radix / AXOL k=1  (>1 = AXOL wins)")
