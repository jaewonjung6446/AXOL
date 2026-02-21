import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Benchmark data
ns = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]

time_data = {
    'CPU std::sort':    [1.2, 8.2, 107.8, 1448.7, 17380.3, 198915.8],
    'CPU scatter':      [0.4, 1.7, 22.7, 402.7, 17526.6, 335077.9],
    'CPU radix':        [1.3, 8.1, 74.8, 813.4, 12678.8, 155200.7],
    'GPU scatter (k=1)':[479.1, 663.2, 657.2, 1024.1, 45083.5, 117212.1],
    'GPU collapse k=3': [657.2, 701.7, 836.9, 2085.6, 53272.5, 226062.6],
    'GPU collapse k=5': [639.8, 818.9, 816.1, 2693.1, 59885.7, 296755.2],
    'GPU radix':        [1616.2, 1767.2, 1645.4, 2149.0, 7962.4, 56449.0],
}

acc_data = {
    'GPU scatter (k=1)': [64.0, 59.9, 63.1, 63.2, 65.4, 83.2],
    'GPU collapse k=3':  [84.0, 83.2, 84.1, 84.9, 87.7, 100.0],
    'GPU collapse k=5':  [90.0, 90.9, 90.2, 90.7, 93.3, 100.0],
    'GPU radix':         [100.0, 0.4, 0.1, 0.04, 0.06, 0.08],
}

colors = {
    'CPU std::sort':     '#888888',
    'CPU scatter':       '#4A90D9',
    'CPU radix':         '#D9534F',
    'GPU scatter (k=1)': '#F0AD4E',
    'GPU collapse k=3':  '#5CB85C',
    'GPU collapse k=5':  '#8E44AD',
    'GPU radix':         '#E74C3C',
}

markers = {
    'CPU std::sort':     's',
    'CPU scatter':       'D',
    'CPU radix':         '^',
    'GPU scatter (k=1)': 'o',
    'GPU collapse k=3':  'P',
    'GPU collapse k=5':  '*',
    'GPU radix':         'X',
}

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# ── Chart 1: Time (log-log) ──
ax1 = axes[0]
for label, times in time_data.items():
    ax1.plot(ns, times, marker=markers[label], color=colors[label],
             label=label, linewidth=2, markersize=8)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('n (elements)', fontsize=13)
ax1.set_ylabel('Time (us)', fontsize=13)
ax1.set_title('Sorting Time Comparison (log-log)', fontsize=15, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, which='both', alpha=0.3)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Crossover annotation
ax1.annotate('GPU wins\n(n > 100K)', xy=(100_000, 1200), fontsize=10,
             ha='center', color='#5CB85C', fontweight='bold')

# ── Chart 2: Accuracy ──
ax2 = axes[1]
for label, accs in acc_data.items():
    style = '-' if 'radix' not in label else '--'
    ax2.plot(ns, accs, marker=markers[label], color=colors[label],
             label=label, linewidth=2, markersize=8, linestyle=style)

ax2.set_xscale('log')
ax2.set_xlabel('n (elements)', fontsize=13)
ax2.set_ylabel('Accuracy (%)', fontsize=13)
ax2.set_title('Sorting Accuracy Comparison', fontsize=15, fontweight='bold')
ax2.set_ylim(-5, 105)
ax2.legend(fontsize=10, loc='center right')
ax2.grid(True, which='both', alpha=0.3)
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Reference lines
ax2.axhline(y=63.2, color='#F0AD4E', linestyle=':', alpha=0.5, linewidth=1)
ax2.text(150, 60, '1/e theory (~63%)', fontsize=9, color='#F0AD4E')

plt.suptitle('AXOL GPU Collapse Sort vs Other Algorithms\n(RTX 4050 Laptop GPU, Vulkan)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('d:/Unity/Izakoza/AXOL/axol-lang/bench_collapse_chart.png',
            dpi=150, bbox_inches='tight')
print("Saved: bench_collapse_chart.png")
plt.close()

# ── Chart 3: Speedup vs CPU std::sort ──
fig2, ax3 = plt.subplots(figsize=(12, 7))

speedup_data = {}
baseline = time_data['CPU std::sort']
for label, times in time_data.items():
    if label == 'CPU std::sort':
        continue
    speedup_data[label] = [baseline[i] / times[i] for i in range(len(ns))]

for label, speedups in speedup_data.items():
    ax3.plot(ns, speedups, marker=markers[label], color=colors[label],
             label=label, linewidth=2, markersize=8)

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax3.set_xlabel('n (elements)', fontsize=13)
ax3.set_ylabel('Speedup vs CPU std::sort', fontsize=13)
ax3.set_title('Speedup Comparison (> 1 = faster than std::sort)\n(RTX 4050, Vulkan)',
              fontsize=15, fontweight='bold')
ax3.legend(fontsize=10, loc='best')
ax3.grid(True, which='both', alpha=0.3)
ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.tight_layout()
plt.savefig('d:/Unity/Izakoza/AXOL/axol-lang/bench_collapse_speedup.png',
            dpi=150, bbox_inches='tight')
print("Saved: bench_collapse_speedup.png")
plt.close()
