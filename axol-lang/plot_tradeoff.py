import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════════════════════════
# Radar chart: multi-dimensional comparison
# ═══════════════════════════════════════════════

categories = [
    'Speed\n(small n)',       # n <= 1M
    'Speed\n(large n)',       # n >= 10M
    'Accuracy',
    'Memory\nefficiency',
    'Data type\nflexibility',
    'Distribution\nindependence',
    'GPU dispatch\ncount',
    'Atomic\ncontention',
    'Latency\n(first result)',
    'Float native\nsupport',
]
N = len(categories)

# Scores 0-10 (higher = better)
scores = {
    'AXOL scatter (k=1)': [10, 5, 4, 7, 9, 3, 10, 9, 10, 10],
    'AXOL collapse (k=3)': [9, 4, 7, 5, 9, 3, 9, 9, 9, 10],
    'AXOL collapse (k=5)': [8, 3, 8, 4, 9, 3, 8, 9, 8, 10],
    'Radix sort':          [3, 9, 10, 6, 3, 10, 3, 4, 3, 3],
}

colors = {
    'AXOL scatter (k=1)': '#F0AD4E',
    'AXOL collapse (k=3)': '#5CB85C',
    'AXOL collapse (k=5)': '#8E44AD',
    'Radix sort': '#E74C3C',
}

fig = plt.figure(figsize=(20, 16))

# ── Radar chart ──
ax_radar = fig.add_subplot(2, 2, 1, polar=True)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for label, vals in scores.items():
    vals_closed = vals + vals[:1]
    ax_radar.plot(angles, vals_closed, 'o-', lw=2.5, ms=7, label=label, color=colors[label])
    ax_radar.fill(angles, vals_closed, alpha=0.1, color=colors[label])

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories, fontsize=9)
ax_radar.set_ylim(0, 10)
ax_radar.set_yticks([2, 4, 6, 8, 10])
ax_radar.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
ax_radar.set_title('Multi-dimensional Trade-off Radar', fontsize=14, fontweight='bold', pad=20)

# ── Trade-off matrix (table) ──
ax_table = fig.add_subplot(2, 2, 2)
ax_table.axis('off')

rows = [
    ['Passes',           '1',         '1',         '1',         '4 (LSD)'],
    ['Time/elem',        'O(1)',      'O(k)=O(1)', 'O(k)=O(1)', 'O(d)=O(4)'],
    ['Accuracy',         '~63%',      '~85%',      '~93%',      '100%'],
    ['n=10M accuracy',   '83%',       '100%',      '100%',      '100%*'],
    ['GPU dispatches',   '4-6',       '6-8',       '6-8',       '~20'],
    ['Histogram size',   'n',         'k*n',       'k*n',       '256'],
    ['Memory (n=10M)',   '160MB',     '360MB',     '520MB',     '~120MB'],
    ['Atomic bins',      'n (sparse)','k*n',       'k*n',       '256 (dense)'],
    ['Float support',    'Native',    'Native',    'Native',    'Bit-flip'],
    ['Neg numbers',      'Direct',    'Direct',    'Direct',    'Encoding'],
    ['Distribution dep', 'YES',       'YES',       'YES',       'NO'],
    ['Min/Max required', 'YES',       'YES',       'YES',       'NO'],
    ['Stable sort',      'NO',        'NO',        'NO',        'YES**'],
    ['Key-value pair',   'Extra buf', 'Extra buf', 'Extra buf', 'Natural'],
]

col_labels = ['', 'AXOL k=1', 'AXOL k=3', 'AXOL k=5', 'Radix']
col_colors = ['#f0f0f0', '#FFF3CD', '#D4EDDA', '#E8DAEF', '#F8D7DA']

table = ax_table.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    colColours=col_colors,
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.5)

# Bold header
for j in range(5):
    table[0, j].set_text_props(fontweight='bold', fontsize=10)
# Color the row labels
for i in range(1, len(rows)+1):
    table[i, 0].set_text_props(fontweight='bold', ha='left')

ax_table.set_title('Feature Comparison Matrix', fontsize=14, fontweight='bold')
ax_table.text(0.5, -0.02, '* GPU radix has atomic contention bug (~0.1% actual)\n** Radix is stable per-digit, AXOL scatter ordering is arbitrary within bucket',
              ha='center', fontsize=8, color='gray', transform=ax_table.transAxes)

# ── What you GAIN vs what you LOSE ──
ax_gain = fig.add_subplot(2, 2, 3)
ax_gain.axis('off')

axol_gain = [
    "1-pass O(1): 3~4x faster (n<1M)",
    "Float native: no bit conversion",
    "Low dispatch count: less GPU sync",
    "Low latency: result in 1 pass",
    "Sparse atomics: n bins, low contention",
    "Simple shader: 10 lines per kernel",
]
axol_lose = [
    "Approximate: 63~93% accuracy (not exact)",
    "Distribution dependent: skewed data = bad",
    "Needs min/max: extra CPU pass",
    "Memory: n buckets (not 256)",
    "Unstable: same-bucket order is random",
    "No key-value: needs separate index buf",
]

radix_gain = [
    "Exact: 100% correct every time",
    "Distribution independent: works on anything",
    "No min/max needed: pure bitwise",
    "Stable: preserves insertion order",
    "Tiny histogram: 256 bins fits cache",
    "Key-value natural: just carry payload",
]
radix_lose = [
    "4 passes: 4x more GPU dispatches",
    "Bit-flip: f32 needs encoding/decoding",
    "High latency: result after 4 full passes",
    "256-bin contention: atomic bottleneck",
    "Fixed radix: can't tune for distribution",
    "Type-specific: different code per dtype",
]

y = 0.95
ax_gain.text(0.25, y, 'AXOL Scatter/Collapse', fontsize=13, fontweight='bold',
             ha='center', color='#5CB85C', transform=ax_gain.transAxes)
ax_gain.text(0.75, y, 'Radix Sort', fontsize=13, fontweight='bold',
             ha='center', color='#E74C3C', transform=ax_gain.transAxes)

y -= 0.06
ax_gain.text(0.0, y, 'GAINS:', fontsize=11, fontweight='bold', color='green', transform=ax_gain.transAxes)
ax_gain.text(0.5, y, 'GAINS:', fontsize=11, fontweight='bold', color='green', transform=ax_gain.transAxes)
for i, (ag, rg) in enumerate(zip(axol_gain, radix_gain)):
    y -= 0.065
    ax_gain.text(0.02, y, f"+ {ag}", fontsize=9, color='#2E7D32', transform=ax_gain.transAxes)
    ax_gain.text(0.52, y, f"+ {rg}", fontsize=9, color='#2E7D32', transform=ax_gain.transAxes)

y -= 0.08
ax_gain.text(0.0, y, 'COSTS:', fontsize=11, fontweight='bold', color='red', transform=ax_gain.transAxes)
ax_gain.text(0.5, y, 'COSTS:', fontsize=11, fontweight='bold', color='red', transform=ax_gain.transAxes)
for i, (al, rl) in enumerate(zip(axol_lose, radix_lose)):
    y -= 0.065
    ax_gain.text(0.02, y, f"- {al}", fontsize=9, color='#C62828', transform=ax_gain.transAxes)
    ax_gain.text(0.52, y, f"- {rl}", fontsize=9, color='#C62828', transform=ax_gain.transAxes)

ax_gain.set_title('What You GAIN vs What You LOSE', fontsize=14, fontweight='bold')

# ── Use case recommendation ──
ax_use = fig.add_subplot(2, 2, 4)
ax_use.axis('off')

cases = [
    ("Game rendering sort\n(depth/distance, n<1M, GPU-resident)",
     "AXOL k=1", "#5CB85C",
     "1-pass, GPU-resident, 63% is enough\n(visual artifacts minimal)"),
    ("Game transparency sort\n(back-to-front, n<100K)",
     "AXOL k=3", "#5CB85C",
     "85%+ accuracy, still fast,\nordering matters more"),
    ("Scientific computing\n(exact sort required, any n)",
     "Radix", "#E74C3C",
     "100% exact, bit-level precision,\nno approximation acceptable"),
    ("Database indexing\n(stable, key-value, large n)",
     "Radix", "#E74C3C",
     "Stable sort preserves order,\nnatural key-value handling"),
    ("Real-time physics\n(broad phase, n~10K-100K)",
     "AXOL k=1", "#5CB85C",
     "Fastest option, approximate OK\nfor spatial partitioning"),
    ("Machine learning\n(top-k, partial sort)",
     "AXOL k=5", "#8E44AD",
     "93%+ accuracy, O(1)/elem,\nfast approximate ranking"),
]

y = 0.95
ax_use.text(0.5, y, 'Use Case Recommendations', fontsize=14, fontweight='bold',
            ha='center', transform=ax_use.transAxes)
y -= 0.04

for scenario, winner, color, reason in cases:
    y -= 0.09
    ax_use.text(0.02, y, scenario, fontsize=10, fontweight='bold',
                transform=ax_use.transAxes, verticalalignment='top')
    y -= 0.04
    ax_use.text(0.55, y+0.04, f"Winner: {winner}", fontsize=11, fontweight='bold',
                color=color, transform=ax_use.transAxes, verticalalignment='top')
    ax_use.text(0.55, y-0.01, reason, fontsize=8, color='#555',
                transform=ax_use.transAxes, verticalalignment='top')

plt.suptitle('AXOL vs Radix Sort — What You Gain, What You Lose\n(RTX 4050, Vulkan)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('d:/Unity/Izakoza/AXOL/axol-lang/bench_tradeoff.png', dpi=150, bbox_inches='tight')
print("Saved: bench_tradeoff.png")
