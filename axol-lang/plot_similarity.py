"""
AXOL vs Radix: Similarity metrics beyond exact accuracy.
Simulates scatter/collapse on CPU to compute rank correlation.
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ── Sort implementations (CPU simulation matching GPU logic) ──

def scatter_sort(data, k=1):
    """k-slit resonance collapse sort (CPU simulation of GPU logic)"""
    n = len(data)
    mn, mx = data.min(), data.max()
    rng = mx - mn if abs(mx - mn) > 1e-10 else 1.0
    inv_range = n / rng  # OffsetTable style: n/range

    # Multi-slit position accumulation
    pos_sum = np.zeros(n, dtype=np.int64)
    for s in range(k):
        offset = s / k
        buckets = np.minimum(((data - mn) * inv_range + offset).astype(np.int64), n - 1)
        pos_sum += buckets

    # Histogram on pos_sum (range: 0 to k*(n-1))
    kn = k * n
    histogram = np.zeros(kn, dtype=np.int64)
    clamped = np.minimum(pos_sum, kn - 1)
    for ps in clamped:
        histogram[ps] += 1

    # Prefix sum
    offsets = np.cumsum(histogram) - histogram  # exclusive prefix sum
    offsets_copy = offsets.copy()

    # Scatter
    result = np.empty(n, dtype=data.dtype)
    for i in range(n):
        b = min(int(pos_sum[i]), kn - 1)
        pos = offsets_copy[b]
        offsets_copy[b] += 1
        result[pos] = data[i]

    return result

def radix_sort_f32(data):
    """LSD radix sort for f32 (exact, CPU)"""
    def f32_to_sortable(v):
        bits = np.float32(v).view(np.uint32)
        if bits & 0x80000000:
            return ~bits & 0xFFFFFFFF
        else:
            return bits ^ 0x80000000

    def sortable_to_f32(bits):
        if bits & 0x80000000:
            raw = bits ^ 0x80000000
        else:
            raw = ~bits & 0xFFFFFFFF
        return np.uint32(raw).view(np.float32)

    n = len(data)
    keys = np.array([f32_to_sortable(v) for v in data], dtype=np.uint32)
    buf = np.zeros(n, dtype=np.uint32)

    for p in range(4):
        shift = p * 8
        counts = np.zeros(256, dtype=np.int64)
        for k in keys:
            counts[(k >> shift) & 0xFF] += 1
        prefix = np.zeros(256, dtype=np.int64)
        for i in range(1, 256):
            prefix[i] = prefix[i-1] + counts[i-1]
        for k in keys:
            d = (k >> shift) & 0xFF
            buf[prefix[d]] = k
            prefix[d] += 1
        keys, buf = buf, keys

    return np.array([sortable_to_f32(k) for k in keys], dtype=np.float32)

# ── Similarity metrics ──

def compute_metrics(ground_truth, result):
    n = len(ground_truth)

    # 1. Exact accuracy
    exact = np.sum(np.abs(ground_truth - result) < 1e-9) / n

    # 2. Spearman rank correlation
    # Rank of each value in the result vs ground truth
    gt_ranks = stats.rankdata(ground_truth)
    res_ranks = stats.rankdata(result)
    # But we need: for each position, is the VALUE at that position correct?
    # Actually, we should compare the ordering: rank of result[i] in the sorted order
    # Simpler: compute rank correlation between the two sequences as-is
    spearman_rho, _ = stats.spearmanr(ground_truth, result)

    # 3. Kendall tau
    if n <= 100000:  # tau is O(n^2), skip for large n
        kendall_tau, _ = stats.kendalltau(ground_truth, result)
    else:
        # Approximate using sample
        idx = np.random.choice(n, min(50000, n), replace=False)
        kendall_tau, _ = stats.kendalltau(ground_truth[idx], result[idx])

    # 4. Mean normalized displacement
    # For each element in result, find how far it is from its correct position
    gt_order = np.argsort(np.argsort(ground_truth))  # rank of each position in gt
    res_order = np.argsort(np.argsort(result))
    displacement = np.abs(gt_order.astype(np.int64) - res_order.astype(np.int64))
    mean_disp = displacement.mean() / n  # normalized by n
    max_disp = displacement.max()
    p99_disp = np.percentile(displacement, 99)

    # 5. Cosine similarity of rank vectors
    cos_sim = np.dot(gt_order.astype(float), res_order.astype(float)) / (
        np.linalg.norm(gt_order.astype(float)) * np.linalg.norm(res_order.astype(float)))

    return {
        'exact_accuracy': exact,
        'spearman_rho': spearman_rho,
        'kendall_tau': kendall_tau,
        'cosine_similarity': cos_sim,
        'mean_displacement': displacement.mean(),
        'mean_disp_normalized': mean_disp,
        'max_displacement': max_disp,
        'p99_displacement': p99_disp,
    }

# ── Run benchmarks ──
sizes = [1000, 10000, 100000, 1000000]
methods = {
    'AXOL k=1': lambda d: scatter_sort(d, k=1),
    'AXOL k=3': lambda d: scatter_sort(d, k=3),
    'AXOL k=5': lambda d: scatter_sort(d, k=5),
    'Radix':    lambda d: radix_sort_f32(d),
}

all_results = {m: [] for m in methods}

for n in sizes:
    print(f"\n{'='*60}")
    print(f"n = {n:,}")
    print(f"{'='*60}")
    data = np.random.rand(n).astype(np.float32)
    gt = np.sort(data)

    for name, fn in methods.items():
        t0 = time.time()
        result = fn(data)
        elapsed = time.time() - t0
        metrics = compute_metrics(gt, result)
        all_results[name].append(metrics)

        print(f"\n  {name}:")
        print(f"    Exact accuracy:    {metrics['exact_accuracy']*100:>8.2f}%")
        print(f"    Spearman rho:      {metrics['spearman_rho']:>8.6f}")
        print(f"    Kendall tau:       {metrics['kendall_tau']:>8.6f}")
        print(f"    Cosine similarity: {metrics['cosine_similarity']:>8.6f}")
        print(f"    Mean displacement: {metrics['mean_displacement']:>8.1f} positions")
        print(f"    P99 displacement:  {metrics['p99_displacement']:>8.0f} positions")
        print(f"    Max displacement:  {metrics['max_displacement']:>8.0f} positions")
        print(f"    Time: {elapsed*1000:.1f}ms")

# ── Plot ──
fig, axes = plt.subplots(2, 3, figsize=(20, 13))

colors = {'AXOL k=1': '#F0AD4E', 'AXOL k=3': '#5CB85C', 'AXOL k=5': '#8E44AD', 'Radix': '#E74C3C'}
markers = {'AXOL k=1': 'o', 'AXOL k=3': 'P', 'AXOL k=5': '*', 'Radix': 'X'}

def plot_metric(ax, metric_key, title, ylabel, ylim=None, invert=False, pct=False):
    for name in methods:
        vals = [r[metric_key] for r in all_results[name]]
        if pct:
            vals = [v * 100 for v in vals]
        ax.plot(sizes, vals, marker=markers[name], color=colors[name],
                label=name, lw=2.5, ms=9)
    ax.set_xscale('log')
    ax.set_xlabel('n', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if ylim: ax.set_ylim(ylim)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# 1. Exact accuracy (the old metric)
plot_metric(axes[0,0], 'exact_accuracy', 'Exact Accuracy (old metric)',
            'Exact match %', ylim=(-5, 105), pct=True)
axes[0,0].axhline(y=63.2, color='#F0AD4E', ls=':', alpha=0.5)
axes[0,0].text(1200, 58, '1/e theory', fontsize=9, color='#F0AD4E')

# 2. Spearman rho
plot_metric(axes[0,1], 'spearman_rho', 'Spearman Rank Correlation (rho)',
            'rho (1.0 = perfect)', ylim=(0.99, 1.001))

# 3. Kendall tau
plot_metric(axes[0,2], 'kendall_tau', "Kendall's Tau",
            'tau (1.0 = perfect)', ylim=(0.95, 1.001))

# 4. Cosine similarity
plot_metric(axes[1,0], 'cosine_similarity', 'Cosine Similarity (rank vectors)',
            'similarity (1.0 = perfect)', ylim=(0.999, 1.0001))

# 5. Mean displacement
ax5 = axes[1,1]
for name in methods:
    vals = [r['mean_displacement'] for r in all_results[name]]
    ax5.plot(sizes, vals, marker=markers[name], color=colors[name],
             label=name, lw=2.5, ms=9)
ax5.set_xscale('log'); ax5.set_yscale('log')
ax5.set_xlabel('n', fontsize=12); ax5.set_ylabel('Mean displacement (positions)', fontsize=12)
ax5.set_title('Mean Position Displacement\n(lower = better)', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10); ax5.grid(True, alpha=0.3)
ax5.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# 6. Summary bar chart at n=100K
ax6 = axes[1,2]
n_idx = 2  # n=100K
bar_metrics = ['exact_accuracy', 'spearman_rho', 'kendall_tau', 'cosine_similarity']
bar_labels = ['Exact\nAccuracy', 'Spearman\nrho', 'Kendall\ntau', 'Cosine\nSim']
x = np.arange(len(bar_metrics))
width = 0.2
for i, name in enumerate(methods):
    vals = [all_results[name][n_idx][m] for m in bar_metrics]
    ax6.bar(x + i*width, vals, width, label=name, color=colors[name], alpha=0.85)
ax6.set_xticks(x + 1.5*width)
ax6.set_xticklabels(bar_labels, fontsize=10)
ax6.set_ylabel('Score (1.0 = perfect)', fontsize=12)
ax6.set_title(f'All Metrics at n=100K\n(Exact accuracy vs Similarity)', fontsize=13, fontweight='bold')
ax6.legend(fontsize=9)
ax6.set_ylim(0.5, 1.02)
ax6.grid(True, axis='y', alpha=0.3)

plt.suptitle('AXOL vs Radix: Exact Accuracy vs Overall Similarity\n"63% accuracy" does NOT mean "37% wrong ordering"',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('d:/Unity/Izakoza/AXOL/axol-lang/bench_similarity.png', dpi=150, bbox_inches='tight')
print("\nSaved: bench_similarity.png")
