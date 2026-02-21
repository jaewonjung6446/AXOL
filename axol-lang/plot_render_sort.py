"""
Rendering-Fair Sort Benchmark Visualization
Reads bench_render_sort_results.json and produces 5 charts.
"""

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Load data ──

json_path = os.path.join(os.path.dirname(__file__), "bench_render_sort_results.json")
if len(sys.argv) > 1:
    json_path = sys.argv[1]

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

scenarios = data["scenarios"]
gpu_estimates = data["gpu_estimates"]
fairness = data["fairness_summary"]

# Color scheme
COLORS = {
    "std_sort":     "#4477AA",
    "axol_pure":    "#EE6677",
    "axol_hybrid":  "#228833",
    "radix_4pass":  "#CCBB44",
    "insertion":    "#AA3377",
}
LABELS = {
    "std_sort":     "std::sort",
    "axol_pure":    "aXOL Pure",
    "axol_hybrid":  "aXOL Hybrid",
    "radix_4pass":  "Radix (4-pass)",
    "insertion":    "Insertion",
}

fig_dir = os.path.join(os.path.dirname(__file__), "render_sort_charts")
os.makedirs(fig_dir, exist_ok=True)


# ══════════════════════════════════════════
# Chart 1: CPU Performance by Scenario (grouped bar)
# ══════════════════════════════════════════

def plot_cpu_performance():
    fig, ax = plt.subplots(figsize=(14, 7))

    scenario_names = [s["scenario"] for s in scenarios]
    # Collect all unique algorithm names in order
    all_algos = []
    for s in scenarios:
        for a in s["algorithms"]:
            if a["name"] not in all_algos:
                all_algos.append(a["name"])

    n_scenarios = len(scenario_names)
    n_algos = len(all_algos)
    bar_width = 0.8 / n_algos
    x = np.arange(n_scenarios)

    for i, algo in enumerate(all_algos):
        times = []
        for s in scenarios:
            match = [a for a in s["algorithms"] if a["name"] == algo]
            times.append(match[0]["total_us"] / 1000.0 if match else 0)  # ms

        offset = (i - n_algos / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, times, bar_width,
                      label=LABELS.get(algo, algo),
                      color=COLORS.get(algo, "#999999"),
                      edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Time (ms)")
    ax.set_title("CPU Sort Performance by Rendering Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=30, ha="right")
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "1_cpu_performance.png"), dpi=150)
    plt.close(fig)
    print("  [1/5] CPU performance chart saved")


# ══════════════════════════════════════════
# Chart 2: GPU Estimate Comparison (stacked bar for hybrid)
# ══════════════════════════════════════════

def plot_gpu_estimates():
    fig, ax = plt.subplots(figsize=(14, 7))

    scenario_names = [g["scenario"] for g in gpu_estimates]
    # Only compare exact algorithms on GPU
    gpu_algos = ["std_sort", "axol_hybrid", "radix_4pass"]

    n_scenarios = len(scenario_names)
    n_algos = len(gpu_algos)
    bar_width = 0.8 / n_algos
    x = np.arange(n_scenarios)

    for i, algo in enumerate(gpu_algos):
        times = []
        for g in gpu_estimates:
            match = [a for a in g["algorithms"] if a["name"] == algo]
            times.append(match[0]["hetero_us"] / 1000.0 if match else 0)  # ms

        offset = (i - n_algos / 2 + 0.5) * bar_width
        ax.bar(x + offset, times, bar_width,
               label=LABELS.get(algo, algo),
               color=COLORS.get(algo, "#999999"),
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Estimated GPU Time (ms)")
    ax.set_title("GPU Sort Estimate by Rendering Scenario\n(GPU=500GB/s, PCIe=25GB/s, CPU=50GB/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=30, ha="right")
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "2_gpu_estimates.png"), dpi=150)
    plt.close(fig)
    print("  [2/5] GPU estimates chart saved")


# ══════════════════════════════════════════
# Chart 3: Accuracy vs Speed (scatter plot, Pareto frontier)
# ══════════════════════════════════════════

def plot_accuracy_vs_speed():
    fig, ax = plt.subplots(figsize=(10, 8))

    for s in scenarios:
        for a in s["algorithms"]:
            color = COLORS.get(a["name"], "#999999")
            marker = "o" if a["is_exact"] else "^"
            size = 60 if a["is_exact"] else 40
            ax.scatter(a["total_us"] / 1000.0, a["accuracy"] * 100,
                       c=color, marker=marker, s=size, alpha=0.7,
                       edgecolors="black", linewidths=0.5)

    # Legend entries
    for algo in COLORS:
        ax.scatter([], [], c=COLORS[algo], label=LABELS.get(algo, algo), s=60)
    ax.scatter([], [], c="gray", marker="o", label="Exact", s=60)
    ax.scatter([], [], c="gray", marker="^", label="Approximate", s=40)

    ax.set_xlabel("Time (ms, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Speed — All Scenarios")
    ax.set_xscale("log")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "3_accuracy_vs_speed.png"), dpi=150)
    plt.close(fig)
    print("  [3/5] Accuracy vs speed chart saved")


# ══════════════════════════════════════════
# Chart 4: Fairness Dashboard (table heatmap)
# ══════════════════════════════════════════

def plot_fairness_dashboard():
    # Build speedup matrix: scenario x algorithm (speedup vs std)
    scenario_names = [s["scenario"] for s in scenarios]
    # Get all exact algos
    exact_algos = ["std_sort", "axol_hybrid", "radix_4pass", "insertion"]

    # Build matrix of times
    time_matrix = []
    algo_present = set()
    for s in scenarios:
        row = {}
        for a in s["algorithms"]:
            if a["is_exact"]:
                row[a["name"]] = a["total_us"]
                algo_present.add(a["name"])
        time_matrix.append(row)

    # Filter to algos that actually appear
    exact_algos = [a for a in exact_algos if a in algo_present]

    fig, ax = plt.subplots(figsize=(12, 6))

    n_scen = len(scenario_names)
    n_algo = len(exact_algos)

    # Rank within each scenario (1=best)
    cell_text = []
    cell_colors = []
    for si, row in enumerate(time_matrix):
        times = [(a, row.get(a, float("inf"))) for a in exact_algos]
        sorted_times = sorted(times, key=lambda x: x[1])
        ranks = {}
        for rank, (name, _) in enumerate(sorted_times):
            if name in row:
                ranks[name] = rank + 1
            else:
                ranks[name] = None

        text_row = []
        color_row = []
        for a in exact_algos:
            if a in row:
                r = ranks[a]
                t = row[a] / 1000.0  # ms
                text_row.append(f"{t:.2f}ms\n(#{r})")
                if r == 1:
                    color_row.append("#c6efce")  # green
                elif r == len([x for x in exact_algos if x in row]):
                    color_row.append("#ffc7ce")  # red
                else:
                    color_row.append("#ffffcc")  # yellow
            else:
                text_row.append("N/A")
                color_row.append("#f0f0f0")
        cell_text.append(text_row)
        cell_colors.append(color_row)

    table = ax.table(
        cellText=cell_text,
        rowLabels=scenario_names,
        colLabels=[LABELS.get(a, a) for a in exact_algos],
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    ax.set_title("Fairness Dashboard — CPU Exact Sort Rankings\n(Green=#1, Red=Last)", pad=20)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "4_fairness_dashboard.png"), dpi=150)
    plt.close(fig)
    print("  [4/5] Fairness dashboard saved")


# ══════════════════════════════════════════
# Chart 5: Distribution Histograms (4 panels)
# ══════════════════════════════════════════

def plot_distribution_histograms():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Pick one representative scenario per distribution type
    dist_scenarios = [
        ("particle_95", "Nearly-sorted (95%)"),
        ("oit_5layer", "Clustered (5 layers)"),
        ("rayhit_heavy", "Power-law (alpha=1.5)"),
        ("uniform_1m", "Uniform"),
    ]

    for idx, (name, title) in enumerate(dist_scenarios):
        ax = axes[idx // 2][idx % 2]
        s = next((s for s in scenarios if s["scenario"] == name), None)
        if s is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        stats = s["distribution_stats"]
        # We don't have raw data in JSON, but we have stats
        # Create a synthetic visualization from stats
        ax.bar(
            ["pre_sort", "mean", "std_dev"],
            [stats["pre_sortedness"], stats["mean"], stats["std_dev"]],
            color=["#4477AA", "#228833", "#EE6677"],
            alpha=0.8,
        )
        ax.set_title(f"{title}\nn={s['n']:,}")
        ax.text(0.02, 0.95,
                f"pre_sort={stats['pre_sortedness']:.4f}\n"
                f"range=[{stats['min']:.4f}, {stats['max']:.4f}]",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Distribution Statistics by Scenario", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "5_distribution_stats.png"), dpi=150)
    plt.close(fig)
    print("  [5/5] Distribution stats chart saved")


# ══════════════════════════════════════════
# Main
# ══════════════════════════════════════════

if __name__ == "__main__":
    print(f"Loading: {json_path}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Output dir: {fig_dir}")
    print()

    # Print fairness summary
    print("=== Fairness Summary ===")
    print("CPU winners:")
    for b in fairness["best_cpu_per_scenario"]:
        print(f"  {b['scenario']:<16} -> {b['winner']} ({b['time_us']:.1f}us)")
    print("GPU winners:")
    for b in fairness["best_gpu_per_scenario"]:
        print(f"  {b['scenario']:<16} -> {b['winner']} ({b['time_us']:.1f}us)")
    print()
    print("aXOL advantages:")
    for a in fairness["axol_advantages"]:
        print(f"  + {a}")
    print("aXOL disadvantages:")
    for d in fairness["axol_disadvantages"]:
        print(f"  - {d}")
    print()

    # Generate charts
    print("Generating charts...")
    plot_cpu_performance()
    plot_gpu_estimates()
    plot_accuracy_vs_speed()
    plot_fairness_dashboard()
    plot_distribution_histograms()
    print(f"\nAll 5 charts saved to: {fig_dir}/")
