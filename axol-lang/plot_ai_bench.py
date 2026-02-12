"""AXOL AI Benchmark Visualization — generates PNG charts from bench_ai_data.txt"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_FILE = Path(__file__).parent / "bench_ai_data.txt"
OUT_DIR = Path(__file__).parent / "bench_plots"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Parse CSV sections from the data file
# ---------------------------------------------------------------------------

def parse_sections(path):
    sections = {}
    current_name = None
    current_header = None
    current_rows = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            if current_name and current_rows:
                sections[current_name] = (current_header, current_rows)
            current_name = None
            current_header = None
            current_rows = []
            continue
        if line.startswith("CSV:"):
            current_name = line[4:]
            continue
        if current_name and current_header is None:
            current_header = line.split(",")
            continue
        if current_name and current_header:
            current_rows.append(line.split(","))

    if current_name and current_rows:
        sections[current_name] = (current_header, current_rows)

    return sections

sections = parse_sections(DATA_FILE)

# Color palette
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
          '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']
BG_COLOR = '#FAFAFA'

# ---------------------------------------------------------------------------
# Chart 1: Task Type — Generation Time + Observations
# ---------------------------------------------------------------------------

header, rows = sections['task_type']
tasks = [r[0] for r in rows]
gen_times = [float(r[1]) for r in rows]
obs_counts = [int(r[2]) for r in rows]
confidences = [float(r[3]) for r in rows]
dims = [int(r[4]) for r in rows]
stmts = [int(r[5]) for r in rows]
src_bytes = [int(r[7]) for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG_COLOR)
fig.suptitle('AXOL AI — 태스크 유형별 벤치마크', fontsize=16, fontweight='bold', y=1.02)

# 1a: Generation time
ax = axes[0]
ax.set_facecolor(BG_COLOR)
bars = ax.bar(tasks, gen_times, color=COLORS[:5], edgecolor='white', linewidth=1.5)
ax.set_ylabel('생성 시간 (ms)', fontsize=12)
ax.set_title('코드 생성 시간', fontsize=13)
for bar, v in zip(bars, gen_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{v:.2f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, max(gen_times) * 1.4)

# 1b: Source size
ax = axes[1]
ax.set_facecolor(BG_COLOR)
bars = ax.bar(tasks, src_bytes, color=COLORS[:5], edgecolor='white', linewidth=1.5)
ax.set_ylabel('소스 크기 (bytes)', fontsize=12)
ax.set_title('생성된 .axol 소스 크기', fontsize=13)
for bar, v in zip(bars, src_bytes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{v}B', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, max(src_bytes) * 1.3)

# 1c: Statements count
ax = axes[2]
ax.set_facecolor(BG_COLOR)
bars = ax.bar(tasks, stmts, color=COLORS[:5], edgecolor='white', linewidth=1.5)
ax.set_ylabel('명령문 수', fontsize=12)
ax.set_title('생성된 AST 명령문 수', fontsize=13)
for bar, v in zip(bars, stmts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(stmts) + 3)

plt.tight_layout()
fig.savefig(OUT_DIR / '1_task_type.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  [1/7] 1_task_type.png")

# ---------------------------------------------------------------------------
# Chart 2: AI Construction Time by Seed
# ---------------------------------------------------------------------------

header, rows = sections['construction']
seeds = [int(r[0]) for r in rows]
construct_ms = [float(r[1]) for r in rows]

fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
bars = ax.bar(range(len(seeds)), construct_ms, color=COLORS[0], edgecolor='white',
              linewidth=1.5, alpha=0.85)
ax.set_xticks(range(len(seeds)))
ax.set_xticklabels([str(s) for s in seeds], rotation=45, ha='right')
ax.set_xlabel('시드 (seed)', fontsize=12)
ax.set_ylabel('생성 시간 (ms)', fontsize=12)
ax.set_title('AXOL AI — 태피스트리 직조 시간 (시드별)', fontsize=14, fontweight='bold')
ax.axhline(y=np.mean(construct_ms), color=COLORS[3], linestyle='--', linewidth=2,
           label=f'평균: {np.mean(construct_ms):.1f}ms')
ax.legend(fontsize=11)

for bar, v in zip(bars, construct_ms):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{v:.1f}', ha='center', va='bottom', fontsize=9)
ax.set_ylim(0, max(construct_ms) * 1.2)

plt.tight_layout()
fig.savefig(OUT_DIR / '2_construction_time.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  [2/7] 2_construction_time.png")

# ---------------------------------------------------------------------------
# Chart 3: Dimension Sweep — Source Size per Task
# ---------------------------------------------------------------------------

header, rows = sections['dim_sweep']
task_names = sorted(set(r[0] for r in rows), key=lambda t: ['logic','classify','pipeline','converge','composite'].index(t))
dim_values = sorted(set(int(r[1]) for r in rows))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
fig.suptitle('AXOL AI — 차원(dim)별 생성 결과', fontsize=14, fontweight='bold', y=1.02)

# 3a: Source bytes by dim
ax = axes[0]
ax.set_facecolor(BG_COLOR)
x = np.arange(len(dim_values))
width = 0.15
for i, task in enumerate(task_names):
    task_rows = [r for r in rows if r[0] == task]
    bytes_vals = [int(r[5]) for r in task_rows]
    ax.bar(x + i * width, bytes_vals, width, label=task, color=COLORS[i],
           edgecolor='white', linewidth=1)

ax.set_xticks(x + width * 2)
ax.set_xticklabels([str(d) for d in dim_values])
ax.set_xlabel('차원 (dim)', fontsize=12)
ax.set_ylabel('소스 크기 (bytes)', fontsize=12)
ax.set_title('소스 크기', fontsize=13)
ax.legend(fontsize=9, ncol=3)

# 3b: Statements by dim
ax = axes[1]
ax.set_facecolor(BG_COLOR)
for i, task in enumerate(task_names):
    task_rows = [r for r in rows if r[0] == task]
    stmts_vals = [int(r[4]) for r in task_rows]
    ax.plot(dim_values, stmts_vals, 'o-', color=COLORS[i], label=task,
            linewidth=2, markersize=8)

ax.set_xlabel('차원 (dim)', fontsize=12)
ax.set_ylabel('명령문 수', fontsize=12)
ax.set_title('명령문 수', fontsize=13)
ax.legend(fontsize=9)
ax.set_xticks(dim_values)

plt.tight_layout()
fig.savefig(OUT_DIR / '3_dim_sweep.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  [3/7] 3_dim_sweep.png")

# ---------------------------------------------------------------------------
# Chart 4: Quality Sweep — Omega/Phi response
# ---------------------------------------------------------------------------

header, rows = sections['quality_sweep']
qualities = [float(r[0]) for r in rows]
omegas = [float(r[3]) for r in rows]
phis = [float(r[4]) for r in rows]
gen_times_q = [float(r[1]) for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
fig.suptitle('AXOL AI — 품질(quality) 파라미터 반응', fontsize=14, fontweight='bold', y=1.02)

# 4a: Quality → omega/phi
ax = axes[0]
ax.set_facecolor(BG_COLOR)
ax.plot(qualities, omegas, 'o-', color=COLORS[0], linewidth=2.5, markersize=8, label='omega (Ω)')
ax.plot(qualities, phis, 's--', color=COLORS[1], linewidth=2.5, markersize=8, label='phi (Φ)')
ax.plot(qualities, qualities, ':', color='gray', linewidth=1.5, alpha=0.5, label='y=x (이상적)')
ax.set_xlabel('요청 품질', fontsize=12)
ax.set_ylabel('출력 파라미터', fontsize=12)
ax.set_title('품질 요청 → Ω, Φ 반응', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# 4b: Quality → gen time
ax = axes[1]
ax.set_facecolor(BG_COLOR)
ax.bar(range(len(qualities)), gen_times_q, color=COLORS[2], edgecolor='white', linewidth=1.5, alpha=0.85)
ax.set_xticks(range(len(qualities)))
ax.set_xticklabels([f'{q:.1f}' for q in qualities])
ax.set_xlabel('요청 품질', fontsize=12)
ax.set_ylabel('생성 시간 (ms)', fontsize=12)
ax.set_title('품질별 생성 시간', fontsize=13)
ax.axhline(y=np.mean(gen_times_q), color=COLORS[3], linestyle='--', linewidth=2,
           label=f'평균: {np.mean(gen_times_q):.2f}ms')
ax.legend(fontsize=11)

plt.tight_layout()
fig.savefig(OUT_DIR / '4_quality_sweep.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  [4/7] 4_quality_sweep.png")

# ---------------------------------------------------------------------------
# Chart 5: Latency Distribution (Histogram)
# ---------------------------------------------------------------------------

header, rows = sections['latency_dist']
latencies = [float(r[1]) for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
fig.suptitle('AXOL AI — 생성 지연시간 분포 (50회 반복)', fontsize=14, fontweight='bold', y=1.02)

# 5a: Histogram
ax = axes[0]
ax.set_facecolor(BG_COLOR)
ax.hist(latencies, bins=20, color=COLORS[0], edgecolor='white', linewidth=1.5, alpha=0.85)
ax.axvline(np.mean(latencies), color=COLORS[3], linestyle='--', linewidth=2,
           label=f'평균: {np.mean(latencies):.0f}μs')
ax.axvline(np.median(latencies), color=COLORS[1], linestyle='--', linewidth=2,
           label=f'중앙값: {np.median(latencies):.0f}μs')
ax.set_xlabel('지연시간 (μs)', fontsize=12)
ax.set_ylabel('빈도', fontsize=12)
ax.set_title('지연시간 히스토그램', fontsize=13)
ax.legend(fontsize=11)

# 5b: Time series
ax = axes[1]
ax.set_facecolor(BG_COLOR)
ax.plot(range(len(latencies)), latencies, 'o-', color=COLORS[0], linewidth=1.5,
        markersize=5, alpha=0.8)
ax.axhline(np.mean(latencies), color=COLORS[3], linestyle='--', linewidth=2, alpha=0.7,
           label=f'평균: {np.mean(latencies):.0f}μs')
ax.fill_between(range(len(latencies)),
                [np.mean(latencies) - np.std(latencies)] * len(latencies),
                [np.mean(latencies) + np.std(latencies)] * len(latencies),
                color=COLORS[3], alpha=0.1, label=f'±1σ ({np.std(latencies):.0f}μs)')
ax.set_xlabel('실행 번호', fontsize=12)
ax.set_ylabel('지연시간 (μs)', fontsize=12)
ax.set_title('시계열 지연시간', fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()
fig.savefig(OUT_DIR / '5_latency_dist.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  [5/7] 5_latency_dist.png")

# ---------------------------------------------------------------------------
# Chart 6: Learn vs AI Comparison
# ---------------------------------------------------------------------------

header, rows = sections['learn_comparison']
gates = [r[0] for r in rows]
learn_times = [float(r[1]) for r in rows]
learn_accs = [float(r[2]) * 100 for r in rows]
learn_evals = [int(r[3]) for r in rows]
ai_times = [float(r[4]) for r in rows]
ai_obs = [int(r[5]) for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG_COLOR)
fig.suptitle('AXOL — Learn vs AI 비교', fontsize=14, fontweight='bold', y=1.02)

# 6a: Time comparison (log scale)
ax = axes[0]
ax.set_facecolor(BG_COLOR)
x = np.arange(len(gates))
width = 0.35
bars1 = ax.bar(x - width/2, learn_times, width, label='Learn', color=COLORS[0],
               edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, ai_times, width, label='AI', color=COLORS[1],
               edgecolor='white', linewidth=1.5)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([g.upper() for g in gates])
ax.set_ylabel('시간 (ms, 로그 스케일)', fontsize=12)
ax.set_title('실행 시간', fontsize=13)
ax.legend(fontsize=11)

for bar, v in zip(bars1, learn_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
            f'{v:.0f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, v in zip(bars2, ai_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
            f'{v:.1f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 6b: Evaluations comparison (log scale)
ax = axes[1]
ax.set_facecolor(BG_COLOR)
bars1 = ax.bar(x - width/2, learn_evals, width, label='Learn (평가 횟수)', color=COLORS[0],
               edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, ai_obs, width, label='AI (관측 횟수)', color=COLORS[1],
               edgecolor='white', linewidth=1.5)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([g.upper() for g in gates])
ax.set_ylabel('횟수 (로그 스케일)', fontsize=12)
ax.set_title('평가/관측 횟수', fontsize=13)
ax.legend(fontsize=10)

for bar, v in zip(bars1, learn_evals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
            str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, v in zip(bars2, ai_obs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
            str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

# 6c: Learn accuracy
ax = axes[2]
ax.set_facecolor(BG_COLOR)
colors_acc = [COLORS[2] if a >= 100 else COLORS[3] if a < 75 else COLORS[4] for a in learn_accs]
bars = ax.bar(gates, learn_accs, color=colors_acc, edgecolor='white', linewidth=1.5)
ax.axhline(100, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax.axhline(75, color=COLORS[3], linestyle='--', linewidth=1.5, alpha=0.3, label='75% 기준')
ax.set_ylabel('정확도 (%)', fontsize=12)
ax.set_title('Learn 정확도', fontsize=13)
ax.set_ylim(0, 115)
ax.legend(fontsize=10)
for bar, v in zip(bars, learn_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{v:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
fig.savefig(OUT_DIR / '6_learn_vs_ai.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  [6/7] 6_learn_vs_ai.png")

# ---------------------------------------------------------------------------
# Chart 7: Summary Dashboard
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 10), facecolor=BG_COLOR)
fig.suptitle('AXOL AI 종합 벤치마크 대시보드', fontsize=18, fontweight='bold', y=0.98)

# Speedup
speedups = [lt / at for lt, at in zip(learn_times, ai_times)]

# 7a: Speedup bar
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_facecolor(BG_COLOR)
bars = ax1.bar([g.upper() for g in gates], speedups, color=COLORS[1], edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, speedups):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{v:.0f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylabel('속도 배율', fontsize=11)
ax1.set_title('AI vs Learn 속도 비교', fontsize=12, fontweight='bold')

# 7b: Construction time stats
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_facecolor(BG_COLOR)
stats = {
    '최소': f'{min(construct_ms):.1f}ms',
    '평균': f'{np.mean(construct_ms):.1f}ms',
    '최대': f'{max(construct_ms):.1f}ms',
    '표준편차': f'{np.std(construct_ms):.1f}ms',
}
ax2.axis('off')
ax2.set_title('AI 생성 통계', fontsize=12, fontweight='bold')
table_data = [[k, v] for k, v in stats.items()]
table = ax2.table(cellText=table_data, colLabels=['지표', '값'],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(COLORS[0])
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#F0F0F0' if row % 2 == 0 else 'white')

# 7c: Latency box plot
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_facecolor(BG_COLOR)
bp = ax3.boxplot(latencies, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor(COLORS[0])
bp['boxes'][0].set_alpha(0.7)
bp['medians'][0].set_color(COLORS[3])
bp['medians'][0].set_linewidth(2)
ax3.set_ylabel('지연시간 (μs)', fontsize=11)
ax3.set_title('생성 지연시간 분포', fontsize=12, fontweight='bold')
ax3.set_xticklabels(['50회 반복'])

# 7d: Pie — time breakdown (construction vs generation)
ax4 = fig.add_subplot(2, 3, 4)
ax4.set_facecolor(BG_COLOR)
avg_construct = np.mean(construct_ms)
avg_gen = np.mean(gen_times)
sizes = [avg_construct, avg_gen]
labels = [f'태피스트리 직조\n{avg_construct:.1f}ms', f'코드 생성\n{avg_gen:.2f}ms']
explode = (0, 0.1)
ax4.pie(sizes, explode=explode, labels=labels, colors=[COLORS[0], COLORS[1]],
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
ax4.set_title('시간 구성 비율', fontsize=12, fontweight='bold')

# 7e: Source size by dim (heatmap-style)
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_facecolor(BG_COLOR)
header_dim, rows_dim = sections['dim_sweep']
heatmap_data = []
for task in task_names:
    task_rows = [r for r in rows_dim if r[0] == task]
    heatmap_data.append([int(r[5]) for r in task_rows])

im = ax5.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax5.set_xticks(range(len(dim_values)))
ax5.set_xticklabels([str(d) for d in dim_values])
ax5.set_yticks(range(len(task_names)))
ax5.set_yticklabels(task_names)
ax5.set_xlabel('차원 (dim)')
ax5.set_title('소스 크기 히트맵 (bytes)', fontsize=12, fontweight='bold')

for i in range(len(task_names)):
    for j in range(len(dim_values)):
        text = ax5.text(j, i, heatmap_data[i][j],
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white' if heatmap_data[i][j] > 900 else 'black')

plt.colorbar(im, ax=ax5, shrink=0.8)

# 7f: Key metrics summary
ax6 = fig.add_subplot(2, 3, 6)
ax6.set_facecolor(BG_COLOR)
ax6.axis('off')
ax6.set_title('핵심 성능 지표', fontsize=12, fontweight='bold')

metrics = [
    ['AI 구축 시간', f'{np.mean(construct_ms):.1f}ms'],
    ['코드 생성 시간', f'{np.mean(gen_times)*1000:.0f}μs'],
    ['생성 지연시간 (중앙값)', f'{np.median(latencies):.0f}μs'],
    ['관측 횟수 (고정)', '48회'],
    ['Learn 대비 속도', f'{np.mean(speedups):.0f}x 빠름'],
    ['생성된 코드 실행', '100% 성공'],
]

table = ax6.table(cellText=metrics, colLabels=['지표', '값'],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(COLORS[2])
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#F0F8F0' if row % 2 == 0 else 'white')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / '7_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  [7/7] 7_dashboard.png")

print(f"\nDone! All charts saved to: {OUT_DIR}")
