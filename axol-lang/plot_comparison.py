"""AXOL AI Benchmark — Before vs After Learning Comparison"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).parent
OUT_DIR = BASE / "bench_plots"
OUT_DIR.mkdir(exist_ok=True)

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
          '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']
BG = '#FAFAFA'

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

before = parse_sections(BASE / "bench_ai_data.txt")
after = parse_sections(BASE / "bench_ai_data2.txt")

# ===================================================================
# Chart 1: Structure Index Distribution (Before vs After)
# ===================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle('구조 선택기 — Weave(학습 전) vs Learn(학습 후)', fontsize=15, fontweight='bold', y=1.02)

# Before
_, rows_b = before['seed_effect']
idx_b = [int(r[4]) for r in rows_b]
ax = axes[0]
ax.set_facecolor(BG)
counts_b = [idx_b.count(i) for i in range(5)]
labels = ['Logic(0)', 'Classify(1)', 'Pipeline(2)', 'Converge(3)', 'Composite(4)']
bars = ax.bar(labels, counts_b, color=COLORS[:5], edgecolor='white', linewidth=1.5)
ax.set_title('학습 전: 항상 Pipeline(2)으로 수렴', fontsize=12, fontweight='bold')
ax.set_ylabel('시드 수 (15개 중)', fontsize=11)
for bar, v in zip(bars, counts_b):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(v), ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylim(0, 17)
ax.tick_params(axis='x', rotation=30)

# After
_, rows_a = after['seed_effect']
idx_a = [int(r[4]) for r in rows_a]
ax = axes[1]
ax.set_facecolor(BG)
counts_a = [idx_a.count(i) for i in range(5)]
bars = ax.bar(labels, counts_a, color=COLORS[:5], edgecolor='white', linewidth=1.5)
ax.set_title('학습 후: 5개 구조 모두 활성화', fontsize=12, fontweight='bold')
ax.set_ylabel('시드 수 (15개 중)', fontsize=11)
for bar, v in zip(bars, counts_a):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(v), ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylim(0, 17)
ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
fig.savefig(OUT_DIR / '8_structure_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  [1/5] 8_structure_comparison.png")

# ===================================================================
# Chart 2: Dim Distribution (Before vs After)
# ===================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle('차원 선택기 — Weave(학습 전) vs Learn(학습 후)', fontsize=15, fontweight='bold', y=1.02)

dim_labels = ['dim=2', 'dim=4', 'dim=8', 'dim=16']
dim_vals = [2, 4, 8, 16]

# Before
dims_b = [int(r[5]) for r in rows_b]
counts_dim_b = [dims_b.count(d) for d in dim_vals]
ax = axes[0]
ax.set_facecolor(BG)
bars = ax.bar(dim_labels, counts_dim_b, color=COLORS[0], edgecolor='white', linewidth=1.5)
ax.set_title('학습 전: 항상 dim=16', fontsize=12, fontweight='bold')
ax.set_ylabel('시드 수 (15개 중)', fontsize=11)
for bar, v in zip(bars, counts_dim_b):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(v), ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylim(0, 17)

# After
dims_a = [int(r[5]) for r in rows_a]
counts_dim_a = [dims_a.count(d) for d in dim_vals]
ax = axes[1]
ax.set_facecolor(BG)
bars = ax.bar(dim_labels, counts_dim_a, color=[COLORS[i] for i in range(4)],
              edgecolor='white', linewidth=1.5)
ax.set_title('학습 후: 4개 차원 모두 활성화', fontsize=12, fontweight='bold')
ax.set_ylabel('시드 수 (15개 중)', fontsize=11)
for bar, v in zip(bars, counts_dim_a):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(v), ha='center', va='bottom', fontsize=13, fontweight='bold')
ax.set_ylim(0, 17)

plt.tight_layout()
fig.savefig(OUT_DIR / '9_dim_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  [2/5] 9_dim_comparison.png")

# ===================================================================
# Chart 3: Construction Time Comparison
# ===================================================================
fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
ax.set_facecolor(BG)

_, rows_cb = before['construction']
_, rows_ca = after['construction']
seeds = [int(r[0]) for r in rows_cb]
time_b = [float(r[1]) for r in rows_cb]
time_a = [float(r[1]) for r in rows_ca]

x = np.arange(len(seeds))
width = 0.35
bars1 = ax.bar(x - width/2, time_b, width, label=f'학습 전 (평균 {np.mean(time_b):.0f}ms)',
               color=COLORS[0], edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, [t/1000 for t in time_a], width,
               label=f'학습 후 (평균 {np.mean(time_a)/1000:.1f}s)',
               color=COLORS[1], edgecolor='white', linewidth=1.5)

ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in seeds], rotation=45, ha='right')
ax.set_xlabel('시드 (seed)', fontsize=12)
ax.set_ylabel('구축 시간', fontsize=12)
ax.set_title('AI 구축 시간 비교 (ms vs 초)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)

# Add second y-axis label
ax.text(0.02, 0.95, '학습 전: ms 단위\n학습 후: 초 단위 (÷1000)',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
fig.savefig(OUT_DIR / '10_construction_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  [3/5] 10_construction_comparison.png")

# ===================================================================
# Chart 4: Task Type — Source Diversity
# ===================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle('태스크별 생성 결과 — 학습 전 vs 학습 후', fontsize=15, fontweight='bold', y=1.02)

_, rows_tb = before['task_type']
_, rows_ta = after['task_type']
tasks = [r[0] for r in rows_tb]

# Source bytes
bytes_b = [int(r[7]) for r in rows_tb]
bytes_a = [int(r[7]) for r in rows_ta]

ax = axes[0]
ax.set_facecolor(BG)
x = np.arange(len(tasks))
width = 0.35
ax.bar(x - width/2, bytes_b, width, label='학습 전', color=COLORS[0], edgecolor='white')
ax.bar(x + width/2, bytes_a, width, label='학습 후', color=COLORS[1], edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylabel('소스 크기 (bytes)', fontsize=11)
ax.set_title('생성된 소스 크기', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

# Unique source byte values = diversity
unique_b = len(set(bytes_b))
unique_a = len(set(bytes_a))

# Statements
stmts_b = [int(r[5]) for r in rows_tb]
stmts_a = [int(r[5]) for r in rows_ta]

ax = axes[1]
ax.set_facecolor(BG)
ax.bar(x - width/2, stmts_b, width, label='학습 전', color=COLORS[0], edgecolor='white')
ax.bar(x + width/2, stmts_a, width, label='학습 후', color=COLORS[1], edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylabel('명령문 수', fontsize=11)
ax.set_title('명령문 수', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

# Add diversity annotation
unique_stmts_b = len(set(stmts_b))
unique_stmts_a = len(set(stmts_a))
axes[0].text(0.5, 0.95, f'고유 크기: {unique_b}종 → {unique_a}종',
             transform=axes[0].transAxes, ha='center', va='top', fontsize=11,
             fontweight='bold', color=COLORS[2] if unique_a > unique_b else COLORS[3],
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1].text(0.5, 0.95, f'고유 구조: {unique_stmts_b}종 → {unique_stmts_a}종',
             transform=axes[1].transAxes, ha='center', va='top', fontsize=11,
             fontweight='bold', color=COLORS[2] if unique_stmts_a > unique_stmts_b else COLORS[3],
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig.savefig(OUT_DIR / '11_task_diversity.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  [4/5] 11_task_diversity.png")

# ===================================================================
# Chart 5: Summary Dashboard — Before vs After
# ===================================================================
fig = plt.figure(figsize=(16, 10), facecolor=BG)
fig.suptitle('AXOL AI — 학습 전 vs 학습 후 종합 비교', fontsize=18, fontweight='bold', y=0.98)

# 5a: Structure distribution pie (before)
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_facecolor(BG)
nonzero_b = [(l, c) for l, c in zip(labels, counts_b) if c > 0]
ax1.pie([c for _, c in nonzero_b],
        labels=[l for l, _ in nonzero_b],
        colors=[COLORS[labels.index(l)] for l, _ in nonzero_b],
        autopct='%1.0f%%', textprops={'fontsize': 10})
ax1.set_title('학습 전 구조 분포', fontsize=12, fontweight='bold')

# 5b: Structure distribution pie (after)
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_facecolor(BG)
nonzero_a = [(l, c) for l, c in zip(labels, counts_a) if c > 0]
ax2.pie([c for _, c in nonzero_a],
        labels=[l for l, _ in nonzero_a],
        colors=[COLORS[labels.index(l)] for l, _ in nonzero_a],
        autopct='%1.0f%%', textprops={'fontsize': 10})
ax2.set_title('학습 후 구조 분포', fontsize=12, fontweight='bold')

# 5c: Key metrics table
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_facecolor(BG)
ax3.axis('off')
ax3.set_title('핵심 변화', fontsize=12, fontweight='bold')

metrics = [
    ['지표', '학습 전', '학습 후'],
    ['활성 구조 수', f'{len([c for c in counts_b if c > 0])}개 / 5개',
                     f'{len([c for c in counts_a if c > 0])}개 / 5개'],
    ['활성 차원 수', f'{len([c for c in counts_dim_b if c > 0])}개 / 4개',
                     f'{len([c for c in counts_dim_a if c > 0])}개 / 4개'],
    ['구축 시간', f'{np.mean(time_b):.0f}ms', f'{np.mean(time_a)/1000:.1f}s'],
    ['생성 시간', '~0.3ms', '~0.3ms'],
    ['소스 다양성', f'{unique_b}종', f'{unique_a}종'],
]

table = ax3.table(cellText=metrics[1:], colLabels=metrics[0],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(COLORS[0])
        cell.set_text_props(color='white', fontweight='bold')
    elif col == 2:  # "after" column
        cell.set_facecolor('#F0F8F0')
    else:
        cell.set_facecolor('#F8F0F0' if col == 1 else '#F0F0F0')

# 5d: Dim distribution comparison
ax4 = fig.add_subplot(2, 3, 4)
ax4.set_facecolor(BG)
x = np.arange(len(dim_labels))
width = 0.35
ax4.bar(x - width/2, counts_dim_b, width, label='학습 전', color=COLORS[0], edgecolor='white')
ax4.bar(x + width/2, counts_dim_a, width, label='학습 후', color=COLORS[1], edgecolor='white')
ax4.set_xticks(x)
ax4.set_xticklabels(dim_labels)
ax4.set_title('차원 선택 분포', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.set_ylabel('시드 수')

# 5e: Construction time comparison (log scale)
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_facecolor(BG)
construct_data = [time_b, [t/1000 for t in time_a]]
bp = ax5.boxplot(construct_data, patch_artist=True, labels=['학습 전\n(ms)', '학습 후\n(초)'])
bp['boxes'][0].set_facecolor(COLORS[0])
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(COLORS[1])
bp['boxes'][1].set_alpha(0.7)
for m in bp['medians']:
    m.set_color(COLORS[3])
    m.set_linewidth(2)
ax5.set_title('구축 시간 분포', fontsize=12, fontweight='bold')
ax5.set_ylabel('시간')

# 5f: Tradeoff summary
ax6 = fig.add_subplot(2, 3, 6)
ax6.set_facecolor(BG)
ax6.axis('off')
ax6.set_title('트레이드오프 분석', fontsize=12, fontweight='bold')

tradeoff_text = (
    "개선점:\n"
    f"  구조 다양성: 1종 → {len([c for c in counts_a if c > 0])}종\n"
    f"  차원 다양성: 1종 → {len([c for c in counts_dim_a if c > 0])}종\n"
    f"  classify 분기 활성화\n"
    "\n비용:\n"
    f"  구축 시간: {np.mean(time_b):.0f}ms → {np.mean(time_a)/1000:.1f}s\n"
    f"  ({np.mean(time_a)/np.mean(time_b):.0f}x 느려짐)\n"
    "\n유지:\n"
    "  생성 시간: 변화 없음 (~0.3ms)\n"
    "  코드 실행: 100% 성공"
)
ax6.text(0.05, 0.95, tradeoff_text, transform=ax6.transAxes,
         fontsize=11, va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / '12_full_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  [5/5] 12_full_comparison.png")

print(f"\nDone! Comparison charts saved to: {OUT_DIR}")
