import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Consistent color scheme
PRIMARY = '#2563EB'      # Blue
SECONDARY = '#10B981'    # Green/Teal
ACCENT = '#F59E0B'       # Amber
DANGER = '#EF4444'       # Red
NEUTRAL = '#6B7280'      # Gray
DARK = '#1F2937'         # Dark gray for text

# Common style settings
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#CCCCCC',
    'text.color': DARK,
    'axes.labelcolor': DARK,
    'xtick.color': DARK,
    'ytick.color': DARK,
})

# ============================================================
# Chart 1: vLLM Throughput Multiplier vs Other Frameworks
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

frameworks = ['HuggingFace\nTransformers', 'HuggingFace\nTGI', 'NVIDIA\nFasterTransformer']
# Midpoints of ranges: (14+24)/2=19, (2.2+3.5)/2=2.85, (2+4)/2=3
midpoints = [19.0, 2.85, 3.0]
# Range bounds for error bars
low_vals = [14, 2.2, 2]
high_vals = [24, 3.5, 4]
errors_low = [m - l for m, l in zip(midpoints, low_vals)]
errors_high = [h - m for h, m in zip(high_vals, midpoints)]

colors = [PRIMARY, SECONDARY, ACCENT]
bars = ax.bar(frameworks, midpoints, color=colors, width=0.55, edgecolor='white', linewidth=1.5, zorder=3)

# Add error bars for ranges
ax.errorbar(frameworks, midpoints,
            yerr=[errors_low, errors_high],
            fmt='none', ecolor=DARK, elinewidth=2, capsize=8, capthick=2, zorder=4)

# Add value labels on bars
for bar, mid, low, high in zip(bars, midpoints, low_vals, high_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f'{low}x - {high}x',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color=DARK)

ax.set_ylabel('Throughput Multiplier (vLLM vs. Framework)', fontsize=13)
ax.set_title('vLLM Throughput Advantage Over Other Frameworks', fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(0, 30)
ax.axhline(y=1, color=NEUTRAL, linestyle='--', linewidth=1, alpha=0.5, label='Baseline (1x)')
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add subtitle
fig.text(0.5, 0.01, 'Source: vLLM PagedAttention paper (SOSP 2023) and vLLM launch blog. Bars show midpoints; whiskers show reported ranges.',
         ha='center', fontsize=9, color=NEUTRAL, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
fig.savefig('/app/files/charts/throughput_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 1 saved: throughput_comparison.png")

# ============================================================
# Chart 2: Memory Waste Reduction
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))

categories = ['Traditional Systems\n(Pre-allocated KV Cache)', 'vLLM\n(PagedAttention)']
# Use midpoint of 60-80% for traditional: 70%. vLLM: under 4%, use 4%.
waste_values = [70, 4]
bar_colors = [DANGER, SECONDARY]

bars = ax.bar(categories, waste_values, color=bar_colors, width=0.5, edgecolor='white', linewidth=2, zorder=3)

# Add range annotation for traditional systems
ax.annotate('60-80% waste\n(fragmentation &\nover-allocation)',
            xy=(0, 70), xytext=(0.42, 82),
            fontsize=11, ha='center', color=DANGER, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=DANGER, lw=1.5))

# Add vLLM annotation
ax.annotate('Under 4% waste\n(block-level allocation)',
            xy=(1, 4), xytext=(1.42, 25),
            fontsize=11, ha='center', color='#059669', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#059669', lw=1.5))

# Add percentage labels on bars
for bar, val in zip(bars, waste_values):
    y_pos = bar.get_height() + 1
    label = f'{val}%'
    if val == 70:
        label = '~70%'
    elif val == 4:
        label = '<4%'
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
            label, ha='center', va='bottom', fontsize=15, fontweight='bold', color=DARK)

# Add a "reduction" arrow between the two bars
ax.annotate('',
            xy=(0.85, 40), xytext=(0.15, 40),
            arrowprops=dict(arrowstyle='<->', color=PRIMARY, lw=2.5))
ax.text(0.5, 43, '~95% reduction\nin memory waste', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color=PRIMARY)

ax.set_ylabel('KV Cache Memory Waste (%)', fontsize=13)
ax.set_title('KV Cache Memory Waste: Traditional Systems vs. vLLM', fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')

fig.text(0.5, 0.01,
         'Source: Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" (SOSP 2023).',
         ha='center', fontsize=9, color=NEUTRAL, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.07)
fig.savefig('/app/files/charts/memory_waste_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 2 saved: memory_waste_comparison.png")

# ============================================================
# Chart 3: Community / Adoption Metrics
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['GitHub Stars', 'Forks', 'Downstream\nProjects', 'Contributors']
values = [70900, 13600, 7600, 2205]
colors_chart3 = [PRIMARY, SECONDARY, ACCENT, '#8B5CF6']  # Purple for contributors

bars = ax.barh(metrics, values, color=colors_chart3, height=0.55, edgecolor='white', linewidth=1.5, zorder=3)

# Add value labels
for bar, val in zip(bars, values):
    label = f'{val:,}'
    ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
            label, ha='left', va='center', fontsize=14, fontweight='bold', color=DARK)

ax.set_xlabel('Count', fontsize=13)
ax.set_title('vLLM Open-Source Community & Adoption Metrics', fontsize=16, fontweight='bold', pad=15)
ax.set_xlim(0, max(values) * 1.25)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.invert_yaxis()

fig.text(0.5, 0.01,
         'Source: GitHub repository (github.com/vllm-project/vllm), as of February 2026.',
         ha='center', fontsize=9, color=NEUTRAL, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.07)
fig.savefig('/app/files/charts/community_adoption_metrics.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Chart 3 saved: community_adoption_metrics.png")

print("\nAll charts generated successfully!")
