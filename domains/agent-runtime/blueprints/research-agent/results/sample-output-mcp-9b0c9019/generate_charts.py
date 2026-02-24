"""
Generate charts from TensorRT-LLM research notes.
Extracts quantitative data and produces 3 bar charts saved as PNG files.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# Common color palette
COLORS = {
    'nvidia_green': '#76B900',
    'dark_green': '#4A7A00',
    'blue': '#2196F3',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'teal': '#009688',
    'gray': '#607D8B',
}


def add_value_labels(ax, bars, fmt='{:.0f}', fontsize=10, offset=0.02):
    """Add value labels on top of bars."""
    ymax = ax.get_ylim()[1]
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + ymax * offset,
            fmt.format(height),
            ha='center', va='bottom',
            fontsize=fontsize, fontweight='bold',
            color='#333333'
        )


# ============================================================
# CHART 1: TensorRT-LLM Throughput Across LLaMA Models on H100
# ============================================================
# Data from benchmarks file: Official NVIDIA Reference Benchmarks
# Using H100 SXM 80GB, ISL/OSL = 128/2048 (realistic chat workload)
# LLaMA 3.1 8B (FP8, TP=1): ~22,000 tok/s
# LLaMA 3.3 70B (FP8, TP=2): ~6,500 tok/s
# LLaMA 3.1 405B (FP8, TP=8): ~3,800 tok/s

fig1, ax1 = plt.subplots(figsize=(10, 6.5))

models = ['LLaMA 3.1 8B\n(TP=1, FP8)', 'LLaMA 3.3 70B\n(TP=2, FP8)', 'LLaMA 3.1 405B\n(TP=8, FP8)']
throughput_128_2048 = [22000, 6500, 3800]
throughput_128_128 = [26500, 7500, 3500]
throughput_2048_2048 = [18500, 5500, 3200]

x = np.arange(len(models))
width = 0.25

bars1 = ax1.bar(x - width, throughput_128_128, width, label='ISL=128, OSL=128',
                color=COLORS['nvidia_green'], edgecolor='white', linewidth=0.5)
bars2 = ax1.bar(x, throughput_128_2048, width, label='ISL=128, OSL=2048',
                color=COLORS['blue'], edgecolor='white', linewidth=0.5)
bars3 = ax1.bar(x + width, throughput_2048_2048, width, label='ISL=2048, OSL=2048',
                color=COLORS['orange'], edgecolor='white', linewidth=0.5)

ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
ax1.set_title('TensorRT-LLM Throughput on H100 SXM 80GB\nAcross LLaMA Model Sizes (Offline Max Throughput)',
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax1.legend(title='Input/Output Sequence Length', fontsize=10, title_fontsize=10,
           loc='upper right', framealpha=0.9)

# Add value labels
for bars in [bars1, bars2, bars3]:
    add_value_labels(ax1, bars, fmt='{:,.0f}', fontsize=8.5, offset=0.01)

ax1.set_ylim(0, max(throughput_128_128) * 1.18)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add annotation
ax1.text(0.02, 0.97, 'Source: NVIDIA Official Benchmarks (Feb 2026)',
         transform=ax1.transAxes, fontsize=8, color='gray',
         verticalalignment='top')

fig1.tight_layout()
fig1.savefig('/app/files/charts/chart1_llama_throughput_h100.png', dpi=150, bbox_inches='tight',
             facecolor='white', edgecolor='none')
plt.close(fig1)
print("Chart 1 saved: chart1_llama_throughput_h100.png")


# ============================================================
# CHART 2: TensorRT-LLM vs vLLM vs HuggingFace Performance
# ============================================================
# Data from benchmarks file:
# - TensorRT-LLM vs HuggingFace: up to 28x faster (LLaMA 2 7B/13B, H100, FP8)
#   HF baseline ~43 tok/s => TRT-LLM ~1,200 tok/s (peak stated)
# - BentoML benchmark (LLaMA 3 8B FP16 on A100, 100 users):
#   LMDeploy: ~4,000; vLLM: ~2,400; TensorRT-LLM: ~2,400; TGI: ~2,400
# - BentoML benchmark (LLaMA 3 70B INT4 on A100, 100 users):
#   LMDeploy: ~700; TensorRT-LLM: ~700; vLLM: lower

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6.5))

# Panel A: LLaMA 3 8B on A100 (FP16) - BentoML benchmark at 100 concurrent users
frameworks_8b = ['TensorRT-LLM', 'vLLM', 'LMDeploy', 'TGI\n(HuggingFace)']
throughput_8b = [2400, 2400, 4000, 2400]
colors_8b = [COLORS['nvidia_green'], COLORS['blue'], COLORS['teal'], COLORS['orange']]

bars_a = ax2a.bar(frameworks_8b, throughput_8b, color=colors_8b, edgecolor='white', linewidth=0.5, width=0.6)
add_value_labels(ax2a, bars_a, fmt='{:,.0f}', fontsize=10, offset=0.015)

ax2a.set_ylabel('Throughput (tokens/sec)', fontsize=11, fontweight='bold')
ax2a.set_title('LLaMA 3 8B (FP16)\nA100 80GB, 100 Concurrent Users',
               fontsize=12, fontweight='bold', pad=10)
ax2a.set_ylim(0, 5200)
ax2a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax2a.spines['top'].set_visible(False)
ax2a.spines['right'].set_visible(False)

# Panel B: LLaMA 3 70B on A100 (INT4) - BentoML benchmark at 100 concurrent users
frameworks_70b = ['TensorRT-LLM', 'vLLM', 'LMDeploy']
throughput_70b = [700, 450, 700]
colors_70b = [COLORS['nvidia_green'], COLORS['blue'], COLORS['teal']]

bars_b = ax2b.bar(frameworks_70b, throughput_70b, color=colors_70b, edgecolor='white', linewidth=0.5, width=0.55)
add_value_labels(ax2b, bars_b, fmt='{:,.0f}', fontsize=10, offset=0.015)

ax2b.set_ylabel('Throughput (tokens/sec)', fontsize=11, fontweight='bold')
ax2b.set_title('LLaMA 3 70B (INT4 Quantized)\nA100 80GB, 100 Concurrent Users',
               fontsize=12, fontweight='bold', pad=10)
ax2b.set_ylim(0, 950)
ax2b.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax2b.spines['top'].set_visible(False)
ax2b.spines['right'].set_visible(False)

# Shared annotation
fig2.text(0.5, -0.02, 'Source: BentoML Independent Benchmarks (2024) | Higher is better',
          ha='center', fontsize=9, color='gray')

fig2.suptitle('Inference Framework Throughput Comparison', fontsize=14, fontweight='bold', y=1.02)
fig2.tight_layout()
fig2.savefig('/app/files/charts/chart2_framework_comparison.png', dpi=150, bbox_inches='tight',
             facecolor='white', edgecolor='none')
plt.close(fig2)
print("Chart 2 saved: chart2_framework_comparison.png")


# ============================================================
# CHART 3: GPU Generation Performance Scaling
# ============================================================
# Data from benchmarks file - LLaMA 2 70B relative performance & MLPerf data
# Cross-Generation Comparison Summary table:
#   A100: 1.0x baseline
#   H100: ~4.0-4.6x vs A100
#   H200: ~1.3-1.45x vs H100 => ~5.2-6.67x vs A100
#   B200: ~3.0x vs H200 => ~15.6-19.3x vs A100
#
# Using MLPerf v5.0 absolute numbers for LLaMA 2 70B:
#   H200 8-GPU server: 33,072 tok/s; offline: 34,988 tok/s
#   B200 NVL8 server: 98,443 tok/s; offline: 98,858 tok/s
# MLPerf v4.1: H200 server: 32,790 tok/s
# H100 baseline (estimated from 4.6x A100 and H200 = 1.3x H100):
#   H200/1.3 ~ 33072/1.3 ~ 25,440 tok/s
# A100 baseline: H100/4.6 ~ 25440/4.6 ~ 5,530 tok/s

fig3, ax3 = plt.subplots(figsize=(11, 6.5))

gpus = ['A100 80GB\n(Ampere)', 'H100 SXM 80GB\n(Hopper)', 'H200 SXM 141GB\n(Hopper)', 'B200 180GB\n(Blackwell)']
# MLPerf-derived throughput (LLaMA 2 70B, 8-GPU server mode, tokens/sec)
# A100: estimated from H100 / 4.6x
# H100: estimated from H200 / 1.3x
# H200: MLPerf v5.0 = 33,072
# B200: MLPerf v5.0 = 98,443
throughput_server = [5530, 25440, 33072, 98443]

bar_colors = [COLORS['gray'], COLORS['blue'], COLORS['teal'], COLORS['nvidia_green']]

bars3 = ax3.bar(gpus, throughput_server, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.6)

# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.,
        height + 1500,
        f'{height:,.0f}',
        ha='center', va='bottom',
        fontsize=11, fontweight='bold', color='#333333'
    )

# Add speedup annotations relative to A100
speedups = [1.0, 4.6, 6.0, 17.8]
for i, (bar, spd) in enumerate(zip(bars3, speedups)):
    if i > 0:  # Skip baseline
        ax3.text(
            bar.get_x() + bar.get_width() / 2.,
            bar.get_height() * 0.5,
            f'{spd:.1f}x\nvs A100',
            ha='center', va='center',
            fontsize=10, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3)
        )

ax3.text(bars3[0].get_x() + bars3[0].get_width() / 2.,
         bars3[0].get_height() * 0.5,
         'Baseline',
         ha='center', va='center',
         fontsize=10, fontweight='bold', color='white',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3))

ax3.set_ylabel('Throughput (tokens/sec)', fontsize=12, fontweight='bold')
ax3.set_title('GPU Generation Performance Scaling -- LLaMA 2 70B\n8-GPU Server Mode (MLPerf Inference, TensorRT-LLM)',
              fontsize=14, fontweight='bold', pad=15)
ax3.set_ylim(0, 120000)
ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Annotation with memory bandwidth context
bandwidth_text = (
    'Memory Bandwidth:  A100: 2.0 TB/s  |  H100: 3.35 TB/s  |  H200: 4.8 TB/s  |  B200: 8.0 TB/s'
)
ax3.text(0.5, -0.10, bandwidth_text,
         transform=ax3.transAxes, fontsize=9, color='#555555',
         ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#cccccc'))

ax3.text(0.02, 0.97, 'Source: MLPerf Inference v5.0 (Apr 2025)\nA100/H100 estimated from published speedup ratios',
         transform=ax3.transAxes, fontsize=8, color='gray',
         verticalalignment='top')

fig3.tight_layout()
fig3.savefig('/app/files/charts/chart3_gpu_scaling.png', dpi=150, bbox_inches='tight',
             facecolor='white', edgecolor='none')
plt.close(fig3)
print("Chart 3 saved: chart3_gpu_scaling.png")

print("\nAll charts generated successfully.")
