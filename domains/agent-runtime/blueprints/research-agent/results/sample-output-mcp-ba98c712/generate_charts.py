"""
Generate SGLang performance benchmark charts from research notes data.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ============================================================
# Global styling
# ============================================================
plt.rcParams.update({
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#F8F9FA',
    'axes.edgecolor': '#DEE2E6',
    'axes.grid': True,
    'grid.color': '#E9ECEF',
    'grid.linewidth': 0.7,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# Color palette
SGLANG_COLOR = '#2563EB'    # Blue
VLLM_COLOR = '#DC2626'      # Red
TRTLLM_COLOR = '#16A34A'    # Green
TGI_COLOR = '#9333EA'       # Purple
ACCENT_COLORS = ['#2563EB', '#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE']

# ============================================================
# Chart 1: SGLang Throughput vs Other Frameworks
# ============================================================
def chart1_throughput_comparison():
    fig, ax = plt.subplots(figsize=(14, 7))

    # Data from research notes: comparative throughput multipliers
    # All numbers are throughput relative to vLLM (vLLM = 1.0x baseline)
    scenarios = [
        'Llama-8B\n1xA100, bf16',
        'Llama-70B\n8xA100, bf16',
        'Llama-70B\n8xH100, fp8\n(short input)',
        'DeepSeek MLA\n8xH100',
        'Structured Output\n(JSON)',
        'LLaVA-OneVision\nvs HF',
    ]

    # Throughput in normalized units (tok/s where available, or multiplier)
    # Using absolute values where available, else normalized to vLLM=1.0
    sglang_vals     = [5000, 3.1, 3.0, 5.0, 10.0, 4.5]
    vllm_vals       = [3300, 1.0, 1.0, 1.0, 1.0,  1.0]
    trtllm_vals     = [5000, 1.5, 2.0, None, None, None]

    # For the chart, normalize everything as "x faster than baseline"
    # Use the multiplier data directly
    labels = [
        'Llama-8B\n1xA100',
        'Llama-70B\n8xA100',
        'Llama-70B\n8xH100\n(short input)',
        'DeepSeek MLA\n8xH100',
        'Structured\nOutput (JSON)',
        'LLaVA-OneVision\nvs HuggingFace',
    ]

    # Throughput multipliers vs vLLM baseline (vLLM = 1.0)
    # From the notes:
    # - Llama-8B: SGLang ~5000, vLLM ~3300 => SGLang ~1.5x. TRT-LLM ~5000 => ~1.5x
    # - Llama-70B 8xA100: SGLang 3.1x vs vLLM. TRT-LLM roughly comparable to SGLang on latency but not throughput
    # - Llama-70B 8xH100 short: SGLang wins vs both
    # - DeepSeek MLA: 3-7x vs baseline (use 5x median)
    # - Structured output: 3-10x (use 10x for xgrammar)
    # - LLaVA: 4.5x vs HuggingFace (not vLLM)

    # Let's use absolute throughput where we have it, else relative
    # Better approach: show multiplier vs baseline for each scenario

    scenarios_clean = [
        'Llama-8B\n1xA100, bf16',
        'Llama-70B\n8xA100, bf16',
        'DeepSeek MLA\nH100',
        'JSON Decoding\n(xgrammar)',
        'LLaVA-OneVision',
        'Overall\n(paper peak)',
    ]

    # All values as "throughput multiplier vs baseline system"
    sglang_mult = [1.52, 3.1, 5.0, 10.0, 4.5, 6.4]
    baseline_labels = [
        'vs vLLM',
        'vs vLLM',
        'vs baseline',
        'vs alternatives',
        'vs HuggingFace',
        'vs SoTA',
    ]

    x = np.arange(len(scenarios_clean))
    width = 0.5

    bars = ax.bar(x, sglang_mult, width, color=SGLANG_COLOR, edgecolor='white',
                  linewidth=1.2, zorder=3, label='SGLang speedup')

    # Add a dashed line at 1.0 (baseline)
    ax.axhline(y=1.0, color='#6B7280', linestyle='--', linewidth=1.5, zorder=2,
               label='Baseline (1.0x)')

    # Add value labels on bars
    for bar, mult, bl in zip(bars, sglang_mult, baseline_labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.15,
                f'{mult}x',
                ha='center', va='bottom', fontweight='bold', fontsize=12,
                color=SGLANG_COLOR)
        ax.text(bar.get_x() + bar.get_width() / 2., 0.05,
                bl, ha='center', va='bottom', fontsize=8,
                color='white', fontweight='bold')

    ax.set_xlabel('Benchmark Scenario', fontsize=12)
    ax.set_ylabel('Throughput Multiplier (higher is better)', fontsize=12)
    ax.set_title('SGLang Throughput Advantage Across Benchmarks', fontsize=15, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_clean, fontsize=9.5)
    ax.set_ylim(0, max(sglang_mult) * 1.25)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

    # Add subtle gradient effect on bars
    for bar in bars:
        bar.set_alpha(0.9)

    plt.tight_layout()
    plt.savefig('/app/files/charts/sglang_throughput_comparison.png', dpi=180,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Chart 1 saved: sglang_throughput_comparison.png")


# ============================================================
# Chart 2: SGLang Key Speedup Factors (horizontal bar)
# ============================================================
def chart2_speedup_factors():
    fig, ax = plt.subplots(figsize=(13, 8))

    # All documented speedup factors from the research notes
    features = [
        'xgrammar JSON Decoding\n(vs alternatives)',
        'DeepSeek MLA Kernels\n(vs baseline, H100)',
        'Overall Peak\n(paper, diverse workloads)',
        'RadixAttention\nPrefix Caching',
        'LLaVA-OneVision\n(vs HuggingFace)',
        'GB200 NVL72\nDecode Throughput',
        'Prefill-Decode\nDisaggregation (high)',
        'Llama-70B\n(vs vLLM, 8xA100)',
        'Prefill-Decode\nDisaggregation (low)',
        'Compressed FSM\n(vs Outlines+vLLM)',
        'EAGLE-3 Speculative\n(vs baseline)',
        'Cache-Aware\nLoad Balancer',
        'DP Attention for\nDeepSeek (8xH100)',
        'torch.compile\n(batch 1-32)',
        'Zero-Overhead\nScheduler (vs SoTA)',
    ]

    speedups = [10.0, 7.0, 6.4, 5.0, 4.5, 4.8, 3.8, 3.1, 2.7, 2.5, 2.36, 1.9, 1.9, 1.5, 1.3]

    # Sort by speedup descending
    sorted_pairs = sorted(zip(speedups, features), reverse=True)
    speedups_sorted = [s for s, f in sorted_pairs]
    features_sorted = [f for s, f in sorted_pairs]

    y_pos = np.arange(len(features_sorted))

    # Color gradient based on magnitude
    norm = plt.Normalize(min(speedups_sorted), max(speedups_sorted))
    cmap = plt.cm.Blues
    colors = [cmap(0.35 + 0.6 * norm(s)) for s in speedups_sorted]

    bars = ax.barh(y_pos, speedups_sorted, height=0.65, color=colors,
                   edgecolor='white', linewidth=1.0, zorder=3)

    # Add value annotations
    for bar, val in zip(bars, speedups_sorted):
        w = bar.get_width()
        ax.text(w + 0.12, bar.get_y() + bar.get_height() / 2.,
                f'{val}x', va='center', ha='left', fontweight='bold',
                fontsize=11, color='#1E3A5F')

    ax.axvline(x=1.0, color='#DC2626', linestyle='--', linewidth=1.5, zorder=2,
               label='Baseline (1.0x)', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_sorted, fontsize=9.5)
    ax.set_xlabel('Speedup Factor (x times faster)', fontsize=12)
    ax.set_title('SGLang Optimization Speedup Factors', fontsize=15, pad=15)
    ax.set_xlim(0, max(speedups_sorted) * 1.2)
    ax.invert_yaxis()
    ax.legend(loc='lower right', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    plt.savefig('/app/files/charts/sglang_speedup_factors.png', dpi=180,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Chart 2 saved: sglang_speedup_factors.png")


# ============================================================
# Chart 3: Speculative Decoding + Llama-70B Comparison
# ============================================================
def chart3_detailed_benchmarks():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left panel: Speculative decoding throughput on Llama-8B ---
    methods = ['Baseline\n(no speculation)', 'EAGLE-2', 'EAGLE-3']
    throughputs = [158.34, 244.10, 373.25]

    bars1 = ax1.bar(methods, throughputs, color=[VLLM_COLOR, '#F59E0B', SGLANG_COLOR],
                    edgecolor='white', linewidth=1.2, width=0.55, zorder=3)

    for bar, val in zip(bars1, throughputs):
        ax1.text(bar.get_x() + bar.get_width() / 2., val + 8,
                 f'{val:.1f}\ntok/s', ha='center', va='bottom',
                 fontweight='bold', fontsize=11)

    # Add speedup annotations
    ax1.annotate('1.54x', xy=(1, 244.10), xytext=(1.3, 195),
                 fontsize=10, fontweight='bold', color='#F59E0B',
                 arrowprops=dict(arrowstyle='->', color='#F59E0B', lw=1.5))
    ax1.annotate('2.36x', xy=(2, 373.25), xytext=(2.3, 300),
                 fontsize=10, fontweight='bold', color=SGLANG_COLOR,
                 arrowprops=dict(arrowstyle='->', color=SGLANG_COLOR, lw=1.5))

    ax1.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax1.set_title('Speculative Decoding on Llama-8B', fontsize=13, pad=12)
    ax1.set_ylim(0, max(throughputs) * 1.3)

    # --- Right panel: Llama-70B SGLang vs vLLM vs TRT-LLM ---
    configs = ['8xA100\nbf16', '8xH100\nfp8 (short)', '8xH100\nfp8 (long)']

    # Relative throughput (vLLM = 1.0)
    sglang_rel = [3.1, 3.0, 2.0]
    vllm_rel = [1.0, 1.0, 1.0]
    trtllm_rel = [1.5, 1.8, 2.5]

    x = np.arange(len(configs))
    width = 0.25

    b1 = ax2.bar(x - width, sglang_rel, width, label='SGLang',
                 color=SGLANG_COLOR, edgecolor='white', linewidth=1.0, zorder=3)
    b2 = ax2.bar(x, vllm_rel, width, label='vLLM',
                 color=VLLM_COLOR, edgecolor='white', linewidth=1.0, zorder=3)
    b3 = ax2.bar(x + width, trtllm_rel, width, label='TensorRT-LLM',
                 color=TRTLLM_COLOR, edgecolor='white', linewidth=1.0, zorder=3)

    for bars, vals in [(b1, sglang_rel), (b2, vllm_rel), (b3, trtllm_rel)]:
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2., val + 0.06,
                     f'{val}x', ha='center', va='bottom', fontsize=9.5,
                     fontweight='bold')

    ax2.axhline(y=1.0, color='#6B7280', linestyle=':', linewidth=1.0, zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=10)
    ax2.set_ylabel('Relative Throughput (vLLM = 1.0x)', fontsize=12)
    ax2.set_title('Llama-70B: Framework Comparison', fontsize=13, pad=12)
    ax2.set_ylim(0, max(max(sglang_rel), max(trtllm_rel)) * 1.3)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=10)

    plt.suptitle('SGLang Detailed Performance Benchmarks', fontsize=16,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/app/files/charts/sglang_detailed_benchmarks.png', dpi=180,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Chart 3 saved: sglang_detailed_benchmarks.png")


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    chart1_throughput_comparison()
    chart2_speedup_factors()
    chart3_detailed_benchmarks()
    print("\nAll charts generated successfully.")
