#!/usr/bin/env python3
"""Generate a 1-page PDF research summary on SGLang."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
import os

OUTPUT_PATH = "/app/files/reports/sglang_research_summary.pdf"
CHART_DIR = "/app/files/charts"

# Page setup: letter size with tight margins for 1-page fit
doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=letter,
    topMargin=0.4 * inch,
    bottomMargin=0.35 * inch,
    leftMargin=0.55 * inch,
    rightMargin=0.55 * inch,
)

# ── Styles ────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

DARK_BLUE = HexColor("#1a3a5c")
ACCENT_BLUE = HexColor("#2563EB")
LIGHT_BG = HexColor("#f0f4f8")
DARK_TEXT = HexColor("#1e293b")
MEDIUM_TEXT = HexColor("#334155")

title_style = ParagraphStyle(
    "CustomTitle",
    parent=styles["Title"],
    fontSize=15,
    leading=18,
    textColor=DARK_BLUE,
    alignment=TA_CENTER,
    spaceAfter=1,
    fontName="Helvetica-Bold",
)

subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontSize=8.5,
    leading=10,
    textColor=MEDIUM_TEXT,
    alignment=TA_CENTER,
    spaceAfter=4,
    fontName="Helvetica",
)

section_style = ParagraphStyle(
    "SectionHead",
    parent=styles["Heading2"],
    fontSize=10,
    leading=12,
    textColor=ACCENT_BLUE,
    spaceBefore=5,
    spaceAfter=2,
    fontName="Helvetica-Bold",
)

body_style = ParagraphStyle(
    "BodyText2",
    parent=styles["Normal"],
    fontSize=7.5,
    leading=9.5,
    textColor=DARK_TEXT,
    alignment=TA_JUSTIFY,
    spaceAfter=2,
    fontName="Helvetica",
)

bullet_style = ParagraphStyle(
    "BulletItem",
    parent=body_style,
    leftIndent=12,
    bulletIndent=4,
    spaceAfter=1,
    fontSize=7.5,
    leading=9.2,
)

small_style = ParagraphStyle(
    "SmallText",
    parent=body_style,
    fontSize=6.5,
    leading=8,
    textColor=MEDIUM_TEXT,
)

# ── Helper ────────────────────────────────────────────────────────────
def section(title_text):
    return Paragraph(title_text, section_style)

def body(text):
    return Paragraph(text, body_style)

def bullet(text):
    return Paragraph(f"<bullet>&bull;</bullet> {text}", bullet_style)

def hr():
    return HRFlowable(
        width="100%", thickness=0.5, color=HexColor("#cbd5e1"),
        spaceBefore=3, spaceAfter=3
    )

# ── Build story ───────────────────────────────────────────────────────
story = []

# Title block
story.append(Paragraph("SGLang: A Research Summary on LLM Inference Framework", title_style))
story.append(Paragraph("February 22, 2026  |  Prepared from LMSYS research publications and official documentation", subtitle_style))
story.append(hr())

# ── Introduction ──────────────────────────────────────────────────────
story.append(section("1. Introduction"))
story.append(body(
    "<b>SGLang</b> (Structured Generation Language) is a high-performance open-source serving framework "
    "for large language models (LLMs) and multimodal models, developed at <b>UC Berkeley</b> under the "
    "<b>LMSYS</b> organization. First described by Zheng et al. (arXiv:2312.07104, 2023), it provides "
    "low-latency, high-throughput inference from single GPUs to distributed clusters. As of early 2026, "
    "SGLang powers <b>400,000+ GPUs globally</b> for organizations including xAI, NVIDIA, AMD, LinkedIn, "
    "Cursor, and all major cloud providers. The project has 23,600 GitHub stars, 1,182 contributors, "
    "and is at version <b>v0.5.8</b> (Apache-2.0 license)."
))

# ── Architecture ──────────────────────────────────────────────────────
story.append(section("2. Architecture Highlights"))
story.append(bullet(
    "<b>RadixAttention</b> -- Organizes KV cache in a radix tree for automatic prefix reuse across requests. "
    "Delivers up to <b>5x throughput gain</b> with zero overhead on cache misses. Accelerates few-shot learning, "
    "multi-turn chat, self-consistency sampling, and tree-of-thought search."
))
story.append(bullet(
    "<b>Compressed Finite State Machine</b> -- Jump-forward decoding skips deterministic token sequences during "
    "constrained generation, making structured output <b>faster than unconstrained decoding</b>. With the "
    "xgrammar backend (v0.4+), achieves up to <b>10x faster JSON decoding</b>."
))
story.append(bullet(
    "<b>Zero-Overhead Scheduler</b> -- Overlaps CPU batch scheduling with GPU execution via CUDA event "
    "synchronization. A 4K-line Python scheduler that matches or beats C++ alternatives, yielding "
    "<b>1.3x throughput</b> over competing solutions."
))
story.append(bullet(
    "<b>Three-Process Architecture</b> -- TokenizerManager, Scheduler, and DetokenizerManager communicate via "
    "ZMQ IPC for continuous batching with chunked prefill and priority-based request management."
))

# ── Features ──────────────────────────────────────────────────────────
story.append(section("3. Performance Features"))

# Build a compact 2-column table for features
feature_data = [
    ["Category", "Details"],
    ["Parallelism", "Tensor (TP), Pipeline (PP), Expert (EP up to 96 GPUs), Data (DP), Context (CP)"],
    ["Model Support", "166 configs: Llama, Qwen, DeepSeek, Gemma, Mistral, Mixtral, vision-language, diffusion, embedding"],
    ["Quantization", "FP4, FP8, INT4, AWQ, GPTQ, BitsAndBytes, GGUF formats"],
    ["Hardware", "NVIDIA (GB200/B300/H100/A100), AMD (MI355/MI300), Intel Xeon, Google TPU, Huawei Ascend"],
    ["Speculative Decoding", "EAGLE-2/3, MTP, Ngram -- EAGLE-3 reaches 373 tok/s (2.36x) on Llama-8B"],
    ["Other", "Prefill-decode disaggregation (2.7-3.8x), multi-LoRA batching, cache-aware load balancing (1.9x)"],
]

table_style = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT_BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 7),
    ("LEADING", (0, 0), (-1, -1), 8.5),
    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ("FONTSIZE", (0, 1), (0, -1), 7),
    ("TEXTCOLOR", (0, 1), (0, -1), DARK_BLUE),
    ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BG),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
    ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#94a3b8")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
])

feat_table = Table(
    [[Paragraph(cell, ParagraphStyle("tc", fontSize=7, leading=8.5, fontName="Helvetica"))
      if i > 0 or j > 0 else Paragraph(cell, ParagraphStyle("th", fontSize=7, leading=8.5, fontName="Helvetica-Bold", textColor=colors.white))
      for j, cell in enumerate(row)]
     for i, row in enumerate(feature_data)],
    colWidths=[1.05 * inch, 5.85 * inch],
)
feat_table.setStyle(table_style)
story.append(feat_table)

# ── Benchmarks ────────────────────────────────────────────────────────
story.append(section("4. Benchmark Highlights"))
story.append(body(
    "SGLang demonstrates <b>up to 6.4x throughput</b> over state-of-the-art systems across diverse workloads "
    "(Llama, DeepSeek, LLaVA, structured output). Key results: "
    "<b>3.1x</b> over vLLM on Llama-70B (8xA100), "
    "<b>3-7x</b> on DeepSeek MLA (H100), "
    "<b>10x</b> faster JSON decoding (xgrammar), "
    "<b>4.5x</b> on LLaVA-OneVision vs HuggingFace, and "
    "<b>4.8x decode throughput</b> on GB200 NVL72. "
    "On Qwen3-235B (AMD MI300X), TTFT improved 1.67x (756ms to 451ms) and TPOT improved 2.12x (26ms to 12ms). "
    "SGLang matches or exceeds TensorRT-LLM on most configs despite being implemented in Python."
))

# ── Charts ────────────────────────────────────────────────────────────
story.append(Spacer(1, 3))

# Two charts side by side
chart1_path = os.path.join(CHART_DIR, "sglang_throughput_comparison.png")
chart2_path = os.path.join(CHART_DIR, "sglang_speedup_factors.png")

# Get image dimensions to compute aspect ratios
from PIL import Image as PILImage

img1 = PILImage.open(chart1_path)
img2 = PILImage.open(chart2_path)

# Target widths for side-by-side layout
total_width = 6.9 * inch
gap = 0.15 * inch
chart1_w = 3.7 * inch
chart2_w = total_width - chart1_w - gap

# Compute heights preserving aspect ratio
chart1_h = chart1_w * (img1.height / img1.width)
chart2_h = chart2_w * (img2.height / img2.width)

chart_table = Table(
    [[Image(chart1_path, width=chart1_w, height=chart1_h),
      Image(chart2_path, width=chart2_w, height=chart2_h)]],
    colWidths=[chart1_w + gap / 2, chart2_w + gap / 2],
)
chart_table.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ("TOPPADDING", (0, 0), (-1, -1), 0),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
]))
story.append(chart_table)

story.append(Paragraph(
    "<i>Figure 1 (left): Throughput advantage across benchmarks. "
    "Figure 2 (right): Optimization speedup factors ranked by magnitude.</i>",
    ParagraphStyle("Caption", fontSize=6.5, leading=8, textColor=MEDIUM_TEXT, alignment=TA_CENTER, spaceAfter=2)
))

# ── Conclusion ────────────────────────────────────────────────────────
story.append(hr())
story.append(section("5. Conclusion"))
story.append(body(
    "SGLang has established itself as a leading open-source LLM inference framework through its co-designed "
    "runtime and frontend language. Its RadixAttention KV cache reuse and compressed FSM constrained decoding "
    "represent genuine architectural innovations rather than incremental improvements. With broad hardware support "
    "(NVIDIA, AMD, Intel, TPU, Ascend), 166 model configurations, and adoption by major cloud providers and AI "
    "companies, SGLang is well-positioned as the go-to framework for production LLM serving. Its Python-native "
    "scheduler matching C++ alternatives in performance, combined with rapid community growth (1,182 contributors), "
    "suggest sustained momentum in the evolving LLM infrastructure ecosystem."
))

# Footer
story.append(Spacer(1, 3))
story.append(Paragraph(
    "Sources: Zheng et al., arXiv:2312.07104 (2023) | github.com/sgl-project/sglang | docs.sglang.io | lmsys.org/blog",
    ParagraphStyle("Footer", fontSize=6, leading=7, textColor=HexColor("#94a3b8"), alignment=TA_CENTER)
))

# ── Build ─────────────────────────────────────────────────────────────
doc.build(story)
print(f"PDF generated: {OUTPUT_PATH}")
print(f"File size: {os.path.getsize(OUTPUT_PATH):,} bytes")
