#!/usr/bin/env python3
"""Generate the 6-doc stratified OCR corpus.

Uses Pillow only. Each image is saved as a PNG < 200 KB. Reproducible:
seeded strings, fixed fonts, deterministic layout. Run once locally (no GPU).

Outputs to the directory containing this script.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent

# ---------- font loading (cross-platform fallback) ----------
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",                  # macOS
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",        # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",               # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",              # Linux
]
ITALIC_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
]
HANDWRITING_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Bradley Hand.ttc",
    "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
    "/System/Library/Fonts/Supplemental/Apple Chancery.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
]


def load_font(size: int, candidates=FONT_CANDIDATES):
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ---------- generators ----------
def gen_receipt() -> Image.Image:
    W, H = 400, 600
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    fbig = load_font(18)
    f = load_font(14)
    fsm = load_font(12)

    y = 15
    d.text((W // 2 - 70, y), "BLUEBIRD CAFE", fill="black", font=fbig); y += 25
    d.text((W // 2 - 60, y), "123 Market Street", fill="black", font=fsm); y += 16
    d.text((W // 2 - 60, y), "San Francisco, CA", fill="black", font=fsm); y += 16
    d.text((W // 2 - 55, y), "Tel (415) 555-0142", fill="black", font=fsm); y += 22
    d.line([(20, y), (W - 20, y)], fill="black"); y += 10
    d.text((20, y), "2026-05-13  09:47  #00234", fill="black", font=fsm); y += 22
    d.line([(20, y), (W - 20, y)], fill="black"); y += 10

    items = [
        ("Cappuccino",           "4.50"),
        ("Blueberry Muffin",     "3.75"),
        ("Egg & Cheese Bagel",   "7.25"),
        ("Orange Juice Lg",      "4.00"),
        ("Espresso Doppio",      "3.25"),
        ("Granola Yogurt Bowl",  "6.50"),
    ]
    for name, price in items:
        d.text((25, y), name, fill="black", font=f)
        d.text((W - 70, y), f"${price}", fill="black", font=f)
        y += 20
    y += 4
    d.line([(20, y), (W - 20, y)], fill="black"); y += 8
    d.text((25, y), "Subtotal",          fill="black", font=f); d.text((W - 70, y), "$29.25", fill="black", font=f); y += 18
    d.text((25, y), "Tax 8.5%",          fill="black", font=f); d.text((W - 70, y), "$2.49",  fill="black", font=f); y += 18
    d.text((25, y), "TOTAL",             fill="black", font=fbig); d.text((W - 80, y), "$31.74", fill="black", font=fbig); y += 26
    d.line([(20, y), (W - 20, y)], fill="black"); y += 10
    d.text((25, y), "VISA ****4521   APPROVED", fill="black", font=fsm); y += 18
    d.text((25, y), "AUTH 034921   REF 88201", fill="black", font=fsm); y += 22
    d.text((W // 2 - 70, y), "Thank you, come again!", fill="black", font=f)
    return img


def gen_article() -> Image.Image:
    W, H = 1000, 1400
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    fh = load_font(36)
    fb = load_font(18)
    fm = load_font(14)

    y = 40
    d.text((40, y), "Continuous Batching in Modern LLM Serving", fill="black", font=fh); y += 50
    d.text((40, y), "A Practitioner's Field Note   |   May 2026", fill="gray", font=fm); y += 40

    paras = [
        ("Continuous batching has become the dominant scheduling pattern in "
         "production LLM inference, quietly replacing the static-batch paradigm "
         "that dominated the first generation of transformer serving systems. "
         "Rather than waiting for a full batch to assemble before launching a "
         "forward pass, continuous batching admits new requests into the active "
         "batch at every token step, enabling short completions to finish early "
         "while long ones continue without blocking incoming traffic."),
        ("The benefits are largest for workloads with high variance in output "
         "length, a regime that covers almost every real-world chat, agent, and "
         "tool-using deployment. For OCR and other vision-language tasks, the "
         "output distribution is bounded but still variable: a receipt might "
         "produce forty tokens, a dense multi-column page fifteen hundred. "
         "Static batching would have to pad to the longest; continuous batching "
         "recycles the slot the moment the receipt finishes."),
        ("In practice, the implementation details are where the wins are either "
         "realized or squandered. Token-budget scheduling, prefix caching, "
         "paged KV memory, and careful attention to the decode-vs-prefill "
         "trade-off all compound. A naive system can leave fifty percent of its "
         "throughput on the floor simply by mis-tuning the max-num-seqs knob "
         "relative to the KV cache budget."),
    ]
    for p in paras:
        wrapped = wrap_text(p, fb, W - 80)
        for line in wrapped:
            d.text((40, y), line, fill="black", font=fb)
            y += 26
        y += 14
    return img


def gen_table() -> Image.Image:
    W, H = 1000, 800
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    fh = load_font(22)
    f = load_font(16)

    d.text((40, 30), "Q2 Server Benchmark Results — OCR Engines", fill="black", font=fh)

    headers = ["Engine", "Throughput (req/s)", "Latency p99 (ms)", "Cost / 1K req", "Accuracy (%)"]
    rows = [
        ["DeepSeek-OCR-2 BF16",     "25.4",  "1505", "$0.018", "76.3"],
        ["Qwen2-VL-7B",             "18.2",  "1820", "$0.023", "71.8"],
        ["GPT-4o (API)",            "12.7",  "2410", "$0.290", "82.4"],
        ["Tesseract 5.5 CPU",       "47.1",  "  310", "$0.004", "58.9"],
        ["Mistral-OCR",             "22.8",  "1640", "$0.021", "73.5"],
    ]

    col_x = [40, 260, 470, 660, 830, 990]
    y0 = 110
    row_h = 44

    # header row
    d.rectangle([(col_x[0], y0), (col_x[-1], y0 + row_h)], fill=(230, 230, 230), outline="black")
    for i, h in enumerate(headers):
        d.text((col_x[i] + 8, y0 + 12), h, fill="black", font=f)

    # data rows
    for r_idx, row in enumerate(rows):
        y = y0 + (r_idx + 1) * row_h
        d.rectangle([(col_x[0], y), (col_x[-1], y + row_h)], outline="black")
        for i, cell in enumerate(row):
            d.text((col_x[i] + 8, y + 12), cell, fill="black", font=f)

    # verticals
    for x in col_x:
        d.line([(x, y0), (x, y0 + row_h * (len(rows) + 1))], fill="black")

    note = ("All numbers measured on g6e.2xlarge (1x L40S 48GB) with bf16 precision "
            "except GPT-4o (managed service).")
    wrapped = wrap_text(note, load_font(13), W - 80)
    yy = y0 + row_h * (len(rows) + 1) + 30
    for line in wrapped:
        d.text((40, yy), line, fill="black", font=load_font(13)); yy += 20
    return img


def gen_formula() -> Image.Image:
    W, H = 1000, 800
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    fh = load_font(24)
    fb = load_font(17)
    fform = load_font(22)

    d.text((40, 30), "Attention Mechanics — Reference Sheet", fill="black", font=fh)

    # Intro paragraph
    intro = ("Scaled dot-product attention is parameterized by query, key, and "
             "value matrices. The softmax-normalized compatibility scores "
             "determine how much each value contributes to the output:")
    y = 90
    for line in wrap_text(intro, fb, W - 80):
        d.text((40, y), line, fill="black", font=fb); y += 24
    y += 10

    # Equation 1 (rendered as text with unicode math)
    d.text((80, y), "Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V", fill="black", font=fform); y += 50

    d.text((40, y), "For multi-head attention with h heads:", fill="black", font=fb); y += 34
    d.text((80, y), "MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O", fill="black", font=fform); y += 40
    d.text((80, y), "head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)", fill="black", font=fform); y += 50

    d.text((40, y), "FLOPs per attention layer (batch=B, seq=N, dim=D):", fill="black", font=fb); y += 34
    d.text((80, y), "F = 4 * B * N^2 * D + 4 * B * N * D^2", fill="black", font=fform); y += 40
    d.text((80, y), "memory = O(B * N^2)  (pre-FlashAttention)", fill="black", font=fform); y += 50

    note = ("FlashAttention reduces the memory term to O(B * N * D) by tiling "
            "over the sequence dimension and recomputing softmax statistics "
            "inside SRAM. The FLOP count is unchanged; the wall-clock speedup "
            "comes entirely from avoiding HBM round-trips.")
    for line in wrap_text(note, fb, W - 80):
        d.text((40, y), line, fill="black", font=fb); y += 24
    return img


def gen_dense() -> Image.Image:
    W, H = 1200, 1600
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    fh = load_font(28)
    fbyline = load_font(12)
    fb = load_font(13)  # small body to pack tokens

    d.text((40, 30), "THE DAILY GPU GAZETTE", fill="black", font=fh)
    d.text((40, 70), "Issue No. 47   |   Wednesday, May 13, 2026   |   All the silicon that's fit to print",
           fill="gray", font=fbyline)
    d.line([(40, 92), (W - 40, 92)], fill="black")

    left_col_x = 40
    right_col_x = W // 2 + 10
    col_w = (W // 2) - 60

    # Left column
    left_blocks = [
        ("BLACKWELL SUPPLY EASES AS Q3 WAFERS SHIP",
         ("Taiwan Semiconductor confirmed Tuesday that the third-quarter "
          "production run of Blackwell B300 dies has entered final packaging, "
          "a development that industry analysts say should relieve the acute "
          "capacity constraints that have dominated the inference-hardware "
          "market since late February. Hyperscaler allocations remain tightly "
          "controlled, but cloud providers are expected to begin publishing "
          "capacity-block pricing for the new generation within the next six "
          "weeks. At least three major platforms-as-a-service operators have "
          "briefed investors on their Q4 B300 fleet targets, with one citing "
          "an installed base approaching twenty thousand devices by year-end. "
          "Independent benchmarking suggests the B300 delivers between 1.6x "
          "and 2.4x the throughput of the preceding B200 on representative "
          "transformer workloads, with the upper end of that range reserved "
          "for fp4-native kernels that have only just landed in upstream "
          "attention libraries.")),
        ("L40S UTILIZATION SURGES IN EDGE OCR DEPLOYMENTS",
         ("Outside the frontier-training market, the unsung workhorse of "
          "mid-2026 has been the Ada-generation L40S. New deployments in "
          "document-AI and video-transcoding pipelines have pushed fleet "
          "utilization above seventy-five percent at three major managed "
          "service providers, according to telemetry published this morning "
          "by the GPU Operator working group. The chip's 48 GB of GDDR6 and "
          "its robust fp8 support make it a natural fit for the new "
          "generation of sub-10B parameter OCR models, which trade raw "
          "parameter count for specialized vision encoders and aggressive "
          "quantization. Industry sources indicate that unit economics at "
          "L40S inference have reached a threshold where on-premises "
          "deployment begins to rival managed API pricing for sustained "
          "workloads above roughly fifty thousand pages per day.")),
    ]

    # Right column
    right_blocks = [
        ("NCCL 2.28 RELEASE NOTES: A CLOSER LOOK",
         ("The long-awaited NCCL 2.28 release has landed in the nightly "
          "channel, and early adopters report that the collective operation "
          "latency improvements on PCIe-only topologies finally close the "
          "gap that opened with the original Blackwell launch. Prior "
          "versions of the library suffered a known sm_120 shared-memory "
          "bug that rendered allreduce unusable on PCIe-connected Blackwell "
          "fleets; 2.26.2 patched the immediate crash, but the performance "
          "gap relative to the Hopper-generation baseline persisted for "
          "nearly six months. The 2.28 patch addresses the underlying "
          "scheduling inefficiency. Measured allreduce bandwidth on a "
          "four-GPU ring now reaches 182 GB/s on representative 256 MB "
          "payloads.")),
        ("OPEN-SOURCE AGENT HARNESSES CONSOLIDATE",
         ("The fragmentation of the open-source agent-harness ecosystem "
          "appears to be ending, with three of the leading projects "
          "announcing interoperability commitments this week. The shared "
          "protocol will specify tool-call schemas, context compaction "
          "policies, and verification-gate semantics, allowing benchmark "
          "suites to compare harnesses on an apples-to-apples basis for "
          "the first time. Early reports from the joint working group "
          "suggest that the quality gap between harnesses is narrower than "
          "previously thought, and that the dominant contributor to fix "
          "rate is in fact the base model, followed closely by prompt "
          "engineering quality. Harness architecture accounts for perhaps "
          "ten to fifteen percent of observed variance.")),
    ]

    def draw_block(x, y, title, body, col_w):
        ft = load_font(16)
        d.text((x, y), title, fill="black", font=ft)
        yy = y + 26
        for line in wrap_text(body, fb, col_w):
            d.text((x, yy), line, fill="black", font=fb); yy += 18
        return yy + 14

    y_left = 110
    for title, body in left_blocks:
        y_left = draw_block(left_col_x, y_left, title, body, col_w)

    y_right = 110
    for title, body in right_blocks:
        y_right = draw_block(right_col_x, y_right, title, body, col_w)

    # vertical divider
    d.line([(W // 2, 100), (W // 2, max(y_left, y_right) + 20)], fill="black")
    return img


def gen_handwritten() -> Image.Image:
    W, H = 800, 1000
    # lined-paper background: white with faint blue horizontal lines
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for y in range(60, H, 40):
        d.line([(30, y), (W - 30, y)], fill=(200, 220, 240))
    d.line([(80, 30), (80, H - 30)], fill=(240, 180, 180))  # margin rule

    fhand = load_font(26, candidates=HANDWRITING_CANDIDATES)
    fdate = load_font(20, candidates=HANDWRITING_CANDIDATES)

    lines = [
        "Dear Claire,",
        "",
        "Just a quick note to say thank you",
        "for the books you left on the porch",
        "last Sunday. I started the Atwood one",
        "the same evening and finished it by",
        "Tuesday. The second chapter in",
        "particular was remarkable.",
        "",
        "We should catch up for coffee soon.",
        "How about Saturday at the usual spot,",
        "around ten in the morning?",
        "",
        "Warmly,",
        "Eleanor",
    ]

    # Date, top-right
    d.text((W - 260, 40), "May 13, 2026", fill=(30, 30, 90), font=fdate)

    y = 100
    for line in lines:
        d.text((100, y), line, fill=(30, 30, 90), font=fhand)
        y += 42
    return img


# ---------- helpers ----------
def wrap_text(text: str, font, max_w: int) -> list[str]:
    """Greedy word-wrap to max_w pixels."""
    tmp = Image.new("RGB", (10, 10))
    td = ImageDraw.Draw(tmp)
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        bbox = td.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_w:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def save_png(img: Image.Image, name: str) -> int:
    # optimize + aggressive compression to stay <200 KB
    path = OUT_DIR / name
    img.save(path, format="PNG", optimize=True, compress_level=9)
    return path.stat().st_size


def main() -> None:
    specs = [
        ("receipt.png",     gen_receipt),
        ("article.png",     gen_article),
        ("table.png",       gen_table),
        ("formula.png",     gen_formula),
        ("dense.png",       gen_dense),
        ("handwritten.png", gen_handwritten),
    ]
    for name, fn in specs:
        img = fn()
        size = save_png(img, name)
        print(f"{name}: {img.size[0]}x{img.size[1]}  {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
