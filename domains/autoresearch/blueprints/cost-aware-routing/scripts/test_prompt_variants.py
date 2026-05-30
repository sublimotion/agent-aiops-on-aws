"""Test whether DeepSeek-bias is prompt-driven or model-prior-driven.

Generates 256 routing decisions on the train.jsonl mix, varying the
ROUTER PROMPT in three ways:

  V0 (control):    current prompt (worker_pool.build_metadata_prompt
                   + few_shot.render_few_shot_block).
  V1 (no-DeepSeek-fewshot):
                   replace the DeepSeek example with a Mistral example
                   (same question, different pick), keep metadata as-is.
  V2 (no-DeepSeek-metadata):
                   keep few-shot as-is but reword DeepSeek's metadata
                   from "good MATH/AIME mid-tier choice" to a more
                   neutral description ("strong general reasoning,
                   moderate cost").
  V3 (both):       V1 + V2.

Compare pick distributions across V0..V3 on the same 256 questions
(deterministic seed). Reveals whether DeepSeek's 24% iter-0 share is
inflated by prompt design or is a baked-in model preference.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time

import boto3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/opt/dlami/nvme/cost-aware-routing/scripts")
from train_cost_aware_router import batch_generate, parse_pick
from worker_pool import POOL


# === Variant prompts ===
def build_v0_prompt(question: str) -> str:
    """Original prompt — control."""
    from worker_pool import build_metadata_prompt
    from few_shot import render_few_shot_block
    return (
        build_metadata_prompt() + "\n\n"
        + render_few_shot_block()
        + f"\nQuestion: {question}\nAnswer:"
    )


def build_v1_prompt(question: str) -> str:
    """V1: drop the DeepSeek-promoting math example from few-shot.

    Replace EXAMPLES[5] (math word problem -> ord_5 DeepSeek) with the
    same question -> ord_4 Mistral. Keep the worker pool listing as-is.
    """
    from worker_pool import build_metadata_prompt
    from few_shot import EXAMPLES

    parts = ["", "Examples (one per worker, balanced across capabilities):", ""]
    modified_examples = list(EXAMPLES)
    # Replace index 5 (DeepSeek) — same question, point at Mistral instead.
    orig_q, _, _ = modified_examples[5]
    modified_examples[5] = (
        orig_q,
        4,
        "Multi-step word problem with arithmetic reasoning — Mistral Large 3 is a strong reasoning mid-tier.",
    )
    for i, (q, ord_, why) in enumerate(modified_examples, 1):
        parts.append(f"Example {i}:")
        parts.append(f"  Question: {q}")
        parts.append(f"  PICK ord_{ord_}")
        parts.append(f"  Justification: {why}")
        parts.append("")
    parts.append("Now answer the actual question:")
    parts.append("")
    return (
        build_metadata_prompt() + "\n\n"
        + "\n".join(parts)
        + f"\nQuestion: {question}\nAnswer:"
    )


def build_v2_prompt(question: str) -> str:
    """V2: rewrite DeepSeek's metadata description to remove MATH/AIME framing."""
    from few_shot import render_few_shot_block
    from worker_pool import POOL, reference_cost_usd

    lines = [
        "You are a routing controller. Pick exactly ONE worker (by ord) to answer the user's question.",
        "Each worker has a distinct cost-quality profile. Pick the cheapest worker likely to answer correctly.",
        "",
        "Available workers:",
    ]
    # Override DeepSeek's strength only:
    NEUTRAL_DEEPSEEK = "Strong general reasoning at moderate cost; broadly capable but not specialized."
    for w in POOL:
        ref_q = reference_cost_usd(w.ord)
        strength = NEUTRAL_DEEPSEEK if w.ord == 5 else w.qualitative_strength
        lines.append(
            f"  ord_{w.ord}: {w.name}  (${w.in_per_1M:.2f}/${w.out_per_1M:.2f} per 1M tok in/out → "
            f"${ref_q:.5f}/query at typical CoT length)\n"
            f"           {strength}"
        )
    lines.append("")
    lines.append("Respond with EXACTLY one line: PICK ord_N (where N is 0..8). "
                 "Optionally add one short sentence of justification.")
    metadata = "\n".join(lines)

    return (
        metadata + "\n\n"
        + render_few_shot_block()
        + f"\nQuestion: {question}\nAnswer:"
    )


def build_v3_prompt(question: str) -> str:
    """V3: both V1 + V2."""
    from worker_pool import POOL, reference_cost_usd
    from few_shot import EXAMPLES

    NEUTRAL_DEEPSEEK = "Strong general reasoning at moderate cost; broadly capable but not specialized."
    metadata_lines = [
        "You are a routing controller. Pick exactly ONE worker (by ord) to answer the user's question.",
        "Each worker has a distinct cost-quality profile. Pick the cheapest worker likely to answer correctly.",
        "",
        "Available workers:",
    ]
    for w in POOL:
        ref_q = reference_cost_usd(w.ord)
        strength = NEUTRAL_DEEPSEEK if w.ord == 5 else w.qualitative_strength
        metadata_lines.append(
            f"  ord_{w.ord}: {w.name}  (${w.in_per_1M:.2f}/${w.out_per_1M:.2f} per 1M tok in/out → "
            f"${ref_q:.5f}/query at typical CoT length)\n"
            f"           {strength}"
        )
    metadata_lines.append("")
    metadata_lines.append("Respond with EXACTLY one line: PICK ord_N. Optionally add one short sentence.")
    metadata = "\n".join(metadata_lines)

    fs_parts = ["", "Examples (one per worker, balanced across capabilities):", ""]
    modified = list(EXAMPLES)
    orig_q, _, _ = modified[5]
    modified[5] = (
        orig_q, 4,
        "Multi-step word problem with arithmetic reasoning — Mistral Large 3 is a strong reasoning mid-tier.",
    )
    for i, (q, ord_, why) in enumerate(modified, 1):
        fs_parts.append(f"Example {i}:")
        fs_parts.append(f"  Question: {q}")
        fs_parts.append(f"  PICK ord_{ord_}")
        fs_parts.append(f"  Justification: {why}")
        fs_parts.append("")
    fs_parts.append("Now answer the actual question:")
    fs_parts.append("")

    return (
        metadata + "\n\n"
        + "\n".join(fs_parts)
        + f"\nQuestion: {question}\nAnswer:"
    )


PROMPT_BUILDERS = {
    "V0_control": build_v0_prompt,
    "V1_no_ds_fewshot": build_v1_prompt,
    "V2_no_ds_metadata": build_v2_prompt,
    "V3_both": build_v3_prompt,
}


def run_variant(variant: str, model, tokenizer, questions: list[dict], gen_batch: int) -> dict:
    builder = PROMPT_BUILDERS[variant]
    prompts = [builder(q["question"]) for q in questions]
    print(f"\n=== {variant}: {len(prompts)} rollouts ===")
    t0 = time.time()
    texts = batch_generate(
        model, tokenizer, prompts,
        batch_size=gen_batch, max_new_tokens=64, temperature=0.7,
    )
    wall = time.time() - t0
    picks = collections.Counter()
    parse_fail = 0
    for t in texts:
        p = parse_pick(t)
        if p is None:
            parse_fail += 1
        else:
            picks[p] += 1
    print(f"  Generated in {wall:.0f}s")
    n = len(prompts)
    print(f"  parse_fail: {parse_fail} ({parse_fail/n:.1%})")
    print(f"  picks (count, %):")
    for ord_ in range(len(POOL)):
        c = picks.get(ord_, 0)
        print(f"    ord_{ord_} {POOL[ord_].name:18s} {c:>4d}  {100*c/n:>5.1f}%")
    return {
        "variant": variant,
        "n": n,
        "parse_fail": parse_fail,
        "pick_counts": dict(picks),
        "pick_pct": {ord_: round(100 * picks.get(ord_, 0) / n, 1) for ord_ in range(len(POOL))},
        "wall_s": round(wall, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/opt/dlami/nvme/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default="/opt/dlami/nvme/cost-aware-routing/data/train.jsonl")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", default="/opt/dlami/nvme/cost-aware-routing/runs/prompt_variants.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train_data = []
    with open(args.data) as f:
        for line in f:
            train_data.append(json.loads(line))
    questions = [rng.choice(train_data) for _ in range(args.n)]

    print(f"Loading {args.checkpoint}...")
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map={"": 0},
    )
    model.eval()

    results = {}
    for variant in PROMPT_BUILDERS:
        results[variant] = run_variant(variant, model, tok, questions, args.gen_batch)

    # Headline comparison
    print("\n" + "=" * 80)
    print(f"{'variant':22s} {'parse_fail':>11s}  " + "  ".join(f"ord_{i}" for i in range(9)))
    for v, r in results.items():
        pcts = "  ".join(f"{r['pick_pct'][i]:>5.1f}" for i in range(9))
        print(f"  {v:20s} {r['parse_fail']:>4d} ({r['parse_fail']/r['n']:>4.0%})  {pcts}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
