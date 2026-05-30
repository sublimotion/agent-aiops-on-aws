"""Compare base Qwen2.5-7B vs trained checkpoint on routing decisions.

Generates 32 rollouts on each of 6 fixed questions (2 math, 2 aime, 2
wildchat) from both the base model and a trained checkpoint, then:

  1. Categorizes parse outcomes:
       - parse_ok: PICK ord_N parsed cleanly
       - empty: response < 5 chars
       - wrong_format: response present but no PICK ord_N
       - other: anything else
  2. Per-source worker pick distribution (base vs trained).
  3. Sample of parse failures (text snippets) so we can see what the
     model is actually emitting when it doesn't follow the format.

Output: results/runs/introspect_<base|v1>.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, "/opt/dlami/nvme/cost-aware-routing/scripts")
from train_cost_aware_router import (
    build_router_prompt,
    parse_pick,
    PICK_RE,
    batch_generate,
)


def categorize(gen_text: str) -> tuple[str, int | None]:
    """Returns (category, picked_ord)."""
    stripped = gen_text.strip()
    if len(stripped) < 5:
        return "empty", None
    picked = parse_pick(gen_text)
    if picked is not None:
        return "parse_ok", picked
    if "PICK" in gen_text.upper():
        return "wrong_pick_format", None
    return "no_pick_word", None


def run(ckpt_path: str, questions: list[dict], rollouts_per_q: int, gen_batch: int):
    print(f"Loading {ckpt_path}...")
    tok = AutoTokenizer.from_pretrained(ckpt_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()

    prompts = []
    for q in questions:
        p = build_router_prompt(q["question"])
        prompts.extend([p] * rollouts_per_q)
    print(f"Generating {len(prompts)} rollouts...")
    texts = batch_generate(
        model, tok, prompts,
        batch_size=gen_batch, max_new_tokens=96, temperature=0.7,
    )

    results = []
    for i, t in enumerate(texts):
        q_idx = i // rollouts_per_q
        cat, ord_ = categorize(t)
        results.append({
            "q_idx": q_idx,
            "source": questions[q_idx]["source"],
            "category": cat,
            "picked_ord": ord_,
            "gen_tail": t[:300],
        })

    # Aggregate
    by_q: dict[int, list] = {}
    for r in results:
        by_q.setdefault(r["q_idx"], []).append(r)

    summary_per_q = []
    for q_idx, qresults in by_q.items():
        cats = collections.Counter(r["category"] for r in qresults)
        picks = collections.Counter(r["picked_ord"] for r in qresults if r["picked_ord"] is not None)
        # Sample a parse failure (if any)
        sample_fail = next((r for r in qresults if r["category"] != "parse_ok"), None)
        summary_per_q.append({
            "q_idx": q_idx,
            "source": questions[q_idx]["source"],
            "question": questions[q_idx]["question"][:150],
            "n_rollouts": len(qresults),
            "categories": dict(cats),
            "parse_ok_rate": round(cats["parse_ok"] / max(len(qresults), 1), 3),
            "pick_distribution": dict(picks),
            "sample_failure": (
                {"category": sample_fail["category"], "gen_tail": sample_fail["gen_tail"]}
                if sample_fail else None
            ),
        })

    # Cleanup model
    del model
    torch.cuda.empty_cache()

    return {
        "checkpoint": ckpt_path,
        "n_questions": len(questions),
        "rollouts_per_q": rollouts_per_q,
        "summary_per_q": summary_per_q,
        "all_results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    ap.add_argument("--data", default="/opt/dlami/nvme/cost-aware-routing/data/train.jsonl")
    ap.add_argument("--n-per-source", type=int, default=2)
    ap.add_argument("--rollouts-per-q", type=int, default=32)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Pick fixed questions: first n_per_source per source.
    by_src: dict[str, list[dict]] = {}
    with open(args.data) as f:
        for line in f:
            r = json.loads(line)
            by_src.setdefault(r["source"], []).append(r)
    questions = []
    for src in ("math500", "aime25", "wildchat"):
        questions.extend(by_src[src][: args.n_per_source])
    print(f"Selected {len(questions)} questions:")
    for q in questions:
        print(f"  [{q['source']}] {q['question'][:100]!r}")

    result = run(args.checkpoint, questions, args.rollouts_per_q, args.gen_batch)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print()
    print("=== INTROSPECTION ===")
    for s in result["summary_per_q"]:
        print(f"\nq_idx={s['q_idx']} [{s['source']}]: {s['question']!r}")
        print(f"  parse_ok_rate: {s['parse_ok_rate']:.0%}")
        print(f"  categories: {s['categories']}")
        print(f"  picks: {s['pick_distribution']}")
        if s["sample_failure"]:
            print(f"  sample failure ({s['sample_failure']['category']}):")
            print(f"    {s['sample_failure']['gen_tail'][:200]!r}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
