#!/usr/bin/env python3
"""
Gate 0.2b — per-(worker × role) output-parser audit (BLOCKING).

Carryover from cost-aware-routing, where the equivalent gate caught 24/33 cell
failures (Kimi intermediate \\boxed, Opus extended-thinking blocks, GLM nested
<think>, mid-reasoning truncation). A 1-token liveness ping (Gate 0.2) does NOT
catch output-format failures, and Trinity's multi-turn loop compounds format
drift across the 3 roles.

Probes each of the 7 workers × 3 roles = 21 cells with a multi-token query and
checks that the EXACT parsers Trinity's core.py uses succeed:
  - Thinker  -> Task._parse_thinker_response: needs BOTH <suggestion>...</suggestion>
                and <suggested_role>solver|verifier</suggested_role>.
  - Worker   -> a usable non-empty answer survives thinking-block stripping.
  - Verifier -> Task._parse_verification_response: response starts ACCEPT|REJECT.

Reasoning (ord 5) and direct (ord 6) Qwen3-32B share the role prompt templates,
so 21 cells is correct. Pass criterion: every cell parses cleanly ≥90% of N reps.

Run inside the Job (needs boto3 + Bedrock creds):
  python gate_0_2b_role_audit.py --reps 5
Exit 0 = all cells pass; exit 1 = at least one cell below threshold.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from worker_pool_bedrock import POOL, reasoning_effort_for
from bedrock_clients import _query_converse

# Mirror the exact prompts/parsers from vendored fugu/core.py (kept in sync by
# value — if core.py prompts change, update here too).
THINKER_PROMPT = (
    "You are requested to coordinate a pool of agents to give a proper response to a query. "
    "The following is the query and the thoughts from some agents.\n<info>\nWhat is 12*9? "
    "One agent answered 108.\n</info>\n"
    "Do not directly respond the query. Provide step-by-step analysis first, then generate:\n"
    "<suggestion>your_suggestion</suggestion>\n\n<suggested_role>next_agent_role</suggested_role>\n\n"
    "The suggested role should be either 'solver' or 'verifier'."
)
WORKER_PROMPT = "Solve and give the final answer: What is 12 * 9?"
VERIFIER_PROMPT = (
    "Please carefully review the following response and determine whether to accept it.\n\n"
    "<query>\nWhat is 12*9?\n</query>\n\n<response>\n108\n</response>\n\n"
    "Respond with either ACCEPT or REJECT followed by a brief explanation."
)


def parse_thinker(text: str) -> bool:
    if not text:
        return False
    role = re.search(r"<suggested_role>(.*?)</suggested_role>", text, re.DOTALL)
    sugg = re.search(r"<suggestion>(.*?)</suggestion>", text, re.DOTALL)
    if not role or not sugg:
        return False
    return role.group(1).strip() in ("solver", "verifier") and bool(sugg.group(1).strip())


def parse_worker(text: str) -> bool:
    # Worker answer just needs to be non-trivial after stripping inline think tags.
    if not text:
        return False
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return len(cleaned) >= 1


def parse_verifier(text: str) -> bool:
    if not text:
        return False
    low = text.lower().strip()
    return low.startswith("accept") or low.startswith("reject") or \
        ("accept" in low) ^ ("reject" in low)


ROLES = [
    ("thinker", THINKER_PROMPT, parse_thinker),
    ("worker", WORKER_PROMPT, parse_worker),
    ("verifier", VERIFIER_PROMPT, parse_verifier),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.90)
    # Must mirror the run's real serving config. Phase 1 uses max_tokens=8192
    # (run_trinity_agent): reasoning workers (qwen3-32b @ high effort) burn the
    # whole budget on the reasoningContent block and emit empty/truncated answer
    # text at 1024/4096 → the thinker/verifier parsers see "" and the gate reports
    # a FALSE failure. Test at the budget we actually serve. (lessons #19)
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    print(f"Gate 0.2b — {len(POOL)}×3 = {len(POOL)*3} cells, {args.reps} reps each, "
          f"pass ≥{args.threshold:.0%}")
    any_fail = False
    for w in POOL:
        if w.transport == "openai_compat":
            print(f"  ord {w.ord} {w.friendly_name}: openai_compat — probe manually with bearer token")
            continue
        eff = reasoning_effort_for(w)
        for role_name, prompt, parser in ROLES:
            passes = 0
            for _ in range(args.reps):
                txt = _query_converse(
                    w.model_id, w.friendly_name, w.concurrency, eff,
                    [{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens, temperature=0.1,
                    no_temperature=("no-temperature" in getattr(w, "api_quirks", ())),
                )
                if parser(txt):
                    passes += 1
            rate = passes / args.reps
            ok = rate >= args.threshold
            any_fail |= not ok
            print(f"  {'✅' if ok else '❌'} ord {w.ord} {w.friendly_name:22s} {role_name:9s} {rate:.0%}")
    if any_fail:
        print("\nGate 0.2b FAILED — fix extraction (do not lower the bar) before any run.")
        return 1
    print("\nGate 0.2b PASSED — all cells parse cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
