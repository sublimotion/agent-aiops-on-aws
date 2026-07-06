"""Router prompt construction with neutral codes.

Per spec §"Worker anonymization": real model names NEVER appear in the router
prompt. Workers are exposed as `worker_alpha` ... `worker_kappa` aliases plus
their capability_summary string from configs/pool.yaml.

The router emits `Answer: ord_<N>` where N is the position-shuffled ord. The
caller (trainer) maps ord → code → real bedrock_id via WorkerPool.

CRITICAL: this file MUST be the only place the router prompt is constructed.
Any drift between this and the prompt fed to the GRPO loss will silently
re-introduce the chat-template bug we just engineered around. See Gate 0.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Greek letters indexed 0..10. Add more if pool grows past 11 workers.
NEUTRAL_CODES = (
    "worker_alpha", "worker_beta", "worker_gamma", "worker_delta",
    "worker_epsilon", "worker_zeta", "worker_eta", "worker_theta",
    "worker_iota", "worker_kappa", "worker_lambda",
)


@dataclass(frozen=True)
class WorkerCard:
    """One row in the worker description block. Provider/family names redacted."""
    ord_: int
    code: str
    capability_summary: str    # e.g. "~$0.003/q, fast generalist, 200K context"


def build_worker_block(cards: Iterable[WorkerCard]) -> str:
    """One line per worker, deterministic order by ord."""
    sorted_cards = sorted(cards, key=lambda c: c.ord_)
    lines = []
    for c in sorted_cards:
        lines.append(f"ord_{c.ord_} ({c.code}): {c.capability_summary}")
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """\
You are a routing controller. For each user question you select exactly one worker from the pool below to answer it. Your goal is to maximize answer quality while minimizing cost: pick the cheapest worker that is capable of answering correctly.

Worker pool:
{worker_block}

Output format (STRICT — any deviation gets reward 0):
  Answer: ord_<N>

where <N> is the integer ord of the worker you select. You may include a brief reasoning prefix wrapped in <thinking>...</thinking>; everything outside the <thinking> block must be exactly the `Answer: ord_<N>` line.

Examples of valid output:
  Answer: ord_3
  <thinking>This is a basic factual question; the cheapest worker can handle it.</thinking>
  Answer: ord_0

Do NOT emit the worker's actual answer; only choose which worker should answer."""

USER_TEMPLATE = "Question: {question}"


def build_router_messages(question: str, cards: Iterable[WorkerCard]) -> list[dict]:
    """Construct ChatML messages for the router. Pass through
    tokenizer.apply_chat_template(add_generation_prompt=True) to render.

    Returns list of {role, content} dicts ready for apply_chat_template.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(
            worker_block=build_worker_block(cards))},
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]


def render_for_generation(messages: list[dict], tokenizer) -> str:
    """Apply chat template with add_generation_prompt=True. NEVER call
    tokenizer(raw_text) directly — that bypasses the chat template and
    triggers the rl-conductor "ordinal bias" bug."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # Gate 0.1 invariant: rendered prompt must contain ChatML markers.
    if "<|im_start|>" not in text:
        raise RuntimeError(
            "Router prompt missing ChatML markers — chat template not applied. "
            "Verify tokenizer is loaded from a Qwen3 (or Qwen2.5) Instruct model."
        )
    return text


def cards_from_pool(pool) -> list[WorkerCard]:
    """Convert a WorkerPool's WorkerConfigs into anonymized WorkerCards.

    The pool's `seed` argument already shuffled the code→bedrock mapping;
    here we just attach neutral codes by ord position. The mapping is
    deterministic given the same seed.
    """
    if pool.num_workers > len(NEUTRAL_CODES):
        raise ValueError(f"Pool has {pool.num_workers} workers but only {len(NEUTRAL_CODES)} codes defined")
    cards = []
    for ord_, w in sorted(pool.workers.items()):
        cards.append(WorkerCard(
            ord_=ord_,
            code=NEUTRAL_CODES[ord_],
            capability_summary=w.capability_summary,
        ))
    return cards
