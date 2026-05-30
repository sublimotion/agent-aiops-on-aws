"""
Balanced 9-shot examples for the router prompt.

Goal: at iter 0, before any GRPO updates, the base Qwen2.5-7B-Instruct
router should pick each worker between 5-20% of the time on a held-out
mix of MATH/MMLU/HumanEval questions. The brand-bias mitigation
(plan-addendum §4) is to prepend these examples so the model sees:

  - one example per worker
  - each pick is justified by the worker's qualitative strength
  - cheap workers handle "easy" questions, expensive workers handle "hard" ones
  - cost is mentioned in the justification when relevant

Examples are intentionally diverse across MATH/MMLU/code/factual to exercise
each worker's strength. Designed to be `apply_chat_template`-friendly: each
example renders as one (user, assistant) turn pair before the real question.
"""
from __future__ import annotations

# (question, picked_ord, one-sentence justification)
EXAMPLES: list[tuple[str, int, str]] = [
    (
        "What is 2 + 2?",
        0,  # gemma-3-27b-it
        "Trivial arithmetic — pick the cheapest worker (ord_0, Gemma).",
    ),
    (
        "Write a single-line Python lambda that returns the absolute value of x.",
        1,  # gpt-oss-120b
        "Tiny code lookup with no reasoning — gpt-oss-120b is cheapest for short code.",
    ),
    (
        "List the 3 main events of the French Revolution in chronological order. Reply with a JSON array of strings.",
        2,  # qwen3-32b
        "Short structured output with light factual recall — Qwen3-32B handles structured outputs well at low cost.",
    ),
    (
        "Implement a Python function `def levenshtein(a: str, b: str) -> int` that returns the edit distance between two strings using bottom-up dynamic programming.",
        3,  # qwen3-coder-480b
        "Code generation requiring DP correctness — Qwen3-Coder-480B is the code specialist.",
    ),
    (
        "Translate the following English paragraph into formal French, preserving the meaning and tone: \"Despite the difficulties, the council reached a unanimous decision after three hours of deliberation.\"",
        4,  # mistral-large-3
        "Multilingual translation with register preservation — Mistral Large 3 has the strongest multilingual coverage.",
    ),
    (
        "Solve: A train leaves city A at 60 km/h. A second train leaves city B (300 km away) at 40 km/h heading toward A two hours later. When do they meet?",
        5,  # deepseek-v3.2
        "Multi-step word problem with arithmetic reasoning — DeepSeek V3.2 is a strong reasoning mid-tier.",
    ),
    (
        "What is the capital of Australia, and which year did it become the capital?",
        6,  # haiku-4-5
        "Direct factual lookup — Haiku 4.5 answers reliably and fast.",
    ),
    (
        "Prove that there are infinitely many primes. Give a complete proof, not just a sketch.",
        7,  # sonnet-4-6
        "Formal proof requiring careful exposition — Sonnet 4.6 handles hard reasoning at moderate cost.",
    ),
    (
        "Find all positive integer solutions (a, b, c) to a^3 + b^3 = c^3 + 1 with a, b, c <= 10. Show your reasoning and verify each solution.",
        8,  # opus-4-7
        "Frontier-hard combinatorial reasoning with case analysis — Opus 4.7 is worth the cost on this regime.",
    ),
]


def render_few_shot_block() -> str:
    """Render the 9-shot block as plain text (router prompt header).

    Format mirrors how the real router prompt looks: each example shows the
    question, the PICK line, and a justification. Designed to be appended
    after the metadata prompt header from worker_pool.build_metadata_prompt().
    """
    parts = ["", "Examples (one per worker, balanced across capabilities):", ""]
    for i, (q, ord_, why) in enumerate(EXAMPLES, 1):
        parts.append(f"Example {i}:")
        parts.append(f"  Question: {q}")
        parts.append(f"  PICK ord_{ord_}")
        parts.append(f"  Justification: {why}")
        parts.append("")
    parts.append("Now answer the actual question:")
    parts.append("")
    return "\n".join(parts)


def render_chat_messages() -> list[dict]:
    """Render the 9-shot block as a list of {role, content} pairs.

    Use this when feeding into apply_chat_template — each pair becomes a
    (user, assistant) turn. The actual question is appended as the final
    user turn by the caller.
    """
    msgs = []
    for q, ord_, why in EXAMPLES:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": f"PICK ord_{ord_}\n{why}"})
    return msgs


if __name__ == "__main__":
    print(render_few_shot_block())
    print()
    print(f"Total examples: {len(EXAMPLES)}")
    counts = {}
    for _, ord_, _ in EXAMPLES:
        counts[ord_] = counts.get(ord_, 0) + 1
    print(f"Picks per ord: {counts}")
    assert all(counts.get(i, 0) == 1 for i in range(9)), "Not balanced!"
    print("OK: exactly one example per worker.")
