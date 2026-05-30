"""Format-only few-shot: 9 examples on the SAME question, one per worker.

The original few_shot.EXAMPLES paired each worker with a question type
that fit its strength (math→DeepSeek, code→Qwen-Coder, etc.). We
discovered (results/runs/prompt_variants.json) this acts as a TYPE
PRESCRIPTION — the model learns "math→DeepSeek" from the examples.

This format-only variant uses ONE generic question and rotates the
picked worker through all 9 ords with a generic justification. Goal:
the model learns the OUTPUT FORMAT but not which worker to pick.
"""
from __future__ import annotations

# Single neutral question — any of these would work; chose something
# innocuous that isn't strongly math-coded or code-coded.
GENERIC_Q = "What's the weather like in Paris in March?"

# Picks rotate 0..8; each justification is generic ("appropriate worker
# for this question") to avoid teaching capability-grounded routing.
EXAMPLES_FORMAT_ONLY = [
    (GENERIC_Q, ord_, f"This worker is suitable for the question.")
    for ord_ in range(9)
]


def render_format_only_block() -> str:
    parts = ["", "Format examples (output exactly this format):", ""]
    for i, (q, ord_, why) in enumerate(EXAMPLES_FORMAT_ONLY, 1):
        parts.append(f"Example {i}:")
        parts.append(f"  Question: {q}")
        parts.append(f"  PICK ord_{ord_}")
        parts.append(f"  Justification: {why}")
        parts.append("")
    parts.append("Now answer the actual question:")
    parts.append("")
    return "\n".join(parts)


if __name__ == "__main__":
    print(render_format_only_block())
