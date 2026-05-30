"""Workflow runtime for RL Conductor.

Parses Conductor output into a multi-step workflow graph,
executes each step by calling workers via the pool,
and returns the final aggregated answer.

Both sync and async versions available.
"""

import re
from typing import Optional

from reward import parse_conductor_output
from worker_proxy import WorkerPool


def execute_workflow_sync(
    conductor_output: str,
    worker_pool: WorkerPool,
    max_depth: int = 5,
) -> str:
    """Execute a multi-step workflow synchronously."""
    steps = split_into_steps(conductor_output)

    if not steps:
        parsed = parse_conductor_output(conductor_output)
        if parsed:
            steps = [parsed]
        else:
            return extract_final_answer(conductor_output)

    all_outputs: list[str] = []

    for i, step in enumerate(steps[:max_depth]):
        for j, (subtask, mid, access) in enumerate(
            zip(step["subtasks"], step["model_id"], step["access_list"])
        ):
            if isinstance(access, int):
                access = [access]
            if not isinstance(access, list):
                access = []

            context_parts = []
            for idx in access:
                if isinstance(idx, int) and 0 <= idx < len(all_outputs):
                    context_parts.append(f"[Previous output {idx}]: {all_outputs[idx]}")

            prompt = subtask if isinstance(subtask, str) else str(subtask)
            if context_parts:
                prompt = "\n".join(context_parts) + f"\n\nTask: {prompt}"

            mid_clamped = max(0, min(int(mid) if isinstance(mid, (int, float)) else 0, worker_pool.num_workers - 1))
            result = worker_pool.call_worker_sync(mid_clamped, prompt)
            all_outputs.append(result)

    final = extract_final_answer(conductor_output)
    if not final and all_outputs:
        final = all_outputs[-1]

    return final


async def execute_workflow(
    conductor_output: str,
    worker_pool: WorkerPool,
    max_depth: int = 5,
) -> str:
    """Async version (delegates to sync)."""
    return execute_workflow_sync(conductor_output, worker_pool, max_depth)


def split_into_steps(text: str) -> list[dict]:
    """Split conductor output into multiple workflow steps."""
    step_pattern = r'(?:Step\s+\d+|STEP\s+\d+|step\s+\d+)[:\s]'
    parts = re.split(step_pattern, text)

    steps = []
    for part in parts:
        if not part.strip():
            continue
        parsed = parse_conductor_output(part)
        if parsed:
            steps.append(parsed)

    return steps


def extract_final_answer(text: str) -> str:
    """Extract the final answer from conductor output."""
    match = re.search(r'FINAL\s+ANSWER\s*:\s*(.*?)(?:\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    boxed = re.search(r'\\boxed\{(.*?)\}', text)
    if boxed:
        return boxed.group(1)

    answer_match = re.search(r'(?:the answer is|answer:)\s*(.*?)(?:\n|\Z)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    return ""
