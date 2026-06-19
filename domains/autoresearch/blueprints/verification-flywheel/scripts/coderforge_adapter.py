"""
CoderForge-Preview → TraceInput adapter.

CoderForge trajectories use OpenHands-style chat messages with tool_calls:
  - assistant messages with tool_calls: [{function: {name, arguments}}]
  - tool response messages with tool_call_id
  - Tools: execute_bash, str_replace_editor, think, finish

This adapter extracts behavioral features from the chat message format
(not the OpenHands event stream format used by the openhands adapter).
"""

from __future__ import annotations

import json
from typing import Optional

from learned_verifier.schemas import TraceInput
from learned_verifier.telemetry import compute_derived_features

# CoderForge tool → category mapping (matches OpenHands categories)
TOOL_CATEGORIES = {
    "str_replace_editor": "edit",
    "execute_bash": "bash",
    "think": "other",
    "finish": "other",
}


def from_coderforge_messages(
    messages: list[dict],
    instance_id: str = "",
    reward: Optional[float] = None,
) -> TraceInput:
    """Build TraceInput from CoderForge chat messages.

    Args:
        messages: List of {role, content, tool_calls?, tool_call_id?} dicts.
        instance_id: Task identifier (trajectory_id).
        reward: Docker-verified gold label (0.0 or 1.0).
    """
    edit_count = 0
    search_count = 0
    bash_count = 0
    total_actions = 0
    loop_count = 0

    prev_tool = None
    repeat_streak = 0
    first_edit_action = None

    total_input_tokens = 0
    total_output_tokens = 0

    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant":
            # Estimate output tokens from content length
            content = msg.get("content") or ""
            total_output_tokens += len(content) // 4  # rough 4 chars/token

            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                category = TOOL_CATEGORIES.get(name, "other")
                total_actions += 1

                if category == "edit":
                    edit_count += 1
                    if first_edit_action is None:
                        first_edit_action = total_actions - 1
                elif category == "bash":
                    bash_count += 1
                elif category == "search":
                    search_count += 1

                # Loop detection
                if name == prev_tool:
                    repeat_streak += 1
                    if repeat_streak >= 2:
                        loop_count += 1
                else:
                    repeat_streak = 0
                prev_tool = name

                # Estimate tokens from arguments
                args = func.get("arguments", "")
                if isinstance(args, str):
                    total_output_tokens += len(args) // 4

        elif role == "tool":
            # Tool response → estimate input tokens
            content = msg.get("content") or ""
            total_input_tokens += len(content) // 4

        elif role == "user":
            content = msg.get("content") or ""
            total_input_tokens += len(content) // 4

    first_edit_pct = None
    if first_edit_action is not None and total_actions > 0:
        first_edit_pct = first_edit_action / total_actions

    # Estimate cost: CoderForge used Qwen3-Coder-480B at ~$0.001/1K tokens
    total_tokens = total_input_tokens + total_output_tokens
    estimated_cost = total_tokens * 0.001 / 1000

    return compute_derived_features(TraceInput(
        instance_id=instance_id,
        total_cost_usd=estimated_cost,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_actions=total_actions,
        edit_count=edit_count,
        search_count=search_count,
        bash_count=bash_count,
        loop_count=loop_count,
        total_turns=total_actions,
        first_edit_pct=first_edit_pct,
        source_format="coderforge",
    ))


def from_coderforge_row(row: dict) -> tuple[TraceInput, float, str]:
    """Convert a single HuggingFace dataset row to (TraceInput, gold_label, diff).

    Returns:
        (trace, reward, patch_diff) where reward is 0.0 or 1.0.
        patch_diff is extracted from the last str_replace_editor calls if available.
    """
    messages = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    reward = float(row.get("reward", 0.0))
    instance_id = row.get("trajectory_id", "")

    trace = from_coderforge_messages(messages, instance_id=instance_id, reward=reward)

    # Extract problem statement from first user message
    problem = ""
    for msg in messages:
        if msg.get("role") == "user":
            problem = msg.get("content", "")[:8000]  # cap at 8K chars
            break

    # Extract patch diff from str_replace_editor calls
    diff_parts = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                if func.get("name") == "str_replace_editor":
                    args = func.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    if isinstance(args, dict):
                        cmd = args.get("command", "")
                        path = args.get("path", "")
                        old = args.get("old_str", "")
                        new = args.get("new_str", "")
                        if cmd == "str_replace" and old and new:
                            diff_parts.append(
                                f"--- a{path}\n+++ b{path}\n"
                                f"-{old}\n+{new}"
                            )
                        elif cmd == "create":
                            file_text = str(args.get("file_text", ""))
                            diff_parts.append(
                                f"--- /dev/null\n+++ b{path}\n"
                                f"+{file_text[:2000]}"
                            )

    patch_diff = "\n".join(diff_parts) if diff_parts else ""

    trace.problem_statement = problem
    trace.patch_diff = patch_diff

    return trace, reward, patch_diff
