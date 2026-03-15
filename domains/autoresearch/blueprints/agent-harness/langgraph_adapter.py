#!/usr/bin/env python3
"""
LangGraph-style ReAct adapter for SWE-bench evaluation.

Uses LangChain's ChatOpenAI with structured tool calling in a ReAct loop.
Includes context compaction to stay within Mistral's token limit.

Same 5 tools as SERA, but uses LangChain's native tool calling protocol
instead of SERA's manual JSON parsing.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

WORKSPACE = Path(".")

MAX_FILE_CHARS = 4000
MAX_CMD_CHARS = 4000


def run_cmd(cmd, cwd, timeout=30):
    try:
        if not os.path.isdir(cwd):
            return f"Error: working directory does not exist: {cwd}", -1
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + result.stderr
        if len(output) > MAX_CMD_CHARS:
            output = output[:MAX_CMD_CHARS // 2] + "\n...(truncated)...\n" + output[-MAX_CMD_CHARS // 2:]
        return output, result.returncode
    except subprocess.TimeoutExpired:
        return "(command timed out after 30s)", -1
    except Exception as e:
        return f"Error: {e}", -1


@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path relative to repo root."""
    fpath = WORKSPACE / path
    if not fpath.exists():
        return f"Error: File not found: {path}"
    try:
        content = fpath.read_text()
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + f"\n\n... (truncated at {MAX_FILE_CHARS} chars)"
        lines = content.split("\n")
        return "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def edit_file(path: str, old_str: str, new_str: str) -> str:
    """Replace old_str with new_str in the file. old_str must match exactly."""
    fpath = WORKSPACE / path
    if not fpath.exists():
        return f"Error: File not found: {path}"
    content = fpath.read_text()
    if old_str not in content:
        return f"Error: old_str not found in {path}. Make sure it matches exactly."
    if content.count(old_str) > 1:
        return f"Error: old_str appears {content.count(old_str)} times. Provide more context."
    fpath.write_text(content.replace(old_str, new_str, 1))
    return f"Successfully edited {path}"


@tool
def run_command(command: str) -> str:
    """Run a shell command in the repo directory. Timeout: 30s."""
    dangerous = ["rm -rf", "sudo", "curl", "wget", "pip install", "git clean", "git reset --hard"]
    if any(d in command for d in dangerous):
        return f"Error: Command blocked for safety: {command}"
    output, rc = run_cmd(command, str(WORKSPACE))
    return f"Exit code: {rc}\n{output}"


@tool
def search_files(pattern: str, path: str = ".") -> str:
    """Search for files matching a glob pattern."""
    import glob as globmod
    search_path = WORKSPACE / path
    matches = sorted(globmod.glob(str(search_path / pattern), recursive=True))
    matches = [str(Path(m).relative_to(WORKSPACE)) for m in matches[:50]]
    return "\n".join(matches) if matches else "No files found matching pattern."


@tool
def search_code(query: str, path: str = ".") -> str:
    """Search file contents for a string."""
    output, _ = run_cmd(
        f"grep -rn --include='*.py' '{query}' {path} | head -30", str(WORKSPACE)
    )
    return output if output.strip() else "No matches found."


ALL_TOOLS = [read_file, edit_file, run_command, search_files, search_code]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}

SYSTEM_PROMPT = """You are an expert software engineer tasked with fixing a bug in a Python repository.

You have access to tools: read_file, edit_file, run_command, search_files, search_code.

You have {max_turns} turns total. You MUST attempt a fix using edit_file before your turns run out.

Strategy:
1. Quickly locate the relevant code (2-3 searches + 1-2 file reads max)
2. Make your fix using edit_file with exact string matching
3. Run failing tests to verify: run_command("python -m pytest <test_path> -x --tb=short")
4. Iterate based on test results

CRITICAL: Do NOT spend too many turns just reading. Make your best edit attempt early.
Never install packages, modify test files, or repeat the same action."""


def compact_messages(messages):
    """Compact old tool results to stay within token limits.

    Preserves Mistral-compatible ordering:
    [system, user, (ai+tool_calls, tool_results)*, ai_final]
    """
    if len(messages) <= 6:
        return messages

    # Keep system + user (first 2), compact middle, keep last 6
    compacted = list(messages[:2])
    middle = messages[2:]
    keep_recent = 6  # Keep last 6 messages uncompacted

    cutoff = max(0, len(middle) - keep_recent)
    for i, msg in enumerate(middle):
        if i < cutoff and isinstance(msg, ToolMessage):
            content = msg.content or ""
            if len(content) > 150:
                content = content[:75] + "\n...(compacted)...\n" + content[-75:]
            compacted.append(ToolMessage(content=content, tool_call_id=msg.tool_call_id))
        else:
            compacted.append(msg)

    return compacted


def run_langgraph_agent(instance_id, problem_statement, workspace_path, endpoint, max_turns=30, model_name="devstral-small-2"):
    """Run a LangGraph-style ReAct agent with context compaction."""
    global WORKSPACE
    WORKSPACE = Path(workspace_path)

    llm = ChatOpenAI(
        model=model_name,
        base_url=endpoint,
        api_key="not-needed",
        temperature=0.0,
        max_tokens=4096,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    explore_budget = max(2, max_turns // 5)
    system_msg = SystemMessage(content=SYSTEM_PROMPT.format(max_turns=max_turns))
    user_msg = HumanMessage(content=f"""Fix this issue:

{problem_statement}

Repository is checked out at the correct commit. Start by locating the relevant code, make your fix, and verify with tests.""")

    messages = [system_msg, user_msg]

    turn = 0
    edit_count = 0
    first_edit_turn = None
    has_edited = False

    while turn < max_turns:
        turn += 1

        # Context compaction check
        est_tokens = sum(len(str(getattr(m, 'content', ''))) // 3 for m in messages)
        if est_tokens > 18000:
            messages = compact_messages(messages)

        # Turn pressure injection (modify last tool message if exists)
        if turn >= explore_budget and not has_edited and messages:
            last = messages[-1]
            if isinstance(last, ToolMessage):
                turns_left = max_turns - turn
                if turns_left <= 3:
                    messages[-1] = ToolMessage(
                        content=last.content + f"\n\n[URGENT: Only {turns_left} turns left! Use edit_file RIGHT NOW.]",
                        tool_call_id=last.tool_call_id,
                    )
                elif turn % 2 == 0:
                    messages[-1] = ToolMessage(
                        content=last.content + f"\n\n[SYSTEM: Turn {turn}/{max_turns}. You MUST use edit_file now.]",
                        tool_call_id=last.tool_call_id,
                    )

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            err = str(e)
            if "400" in err or "502" in err:
                # Context too large — try aggressive compaction
                messages = compact_messages(messages)
                try:
                    response = llm_with_tools.invoke(messages)
                except Exception as e2:
                    print(f"  Agent error after compaction: {str(e2)[:200]}", file=sys.stderr)
                    break
            else:
                print(f"  Agent error: {err[:200]}", file=sys.stderr)
                break

        messages.append(response)

        if not response.tool_calls:
            # Model wants to stop — check if it's just thinking or done
            if turn >= max_turns - 2:
                break
            continue

        # Execute tool calls
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            if tool_name in TOOL_MAP:
                try:
                    result = TOOL_MAP[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tool_name}"

            if tool_name == "edit_file" and "Successfully" in str(result):
                edit_count += 1
                has_edited = True
                if first_edit_turn is None:
                    first_edit_turn = turn

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # Check for diff
    diff_output, _ = run_cmd("git diff", str(WORKSPACE))
    fix_generated = bool(diff_output.strip())

    return {
        "instance_id": instance_id,
        "harness": "langgraph",
        "fix_generated": fix_generated,
        "turns_used": turn,
        "edit_count": edit_count,
        "first_edit_turn": first_edit_turn,
        "diff_size": len(diff_output.strip()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--model", default="devstral-small-2")
    parser.add_argument("--max-turns", type=int, default=30)
    args = parser.parse_args()

    result = run_langgraph_agent(
        args.instance_id, args.problem, args.workspace,
        args.endpoint, args.max_turns, args.model
    )
    print(json.dumps(result))
