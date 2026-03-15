#!/usr/bin/env python3
"""
SWE-bench Lite Agent Evaluator for Phase 1: Turn Degradation Analysis.

Runs a coding agent loop against SWE-bench Lite issues with configurable
turn budgets and context management strategies.

Usage:
    python3 sera-eval.py --issues eval_subset.txt --max-turns 30 \
        --output phase1_D.jsonl --per-turn-log phase1_D_turns.jsonl

    python3 sera-eval.py --issues eval_subset.txt --max-turns 15 \
        --restart-with-summary --output phase1_E.jsonl

    python3 sera-eval.py --list-issues 50 --seed 42 > eval_subset.txt
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import openai
from datasets import load_dataset

ENDPOINT = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:9000/v1")
MODEL = os.environ.get("MODEL_NAME", "devstral-small-2")
WORKSPACE_ROOT = Path("/mnt/nvme/sera-workspaces")
REPO_CACHE = WORKSPACE_ROOT / "_repo_cache"

SYSTEM_PROMPT_TEMPLATE = """You are an expert software engineer tasked with fixing a bug in a Python repository.

You have access to the following tools:
- read_file(path): Read a file's contents
- edit_file(path, old_str, new_str): Replace old_str with new_str in the file
- run_command(command): Run a shell command and get its output (timeout: 30s)
- search_files(pattern, path): Search for files matching a glob pattern
- search_code(query, path): Search file contents for a string

You have {max_turns} turns total. You MUST attempt a fix using edit_file before your turns run out.

Strategy:
1. Turns 1-{explore_budget}: Quickly locate the relevant code (2-3 searches + 1-2 file reads max)
2. Turns {explore_budget}-{fix_budget}: Make your fix using edit_file, then run the failing tests
3. Remaining turns: Iterate based on test results

CRITICAL RULES:
- Do NOT spend more than {explore_budget} turns just reading/searching. Make your best edit attempt early.
- Use edit_file with exact string matching. Copy the exact code you want to change.
- After editing, run the failing tests to verify: run_command("python -m pytest <test_path> -x --tb=short")
- If a test fails after your fix, read the error and try a different approach.
- Never install packages, modify test files, or repeat the same action.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace old_str with new_str in the file. old_str must match exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root"},
                    "old_str": {"type": "string", "description": "Exact string to find and replace"},
                    "new_str": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the repo directory. Timeout: 30s.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Find files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py')"},
                    "path": {"type": "string", "description": "Directory to search in (default: repo root)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search file contents for a string (like grep -r)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "String to search for"},
                    "path": {"type": "string", "description": "Directory to search in (default: repo root)"},
                },
                "required": ["query"],
            },
        },
    },
]


def run_cmd(cmd, cwd, timeout=30):
    """Run a command and return (stdout+stderr, returncode)."""
    try:
        if not os.path.isdir(cwd):
            return f"Error: working directory does not exist: {cwd}", -1
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + result.stderr
        # Truncate long output
        if len(output) > 8000:
            output = output[:4000] + "\n\n... (truncated) ...\n\n" + output[-4000:]
        return output, result.returncode
    except subprocess.TimeoutExpired:
        return "(command timed out after 30s)", -1
    except Exception as e:
        return f"Error running command: {e}", -1


def setup_workspace(instance):
    """Clone repo at base_commit into a workspace directory."""
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]

    workspace = WORKSPACE_ROOT / instance_id.replace("/", "__")

    # Clean workspace if it exists
    if workspace.exists():
        subprocess.run(["rm", "-rf", str(workspace)], check=True)

    # Clone from GitHub (full history needed for arbitrary base_commits)
    repo_dir = repo.replace("/", "__")
    cache_path = REPO_CACHE / repo_dir

    if cache_path.exists():
        # Check if cache has the commit
        check = subprocess.run(
            ["git", "cat-file", "-t", base_commit],
            cwd=cache_path, capture_output=True
        )
        if check.returncode == 0:
            subprocess.run(["git", "clone", str(cache_path), str(workspace)],
                           capture_output=True, check=True)
        else:
            # Fetch from origin into cache, then clone
            subprocess.run(
                ["git", "fetch", "origin", "--depth=1", base_commit],
                cwd=cache_path, capture_output=True, timeout=120
            )
            # If shallow fetch worked, clone from cache
            check2 = subprocess.run(
                ["git", "cat-file", "-t", base_commit],
                cwd=cache_path, capture_output=True
            )
            if check2.returncode == 0:
                subprocess.run(["git", "clone", str(cache_path), str(workspace)],
                               capture_output=True, check=True)
            else:
                # Full clone from GitHub as fallback
                subprocess.run(
                    ["git", "clone", f"https://github.com/{repo}.git", str(workspace)],
                    capture_output=True, check=True, timeout=300
                )
    else:
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git", str(workspace)],
            capture_output=True, check=True, timeout=300
        )

    # Checkout base commit
    subprocess.run(["git", "checkout", base_commit], cwd=workspace,
                   capture_output=True, check=True)
    subprocess.run(["git", "clean", "-fdx"], cwd=workspace, capture_output=True)

    return workspace


def execute_tool(tool_name, args, workspace):
    """Execute a tool call and return the result string."""
    cwd = str(workspace)

    if tool_name == "read_file":
        fpath = workspace / args.get("path", "")
        if not fpath.exists():
            return f"Error: File not found: {args.get('path')}"
        try:
            content = fpath.read_text()
            if len(content) > 10000:
                content = content[:10000] + "\n\n... (file truncated, showing first 10000 chars)"
            # Add line numbers
            lines = content.split("\n")
            numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
            return numbered
        except Exception as e:
            return f"Error reading file: {e}"

    elif tool_name == "edit_file":
        fpath = workspace / args.get("path", "")
        if not fpath.exists():
            return f"Error: File not found: {args.get('path')}"
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")
        content = fpath.read_text()
        if old_str not in content:
            return f"Error: old_str not found in {args.get('path')}. Make sure it matches exactly."
        if content.count(old_str) > 1:
            return f"Error: old_str appears {content.count(old_str)} times. Provide more context to match uniquely."
        new_content = content.replace(old_str, new_str, 1)
        fpath.write_text(new_content)
        return f"Successfully edited {args.get('path')}"

    elif tool_name == "run_command":
        cmd = args.get("command", "")
        # Safety: block dangerous commands
        dangerous = ["rm -rf", "sudo", "curl", "wget", "pip install", "git clean", "git reset --hard"]
        if any(d in cmd for d in dangerous):
            return f"Error: Command blocked for safety: {cmd}"
        output, rc = run_cmd(cmd, cwd)
        return f"Exit code: {rc}\n{output}"

    elif tool_name == "search_files":
        import glob
        pattern = args.get("pattern", "")
        search_path = workspace / args.get("path", "")
        matches = sorted(glob.glob(str(search_path / pattern), recursive=True))
        # Make paths relative
        matches = [str(Path(m).relative_to(workspace)) for m in matches[:50]]
        return "\n".join(matches) if matches else "No files found matching pattern."

    elif tool_name == "search_code":
        query = args.get("query", "")
        search_path = args.get("path", ".")
        output, _ = run_cmd(
            f"grep -rn --include='*.py' '{query}' {search_path} | head -30", cwd
        )
        return output if output.strip() else "No matches found."

    return f"Unknown tool: {tool_name}"


def compact_context(messages, keep_system=True):
    """Compact conversation context by truncating old tool results.

    Preserves message ordering (system, user, [assistant+tool]*)
    which is required by Mistral chat template.
    """
    if len(messages) <= 4:
        return messages

    compacted = []
    # Keep system + first user message
    for msg in messages[:2]:
        compacted.append(msg)

    # For remaining messages, keep structure but truncate old tool results
    remaining = messages[2:]
    # Keep last 6 messages intact, truncate older tool results aggressively
    cutoff = max(0, len(remaining) - 6)

    for i, msg in enumerate(remaining):
        if i < cutoff and msg["role"] == "tool":
            content = msg.get("content", "")
            if len(content) > 200:
                content = content[:100] + "\n...(compacted)...\n" + content[-100:]
            compacted.append({**msg, "content": content})
        else:
            compacted.append(msg)

    return compacted


def make_restart_summary(messages):
    """Summarize a failed attempt for restart-with-summary strategy."""
    edits = []
    errors = []
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc["function"]["name"] == "edit_file":
                    args = json.loads(tc["function"]["arguments"])
                    edits.append(f"- Edited {args.get('path', '?')}")
        if msg["role"] == "tool":
            content = msg.get("content", "")
            if "Error" in content or "FAILED" in content:
                errors.append(content[:200])

    summary = "Previous attempt summary:\n"
    summary += f"- Made {len(edits)} edits:\n" + "\n".join(edits[:5]) + "\n"
    if errors:
        summary += f"- Encountered {len(errors)} errors:\n"
        for e in errors[:3]:
            summary += f"  {e}\n"
    summary += "\nTry a DIFFERENT approach this time."
    return summary


def run_agent_loop(client, instance, max_turns, compact_at=None,
                   restart_with_summary=False, per_turn_log=None):
    """Run the agent loop for a single issue. Returns (result_dict, turn_logs)."""
    instance_id = instance["instance_id"]
    problem = instance["problem_statement"]
    fail_to_pass = json.loads(instance["FAIL_TO_PASS"])

    print(f"  Setting up workspace for {instance_id}...")
    try:
        workspace = setup_workspace(instance)
    except Exception as e:
        return {"instance_id": instance_id, "error": f"workspace_setup_failed: {e}",
                "pass": False, "turns_used": 0}, []

    explore_budget = max(3, max_turns // 5)
    fix_budget = max(explore_budget + 2, max_turns // 2)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        max_turns=max_turns,
        explore_budget=explore_budget,
        fix_budget=fix_budget,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Fix the following issue:\n\n{problem}\n\n"
                                     f"The repository is checked out at the correct commit. "
                                     f"The failing tests are: {fail_to_pass}"}
    ]

    turn_logs = []
    prev_actions = set()
    total_turns = max_turns
    if restart_with_summary:
        total_turns = max_turns * 2  # Two rounds of max_turns

    restart_done = False
    effective_turn = 0

    for turn in range(total_turns):
        effective_turn += 1
        turn_start = time.time()

        # Restart-with-summary: reset at midpoint
        if restart_with_summary and turn == max_turns and not restart_done:
            summary = make_restart_summary(messages)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Fix the following issue:\n\n{problem}\n\n"
                                             f"The failing tests are: {fail_to_pass}\n\n{summary}"}
            ]
            restart_done = True
            effective_turn = 1  # Reset turn counter for second round

        # Context compaction
        if compact_at and effective_turn == compact_at:
            messages = compact_context(messages)

        # Auto-compact when context is getting large (avoid 400 errors)
        estimated_tokens = sum(
            len(str(m.get("content", ""))) // 3 for m in messages
        )
        if estimated_tokens > 20000:
            messages = compact_context(messages)

        # Inject turn pressure reminders (must come after assistant+tool pairs)
        turns_left = total_turns - turn
        has_edited = any(
            t.get("action_type") == "edit_file" for t in turn_logs
        )
        if effective_turn >= explore_budget and not has_edited:
            if messages and messages[-1]["role"] == "tool":
                if turns_left <= 3:
                    messages[-1]["content"] += (
                        f"\n\n[URGENT: Only {turns_left} turns left! Use edit_file RIGHT NOW "
                        f"with your best guess fix or you will fail this issue.]"
                    )
                elif effective_turn % 2 == 0:  # Every other turn
                    messages[-1]["content"] += (
                        f"\n\n[SYSTEM: Turn {effective_turn}/{max_turns}. "
                        f"You MUST use edit_file to attempt a fix. "
                        f"Stop reading files and make your best edit now.]"
                    )

        # Call model
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=4096,
                temperature=0.0,
            )
        except Exception as e:
            err_str = str(e)
            # If context too large, try compacting and retrying once
            if "400" in err_str or "context" in err_str.lower():
                messages = compact_context(messages)
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        max_tokens=4096,
                        temperature=0.0,
                    )
                except Exception as e2:
                    turn_logs.append({
                        "turn": turn + 1, "error": str(e2),
                        "duration_s": time.time() - turn_start
                    })
                    break
            else:
                turn_logs.append({
                    "turn": turn + 1, "error": err_str,
                    "duration_s": time.time() - turn_start
                })
                break

        choice = response.choices[0]
        assistant_msg = choice.message

        # Build assistant message dict
        msg_dict = {"role": "assistant", "content": assistant_msg.content or ""}
        if assistant_msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in assistant_msg.tool_calls
            ]
        messages.append(msg_dict)

        # If no tool calls, model is done (or gave up)
        if not assistant_msg.tool_calls:
            turn_log = {
                "turn": turn + 1, "action_type": "text_only",
                "context_tokens": response.usage.total_tokens if response.usage else 0,
                "repeated_action": False, "duration_s": time.time() - turn_start,
            }
            turn_logs.append(turn_log)
            if choice.finish_reason == "stop":
                break
            continue

        # Execute tool calls
        for tc in assistant_msg.tool_calls:
            func_name = tc.function.name
            try:
                func_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                func_args = {}

            # Track repetition
            action_key = f"{func_name}:{json.dumps(func_args, sort_keys=True)}"
            is_repeated = action_key in prev_actions
            prev_actions.add(action_key)

            result = execute_tool(func_name, func_args, workspace)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

            turn_log = {
                "turn": turn + 1,
                "action_type": func_name,
                "repeated_action": is_repeated,
                "context_tokens": response.usage.total_tokens if response.usage else 0,
                "duration_s": time.time() - turn_start,
            }
            turn_logs.append(turn_log)

    # Evaluate: generate diff and save patch
    diff_output, _ = run_cmd("git diff", str(workspace))
    fix_generated = bool(diff_output.strip())

    # Save patch for offline swebench evaluation
    if fix_generated:
        patch_dir = Path("/mnt/nvme/agent-harness/patches")
        patch_dir.mkdir(exist_ok=True)
        (patch_dir / f"{instance_id.replace('/', '__')}.patch").write_text(diff_output)

    # Find turn of first edit
    first_edit_turn = None
    for tl in turn_logs:
        if tl.get("action_type") == "edit_file":
            first_edit_turn = tl["turn"]
            break

    result = {
        "instance_id": instance_id,
        "fix_generated": fix_generated,
        "turns_used": len([t for t in turn_logs if t.get("action_type")]),
        "total_turns": effective_turn,
        "diff_size": len(diff_output),
        "first_edit_turn": first_edit_turn,
        "edit_count": sum(1 for t in turn_logs if t.get("action_type") == "edit_file"),
        "repeat_count": sum(1 for t in turn_logs if t.get("repeated_action")),
    }

    # Cleanup workspace
    subprocess.run(["rm", "-rf", str(workspace)], capture_output=True)

    return result, turn_logs


def load_issue_subset(issues_file):
    """Load issue IDs from a file (one per line)."""
    with open(issues_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def select_subset(dataset, n=50, seed=42):
    """Select a diverse subset of issues."""
    rng = random.Random(seed)

    # Group by repo for diversity
    by_repo = {}
    for i, item in enumerate(dataset):
        repo = item["repo"]
        if repo not in by_repo:
            by_repo[repo] = []
        by_repo[repo].append(i)

    # Proportional sampling from each repo
    selected = []
    total = len(dataset)
    for repo, indices in by_repo.items():
        count = max(1, round(n * len(indices) / total))
        sampled = rng.sample(indices, min(count, len(indices)))
        selected.extend(sampled)

    # Trim or pad to exactly n
    rng.shuffle(selected)
    if len(selected) > n:
        selected = selected[:n]
    elif len(selected) < n:
        remaining = [i for i in range(total) if i not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[:n - len(selected)])

    return selected


def main():
    parser = argparse.ArgumentParser(description="SWE-bench Lite Agent Evaluator")
    parser.add_argument("--list-issues", type=int, metavar="N",
                        help="Just print N issue IDs for the eval subset and exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--issues", type=str, help="File with issue IDs (one per line)")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--compact-at", type=int, default=None,
                        help="Compact context at this turn number")
    parser.add_argument("--restart-with-summary", action="store_true",
                        help="Restart with failure summary after max-turns")
    parser.add_argument("--output", type=str, default="results.jsonl")
    parser.add_argument("--per-turn-log", type=str, default=None)
    parser.add_argument("--endpoint", type=str, default=None)
    args = parser.parse_args()

    if args.endpoint:
        global ENDPOINT
        ENDPOINT = args.endpoint

    # Load dataset
    print("Loading SWE-bench Lite...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    # List-only mode
    if args.list_issues:
        indices = select_subset(ds, n=args.list_issues, seed=args.seed)
        for idx in sorted(indices):
            print(ds[idx]["instance_id"])
        return

    # Load issue subset
    if args.issues:
        target_ids = load_issue_subset(args.issues)
        issues = [item for item in ds if item["instance_id"] in set(target_ids)]
    else:
        indices = select_subset(ds, n=50, seed=args.seed)
        issues = [ds[i] for i in indices]

    print(f"Evaluating {len(issues)} issues, max_turns={args.max_turns}, "
          f"compact_at={args.compact_at}, restart={args.restart_with_summary}")

    # Setup client
    client = openai.OpenAI(base_url=ENDPOINT, api_key="not-needed")

    # Run evaluation
    results = []
    all_turn_logs = []
    fix_count = 0

    for i, instance in enumerate(issues):
        iid = instance["instance_id"]
        print(f"[{i+1}/{len(issues)}] {iid}...")

        result, turn_logs = run_agent_loop(
            client, instance,
            max_turns=args.max_turns,
            compact_at=args.compact_at,
            restart_with_summary=args.restart_with_summary,
        )

        results.append(result)
        all_turn_logs.extend(turn_logs)

        if result.get("fix_generated"):
            fix_count += 1

        print(f"  -> fix={result.get('fix_generated')}, edits={result.get('edit_count')}, "
              f"first_edit={result.get('first_edit_turn')}, turns={result.get('turns_used')}")

        # Write results incrementally
        with open(args.output, "a") as f:
            f.write(json.dumps(result) + "\n")

        if args.per_turn_log:
            with open(args.per_turn_log, "a") as f:
                for tl in turn_logs:
                    tl["instance_id"] = iid
                    f.write(json.dumps(tl) + "\n")

    # Summary
    total = len(issues)
    print(f"\n{'='*50}")
    print(f"Fixes generated: {fix_count}/{total} ({100*fix_count/total:.1f}%)")
    avg_turns = sum(r.get("turns_used", 0) for r in results) / max(total, 1)
    print(f"Avg turns: {avg_turns:.1f}")
    edit_issues = sum(1 for r in results if r.get("edit_count", 0) > 0)
    print(f"Issues with edits: {edit_issues}/{total} ({100*edit_issues/total:.1f}%)")
    avg_first_edit = [r["first_edit_turn"] for r in results if r.get("first_edit_turn")]
    if avg_first_edit:
        print(f"Avg first edit turn: {sum(avg_first_edit)/len(avg_first_edit):.1f}")
    repetitions = sum(1 for t in all_turn_logs if t.get("repeated_action"))
    print(f"Repeated actions: {repetitions}/{len(all_turn_logs)} "
          f"({100*repetitions/max(len(all_turn_logs),1):.1f}%)")
    print(f"Patches saved to: /mnt/nvme/agent-harness/patches/")
    print(f"Output: {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
