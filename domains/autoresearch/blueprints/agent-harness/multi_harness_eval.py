#!/usr/bin/env python3
"""
Phase 2: Multi-Harness Comparison for SWE-bench Lite.

Runs multiple agent harnesses against the same 50-issue subset and produces
a comparative leaderboard.

Usage:
    python3 multi_harness_eval.py --harness langgraph --endpoint http://localhost:9000/v1
    python3 multi_harness_eval.py --harness sera --endpoint http://localhost:9000/v1
    python3 multi_harness_eval.py --harness aider --endpoint http://localhost:9000/v1
    python3 multi_harness_eval.py --report results/phase2_*.jsonl
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

from datasets import load_dataset

WORKSPACE_ROOT = Path("/mnt/nvme/sera-workspaces")
REPO_CACHE = WORKSPACE_ROOT / "_repo_cache"
RESULTS_DIR = Path("/mnt/nvme/agent-harness/results")
PATCHES_DIR = Path("/mnt/nvme/agent-harness/patches")
SCRIPT_DIR = Path(__file__).parent


def select_subset(dataset, n=50, seed=42):
    """Select the same diverse subset as Phase 1 (seed 42)."""
    rng = random.Random(seed)
    by_repo = {}
    for i, item in enumerate(dataset):
        repo = item["repo"]
        if repo not in by_repo:
            by_repo[repo] = []
        by_repo[repo].append(i)

    selected = []
    total = len(dataset)
    for repo, indices in by_repo.items():
        count = max(1, round(n * len(indices) / total))
        sampled = rng.sample(indices, min(count, len(indices)))
        selected.extend(sampled)

    rng.shuffle(selected)
    if len(selected) > n:
        selected = selected[:n]
    elif len(selected) < n:
        remaining = [i for i in range(total) if i not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[:n - len(selected)])

    return selected


def setup_workspace(instance):
    """Clone repo at base_commit into a workspace directory."""
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]

    workspace = WORKSPACE_ROOT / instance_id.replace("/", "__")

    if workspace.exists():
        subprocess.run(["rm", "-rf", str(workspace)], check=True)

    repo_dir = repo.replace("/", "__")
    cache_path = REPO_CACHE / repo_dir

    if cache_path.exists():
        check = subprocess.run(
            ["git", "cat-file", "-t", base_commit],
            cwd=cache_path, capture_output=True
        )
        if check.returncode == 0:
            subprocess.run(["git", "clone", str(cache_path), str(workspace)],
                           capture_output=True, check=True)
        else:
            subprocess.run(
                ["git", "fetch", "origin", "--depth=1", base_commit],
                cwd=cache_path, capture_output=True, timeout=120
            )
            check2 = subprocess.run(
                ["git", "cat-file", "-t", base_commit],
                cwd=cache_path, capture_output=True
            )
            if check2.returncode == 0:
                subprocess.run(["git", "clone", str(cache_path), str(workspace)],
                               capture_output=True, check=True)
            else:
                subprocess.run(
                    ["git", "clone", f"https://github.com/{repo}.git", str(workspace)],
                    capture_output=True, check=True, timeout=300
                )
    else:
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo}.git", str(workspace)],
            capture_output=True, check=True, timeout=300
        )

    subprocess.run(["git", "checkout", base_commit], cwd=workspace,
                   capture_output=True, check=True)
    subprocess.run(["git", "clean", "-fdx"], cwd=workspace, capture_output=True)

    return workspace


def run_langgraph(instance, workspace, endpoint, model, max_turns):
    """Run LangGraph ReAct agent."""
    from langgraph_adapter import run_langgraph_agent
    return run_langgraph_agent(
        instance["instance_id"],
        instance["problem_statement"],
        str(workspace),
        endpoint,
        max_turns,
        model,
    )


def run_sera(instance, workspace, endpoint, model, max_turns):
    """Run SERA agent (Phase 1 baseline with Config D settings)."""
    # Import sera-eval functions
    sys.path.insert(0, str(SCRIPT_DIR))
    import importlib
    sera = importlib.import_module("sera-eval") if Path(SCRIPT_DIR / "sera-eval.py").exists() else None

    if sera is None:
        # Fallback: run as subprocess
        cmd = [
            sys.executable, str(SCRIPT_DIR / "sera-eval.py"),
            "--issues-inline", instance["instance_id"],
            "--max-turns", str(max_turns),
            "--output", "/dev/stdout",
            "--endpoint", endpoint,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip().split("\n")[-1])
                data["harness"] = "sera"
                return data
        except Exception as e:
            pass

    # Direct inline SERA-like loop (simplified version matching sera-eval.py logic)
    import openai as openai_mod

    client = openai_mod.OpenAI(base_url=endpoint, api_key="not-needed")

    explore_budget = max(2, max_turns // 5)
    fix_budget = max_turns - 3

    system_prompt = f"""You are an expert software engineer tasked with fixing a bug in a Python repository.

You have access to the following tools:
- read_file(path): Read a file's contents
- edit_file(path, old_str, new_str): Replace old_str with new_str in the file
- run_command(command): Run a shell command and get its output (timeout: 30s)
- search_files(pattern, path): Search for files matching a glob pattern
- search_code(query, path): Search file contents for a string

You have {max_turns} turns total. You MUST attempt a fix using edit_file before your turns run out.

Strategy:
1. Turns 1-{explore_budget}: Quickly locate the relevant code
2. Turns {explore_budget}-{fix_budget}: Make your fix using edit_file, then run the failing tests
3. Remaining turns: Iterate based on test results

CRITICAL: Do NOT spend more than {explore_budget} turns just reading. Make your best edit attempt early."""

    TOOLS_SPEC = [
        {"type": "function", "function": {"name": "read_file", "description": "Read file contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "edit_file", "description": "Replace old_str with new_str", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}}, "required": ["path", "old_str", "new_str"]}}},
        {"type": "function", "function": {"name": "run_command", "description": "Run shell command (30s timeout)", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
        {"type": "function", "function": {"name": "search_files", "description": "Glob search for files", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}}, "required": ["pattern"]}}},
        {"type": "function", "function": {"name": "search_code", "description": "Search file contents", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string", "default": "."}}, "required": ["query"]}}},
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Fix this issue:\n\n{instance['problem_statement']}\n\nThe repo is checked out at the correct commit."},
    ]

    turn = 0
    edit_count = 0
    first_edit_turn = None
    has_edited = False
    repeat_count = 0
    prev_actions = set()

    while turn < max_turns:
        turn += 1

        # Auto-compaction
        estimated_tokens = sum(len(str(m.get("content", ""))) // 3 for m in messages)
        if estimated_tokens > 20000:
            messages = compact_sera_context(messages)

        # Turn pressure
        if turn >= explore_budget and not has_edited:
            turns_left = max_turns - turn
            if messages and messages[-1]["role"] == "tool":
                if turns_left <= 3:
                    messages[-1]["content"] += (
                        f"\n\n[URGENT: Only {turns_left} turns left! Use edit_file RIGHT NOW.]"
                    )
                elif turn % 2 == 0:
                    messages[-1]["content"] += (
                        f"\n\n[SYSTEM: Turn {turn}/{max_turns}. You MUST use edit_file now.]"
                    )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS_SPEC,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=4096,
            )
        except Exception as e:
            print(f"  API error at turn {turn}: {e}", file=sys.stderr)
            break

        msg = response.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            if turn >= max_turns - 2:
                break
            continue

        # Process tool calls
        assistant_msg = {"role": "assistant", "content": msg.content or "", "tool_calls": []}
        for tc in msg.tool_calls:
            assistant_msg["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })
        messages.append(assistant_msg)

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # Track repeats
            action_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
            if action_key in prev_actions:
                repeat_count += 1
            prev_actions.add(action_key)

            # Execute tool
            result = execute_sera_tool(tool_name, args, workspace)

            if tool_name == "edit_file" and "Successfully" in result:
                edit_count += 1
                has_edited = True
                if first_edit_turn is None:
                    first_edit_turn = turn

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Check for diff
    diff_cmd = subprocess.run(["git", "diff"], cwd=workspace, capture_output=True, text=True)
    diff_output = diff_cmd.stdout

    fix_generated = bool(diff_output.strip())

    return {
        "instance_id": instance["instance_id"],
        "harness": "sera",
        "fix_generated": fix_generated,
        "turns_used": turn,
        "edit_count": edit_count,
        "first_edit_turn": first_edit_turn,
        "diff_size": len(diff_output.strip()),
        "repeat_count": repeat_count,
    }


def compact_sera_context(messages):
    """Compact context preserving Mistral role ordering."""
    if len(messages) <= 4:
        return messages
    compacted = []
    for msg in messages[:2]:
        compacted.append(msg)
    remaining = messages[2:]
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


def execute_sera_tool(tool_name, args, workspace):
    """Execute a tool (same logic as sera-eval.py)."""
    cwd = str(workspace)

    if tool_name == "read_file":
        fpath = workspace / args.get("path", "")
        if not fpath.exists():
            return f"Error: File not found: {args.get('path')}"
        try:
            content = fpath.read_text()
            if len(content) > 10000:
                content = content[:10000] + "\n\n... (truncated)"
            lines = content.split("\n")
            return "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
        except Exception as e:
            return f"Error: {e}"

    elif tool_name == "edit_file":
        fpath = workspace / args.get("path", "")
        if not fpath.exists():
            return f"Error: File not found: {args.get('path')}"
        old_str = args.get("old_str", "")
        new_str = args.get("new_str", "")
        content = fpath.read_text()
        if old_str not in content:
            return f"Error: old_str not found in {args.get('path')}."
        if content.count(old_str) > 1:
            return f"Error: old_str appears {content.count(old_str)} times."
        fpath.write_text(content.replace(old_str, new_str, 1))
        return f"Successfully edited {args.get('path')}"

    elif tool_name == "run_command":
        cmd = args.get("command", "")
        dangerous = ["rm -rf", "sudo", "curl", "wget", "pip install", "git clean", "git reset --hard"]
        if any(d in cmd for d in dangerous):
            return f"Error: Command blocked: {cmd}"
        try:
            if not os.path.isdir(cwd):
                return "Error: workspace missing"
            result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr
            if len(output) > 8000:
                output = output[:4000] + "\n...(truncated)...\n" + output[-4000:]
            return f"Exit code: {result.returncode}\n{output}"
        except subprocess.TimeoutExpired:
            return "(timed out)"
        except Exception as e:
            return f"Error: {e}"

    elif tool_name == "search_files":
        import glob as globmod
        pattern = args.get("pattern", "")
        search_path = workspace / args.get("path", "")
        matches = sorted(globmod.glob(str(search_path / pattern), recursive=True))
        matches = [str(Path(m).relative_to(workspace)) for m in matches[:50]]
        return "\n".join(matches) if matches else "No files found."

    elif tool_name == "search_code":
        query = args.get("query", "")
        path = args.get("path", ".")
        try:
            result = subprocess.run(
                f"grep -rn --include='*.py' '{query}' {path} | head -30",
                shell=True, cwd=cwd, capture_output=True, text=True, timeout=15
            )
            return result.stdout if result.stdout.strip() else "No matches."
        except Exception:
            return "Search error."

    return f"Unknown tool: {tool_name}"


def run_aider(instance, workspace, endpoint, model, max_turns):
    """Run Aider in message mode (single-turn)."""
    problem = instance["problem_statement"]
    instance_id = instance["instance_id"]

    # Aider uses litellm model naming for OpenAI-compatible endpoints
    env = os.environ.copy()
    env["OPENAI_API_BASE"] = endpoint
    env["OPENAI_API_KEY"] = "not-needed"

    cmd = [
        sys.executable, "-m", "aider",
        "--model", f"openai/{model}",
        "--no-auto-commits",
        "--no-git",
        "--yes",
        "--edit-format", "diff",
        "--message", f"Fix this issue:\n\n{problem}",
    ]

    try:
        result = subprocess.run(
            cmd, cwd=str(workspace), capture_output=True, text=True,
            timeout=120, env=env
        )
    except subprocess.TimeoutExpired:
        result = type("R", (), {"stdout": "", "stderr": "timeout", "returncode": -1})()
    except Exception as e:
        result = type("R", (), {"stdout": "", "stderr": str(e), "returncode": -1})()

    # Check diff
    diff_cmd = subprocess.run(["git", "diff"], cwd=workspace, capture_output=True, text=True)
    diff_output = diff_cmd.stdout
    fix_generated = bool(diff_output.strip())

    return {
        "instance_id": instance_id,
        "harness": "aider",
        "fix_generated": fix_generated,
        "turns_used": 1 if fix_generated else 0,
        "edit_count": 1 if fix_generated else 0,
        "first_edit_turn": 1 if fix_generated else None,
        "diff_size": len(diff_output.strip()),
    }


HARNESS_RUNNERS = {
    "langgraph": run_langgraph,
    "sera": run_sera,
    "aider": run_aider,
}


def run_harness_eval(harness_name, endpoint, model, max_turns, resume_from=0):
    """Run a single harness against the 50-issue subset."""
    print(f"\n{'='*60}")
    print(f"Phase 2: {harness_name.upper()} harness")
    print(f"Endpoint: {endpoint}, Model: {model}, Max turns: {max_turns}")
    print(f"{'='*60}\n")

    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    indices = select_subset(dataset, n=50, seed=42)
    issues = [dataset[i] for i in indices]

    runner = HARNESS_RUNNERS[harness_name]
    output_file = RESULTS_DIR / f"phase2_{harness_name}.jsonl"
    log_file = RESULTS_DIR / f"phase2_{harness_name}.log"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    # Resume support
    completed = set()
    if resume_from > 0 and output_file.exists():
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                completed.add(data["instance_id"])

    results = []
    for idx, instance in enumerate(issues):
        iid = instance["instance_id"]
        if iid in completed:
            print(f"[{idx+1}/50] {iid}... SKIPPED (already done)")
            continue

        print(f"[{idx+1}/50] {iid}...")
        print(f"  Setting up workspace for {iid}...")

        try:
            workspace = setup_workspace(instance)
        except Exception as e:
            print(f"  Workspace setup failed: {e}")
            result = {
                "instance_id": iid,
                "harness": harness_name,
                "fix_generated": False,
                "turns_used": 0,
                "edit_count": 0,
                "first_edit_turn": None,
                "diff_size": 0,
                "error": True,
            }
            results.append(result)
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")
            continue

        t0 = time.time()
        try:
            result = runner(instance, workspace, endpoint, model, max_turns)
        except Exception as e:
            print(f"  Harness error: {e}")
            result = {
                "instance_id": iid,
                "harness": harness_name,
                "fix_generated": False,
                "turns_used": 0,
                "edit_count": 0,
                "first_edit_turn": None,
                "diff_size": 0,
                "error": True,
            }
        elapsed = time.time() - t0
        result["elapsed_seconds"] = round(elapsed, 1)

        # Save patch if fix generated
        if result.get("fix_generated"):
            diff_cmd = subprocess.run(["git", "diff"], cwd=workspace, capture_output=True, text=True)
            patch_file = PATCHES_DIR / f"{harness_name}__{iid.replace('/', '__')}.patch"
            patch_file.write_text(diff_cmd.stdout)

        results.append(result)
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        fix_str = "fix" if result.get("fix_generated") else "no-fix"
        edits = result.get("edit_count", 0)
        turns = result.get("turns_used", 0)
        first = result.get("first_edit_turn", "None")
        print(f"  -> {fix_str}, edits={edits}, first_edit={first}, turns={turns}, {elapsed:.1f}s")

        # Log to file
        with open(log_file, "a") as f:
            f.write(f"[{idx+1}/50] {iid}: {fix_str}, edits={edits}, turns={turns}, {elapsed:.1f}s\n")

    # Summary
    total = len(results)
    fixes = sum(1 for r in results if r.get("fix_generated"))
    edits_any = sum(1 for r in results if r.get("edit_count", 0) > 0)
    first_edits = [r["first_edit_turn"] for r in results if r.get("first_edit_turn") is not None]
    times = [r.get("elapsed_seconds", 0) for r in results]

    print(f"\n{'='*50}")
    print(f"PHASE 2 SUMMARY: {harness_name.upper()}")
    print(f"{'='*50}")
    print(f"Fix rate: {fixes}/{total} ({100*fixes/max(total,1):.0f}%)")
    print(f"Edit rate: {edits_any}/{total} ({100*edits_any/max(total,1):.0f}%)")
    if first_edits:
        print(f"Avg first edit turn: {sum(first_edits)/len(first_edits):.1f}")
    if times:
        print(f"Avg latency: {sum(times)/len(times):.1f}s")
    print(f"Output: {output_file}")
    print(f"{'='*50}")


def generate_report(result_files):
    """Generate comparison report from Phase 2 result files."""
    print(f"\n{'='*60}")
    print("PHASE 2 HARNESS LEADERBOARD")
    print(f"{'='*60}\n")

    harness_data = {}
    for fpath in result_files:
        with open(fpath) as f:
            results = [json.loads(line) for line in f if line.strip()]
        if not results:
            continue
        harness = results[0].get("harness", Path(fpath).stem.replace("phase2_", ""))
        fixes = sum(1 for r in results if r.get("fix_generated"))
        edits = sum(1 for r in results if r.get("edit_count", 0) > 0)
        first_edits = [r["first_edit_turn"] for r in results if r.get("first_edit_turn") is not None]
        times = [r.get("elapsed_seconds", 0) for r in results]
        turns = [r.get("turns_used", 0) for r in results if r.get("turns_used", 0) > 0]

        harness_data[harness] = {
            "total": len(results),
            "fixes": fixes,
            "fix_rate": fixes / max(len(results), 1),
            "edits": edits,
            "avg_first_edit": sum(first_edits) / len(first_edits) if first_edits else 0,
            "avg_turns": sum(turns) / len(turns) if turns else 0,
            "avg_latency": sum(times) / len(times) if times else 0,
            "fix_ids": [r["instance_id"] for r in results if r.get("fix_generated")],
        }

    # Sort by fix rate
    ranked = sorted(harness_data.items(), key=lambda x: -x[1]["fix_rate"])

    print(f"| Rank | Harness | Fix Rate | Edit Rate | Avg 1st Edit | Avg Turns | Avg Latency |")
    print(f"|------|---------|----------|-----------|-------------|-----------|-------------|")
    for i, (name, d) in enumerate(ranked, 1):
        fr = f"{d['fixes']}/{d['total']} ({d['fix_rate']*100:.0f}%)"
        er = f"{d['edits']}/{d['total']} ({d['edits']/max(d['total'],1)*100:.0f}%)"
        fe = f"{d['avg_first_edit']:.1f}" if d['avg_first_edit'] > 0 else "—"
        at = f"{d['avg_turns']:.1f}" if d['avg_turns'] > 0 else "—"
        al = f"{d['avg_latency']:.1f}s"
        print(f"| {i} | {name} | {fr} | {er} | {fe} | {at} | {al} |")

    # Complementary analysis
    if len(harness_data) >= 2:
        print(f"\n### Complementary Fixes")
        all_fix_sets = {name: set(d["fix_ids"]) for name, d in harness_data.items()}
        union = set()
        for s in all_fix_sets.values():
            union |= s
        print(f"Union of all harnesses: {len(union)}/50 ({len(union)*100/50:.0f}%)")
        for name, fids in all_fix_sets.items():
            unique = fids - set().union(*(s for n, s in all_fix_sets.items() if n != name))
            if unique:
                print(f"  {name}-only: {len(unique)} ({', '.join(sorted(unique)[:5])}{'...' if len(unique) > 5 else ''})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Multi-Harness Comparison")
    parser.add_argument("--harness", choices=list(HARNESS_RUNNERS.keys()),
                        help="Which harness to run")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--model", default="devstral-small-2")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--resume", type=int, default=0, help="Resume from issue N")
    parser.add_argument("--report", nargs="+", help="Generate report from result files")
    args = parser.parse_args()

    if args.report:
        generate_report(args.report)
    elif args.harness:
        run_harness_eval(args.harness, args.endpoint, args.model, args.max_turns, args.resume)
    else:
        parser.print_help()
