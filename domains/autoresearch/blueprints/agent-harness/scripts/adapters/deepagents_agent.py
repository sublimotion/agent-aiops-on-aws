#!/usr/bin/env python3
"""
DeepAgents adapter for SWE-bench evaluation.

Uses langchain-ai/deepagents' batteries-included agent with built-in
filesystem tools, shell execution, sub-agents, and auto-summarization.
Key differentiators vs vanilla LangGraph ReAct:
  - SummarizationMiddleware auto-compacts long conversations
  - write_todos planning tool for structured task breakdown
  - task tool spawns isolated sub-agents with separate context windows
  - Opinionated system prompt (Claude Code-inspired)

Output: JSON on last line: {"pass": bool, "turns": int, "tokens": int, "fix_generated": bool}
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--test-cmd", required=True)
    parser.add_argument("--max-turns", type=int, default=30)
    args = parser.parse_args()

    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        print(json.dumps({
            "pass": False, "turns": 0, "tokens": 0,
            "fix_generated": False, "error": f"import failed: {e}",
        }))
        sys.exit(0)

    llm = ChatOpenAI(
        model=args.model,
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:9000/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        temperature=0.0,
        max_tokens=4096,
    )

    # FilesystemBackend scopes all file tools to the workspace directory
    backend = FilesystemBackend(root=args.workspace)

    try:
        agent = create_deep_agent(
            model=llm,
            backend=backend,
            system_prompt=(
                f"You are a coding agent fixing a bug in this repository.\n"
                f"Working directory: {args.workspace}\n\n"
                f"Fix this issue:\n{args.issue}\n\n"
                f"After fixing, verify with: {args.test_cmd}\n\n"
                f"Use edit_file for targeted changes. Do not rewrite entire files."
            ),
        )

        result = agent.invoke(
            {"messages": [("user", "Fix the issue described in the system prompt. Start by reading relevant files.")]},
            config={"recursion_limit": args.max_turns * 2},
        )

        # Count tool-use messages as turns, accumulate tokens
        turns = 0
        total_tokens = 0
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                turns += 1
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                total_tokens += msg.usage_metadata.get("total_tokens", 0)

    except Exception as e:
        print(json.dumps({
            "pass": False, "turns": 0, "tokens": 0,
            "fix_generated": False, "error": str(e)[:200],
        }))
        return

    # Check tests
    venv = os.path.join(args.workspace, ".venv", "bin", "activate")
    test_command = args.test_cmd
    if os.path.exists(venv):
        test_command = f"source {venv} && {test_command}"
    try:
        proc = subprocess.run(
            test_command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=args.workspace, executable="/bin/bash",
        )
        test_output = (proc.stdout + "\n" + proc.stderr).lower()
        tests_pass = False
        if "passed" in test_output and "failed" not in test_output.split("passed")[0]:
            tests_pass = True
        if ("\nok" in test_output or test_output.rstrip().endswith("ok")) and "fail" not in test_output:
            tests_pass = True
    except (subprocess.TimeoutExpired, Exception):
        tests_pass = False

    # Check for fix
    diff_proc = subprocess.run(
        "git diff", shell=True, capture_output=True, text=True, cwd=args.workspace,
    )
    fix_generated = bool(diff_proc.stdout.strip())

    print(json.dumps({
        "pass": tests_pass,
        "turns": turns,
        "tokens": total_tokens,
        "fix_generated": fix_generated,
    }))


if __name__ == "__main__":
    main()
