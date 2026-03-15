#!/usr/bin/env python3
"""
Hashline edit agent for SWE-bench evaluation.

Same LangGraph ReAct loop as langgraph_agent.py but with hash-anchored
LINE#ID editing instead of str_replace. This isolates the effect of the
edit tool format (the "harness problem" from can.ac blog).

read_file returns: "42:a3|    def method(self):"
edit_file takes: start_hash="42:a3", end_hash="45:f1", new_content="..."

Output: JSON on last line: {"pass": bool, "turns": int, "tokens": int, "fix_generated": bool}
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys


def line_hash(line):
    """2-char content hash for a line."""
    return hashlib.md5(line.encode()).hexdigest()[:2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--test-cmd", required=True)
    parser.add_argument("--max-turns", type=int, default=30)
    args = parser.parse_args()

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        print('{"pass": false, "turns": 0, "tokens": 0, "fix_generated": false, "error": "langgraph not installed"}')
        sys.exit(0)

    @tool
    def read_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
        """Read a file. Returns lines with hash anchors: LINE:HASH|content.
        Use LINE:HASH references in edit_file to identify lines precisely."""
        full = os.path.join(args.workspace, path.lstrip("/"))
        if not os.path.isfile(full):
            return f"Error: file not found: {path}"
        with open(full, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        start = max(1, start_line) if start_line else 1
        end = min(len(lines), end_line) if end_line else len(lines)
        selected = lines[start - 1:end]
        numbered = []
        for i, raw_line in enumerate(selected):
            ln = start + i
            h = line_hash(raw_line.rstrip('\n'))
            numbered.append(f"{ln}:{h}|{raw_line.rstrip(chr(10))}")
        return "\n".join(numbered)[:50000]

    @tool
    def edit_file(path: str, start_hash: str, end_hash: str, new_content: str) -> str:
        """Edit a file using hash-anchored line references.
        start_hash and end_hash are LINE:HASH from read_file output (e.g. "42:a3").
        Lines from start to end (inclusive) are replaced with new_content."""
        full = os.path.join(args.workspace, path.lstrip("/"))
        if not os.path.isfile(full):
            return f"Error: file not found: {path}"
        with open(full, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        def parse_ref(ref):
            parts = ref.strip().split(":")
            if len(parts) != 2:
                return None, "bad format"
            try:
                ln = int(parts[0])
            except ValueError:
                return None, "bad line number"
            h = parts[1]
            if ln < 1 or ln > len(lines):
                return None, f"line {ln} out of range (1-{len(lines)})"
            actual = line_hash(lines[ln - 1].rstrip('\n'))
            if actual != h:
                return None, f"hash mismatch at line {ln}: expected {h}, got {actual}"
            return ln, None

        start_ln, err = parse_ref(start_hash)
        if err:
            return f"Error: start_hash '{start_hash}': {err}"
        end_ln, err = parse_ref(end_hash)
        if err:
            return f"Error: end_hash '{end_hash}': {err}"
        if start_ln > end_ln:
            return "Error: start must be before end"

        new_lines = new_content.split("\n")
        # Preserve trailing newline convention
        new_lines_with_nl = [l + "\n" for l in new_lines]
        lines[start_ln - 1:end_ln] = new_lines_with_nl
        with open(full, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return f"Edited {path}: replaced lines {start_ln}-{end_ln} with {len(new_lines)} lines"

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file."""
        full = os.path.join(args.workspace, path.lstrip("/"))
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File written: {path}"

    @tool
    def run_command(command: str, timeout: int = 60) -> str:
        """Execute a shell command."""
        timeout = min(timeout, 120)
        venv = os.path.join(args.workspace, ".venv", "bin", "activate")
        if os.path.exists(venv):
            command = f"source {venv} && {command}"
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=args.workspace, executable="/bin/bash",
        )
        output = proc.stdout
        if proc.stderr:
            output += f"\nSTDERR:\n{proc.stderr}"
        if proc.returncode != 0:
            output += f"\n(exit code {proc.returncode})"
        return output[:8000]

    @tool
    def search_files(pattern: str, path: str = ".", file_glob: str = "") -> str:
        """Search for a pattern using grep."""
        full = os.path.join(args.workspace, path.lstrip("/"))
        cmd = f"grep -rn --include='{file_glob}' '{pattern}' '{full}'" if file_glob else f"grep -rn '{pattern}' '{full}'"
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=args.workspace)
        output = proc.stdout.replace(args.workspace + "/", "")
        if not output:
            return "No matches found."
        lines = output.split("\n")
        if len(lines) > 50:
            output = "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more)"
        return output[:8000]

    tools = [read_file, edit_file, write_file, run_command, search_files]

    llm = ChatOpenAI(model=args.model, temperature=0.0, max_tokens=4096)
    agent = create_react_agent(llm, tools)

    prompt = (
        f"You are a coding agent. Fix this issue in the repository:\n\n"
        f"{args.issue}\n\n"
        f"After fixing, verify with: {args.test_cmd}\n\n"
        f"read_file returns lines as LINE:HASH|content. Use the LINE:HASH references "
        f"in edit_file's start_hash and end_hash parameters to identify lines precisely. "
        f"This avoids exact-text-match failures."
    )

    turns = 0
    total_tokens = 0

    try:
        result = agent.invoke(
            {"messages": [("user", prompt)]},
            config={"recursion_limit": args.max_turns * 2},
        )
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                turns += 1
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                total_tokens += msg.usage_metadata.get("total_tokens", 0)
    except Exception as e:
        print(json.dumps({
            "pass": False, "turns": turns, "tokens": total_tokens,
            "fix_generated": False, "error": str(e)[:200],
        }))
        return

    # Check tests
    test_output = run_command.invoke({"command": args.test_cmd, "timeout": 120})
    test_lower = test_output.lower()
    tests_pass = False
    if "passed" in test_lower and "failed" not in test_lower.split("passed")[0]:
        tests_pass = True
    if ("\nok" in test_lower or test_lower.rstrip().endswith("ok")) and "fail" not in test_lower:
        tests_pass = True

    diff_proc = subprocess.run("git diff", shell=True, capture_output=True, text=True, cwd=args.workspace)
    fix_generated = bool(diff_proc.stdout.strip())

    print(json.dumps({
        "pass": tests_pass,
        "turns": turns,
        "tokens": total_tokens,
        "fix_generated": fix_generated,
    }))


if __name__ == "__main__":
    main()
