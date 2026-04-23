#!/usr/bin/env python3
"""
Run generated tests in the current workspace.

Usage:
    python3 verify/run_tests.py --test-file tests.py
    python3 verify/run_tests.py --test-file tests.py --workspace /path/to/repo
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def log_telemetry(tool: str, inputs: dict, outputs: dict, elapsed_s: float, cost_usd: float = 0.0):
    telemetry_path = Path(__file__).parent / "telemetry.jsonl"
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "inputs": inputs,
        "outputs": outputs,
        "elapsed_s": round(elapsed_s, 2),
        "cost_usd": round(cost_usd, 6),
    }
    with open(telemetry_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run generated tests in workspace")
    parser.add_argument("--test-file", required=True, help="Path to test file")
    parser.add_argument("--workspace", default=".", help="Repo workspace (default: cwd)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    args = parser.parse_args()

    test_code = Path(args.test_file).read_text()
    workspace = os.path.abspath(args.workspace)
    test_filename = "test_verification.py"
    test_path = os.path.join(workspace, test_filename)

    start = time.monotonic()
    result = {"passed": 0, "failed": 0, "errors": 0, "total": 0, "test_results": []}

    try:
        # Write test file to workspace
        Path(test_path).write_text(test_code)

        # Check syntax
        try:
            compile(test_code, test_filename, "exec")
        except SyntaxError as e:
            elapsed = time.monotonic() - start
            print(json.dumps({"error": f"Syntax error: {e}", "passed": 0, "failed": 0, "total": 0}))
            log_telemetry("run_tests", {"test_file": args.test_file}, {"error": f"syntax: {e}"}, elapsed)
            return

        # Run pytest
        proc = subprocess.run(
            ["python3", "-m", "pytest", test_filename, "-v", "--tb=short", "--no-header", "-x"],
            capture_output=True, text=True, timeout=args.timeout, cwd=workspace,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        # Parse results
        for line in proc.stdout.split("\n"):
            line = line.strip()
            if "::" not in line:
                continue
            if " PASSED" in line:
                result["test_results"].append({"name": line.split(" PASSED")[0].strip(), "status": "passed"})
                result["passed"] += 1
                result["total"] += 1
            elif " FAILED" in line:
                result["test_results"].append({"name": line.split(" FAILED")[0].strip(), "status": "failed"})
                result["failed"] += 1
                result["total"] += 1
            elif " ERROR" in line:
                result["test_results"].append({"name": line.split(" ERROR")[0].strip(), "status": "error"})
                result["errors"] += 1
                result["total"] += 1

        if result["total"] == 0:
            output = proc.stdout + proc.stderr
            if "ImportError" in output or "ModuleNotFoundError" in output:
                result["error"] = "Import error in generated tests"
            elif "no tests ran" in output.lower():
                result["error"] = "No tests collected"
            elif proc.returncode != 0:
                result["error"] = f"pytest exited with code {proc.returncode}"

    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {args.timeout}s"
    except Exception as e:
        result["error"] = str(e)[:500]
    finally:
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        cache_dir = os.path.join(workspace, "__pycache__")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

    elapsed = time.monotonic() - start

    log_telemetry(
        tool="run_tests",
        inputs={"test_file": args.test_file, "workspace": workspace},
        outputs={"passed": result["passed"], "failed": result["failed"],
                 "total": result["total"], "error": result.get("error")},
        elapsed_s=elapsed,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
