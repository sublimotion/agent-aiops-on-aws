#!/usr/bin/env python3
"""
Verification primitive: run tests in a sandboxed workspace.

Executes a test file against a patched repo and returns per-test results.
Reuses the agent-harness workspace infrastructure.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def run_tests(
    test_code: str,
    workspace: str,
    timeout: int = 60,
    test_filename: str = "test_verification.py",
) -> dict:
    """
    Run generated test code in a workspace.

    Args:
        test_code: Python test file content
        workspace: Path to the repo workspace (already at base_commit + patch applied)
        timeout: Max seconds for test execution
        test_filename: Name for the test file

    Returns:
        dict with: passed, total, failed, errors, test_results (list),
                   stdout, stderr, elapsed_ms, error
    """
    result = {
        "passed": 0,
        "total": 0,
        "failed": 0,
        "errors": 0,
        "test_results": [],
        "stdout": "",
        "stderr": "",
        "elapsed_ms": 0,
        "compile_error": False,
        "error": None,
    }

    if not test_code.strip():
        result["error"] = "Empty test code"
        return result

    # Write test file to workspace
    test_path = os.path.join(workspace, test_filename)
    try:
        Path(test_path).write_text(test_code)
    except Exception as e:
        result["error"] = f"Failed to write test file: {e}"
        return result

    # First check if the test file compiles
    try:
        compile(test_code, test_filename, "exec")
    except SyntaxError as e:
        result["error"] = f"Syntax error: {e}"
        result["compile_error"] = True
        _cleanup(test_path)
        return result

    # Run with pytest, capture per-test results via JSON report
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [
                "python3", "-m", "pytest", test_filename,
                "-v", "--tb=short", "--no-header",
                "-x",  # stop on first failure for efficiency
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workspace,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        result["stdout"] = proc.stdout[-4000:]
        result["stderr"] = proc.stderr[-2000:]

    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {timeout}s"
        result["elapsed_ms"] = timeout * 1000
        _cleanup(test_path)
        return result

    except Exception as e:
        result["error"] = str(e)[:500]
        _cleanup(test_path)
        return result

    result["elapsed_ms"] = int((time.monotonic() - start) * 1000)

    # Parse pytest verbose output
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

    # If no individual results parsed, check for import/collection errors
    if result["total"] == 0:
        output = proc.stdout + proc.stderr
        if "ImportError" in output or "ModuleNotFoundError" in output:
            result["error"] = "Import error in generated tests"
            result["compile_error"] = True
        elif "no tests ran" in output.lower():
            result["error"] = "No tests collected"
        elif proc.returncode != 0:
            result["error"] = f"pytest exited with code {proc.returncode}"

    _cleanup(test_path)
    return result


def _cleanup(test_path: str):
    """Remove generated test file."""
    try:
        os.remove(test_path)
    except OSError:
        pass
    # Also remove __pycache__ if created
    cache_dir = os.path.join(os.path.dirname(test_path), "__pycache__")
    if os.path.isdir(cache_dir):
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run generated tests in workspace")
    parser.add_argument("--test-file", required=True, help="Path to test file")
    parser.add_argument("--workspace", required=True, help="Path to repo workspace")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    test_code = Path(args.test_file).read_text()
    result = run_tests(test_code, args.workspace, args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
