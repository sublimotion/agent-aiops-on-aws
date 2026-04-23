#!/usr/bin/env python3
"""
Pre-flight smoke test for agent harness evaluations.

Validates that the vLLM endpoint is healthy, the model responds correctly,
tool calling works, and non-compliant tool_call_ids are handled.

Usage:
    python3 smoke_test.py --endpoint http://localhost:8000 --model devstral-small-2

Exit code 0 = all checks pass, non-zero = failure (do not proceed with eval).
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def check(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def test_health(endpoint: str) -> bool:
    try:
        req = urllib.request.Request(f"{endpoint}/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return check("health", resp.status == 200, f"status={resp.status}")
    except Exception as e:
        return check("health", False, str(e))


def test_model_available(endpoint: str, model: str) -> bool:
    try:
        req = urllib.request.Request(f"{endpoint}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            found = model in models
            return check("model_available", found,
                         f"wanted={model}, available={models}")
    except Exception as e:
        return check("model_available", False, str(e))


def test_basic_completion(endpoint: str, model: str) -> bool:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
        "max_tokens": 16,
    }).encode()
    try:
        req = urllib.request.Request(
            f"{endpoint}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            has_choices = "choices" in data and len(data["choices"]) > 0
            content = data["choices"][0]["message"]["content"][:50] if has_choices else ""
            return check("basic_completion", has_choices, f"response={content!r}")
    except Exception as e:
        return check("basic_completion", False, str(e))


def test_tool_calling(endpoint: str, model: str) -> bool:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Read the file test.py"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }],
        "max_tokens": 256,
    }).encode()
    try:
        req = urllib.request.Request(
            f"{endpoint}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls", [])
            has_tool_call = len(tool_calls) > 0
            detail = ""
            if has_tool_call:
                tc = tool_calls[0]
                detail = f"id={tc.get('id')}, fn={tc['function']['name']}"
            else:
                detail = f"finish_reason={choice.get('finish_reason')}, content={msg.get('content', '')[:80]}"
            return check("tool_calling", has_tool_call, detail)
    except Exception as e:
        return check("tool_calling", False, str(e))


def test_tool_call_id_normalization(endpoint: str, model: str) -> bool:
    """Test that non-compliant tool_call_ids (droid/claude style) are accepted."""
    test_ids = [
        ("droid_style", "obC25iz_0"),
        ("claude_style", "toulu_01ABCdefGH"),
        ("long_id", "chatcmpl-tool-abc123def456"),
    ]
    all_ok = True
    for label, bad_id in test_ids:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "user", "content": "Read test.py"},
                {
                    "role": "assistant", "content": "",
                    "tool_calls": [{
                        "id": bad_id, "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"test.py"}'},
                    }],
                },
                {
                    "role": "tool", "tool_call_id": bad_id,
                    "name": "read_file", "content": "print('hello')",
                },
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }],
            "max_tokens": 32,
        }).encode()
        try:
            req = urllib.request.Request(
                f"{endpoint}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                ok = "choices" in data
                if not ok:
                    all_ok = False
                    check(f"tool_id_{label}", False, json.dumps(data.get("error", {}))[:100])
        except urllib.error.HTTPError as e:
            all_ok = False
            body = e.read().decode()[:200]
            check(f"tool_id_{label}", False, body)
        except Exception as e:
            all_ok = False
            check(f"tool_id_{label}", False, str(e))

    return check("tool_call_id_normalization", all_ok,
                 f"tested {len(test_ids)} non-compliant ID formats")


def test_diff_capture(workspace_dir: str) -> bool:
    """Verify git is available and diff capture will work."""
    import subprocess
    try:
        proc = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
        ok = proc.returncode == 0
        return check("git_available", ok, proc.stdout.strip())
    except Exception as e:
        return check("git_available", False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Pre-flight smoke test")
    parser.add_argument("--endpoint", required=True, help="vLLM endpoint URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--workspace-dir", default="/mnt/nvme/sera-workspaces")
    args = parser.parse_args()

    print(f"Smoke test: {args.endpoint} / {args.model}")
    print(f"{'='*60}")

    start = time.monotonic()
    results = [
        test_health(args.endpoint),
        test_model_available(args.endpoint, args.model),
        test_basic_completion(args.endpoint, args.model),
        test_tool_calling(args.endpoint, args.model),
        test_tool_call_id_normalization(args.endpoint, args.model),
        test_diff_capture(args.workspace_dir),
    ]
    elapsed = time.monotonic() - start

    passed = sum(results)
    total = len(results)
    print(f"{'='*60}")
    print(f"Result: {passed}/{total} passed in {elapsed:.1f}s")

    if passed < total:
        print("\nFAILED — do not proceed with evaluation.")
        sys.exit(1)
    else:
        print("\nAll checks passed — safe to proceed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
