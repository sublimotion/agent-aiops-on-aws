#!/usr/bin/env python3
"""
BFCL-style tool-use evaluation for Qwen3-Next on SGLang.

Evaluates tool-calling capability across multiple categories:
  - Simple function calling (single tool, clear intent)
  - Multi-tool selection (choose correct tool from N options)
  - Parallel tool calls (invoke multiple tools in one turn)
  - Multi-turn tool use (sequential tool calls with results)
  - Structured output (complex nested arguments)

Scoring:
  - First-call success rate: Did the model call the RIGHT tool with VALID args?
  - Argument accuracy: Are required params present and correctly typed?
  - Multi-turn completion: Can the model chain tool results across turns?
  - Overall BFCL score: Weighted composite (maps to BFCL-V4 categories)

Usage:
  python3 bfcl-eval.py --endpoint http://localhost:30000 --model qwen3-next
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool Definitions (BFCL-V4 inspired categories)
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'Tokyo'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
            },
            "required": ["location"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results to return", "default": 5},
            },
            "required": ["query"],
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '2 * (3 + 4)'"},
            },
            "required": ["expression"],
        },
    },
}

FILE_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
            },
            "required": ["path"],
        },
    },
}

FILE_WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
}

RUN_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Execute a shell command and return stdout/stderr",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
            },
            "required": ["command"],
        },
    },
}

CREATE_PR_TOOL = {
    "type": "function",
    "function": {
        "name": "create_pull_request",
        "description": "Create a GitHub pull request",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "PR title"},
                "body": {"type": "string", "description": "PR description"},
                "base": {"type": "string", "description": "Base branch", "default": "main"},
                "head": {"type": "string", "description": "Head branch"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels to apply",
                },
            },
            "required": ["title", "head"],
        },
    },
}

DB_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Execute a SQL query against the database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query to execute"},
                "database": {"type": "string", "description": "Database name"},
                "params": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Query parameters for prepared statements",
                },
            },
            "required": ["query", "database"],
        },
    },
}

ALL_TOOLS = [
    WEATHER_TOOL, SEARCH_TOOL, CALCULATOR_TOOL,
    FILE_READ_TOOL, FILE_WRITE_TOOL, RUN_COMMAND_TOOL,
    CREATE_PR_TOOL, DB_QUERY_TOOL,
]

CODING_TOOLS = [FILE_READ_TOOL, FILE_WRITE_TOOL, RUN_COMMAND_TOOL, CREATE_PR_TOOL]

# ---------------------------------------------------------------------------
# Test Scenarios
# ---------------------------------------------------------------------------


def build_scenarios():
    """Build BFCL-style evaluation scenarios."""
    scenarios = []

    # --- Category 1: Simple Function Calling ---
    simple = [
        {
            "id": "simple_weather",
            "messages": [{"role": "user", "content": "What's the weather like in San Francisco?"}],
            "tools": [WEATHER_TOOL, SEARCH_TOOL, CALCULATOR_TOOL],
            "expected_tool": "get_weather",
            "expected_args": {"location": "San Francisco"},
            "required_args": ["location"],
        },
        {
            "id": "simple_search",
            "messages": [{"role": "user", "content": "Search for the latest Python 3.13 release notes."}],
            "tools": [WEATHER_TOOL, SEARCH_TOOL, CALCULATOR_TOOL],
            "expected_tool": "web_search",
            "expected_args": {},
            "required_args": ["query"],
        },
        {
            "id": "simple_calc",
            "messages": [{"role": "user", "content": "What is 2^10 + 3^7?"}],
            "tools": [WEATHER_TOOL, SEARCH_TOOL, CALCULATOR_TOOL],
            "expected_tool": "calculate",
            "expected_args": {},
            "required_args": ["expression"],
        },
        {
            "id": "simple_read_file",
            "messages": [{"role": "user", "content": "Read the contents of /etc/hostname"}],
            "tools": CODING_TOOLS,
            "expected_tool": "read_file",
            "expected_args": {"path": "/etc/hostname"},
            "required_args": ["path"],
        },
        {
            "id": "simple_run_cmd",
            "messages": [{"role": "user", "content": "Run 'ls -la /tmp' and show me the output."}],
            "tools": CODING_TOOLS,
            "expected_tool": "run_command",
            "expected_args": {},
            "required_args": ["command"],
        },
    ]
    for s in simple:
        s["category"] = "simple"
    scenarios.extend(simple)

    # --- Category 2: Multi-Tool Selection ---
    multi_select = [
        {
            "id": "select_file_not_search",
            "category": "multi_select",
            "messages": [
                {"role": "user", "content": "I have a config file at /app/config.yaml. Read it and tell me what port the server runs on."}
            ],
            "tools": ALL_TOOLS,
            "expected_tool": "read_file",
            "required_args": ["path"],
        },
        {
            "id": "select_db_not_file",
            "category": "multi_select",
            "messages": [
                {"role": "user", "content": "How many users signed up in the last 30 days? Check the 'analytics' database."}
            ],
            "tools": ALL_TOOLS,
            "expected_tool": "query_database",
            "required_args": ["query", "database"],
        },
        {
            "id": "select_write_not_cmd",
            "category": "multi_select",
            "messages": [
                {"role": "user", "content": "Create a new file at /app/README.md with the content '# My Project\\nA sample project.'"}
            ],
            "tools": CODING_TOOLS,
            "expected_tool": "write_file",
            "required_args": ["path", "content"],
        },
        {
            "id": "select_pr_workflow",
            "category": "multi_select",
            "messages": [
                {"role": "user", "content": "Create a pull request with title 'Fix login bug' from branch 'fix/login-redirect' to main."}
            ],
            "tools": ALL_TOOLS,
            "expected_tool": "create_pull_request",
            "required_args": ["title", "head"],
        },
    ]
    scenarios.extend(multi_select)

    # --- Category 3: Parallel Tool Calls ---
    parallel = [
        {
            "id": "parallel_weather_two_cities",
            "category": "parallel",
            "messages": [
                {"role": "user", "content": "What's the weather in both Tokyo and London right now?"}
            ],
            "tools": [WEATHER_TOOL],
            "expected_tool": "get_weather",
            "expected_parallel_count": 2,
            "required_args": ["location"],
        },
        {
            "id": "parallel_read_two_files",
            "category": "parallel",
            "messages": [
                {"role": "user", "content": "Read both /app/src/main.py and /app/tests/test_main.py so I can compare them."}
            ],
            "tools": [FILE_READ_TOOL],
            "expected_tool": "read_file",
            "expected_parallel_count": 2,
            "required_args": ["path"],
        },
    ]
    scenarios.extend(parallel)

    # --- Category 4: Multi-Turn Tool Use ---
    multi_turn = [
        {
            "id": "mt_read_then_modify",
            "category": "multi_turn",
            "turns": [
                {
                    "messages": [
                        {"role": "system", "content": "You are a coding assistant. Use tools to help the user."},
                        {"role": "user", "content": "Read the file /app/config.json and change the port from 3000 to 8080."},
                    ],
                    "expected_tool": "read_file",
                    "tool_result": '{"port": 3000, "host": "0.0.0.0", "debug": false}',
                },
                {
                    "expected_tool": "write_file",
                    "required_args": ["path", "content"],
                    "validate_content": lambda args: "8080" in args.get("content", ""),
                },
            ],
            "tools": CODING_TOOLS,
        },
        {
            "id": "mt_search_then_calc",
            "category": "multi_turn",
            "turns": [
                {
                    "messages": [
                        {"role": "user", "content": "Search for the population of Japan and South Korea, then calculate the total."},
                    ],
                    "expected_tool": "web_search",
                    "tool_result": "Japan: 125 million, South Korea: 52 million",
                },
                {
                    "expected_tool": "calculate",
                    "required_args": ["expression"],
                },
            ],
            "tools": [SEARCH_TOOL, CALCULATOR_TOOL],
        },
        {
            "id": "mt_run_test_then_fix",
            "category": "multi_turn",
            "turns": [
                {
                    "messages": [
                        {"role": "system", "content": "You are a coding assistant. Run tests and fix failures."},
                        {"role": "user", "content": "Run the test suite with 'pytest /app/tests/' and fix any failures."},
                    ],
                    "expected_tool": "run_command",
                    "tool_result": "FAILED tests/test_utils.py::test_parse_date - AssertionError: expected '2024-01-01' got None\n1 failed, 12 passed",
                },
                {
                    "expected_tool": "read_file",
                    "tool_result": 'def parse_date(s):\n    """Parse date string."""\n    return None  # TODO: implement\n',
                },
                {
                    "expected_tool": "write_file",
                    "required_args": ["path", "content"],
                },
            ],
            "tools": CODING_TOOLS,
        },
    ]
    scenarios.extend(multi_turn)

    # --- Category 5: Structured Output (Complex Args) ---
    structured = [
        {
            "id": "struct_pr_with_labels",
            "category": "structured",
            "messages": [
                {
                    "role": "user",
                    "content": "Create a PR titled 'Add caching layer' from branch 'feat/redis-cache' with labels 'enhancement' and 'performance'. The description should explain that we're adding Redis caching to reduce DB load.",
                }
            ],
            "tools": [CREATE_PR_TOOL],
            "expected_tool": "create_pull_request",
            "required_args": ["title", "head"],
            "validate_args": lambda args: (
                isinstance(args.get("labels"), list)
                and len(args.get("labels", [])) >= 1
                and args.get("body", "") != ""
            ),
        },
        {
            "id": "struct_db_with_params",
            "category": "structured",
            "messages": [
                {
                    "role": "user",
                    "content": "Query the 'users' database to find all users with email ending in '@example.com' who signed up after January 2024. Use parameterized queries.",
                }
            ],
            "tools": [DB_QUERY_TOOL],
            "expected_tool": "query_database",
            "required_args": ["query", "database"],
        },
    ]
    scenarios.extend(structured)

    return scenarios


# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    id: str
    category: str
    passed: bool
    tool_called: str | None = None
    expected_tool: str | None = None
    args_valid: bool = False
    error: str | None = None
    latency_ms: float = 0.0
    turns_completed: int = 0
    turns_total: int = 1
    details: dict = field(default_factory=dict)


async def call_chat(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
) -> dict:
    """Make a single chat completion request with tools."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    async with session.post(
        f"{endpoint}/v1/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
        return await resp.json()


def extract_tool_calls(response: dict) -> list[dict]:
    """Extract tool calls from chat completion response.

    Handles two formats:
    1. Standard OpenAI tool_calls array in the message
    2. Qwen3-style <tool_call> XML tags in the content field
       (SGLang's qwen3_coder parser may put tool calls in content
       instead of the tool_calls array)
    """
    import re

    msg = response.get("choices", [{}])[0].get("message", {})
    tool_calls = msg.get("tool_calls", [])

    # Standard path: tool_calls array is populated
    if tool_calls:
        return [
            {
                "id": tc.get("id", f"call_{i}"),
                "name": tc["function"]["name"],
                "arguments": json.loads(tc["function"]["arguments"])
                if isinstance(tc["function"]["arguments"], str)
                else tc["function"]["arguments"],
            }
            for i, tc in enumerate(tool_calls)
        ]

    # Fallback: parse <tool_call> tags from content
    content = msg.get("content", "") or ""
    tag_pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    matches = tag_pattern.findall(content)
    results = []
    for i, m in enumerate(matches):
        try:
            obj = json.loads(m)
            name = obj.get("name", "")
            arguments = obj.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            results.append({"id": f"call_{i}", "name": name, "arguments": arguments})
        except (json.JSONDecodeError, TypeError):
            continue
    return results


def validate_args(tool_call: dict, required_args: list[str]) -> bool:
    """Check that all required arguments are present and non-empty."""
    args = tool_call.get("arguments", {})
    for arg in required_args:
        if arg not in args or args[arg] is None or args[arg] == "":
            return False
    return True


async def eval_single_turn(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    scenario: dict,
) -> ScenarioResult:
    """Evaluate a single-turn scenario."""
    start = time.monotonic()
    try:
        resp = await call_chat(session, endpoint, model, scenario["messages"], scenario["tools"])
        latency = (time.monotonic() - start) * 1000

        tool_calls = extract_tool_calls(resp)
        if not tool_calls:
            return ScenarioResult(
                id=scenario["id"],
                category=scenario["category"],
                passed=False,
                error="No tool calls in response",
                latency_ms=latency,
            )

        first_call = tool_calls[0]
        correct_tool = first_call["name"] == scenario["expected_tool"]

        required = scenario.get("required_args", [])
        args_ok = validate_args(first_call, required)

        # Check parallel count if expected
        expected_parallel = scenario.get("expected_parallel_count")
        parallel_ok = True
        if expected_parallel:
            parallel_ok = len(tool_calls) >= expected_parallel

        # Custom validation
        custom_ok = True
        validator = scenario.get("validate_args")
        if validator and callable(validator):
            custom_ok = validator(first_call.get("arguments", {}))

        passed = correct_tool and args_ok and parallel_ok and custom_ok

        return ScenarioResult(
            id=scenario["id"],
            category=scenario["category"],
            passed=passed,
            tool_called=first_call["name"],
            expected_tool=scenario["expected_tool"],
            args_valid=args_ok,
            latency_ms=latency,
            details={
                "num_tool_calls": len(tool_calls),
                "parallel_ok": parallel_ok,
                "custom_ok": custom_ok,
                "args": first_call.get("arguments", {}),
            },
        )
    except Exception as e:
        return ScenarioResult(
            id=scenario["id"],
            category=scenario["category"],
            passed=False,
            error=str(e),
            latency_ms=(time.monotonic() - start) * 1000,
        )


async def eval_multi_turn(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    scenario: dict,
) -> ScenarioResult:
    """Evaluate a multi-turn scenario."""
    turns = scenario["turns"]
    tools = scenario["tools"]
    messages = list(turns[0]["messages"])
    turns_completed = 0

    last_tool_call_id = None  # Track actual ID from model response
    start = time.monotonic()
    try:
        for i, turn in enumerate(turns):
            if i > 0:
                # Append tool result from previous turn using the actual tool_call_id
                prev_result = turns[i - 1].get("tool_result", "OK")
                messages.append({
                    "role": "tool",
                    "content": prev_result,
                    "tool_call_id": last_tool_call_id or f"call_{i-1}",
                })

            resp = await call_chat(session, endpoint, model, messages, tools)
            tool_calls = extract_tool_calls(resp)

            if not tool_calls:
                return ScenarioResult(
                    id=scenario["id"],
                    category="multi_turn",
                    passed=False,
                    error=f"No tool call at turn {i+1}",
                    latency_ms=(time.monotonic() - start) * 1000,
                    turns_completed=turns_completed,
                    turns_total=len(turns),
                )

            first_call = tool_calls[0]
            expected = turn.get("expected_tool", turns[i]["expected_tool"])
            if first_call["name"] != expected:
                return ScenarioResult(
                    id=scenario["id"],
                    category="multi_turn",
                    passed=False,
                    tool_called=first_call["name"],
                    expected_tool=expected,
                    error=f"Turn {i+1}: expected {expected}, got {first_call['name']}",
                    latency_ms=(time.monotonic() - start) * 1000,
                    turns_completed=turns_completed,
                    turns_total=len(turns),
                )

            # Check required args
            required = turn.get("required_args", [])
            if required and not validate_args(first_call, required):
                return ScenarioResult(
                    id=scenario["id"],
                    category="multi_turn",
                    passed=False,
                    error=f"Turn {i+1}: missing required args {required}",
                    latency_ms=(time.monotonic() - start) * 1000,
                    turns_completed=turns_completed,
                    turns_total=len(turns),
                )

            # Custom content validation
            content_validator = turn.get("validate_content")
            if content_validator and callable(content_validator):
                if not content_validator(first_call.get("arguments", {})):
                    return ScenarioResult(
                        id=scenario["id"],
                        category="multi_turn",
                        passed=False,
                        error=f"Turn {i+1}: content validation failed",
                        latency_ms=(time.monotonic() - start) * 1000,
                        turns_completed=turns_completed,
                        turns_total=len(turns),
                    )

            turns_completed += 1

            # Use the actual tool_call ID from the model response
            last_tool_call_id = first_call.get("id", f"call_{i}")

            # Add the assistant's tool call to messages for next turn
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": last_tool_call_id,
                        "type": "function",
                        "function": {
                            "name": first_call["name"],
                            "arguments": json.dumps(first_call["arguments"]),
                        },
                    }
                ],
            })

        return ScenarioResult(
            id=scenario["id"],
            category="multi_turn",
            passed=True,
            turns_completed=turns_completed,
            turns_total=len(turns),
            latency_ms=(time.monotonic() - start) * 1000,
        )
    except Exception as e:
        return ScenarioResult(
            id=scenario["id"],
            category="multi_turn",
            passed=False,
            error=str(e),
            latency_ms=(time.monotonic() - start) * 1000,
            turns_completed=turns_completed,
            turns_total=len(turns),
        )


async def run_evaluation(
    endpoint: str,
    model: str,
    output_dir: str,
    num_scenarios: int = 200,
    max_concurrent: int = 4,
):
    """Run the full BFCL evaluation."""
    scenarios = build_scenarios()

    # Repeat scenarios to reach num_scenarios
    full_list: list[dict] = []
    while len(full_list) < num_scenarios:
        for s in scenarios:
            if len(full_list) >= num_scenarios:
                break
            full_list.append(s)

    log.info(f"Running {len(full_list)} scenarios ({len(scenarios)} unique) against {endpoint}")

    results: list[ScenarioResult] = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(scenario: dict) -> ScenarioResult:
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                if scenario.get("category") == "multi_turn":
                    return await eval_multi_turn(session, endpoint, model, scenario)
                else:
                    return await eval_single_turn(session, endpoint, model, scenario)

    tasks = [run_one(s) for s in full_list]
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        if completed % 20 == 0:
            passed = sum(1 for r in results if r.passed)
            log.info(f"Progress: {completed}/{len(full_list)} ({passed}/{completed} passed)")

    # --- Compute Summary ---
    by_category: dict[str, list[ScenarioResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    category_scores = {}
    for cat, cat_results in by_category.items():
        passed = sum(1 for r in cat_results if r.passed)
        total = len(cat_results)
        category_scores[cat] = {
            "passed": passed,
            "total": total,
            "score": passed / total if total > 0 else 0,
            "avg_latency_ms": sum(r.latency_ms for r in cat_results) / total if total > 0 else 0,
        }

    # BFCL-style weighted composite
    # Weights approximate BFCL-V4 category distribution
    weights = {
        "simple": 0.30,
        "multi_select": 0.20,
        "parallel": 0.10,
        "multi_turn": 0.25,
        "structured": 0.15,
    }
    weighted_score = 0.0
    total_weight = 0.0
    for cat, weight in weights.items():
        if cat in category_scores:
            weighted_score += weight * category_scores[cat]["score"]
            total_weight += weight
    overall_score = (weighted_score / total_weight * 100) if total_weight > 0 else 0

    # Multi-turn specific metrics
    mt_results = by_category.get("multi_turn", [])
    mt_completion = (
        sum(r.turns_completed for r in mt_results) / sum(r.turns_total for r in mt_results)
        if mt_results
        else 0
    )

    first_call_success = sum(1 for r in results if r.passed) / len(results) if results else 0

    summary = {
        "overall_score": round(overall_score, 1),
        "first_call_success_rate": round(first_call_success, 4),
        "multi_turn_completion_rate": round(mt_completion, 4),
        "total_scenarios": len(results),
        "total_passed": sum(1 for r in results if r.passed),
        "category_scores": category_scores,
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / len(results), 1) if results else 0,
        "model": model,
        "endpoint": endpoint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # --- Write Results ---
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "bfcl_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    detailed = [asdict(r) for r in results]
    # Remove non-serializable items
    for d in detailed:
        d.pop("details", None)
    with open(os.path.join(output_dir, "bfcl_detailed.json"), "w") as f:
        json.dump(detailed, f, indent=2)

    # Print report
    print("\n" + "=" * 60)
    print("  BFCL Tool-Use Evaluation Results")
    print("=" * 60)
    print(f"  Model:    {model}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Scenarios: {len(results)} ({sum(1 for r in results if r.passed)} passed)")
    print()
    print(f"  {'Category':<15} {'Passed':>8} {'Total':>8} {'Score':>8} {'Avg ms':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cat in ["simple", "multi_select", "parallel", "multi_turn", "structured"]:
        if cat in category_scores:
            cs = category_scores[cat]
            print(f"  {cat:<15} {cs['passed']:>8} {cs['total']:>8} {cs['score']*100:>7.1f}% {cs['avg_latency_ms']:>7.0f}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'OVERALL':<15} {summary['total_passed']:>8} {summary['total_scenarios']:>8} {overall_score:>7.1f}%")
    print()

    if overall_score >= 80:
        verdict = "STRONG — competitive with Claude Sonnet for tool orchestration"
    elif overall_score >= 75:
        verdict = "PROCEED — viable for both swarm and interactive coding agent"
    elif overall_score >= 70:
        verdict = "CAUTION — viable for swarms (retry at machine speed), not interactive"
    else:
        verdict = "STOP — model is not viable for coding agents"

    print(f"  Verdict: {verdict}")
    print(f"  Multi-turn completion: {mt_completion*100:.1f}%")
    print(f"  Avg latency: {summary['avg_latency_ms']:.0f}ms")
    print("=" * 60)
    print(f"\n  Results saved to: {output_dir}/")

    # Print failures for debugging
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n  Failed scenarios ({len(failures)}):")
        seen = set()
        for r in failures:
            if r.id not in seen:
                seen.add(r.id)
                err = r.error or f"called {r.tool_called}, expected {r.expected_tool}"
                print(f"    - {r.id}: {err}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="BFCL-style tool-use evaluation")
    parser.add_argument("--endpoint", required=True, help="SGLang endpoint URL")
    parser.add_argument("--model", default="qwen3-next", help="Model name")
    parser.add_argument("--output-dir", default="./results/s2_bfcl", help="Output directory")
    parser.add_argument("--num-scenarios", type=int, default=200, help="Number of scenarios to run")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent requests")
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            endpoint=args.endpoint,
            model=args.model,
            output_dir=args.output_dir,
            num_scenarios=args.num_scenarios,
            max_concurrent=args.max_concurrent,
        )
    )


if __name__ == "__main__":
    main()
