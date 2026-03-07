#!/usr/bin/env python3
"""
Swarm concurrency simulator for coding agent workloads.

Simulates N concurrent coding agents, each running a multi-turn workflow:
  1. Send a coding request (prefill + generation)
  2. Wait for "tool execution" (simulated delay — compile, test, git, etc.)
  3. Send follow-up with tool results
  4. Repeat until task is "complete" (3-6 turns per agent)

This models real coding agent behavior where the GPU is idle during tool
execution, and multiple agents can share the same GPU by interleaving
their inference requests.

Key metrics:
  - Per-agent TTFT (should stay <10s even at high concurrency with prefix caching)
  - Aggregate throughput (tok/s across all agents)
  - GPU utilization at different concurrency levels
  - Memory pressure (via /metrics)
  - Failure rate (OOM, timeouts)

Usage:
  python3 swarm-simulator.py --endpoint http://localhost:30000 --num-agents 16 --duration 900
"""

import argparse
import asyncio
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict

try:
    import aiohttp
except ImportError:
    print("ERROR: aiohttp required. Install with: pip install aiohttp")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent Workload Templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior software engineer working on a Python web application.
You have access to tools for reading files, writing files, running commands, and running tests.
You are methodical: read before you edit, test after you change, and commit when done.
Always explain your reasoning before taking action."""

# Realistic coding agent task prompts with shared system prompt (tests prefix caching)
TASK_PROMPTS = [
    "There's a bug in the authentication middleware where expired JWT tokens are not being rejected. Read the auth module, identify the issue, and fix it.",
    "Add rate limiting to the /api/v1/upload endpoint. It should allow 10 requests per minute per user. Use Redis for the counter.",
    "The database migration for adding the 'preferences' JSON column to the users table is failing. Debug and fix the migration script.",
    "Refactor the payment processing module to use the Strategy pattern instead of the current if/else chain for different payment providers.",
    "Write comprehensive unit tests for the order processing pipeline. Cover edge cases like partial fulfillment, cancellation, and refunds.",
    "The search endpoint is slow (>2s p99). Profile the query, add appropriate database indexes, and implement query result caching.",
    "Implement a WebSocket endpoint for real-time notifications. Include connection management, heartbeat, and graceful disconnection.",
    "The CI pipeline is failing on the integration tests. The error is a race condition in the test setup. Investigate and fix.",
    "Add OpenTelemetry tracing to the API gateway. Instrument all outgoing HTTP calls and database queries with proper span attributes.",
    "The CSV export for the analytics dashboard is running out of memory on large datasets. Implement streaming export with chunked processing.",
    "Implement a circuit breaker for the external payment gateway API. Include exponential backoff and a fallback queue for failed transactions.",
    "Add RBAC (Role-Based Access Control) to the admin API. Define roles: viewer, editor, admin. Implement middleware and permission checks.",
]

# Simulated tool results that the model would receive between turns
TOOL_RESULTS = [
    "File contents:\n```python\ndef authenticate(token):\n    payload = jwt.decode(token, SECRET)\n    return payload\n```\nNote: no expiration check.",
    "Command output:\n$ pytest tests/test_auth.py -v\nFAILED test_expired_token - AssertionError: expected 401, got 200\n11 passed, 1 failed",
    "File written successfully: /app/src/middleware/rate_limiter.py (42 lines)",
    "Command output:\n$ python manage.py migrate\nApplying 0042_add_preferences... OK",
    "Command output:\n$ pytest tests/ -v --tb=short\n24 passed, 0 failed in 3.2s",
    "Command output:\n$ git diff --stat\n 3 files changed, 87 insertions(+), 23 deletions(-)",
    "Command output:\n$ curl -w '%{time_total}' localhost:8000/api/search?q=test\n0.087s (was 2.3s before index)",
    "File contents:\n```python\nclass PaymentProcessor:\n    def process(self, method, amount):\n        if method == 'stripe': ...\n        elif method == 'paypal': ...\n        elif method == 'crypto': ...\n```",
]


@dataclass
class RequestMetrics:
    agent_id: int
    turn: int
    ttft_ms: float
    total_ms: float
    output_tokens: int
    input_tokens: int  # estimated
    success: bool
    error: str | None = None
    timestamp: float = 0.0


@dataclass
class AgentMetrics:
    agent_id: int
    task: str
    turns_completed: int
    total_tokens_generated: int
    total_inference_ms: float
    total_tool_delay_ms: float
    requests: list[RequestMetrics] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GPU Utilization Sampling
# ---------------------------------------------------------------------------


async def sample_gpu_utilization() -> list[dict]:
    """Sample GPU utilization via nvidia-smi."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        gpus = []
        for line in stdout.decode().strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index": int(parts[0]),
                    "utilization_pct": float(parts[1]),
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                })
        return gpus
    except Exception:
        return []


async def gpu_monitor(
    interval: float, duration: float, samples: list[list[dict]], stop_event: asyncio.Event
):
    """Background task that samples GPU utilization at regular intervals."""
    start = time.monotonic()
    while not stop_event.is_set() and (time.monotonic() - start) < duration:
        gpu_data = await sample_gpu_utilization()
        if gpu_data:
            samples.append(gpu_data)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------
# Agent Simulation
# ---------------------------------------------------------------------------


async def simulate_agent(
    agent_id: int,
    endpoint: str,
    model: str,
    duration: float,
    tool_delay_min: float,
    tool_delay_max: float,
    start_time: float,
) -> AgentMetrics:
    """Simulate a single coding agent's multi-turn workflow."""
    task = random.choice(TASK_PROMPTS)
    turns_per_task = random.randint(3, 6)
    metrics = AgentMetrics(
        agent_id=agent_id,
        task=task[:80],
        turns_completed=0,
        total_tokens_generated=0,
        total_inference_ms=0,
        total_tool_delay_ms=0,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    async with aiohttp.ClientSession() as session:
        turn = 0
        while (time.monotonic() - start_time) < duration:
            turn += 1
            req_start = time.monotonic()
            ttft = None
            output_tokens = 0

            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": 512,
                    "temperature": 0.3,
                    "stream": True,
                }

                async with session.post(
                    f"{endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")

                    content_chunks = []
                    async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content"):
                                if ttft is None:
                                    ttft = (time.monotonic() - req_start) * 1000
                                content_chunks.append(delta["content"])
                                output_tokens += 1  # approximate: 1 chunk ≈ 1 token
                        except json.JSONDecodeError:
                            continue

                total_ms = (time.monotonic() - req_start) * 1000
                content = "".join(content_chunks)

                rm = RequestMetrics(
                    agent_id=agent_id,
                    turn=turn,
                    ttft_ms=ttft or total_ms,
                    total_ms=total_ms,
                    output_tokens=output_tokens,
                    input_tokens=len(json.dumps(messages)) // 4,  # rough estimate
                    success=True,
                    timestamp=time.monotonic() - start_time,
                )
                metrics.requests.append(rm)
                metrics.total_tokens_generated += output_tokens
                metrics.total_inference_ms += total_ms
                metrics.turns_completed += 1

                # Simulate tool execution delay
                tool_delay = random.uniform(tool_delay_min, tool_delay_max)
                metrics.total_tool_delay_ms += tool_delay * 1000
                await asyncio.sleep(tool_delay)

                # Add assistant response and simulated tool result for next turn
                messages.append({"role": "assistant", "content": content})
                tool_result = random.choice(TOOL_RESULTS)
                messages.append({"role": "user", "content": f"Tool result:\n{tool_result}\n\nContinue with the next step."})

                # Start a new task after completing turns_per_task
                if turn >= turns_per_task:
                    turn = 0
                    turns_per_task = random.randint(3, 6)
                    task = random.choice(TASK_PROMPTS)
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": task},
                    ]
                    metrics.task = task[:80]

            except Exception as e:
                total_ms = (time.monotonic() - req_start) * 1000
                metrics.errors.append(f"Turn {turn}: {str(e)[:200]}")
                metrics.requests.append(RequestMetrics(
                    agent_id=agent_id,
                    turn=turn,
                    ttft_ms=0,
                    total_ms=total_ms,
                    output_tokens=0,
                    input_tokens=0,
                    success=False,
                    error=str(e)[:200],
                    timestamp=time.monotonic() - start_time,
                ))
                # Brief backoff on error, then retry with fresh task
                await asyncio.sleep(5)
                turn = 0
                task = random.choice(TASK_PROMPTS)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                ]

    return metrics


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------


async def run_swarm(
    endpoint: str,
    model: str,
    num_agents: int,
    duration: float,
    tool_delay_min: float,
    tool_delay_max: float,
    output_dir: str,
):
    """Run swarm simulation with N concurrent agents."""
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Starting swarm: {num_agents} agents, {duration}s duration, tool delay {tool_delay_min}-{tool_delay_max}s")
    log.info(f"Endpoint: {endpoint}, Model: {model}")

    # Start GPU monitor
    gpu_samples: list[list[dict]] = []
    stop_gpu = asyncio.Event()
    gpu_task = asyncio.create_task(gpu_monitor(5.0, duration + 60, gpu_samples, stop_gpu))

    # Stagger agent starts (1s apart) to avoid thundering herd
    start_time = time.monotonic()
    agent_tasks = []
    for i in range(num_agents):
        await asyncio.sleep(min(1.0, duration / num_agents / 2))
        task = asyncio.create_task(
            simulate_agent(i, endpoint, model, duration, tool_delay_min, tool_delay_max, start_time)
        )
        agent_tasks.append(task)

    # Wait for all agents
    agent_results: list[AgentMetrics] = await asyncio.gather(*agent_tasks)

    # Stop GPU monitor
    stop_gpu.set()
    await gpu_task

    wall_time = time.monotonic() - start_time

    # --- Aggregate Results ---
    all_requests = [r for a in agent_results for r in a.requests]
    successful = [r for r in all_requests if r.success]
    failed = [r for r in all_requests if not r.success]

    ttfts = sorted([r.ttft_ms for r in successful]) if successful else [0]
    total_tokens = sum(r.output_tokens for r in successful)
    total_inference = sum(a.total_inference_ms for a in agent_results) / 1000  # seconds

    # GPU utilization stats
    all_gpu_utils = [g["utilization_pct"] for sample in gpu_samples for g in sample]
    avg_gpu_util = sum(all_gpu_utils) / len(all_gpu_utils) if all_gpu_utils else 0
    max_gpu_util = max(all_gpu_utils) if all_gpu_utils else 0

    # Memory stats
    all_mem_used = [g["memory_used_mb"] for sample in gpu_samples for g in sample]
    avg_mem_used = sum(all_mem_used) / len(all_mem_used) if all_mem_used else 0
    max_mem_used = max(all_mem_used) if all_mem_used else 0

    def percentile(data, p):
        if not data:
            return 0
        k = (len(data) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])

    summary = {
        "num_agents": num_agents,
        "duration_s": round(wall_time, 1),
        "total_requests": len(all_requests),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "failure_rate_pct": round(len(failed) / len(all_requests) * 100, 2) if all_requests else 0,
        "total_tokens_generated": total_tokens,
        "aggregate_throughput_tok_s": round(total_tokens / wall_time, 1) if wall_time > 0 else 0,
        "ttft_ms": {
            "p50": round(percentile(ttfts, 50), 1),
            "p90": round(percentile(ttfts, 90), 1),
            "p95": round(percentile(ttfts, 95), 1),
            "p99": round(percentile(ttfts, 99), 1),
            "max": round(ttfts[-1], 1) if ttfts else 0,
        },
        "avg_gpu_utilization_pct": round(avg_gpu_util, 1),
        "max_gpu_utilization_pct": round(max_gpu_util, 1),
        "avg_gpu_memory_used_mb": round(avg_mem_used, 0),
        "max_gpu_memory_used_mb": round(max_mem_used, 0),
        "turns_completed": sum(a.turns_completed for a in agent_results),
        "avg_turns_per_agent": round(
            sum(a.turns_completed for a in agent_results) / num_agents, 1
        ),
        "tool_delay_range": f"{tool_delay_min}-{tool_delay_max}s",
        "model": model,
        "endpoint": endpoint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Write results
    with open(os.path.join(output_dir, f"{num_agents}_agents_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    detailed = {
        "summary": summary,
        "per_agent": [
            {
                "agent_id": a.agent_id,
                "task": a.task,
                "turns_completed": a.turns_completed,
                "total_tokens": a.total_tokens_generated,
                "total_inference_ms": round(a.total_inference_ms, 1),
                "total_tool_delay_ms": round(a.total_tool_delay_ms, 1),
                "errors": a.errors,
                "requests": [asdict(r) for r in a.requests],
            }
            for a in agent_results
        ],
        "gpu_samples": [
            [{"index": g["index"], "util": g["utilization_pct"], "mem_mb": g["memory_used_mb"]}
             for g in sample]
            for sample in gpu_samples[::6]  # every 30s sample (5s interval × 6)
        ],
    }
    with open(os.path.join(output_dir, f"{num_agents}_agents_detailed.json"), "w") as f:
        json.dump(detailed, f, indent=2)

    # Print report
    print(f"\n{'='*60}")
    print(f"  Swarm Simulation Results: {num_agents} Agents")
    print(f"{'='*60}")
    print(f"  Duration:       {wall_time:.0f}s")
    print(f"  Requests:       {len(successful)} ok / {len(failed)} failed ({summary['failure_rate_pct']}% fail)")
    print(f"  Turns:          {summary['turns_completed']} ({summary['avg_turns_per_agent']} avg/agent)")
    print(f"  Throughput:     {summary['aggregate_throughput_tok_s']} tok/s aggregate")
    print(f"  TTFT p50/p95:   {summary['ttft_ms']['p50']:.0f}ms / {summary['ttft_ms']['p95']:.0f}ms")
    print(f"  TTFT p99/max:   {summary['ttft_ms']['p99']:.0f}ms / {summary['ttft_ms']['max']:.0f}ms")
    print(f"  GPU util avg:   {summary['avg_gpu_utilization_pct']:.0f}%")
    print(f"  GPU util max:   {summary['max_gpu_utilization_pct']:.0f}%")
    print(f"  GPU mem avg:    {summary['avg_gpu_memory_used_mb']:.0f} MB")
    print(f"  GPU mem max:    {summary['max_gpu_memory_used_mb']:.0f} MB")
    print(f"  Tool delay:     {tool_delay_min}-{tool_delay_max}s")

    # Pass/fail checks from spec
    ttft_p50 = summary["ttft_ms"]["p50"]
    gpu_util = summary["avg_gpu_utilization_pct"]
    fail_rate = summary["failure_rate_pct"]

    print(f"\n  Spec Checks:")
    print(f"    TTFT p50 < 10s:     {'PASS' if ttft_p50 < 10000 else 'FAIL'} ({ttft_p50:.0f}ms)")
    print(f"    GPU util >= 80%:    {'PASS' if gpu_util >= 80 else f'INFO ({gpu_util:.0f}%)'} (at {num_agents} agents)")
    print(f"    Zero failures:      {'PASS' if fail_rate == 0 else f'FAIL ({len(failed)} failures)'}")
    print(f"{'='*60}")
    print(f"  Results: {output_dir}/{num_agents}_agents_*.json")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Swarm concurrency simulator for coding agents")
    parser.add_argument("--endpoint", required=True, help="SGLang endpoint URL")
    parser.add_argument("--model", default="qwen3-next", help="Model name")
    parser.add_argument("--num-agents", type=int, default=8, help="Number of concurrent agents")
    parser.add_argument("--duration", type=int, default=900, help="Test duration in seconds")
    parser.add_argument("--tool-delay-min", type=float, default=5, help="Min tool execution delay (seconds)")
    parser.add_argument("--tool-delay-max", type=float, default=30, help="Max tool execution delay (seconds)")
    parser.add_argument("--output-dir", default="./results/s4_swarm", help="Output directory")
    args = parser.parse_args()

    asyncio.run(
        run_swarm(
            endpoint=args.endpoint,
            model=args.model,
            num_agents=args.num_agents,
            duration=args.duration,
            tool_delay_min=args.tool_delay_min,
            tool_delay_max=args.tool_delay_max,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
