#!/usr/bin/env python3
"""
vLLM KV Cache Benchmark Runner

Runs benchmarks following the spec in specs/vllm-kv-cache-benchmark.md
Measures TTFT, ITL, E2E latency at different QPS levels.
"""

import argparse
import asyncio
import json
import time
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import aiohttp
import random

# Workload definitions
WORKLOADS = {
    "synthetic": {
        "description": "Multi-round QA with shared prefix",
        "system_prompt": "You are a helpful assistant. Answer questions concisely.",
        "prompts": [
            "What is the capital of France?",
            "What is the population of Paris?",
            "What are the main tourist attractions in Paris?",
            "How do I get from the Eiffel Tower to the Louvre?",
            "What is the best time to visit Paris?",
        ],
        "shared_context": "Context: We are discussing travel to France, specifically Paris.",
    },
    "rag": {
        "description": "Long context retrieval with short output",
        "system_prompt": "Answer based only on the provided context.",
        "prompts": [
            "What is the main topic discussed?",
            "Summarize the key points in one sentence.",
            "What conclusions can be drawn?",
        ],
        "shared_context": """Retrieved Documents:
Document 1: Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Document 2: Deep learning uses neural networks with multiple layers to model complex patterns.
Document 3: Natural language processing allows computers to understand human language.
Document 4: Computer vision enables machines to interpret visual information from images.
Document 5: Reinforcement learning trains agents through rewards and penalties.
""" * 10,  # Repeat to create longer context
    },
    "agentic": {
        "description": "Multi-turn agent conversation with tool calls",
        "system_prompt": "You are an AI assistant that can use tools. Available tools: search, calculate, weather.",
        "prompts": [
            "Search for the weather in New York",
            "Calculate 15% tip on $85.50",
            "Search for restaurants near Times Square",
            "What's the weather like in Los Angeles?",
            "Calculate the total cost of 3 tickets at $45 each plus tax",
        ],
        "shared_context": "Previous tool results: search('weather NYC') -> 'Sunny, 72F'",
    },
}

QPS_LEVELS = {
    "low": 0.5,
    "medium": 2.0,
    "high": 4.0,
}


@dataclass
class RequestMetrics:
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float  # Time to first token
    e2e_ms: float   # End-to-end latency
    itl_ms: float   # Inter-token latency (avg)
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    baseline: str
    workload: str
    qps: float
    timestamp: str
    num_requests: int
    metrics: dict


async def make_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    messages: list,
    max_tokens: int = 512,
) -> RequestMetrics:
    """Make a single request and measure metrics."""
    start_time = time.perf_counter()
    ttft = None
    tokens_received = 0
    token_times = []

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    try:
        async with session.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                return RequestMetrics(
                    prompt_tokens=0,
                    completion_tokens=0,
                    ttft_ms=0,
                    e2e_ms=0,
                    itl_ms=0,
                    success=False,
                    error=f"HTTP {response.status}: {error_text[:200]}",
                )

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break

                try:
                    data = json.loads(line[6:])
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        if delta.get("content"):
                            now = time.perf_counter()
                            if ttft is None:
                                ttft = (now - start_time) * 1000
                            token_times.append(now)
                            tokens_received += 1
                except json.JSONDecodeError:
                    continue

            e2e_time = (time.perf_counter() - start_time) * 1000

            # Calculate ITL
            itl = 0
            if len(token_times) > 1:
                itls = [(token_times[i] - token_times[i-1]) * 1000
                        for i in range(1, len(token_times))]
                itl = statistics.mean(itls) if itls else 0

            return RequestMetrics(
                prompt_tokens=len(str(messages)) // 4,  # Rough estimate
                completion_tokens=tokens_received,
                ttft_ms=ttft or 0,
                e2e_ms=e2e_time,
                itl_ms=itl,
                success=True,
            )

    except asyncio.TimeoutError:
        return RequestMetrics(
            prompt_tokens=0, completion_tokens=0, ttft_ms=0, e2e_ms=0, itl_ms=0,
            success=False, error="Request timeout",
        )
    except Exception as e:
        return RequestMetrics(
            prompt_tokens=0, completion_tokens=0, ttft_ms=0, e2e_ms=0, itl_ms=0,
            success=False, error=str(e),
        )


async def run_workload(
    endpoint: str,
    model: str,
    workload_name: str,
    qps: float,
    num_requests: int,
    warmup_requests: int = 30,
) -> list[RequestMetrics]:
    """Run a workload at specified QPS."""
    workload = WORKLOADS[workload_name]
    results = []

    # Build messages with shared context for prefix caching benefit
    def build_messages(prompt_idx: int) -> list:
        messages = [
            {"role": "system", "content": workload["system_prompt"]},
            {"role": "user", "content": workload["shared_context"]},
            {"role": "assistant", "content": "I understand the context. How can I help?"},
            {"role": "user", "content": workload["prompts"][prompt_idx % len(workload["prompts"])]},
        ]
        return messages

    async with aiohttp.ClientSession() as session:
        # Warmup phase
        print(f"  Warmup: {warmup_requests} requests...")
        warmup_tasks = []
        for i in range(warmup_requests):
            messages = build_messages(i)
            warmup_tasks.append(make_request(session, endpoint, model, messages))
        await asyncio.gather(*warmup_tasks)
        print("  Warmup complete.")

        # Measurement phase
        print(f"  Running {num_requests} requests at {qps} QPS...")
        interval = 1.0 / qps

        for i in range(num_requests):
            start = time.perf_counter()
            messages = build_messages(i)
            result = await make_request(session, endpoint, model, messages)
            results.append(result)

            # Rate limiting
            elapsed = time.perf_counter() - start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

            if (i + 1) % 10 == 0:
                print(f"    Progress: {i + 1}/{num_requests}")

    return results


def calculate_percentiles(values: list[float], percentiles: list[int] = [50, 90, 99]) -> dict:
    """Calculate percentiles for a list of values."""
    if not values:
        return {f"p{p}": 0 for p in percentiles}

    sorted_values = sorted(values)
    result = {}
    for p in percentiles:
        idx = int(len(sorted_values) * p / 100)
        idx = min(idx, len(sorted_values) - 1)
        result[f"p{p}"] = sorted_values[idx]
    return result


def aggregate_results(results: list[RequestMetrics]) -> dict:
    """Aggregate metrics from multiple requests."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        return {
            "success_rate": 0,
            "total_requests": len(results),
            "failed_requests": len(failed),
            "errors": [r.error for r in failed[:5]],
        }

    ttft_values = [r.ttft_ms for r in successful]
    itl_values = [r.itl_ms for r in successful if r.itl_ms > 0]
    e2e_values = [r.e2e_ms for r in successful]

    total_tokens = sum(r.completion_tokens for r in successful)
    total_time = sum(r.e2e_ms for r in successful) / 1000  # Convert to seconds

    return {
        "success_rate": len(successful) / len(results),
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "ttft_ms": {
            "mean": statistics.mean(ttft_values),
            **calculate_percentiles(ttft_values),
        },
        "itl_ms": {
            "mean": statistics.mean(itl_values) if itl_values else 0,
            **calculate_percentiles(itl_values),
        },
        "e2e_ms": {
            "mean": statistics.mean(e2e_values),
            **calculate_percentiles(e2e_values),
        },
        "throughput": {
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "requests_per_second": len(successful) / total_time if total_time > 0 else 0,
        },
        "tokens": {
            "total_completion": total_tokens,
            "avg_completion": statistics.mean([r.completion_tokens for r in successful]),
        },
    }


async def run_benchmark(
    endpoint: str,
    model: str,
    baseline: str,
    workloads: list[str],
    qps_levels: list[str],
    requests_per_run: int = 50,
    runs_per_config: int = 5,
    warmup_requests: int = 30,
    output_dir: str = "results",
) -> list[BenchmarkResult]:
    """Run full benchmark suite."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    for workload_name in workloads:
        for qps_name in qps_levels:
            qps = QPS_LEVELS[qps_name]
            print(f"\n{'='*60}")
            print(f"Benchmark: {baseline} / {workload_name} / {qps_name} ({qps} QPS)")
            print(f"{'='*60}")

            run_results = []
            for run in range(runs_per_config):
                print(f"\nRun {run + 1}/{runs_per_config}")
                results = await run_workload(
                    endpoint=endpoint,
                    model=model,
                    workload_name=workload_name,
                    qps=qps,
                    num_requests=requests_per_run,
                    warmup_requests=warmup_requests if run == 0 else 5,
                )
                run_results.extend(results)

                # Cooldown between runs
                if run < runs_per_config - 1:
                    print("  Cooldown: 10s...")
                    await asyncio.sleep(10)

            # Aggregate all runs
            aggregated = aggregate_results(run_results)

            timestamp = datetime.now().isoformat()
            result = BenchmarkResult(
                baseline=baseline,
                workload=workload_name,
                qps=qps,
                timestamp=timestamp,
                num_requests=len(run_results),
                metrics=aggregated,
            )
            all_results.append(result)

            # Save individual result
            filename = f"{baseline}_{workload_name}_{qps_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = Path(output_dir) / filename
            with open(filepath, "w") as f:
                json.dump(asdict(result), f, indent=2)
            print(f"\nSaved: {filepath}")

            # Print summary
            print(f"\nResults Summary:")
            print(f"  Success Rate: {aggregated.get('success_rate', 0)*100:.1f}%")
            if 'ttft_ms' in aggregated:
                print(f"  TTFT p50/p90/p99: {aggregated['ttft_ms']['p50']:.1f} / {aggregated['ttft_ms']['p90']:.1f} / {aggregated['ttft_ms']['p99']:.1f} ms")
                print(f"  E2E p50/p90/p99: {aggregated['e2e_ms']['p50']:.1f} / {aggregated['e2e_ms']['p90']:.1f} / {aggregated['e2e_ms']['p99']:.1f} ms")
                print(f"  Throughput: {aggregated['throughput']['tokens_per_second']:.1f} tok/s")

            # Cooldown between configs
            print("\nCooldown: 30s...")
            await asyncio.sleep(30)

    return all_results


async def main():
    parser = argparse.ArgumentParser(description="vLLM KV Cache Benchmark")
    parser.add_argument("--endpoint", default="http://localhost:30080", help="vLLM API endpoint")
    parser.add_argument("--model", default=None, help="Model name (auto-detected if not specified)")
    parser.add_argument("--baseline", default="vllm-baseline", help="Baseline configuration name")
    parser.add_argument("--workloads", nargs="+", default=["synthetic", "rag", "agentic"],
                        choices=list(WORKLOADS.keys()), help="Workloads to run")
    parser.add_argument("--qps", nargs="+", default=["low", "medium", "high"],
                        choices=list(QPS_LEVELS.keys()), help="QPS levels to test")
    parser.add_argument("--requests", type=int, default=50, help="Requests per run")
    parser.add_argument("--runs", type=int, default=5, help="Runs per configuration")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup requests")
    parser.add_argument("--output", default="results", help="Output directory")

    args = parser.parse_args()

    # Auto-detect model if not specified
    model = args.model
    if not model:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.endpoint}/v1/models") as resp:
                data = await resp.json()
                model = data["data"][0]["id"]
                print(f"Auto-detected model: {model}")

    print(f"\n{'#'*60}")
    print(f"# vLLM KV Cache Benchmark")
    print(f"# Endpoint: {args.endpoint}")
    print(f"# Model: {model}")
    print(f"# Baseline: {args.baseline}")
    print(f"# Workloads: {args.workloads}")
    print(f"# QPS Levels: {args.qps}")
    print(f"# Requests/run: {args.requests}, Runs: {args.runs}")
    print(f"{'#'*60}")

    results = await run_benchmark(
        endpoint=args.endpoint,
        model=model,
        baseline=args.baseline,
        workloads=args.workloads,
        qps_levels=args.qps,
        requests_per_run=args.requests,
        runs_per_config=args.runs,
        warmup_requests=args.warmup,
        output_dir=args.output,
    )

    # Save combined results
    combined_path = Path(args.output) / f"combined_{args.baseline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n\nCombined results saved to: {combined_path}")


if __name__ == "__main__":
    asyncio.run(main())
