#!/usr/bin/env python3
"""
Kimi-K2.5 Benchmark Suite for p5e.48xlarge (8x H100)

Benchmarks designed to test:
1. Baseline throughput and latency
2. KV cache prefix caching effectiveness
3. Multi-turn conversation (LMCache-friendly workloads)
4. Reasoning-heavy workloads
5. Stress testing under high concurrency
"""

import asyncio
import aiohttp
import json
import time
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import random

# Kimi-K2.5 specific configuration
MODEL_ID = "moonshotai/Kimi-K2.5"
ENDPOINT = "http://localhost:30080"
MAX_MODEL_LEN = 32768
TENSOR_PARALLEL = 8

# Workloads optimized for Kimi-K2.5 with reasoning capabilities
WORKLOADS = {
    "reasoning_math": {
        "description": "Math reasoning problems - tests reasoning parser",
        "system_prompt": "You are a helpful math tutor. Show your step-by-step reasoning.",
        "prompts": [
            "If a car travels 60 mph for 2 hours, then 80 mph for 1.5 hours, what's the total distance?",
            "A store sells apples for $1.50 each. If I buy 6 apples with a 10% discount, how much do I pay?",
            "What is the sum of the first 10 prime numbers?",
            "A rectangle has perimeter 24cm. If length is twice width, what's the area?",
            "If 5x + 3 = 23, what is x?",
        ],
        "shared_context": "Previous context: We are practicing math problems step by step.",
        "max_tokens": 500,
    },
    "code_generation": {
        "description": "Code generation tasks",
        "system_prompt": "You are an expert programmer. Write clean, efficient code.",
        "prompts": [
            "Write a Python function to check if a string is a palindrome.",
            "Implement binary search in Python.",
            "Write a function to find the nth Fibonacci number.",
            "Create a Python class for a linked list with append and print methods.",
            "Write code to merge two sorted lists.",
        ],
        "shared_context": "",
        "max_tokens": 800,
    },
    "multi_turn_qa": {
        "description": "Multi-turn Q&A with long shared context (tests prefix caching)",
        "system_prompt": "You are a helpful assistant. Answer based on the context provided.",
        "prompts": [
            "What is the main topic discussed in the context?",
            "Can you summarize the key points?",
            "What are the implications mentioned?",
            "How does this relate to real-world applications?",
            "What conclusions can be drawn?",
        ],
        "shared_context": """Context Document: Understanding Large Language Models

Large Language Models (LLMs) represent a significant advancement in artificial intelligence.
These models are trained on vast amounts of text data and can perform various natural language tasks.

Key Characteristics:
1. Scale: Modern LLMs contain billions of parameters
2. Architecture: Most use the Transformer architecture with attention mechanisms
3. Training: Self-supervised learning on large text corpora
4. Fine-tuning: Models can be adapted to specific tasks

Applications include:
- Text generation and completion
- Question answering
- Code generation
- Translation
- Summarization

Challenges:
- Computational resources for training and inference
- Hallucination and factual accuracy
- Bias in training data
- Environmental impact

Recent developments focus on efficiency improvements like quantization,
pruning, and knowledge distillation to make LLMs more accessible.
""" * 5,  # ~2500 tokens of shared context
        "max_tokens": 300,
    },
    "long_context_rag": {
        "description": "Long context RAG (tests KV cache offloading benefits)",
        "system_prompt": "Answer questions using only the provided documents.",
        "prompts": [
            "According to the documents, what is the primary benefit?",
            "What limitations are mentioned?",
            "How does the implementation work?",
            "What are the recommended best practices?",
            "Compare the approaches discussed.",
        ],
        "shared_context": """
## Document Collection for Analysis

### Document 1: System Architecture
The distributed system architecture consists of multiple microservices communicating via
message queues. Each service is independently deployable and scalable. The architecture
follows the principle of loose coupling and high cohesion. Load balancing is achieved
through a combination of DNS-based and application-level strategies.

### Document 2: Performance Optimization
Performance optimization strategies include caching at multiple levels, connection pooling,
and asynchronous processing. The cache layer uses a combination of in-memory (Redis) and
distributed caching solutions. Query optimization involves proper indexing and query planning.

### Document 3: Security Considerations
Security is implemented through multiple layers including network segmentation, encryption
at rest and in transit, and role-based access control. Authentication uses OAuth 2.0 with
JWT tokens. Regular security audits and penetration testing are conducted.

### Document 4: Deployment Strategy
The deployment follows a blue-green strategy with automatic rollback capabilities.
Infrastructure is managed as code using Terraform. Container orchestration uses Kubernetes
with custom operators for application-specific requirements.

### Document 5: Monitoring and Observability
The observability stack includes metrics (Prometheus), logs (ELK stack), and traces (Jaeger).
Custom dashboards provide real-time visibility into system health. Alerting is configured
for SLO violations with PagerDuty integration.
""" * 8,  # ~4000 tokens of shared context
        "max_tokens": 400,
    },
    "agentic_tool_use": {
        "description": "Agentic workload with tool calling (tests tool-call-parser)",
        "system_prompt": """You are an AI assistant with access to the following tools:
- search(query): Search the web for information
- calculate(expression): Perform mathematical calculations
- get_weather(location): Get current weather for a location
- get_time(timezone): Get current time in a timezone

Use tools when needed to answer questions.""",
        "prompts": [
            "What's the weather in Tokyo?",
            "Calculate the compound interest on $1000 at 5% for 3 years",
            "Search for the latest news about AI",
            "What time is it in London?",
            "Calculate 15% tip on a $85.50 bill",
        ],
        "shared_context": "Previous tool results: search('AI news') -> 'Latest AI developments...'",
        "max_tokens": 400,
    },
}

# QPS levels for different test scenarios
QPS_LEVELS = {
    "low": 0.5,      # Light load - individual request latency focus
    "medium": 2.0,    # Moderate load - balanced test
    "high": 5.0,      # High load - throughput focus
    "stress": 10.0,   # Stress test - find breaking point
}


@dataclass
class RequestMetrics:
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    e2e_ms: float
    itl_ms: float
    success: bool
    has_reasoning: bool = False
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    model: str
    instance_type: str
    workload: str
    qps: float
    timestamp: str
    config: dict
    num_requests: int
    metrics: dict


async def make_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    messages: list,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> RequestMetrics:
    """Make a single request and measure metrics."""
    start_time = time.perf_counter()
    ttft = None
    tokens_received = 0
    token_times = []
    has_reasoning = False

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
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
                    prompt_tokens=0, completion_tokens=0, ttft_ms=0, e2e_ms=0, itl_ms=0,
                    success=False, error=f"HTTP {response.status}: {error_text[:200]}",
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
                        # Check for reasoning content (Kimi-K2.5 specific)
                        if delta.get("reasoning") or delta.get("reasoning_content"):
                            has_reasoning = True
                        if delta.get("content"):
                            now = time.perf_counter()
                            if ttft is None:
                                ttft = (now - start_time) * 1000
                            token_times.append(now)
                            tokens_received += 1
                except json.JSONDecodeError:
                    continue

            e2e_time = (time.perf_counter() - start_time) * 1000

            itl = 0
            if len(token_times) > 1:
                itls = [(token_times[i] - token_times[i-1]) * 1000
                        for i in range(1, len(token_times))]
                itl = statistics.mean(itls) if itls else 0

            return RequestMetrics(
                prompt_tokens=len(str(messages)) // 4,
                completion_tokens=tokens_received,
                ttft_ms=ttft or 0,
                e2e_ms=e2e_time,
                itl_ms=itl,
                success=True,
                has_reasoning=has_reasoning,
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
    warmup_requests: int = 10,
) -> list[RequestMetrics]:
    """Run a workload at specified QPS."""
    workload = WORKLOADS[workload_name]
    results = []

    def build_messages(prompt_idx: int) -> list:
        messages = [{"role": "system", "content": workload["system_prompt"]}]
        if workload["shared_context"]:
            messages.append({"role": "user", "content": workload["shared_context"]})
            messages.append({"role": "assistant", "content": "I understand the context. Please ask your question."})
        messages.append({"role": "user", "content": workload["prompts"][prompt_idx % len(workload["prompts"])]})
        return messages

    async with aiohttp.ClientSession() as session:
        # Warmup
        print(f"  Warmup: {warmup_requests} requests...")
        for i in range(warmup_requests):
            messages = build_messages(i)
            await make_request(session, endpoint, model, messages, workload.get("max_tokens", 512))
        print("  Warmup complete.")

        # Measurement
        print(f"  Running {num_requests} requests at {qps} QPS...")
        interval = 1.0 / qps

        for i in range(num_requests):
            start = time.perf_counter()
            messages = build_messages(i)
            result = await make_request(session, endpoint, model, messages, workload.get("max_tokens", 512))
            results.append(result)

            elapsed = time.perf_counter() - start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

            if (i + 1) % 10 == 0:
                success_rate = sum(1 for r in results if r.success) / len(results) * 100
                print(f"    Progress: {i + 1}/{num_requests} ({success_rate:.0f}% success)")

    return results


def calculate_percentiles(values: list[float], percentiles: list[int] = [50, 90, 95, 99]) -> dict:
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
    reasoning_count = sum(1 for r in successful if r.has_reasoning)

    total_tokens = sum(r.completion_tokens for r in successful)
    total_time = sum(r.e2e_ms for r in successful) / 1000

    return {
        "success_rate": len(successful) / len(results),
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "reasoning_responses": reasoning_count,
        "reasoning_rate": reasoning_count / len(successful) if successful else 0,
        "ttft_ms": {
            "mean": statistics.mean(ttft_values),
            "min": min(ttft_values),
            "max": max(ttft_values),
            **calculate_percentiles(ttft_values),
        },
        "itl_ms": {
            "mean": statistics.mean(itl_values) if itl_values else 0,
            **calculate_percentiles(itl_values),
        },
        "e2e_ms": {
            "mean": statistics.mean(e2e_values),
            "min": min(e2e_values),
            "max": max(e2e_values),
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


async def run_benchmark_suite(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    workloads: list[str] = None,
    qps_levels: list[str] = None,
    requests_per_workload: int = 30,
    warmup_requests: int = 10,
    output_dir: str = "results/kimi-k2.5-p5e",
):
    """Run the full benchmark suite."""
    if workloads is None:
        workloads = list(WORKLOADS.keys())
    if qps_levels is None:
        qps_levels = ["low", "medium", "high"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    print(f"\n{'#'*70}")
    print(f"# Kimi-K2.5 Benchmark Suite on p5e.48xlarge (8x H100)")
    print(f"# Endpoint: {endpoint}")
    print(f"# Model: {model}")
    print(f"# Workloads: {workloads}")
    print(f"# QPS Levels: {qps_levels}")
    print(f"# Requests per config: {requests_per_workload}")
    print(f"{'#'*70}\n")

    for workload_name in workloads:
        for qps_name in qps_levels:
            qps = QPS_LEVELS[qps_name]
            print(f"\n{'='*60}")
            print(f"Benchmark: {workload_name} @ {qps_name} ({qps} QPS)")
            print(f"Description: {WORKLOADS[workload_name]['description']}")
            print(f"{'='*60}")

            results = await run_workload(
                endpoint=endpoint,
                model=model,
                workload_name=workload_name,
                qps=qps,
                num_requests=requests_per_workload,
                warmup_requests=warmup_requests,
            )

            aggregated = aggregate_results(results)

            timestamp = datetime.now().isoformat()
            result = BenchmarkResult(
                model=model,
                instance_type="p5e.48xlarge",
                workload=workload_name,
                qps=qps,
                timestamp=timestamp,
                config={
                    "tensor_parallel": TENSOR_PARALLEL,
                    "max_model_len": MAX_MODEL_LEN,
                    "workload_config": WORKLOADS[workload_name],
                },
                num_requests=len(results),
                metrics=aggregated,
            )
            all_results.append(result)

            # Save individual result
            filename = f"kimi_k2.5_{workload_name}_{qps_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = Path(output_dir) / filename
            with open(filepath, "w") as f:
                json.dump(asdict(result), f, indent=2)
            print(f"\nSaved: {filepath}")

            # Print summary
            print(f"\n--- Results Summary ---")
            print(f"  Success Rate: {aggregated.get('success_rate', 0)*100:.1f}%")
            if 'reasoning_rate' in aggregated:
                print(f"  Reasoning Response Rate: {aggregated['reasoning_rate']*100:.1f}%")
            if 'ttft_ms' in aggregated:
                print(f"  TTFT p50/p90/p99: {aggregated['ttft_ms']['p50']:.1f} / {aggregated['ttft_ms']['p90']:.1f} / {aggregated['ttft_ms']['p99']:.1f} ms")
                print(f"  E2E p50/p90/p99: {aggregated['e2e_ms']['p50']:.1f} / {aggregated['e2e_ms']['p90']:.1f} / {aggregated['e2e_ms']['p99']:.1f} ms")
                print(f"  Throughput: {aggregated['throughput']['tokens_per_second']:.1f} tok/s")

            # Cooldown
            print("\nCooldown: 15s...")
            await asyncio.sleep(15)

    # Save combined results
    combined_path = Path(output_dir) / f"combined_kimi_k2.5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\n\nCombined results saved to: {combined_path}")

    return all_results


async def run_stress_test(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    duration_seconds: int = 120,
    target_qps: float = 10.0,
    output_dir: str = "results/kimi-k2.5-p5e",
):
    """Run stress test with high concurrency."""
    print(f"\n{'#'*70}")
    print(f"# STRESS TEST: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Duration: {duration_seconds}s @ {target_qps} QPS")
    print(f"{'#'*70}\n")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Simple prompts for stress testing
    prompts = [
        "What is 2+2?",
        "Name a color.",
        "Say hello.",
        "Count to 5.",
        "What day is it?",
    ]

    results = []
    start_time = time.perf_counter()
    interval = 1.0 / target_qps
    request_count = 0

    async with aiohttp.ClientSession() as session:
        while (time.perf_counter() - start_time) < duration_seconds:
            req_start = time.perf_counter()

            messages = [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": prompts[request_count % len(prompts)]},
            ]

            result = await make_request(session, endpoint, model, messages, max_tokens=50)
            results.append(result)
            request_count += 1

            elapsed = time.perf_counter() - req_start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

            if request_count % 50 == 0:
                elapsed_total = time.perf_counter() - start_time
                actual_qps = request_count / elapsed_total
                success_rate = sum(1 for r in results if r.success) / len(results) * 100
                print(f"  {request_count} requests, {elapsed_total:.0f}s elapsed, "
                      f"{actual_qps:.1f} QPS, {success_rate:.0f}% success")

    aggregated = aggregate_results(results)

    result = BenchmarkResult(
        model=model,
        instance_type="p5e.48xlarge",
        workload="stress_test",
        qps=target_qps,
        timestamp=datetime.now().isoformat(),
        config={
            "duration_seconds": duration_seconds,
            "target_qps": target_qps,
        },
        num_requests=len(results),
        metrics=aggregated,
    )

    filename = f"stress_test_{target_qps}qps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = Path(output_dir) / filename
    with open(filepath, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"\n--- Stress Test Results ---")
    print(f"  Total Requests: {len(results)}")
    print(f"  Duration: {duration_seconds}s")
    print(f"  Actual QPS: {len(results)/duration_seconds:.2f}")
    print(f"  Success Rate: {aggregated.get('success_rate', 0)*100:.1f}%")
    if 'ttft_ms' in aggregated:
        print(f"  TTFT p50/p99: {aggregated['ttft_ms']['p50']:.1f} / {aggregated['ttft_ms']['p99']:.1f} ms")
        print(f"  Throughput: {aggregated['throughput']['tokens_per_second']:.1f} tok/s")
    print(f"\nSaved: {filepath}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kimi-K2.5 Benchmark Suite")
    parser.add_argument("--mode", choices=["full", "quick", "stress"], default="quick",
                        help="Benchmark mode: full (all workloads), quick (subset), stress (stress test)")
    parser.add_argument("--endpoint", default=ENDPOINT, help="vLLM API endpoint")
    parser.add_argument("--requests", type=int, default=30, help="Requests per workload")
    parser.add_argument("--output", default="results/kimi-k2.5-p5e", help="Output directory")

    args = parser.parse_args()

    if args.mode == "full":
        asyncio.run(run_benchmark_suite(
            endpoint=args.endpoint,
            workloads=list(WORKLOADS.keys()),
            qps_levels=["low", "medium", "high"],
            requests_per_workload=args.requests,
            output_dir=args.output,
        ))
    elif args.mode == "quick":
        asyncio.run(run_benchmark_suite(
            endpoint=args.endpoint,
            workloads=["reasoning_math", "multi_turn_qa", "agentic_tool_use"],
            qps_levels=["low", "medium"],
            requests_per_workload=20,
            output_dir=args.output,
        ))
    elif args.mode == "stress":
        asyncio.run(run_stress_test(
            endpoint=args.endpoint,
            duration_seconds=120,
            target_qps=10.0,
            output_dir=args.output,
        ))
