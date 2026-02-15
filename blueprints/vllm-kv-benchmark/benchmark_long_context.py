#!/usr/bin/env python3
"""
Long Context Benchmark for KV Cache Offloading

Tests 16K+ token contexts to stress KV cache offloading with LMCache + FSx.
Measures TTFT improvement when prefixes are cached on FSx vs recomputed.
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

ENDPOINT = "http://localhost:30080"
MODEL_ID = "moonshotai/Kimi-K2.5"

# Generate long context documents
def generate_document(doc_num: int, paragraphs: int = 20) -> str:
    """Generate a synthetic document with specified length."""
    topics = [
        ("distributed systems", "microservices", "load balancing", "fault tolerance"),
        ("machine learning", "neural networks", "deep learning", "transformers"),
        ("cloud computing", "containerization", "kubernetes", "serverless"),
        ("data engineering", "streaming", "batch processing", "ETL pipelines"),
        ("security", "encryption", "authentication", "zero trust"),
    ]
    topic = topics[doc_num % len(topics)]

    paragraphs_text = []
    for i in range(paragraphs):
        paragraphs_text.append(f"""
Section {i+1}: {topic[i % len(topic)].title()}

This section discusses {topic[i % len(topic)]} in detail. The implementation involves
multiple components working together to achieve the desired outcome. Key considerations
include performance optimization, scalability, and maintainability of the system.

The architecture follows established patterns for {topic[i % len(topic)]}. Best practices
recommend using proven approaches that have been validated in production environments.
Testing and monitoring are essential for ensuring system reliability and performance.

Integration with other components requires careful attention to interface design and
error handling. The system should gracefully handle failures and provide meaningful
feedback to operators. Logging and tracing help with debugging and performance analysis.
""")

    return f"""
# Document {doc_num}: {topic[0].title()} Architecture Guide

## Overview
This comprehensive guide covers the implementation of {topic[0]} systems,
including design patterns, best practices, and operational considerations.

{''.join(paragraphs_text)}

## Conclusion
Implementing {topic[0]} requires careful consideration of multiple factors.
This guide provides a foundation for building robust and scalable systems.
"""


# Long context workloads (16K, 20K, 24K tokens)
WORKLOADS = {
    "context_16k": {
        "description": "16K token shared context - tests KV cache offloading",
        "context_tokens": 16000,
        "num_documents": 8,
        "paragraphs_per_doc": 15,
        "system_prompt": "You are an expert technical consultant. Answer based only on the provided documents.",
        "prompts": [
            "What are the main architectural patterns discussed across the documents?",
            "Summarize the key best practices mentioned in the documentation.",
            "What are the common challenges and how do the documents address them?",
            "Compare the approaches discussed in different sections.",
            "What recommendations would you give based on this documentation?",
        ],
        "max_tokens": 300,
    },
    "context_20k": {
        "description": "20K token shared context - heavy KV cache pressure",
        "context_tokens": 20000,
        "num_documents": 10,
        "paragraphs_per_doc": 18,
        "system_prompt": "You are a senior architect reviewing technical documentation. Provide detailed analysis.",
        "prompts": [
            "Analyze the overall system architecture described in the documents.",
            "What are the trade-offs discussed in the documentation?",
            "Identify any gaps or missing considerations in the documentation.",
            "How do the different components interact based on the documents?",
            "What would you prioritize for implementation based on this information?",
        ],
        "max_tokens": 400,
    },
    "context_24k": {
        "description": "24K token shared context - stress test for KV cache",
        "context_tokens": 24000,
        "num_documents": 12,
        "paragraphs_per_doc": 20,
        "system_prompt": "You are a technical lead conducting a comprehensive review. Provide thorough analysis.",
        "prompts": [
            "Provide a comprehensive summary of all the documentation.",
            "What are the critical success factors mentioned across all documents?",
            "Identify potential risks and mitigation strategies from the documentation.",
            "How would you structure an implementation plan based on this information?",
            "What metrics and monitoring approaches are recommended?",
        ],
        "max_tokens": 500,
    },
    "context_28k": {
        "description": "28K token shared context - extreme KV cache stress",
        "context_tokens": 28000,
        "num_documents": 14,
        "paragraphs_per_doc": 18,
        "system_prompt": "You are an expert consultant. Be very concise.",
        "prompts": [
            "What are the key themes?",
            "Summarize the recommendations.",
            "What are the main decisions?",
            "Identify the top risks.",
            "What would you prioritize?",
        ],
        "max_tokens": 200,
    },
    "context_30k": {
        "description": "30K token shared context - near max context limit",
        "context_tokens": 30000,
        "num_documents": 15,
        "paragraphs_per_doc": 18,
        "system_prompt": "Be very concise.",
        "prompts": [
            "What is the focus?",
            "List main topics.",
            "Key takeaways?",
            "One sentence summary.",
            "What is missing?",
        ],
        "max_tokens": 150,
    },
}

QPS_LEVELS = {
    "very_low": 0.25,  # Long context needs lower QPS
    "low": 0.5,
    "medium": 1.0,
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
    context_size: int
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
        "temperature": 0.7,
        "stream": True,
    }

    try:
        async with session.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),  # Longer timeout for long context
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
            success=False, error="Request timeout (600s)",
        )
    except Exception as e:
        return RequestMetrics(
            prompt_tokens=0, completion_tokens=0, ttft_ms=0, e2e_ms=0, itl_ms=0,
            success=False, error=str(e),
        )


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


async def run_long_context_workload(
    endpoint: str,
    model: str,
    workload_name: str,
    qps: float,
    num_requests: int,
    warmup_requests: int = 5,
) -> tuple[list[RequestMetrics], str]:
    """Run a long context workload at specified QPS."""
    workload = WORKLOADS[workload_name]
    results = []

    # Generate the shared long context
    print(f"  Generating {workload['num_documents']} documents...")
    documents = []
    for i in range(workload["num_documents"]):
        doc = generate_document(i, workload["paragraphs_per_doc"])
        documents.append(doc)

    shared_context = "\n\n---\n\n".join(documents)
    estimated_tokens = len(shared_context) // 4
    print(f"  Shared context: ~{estimated_tokens:,} tokens")

    def build_messages(prompt_idx: int) -> list:
        return [
            {"role": "system", "content": workload["system_prompt"]},
            {"role": "user", "content": f"Documents:\n\n{shared_context}"},
            {"role": "assistant", "content": "I have reviewed all the provided documentation. Please ask your question."},
            {"role": "user", "content": workload["prompts"][prompt_idx % len(workload["prompts"])]},
        ]

    async with aiohttp.ClientSession() as session:
        # Warmup phase - critical for KV cache population
        print(f"  Warmup: {warmup_requests} requests (populating KV cache)...")
        warmup_results = []
        for i in range(warmup_requests):
            messages = build_messages(i)
            result = await make_request(session, endpoint, model, messages, workload.get("max_tokens", 512))
            warmup_results.append(result)
            success = "OK" if result.success else f"FAIL: {result.error}"
            print(f"    Warmup {i+1}/{warmup_requests}: TTFT={result.ttft_ms:.0f}ms, E2E={result.e2e_ms:.0f}ms [{success}]")

        warmup_success = sum(1 for r in warmup_results if r.success)
        print(f"  Warmup complete: {warmup_success}/{warmup_requests} successful")

        # Measurement phase - should benefit from cached KV
        print(f"\n  Running {num_requests} measurement requests at {qps} QPS...")
        print(f"  (KV cache should be warm - expect lower TTFT)")
        interval = 1.0 / qps

        for i in range(num_requests):
            start = time.perf_counter()
            messages = build_messages(i)
            result = await make_request(session, endpoint, model, messages, workload.get("max_tokens", 512))
            results.append(result)

            elapsed = time.perf_counter() - start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)

            if (i + 1) % 5 == 0 or (i + 1) == num_requests:
                success_rate = sum(1 for r in results if r.success) / len(results) * 100
                avg_ttft = statistics.mean([r.ttft_ms for r in results if r.success]) if results else 0
                print(f"    Progress: {i + 1}/{num_requests} ({success_rate:.0f}% success, avg TTFT: {avg_ttft:.0f}ms)")

    return results, shared_context


async def run_benchmark(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    workloads: list[str] = None,
    qps_levels: list[str] = None,
    requests_per_workload: int = 15,
    warmup_requests: int = 5,
    output_dir: str = "results/kimi-k2.5-p5e-lmcache",
):
    """Run the long context benchmark suite."""
    if workloads is None:
        workloads = ["context_16k"]  # Default to 16K
    if qps_levels is None:
        qps_levels = ["very_low", "low"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    print(f"\n{'#'*70}")
    print(f"# Long Context KV Cache Benchmark (16K+ tokens)")
    print(f"# Endpoint: {endpoint}")
    print(f"# Model: {model}")
    print(f"# Workloads: {workloads}")
    print(f"# QPS Levels: {qps_levels}")
    print(f"# Requests per config: {requests_per_workload}")
    print(f"# Purpose: Test KV cache offloading benefits with long context")
    print(f"{'#'*70}\n")

    for workload_name in workloads:
        workload = WORKLOADS[workload_name]
        for qps_name in qps_levels:
            qps = QPS_LEVELS[qps_name]
            print(f"\n{'='*70}")
            print(f"Benchmark: {workload_name} @ {qps_name} ({qps} QPS)")
            print(f"Description: {workload['description']}")
            print(f"Target context: ~{workload['context_tokens']:,} tokens")
            print(f"{'='*70}")

            results, shared_context = await run_long_context_workload(
                endpoint=endpoint,
                model=model,
                workload_name=workload_name,
                qps=qps,
                num_requests=requests_per_workload,
                warmup_requests=warmup_requests,
            )

            aggregated = aggregate_results(results)
            actual_context_tokens = len(shared_context) // 4

            timestamp = datetime.now().isoformat()
            result = BenchmarkResult(
                model=model,
                instance_type="p5e.48xlarge",
                workload=workload_name,
                context_size=actual_context_tokens,
                qps=qps,
                timestamp=timestamp,
                config={
                    "num_documents": workload["num_documents"],
                    "paragraphs_per_doc": workload["paragraphs_per_doc"],
                    "target_context_tokens": workload["context_tokens"],
                    "actual_context_tokens": actual_context_tokens,
                },
                num_requests=len(results),
                metrics=aggregated,
            )
            all_results.append(result)

            # Save individual result
            filename = f"long_context_{workload_name}_{qps_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = Path(output_dir) / filename
            with open(filepath, "w") as f:
                json.dump(asdict(result), f, indent=2)
            print(f"\nSaved: {filepath}")

            # Print summary
            print(f"\n--- Results Summary ---")
            print(f"  Context Size: ~{actual_context_tokens:,} tokens")
            print(f"  Success Rate: {aggregated.get('success_rate', 0)*100:.1f}%")
            if 'reasoning_rate' in aggregated:
                print(f"  Reasoning Response Rate: {aggregated['reasoning_rate']*100:.1f}%")
            if 'ttft_ms' in aggregated:
                print(f"  TTFT p50/p90/p99: {aggregated['ttft_ms']['p50']:.1f} / {aggregated['ttft_ms']['p90']:.1f} / {aggregated['ttft_ms']['p99']:.1f} ms")
                print(f"  E2E p50/p90/p99: {aggregated['e2e_ms']['p50']:.1f} / {aggregated['e2e_ms']['p90']:.1f} / {aggregated['e2e_ms']['p99']:.1f} ms")
                print(f"  Throughput: {aggregated['throughput']['tokens_per_second']:.1f} tok/s")

                # Calculate cache benefit indicator
                if aggregated['ttft_ms']['min'] > 0:
                    cache_benefit = aggregated['ttft_ms']['max'] / aggregated['ttft_ms']['min']
                    print(f"  TTFT Variance (max/min): {cache_benefit:.1f}x")
                    if cache_benefit > 2:
                        print(f"  >> High variance suggests cache misses on some requests")
                    else:
                        print(f"  >> Low variance suggests consistent cache hits")

            # Cooldown between workloads
            print("\nCooldown: 20s...")
            await asyncio.sleep(20)

    # Save combined results
    combined_path = Path(output_dir) / f"long_context_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\n\nCombined results saved to: {combined_path}")

    return all_results


async def compare_cold_vs_warm(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    output_dir: str = "results/kimi-k2.5-p5e-lmcache",
):
    """Compare cold cache vs warm cache TTFT for long context."""
    print(f"\n{'#'*70}")
    print(f"# Cold vs Warm Cache Comparison (16K context)")
    print(f"# This test demonstrates KV cache offloading benefit")
    print(f"{'#'*70}\n")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate 16K context
    print("Generating 16K token context...")
    documents = [generate_document(i, 15) for i in range(8)]
    shared_context = "\n\n---\n\n".join(documents)
    estimated_tokens = len(shared_context) // 4
    print(f"Context size: ~{estimated_tokens:,} tokens")

    system_prompt = "You are an expert technical consultant. Answer based only on the provided documents."
    prompts = [
        "What are the main architectural patterns discussed?",
        "Summarize the key best practices.",
        "What are the common challenges?",
        "Compare the different approaches.",
        "What recommendations would you give?",
    ]

    def build_messages(prompt_idx: int) -> list:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Documents:\n\n{shared_context}"},
            {"role": "assistant", "content": "I have reviewed all the documentation. Please ask your question."},
            {"role": "user", "content": prompts[prompt_idx % len(prompts)]},
        ]

    results = {"cold": [], "warm": []}

    async with aiohttp.ClientSession() as session:
        # Cold request (first request, no cache)
        print("\n=== COLD CACHE TEST (first request) ===")
        print("This request must compute all KV tensors from scratch...")

        messages = build_messages(0)
        cold_result = await make_request(session, endpoint, model, messages, 300)
        results["cold"].append(cold_result)

        if cold_result.success:
            print(f"  Cold TTFT: {cold_result.ttft_ms:.0f}ms")
            print(f"  Cold E2E: {cold_result.e2e_ms:.0f}ms")
        else:
            print(f"  Cold request failed: {cold_result.error}")

        # Warm requests (subsequent, should hit cache)
        print("\n=== WARM CACHE TEST (5 subsequent requests) ===")
        print("These requests should retrieve KV from FSx cache...")

        for i in range(5):
            messages = build_messages(i + 1)
            warm_result = await make_request(session, endpoint, model, messages, 300)
            results["warm"].append(warm_result)

            if warm_result.success:
                print(f"  Warm #{i+1} TTFT: {warm_result.ttft_ms:.0f}ms, E2E: {warm_result.e2e_ms:.0f}ms")
            else:
                print(f"  Warm #{i+1} failed: {warm_result.error}")

            await asyncio.sleep(1)  # Small delay between requests

    # Calculate and display comparison
    print("\n" + "="*70)
    print("COLD vs WARM COMPARISON")
    print("="*70)

    cold_ttft = results["cold"][0].ttft_ms if results["cold"][0].success else 0
    warm_ttfts = [r.ttft_ms for r in results["warm"] if r.success]

    if cold_ttft and warm_ttfts:
        avg_warm_ttft = statistics.mean(warm_ttfts)
        speedup = cold_ttft / avg_warm_ttft if avg_warm_ttft > 0 else 0

        print(f"\n  Cold TTFT:     {cold_ttft:.0f}ms")
        print(f"  Avg Warm TTFT: {avg_warm_ttft:.0f}ms")
        print(f"  Cache Speedup: {speedup:.2f}x")

        if speedup > 1.5:
            print(f"\n  >> KV cache offloading is EFFECTIVE!")
            print(f"  >> FSx cache retrieval faster than recomputation")
        elif speedup > 1.0:
            print(f"\n  >> Modest cache benefit detected")
        else:
            print(f"\n  >> Cache benefit not observed (may need more warmup)")

    # Save results
    comparison_result = {
        "context_tokens": estimated_tokens,
        "cold_ttft_ms": cold_ttft,
        "warm_ttft_ms": warm_ttfts,
        "avg_warm_ttft_ms": statistics.mean(warm_ttfts) if warm_ttfts else 0,
        "speedup": cold_ttft / statistics.mean(warm_ttfts) if warm_ttfts and statistics.mean(warm_ttfts) > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }

    filepath = Path(output_dir) / f"cold_vs_warm_16k_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filepath, "w") as f:
        json.dump(comparison_result, f, indent=2)
    print(f"\nSaved: {filepath}")

    return comparison_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Long Context KV Cache Benchmark")
    parser.add_argument("--mode", choices=["16k", "20k", "24k", "28k", "30k", "all", "extreme", "compare"], default="16k",
                        help="Benchmark mode: 16k, 20k, 24k, 28k, 30k context, all (16k-24k), extreme (28k-30k), or compare")
    parser.add_argument("--endpoint", default=ENDPOINT, help="vLLM API endpoint")
    parser.add_argument("--requests", type=int, default=15, help="Requests per workload")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup requests")
    parser.add_argument("--output", default="results/kimi-k2.5-p5e-lmcache", help="Output directory")

    args = parser.parse_args()

    if args.mode == "compare":
        asyncio.run(compare_cold_vs_warm(
            endpoint=args.endpoint,
            output_dir=args.output,
        ))
    elif args.mode == "all":
        asyncio.run(run_benchmark(
            endpoint=args.endpoint,
            workloads=["context_16k", "context_20k", "context_24k"],
            qps_levels=["very_low", "low"],
            requests_per_workload=args.requests,
            warmup_requests=args.warmup,
            output_dir=args.output,
        ))
    elif args.mode == "extreme":
        asyncio.run(run_benchmark(
            endpoint=args.endpoint,
            workloads=["context_28k", "context_30k"],
            qps_levels=["very_low"],
            requests_per_workload=args.requests,
            warmup_requests=args.warmup,
            output_dir=args.output,
        ))
    else:
        workload = f"context_{args.mode}"
        asyncio.run(run_benchmark(
            endpoint=args.endpoint,
            workloads=[workload],
            qps_levels=["very_low", "low"],
            requests_per_workload=args.requests,
            warmup_requests=args.warmup,
            output_dir=args.output,
        ))
