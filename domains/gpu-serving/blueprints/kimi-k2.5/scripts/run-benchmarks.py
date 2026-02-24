#!/usr/bin/env python3
"""
Kimi-K2.5 Benchmark Suite for p5e.48xlarge (8x H200)

Benchmarks designed to test:
1. Baseline throughput and latency
2. KV cache prefix caching effectiveness
3. Multi-turn conversation (LMCache-friendly workloads)
4. Reasoning-heavy workloads
5. Stress testing under high concurrency
6. Long context scaling (16K-48K tokens)
7. Multi-tenant cache pressure (50 unique system prompts)
8. Cache hit vs miss A/B comparison (KV reuse 0% vs 100%)
9. TraceReplayer with GMI production traces
10. FSx cache growth tracking across runs
"""

import asyncio
import aiohttp
import json
import time
import statistics
import subprocess
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import random

# Kimi-K2.5 specific configuration
MODEL_ID = "/mnt/nvme/models/Kimi-K2.5"
ENDPOINT = "http://localhost:8000"
MAX_MODEL_LEN = 32768
TENSOR_PARALLEL = 8

# Serving stack configurations
SERVING_CONFIGS = {
    "baseline": {
        "name": "Baseline vLLM",
        "description": "Native prefix caching, no external KV cache",
        "endpoint_port": 8000,
        "launch_script": "baseline-vllm/run-kimi-k2.5.sh",
    },
    "lmcache": {
        "name": "vLLM + LMCache (GDS)",
        "description": "KV cache offloading to FSx via GPU Direct Storage",
        "endpoint_port": 8000,
        "launch_script": "lmcache/run-kimi-k2.5.sh",
    },
    "lmcache-posix": {
        "name": "vLLM + LMCache (POSIX)",
        "description": "KV cache offloading to FSx via CPU bounce (no GDS)",
        "endpoint_port": 8000,
        "launch_script": "lmcache/run-kimi-k2.5.sh",  # needs LMCACHE_USE_EXPERIMENTAL=False
    },
    "dynamo": {
        "name": "NVIDIA Dynamo",
        "description": "Tiered KV cache manager (VRAM→DRAM→NVMe→FSx)",
        "endpoint_port": 8000,
        "launch_script": "dynamo/kimi-k2.5-gds.yaml",
    },
    "sglang-hicache": {
        "name": "SGLang HiCache",
        "description": "SGLang cascading eviction (GPU→CPU), no external storage",
        "endpoint_port": 8000,
        "launch_script": "configs/sglang-hicache.sh",
    },
    "sglang-hicache-nvme": {
        "name": "SGLang HiCache + NVMe",
        "description": "SGLang cascading eviction with NVMe L3 tier",
        "endpoint_port": 8000,
        "launch_script": "configs/sglang-hicache-nvme.sh",
    },
    "sglang-mooncake": {
        "name": "SGLang HiCache + Mooncake",
        "description": "SGLang cascading eviction with Mooncake L3 via RDMA",
        "endpoint_port": 8000,
        "launch_script": "configs/sglang-mooncake.sh",
    },
}

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
        "max_tokens": 200,
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
        "max_tokens": 300,
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
        "max_tokens": 150,
    },
    "long_context_rag": {
        "description": "Long context RAG (tests KV cache offloading benefits, 20K chat history)",
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
""" * 40,  # ~20000 tokens of shared context (matches BENCHMARK_REPORT.md)
        "max_tokens": 200,
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
        "max_tokens": 150,
    },
    "strict_no_reuse": {
        "description": "StrictSynthetic 0% KV reuse (control for cache benefit A/B test)",
        "system_prompt": "You are a helpful assistant. Answer the question.",
        "prompts": [
            "What are the main differences between TCP and UDP?",
            "Explain how a hash table works.",
            "What is the CAP theorem?",
            "Describe the observer pattern in software design.",
            "How does garbage collection work in Java?",
        ],
        "shared_context": "",  # No shared context — each request is unique
        "max_tokens": 150,
    },
}

# QPS levels for different test scenarios
QPS_LEVELS = {
    "low": 0.5,      # Light load - individual request latency focus
    "medium": 1.0,    # Moderate load - matches BENCHMARK_REPORT.md
    "high": 2.0,      # High load - balanced test
    "stress": 5.0,    # Stress test - find breaking point
    "extreme": 10.0,  # Extreme load - throughput ceiling
}

# FSx cache paths by serving config
FSX_CACHE_PATHS = {
    "baseline": None,  # No external cache
    "lmcache": "/mnt/fsx/kv-cache/lmcache",
    "lmcache-posix": "/mnt/fsx/kv-cache/lmcache",  # Same path, different I/O backend
    "dynamo": "/mnt/fsx/kv-cache/dynamo",
}


@dataclass
class RequestMetrics:
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    ttft_ms: float
    ttft_content_ms: float  # time to first content token (after reasoning)
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
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> RequestMetrics:
    """Make a single request and measure metrics."""
    start_time = time.perf_counter()
    ttft = None           # first token of any kind (reasoning or content)
    ttft_content = None   # first content token (after reasoning)
    tokens_received = 0
    reasoning_tokens_received = 0
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
                    prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
                    ttft_ms=0, ttft_content_ms=0, e2e_ms=0, itl_ms=0,
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
                        now = time.perf_counter()
                        # Reasoning tokens (Kimi-K2.5 emits these before content)
                        if delta.get("reasoning") or delta.get("reasoning_content"):
                            has_reasoning = True
                            reasoning_tokens_received += 1
                            if ttft is None:
                                ttft = (now - start_time) * 1000
                            token_times.append(now)
                        # Content tokens
                        if delta.get("content"):
                            if ttft is None:
                                ttft = (now - start_time) * 1000
                            if ttft_content is None:
                                ttft_content = (now - start_time) * 1000
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
                reasoning_tokens=reasoning_tokens_received,
                ttft_ms=ttft or 0,
                ttft_content_ms=ttft_content or 0,
                e2e_ms=e2e_time,
                itl_ms=itl,
                success=True,
                has_reasoning=has_reasoning,
            )

    except asyncio.TimeoutError:
        return RequestMetrics(
            prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
            ttft_ms=0, ttft_content_ms=0, e2e_ms=0, itl_ms=0,
            success=False, error="Request timeout",
        )
    except Exception as e:
        return RequestMetrics(
            prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
            ttft_ms=0, ttft_content_ms=0, e2e_ms=0, itl_ms=0,
            success=False, error=str(e),
        )


async def run_workload(
    endpoint: str,
    model: str,
    workload_name: str,
    qps: float,
    num_requests: int,
    warmup_requests: int = 3,
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
            await make_request(session, endpoint, model, messages, workload.get("max_tokens", 200))
        print("  Warmup complete.")

        # Measurement
        print(f"  Running {num_requests} requests at {qps} QPS...")
        interval = 1.0 / qps

        for i in range(num_requests):
            start = time.perf_counter()
            messages = build_messages(i)
            result = await make_request(session, endpoint, model, messages, workload.get("max_tokens", 200))
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
    ttft_content_values = [r.ttft_content_ms for r in successful if r.ttft_content_ms > 0]
    itl_values = [r.itl_ms for r in successful if r.itl_ms > 0]
    e2e_values = [r.e2e_ms for r in successful]
    reasoning_count = sum(1 for r in successful if r.has_reasoning)

    total_content_tokens = sum(r.completion_tokens for r in successful)
    total_reasoning_tokens = sum(r.reasoning_tokens for r in successful)
    total_tokens = total_content_tokens + total_reasoning_tokens
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
        "ttft_content_ms": {
            "mean": statistics.mean(ttft_content_values) if ttft_content_values else 0,
            **calculate_percentiles(ttft_content_values),
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
            "total_tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "content_tokens_per_second": total_content_tokens / total_time if total_time > 0 else 0,
            "reasoning_tokens_per_second": total_reasoning_tokens / total_time if total_time > 0 else 0,
            "requests_per_second": len(successful) / total_time if total_time > 0 else 0,
        },
        "tokens": {
            "total_content": total_content_tokens,
            "total_reasoning": total_reasoning_tokens,
            "total_all": total_tokens,
            "avg_content": statistics.mean([r.completion_tokens for r in successful]),
            "avg_reasoning": statistics.mean([r.reasoning_tokens for r in successful]),
        },
    }


async def run_benchmark_suite(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    workloads: list[str] = None,
    qps_levels: list[str] = None,
    requests_per_workload: int = 30,
    warmup_requests: int = 3,
    output_dir: str = "results/kimi-k2.5-p5e",
    config: str = "baseline",
):
    """Run the full benchmark suite."""
    if workloads is None:
        workloads = list(WORKLOADS.keys())
    if qps_levels is None:
        qps_levels = ["low", "medium", "high"]

    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])

    # Use config-specific output subdirectory
    output_dir = f"{output_dir}/{config}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    print(f"\n{'#'*70}")
    print(f"# Kimi-K2.5 Benchmark Suite on p5e.48xlarge (8x H200)")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Description: {serving_config['description']}")
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

            # Scrape prefix cache counters BEFORE workload
            pre_cache = await scrape_prefix_cache_metrics(endpoint)

            results = await run_workload(
                endpoint=endpoint,
                model=model,
                workload_name=workload_name,
                qps=qps,
                num_requests=requests_per_workload,
                warmup_requests=warmup_requests,
            )

            aggregated = aggregate_results(results)

            # Scrape prefix cache counters AFTER workload and compute delta
            post_cache = await scrape_prefix_cache_metrics(endpoint)
            prefix_cache_delta = {}
            if pre_cache and post_cache:
                hits = post_cache.get("hits", 0) - pre_cache.get("hits", 0)
                misses = post_cache.get("misses", 0) - pre_cache.get("misses", 0)
                total = hits + misses
                prefix_cache_delta = {
                    "hits": hits,
                    "misses": misses,
                    "hit_rate": hits / total if total > 0 else None,
                    "post_hit_rate_instant": post_cache.get("hit_rate"),
                }
                # Include SGLang HiCache per-tier metrics if present
                for tier in ["l1", "l2", "l3"]:
                    h_key, m_key = f"{tier}_hits", f"{tier}_misses"
                    if h_key in post_cache:
                        prefix_cache_delta[h_key] = post_cache[h_key] - pre_cache.get(h_key, 0)
                        prefix_cache_delta[m_key] = post_cache[m_key] - pre_cache.get(m_key, 0)

            # Track FSx cache growth and LMCache metrics
            cache_snapshot = measure_fsx_cache(config)
            lmcache_metrics = await scrape_lmcache_metrics(endpoint)

            timestamp = datetime.now().isoformat()
            result = BenchmarkResult(
                model=model,
                instance_type="p5e.48xlarge",
                workload=workload_name,
                qps=qps,
                timestamp=timestamp,
                config={
                    "serving_config": config,
                    "serving_name": serving_config["name"],
                    "tensor_parallel": TENSOR_PARALLEL,
                    "max_model_len": MAX_MODEL_LEN,
                    "workload_config": WORKLOADS[workload_name],
                    "fsx_cache_snapshot": cache_snapshot,
                    "lmcache_metrics": lmcache_metrics,
                    "prefix_cache_metrics": prefix_cache_delta if prefix_cache_delta else {"info": "no prefix cache metrics found"},
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
                if aggregated.get('ttft_content_ms', {}).get('mean', 0) > 0:
                    print(f"  TTFT-content p50/p90: {aggregated['ttft_content_ms']['p50']:.1f} / {aggregated['ttft_content_ms']['p90']:.1f} ms")
                print(f"  E2E p50/p90/p99: {aggregated['e2e_ms']['p50']:.1f} / {aggregated['e2e_ms']['p90']:.1f} / {aggregated['e2e_ms']['p99']:.1f} ms")
                print(f"  Throughput: {aggregated['throughput']['total_tokens_per_second']:.1f} tok/s (reasoning: {aggregated['throughput']['reasoning_tokens_per_second']:.1f}, content: {aggregated['throughput']['content_tokens_per_second']:.1f})")
            if prefix_cache_delta and "info" not in prefix_cache_delta:
                pc_hits = prefix_cache_delta.get("hits", 0)
                pc_misses = prefix_cache_delta.get("misses", 0)
                pc_rate = prefix_cache_delta.get("hit_rate")
                rate_str = f"{pc_rate:.2%}" if pc_rate is not None else "N/A"
                print(f"  Prefix Cache: hits={pc_hits:.0f}, misses={pc_misses:.0f}, hit_rate={rate_str}")
            if lmcache_metrics and "error" not in lmcache_metrics and "info" not in lmcache_metrics:
                hit_rate = lmcache_metrics.get("lmcache:retrieve_hit_rate", 0)
                stores = lmcache_metrics.get("lmcache:num_store_requests", 0)
                retrieves = lmcache_metrics.get("lmcache:num_retrieve_requests", 0)
                remote_usage = lmcache_metrics.get("lmcache:remote_cache_usage", 0)
                print(f"  LMCache: hit_rate={hit_rate:.2%}, stores={stores:.0f}, retrieves={retrieves:.0f}, remote_cache={remote_usage/(1024*1024):.1f}MB")

            # Cooldown
            print("\nCooldown: 15s...")
            await asyncio.sleep(15)

    # Save combined results
    combined_path = Path(output_dir) / f"combined_kimi_k2.5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\n\nCombined results saved to: {combined_path}")

    return all_results


async def scrape_lmcache_metrics(endpoint: str) -> dict:
    """Scrape LMCache metrics from the vLLM /metrics Prometheus endpoint."""
    lmcache_keys = [
        "lmcache:num_retrieve_requests",
        "lmcache:num_store_requests",
        "lmcache:retrieve_hit_rate",
        "lmcache:lookup_hit_rate",
        "lmcache:num_hit_tokens",
        "lmcache:local_cache_usage",
        "lmcache:remote_cache_usage",
        "lmcache:local_storage_usage",
        "lmcache:num_remote_read_requests",
        "lmcache:num_remote_write_requests",
    ]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{endpoint}/metrics",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    return {"error": f"HTTP {response.status}"}
                text = await response.text()
                metrics = {}
                for line in text.splitlines():
                    if line.startswith("#"):
                        continue
                    for key in lmcache_keys:
                        if line.startswith(key):
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    metrics[key] = float(parts[-1])
                                except ValueError:
                                    pass
                return metrics if metrics else {"info": "no lmcache metrics found"}
    except Exception as e:
        return {"error": str(e)}


def measure_fsx_cache(config: str) -> dict:
    """Measure FSx cache size and file count for a serving config.

    Tries local filesystem first. Falls back to kubectl exec if the path
    doesn't exist locally (e.g., running benchmarks from a Mac via port-forward).
    """
    cache_path = FSX_CACHE_PATHS.get(config)
    if not cache_path:
        return {"path": None, "size_bytes": 0, "size_mb": 0, "file_count": 0}

    cache_dir = Path(cache_path)
    if cache_dir.exists():
        # Local measurement (running on the p5e instance)
        try:
            result = subprocess.run(
                ["du", "-sb", cache_path], capture_output=True, text=True, timeout=30
            )
            size_bytes = int(result.stdout.split()[0]) if result.returncode == 0 else 0
        except Exception:
            size_bytes = 0
        try:
            result = subprocess.run(
                ["find", cache_path, "-type", "f"], capture_output=True, text=True, timeout=30
            )
            file_count = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
        except Exception:
            file_count = 0
    else:
        # Remote measurement via kubectl (running from Mac via port-forward)
        size_bytes = 0
        file_count = 0
        pod_label = "app=vllm-kimi-k2"
        namespace = "ml-inference"
        try:
            result = subprocess.run(
                ["kubectl", "exec", "-n", namespace,
                 f"deploy/vllm-kimi-k2", "--",
                 "du", "-sb", cache_path],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                size_bytes = int(result.stdout.split()[0])
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["kubectl", "exec", "-n", namespace,
                 f"deploy/vllm-kimi-k2", "--",
                 "find", cache_path, "-type", "f"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                file_count = len(result.stdout.strip().split("\n"))
        except Exception:
            pass

    return {
        "path": cache_path,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 1),
        "file_count": file_count,
    }


def generate_context(target_tokens: int) -> str:
    """Generate synthetic context of approximately target_tokens length.

    Uses ~4 chars per token as approximation.
    """
    base_paragraph = (
        "The distributed system architecture consists of multiple microservices "
        "communicating via message queues. Each service is independently deployable "
        "and scalable. The architecture follows the principle of loose coupling and "
        "high cohesion. Load balancing is achieved through a combination of DNS-based "
        "and application-level strategies. Performance optimization strategies include "
        "caching at multiple levels, connection pooling, and asynchronous processing. "
        "The cache layer uses a combination of in-memory and distributed caching "
        "solutions. Query optimization involves proper indexing and query planning. "
        "Security is implemented through multiple layers including network segmentation, "
        "encryption at rest and in transit, and role-based access control. "
    )
    target_chars = target_tokens * 4
    repetitions = max(1, target_chars // len(base_paragraph))
    return (base_paragraph * repetitions)[:target_chars]


async def run_long_context_scaling(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    context_sizes: list[int] = None,
    num_requests_per_size: int = 15,
    warmup_requests: int = 5,
    output_dir: str = "results/kimi-k2.5-p5e",
    config: str = "baseline",
) -> list[BenchmarkResult]:
    """Run long context scaling benchmark (16K → 48K tokens).

    Reproduces BENCHMARK_REPORT.md Long Context benchmarks.
    Tests sub-linear latency scaling with increasing shared context sizes.
    """
    if context_sizes is None:
        context_sizes = [16000, 24000, 36000, 48000]

    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    output_dir = f"{output_dir}/{config}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    print(f"\n{'#'*70}")
    print(f"# LONG CONTEXT SCALING: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Context sizes: {context_sizes}")
    print(f"# Requests per size: {num_requests_per_size}")
    print(f"{'#'*70}\n")

    prompts = [
        "What is the main topic discussed in this context?",
        "Summarize the key points mentioned.",
        "What are the implications described?",
        "How does this relate to real-world applications?",
        "What conclusions can be drawn from this material?",
    ]

    async with aiohttp.ClientSession() as session:
        for ctx_size in context_sizes:
            cache_before = measure_fsx_cache(config)
            context = generate_context(ctx_size)
            print(f"\n{'='*60}")
            print(f"Long Context: ~{ctx_size} tokens ({len(context)} chars)")
            print(f"FSx cache before: {cache_before['size_mb']} MB, {cache_before['file_count']} files")
            print(f"{'='*60}")

            # Scrape prefix cache counters BEFORE
            pre_cache_lcs = await scrape_prefix_cache_metrics(endpoint)

            # Warmup with this context size
            print(f"  Warmup: {warmup_requests} requests...")
            for i in range(warmup_requests):
                messages = [
                    {"role": "system", "content": "Answer questions using the provided context."},
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": "I have read the context. Please ask your question."},
                    {"role": "user", "content": prompts[i % len(prompts)]},
                ]
                await make_request(session, endpoint, model, messages, max_tokens=150)

            # Measurement
            results = []
            print(f"  Running {num_requests_per_size} requests...")
            for i in range(num_requests_per_size):
                messages = [
                    {"role": "system", "content": "Answer questions using the provided context."},
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": "I have read the context. Please ask your question."},
                    {"role": "user", "content": prompts[i % len(prompts)]},
                ]
                result = await make_request(session, endpoint, model, messages, max_tokens=150)
                results.append(result)
                await asyncio.sleep(2.0)  # 0.5 QPS

            cache_after = measure_fsx_cache(config)
            aggregated = aggregate_results(results)

            # Scrape prefix cache counters AFTER and compute delta
            post_cache_lcs = await scrape_prefix_cache_metrics(endpoint)
            pc_delta_lcs = {}
            if pre_cache_lcs and post_cache_lcs:
                pc_h = post_cache_lcs.get("hits", 0) - pre_cache_lcs.get("hits", 0)
                pc_m = post_cache_lcs.get("misses", 0) - pre_cache_lcs.get("misses", 0)
                pc_t = pc_h + pc_m
                pc_delta_lcs = {
                    "hits": pc_h, "misses": pc_m,
                    "hit_rate": pc_h / pc_t if pc_t > 0 else None,
                    "post_hit_rate_instant": post_cache_lcs.get("hit_rate"),
                }

            bench_result = BenchmarkResult(
                model=model,
                instance_type="p5e.48xlarge",
                workload=f"long_context_{ctx_size}",
                qps=0.5,
                timestamp=datetime.now().isoformat(),
                config={
                    "serving_config": config,
                    "serving_name": serving_config["name"],
                    "context_tokens": ctx_size,
                    "context_chars": len(context),
                    "fsx_cache_before": cache_before,
                    "fsx_cache_after": cache_after,
                    "prefix_cache_metrics": pc_delta_lcs if pc_delta_lcs else {"info": "no prefix cache metrics found"},
                },
                num_requests=len(results),
                metrics=aggregated,
            )
            all_results.append(bench_result)

            filename = f"long_context_{ctx_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = Path(output_dir) / filename
            with open(filepath, "w") as f:
                json.dump(asdict(bench_result), f, indent=2)

            print(f"\n--- {ctx_size} Token Results ---")
            if "e2e_ms" in aggregated:
                print(f"  E2E p50/p90/p99: {aggregated['e2e_ms']['p50']:.1f} / {aggregated['e2e_ms']['p90']:.1f} / {aggregated['e2e_ms']['p99']:.1f} ms")
                print(f"  TTFT p50: {aggregated['ttft_ms']['p50']:.1f} ms")
            print(f"  FSx cache after: {cache_after['size_mb']} MB, {cache_after['file_count']} files")
            if pc_delta_lcs and "info" not in pc_delta_lcs:
                rate_s = f"{pc_delta_lcs['hit_rate']:.2%}" if pc_delta_lcs.get("hit_rate") is not None else "N/A"
                print(f"  Prefix Cache: hits={pc_delta_lcs['hits']:.0f}, misses={pc_delta_lcs['misses']:.0f}, hit_rate={rate_s}")
            print(f"  Saved: {filepath}")

            print("\nCooldown: 15s...")
            await asyncio.sleep(15)

    return all_results


async def run_multi_tenant(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    num_tenants: int = 50,
    users_per_tenant: int = 3,
    system_prompt_tokens: int = 4000,
    output_dir: str = "results/kimi-k2.5-p5e",
    config: str = "baseline",
) -> BenchmarkResult:
    """Run multi-tenant benchmark with unique system prompts per tenant.

    Reproduces BENCHMARK_REPORT.md Multi-Tenant benchmark.
    Tests cache pressure with high prefix variety (worst case for caching).
    """
    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    output_dir = f"{output_dir}/{config}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# MULTI-TENANT: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Tenants: {num_tenants}, Users/tenant: {users_per_tenant}")
    print(f"# System prompt: ~{system_prompt_tokens} tokens each")
    print(f"{'#'*70}\n")

    # Generate unique system prompts per tenant
    tenant_prompts = []
    for t in range(num_tenants):
        base = (
            f"You are the AI assistant for Tenant #{t+1}. "
            f"Your company specializes in {'technology' if t % 5 == 0 else 'healthcare' if t % 5 == 1 else 'finance' if t % 5 == 2 else 'education' if t % 5 == 3 else 'manufacturing'}. "
            f"Company ID: TENANT-{t+1:04d}. "
        )
        padding = generate_context(system_prompt_tokens - len(base) // 4)
        tenant_prompts.append(base + padding)

    user_questions = [
        "What services do you offer?",
        "Can you help me with my account?",
        "What are your business hours?",
        "How do I contact support?",
        "What's the pricing for enterprise plans?",
        "Tell me about your security certifications.",
    ]

    cache_before = measure_fsx_cache(config)
    pre_cache_mt = await scrape_prefix_cache_metrics(endpoint)
    results = []
    total_requests = num_tenants * users_per_tenant

    print(f"  Total requests: {total_requests}")
    print(f"  FSx cache before: {cache_before['size_mb']} MB, {cache_before['file_count']} files")

    async with aiohttp.ClientSession() as session:
        request_idx = 0
        for tenant_id in range(num_tenants):
            for user_id in range(users_per_tenant):
                messages = [
                    {"role": "system", "content": tenant_prompts[tenant_id]},
                    {"role": "user", "content": user_questions[request_idx % len(user_questions)]},
                ]
                result = await make_request(session, endpoint, model, messages, max_tokens=150)
                results.append(result)
                request_idx += 1

                # ~3 QPS
                await asyncio.sleep(0.33)

                if request_idx % 25 == 0:
                    success_rate = sum(1 for r in results if r.success) / len(results) * 100
                    print(f"    Progress: {request_idx}/{total_requests} ({success_rate:.0f}% success)")

    cache_after = measure_fsx_cache(config)
    aggregated = aggregate_results(results)

    # Scrape prefix cache counters AFTER and compute delta
    post_cache_mt = await scrape_prefix_cache_metrics(endpoint)
    pc_delta_mt = {}
    if pre_cache_mt and post_cache_mt:
        pc_h = post_cache_mt.get("hits", 0) - pre_cache_mt.get("hits", 0)
        pc_m = post_cache_mt.get("misses", 0) - pre_cache_mt.get("misses", 0)
        pc_t = pc_h + pc_m
        pc_delta_mt = {
            "hits": pc_h, "misses": pc_m,
            "hit_rate": pc_h / pc_t if pc_t > 0 else None,
            "post_hit_rate_instant": post_cache_mt.get("hit_rate"),
        }

    # Separate cold (first request per tenant) vs warm (subsequent)
    cold_results = [results[i * users_per_tenant] for i in range(num_tenants) if i * users_per_tenant < len(results)]
    warm_results = [r for i, r in enumerate(results) if i % users_per_tenant != 0]
    cold_agg = aggregate_results(cold_results)
    warm_agg = aggregate_results(warm_results)

    bench_result = BenchmarkResult(
        model=model,
        instance_type="p5e.48xlarge",
        workload="multi_tenant",
        qps=3.0,
        timestamp=datetime.now().isoformat(),
        config={
            "serving_config": config,
            "serving_name": serving_config["name"],
            "num_tenants": num_tenants,
            "users_per_tenant": users_per_tenant,
            "system_prompt_tokens": system_prompt_tokens,
            "fsx_cache_before": cache_before,
            "fsx_cache_after": cache_after,
            "prefix_cache_metrics": pc_delta_mt if pc_delta_mt else {"info": "no prefix cache metrics found"},
        },
        num_requests=len(results),
        metrics={
            **aggregated,
            "cold_requests": cold_agg,
            "warm_requests": warm_agg,
            "cache_benefit": {
                "cold_e2e_mean": cold_agg.get("e2e_ms", {}).get("mean", 0),
                "warm_e2e_mean": warm_agg.get("e2e_ms", {}).get("mean", 0),
                "speedup": (
                    cold_agg.get("e2e_ms", {}).get("mean", 1)
                    / warm_agg.get("e2e_ms", {}).get("mean", 1)
                    if warm_agg.get("e2e_ms", {}).get("mean", 0) > 0 else 0
                ),
            },
        },
    )

    filename = f"multi_tenant_{num_tenants}t_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = Path(output_dir) / filename
    with open(filepath, "w") as f:
        json.dump(asdict(bench_result), f, indent=2)

    print(f"\n--- Multi-Tenant Results ---")
    print(f"  Total Requests: {len(results)}")
    print(f"  Success Rate: {aggregated.get('success_rate', 0)*100:.1f}%")
    if "e2e_ms" in cold_agg and "e2e_ms" in warm_agg:
        print(f"  Cold (first per tenant): {cold_agg['e2e_ms']['mean']:.0f} ms")
        print(f"  Warm (subsequent): {warm_agg['e2e_ms']['mean']:.0f} ms")
        speedup = bench_result.metrics["cache_benefit"]["speedup"]
        print(f"  Cache Benefit: {speedup:.2f}x")
    print(f"  FSx cache after: {cache_after['size_mb']} MB, {cache_after['file_count']} files")
    if pc_delta_mt and "info" not in pc_delta_mt:
        rate_s = f"{pc_delta_mt['hit_rate']:.2%}" if pc_delta_mt.get("hit_rate") is not None else "N/A"
        print(f"  Prefix Cache: hits={pc_delta_mt['hits']:.0f}, misses={pc_delta_mt['misses']:.0f}, hit_rate={rate_s}")
    print(f"  Saved: {filepath}")

    return bench_result


async def run_trace_replay(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    trace_file: str = "traces/gmi_trace.jsonl",
    preserve_timing: bool = True,
    output_dir: str = "results/kimi-k2.5-p5e",
    config: str = "baseline",
) -> Optional[BenchmarkResult]:
    """Replay production traces from a JSONL file.

    Reproduces BENCHMARK_REPORT.md TraceReplayer benchmark.
    Each line in the trace file should be a JSON object with:
      {"messages": [...], "max_tokens": N, "timestamp": <epoch_seconds>}
    """
    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    output_dir = f"{output_dir}/{config}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trace_path = Path(trace_file)
    if not trace_path.exists():
        print(f"\n[WARN] Trace file not found: {trace_file}")
        print("  Skipping TraceReplayer. To run this benchmark, provide traces/gmi_trace.jsonl")
        return None

    # Load traces
    traces = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    if not traces:
        print(f"[WARN] Trace file is empty: {trace_file}")
        return None

    print(f"\n{'#'*70}")
    print(f"# TRACE REPLAYER: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Trace file: {trace_file} ({len(traces)} requests)")
    print(f"# Preserve timing: {preserve_timing}")
    print(f"{'#'*70}\n")

    cache_before = measure_fsx_cache(config)
    results = []
    total_tokens_generated = 0

    async with aiohttp.ClientSession() as session:
        prev_ts = traces[0].get("timestamp", 0) if preserve_timing else 0

        for i, trace in enumerate(traces):
            # Preserve original inter-request timing
            if preserve_timing and i > 0:
                curr_ts = trace.get("timestamp", prev_ts)
                delay = max(0, curr_ts - prev_ts)
                if delay > 0 and delay < 60:  # Cap at 60s
                    await asyncio.sleep(delay)
                prev_ts = curr_ts

            messages = trace.get("messages", [
                {"role": "user", "content": trace.get("prompt", "Hello")},
            ])
            max_tokens = trace.get("max_tokens", 200)

            result = await make_request(session, endpoint, model, messages, max_tokens=max_tokens)
            results.append(result)
            if result.success:
                total_tokens_generated += result.completion_tokens

            if (i + 1) % 10 == 0:
                success_rate = sum(1 for r in results if r.success) / len(results) * 100
                print(f"    Progress: {i + 1}/{len(traces)} ({success_rate:.0f}% success)")

    cache_after = measure_fsx_cache(config)
    aggregated = aggregate_results(results)

    bench_result = BenchmarkResult(
        model=model,
        instance_type="p5e.48xlarge",
        workload="trace_replay",
        qps=len(results) / (sum(r.e2e_ms for r in results if r.success) / 1000) if results else 0,
        timestamp=datetime.now().isoformat(),
        config={
            "serving_config": config,
            "serving_name": serving_config["name"],
            "trace_file": trace_file,
            "num_traces": len(traces),
            "preserve_timing": preserve_timing,
            "total_tokens_generated": total_tokens_generated,
            "fsx_cache_before": cache_before,
            "fsx_cache_after": cache_after,
        },
        num_requests=len(results),
        metrics=aggregated,
    )

    filename = f"trace_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = Path(output_dir) / filename
    with open(filepath, "w") as f:
        json.dump(asdict(bench_result), f, indent=2)

    print(f"\n--- Trace Replay Results ---")
    print(f"  Requests Replayed: {len(results)}")
    print(f"  Success Rate: {aggregated.get('success_rate', 0)*100:.1f}%")
    print(f"  Tokens Generated: {total_tokens_generated}")
    if "ttft_ms" in aggregated:
        print(f"  TTFT mean: {aggregated['ttft_ms']['mean']:.0f} ms")
        print(f"  E2E mean: {aggregated['e2e_ms']['mean']:.0f} ms")
    print(f"  FSx cache: {cache_after['size_mb']} MB, {cache_after['file_count']} files")
    print(f"  Saved: {filepath}")

    return bench_result


async def run_stress_test(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    duration_seconds: int = 120,
    target_qps: float = 10.0,
    output_dir: str = "results/kimi-k2.5-p5e",
    config: str = "baseline",
):
    """Run stress test with high concurrency."""
    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    output_dir = f"{output_dir}/{config}"

    print(f"\n{'#'*70}")
    print(f"# STRESS TEST: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
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
        print(f"  Throughput: {aggregated['throughput']['total_tokens_per_second']:.1f} tok/s")
    print(f"\nSaved: {filepath}")

    return result


async def run_all_benchmarks(
    endpoint: str = ENDPOINT,
    config: str = "baseline",
    output_dir: str = "results/kimi-k2.5-p5e",
    requests_per_workload: int = 30,
    trace_file: str = "traces/gmi_trace.jsonl",
):
    """Run the complete benchmark suite matching BENCHMARK_REPORT.md coverage.

    Runs all 11 benchmark runs in order:
    1-7: Standard workloads (StrictSynthetic, LMCacheSynthetic, Agentic, etc.)
    8: Long context scaling (16K → 48K)
    9: Multi-tenant (50 tenants)
    10: StrictSynthetic 0% reuse (A/B control)
    11: TraceReplayer (GMI traces)

    FSx cache is measured before/after each run.
    """
    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    config_output = f"{output_dir}/{config}"
    Path(config_output).mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# COMPLETE BENCHMARK SUITE: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Output: {config_output}")
    print(f"{'#'*70}\n")

    initial_cache = measure_fsx_cache(config)
    print(f"Initial FSx cache: {initial_cache['size_mb']} MB, {initial_cache['file_count']} files\n")

    # Phase 1: Standard workloads (Runs 1-7)
    print("=" * 70)
    print("PHASE 1: Standard workloads (Runs 1-7)")
    print("=" * 70)
    await run_benchmark_suite(
        endpoint=endpoint,
        workloads=list(WORKLOADS.keys()),
        qps_levels=["low", "medium", "high"],
        requests_per_workload=requests_per_workload,
        output_dir=output_dir,
        config=config,
    )

    # Phase 2: Long context scaling (Run 8)
    print("\n" + "=" * 70)
    print("PHASE 2: Long context scaling (Run 8)")
    print("=" * 70)
    await run_long_context_scaling(
        endpoint=endpoint,
        context_sizes=[16000, 24000, 36000, 48000],
        num_requests_per_size=15,
        output_dir=output_dir,
        config=config,
    )

    # Phase 3: Multi-tenant (Run 9)
    print("\n" + "=" * 70)
    print("PHASE 3: Multi-tenant (Run 9)")
    print("=" * 70)
    await run_multi_tenant(
        endpoint=endpoint,
        num_tenants=50,
        users_per_tenant=3,
        output_dir=output_dir,
        config=config,
    )

    # Phase 4: TraceReplayer (Run 11)
    print("\n" + "=" * 70)
    print("PHASE 4: TraceReplayer (Run 11)")
    print("=" * 70)
    await run_trace_replay(
        endpoint=endpoint,
        trace_file=trace_file,
        output_dir=output_dir,
        config=config,
    )

    # Phase 5: Stress test
    print("\n" + "=" * 70)
    print("PHASE 5: Stress test")
    print("=" * 70)
    await run_stress_test(
        endpoint=endpoint,
        duration_seconds=120,
        target_qps=10.0,
        output_dir=output_dir,
        config=config,
    )

    final_cache = measure_fsx_cache(config)
    print(f"\n{'#'*70}")
    print(f"# COMPLETE BENCHMARK SUITE FINISHED")
    print(f"# FSx cache: {initial_cache['size_mb']} MB → {final_cache['size_mb']} MB")
    print(f"# Cache files: {initial_cache['file_count']} → {final_cache['file_count']}")
    print(f"# Results in: {config_output}")
    print(f"{'#'*70}")

    # Save cache growth summary
    cache_summary = {
        "config": config,
        "initial": initial_cache,
        "final": final_cache,
        "growth_mb": final_cache["size_mb"] - initial_cache["size_mb"],
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = Path(config_output) / f"fsx_cache_growth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(cache_summary, f, indent=2)
    print(f"Cache growth summary: {summary_path}")


async def run_multi_turn_conversation(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    num_users: int = 8,
    num_rounds: int = 20,
    history_tokens_per_round: int = 1000,
    output_dir: str = "results/kimi-k2.5-p5e",
    config: str = "baseline",
) -> BenchmarkResult:
    """Run multi-turn conversation benchmark with growing history per user.

    Reproduces BENCHMARK_REPORT.md "LMCacheSynthetic 20K Chat History".
    Each user has a multi-turn conversation where the context grows by
    ~history_tokens_per_round per turn. Measures per-turn TTFT to show
    increasing cache benefit as more history is reused.

    This is LMCache's strongest use case: each turn reuses KV from all
    previous turns, so turn 20 has ~95% cache hit rate.
    """
    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    output_dir = f"{output_dir}/{config}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# MULTI-TURN CONVERSATION: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Users: {num_users}, Rounds: {num_rounds}")
    print(f"# History growth: ~{history_tokens_per_round} tokens/round")
    print(f"# Total context at round {num_rounds}: ~{num_rounds * history_tokens_per_round} tokens")
    print(f"{'#'*70}\n")

    cache_before = measure_fsx_cache(config)

    # Generate unique system prompts per user
    system_prompts = [
        f"You are a helpful AI assistant for User #{u+1}. "
        f"Always provide detailed, thoughtful responses."
        for u in range(num_users)
    ]

    # Conversation topics per round
    round_questions = [
        "Tell me about distributed systems architecture.",
        "How does caching work in distributed systems?",
        "What are the trade-offs of eventual consistency?",
        "Explain the CAP theorem with examples.",
        "How do message queues help with decoupling?",
        "What is the role of load balancers?",
        "Describe microservices vs monolith trade-offs.",
        "How does database sharding work?",
        "What are circuit breakers in distributed systems?",
        "Explain consensus algorithms like Raft.",
        "How does service discovery work?",
        "What is the saga pattern for transactions?",
        "Describe event sourcing and CQRS.",
        "How do you handle distributed tracing?",
        "What are the benefits of immutable infrastructure?",
        "Explain blue-green deployments.",
        "How does container orchestration work?",
        "What is infrastructure as code?",
        "Describe observability vs monitoring.",
        "How do you design for fault tolerance?",
    ]

    # Generate filler history content
    history_filler = generate_context(history_tokens_per_round)

    all_results = []
    per_round_metrics = []

    async with aiohttp.ClientSession() as session:
        for round_num in range(num_rounds):
            round_results = []
            print(f"\n  Round {round_num + 1}/{num_rounds} "
                  f"(~{(round_num + 1) * history_tokens_per_round} tokens context)")

            for user_id in range(num_users):
                # Build conversation history: system prompt + all prior rounds
                messages = [{"role": "system", "content": system_prompts[user_id]}]

                # Add history from all previous rounds
                for prev_round in range(round_num):
                    messages.append({
                        "role": "user",
                        "content": round_questions[prev_round % len(round_questions)]
                    })
                    # Simulate previous assistant response as filler
                    messages.append({
                        "role": "assistant",
                        "content": history_filler[:history_tokens_per_round * 4]
                    })

                # Current round question
                messages.append({
                    "role": "user",
                    "content": round_questions[round_num % len(round_questions)]
                })

                result = await make_request(
                    session, endpoint, model, messages, max_tokens=150
                )
                round_results.append(result)
                all_results.append(result)

                await asyncio.sleep(0.5)  # ~2 QPS

            # Aggregate per-round metrics
            round_agg = aggregate_results(round_results)
            per_round_metrics.append({
                "round": round_num + 1,
                "approx_context_tokens": (round_num + 1) * history_tokens_per_round,
                "ttft_mean": round_agg.get("ttft_ms", {}).get("mean", 0),
                "ttft_p50": round_agg.get("ttft_ms", {}).get("p50", 0),
                "e2e_mean": round_agg.get("e2e_ms", {}).get("mean", 0),
                "success_rate": round_agg.get("success_rate", 0),
            })

            ttft_mean = round_agg.get("ttft_ms", {}).get("mean", 0)
            e2e_mean = round_agg.get("e2e_ms", {}).get("mean", 0)
            print(f"    TTFT: {ttft_mean:.0f}ms, E2E: {e2e_mean:.0f}ms")

    cache_after = measure_fsx_cache(config)
    overall_agg = aggregate_results(all_results)

    # Separate first round (cold) vs subsequent rounds (warm)
    cold_results = all_results[:num_users]
    warm_results = all_results[num_users:]
    cold_agg = aggregate_results(cold_results)
    warm_agg = aggregate_results(warm_results)

    bench_result = BenchmarkResult(
        model=model,
        instance_type="p5e.48xlarge",
        workload="multi_turn_conversation",
        qps=2.0,
        timestamp=datetime.now().isoformat(),
        config={
            "serving_config": config,
            "serving_name": serving_config["name"],
            "num_users": num_users,
            "num_rounds": num_rounds,
            "history_tokens_per_round": history_tokens_per_round,
            "total_context_at_final_round": num_rounds * history_tokens_per_round,
            "fsx_cache_before": cache_before,
            "fsx_cache_after": cache_after,
        },
        num_requests=len(all_results),
        metrics={
            **overall_agg,
            "per_round": per_round_metrics,
            "cold_requests": cold_agg,
            "warm_requests": warm_agg,
            "cache_benefit": {
                "cold_e2e_mean": cold_agg.get("e2e_ms", {}).get("mean", 0),
                "warm_e2e_mean": warm_agg.get("e2e_ms", {}).get("mean", 0),
                "cold_ttft_mean": cold_agg.get("ttft_ms", {}).get("mean", 0),
                "warm_ttft_mean": warm_agg.get("ttft_ms", {}).get("mean", 0),
                "e2e_speedup": (
                    cold_agg.get("e2e_ms", {}).get("mean", 1)
                    / warm_agg.get("e2e_ms", {}).get("mean", 1)
                    if warm_agg.get("e2e_ms", {}).get("mean", 0) > 0 else 0
                ),
                "ttft_speedup": (
                    cold_agg.get("ttft_ms", {}).get("mean", 1)
                    / warm_agg.get("ttft_ms", {}).get("mean", 1)
                    if warm_agg.get("ttft_ms", {}).get("mean", 0) > 0 else 0
                ),
            },
        },
    )

    filename = f"multi_turn_{num_users}u_{num_rounds}r_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = Path(output_dir) / filename
    with open(filepath, "w") as f:
        json.dump(asdict(bench_result), f, indent=2)

    print(f"\n--- Multi-Turn Conversation Results ---")
    print(f"  Users: {num_users}, Rounds: {num_rounds}")
    print(f"  Total Requests: {len(all_results)}")
    print(f"  Round 1 (cold) TTFT: {cold_agg.get('ttft_ms', {}).get('mean', 0):.0f}ms")
    print(f"  Round 2+ (warm) TTFT: {warm_agg.get('ttft_ms', {}).get('mean', 0):.0f}ms")
    print(f"  TTFT Speedup: {bench_result.metrics['cache_benefit']['ttft_speedup']:.2f}x")
    print(f"  Round 1 (cold) E2E: {cold_agg.get('e2e_ms', {}).get('mean', 0):.0f}ms")
    print(f"  Round 2+ (warm) E2E: {warm_agg.get('e2e_ms', {}).get('mean', 0):.0f}ms")
    print(f"  E2E Speedup: {bench_result.metrics['cache_benefit']['e2e_speedup']:.2f}x")
    print(f"  FSx cache: {cache_before['size_mb']} MB → {cache_after['size_mb']} MB")
    print(f"  Saved: {filepath}")

    # Print per-round summary
    print(f"\n  Per-Round TTFT Progression:")
    for rm in per_round_metrics:
        bar = "█" * max(1, int(rm["ttft_mean"] / 20))
        print(f"    Round {rm['round']:2d} ({rm['approx_context_tokens']:5d} tok): "
              f"{rm['ttft_mean']:7.0f}ms {bar}")

    return bench_result


async def run_shared_prompt_sweep(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    tenant_counts: list[int] = None,
    users_per_tenant: int = 5,
    system_prompt_tokens: int = 4000,
    output_dir: str = "results/kimi-k2.5-p5e",
    config: str = "baseline",
) -> list[BenchmarkResult]:
    """Run multi-tenant benchmark with varying tenant counts.

    Finds the sweet spot where LMCache provides maximum benefit.
    Fewer tenants = more prefix reuse = more cache benefit.
    More tenants = cache pressure = less benefit.
    """
    if tenant_counts is None:
        tenant_counts = [5, 10, 25, 50]

    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    all_results = []

    print(f"\n{'#'*70}")
    print(f"# SHARED PROMPT SWEEP: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Tenant counts: {tenant_counts}")
    print(f"# Users per tenant: {users_per_tenant}")
    print(f"# System prompt: ~{system_prompt_tokens} tokens each")
    print(f"{'#'*70}\n")

    for num_tenants in tenant_counts:
        print(f"\n{'='*60}")
        print(f"Sweep: {num_tenants} tenants × {users_per_tenant} users = "
              f"{num_tenants * users_per_tenant} requests")
        print(f"{'='*60}")

        result = await run_multi_tenant(
            endpoint=endpoint,
            model=model,
            num_tenants=num_tenants,
            users_per_tenant=users_per_tenant,
            system_prompt_tokens=system_prompt_tokens,
            output_dir=output_dir,
            config=config,
        )
        all_results.append(result)

        print("\nCooldown: 20s...")
        await asyncio.sleep(20)

    # Print sweep summary
    print(f"\n{'='*70}")
    print(f"SHARED PROMPT SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Tenants':>8} {'Cold TTFT':>12} {'Warm TTFT':>12} {'TTFT Speedup':>14} {'E2E Speedup':>13}")
    print(f"{'-'*8:>8} {'-'*12:>12} {'-'*12:>12} {'-'*14:>14} {'-'*13:>13}")
    for r in all_results:
        cold_ttft = r.metrics.get("cold_requests", {}).get("ttft_ms", {}).get("mean", 0)
        warm_ttft = r.metrics.get("warm_requests", {}).get("ttft_ms", {}).get("mean", 0)
        ttft_speedup = cold_ttft / warm_ttft if warm_ttft > 0 else 0
        e2e_speedup = r.metrics.get("cache_benefit", {}).get("speedup", 0)
        tenants = r.config.get("num_tenants", "?")
        print(f"{tenants:>8} {cold_ttft:>10.0f}ms {warm_ttft:>10.0f}ms {ttft_speedup:>12.2f}x {e2e_speedup:>11.2f}x")

    return all_results


async def run_cold_start_recovery(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    num_warmup_requests: int = 20,
    num_test_requests: int = 10,
    system_prompt_tokens: int = 4000,
    output_dir: str = "results/kimi-k2.5-p5e",
    config: str = "baseline",
) -> BenchmarkResult:
    """Test cold start recovery with pre-populated FSx cache.

    Measures the benefit of persistent FSx cache when vLLM's in-memory
    prefix cache is cold (e.g., after pod restart).

    Protocol:
    1. Send warmup requests to populate FSx cache (and vLLM prefix cache)
    2. Clear vLLM's in-memory cache by sending diverse filler requests
    3. Re-send the SAME requests and measure TTFT (should hit FSx cache)
    4. Compare warmup TTFT vs recovery TTFT

    This demonstrates LMCache's production value: cache persists across
    pod restarts, scaling events, and rolling updates.
    """
    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    output_dir = f"{output_dir}/{config}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# COLD START RECOVERY: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Warmup requests: {num_warmup_requests}")
    print(f"# Test requests: {num_test_requests}")
    print(f"# System prompt: ~{system_prompt_tokens} tokens")
    print(f"{'#'*70}\n")

    cache_before = measure_fsx_cache(config)

    # Build a shared system prompt (long, unique per test run)
    system_prompt = (
        "You are an enterprise AI assistant for Acme Corp. "
        "Your role is to assist with technical documentation, code reviews, "
        "and architectural decisions. Always provide thorough, detailed responses. "
    ) + generate_context(system_prompt_tokens)

    test_questions = [
        "What is the best approach for database migration?",
        "How should we handle authentication in microservices?",
        "Describe our deployment pipeline best practices.",
        "What monitoring strategy should we use?",
        "How do we handle secrets management?",
        "What's the recommended approach for API versioning?",
        "How should we implement rate limiting?",
        "Describe our disaster recovery plan.",
        "What CI/CD tools should we standardize on?",
        "How do we handle cross-service communication?",
    ]

    async with aiohttp.ClientSession() as session:
        # Phase 1: Warmup — populate both in-memory and FSx cache
        print("  Phase 1: Populating cache (warmup requests)...")
        warmup_results = []
        for i in range(num_warmup_requests):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test_questions[i % len(test_questions)]},
            ]
            result = await make_request(session, endpoint, model, messages, max_tokens=150)
            warmup_results.append(result)
            await asyncio.sleep(0.5)
            if (i + 1) % 5 == 0:
                print(f"    Warmup: {i+1}/{num_warmup_requests}")

        warmup_agg = aggregate_results(warmup_results)
        cache_after_warmup = measure_fsx_cache(config)
        print(f"  Warmup TTFT mean: {warmup_agg.get('ttft_ms', {}).get('mean', 0):.0f}ms")
        print(f"  FSx cache: {cache_after_warmup['size_mb']} MB, {cache_after_warmup['file_count']} files")

        # Phase 2: Evict in-memory cache by flooding with diverse requests
        print("\n  Phase 2: Evicting in-memory prefix cache (diverse requests)...")
        eviction_prompts = [generate_context(4000) for _ in range(30)]
        for i, ep in enumerate(eviction_prompts):
            messages = [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": ep[:16000]},  # unique context to evict prefix cache
            ]
            await make_request(session, endpoint, model, messages, max_tokens=10)
            if (i + 1) % 10 == 0:
                print(f"    Eviction: {i+1}/{len(eviction_prompts)}")

        print("  In-memory cache evicted (filled with diverse prefixes)")

        # Phase 3: Recovery — re-send original requests (should hit FSx cache)
        print("\n  Phase 3: Recovery requests (same prompts, cold in-memory, warm FSx)...")
        recovery_results = []
        for i in range(num_test_requests):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test_questions[i % len(test_questions)]},
            ]
            result = await make_request(session, endpoint, model, messages, max_tokens=150)
            recovery_results.append(result)
            await asyncio.sleep(0.5)

        recovery_agg = aggregate_results(recovery_results)
        cache_after = measure_fsx_cache(config)

    # Compare warmup (first time) vs recovery (from FSx)
    # Use only the first num_test_requests from warmup for fair comparison
    first_warmup = aggregate_results(warmup_results[:num_test_requests])

    bench_result = BenchmarkResult(
        model=model,
        instance_type="p5e.48xlarge",
        workload="cold_start_recovery",
        qps=2.0,
        timestamp=datetime.now().isoformat(),
        config={
            "serving_config": config,
            "serving_name": serving_config["name"],
            "num_warmup_requests": num_warmup_requests,
            "num_test_requests": num_test_requests,
            "system_prompt_tokens": system_prompt_tokens,
            "num_eviction_requests": len(eviction_prompts),
            "fsx_cache_before": cache_before,
            "fsx_cache_after_warmup": cache_after_warmup,
            "fsx_cache_after": cache_after,
        },
        num_requests=num_test_requests,
        metrics={
            **recovery_agg,
            "warmup_metrics": first_warmup,
            "recovery_metrics": recovery_agg,
            "cache_benefit": {
                "warmup_ttft_mean": first_warmup.get("ttft_ms", {}).get("mean", 0),
                "recovery_ttft_mean": recovery_agg.get("ttft_ms", {}).get("mean", 0),
                "warmup_e2e_mean": first_warmup.get("e2e_ms", {}).get("mean", 0),
                "recovery_e2e_mean": recovery_agg.get("e2e_ms", {}).get("mean", 0),
                "ttft_recovery_ratio": (
                    recovery_agg.get("ttft_ms", {}).get("mean", 0)
                    / first_warmup.get("ttft_ms", {}).get("mean", 1)
                    if first_warmup.get("ttft_ms", {}).get("mean", 0) > 0 else 0
                ),
                "e2e_recovery_ratio": (
                    recovery_agg.get("e2e_ms", {}).get("mean", 0)
                    / first_warmup.get("e2e_ms", {}).get("mean", 1)
                    if first_warmup.get("e2e_ms", {}).get("mean", 0) > 0 else 0
                ),
            },
        },
    )

    filename = f"cold_start_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = Path(output_dir) / filename
    with open(filepath, "w") as f:
        json.dump(asdict(bench_result), f, indent=2)

    warmup_ttft = first_warmup.get("ttft_ms", {}).get("mean", 0)
    recovery_ttft = recovery_agg.get("ttft_ms", {}).get("mean", 0)
    warmup_e2e = first_warmup.get("e2e_ms", {}).get("mean", 0)
    recovery_e2e = recovery_agg.get("e2e_ms", {}).get("mean", 0)

    print(f"\n--- Cold Start Recovery Results ---")
    print(f"  {'':20} {'First Time':>12} {'Recovery':>12} {'Ratio':>8}")
    print(f"  {'TTFT':20} {warmup_ttft:>10.0f}ms {recovery_ttft:>10.0f}ms {recovery_ttft/warmup_ttft if warmup_ttft > 0 else 0:>7.2f}x")
    print(f"  {'E2E':20} {warmup_e2e:>10.0f}ms {recovery_e2e:>10.0f}ms {recovery_e2e/warmup_e2e if warmup_e2e > 0 else 0:>7.2f}x")
    print(f"  FSx cache: {cache_after['size_mb']} MB, {cache_after['file_count']} files")
    if recovery_ttft < warmup_ttft:
        print(f"  ✓ Recovery faster than first time — FSx cache working!")
    else:
        print(f"  ✗ Recovery not faster — FSx cache may not be providing benefit")
    print(f"  Saved: {filepath}")

    return bench_result


async def run_document_library_rag(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    num_documents: int = 15,
    document_tokens: int = 2000,
    num_queries: int = 60,
    docs_per_query: int = 3,
    output_dir: str = "results/kimi-k2.5-p5e",
    config: str = "baseline",
) -> BenchmarkResult:
    """Run RAG benchmark with a shared document library.

    Simulates a retrieval-augmented generation system where:
    - A library of documents exists (e.g., company policies, FAQs, product docs)
    - Each query selects 2-3 documents as context (via retrieval)
    - Popular documents appear in many queries (Zipf distribution)
    - LMCache benefit: popular docs are cached after first access

    This differs from long_context_rag which uses one monolithic context.
    Real RAG has overlapping document selections across queries.

    Expected: 2-4x TTFT reduction as popular docs get cached.
    """
    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    output_dir = f"{output_dir}/{config}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# DOCUMENT LIBRARY RAG: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Documents: {num_documents} × ~{document_tokens} tokens")
    print(f"# Queries: {num_queries}, {docs_per_query} docs per query")
    print(f"{'#'*70}\n")

    cache_before = measure_fsx_cache(config)

    # Generate document library with distinct topics
    doc_topics = [
        "Cloud Infrastructure Architecture and Best Practices",
        "Security and Compliance Requirements for Financial Services",
        "Machine Learning Operations and Model Serving Pipeline",
        "Kubernetes Cluster Management and Scaling Strategies",
        "Database Design Patterns for High-Throughput Applications",
        "API Gateway Configuration and Rate Limiting Policies",
        "Disaster Recovery and Business Continuity Planning",
        "Container Image Security Scanning and Supply Chain",
        "Network Architecture with VPCs Subnets and Security Groups",
        "Cost Optimization Strategies for Cloud Workloads",
        "Observability Stack Setup with Prometheus Grafana Jaeger",
        "CI/CD Pipeline Design with GitOps and ArgoCD",
        "Data Lake Architecture with S3 Glue and Athena",
        "Identity and Access Management Best Practices",
        "Performance Testing and Load Generation Strategies",
    ]

    documents = []
    for i in range(num_documents):
        topic = doc_topics[i % len(doc_topics)]
        doc_content = (
            f"## Document {i+1}: {topic}\n\n"
            + generate_context(document_tokens)
        )
        documents.append(doc_content)

    # Generate queries with Zipf-like document selection
    # First few docs are "popular" (appear more often)
    questions = [
        "Based on the provided documents, what is the recommended architecture?",
        "What are the key security considerations mentioned in these documents?",
        "Summarize the main best practices described in the context.",
        "What deployment strategy is recommended and why?",
        "How should monitoring and alerting be configured according to the docs?",
        "What are the cost optimization opportunities identified?",
        "Describe the disaster recovery approach outlined in the documents.",
        "What compliance requirements are mentioned?",
        "How should the team handle scaling challenges?",
        "What are the performance benchmarks and targets described?",
    ]

    system_prompt = (
        "You are a technical documentation assistant. Answer questions "
        "using ONLY the provided documents. Cite document numbers in your response."
    )

    results = []
    doc_hit_count = [0] * num_documents

    async with aiohttp.ClientSession() as session:
        for q in range(num_queries):
            # Zipf-like selection: popular docs (low index) selected more often
            weights = [1.0 / (i + 1) for i in range(num_documents)]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            selected_indices = sorted(random.choices(
                range(num_documents), weights=weights, k=docs_per_query
            ))

            # Track hits
            for idx in selected_indices:
                doc_hit_count[idx] += 1

            # Build context from selected documents
            context = "\n\n---\n\n".join(documents[idx] for idx in selected_indices)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context + "\n\nQuestion: " + questions[q % len(questions)]},
            ]

            result = await make_request(session, endpoint, model, messages, max_tokens=200)
            results.append(result)

            await asyncio.sleep(0.5)  # ~2 QPS

            if (q + 1) % 15 == 0:
                success_rate = sum(1 for r in results if r.success) / len(results) * 100
                avg_ttft = statistics.mean([r.ttft_ms for r in results if r.success and r.ttft_ms > 0]) if any(r.success for r in results) else 0
                print(f"    Progress: {q+1}/{num_queries} ({success_rate:.0f}% success, avg TTFT: {avg_ttft:.0f}ms)")

    cache_after = measure_fsx_cache(config)
    aggregated = aggregate_results(results)

    # Split into early (cold) and late (warm) requests
    split_point = num_queries // 3
    early_results = results[:split_point]
    late_results = results[split_point:]
    early_agg = aggregate_results(early_results)
    late_agg = aggregate_results(late_results)

    bench_result = BenchmarkResult(
        model=model,
        instance_type="p5e.48xlarge",
        workload="document_library_rag",
        qps=2.0,
        timestamp=datetime.now().isoformat(),
        config={
            "serving_config": config,
            "serving_name": serving_config["name"],
            "num_documents": num_documents,
            "document_tokens": document_tokens,
            "docs_per_query": docs_per_query,
            "document_hit_distribution": doc_hit_count,
            "fsx_cache_before": cache_before,
            "fsx_cache_after": cache_after,
        },
        num_requests=len(results),
        metrics={
            **aggregated,
            "early_requests": early_agg,
            "late_requests": late_agg,
            "cache_benefit": {
                "early_ttft_mean": early_agg.get("ttft_ms", {}).get("mean", 0),
                "late_ttft_mean": late_agg.get("ttft_ms", {}).get("mean", 0),
                "early_e2e_mean": early_agg.get("e2e_ms", {}).get("mean", 0),
                "late_e2e_mean": late_agg.get("e2e_ms", {}).get("mean", 0),
                "ttft_speedup": (
                    early_agg.get("ttft_ms", {}).get("mean", 1)
                    / late_agg.get("ttft_ms", {}).get("mean", 1)
                    if late_agg.get("ttft_ms", {}).get("mean", 0) > 0 else 0
                ),
            },
        },
    )

    filename = f"doc_library_rag_{num_documents}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = Path(output_dir) / filename
    with open(filepath, "w") as f:
        json.dump(asdict(bench_result), f, indent=2)

    early_ttft = early_agg.get("ttft_ms", {}).get("mean", 0)
    late_ttft = late_agg.get("ttft_ms", {}).get("mean", 0)
    print(f"\n--- Document Library RAG Results ---")
    print(f"  Documents: {num_documents}, Queries: {num_queries}")
    print(f"  Early (cold) TTFT: {early_ttft:.0f}ms")
    print(f"  Late (warm) TTFT: {late_ttft:.0f}ms")
    print(f"  TTFT Speedup: {early_ttft/late_ttft if late_ttft > 0 else 0:.2f}x")
    print(f"  Doc hit distribution (top 5): {sorted(doc_hit_count, reverse=True)[:5]}")
    print(f"  FSx cache: {cache_before['size_mb']} MB → {cache_after['size_mb']} MB")
    print(f"  Saved: {filepath}")

    return bench_result


async def run_enterprise_api_gateway(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    num_requests: int = 50,
    api_schema_tokens: int = 8000,
    output_dir: str = "results/kimi-k2.5-p5e",
    config: str = "baseline",
) -> BenchmarkResult:
    """Run enterprise API gateway benchmark with large shared tool schema.

    Simulates an API gateway where ALL requests share a massive tool/API
    definition schema (~8K tokens). This is the maximum prefix sharing case:
    100% of the system prompt is shared across all users.

    Common in production:
    - Agentic systems with 50+ tool definitions
    - Enterprise APIs with OpenAPI schemas
    - Customer support bots with CRM integration schemas

    Expected: 3-5x TTFT reduction after first request caches the schema.
    """
    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    output_dir = f"{output_dir}/{config}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# ENTERPRISE API GATEWAY: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# API schema: ~{api_schema_tokens} tokens (shared across ALL requests)")
    print(f"# Requests: {num_requests}")
    print(f"{'#'*70}\n")

    cache_before = measure_fsx_cache(config)

    # Build a large API/tool schema system prompt
    tool_definitions = """You are an enterprise AI assistant with access to the following tools and APIs.
Use the appropriate tool to answer user questions.

## Available Tools

### 1. CustomerService
- `get_customer(customer_id: str) -> Customer` - Retrieve customer profile
- `update_customer(customer_id: str, fields: dict) -> Customer` - Update customer info
- `get_customer_orders(customer_id: str, limit: int = 10) -> List[Order]` - Get order history
- `create_support_ticket(customer_id: str, subject: str, description: str, priority: str) -> Ticket`
- `get_support_tickets(customer_id: str, status: str = "open") -> List[Ticket]`

### 2. InventoryService
- `get_product(sku: str) -> Product` - Get product details
- `check_availability(sku: str, warehouse: str = "all") -> Availability`
- `reserve_inventory(sku: str, quantity: int, order_id: str) -> Reservation`
- `get_warehouses() -> List[Warehouse]` - List all warehouses
- `transfer_inventory(sku: str, from_warehouse: str, to_warehouse: str, quantity: int) -> Transfer`

### 3. OrderService
- `create_order(customer_id: str, items: List[OrderItem]) -> Order`
- `get_order(order_id: str) -> Order` - Get order details
- `update_order_status(order_id: str, status: str) -> Order`
- `cancel_order(order_id: str, reason: str) -> Order`
- `get_shipping_options(order_id: str) -> List[ShippingOption]`

### 4. PaymentService
- `process_payment(order_id: str, method: str, amount: float) -> Payment`
- `refund_payment(payment_id: str, amount: float, reason: str) -> Refund`
- `get_payment_history(customer_id: str) -> List[Payment]`
- `validate_payment_method(method: str, details: dict) -> ValidationResult`

### 5. AnalyticsService
- `get_sales_report(start_date: str, end_date: str, granularity: str) -> Report`
- `get_customer_segments() -> List[Segment]`
- `get_product_performance(sku: str, period: str) -> Performance`
- `get_churn_prediction(customer_id: str) -> ChurnPrediction`

### 6. NotificationService
- `send_email(to: str, subject: str, template: str, context: dict) -> Notification`
- `send_sms(phone: str, message: str) -> Notification`
- `get_notification_preferences(customer_id: str) -> Preferences`
- `schedule_notification(notification: dict, send_at: str) -> ScheduledNotification`

### 7. AuthService
- `authenticate(username: str, password: str) -> AuthToken`
- `validate_token(token: str) -> TokenInfo`
- `get_permissions(user_id: str) -> List[Permission]`
- `create_api_key(user_id: str, scopes: List[str]) -> ApiKey`

### 8. SearchService
- `search_products(query: str, filters: dict = {}, limit: int = 20) -> SearchResults`
- `search_customers(query: str, filters: dict = {}) -> SearchResults`
- `get_suggestions(partial_query: str) -> List[Suggestion]`

## Data Models

### Customer
```json
{"customer_id": "string", "name": "string", "email": "string", "phone": "string",
 "tier": "bronze|silver|gold|platinum", "created_at": "datetime", "lifetime_value": "float",
 "preferences": {"language": "string", "currency": "string", "notifications": "dict"}}
```

### Order
```json
{"order_id": "string", "customer_id": "string", "items": [{"sku": "string", "quantity": "int", "price": "float"}],
 "status": "pending|processing|shipped|delivered|cancelled", "total": "float",
 "shipping_address": "Address", "created_at": "datetime", "updated_at": "datetime"}
```

### Product
```json
{"sku": "string", "name": "string", "description": "string", "price": "float",
 "category": "string", "tags": ["string"], "availability": "in_stock|low_stock|out_of_stock",
 "rating": "float", "reviews_count": "int"}
```

## Business Rules
1. Gold/Platinum customers get priority support (auto-escalate tickets)
2. Orders over $500 require manager approval for cancellation
3. Refunds must be processed within 30 days of purchase
4. Inventory reservations expire after 15 minutes
5. Free shipping for orders over $100
6. Maximum 5 items per order for standard accounts
7. API rate limit: 100 requests per minute per customer
"""

    # Pad with additional context to reach target token count
    padding = generate_context(max(0, api_schema_tokens - len(tool_definitions) // 4))
    full_schema = tool_definitions + "\n\n## Additional API Documentation\n\n" + padding

    # Diverse user queries that all share the same schema prefix
    user_queries = [
        "Look up customer C-12345 and show their recent orders.",
        "I need to process a refund for order ORD-67890. The reason is defective product.",
        "Check if SKU LAPTOP-PRO-15 is available in the east coast warehouse.",
        "Create a support ticket for customer C-54321 about a delayed shipment.",
        "What's the sales report for last quarter?",
        "Send an email notification to the customer about their order being shipped.",
        "Search for products matching 'wireless headphones' under $100.",
        "What are the permissions for user admin-001?",
        "Transfer 50 units of SKU TABLET-10 from warehouse-west to warehouse-east.",
        "Get the churn prediction for customer C-99999.",
        "Cancel order ORD-11111 because the customer changed their mind.",
        "What shipping options are available for order ORD-22222?",
        "Create a new order for customer C-33333: 2x LAPTOP-PRO-15 and 1x MOUSE-WIFI.",
        "Show me the notification preferences for customer C-44444.",
        "Generate an API key for user dev-team-lead with read and write scopes.",
    ]

    results = []

    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            messages = [
                {"role": "system", "content": full_schema},
                {"role": "user", "content": user_queries[i % len(user_queries)]},
            ]
            result = await make_request(session, endpoint, model, messages, max_tokens=200)
            results.append(result)
            await asyncio.sleep(0.33)  # ~3 QPS

            if (i + 1) % 10 == 0:
                success_rate = sum(1 for r in results if r.success) / len(results) * 100
                recent_ttft = statistics.mean([r.ttft_ms for r in results[-10:] if r.success and r.ttft_ms > 0]) if any(r.success for r in results[-10:]) else 0
                print(f"    Progress: {i+1}/{num_requests} ({success_rate:.0f}% success, recent TTFT: {recent_ttft:.0f}ms)")

    cache_after = measure_fsx_cache(config)
    aggregated = aggregate_results(results)

    # First 5 requests = cold, rest = warm
    cold_results = results[:5]
    warm_results = results[5:]
    cold_agg = aggregate_results(cold_results)
    warm_agg = aggregate_results(warm_results)

    bench_result = BenchmarkResult(
        model=model,
        instance_type="p5e.48xlarge",
        workload="enterprise_api_gateway",
        qps=3.0,
        timestamp=datetime.now().isoformat(),
        config={
            "serving_config": config,
            "serving_name": serving_config["name"],
            "api_schema_tokens": api_schema_tokens,
            "num_unique_queries": len(user_queries),
            "fsx_cache_before": cache_before,
            "fsx_cache_after": cache_after,
        },
        num_requests=len(results),
        metrics={
            **aggregated,
            "cold_requests": cold_agg,
            "warm_requests": warm_agg,
            "cache_benefit": {
                "cold_ttft_mean": cold_agg.get("ttft_ms", {}).get("mean", 0),
                "warm_ttft_mean": warm_agg.get("ttft_ms", {}).get("mean", 0),
                "ttft_speedup": (
                    cold_agg.get("ttft_ms", {}).get("mean", 1)
                    / warm_agg.get("ttft_ms", {}).get("mean", 1)
                    if warm_agg.get("ttft_ms", {}).get("mean", 0) > 0 else 0
                ),
                "cold_e2e_mean": cold_agg.get("e2e_ms", {}).get("mean", 0),
                "warm_e2e_mean": warm_agg.get("e2e_ms", {}).get("mean", 0),
            },
        },
    )

    filename = f"enterprise_api_gateway_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = Path(output_dir) / filename
    with open(filepath, "w") as f:
        json.dump(asdict(bench_result), f, indent=2)

    cold_ttft = cold_agg.get("ttft_ms", {}).get("mean", 0)
    warm_ttft = warm_agg.get("ttft_ms", {}).get("mean", 0)
    print(f"\n--- Enterprise API Gateway Results ---")
    print(f"  Schema: ~{api_schema_tokens} tokens shared across all {num_requests} requests")
    print(f"  Cold (first 5) TTFT: {cold_ttft:.0f}ms")
    print(f"  Warm (remaining) TTFT: {warm_ttft:.0f}ms")
    print(f"  TTFT Speedup: {cold_ttft/warm_ttft if warm_ttft > 0 else 0:.2f}x")
    print(f"  FSx cache: {cache_before['size_mb']} MB → {cache_after['size_mb']} MB")
    print(f"  Saved: {filepath}")

    return bench_result


async def run_conversation_resumption(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    num_users: int = 10,
    turns_before_gap: int = 5,
    turns_after_gap: int = 5,
    history_tokens_per_turn: int = 500,
    num_interleaved_requests: int = 40,
    output_dir: str = "results/kimi-k2.5-p5e",
    config: str = "baseline",
) -> BenchmarkResult:
    """Simulate conversation resumption after a gap.

    Scenario: Users have active conversations, leave for a while, then return.
    During the gap, other traffic evicts their context from vLLM's in-memory
    prefix cache. With LMCache+FSx, the conversation context persists on disk.

    Protocol:
    1. Each user has a 5-turn conversation (populates cache)
    2. Interleave diverse requests to evict in-memory cache
    3. Users resume their conversations (should hit FSx cache)
    4. Compare TTFT of last turn before gap vs first turn after gap

    Expected: Resumed turns have similar TTFT to pre-gap turns (FSx hit),
    much faster than a cold start without FSx.
    """
    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    output_dir = f"{output_dir}/{config}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# CONVERSATION RESUMPTION: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Users: {num_users}, Turns before gap: {turns_before_gap}")
    print(f"# Interleaved requests (gap): {num_interleaved_requests}")
    print(f"# Turns after resumption: {turns_after_gap}")
    print(f"{'#'*70}\n")

    cache_before = measure_fsx_cache(config)

    # User system prompts (unique per user)
    system_prompts = [
        f"You are a personal AI assistant for User {u+1}. "
        f"Remember context from previous messages and provide consistent help."
        for u in range(num_users)
    ]

    topics = [
        "Let's discuss our project architecture.",
        "Can you help me optimize our database queries?",
        "I need advice on our deployment strategy.",
        "Let's review the security audit findings.",
        "Help me plan the team offsite agenda.",
    ]

    filler = generate_context(history_tokens_per_turn)

    pre_gap_results = []
    post_gap_results = []
    conversation_histories = [[] for _ in range(num_users)]

    async with aiohttp.ClientSession() as session:
        # Phase 1: Active conversations (turns_before_gap per user)
        print("  Phase 1: Active conversations (pre-gap)...")
        for turn in range(turns_before_gap):
            for user_id in range(num_users):
                messages = [{"role": "system", "content": system_prompts[user_id]}]
                messages.extend(conversation_histories[user_id])
                messages.append({
                    "role": "user",
                    "content": topics[turn % len(topics)] + f" (Turn {turn+1})"
                })

                result = await make_request(session, endpoint, model, messages, max_tokens=150)
                pre_gap_results.append(result)

                # Add to conversation history
                conversation_histories[user_id].append({
                    "role": "user",
                    "content": topics[turn % len(topics)] + f" (Turn {turn+1})"
                })
                conversation_histories[user_id].append({
                    "role": "assistant",
                    "content": filler[:history_tokens_per_turn * 4]
                })

                await asyncio.sleep(0.25)

            if (turn + 1) % 2 == 0:
                avg_ttft = statistics.mean([r.ttft_ms for r in pre_gap_results[-num_users:] if r.success and r.ttft_ms > 0]) if pre_gap_results else 0
                print(f"    Turn {turn+1}/{turns_before_gap}: avg TTFT {avg_ttft:.0f}ms")

        pre_gap_agg = aggregate_results(pre_gap_results)
        cache_after_active = measure_fsx_cache(config)
        print(f"  Pre-gap TTFT mean: {pre_gap_agg.get('ttft_ms', {}).get('mean', 0):.0f}ms")
        print(f"  FSx cache: {cache_after_active['size_mb']} MB")

        # Phase 2: Gap — interleaved diverse traffic to evict in-memory cache
        print(f"\n  Phase 2: Simulating gap ({num_interleaved_requests} diverse requests)...")
        for i in range(num_interleaved_requests):
            unique_context = generate_context(3000)
            messages = [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": unique_context[:12000]},
            ]
            await make_request(session, endpoint, model, messages, max_tokens=10)
            if (i + 1) % 10 == 0:
                print(f"    Gap traffic: {i+1}/{num_interleaved_requests}")

        print("  Gap complete — in-memory cache likely evicted")

        # Phase 3: Resume conversations
        print(f"\n  Phase 3: Conversation resumption (post-gap)...")
        for turn in range(turns_after_gap):
            for user_id in range(num_users):
                messages = [{"role": "system", "content": system_prompts[user_id]}]
                messages.extend(conversation_histories[user_id])
                messages.append({
                    "role": "user",
                    "content": f"Continuing our discussion — what were the key points from turn {turns_before_gap}? (Resumed turn {turn+1})"
                })

                result = await make_request(session, endpoint, model, messages, max_tokens=150)
                post_gap_results.append(result)

                # Update history
                conversation_histories[user_id].append({
                    "role": "user",
                    "content": f"Resumed turn {turn+1}"
                })
                conversation_histories[user_id].append({
                    "role": "assistant",
                    "content": filler[:history_tokens_per_turn * 4]
                })

                await asyncio.sleep(0.25)

            if (turn + 1) % 2 == 0:
                avg_ttft = statistics.mean([r.ttft_ms for r in post_gap_results[-num_users:] if r.success and r.ttft_ms > 0]) if post_gap_results else 0
                print(f"    Resumed turn {turn+1}/{turns_after_gap}: avg TTFT {avg_ttft:.0f}ms")

    cache_after = measure_fsx_cache(config)
    post_gap_agg = aggregate_results(post_gap_results)

    # Compare last pre-gap turn vs first post-gap turn
    last_pre_gap = aggregate_results(pre_gap_results[-num_users:])
    first_post_gap = aggregate_results(post_gap_results[:num_users])

    bench_result = BenchmarkResult(
        model=model,
        instance_type="p5e.48xlarge",
        workload="conversation_resumption",
        qps=4.0,
        timestamp=datetime.now().isoformat(),
        config={
            "serving_config": config,
            "serving_name": serving_config["name"],
            "num_users": num_users,
            "turns_before_gap": turns_before_gap,
            "turns_after_gap": turns_after_gap,
            "history_tokens_per_turn": history_tokens_per_turn,
            "num_interleaved_requests": num_interleaved_requests,
            "fsx_cache_before": cache_before,
            "fsx_cache_after_active": cache_after_active,
            "fsx_cache_after": cache_after,
        },
        num_requests=len(pre_gap_results) + len(post_gap_results),
        metrics={
            "pre_gap": pre_gap_agg,
            "post_gap": post_gap_agg,
            "last_turn_before_gap": last_pre_gap,
            "first_turn_after_gap": first_post_gap,
            "cache_benefit": {
                "pre_gap_ttft_mean": pre_gap_agg.get("ttft_ms", {}).get("mean", 0),
                "post_gap_ttft_mean": post_gap_agg.get("ttft_ms", {}).get("mean", 0),
                "last_pre_gap_ttft": last_pre_gap.get("ttft_ms", {}).get("mean", 0),
                "first_post_gap_ttft": first_post_gap.get("ttft_ms", {}).get("mean", 0),
                "resumption_ratio": (
                    first_post_gap.get("ttft_ms", {}).get("mean", 0)
                    / last_pre_gap.get("ttft_ms", {}).get("mean", 1)
                    if last_pre_gap.get("ttft_ms", {}).get("mean", 0) > 0 else 0
                ),
            },
        },
    )

    filename = f"conversation_resumption_{num_users}u_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = Path(output_dir) / filename
    with open(filepath, "w") as f:
        json.dump(asdict(bench_result), f, indent=2)

    last_pre_ttft = last_pre_gap.get("ttft_ms", {}).get("mean", 0)
    first_post_ttft = first_post_gap.get("ttft_ms", {}).get("mean", 0)
    ratio = first_post_ttft / last_pre_ttft if last_pre_ttft > 0 else 0

    print(f"\n--- Conversation Resumption Results ---")
    print(f"  Last turn before gap TTFT: {last_pre_ttft:.0f}ms")
    print(f"  First turn after gap TTFT: {first_post_ttft:.0f}ms")
    print(f"  Resumption ratio: {ratio:.2f}x (1.0 = perfect recovery from FSx)")
    if ratio < 1.5:
        print(f"  ✓ Good recovery — FSx cache effectively preserved context")
    elif ratio < 2.5:
        print(f"  ~ Partial recovery — some FSx benefit but not full")
    else:
        print(f"  ✗ Poor recovery — FSx cache may not be helping")
    print(f"  FSx cache: {cache_before['size_mb']} MB → {cache_after['size_mb']} MB")
    print(f"  Saved: {filepath}")

    return bench_result


async def run_lmcache_showcase(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    config: str = "baseline",
    output_dir: str = "results/kimi-k2.5-p5e",
):
    """Run benchmarks that best highlight LMCache's production value.

    7 tests covering all high-value LMCache+FSx use cases:
    1. Multi-turn conversation (20 rounds, growing history) — incremental caching
    2. Enterprise API gateway (8K shared tool schema) — max prefix sharing
    3. Document library RAG (15 docs, Zipf access) — popular doc caching
    4. Conversation resumption (gap + diverse traffic) — FSx persistence
    5. Shared prompt sweep (5/10/25 tenants) — find sweet spot
    6. Cold start recovery (evict + recover from FSx) — persistence value
    7. Strict no-reuse (control) — verify minimal overhead
    """
    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    config_output = f"{output_dir}/{config}"
    Path(config_output).mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# LMCACHE SHOWCASE: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Tests: 7 high-value LMCache+FSx use cases")
    print(f"# Output: {config_output}")
    print(f"{'#'*70}\n")

    initial_cache = measure_fsx_cache(config)
    print(f"Initial FSx cache: {initial_cache['size_mb']} MB, {initial_cache['file_count']} files\n")

    # Test 1: Multi-turn conversation (LMCache's strongest use case)
    print("=" * 70)
    print("TEST 1/7: Multi-Turn Conversation (20 rounds, growing history)")
    print("  Why: Each turn reuses KV from all previous turns → ~95% hit by turn 20")
    print("=" * 70)
    await run_multi_turn_conversation(
        endpoint=endpoint,
        model=model,
        num_users=8,
        num_rounds=20,
        history_tokens_per_round=1000,
        output_dir=output_dir,
        config=config,
    )

    print("\nCooldown: 20s...")
    await asyncio.sleep(20)

    # Test 2: Enterprise API gateway (maximum prefix sharing)
    print("\n" + "=" * 70)
    print("TEST 2/7: Enterprise API Gateway (8K shared tool schema)")
    print("  Why: 100% prefix sharing — every request reuses the same 8K schema")
    print("=" * 70)
    await run_enterprise_api_gateway(
        endpoint=endpoint,
        model=model,
        num_requests=50,
        api_schema_tokens=8000,
        output_dir=output_dir,
        config=config,
    )

    print("\nCooldown: 20s...")
    await asyncio.sleep(20)

    # Test 3: Document library RAG (overlapping document selections)
    print("\n" + "=" * 70)
    print("TEST 3/7: Document Library RAG (15 docs, Zipf access pattern)")
    print("  Why: Popular documents get cached → subsequent queries skip prefill")
    print("=" * 70)
    await run_document_library_rag(
        endpoint=endpoint,
        model=model,
        num_documents=15,
        document_tokens=2000,
        num_queries=60,
        docs_per_query=3,
        output_dir=output_dir,
        config=config,
    )

    print("\nCooldown: 20s...")
    await asyncio.sleep(20)

    # Test 4: Conversation resumption (FSx persistence across traffic gaps)
    print("\n" + "=" * 70)
    print("TEST 4/7: Conversation Resumption (gap + diverse traffic)")
    print("  Why: FSx persists context when in-memory cache is evicted by other users")
    print("=" * 70)
    await run_conversation_resumption(
        endpoint=endpoint,
        model=model,
        num_users=10,
        turns_before_gap=5,
        turns_after_gap=5,
        num_interleaved_requests=40,
        output_dir=output_dir,
        config=config,
    )

    print("\nCooldown: 20s...")
    await asyncio.sleep(20)

    # Test 5: Shared prompt sweep (find the sweet spot)
    print("\n" + "=" * 70)
    print("TEST 5/7: Shared Prompt Sweep (5/10/25 tenants)")
    print("  Why: Find the optimal tenant count for cache benefit vs pressure")
    print("=" * 70)
    await run_shared_prompt_sweep(
        endpoint=endpoint,
        model=model,
        tenant_counts=[5, 10, 25],
        users_per_tenant=5,
        output_dir=output_dir,
        config=config,
    )

    print("\nCooldown: 20s...")
    await asyncio.sleep(20)

    # Test 6: Cold start recovery (persistent cache value)
    print("\n" + "=" * 70)
    print("TEST 6/7: Cold Start Recovery (evict in-memory, recover from FSx)")
    print("  Why: Demonstrates cache persistence across pod restarts/scaling")
    print("=" * 70)
    await run_cold_start_recovery(
        endpoint=endpoint,
        model=model,
        num_warmup_requests=20,
        num_test_requests=10,
        output_dir=output_dir,
        config=config,
    )

    print("\nCooldown: 20s...")
    await asyncio.sleep(20)

    # Test 7: Strict no-reuse control (verify minimal overhead)
    print("\n" + "=" * 70)
    print("TEST 7/7: Strict No-Reuse (control — verify minimal LMCache overhead)")
    print("  Why: Confirms LMCache adds near-zero overhead for non-reuse workloads")
    print("=" * 70)
    await run_benchmark_suite(
        endpoint=endpoint,
        model=model,
        workloads=["strict_no_reuse"],
        qps_levels=["medium"],
        requests_per_workload=30,
        output_dir=output_dir,
        config=config,
    )

    final_cache = measure_fsx_cache(config)
    print(f"\n{'#'*70}")
    print(f"# LMCACHE SHOWCASE COMPLETE")
    print(f"# FSx cache: {initial_cache['size_mb']} MB → {final_cache['size_mb']} MB")
    print(f"# Cache files: {initial_cache['file_count']} → {final_cache['file_count']}")
    print(f"# Results in: {config_output}")
    print(f"{'#'*70}")


async def run_kv_cache_benchmark(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    config: str = "baseline",
    output_dir: str = "results/kimi-k2.5-p5e",
    requests_per_workload: int = 15,
):
    """Lean benchmark focused on KV cache offloading comparison across frameworks.

    Runs only workloads that exercise prefix caching and KV cache offloading:
    1. multi_turn_qa @ medium QPS  (shared context, prefix hits)
    2. long_context_rag @ medium QPS  (20K context, offloading benefit)
    3. strict_no_reuse @ medium QPS  (0% reuse control)
    4. Long context scaling (16K → 48K)
    5. Multi-tenant (50 tenants, cache pressure)
    """
    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    config_output = f"{output_dir}/{config}"
    Path(config_output).mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# KV CACHE BENCHMARK: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Output: {config_output}")
    print(f"{'#'*70}\n")

    initial_cache = measure_fsx_cache(config)
    print(f"Initial FSx cache: {initial_cache['size_mb']} MB, {initial_cache['file_count']} files\n")

    # Phase 1: KV-cache-relevant workloads at medium QPS only
    kv_workloads = ["multi_turn_qa", "long_context_rag", "strict_no_reuse"]
    print("=" * 70)
    print(f"PHASE 1: KV cache workloads ({', '.join(kv_workloads)}) @ medium QPS")
    print("=" * 70)
    await run_benchmark_suite(
        endpoint=endpoint,
        model=model,
        workloads=kv_workloads,
        qps_levels=["medium"],
        requests_per_workload=requests_per_workload,
        output_dir=output_dir,
        config=config,
    )

    # Phase 2: Long context scaling
    print("\n" + "=" * 70)
    print("PHASE 2: Long context scaling (16K → 48K)")
    print("=" * 70)
    await run_long_context_scaling(
        endpoint=endpoint,
        model=model,
        context_sizes=[16000, 24000, 36000, 48000],
        num_requests_per_size=10,
        warmup_requests=3,
        output_dir=output_dir,
        config=config,
    )

    # Phase 3: Multi-tenant
    print("\n" + "=" * 70)
    print("PHASE 3: Multi-tenant (50 tenants)")
    print("=" * 70)
    await run_multi_tenant(
        endpoint=endpoint,
        model=model,
        num_tenants=50,
        users_per_tenant=2,
        output_dir=output_dir,
        config=config,
    )

    final_cache = measure_fsx_cache(config)
    print(f"\n{'#'*70}")
    print(f"# KV CACHE BENCHMARK COMPLETE")
    print(f"# FSx cache: {initial_cache['size_mb']} MB → {final_cache['size_mb']} MB")
    print(f"# Cache files: {initial_cache['file_count']} → {final_cache['file_count']}")
    print(f"# Results in: {config_output}")
    print(f"{'#'*70}")


async def fetch_vllm_metrics(endpoint: str) -> dict:
    """Fetch vLLM Prometheus metrics and return key values."""
    metrics_url = endpoint.rstrip("/") + "/metrics"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                text = await resp.text()
                result = {}
                for line in text.split("\n"):
                    if line.startswith("#"):
                        continue
                    if "kv_cache_usage_perc" in line and "engine=" in line:
                        result["kv_cache_usage"] = float(line.split()[-1])
                    elif "num_requests_running" in line and "engine=" in line:
                        result["running"] = float(line.split()[-1])
                    elif "num_requests_waiting" in line and "engine=" in line:
                        result["waiting"] = float(line.split()[-1])
                    elif "num_preemptions_total" in line and not line.startswith("#"):
                        result["preemptions"] = float(line.split()[-1])
                    elif "num_gpu_blocks" in line and "num_gpu_blocks_override" not in line:
                        # Parse from cache_config_info label
                        import re
                        m = re.search(r'num_gpu_blocks="(\d+)"', line)
                        if m:
                            result["num_gpu_blocks"] = int(m.group(1))
                return result
    except Exception:
        return {}


# Prefix cache hit/miss metrics are now captured in run_benchmark_suite(),
# run_long_context_scaling(), and run_multi_tenant() via scrape_prefix_cache_metrics().
# Addresses Lesson #30 gap from the 2026-02-18/19 rounds.
#
# vLLM exposes: prefix_cache_hit_total, prefix_cache_miss_total, gpu_prefix_cache_hit_rate
# SGLang HiCache exposes: hicache_l1/l2/l3_hits/misses (per-tier)
#
async def scrape_prefix_cache_metrics(endpoint: str) -> dict:
    """Scrape prefix cache hit/miss counters from vLLM or SGLang /metrics endpoint.

    Returns cumulative counters — caller should diff before/after to get per-test rates.
    """
    metrics_url = endpoint.rstrip("/") + "/metrics"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                text = await resp.text()
                result = {}
                for line in text.split("\n"):
                    if line.startswith("#"):
                        continue
                    # vLLM prefix cache metrics
                    if "prefix_cache_hit_total" in line:
                        result["hits"] = float(line.split()[-1])
                    elif "prefix_cache_miss_total" in line:
                        result["misses"] = float(line.split()[-1])
                    elif "gpu_prefix_cache_hit_rate" in line:
                        result["hit_rate"] = float(line.split()[-1])
                    # SGLang HiCache per-tier metrics
                    elif "hicache_l1_hits" in line:
                        result["l1_hits"] = float(line.split()[-1])
                    elif "hicache_l1_misses" in line:
                        result["l1_misses"] = float(line.split()[-1])
                    elif "hicache_l2_hits" in line:
                        result["l2_hits"] = float(line.split()[-1])
                    elif "hicache_l2_misses" in line:
                        result["l2_misses"] = float(line.split()[-1])
                    elif "hicache_l3_hits" in line:
                        result["l3_hits"] = float(line.split()[-1])
                    elif "hicache_l3_misses" in line:
                        result["l3_misses"] = float(line.split()[-1])
                return result
    except Exception:
        return {}


async def run_memory_pressure_throughput(
    endpoint: str = ENDPOINT,
    model: str = MODEL_ID,
    config: str = "baseline",
    output_dir: str = "results/kimi-k2.5-p5e",
    num_background_sessions: int = 25,
    context_tokens_per_session: int = 24000,
    max_tokens_background: int = 500,
    num_foreground_bursts: int = 5,
    burst_size: int = 10,
    foreground_context_tokens: int = 4000,
):
    """Memory pressure benchmark: saturate GPU KV cache, then measure foreground throughput.

    This test demonstrates the real value of KV cache offloading to FSx+GDS:
    when GPU memory is full, offloading allows the system to evict cold KV blocks
    to external storage instead of recomputing prefill or queueing requests.

    Architecture:
    - Phase 1: Open N background sessions with long contexts (fill GPU KV cache)
    - Phase 2: While background sessions are generating, send bursts of foreground
      requests and measure TTFT, queue time, and throughput
    - Phase 3: Measure recovery — after background sessions finish, send new requests
      to contexts that were evicted during pressure

    With offloading (LMCache+GDS):
      - Cold KV evicted to FSx (~20ms reload vs ~300ms recompute)
      - Foreground requests admitted immediately
      - Higher sustained throughput

    Without offloading (baseline):
      - Requests queue or preempt existing sessions
      - Preempted sessions must recompute prefill from scratch
      - Lower throughput, higher tail latency
    """
    serving_config = SERVING_CONFIGS.get(config, SERVING_CONFIGS["baseline"])
    config_output = f"{output_dir}/{config}"
    Path(config_output).mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# MEMORY PRESSURE THROUGHPUT: Kimi-K2.5 on p5e.48xlarge")
    print(f"# Serving Config: {serving_config['name']} ({config})")
    print(f"# Background sessions: {num_background_sessions} × ~{context_tokens_per_session} tokens")
    print(f"# Foreground bursts: {num_foreground_bursts} × {burst_size} requests")
    print(f"# GPU KV blocks: check /metrics for capacity")
    print(f"{'#'*70}\n")

    initial_cache = measure_fsx_cache(config)
    initial_metrics = await fetch_vllm_metrics(endpoint)
    initial_preemptions = initial_metrics.get("preemptions", 0)
    num_gpu_blocks = initial_metrics.get("num_gpu_blocks", 0)
    kv_capacity_tokens = num_gpu_blocks * 64  # block_size=64

    print(f"  GPU KV capacity: {num_gpu_blocks} blocks × 64 = {kv_capacity_tokens:,} tokens")
    print(f"  Background load: {num_background_sessions} × {context_tokens_per_session} = {num_background_sessions * context_tokens_per_session:,} tokens")
    fill_ratio = (num_background_sessions * context_tokens_per_session) / kv_capacity_tokens if kv_capacity_tokens else 0
    print(f"  Target fill ratio: {fill_ratio:.1%}")
    print(f"  Initial KV usage: {initial_metrics.get('kv_cache_usage', 0):.1%}")
    print(f"  Initial preemptions: {initial_preemptions}")
    print(f"  FSx cache: {initial_cache['size_mb']} MB\n")

    # Generate diverse background contexts (each unique to prevent prefix sharing)
    background_contexts = []
    topics = [
        "quantum computing and error correction", "distributed database sharding",
        "climate modeling with machine learning", "autonomous vehicle perception systems",
        "protein folding prediction methods", "financial risk modeling and VaR",
        "natural language processing transformers", "supply chain optimization algorithms",
        "satellite image analysis techniques", "cybersecurity threat detection systems",
        "robotic manipulation and grasping", "gene editing with CRISPR technology",
        "neural architecture search methods", "smart grid energy management",
        "social network graph analysis", "medical imaging with deep learning",
        "reinforcement learning for games", "blockchain consensus mechanisms",
        "time series forecasting models", "computational fluid dynamics",
        "edge computing optimization", "recommender system algorithms",
        "speech recognition advances", "materials science simulation",
        "epidemic modeling approaches", "ocean current prediction models",
        "quantum chemistry orbital calculations", "wireless mesh network routing",
        "compiler optimization techniques", "fusion energy plasma containment",
        "seismological wave propagation", "neural prosthetics signal processing",
        "algorithmic trading strategies", "urban traffic flow modeling",
        "DNA sequencing alignment methods", "cloud infrastructure autoscaling",
        "gravitational wave detection", "adversarial machine learning defense",
        "sustainable agriculture analytics", "lithium battery degradation models",
        "autonomous drone swarm coordination", "phylogenetic tree reconstruction",
        "radio telescope signal processing", "protein-protein interaction networks",
        "federated learning privacy guarantees", "photonic computing architectures",
        "glacier retreat measurement techniques", "dark matter simulation methods",
        "real-time language translation systems", "earthquake early warning networks",
        "gene regulatory network inference", "spacecraft trajectory optimization",
    ]
    for i in range(num_background_sessions):
        topic = topics[i % len(topics)]
        ctx = generate_context(context_tokens_per_session)
        # Prepend unique topic to prevent prefix sharing across sessions
        background_contexts.append(
            f"[Session {i+1}: Expert analysis of {topic}]\n\n{ctx}"
        )

    # ── Phase 1: Launch background sessions to fill GPU KV cache ──
    print("  Phase 1: Filling GPU KV cache with background sessions...")

    async def background_session(session, idx, context):
        """A long-running background session that occupies GPU KV cache."""
        messages = [
            {"role": "system", "content": f"You are an expert analyst for session {idx}. Provide detailed analysis."},
            {"role": "user", "content": context + f"\n\nProvide a comprehensive analysis of this information in session {idx}."},
        ]
        return await make_request(session, endpoint, model, messages,
                                   max_tokens=max_tokens_background, temperature=0.7)

    # Launch all background sessions concurrently
    async with aiohttp.ClientSession() as session:
        bg_start = time.perf_counter()
        bg_tasks = [
            asyncio.create_task(background_session(session, i, ctx))
            for i, ctx in enumerate(background_contexts)
        ]

        # Wait a few seconds for requests to start processing and fill KV cache
        await asyncio.sleep(5)
        pressure_metrics = await fetch_vllm_metrics(endpoint)
        print(f"    KV usage after launch: {pressure_metrics.get('kv_cache_usage', 0):.1%}")
        print(f"    Running: {pressure_metrics.get('running', 0):.0f}, Waiting: {pressure_metrics.get('waiting', 0):.0f}")

        # ── Phase 2: Send foreground bursts while background is running ──
        print(f"\n  Phase 2: Sending {num_foreground_bursts} foreground bursts (GPU under pressure)...")

        foreground_results = []
        metrics_snapshots = []

        for burst_idx in range(num_foreground_bursts):
            # Check metrics before burst
            pre_burst = await fetch_vllm_metrics(endpoint)
            metrics_snapshots.append(pre_burst)

            # Send a burst of foreground requests with moderate context
            fg_context = generate_context(foreground_context_tokens)
            burst_tasks = []
            for req_idx in range(burst_size):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Be concise."},
                    {"role": "user", "content": f"[Query {burst_idx*burst_size + req_idx + 1}] "
                     f"Given this context:\n{fg_context[:foreground_context_tokens*2]}\n\n"
                     f"Summarize the key points in 2-3 sentences."},
                ]
                burst_tasks.append(
                    make_request(session, endpoint, model, messages, max_tokens=100, temperature=0.7)
                )

            burst_start = time.perf_counter()
            burst_results = await asyncio.gather(*burst_tasks)
            burst_elapsed = (time.perf_counter() - burst_start) * 1000

            successful = [r for r in burst_results if r.success]
            failed = [r for r in burst_results if not r.success]

            if successful:
                avg_ttft = statistics.mean(r.ttft_ms for r in successful)
                p99_ttft = sorted(r.ttft_ms for r in successful)[int(len(successful) * 0.99)]
                avg_e2e = statistics.mean(r.e2e_ms for r in successful)
            else:
                avg_ttft = p99_ttft = avg_e2e = 0

            post_burst = await fetch_vllm_metrics(endpoint)
            preemptions_delta = post_burst.get("preemptions", 0) - pre_burst.get("preemptions", 0)

            print(f"    Burst {burst_idx+1}/{num_foreground_bursts}: "
                  f"TTFT avg={avg_ttft:.0f}ms p99={p99_ttft:.0f}ms, "
                  f"E2E avg={avg_e2e:.0f}ms, "
                  f"success={len(successful)}/{burst_size}, "
                  f"preemptions=+{preemptions_delta:.0f}, "
                  f"KV={post_burst.get('kv_cache_usage', 0):.0%}")

            foreground_results.extend(burst_results)

            # Small pause between bursts
            await asyncio.sleep(2)

        # Wait for background sessions to complete
        print(f"\n  Waiting for background sessions to complete...")
        bg_results = await asyncio.gather(*bg_tasks)
        bg_elapsed = (time.perf_counter() - bg_start) * 1000
        bg_successful = [r for r in bg_results if r.success]
        print(f"    Background: {len(bg_successful)}/{num_background_sessions} succeeded in {bg_elapsed/1000:.1f}s")

    # ── Phase 3: Post-pressure recovery ──
    print(f"\n  Phase 3: Post-pressure recovery (re-send evicted contexts)...")
    await asyncio.sleep(5)  # Let vLLM settle

    post_pressure_metrics = await fetch_vllm_metrics(endpoint)
    total_preemptions = post_pressure_metrics.get("preemptions", 0) - initial_preemptions

    # Re-use first 5 background contexts to test recovery from FSx
    recovery_results = []
    async with aiohttp.ClientSession() as session:
        for i in range(min(5, num_background_sessions)):
            messages = [
                {"role": "system", "content": f"You are an expert analyst for session {i}. Be concise."},
                {"role": "user", "content": background_contexts[i][:context_tokens_per_session*2]
                 + f"\n\nBriefly summarize session {i}."},
            ]
            result = await make_request(session, endpoint, model, messages, max_tokens=50, temperature=0.7)
            if result.success:
                recovery_results.append(result)
                print(f"    Recovery {i+1}/5: TTFT={result.ttft_ms:.0f}ms")
            else:
                print(f"    Recovery {i+1}/5: FAILED ({result.error})")

    # ── Compile results ──
    final_cache = measure_fsx_cache(config)
    final_metrics = await fetch_vllm_metrics(endpoint)

    fg_successful = [r for r in foreground_results if r.success]
    fg_ttfts = [r.ttft_ms for r in fg_successful] if fg_successful else [0]
    recovery_ttfts = [r.ttft_ms for r in recovery_results] if recovery_results else [0]

    results = {
        "test": "memory_pressure_throughput",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "serving_config": serving_config["name"],
        "parameters": {
            "num_background_sessions": num_background_sessions,
            "context_tokens_per_session": context_tokens_per_session,
            "num_foreground_bursts": num_foreground_bursts,
            "burst_size": burst_size,
            "foreground_context_tokens": foreground_context_tokens,
            "num_gpu_blocks": num_gpu_blocks,
            "kv_capacity_tokens": kv_capacity_tokens,
            "target_fill_ratio": fill_ratio,
        },
        "background": {
            "total": num_background_sessions,
            "successful": len(bg_successful),
            "elapsed_s": bg_elapsed / 1000,
        },
        "foreground_under_pressure": {
            "total": len(foreground_results),
            "successful": len(fg_successful),
            "ttft_mean_ms": statistics.mean(fg_ttfts),
            "ttft_p50_ms": sorted(fg_ttfts)[len(fg_ttfts)//2],
            "ttft_p99_ms": sorted(fg_ttfts)[int(len(fg_ttfts)*0.99)],
            "e2e_mean_ms": statistics.mean(r.e2e_ms for r in fg_successful) if fg_successful else 0,
        },
        "recovery": {
            "total": len(recovery_results),
            "ttft_mean_ms": statistics.mean(recovery_ttfts),
            "ttfts": recovery_ttfts,
        },
        "preemptions_total": total_preemptions,
        "kv_cache_peak_usage": max(
            (s.get("kv_cache_usage", 0) for s in metrics_snapshots), default=0
        ),
        "fsx_cache_before_mb": initial_cache["size_mb"],
        "fsx_cache_after_mb": final_cache["size_mb"],
    }

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{config_output}/memory_pressure_{num_background_sessions}bg_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  MEMORY PRESSURE RESULTS")
    print(f"{'='*70}")
    print(f"  GPU KV capacity: {kv_capacity_tokens:,} tokens ({num_gpu_blocks} blocks)")
    print(f"  Background load: {num_background_sessions} sessions × {context_tokens_per_session} tok = {num_background_sessions * context_tokens_per_session:,} tok")
    print(f"  Target fill: {fill_ratio:.0%}")
    print(f"  Peak KV usage: {results['kv_cache_peak_usage']:.0%}")
    print(f"  Total preemptions: {total_preemptions:.0f}")
    print(f"")
    print(f"  Foreground under pressure ({len(fg_successful)} successful / {len(foreground_results)} total):")
    print(f"    TTFT mean: {results['foreground_under_pressure']['ttft_mean_ms']:.0f}ms")
    print(f"    TTFT p50:  {results['foreground_under_pressure']['ttft_p50_ms']:.0f}ms")
    print(f"    TTFT p99:  {results['foreground_under_pressure']['ttft_p99_ms']:.0f}ms")
    print(f"    E2E mean:  {results['foreground_under_pressure']['e2e_mean_ms']:.0f}ms")
    print(f"")
    print(f"  Recovery (re-send evicted contexts):")
    print(f"    TTFT mean: {results['recovery']['ttft_mean_ms']:.0f}ms")
    for i, ttft in enumerate(recovery_ttfts):
        print(f"    Session {i+1}: {ttft:.0f}ms")
    print(f"")
    print(f"  FSx cache: {initial_cache['size_mb']} MB → {final_cache['size_mb']} MB")
    print(f"  Saved: {out_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kimi-K2.5 Benchmark Suite")
    parser.add_argument(
        "--mode",
        choices=[
            "kv-cache", "lmcache-showcase", "multi-turn", "shared-prompt-sweep",
            "cold-start", "doc-library-rag", "api-gateway", "conversation-resumption",
            "memory-pressure",
            "full", "quick", "stress", "long-context",
            "multi-tenant", "trace-replay", "all",
        ],
        default="kv-cache",
        help=(
            "Benchmark mode: "
            "lmcache-showcase (best tests for LMCache value — recommended for comparison), "
            "kv-cache (lean KV cache offloading comparison), "
            "multi-turn (20-round growing conversation history), "
            "shared-prompt-sweep (5/10/25/50 tenants to find sweet spot), "
            "cold-start (FSx cache persistence after eviction), "
            "all (complete suite matching BENCHMARK_REPORT.md), "
            "full (all standard workloads), "
            "quick (subset), "
            "stress (stress test), "
            "long-context (16K-48K context scaling), "
            "multi-tenant (50 tenant cache pressure), "
            "trace-replay (GMI production traces)"
        ),
    )
    parser.add_argument("--config", choices=list(SERVING_CONFIGS.keys()), default="baseline",
                        help="Serving stack config: baseline, lmcache, mooncake, dynamo, sglang-hicache, sglang-hicache-nvme, sglang-mooncake")
    parser.add_argument("--endpoint", default=ENDPOINT, help="vLLM API endpoint")
    parser.add_argument("--requests", type=int, default=30, help="Requests per workload")
    parser.add_argument("--output", default="results/kimi-k2.5-p5e", help="Output base directory")
    parser.add_argument("--trace-file", default="traces/gmi_trace.jsonl",
                        help="Path to GMI trace JSONL file (for trace-replay mode)")
    parser.add_argument("--bg-sessions", type=int, default=25,
                        help="Background sessions for memory-pressure mode (default: 25)")
    parser.add_argument("--bg-context", type=int, default=24000,
                        help="Context tokens per background session for memory-pressure mode (default: 24000)")

    args = parser.parse_args()

    if args.mode == "lmcache-showcase":
        asyncio.run(run_lmcache_showcase(
            endpoint=args.endpoint,
            config=args.config,
            output_dir=args.output,
        ))
    elif args.mode == "multi-turn":
        asyncio.run(run_multi_turn_conversation(
            endpoint=args.endpoint,
            num_users=8,
            num_rounds=20,
            output_dir=args.output,
            config=args.config,
        ))
    elif args.mode == "shared-prompt-sweep":
        asyncio.run(run_shared_prompt_sweep(
            endpoint=args.endpoint,
            tenant_counts=[5, 10, 25, 50],
            output_dir=args.output,
            config=args.config,
        ))
    elif args.mode == "cold-start":
        asyncio.run(run_cold_start_recovery(
            endpoint=args.endpoint,
            output_dir=args.output,
            config=args.config,
        ))
    elif args.mode == "doc-library-rag":
        asyncio.run(run_document_library_rag(
            endpoint=args.endpoint,
            output_dir=args.output,
            config=args.config,
        ))
    elif args.mode == "api-gateway":
        asyncio.run(run_enterprise_api_gateway(
            endpoint=args.endpoint,
            output_dir=args.output,
            config=args.config,
        ))
    elif args.mode == "conversation-resumption":
        asyncio.run(run_conversation_resumption(
            endpoint=args.endpoint,
            output_dir=args.output,
            config=args.config,
        ))
    elif args.mode == "memory-pressure":
        asyncio.run(run_memory_pressure_throughput(
            endpoint=args.endpoint,
            config=args.config,
            output_dir=args.output,
            num_background_sessions=args.bg_sessions,
            context_tokens_per_session=args.bg_context,
        ))
    elif args.mode == "kv-cache":
        asyncio.run(run_kv_cache_benchmark(
            endpoint=args.endpoint,
            config=args.config,
            output_dir=args.output,
            requests_per_workload=args.requests,
        ))
    elif args.mode == "all":
        asyncio.run(run_all_benchmarks(
            endpoint=args.endpoint,
            config=args.config,
            output_dir=args.output,
            requests_per_workload=args.requests,
            trace_file=args.trace_file,
        ))
    elif args.mode == "full":
        asyncio.run(run_benchmark_suite(
            endpoint=args.endpoint,
            workloads=list(WORKLOADS.keys()),
            qps_levels=["low", "medium", "high"],
            requests_per_workload=args.requests,
            output_dir=args.output,
            config=args.config,
        ))
    elif args.mode == "quick":
        asyncio.run(run_benchmark_suite(
            endpoint=args.endpoint,
            workloads=["reasoning_math", "multi_turn_qa", "agentic_tool_use"],
            qps_levels=["low", "medium"],
            requests_per_workload=20,
            output_dir=args.output,
            config=args.config,
        ))
    elif args.mode == "stress":
        asyncio.run(run_stress_test(
            endpoint=args.endpoint,
            duration_seconds=120,
            target_qps=10.0,
            output_dir=args.output,
            config=args.config,
        ))
    elif args.mode == "long-context":
        asyncio.run(run_long_context_scaling(
            endpoint=args.endpoint,
            output_dir=args.output,
            config=args.config,
        ))
    elif args.mode == "multi-tenant":
        asyncio.run(run_multi_tenant(
            endpoint=args.endpoint,
            output_dir=args.output,
            config=args.config,
        ))
    elif args.mode == "trace-replay":
        asyncio.run(run_trace_replay(
            endpoint=args.endpoint,
            trace_file=args.trace_file,
            output_dir=args.output,
            config=args.config,
        ))
