#!/usr/bin/env python3
"""
Qwen3-32B EKS Benchmark — Apple-to-Apple comparison with HyperPod.
Identical workloads W1-W6 from gpt-oss-120b-hyperpod/scripts/benchmark.py.
Only change: MODEL name and API_URL for plain EKS Service.
"""

import argparse
import asyncio
import json
import time
import statistics
import os
import sys
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("Installing aiohttp...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])
    import aiohttp


API_URL = os.environ.get("BENCHMARK_API_URL", "http://localhost:8000")
MODEL = os.environ.get("BENCHMARK_MODEL", "/opt/ml/model")


@dataclass
class RequestResult:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    itl_ms: float = 0.0
    tps: float = 0.0
    success: bool = True
    error: str = ""


def percentile(data, p):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def send_request(session, messages, max_tokens=256, stream=True):
    """Send a single chat completion request, measure TTFT and total latency."""
    result = RequestResult()
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    start = time.perf_counter()
    first_token_time = None
    token_times = []

    try:
        if stream:
            async with session.post(
                f"{API_URL}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    result.success = False
                    result.error = f"HTTP {resp.status}: {body[:200]}"
                    return result

                token_count = 0
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "") or ""
                        reasoning = delta.get("reasoning", "") or ""
                        token_text = content + reasoning
                        if token_text:
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                            token_times.append(now)
                            token_count += 1
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

                end = time.perf_counter()
                result.total_latency_ms = (end - start) * 1000
                if first_token_time:
                    result.ttft_ms = (first_token_time - start) * 1000
                result.completion_tokens = token_count

                if len(token_times) > 1:
                    itls = [(token_times[i] - token_times[i-1]) * 1000
                            for i in range(1, len(token_times))]
                    result.itl_ms = statistics.mean(itls) if itls else 0
                if token_count > 0 and result.total_latency_ms > 0:
                    result.tps = token_count / (result.total_latency_ms / 1000)
        else:
            async with session.post(
                f"{API_URL}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                end = time.perf_counter()
                result.total_latency_ms = (end - start) * 1000
                if resp.status != 200:
                    result.success = False
                    result.error = f"HTTP {resp.status}"
                    return result
                body = await resp.json()
                usage = body.get("usage", {})
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)
                result.ttft_ms = result.total_latency_ms
                if result.completion_tokens > 0:
                    result.tps = result.completion_tokens / (result.total_latency_ms / 1000)

    except asyncio.TimeoutError:
        result.success = False
        result.error = "timeout"
    except Exception as e:
        result.success = False
        result.error = str(e)[:200]

    return result


async def run_concurrent(coro_list, qps=None):
    """Run requests with optional QPS rate limiting."""
    if qps and qps > 0:
        results = []
        delay = 1.0 / qps
        tasks = []
        for coro in coro_list:
            tasks.append(asyncio.create_task(coro))
            await asyncio.sleep(delay)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r if isinstance(r, RequestResult) else RequestResult(success=False, error=str(r))
                for r in results]
    else:
        results = await asyncio.gather(*coro_list, return_exceptions=True)
        return [r if isinstance(r, RequestResult) else RequestResult(success=False, error=str(r))
                for r in results]


def build_multiturn_messages(num_rounds, turn_tokens=200):
    filler = "Explain the concept of " + " ".join(["distributed"] * 30) + " systems. "
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(num_rounds):
        messages.append({"role": "user", "content": f"Turn {i+1}: {filler} Question {i+1}: What is important about turn {i+1}?"})
        if i < num_rounds - 1:
            messages.append({"role": "assistant", "content": f"Response to turn {i+1}. " + filler[:100]})
    return messages


def build_shared_prompt_messages(prompt_len_tokens=2000, query="What is the capital of France?"):
    system_text = "You are an expert assistant. " + ("Context: " + "x " * 4) * (prompt_len_tokens // 2)
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": query},
    ]


def build_random_input(input_len_tokens=4096):
    text = "Summarize: " + ("word " * 4) * (input_len_tokens // 4)
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]


def compute_stats(results_list):
    successful = [r for r in results_list if r.success]
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms > 0]
    itls = [r.itl_ms for r in successful if r.itl_ms > 0]
    latencies = [r.total_latency_ms for r in successful if r.total_latency_ms > 0]
    tps_list = [r.tps for r in successful if r.tps > 0]

    return {
        "num_requests": len(results_list),
        "successful": len(successful),
        "failed": len(results_list) - len(successful),
        "ttft_p50_ms": round(percentile(ttfts, 50), 2),
        "ttft_p90_ms": round(percentile(ttfts, 90), 2),
        "ttft_p95_ms": round(percentile(ttfts, 95), 2),
        "ttft_p99_ms": round(percentile(ttfts, 99), 2),
        "ttft_mean_ms": round(statistics.mean(ttfts), 2) if ttfts else 0,
        "itl_p50_ms": round(percentile(itls, 50), 2),
        "itl_p90_ms": round(percentile(itls, 90), 2),
        "itl_p95_ms": round(percentile(itls, 95), 2),
        "total_latency_p50_ms": round(percentile(latencies, 50), 2),
        "total_latency_p95_ms": round(percentile(latencies, 95), 2),
        "throughput_tps": round(statistics.mean(tps_list), 2) if tps_list else 0,
    }


async def benchmark_w1_multiturn(config_name, session):
    """W1: Multi-Turn Chat — sweep rounds and concurrency."""
    print("\n=== W1: Multi-Turn Chat ===")
    all_results = []

    for num_rounds in [1, 5, 10]:
        for concurrent in [1, 4, 8]:
            for qps in [1.0, 4.0]:
                print(f"  rounds={num_rounds} concurrent={concurrent} qps={qps} ...", end=" ", flush=True)
                messages = build_multiturn_messages(num_rounds)
                coros = [send_request(session, messages, max_tokens=128)
                         for _ in range(concurrent)]

                start = time.perf_counter()
                results = await run_concurrent(coros, qps=qps)
                duration = time.perf_counter() - start

                stats = compute_stats(results)
                stats["rounds"] = num_rounds
                stats["concurrent"] = concurrent
                stats["qps"] = qps
                stats["duration_s"] = round(duration, 2)
                all_results.append(stats)

                ok = stats["successful"]
                ttft = stats["ttft_p50_ms"]
                print(f"ok={ok}/{concurrent} ttft_p50={ttft:.0f}ms tps={stats['throughput_tps']:.1f}")

    return all_results


async def benchmark_w2_rag(config_name, session):
    """W2: RAG / Long Document QA — shared document prefix with varying queries."""
    print("\n=== W2: RAG / Long Document QA ===")
    all_results = []

    doc_filler = "The distributed computing paradigm enables scalable processing. " * 50
    queries = [
        "What is the main topic of this document?",
        "Summarize the key points in 3 sentences.",
        "What are the implications for system design?",
        "How does this relate to modern cloud architectures?",
        "What are the potential challenges mentioned?",
        "Describe the scalability aspects discussed.",
        "What recommendations are made?",
        "How does this compare to traditional approaches?",
    ]

    for doc_tokens in [2000, 5000, 10000]:
        doc_text = (doc_filler * (doc_tokens // 50))[:doc_tokens * 4]

        for hit_ratio_label, num_warmup, num_query in [("2:2", 2, 2), ("3:1", 3, 1), ("4:1", 4, 1)]:
            for concurrent in [4, 8]:
                print(f"  doc_tokens={doc_tokens} ratio={hit_ratio_label} concurrent={concurrent} ...", end=" ", flush=True)

                warmup_coros = []
                for i in range(num_warmup):
                    msgs = [
                        {"role": "system", "content": f"You are a document analyst. Document:\n{doc_text}"},
                        {"role": "user", "content": queries[i % len(queries)]},
                    ]
                    warmup_coros.append(send_request(session, msgs, max_tokens=100))

                warmup_results = await run_concurrent(warmup_coros, qps=2.0)
                warmup_stats = compute_stats(warmup_results)

                query_coros = []
                for i in range(concurrent):
                    msgs = [
                        {"role": "system", "content": f"You are a document analyst. Document:\n{doc_text}"},
                        {"role": "user", "content": queries[(i + num_warmup) % len(queries)]},
                    ]
                    query_coros.append(send_request(session, msgs, max_tokens=100))

                start = time.perf_counter()
                query_results = await run_concurrent(query_coros, qps=4.0)
                duration = time.perf_counter() - start

                query_stats = compute_stats(query_results)

                result = {
                    "doc_tokens": doc_tokens,
                    "hit_ratio": hit_ratio_label,
                    "concurrent": concurrent,
                    "warmup_ttft_p50_ms": warmup_stats["ttft_p50_ms"],
                    "warmup_ttft_p95_ms": warmup_stats["ttft_p95_ms"],
                    "query_ttft_p50_ms": query_stats["ttft_p50_ms"],
                    "query_ttft_p95_ms": query_stats["ttft_p95_ms"],
                    "ttft_improvement": round(warmup_stats["ttft_p50_ms"] / query_stats["ttft_p50_ms"], 2) if query_stats["ttft_p50_ms"] > 0 else 0,
                    "warmup_tps": warmup_stats["throughput_tps"],
                    "query_tps": query_stats["throughput_tps"],
                    "num_requests": len(query_results),
                    "successful": query_stats["successful"],
                    "failed": query_stats["failed"],
                    "duration_s": round(duration, 2),
                }
                all_results.append(result)

                improvement = result["ttft_improvement"]
                print(f"warmup_ttft={warmup_stats['ttft_p50_ms']:.0f}ms query_ttft={query_stats['ttft_p50_ms']:.0f}ms improvement={improvement:.2f}x")

    return all_results


async def benchmark_w3_agentic(config_name, session):
    """W3: Agentic Tool Calling — simulates multi-turn agent with tool-call pauses."""
    print("\n=== W3: Agentic Tool Calling ===")
    all_results = []

    tool_results_bank = [
        "Tool returned: {'status': 'success', 'data': {'temperature': 72, 'humidity': 45}}",
        "Tool returned: {'files': ['main.py', 'utils.py', 'config.yaml'], 'count': 3}",
        "Tool returned: {'query_result': [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]}",
        "Tool returned: {'calculation': 3.14159, 'unit': 'radians'}",
        "Tool returned: {'search_results': ['Result 1: Overview of distributed systems', 'Result 2: CAP theorem explained']}",
    ]

    for num_turns in [5, 10]:
        for tool_latency in [0.5, 2.0, 5.0]:
            for concurrent in [4, 8]:
                print(f"  turns={num_turns} tool_latency={tool_latency}s concurrent={concurrent} ...", end=" ", flush=True)

                turn_ttfts = {i: [] for i in range(num_turns)}
                turn_tps = {i: [] for i in range(num_turns)}
                total_failures = 0

                async def run_agent_session(session_id):
                    nonlocal total_failures
                    messages = [{"role": "system", "content": "You are a helpful agent with access to tools. Use tools when needed."}]
                    session_ttfts = []

                    for turn in range(num_turns):
                        if turn == 0:
                            messages.append({"role": "user", "content": f"Agent session {session_id}: Find information about distributed computing and analyze the results."})
                        else:
                            await asyncio.sleep(tool_latency)
                            messages.append({"role": "user", "content": f"Tool result for turn {turn}: {tool_results_bank[turn % len(tool_results_bank)]}"})

                        result = await send_request(session, messages, max_tokens=128, stream=True)
                        if result.success:
                            turn_ttfts[turn].append(result.ttft_ms)
                            turn_tps[turn].append(result.tps)
                            messages.append({"role": "assistant", "content": f"[Response for turn {turn}]"})
                        else:
                            total_failures += 1

                    return session_ttfts

                start = time.perf_counter()
                tasks = [run_agent_session(i) for i in range(concurrent)]
                await asyncio.gather(*tasks)
                duration = time.perf_counter() - start

                turn_stats = {}
                for turn in range(num_turns):
                    ttfts = turn_ttfts[turn]
                    if ttfts:
                        turn_stats[f"turn{turn}_ttft_p50"] = round(percentile(ttfts, 50), 2)
                        turn_stats[f"turn{turn}_ttft_p95"] = round(percentile(ttfts, 95), 2)

                t0_ttft = percentile(turn_ttfts[0], 50) if turn_ttfts[0] else 0
                last_turn = num_turns - 1
                tN_ttft = percentile(turn_ttfts[last_turn], 50) if turn_ttfts[last_turn] else 0
                degradation = round(tN_ttft / t0_ttft, 2) if t0_ttft > 0 else 0

                result = {
                    "num_turns": num_turns,
                    "tool_latency_s": tool_latency,
                    "concurrent": concurrent,
                    "turn0_ttft_p50_ms": round(t0_ttft, 2),
                    "turnN_ttft_p50_ms": round(tN_ttft, 2),
                    "ttft_degradation": degradation,
                    "num_requests": concurrent * num_turns,
                    "successful": concurrent * num_turns - total_failures,
                    "failed": total_failures,
                    "duration_s": round(duration, 2),
                    "ttft_p50_ms": round(t0_ttft, 2),
                    "ttft_p95_ms": round(percentile(turn_ttfts[0], 95) if turn_ttfts[0] else 0, 2),
                    "throughput_tps": round(statistics.mean([t for tl in turn_tps.values() for t in tl]), 2) if any(turn_tps.values()) else 0,
                    **turn_stats,
                }
                all_results.append(result)

                print(f"t0_ttft={t0_ttft:.0f}ms tN_ttft={tN_ttft:.0f}ms degradation={degradation:.2f}x duration={duration:.0f}s")

    return all_results


async def benchmark_w4_shared_prompt(config_name, session):
    """W4: Shared System Prompt — test prefix caching."""
    print("\n=== W4: Shared System Prompt ===")
    all_results = []

    queries = [
        "What is the capital of France?",
        "Explain quantum computing briefly.",
        "How does photosynthesis work?",
        "What are the benefits of exercise?",
        "Describe the water cycle.",
        "What is machine learning?",
        "How do vaccines work?",
        "What causes earthquakes?",
    ]

    for prompt_len in [2000, 4000]:
        for concurrent in [4, 8, 16]:
            for qps in [2.0, 8.0]:
                print(f"  prompt_len={prompt_len} concurrent={concurrent} qps={qps} ...", end=" ", flush=True)
                coros = []
                for i in range(concurrent):
                    msgs = build_shared_prompt_messages(prompt_len, queries[i % len(queries)])
                    coros.append(send_request(session, msgs, max_tokens=128))

                start = time.perf_counter()
                results = await run_concurrent(coros, qps=qps)
                duration = time.perf_counter() - start

                stats = compute_stats(results)
                stats["prompt_len_tokens"] = prompt_len
                stats["concurrent"] = concurrent
                stats["qps"] = qps
                stats["duration_s"] = round(duration, 2)
                all_results.append(stats)

                ok = stats["successful"]
                ttft = stats["ttft_p50_ms"]
                print(f"ok={ok}/{concurrent} ttft_p50={ttft:.0f}ms tps={stats['throughput_tps']:.1f}")

    return all_results


async def benchmark_w5_sharegpt(config_name, session):
    """W5: Simulated ShareGPT-style conversations at various QPS."""
    print("\n=== W5: ShareGPT-style Conversations ===")
    all_results = []

    conversations = [
        [{"role": "user", "content": "Write a Python function to sort a list using merge sort."}],
        [{"role": "user", "content": "Explain the difference between TCP and UDP protocols."}],
        [{"role": "user", "content": "What are the main causes of climate change and how can we address them?"}],
        [{"role": "user", "content": "Describe the architecture of a modern web application."}],
        [{"role": "user", "content": "How does a neural network learn? Explain backpropagation."}],
        [{"role": "user", "content": "Write a SQL query to find the top 10 customers by revenue."}],
        [{"role": "user", "content": "What is the difference between REST and GraphQL APIs?"}],
        [{"role": "user", "content": "Explain how garbage collection works in Java."}],
        [{"role": "user", "content": "Design a URL shortener service. What components would you need?"}],
        [{"role": "user", "content": "What are design patterns? Give examples of 3 common ones."}],
    ]

    for qps in [0.5, 2.0, 4.0, 8.0]:
        num_requests = min(int(qps * 30), 40)
        if num_requests < 4:
            num_requests = 4
        print(f"  qps={qps} num_requests={num_requests} ...", end=" ", flush=True)

        coros = []
        for i in range(num_requests):
            msgs = [{"role": "system", "content": "You are a helpful assistant."}]
            msgs.extend(conversations[i % len(conversations)])
            coros.append(send_request(session, msgs, max_tokens=256))

        start = time.perf_counter()
        results = await run_concurrent(coros, qps=qps)
        duration = time.perf_counter() - start

        stats = compute_stats(results)
        stats["qps_target"] = qps
        stats["qps_actual"] = round(stats["successful"] / duration, 2) if duration > 0 else 0
        stats["num_requests"] = num_requests
        stats["duration_s"] = round(duration, 2)
        all_results.append(stats)

        ok = stats["successful"]
        ttft = stats["ttft_p50_ms"]
        print(f"ok={ok}/{num_requests} ttft_p50={ttft:.0f}ms ttft_p95={stats['ttft_p95_ms']:.0f}ms tps={stats['throughput_tps']:.1f}")

    return all_results


async def benchmark_w6_longctx(config_name, session):
    """W6: Long Context Scaling — sweep input lengths."""
    print("\n=== W6: Long Context Scaling ===")
    all_results = []

    for input_len in [1000, 4000, 8000, 16000]:
        for qps in [0.5, 2.0]:
            num_requests = 4
            print(f"  input_len={input_len} qps={qps} ...", end=" ", flush=True)

            coros = []
            for _ in range(num_requests):
                msgs = build_random_input(input_len)
                coros.append(send_request(session, msgs, max_tokens=256))

            start = time.perf_counter()
            results = await run_concurrent(coros, qps=qps)
            duration = time.perf_counter() - start

            stats = compute_stats(results)
            stats["input_len_tokens"] = input_len
            stats["qps"] = qps
            stats["duration_s"] = round(duration, 2)
            all_results.append(stats)

            ok = stats["successful"]
            ttft = stats["ttft_p50_ms"]
            print(f"ok={ok}/{num_requests} ttft_p50={ttft:.0f}ms tps={stats['throughput_tps']:.1f}")

    return all_results


async def main():
    parser = argparse.ArgumentParser(description="Qwen3-32B EKS Benchmark")
    parser.add_argument("--config", default="config0-nocache", help="Config name")
    parser.add_argument("--workloads", default="w1,w2,w3,w4,w5,w6", help="Comma-separated workloads")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    config_name = args.config
    workloads = args.workloads.split(",")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "results", "benchmarks", config_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Qwen3-32B EKS Benchmark — {config_name}")
    print(f"Workloads: {workloads}")
    print(f"Output: {output_dir}")
    print(f"Timestamp: {timestamp}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                print(f"Health check: {resp.status}")
        except Exception as e:
            print(f"ERROR: Cannot reach {API_URL}/health: {e}")
            return

    all_benchmark_results = {}

    async with aiohttp.ClientSession() as session:
        if "w1" in workloads:
            results = await benchmark_w1_multiturn(config_name, session)
            all_benchmark_results["w1_multiturn"] = results

        if "w2" in workloads:
            results = await benchmark_w2_rag(config_name, session)
            all_benchmark_results["w2_rag"] = results

        if "w3" in workloads:
            results = await benchmark_w3_agentic(config_name, session)
            all_benchmark_results["w3_agentic"] = results

        if "w4" in workloads:
            results = await benchmark_w4_shared_prompt(config_name, session)
            all_benchmark_results["w4_shared_prompt"] = results

        if "w5" in workloads:
            results = await benchmark_w5_sharegpt(config_name, session)
            all_benchmark_results["w5_sharegpt"] = results

        if "w6" in workloads:
            results = await benchmark_w6_longctx(config_name, session)
            all_benchmark_results["w6_longctx"] = results

    output_file = os.path.join(output_dir, f"benchmark_{config_name}_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump({
            "config": config_name,
            "model": MODEL,
            "api_url": API_URL,
            "platform": "eks",
            "instance_type": "g6e.2xlarge",
            "gpu_count": 1,
            "timestamp": timestamp,
            "results": all_benchmark_results,
        }, f, indent=2)

    print(f"\n=== Results saved to {output_file} ===")

    print(f"\n{'='*80}")
    print(f"SUMMARY — {config_name}")
    print(f"{'='*80}")
    for workload_name, results in all_benchmark_results.items():
        print(f"\n--- {workload_name} ---")
        for r in results:
            common_keys = {"num_requests", "successful", "failed", "duration_s",
                           "ttft_p50_ms", "ttft_p90_ms", "ttft_p95_ms", "ttft_p99_ms",
                           "ttft_mean_ms", "itl_p50_ms", "itl_p90_ms", "itl_p95_ms",
                           "total_latency_p50_ms", "total_latency_p95_ms", "throughput_tps",
                           "warmup_ttft_p50_ms", "warmup_ttft_p95_ms", "query_ttft_p50_ms",
                           "query_ttft_p95_ms", "ttft_improvement", "warmup_tps", "query_tps",
                           "turn0_ttft_p50_ms", "turnN_ttft_p50_ms", "ttft_degradation"}
            params = {k: v for k, v in r.items()
                      if k not in common_keys and not k.startswith("turn")}
            if "ttft_p50_ms" in r:
                print(f"  {params} => ok={r['successful']}/{r['num_requests']} "
                      f"TTFT p50={r['ttft_p50_ms']:.0f}ms p95={r['ttft_p95_ms']:.0f}ms "
                      f"TPS={r['throughput_tps']:.1f}")
            elif "warmup_ttft_p50_ms" in r:
                print(f"  {params} => ok={r['successful']}/{r['num_requests']} "
                      f"warmup={r['warmup_ttft_p50_ms']:.0f}ms query={r['query_ttft_p50_ms']:.0f}ms "
                      f"improvement={r['ttft_improvement']:.2f}x")
            elif "turn0_ttft_p50_ms" in r:
                print(f"  {params} => ok={r['successful']}/{r['num_requests']} "
                      f"t0={r['turn0_ttft_p50_ms']:.0f}ms tN={r['turnN_ttft_p50_ms']:.0f}ms "
                      f"degradation={r['ttft_degradation']:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
