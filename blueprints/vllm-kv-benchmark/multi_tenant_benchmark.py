#!/usr/bin/env python3
"""
Multi-Tenant System Prompt Benchmark

Tests LMCache benefits with multiple user groups, each sharing a different system prompt.
This simulates multi-tenant SaaS deployments where each tenant has a unique system prompt.

Without LMCache: Each request recomputes the tenant's system prompt
With LMCache + FSx: First user per tenant computes, subsequent users retrieve from cache
"""

import argparse
import asyncio
import time
import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import openai
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TenantConfig:
    tenant_id: int
    system_prompt: str
    num_users: int


@dataclass
class RequestResult:
    tenant_id: int
    user_id: int
    question_id: int
    prompt_tokens: int
    generation_tokens: int
    ttft: float
    generation_time: float
    launch_time: float
    finish_time: float


def generate_system_prompt(tenant_id: int, token_length: int) -> str:
    """Generate a unique system prompt for a tenant."""
    base_prompt = f"You are an AI assistant for Tenant {tenant_id}. "
    # Pad with unique content per tenant to ensure different prefixes
    padding = f"Tenant {tenant_id} context: " + " ".join([f"word{tenant_id}_{i}" for i in range(token_length // 2)])
    return base_prompt + padding


async def send_request(
    client: openai.AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    user_id: int,
) -> Dict:
    """Send a single request and measure timing."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    start_time = time.time()
    first_token_time = None
    content = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            temperature=0.0,
            stream_options={"include_usage": True},
            extra_headers={"x-user-id": str(user_id)},
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                content += chunk.choices[0].delta.content
            if hasattr(chunk, 'usage') and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

        finish_time = time.time()
        ttft = (first_token_time - start_time) if first_token_time else 0
        generation_time = (finish_time - first_token_time) if first_token_time else 0

        return {
            "prompt_tokens": prompt_tokens,
            "generation_tokens": completion_tokens,
            "ttft": ttft,
            "generation_time": generation_time,
            "launch_time": start_time,
            "finish_time": finish_time,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {
            "prompt_tokens": 0,
            "generation_tokens": 0,
            "ttft": 0,
            "generation_time": 0,
            "launch_time": start_time,
            "finish_time": time.time(),
            "success": False,
        }


async def run_user_session(
    client: openai.AsyncOpenAI,
    model: str,
    tenant: TenantConfig,
    user_id: int,
    num_questions: int,
    question_length: int,
    answer_length: int,
    delay_between_questions: float,
) -> List[RequestResult]:
    """Simulate a user session with multiple questions."""
    results = []

    for q in range(num_questions):
        # Generate a unique question per user/question
        user_message = f"User {user_id} question {q}: " + " ".join(
            [f"q{user_id}_{q}_{i}" for i in range(question_length // 4)]
        ) + " Please respond briefly."

        result = await send_request(
            client=client,
            model=model,
            system_prompt=tenant.system_prompt,
            user_message=user_message,
            max_tokens=answer_length,
            user_id=user_id,
        )

        if result["success"]:
            results.append(RequestResult(
                tenant_id=tenant.tenant_id,
                user_id=user_id,
                question_id=q,
                prompt_tokens=result["prompt_tokens"],
                generation_tokens=result["generation_tokens"],
                ttft=result["ttft"],
                generation_time=result["generation_time"],
                launch_time=result["launch_time"],
                finish_time=result["finish_time"],
            ))

        if q < num_questions - 1:
            await asyncio.sleep(delay_between_questions)

    return results


async def run_benchmark(
    base_url: str,
    model: str,
    num_tenants: int,
    users_per_tenant: int,
    system_prompt_length: int,
    questions_per_user: int,
    question_length: int,
    answer_length: int,
    qps: float,
) -> pd.DataFrame:
    """Run the multi-tenant benchmark."""

    # Initialize client
    if not base_url.endswith('/v1'):
        base_url = base_url.rstrip('/') + '/v1'

    client = openai.AsyncOpenAI(
        api_key="dummy-key",
        base_url=base_url,
    )

    # Create tenant configs
    tenants = []
    for t in range(num_tenants):
        tenants.append(TenantConfig(
            tenant_id=t,
            system_prompt=generate_system_prompt(t, system_prompt_length),
            num_users=users_per_tenant,
        ))

    logger.info(f"Created {num_tenants} tenants with {users_per_tenant} users each")
    logger.info(f"System prompt length: ~{system_prompt_length} tokens per tenant")
    logger.info(f"Total users: {num_tenants * users_per_tenant}")
    logger.info(f"Target QPS: {qps}")

    # Calculate delay to achieve target QPS
    total_users = num_tenants * users_per_tenant
    delay_between_users = 1.0 / qps if qps > 0 else 0
    delay_between_questions = total_users / qps if qps > 0 else 1.0

    # Create all user sessions
    all_results = []
    tasks = []

    # Stagger user starts to achieve target QPS
    user_global_id = 0
    for tenant in tenants:
        for u in range(tenant.num_users):
            user_id = user_global_id
            user_global_id += 1

            # Stagger start times
            start_delay = user_id * delay_between_users

            async def run_with_delay(delay, t, uid):
                await asyncio.sleep(delay)
                return await run_user_session(
                    client=client,
                    model=model,
                    tenant=t,
                    user_id=uid,
                    num_questions=questions_per_user,
                    question_length=question_length,
                    answer_length=answer_length,
                    delay_between_questions=delay_between_questions,
                )

            tasks.append(run_with_delay(start_delay, tenant, user_id))

    # Run all sessions concurrently
    logger.info("Starting benchmark...")
    start_time = time.time()

    results_lists = await asyncio.gather(*tasks)

    end_time = time.time()
    logger.info(f"Benchmark completed in {end_time - start_time:.2f}s")

    # Flatten results
    for results in results_lists:
        all_results.extend(results)

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "tenant_id": r.tenant_id,
            "user_id": r.user_id,
            "question_id": r.question_id,
            "prompt_tokens": r.prompt_tokens,
            "generation_tokens": r.generation_tokens,
            "ttft": r.ttft,
            "generation_time": r.generation_time,
            "launch_time": r.launch_time,
            "finish_time": r.finish_time,
        }
        for r in all_results
    ])

    return df


def analyze_results(df: pd.DataFrame) -> Dict:
    """Analyze benchmark results."""

    # Overall stats
    total_requests = len(df)
    avg_ttft = df["ttft"].mean() * 1000  # ms
    p50_ttft = df["ttft"].quantile(0.5) * 1000
    p90_ttft = df["ttft"].quantile(0.9) * 1000
    p99_ttft = df["ttft"].quantile(0.99) * 1000

    # Per-tenant stats
    tenant_stats = df.groupby("tenant_id").agg({
        "ttft": ["mean", "std", "count"],
        "generation_time": "mean",
    })

    # First request vs subsequent (proxy for cache hit)
    df_sorted = df.sort_values(["tenant_id", "launch_time"])

    first_per_tenant = df_sorted.groupby("tenant_id").first()
    rest_per_tenant = df_sorted.groupby("tenant_id").apply(lambda x: x.iloc[1:] if len(x) > 1 else pd.DataFrame())

    first_ttft = first_per_tenant["ttft"].mean() * 1000
    if len(rest_per_tenant) > 0:
        rest_ttft = rest_per_tenant["ttft"].mean() * 1000
    else:
        rest_ttft = 0

    duration = df["finish_time"].max() - df["launch_time"].min()
    achieved_qps = total_requests / duration if duration > 0 else 0

    return {
        "total_requests": total_requests,
        "duration_seconds": duration,
        "achieved_qps": achieved_qps,
        "avg_ttft_ms": avg_ttft,
        "p50_ttft_ms": p50_ttft,
        "p90_ttft_ms": p90_ttft,
        "p99_ttft_ms": p99_ttft,
        "first_request_ttft_ms": first_ttft,
        "subsequent_request_ttft_ms": rest_ttft,
        "cache_benefit_ratio": first_ttft / rest_ttft if rest_ttft > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-tenant system prompt benchmark")
    parser.add_argument("--base-url", type=str, required=True, help="vLLM server URL")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--num-tenants", type=int, default=5, help="Number of tenants")
    parser.add_argument("--users-per-tenant", type=int, default=20, help="Users per tenant")
    parser.add_argument("--system-prompt-length", type=int, default=2000, help="System prompt length in tokens")
    parser.add_argument("--questions-per-user", type=int, default=3, help="Questions per user")
    parser.add_argument("--question-length", type=int, default=100, help="Question length in tokens")
    parser.add_argument("--answer-length", type=int, default=100, help="Max answer length")
    parser.add_argument("--qps", type=float, default=2.0, help="Target QPS")
    parser.add_argument("--output", type=str, default="multi_tenant_results.csv", help="Output CSV file")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Multi-Tenant System Prompt Benchmark")
    logger.info("=" * 60)
    logger.info(f"Tenants: {args.num_tenants}")
    logger.info(f"Users per tenant: {args.users_per_tenant}")
    logger.info(f"System prompt: {args.system_prompt_length} tokens")
    logger.info(f"Questions per user: {args.questions_per_user}")
    logger.info(f"Target QPS: {args.qps}")
    logger.info("=" * 60)

    df = asyncio.run(run_benchmark(
        base_url=args.base_url,
        model=args.model,
        num_tenants=args.num_tenants,
        users_per_tenant=args.users_per_tenant,
        system_prompt_length=args.system_prompt_length,
        questions_per_user=args.questions_per_user,
        question_length=args.question_length,
        answer_length=args.answer_length,
        qps=args.qps,
    ))

    # Save raw results
    df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

    # Analyze and print summary
    stats = analyze_results(df)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total requests: {stats['total_requests']}")
    logger.info(f"Duration: {stats['duration_seconds']:.2f}s")
    logger.info(f"Achieved QPS: {stats['achieved_qps']:.2f}")
    logger.info("-" * 60)
    logger.info(f"TTFT (avg): {stats['avg_ttft_ms']:.1f}ms")
    logger.info(f"TTFT (p50): {stats['p50_ttft_ms']:.1f}ms")
    logger.info(f"TTFT (p90): {stats['p90_ttft_ms']:.1f}ms")
    logger.info(f"TTFT (p99): {stats['p99_ttft_ms']:.1f}ms")
    logger.info("-" * 60)
    logger.info(f"First request per tenant (cold): {stats['first_request_ttft_ms']:.1f}ms")
    logger.info(f"Subsequent requests (warm): {stats['subsequent_request_ttft_ms']:.1f}ms")
    logger.info(f"Cache benefit ratio: {stats['cache_benefit_ratio']:.2f}x")
    logger.info("=" * 60)

    # Print per-tenant breakdown
    tenant_summary = df.groupby("tenant_id").agg({
        "ttft": ["mean", "count"],
    })
    logger.info("\nPer-Tenant TTFT (ms):")
    for tenant_id in tenant_summary.index:
        avg_ttft = tenant_summary.loc[tenant_id, ("ttft", "mean")] * 1000
        count = tenant_summary.loc[tenant_id, ("ttft", "count")]
        logger.info(f"  Tenant {tenant_id}: {avg_ttft:.1f}ms ({count} requests)")


if __name__ == "__main__":
    main()
