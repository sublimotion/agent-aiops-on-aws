"""Bedrock worker proxy for cost-aware routing.

Adapted from rl-conductor/worker_proxy_v2.py with these changes:
  - Pool loaded from configs/pool.yaml (verified pricing) — single source of truth
  - Records actual input/output token counts (not just word-count estimates)
    by reading Bedrock's response usage fields. Cost model uses real tokens.
  - Anthropic "thinking" content blocks are filtered out: we take only
    type=="text" items so the extractor never sees raw thinking. This was a
    major risk for Opus 4.7 extended-thinking outputs.
  - Cross-region rotation + per-worker semaphores preserved
  - Returns a CallResult dataclass instead of bare string so the trainer can
    log latency / tokens / region / model_id per call

This proxy is BACK-END ONLY. Worker prompts are constructed by the caller
(trainer / parser_audit) and passed verbatim. The proxy does not know about
the routing format or the reward function.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import boto3
import yaml
from botocore.config import Config as BotoConfig

AWS_REGIONS = os.environ.get("BEDROCK_REGIONS", "us-east-1,us-east-2,us-west-2").split(",")
US_PREFIX = "us."


@dataclass
class WorkerConfig:
    ord: int
    name: str
    bedrock_id: str
    needs_us_prefix: bool
    in_per_1m: float
    out_per_1m: float
    output_tok_assumed: int
    capability_summary: str
    concurrency: int
    supports_temperature: bool = True
    allowed_regions: tuple = ()


@dataclass
class CallResult:
    text: str                      # final assistant text, with Anthropic thinking blocks removed
    raw_response: dict             # full Bedrock response body for logging/debug
    input_tok: int
    output_tok: int
    latency_ms: float
    region: str
    model_id: str
    error: Optional[str] = None


@dataclass
class WorkerStats:
    calls: int = 0
    errors: int = 0
    throttled: int = 0
    total_input_tok: int = 0
    total_output_tok: int = 0
    total_latency_ms: float = 0.0
    total_cost_dollars: float = 0.0


def _load_pool(yaml_path: str) -> list[WorkerConfig]:
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    workers = []
    for w in cfg["workers"]:
        workers.append(WorkerConfig(
            ord=w["ord"],
            name=w["name"],
            bedrock_id=w["bedrock_id"],
            needs_us_prefix=w["needs_us_prefix"],
            in_per_1m=w["in_per_1m"],
            out_per_1m=w["out_per_1m"],
            output_tok_assumed=w["output_tok_assumed"],
            capability_summary=w["capability_summary"],
            concurrency=w["concurrency"],
            supports_temperature=w.get("supports_temperature", True),
            allowed_regions=tuple(w.get("allowed_regions", ())),
        ))
    return workers


class WorkerPool:
    """11-worker Bedrock pool. Ordinals are SHUFFLED via `seed` to break
    positional brand-bias (Gate 0.4); the trainer must record the mapping
    so cross-seed eval (RQ#3) can replay or shuffle differently.
    """

    def __init__(self, yaml_path: str, seed: Optional[int] = None):
        configs = _load_pool(yaml_path)
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(configs)
            # Re-assign ords 0..N after shuffle — `ord` field becomes
            # the position the router sees, not the canonical pool index.
            for i, c in enumerate(configs):
                c.ord = i
        self.workers: dict[int, WorkerConfig] = {c.ord: c for c in configs}
        self.stats: dict[int, WorkerStats] = {o: WorkerStats() for o in self.workers}
        # Semaphores are tied to the running event loop; create lazily inside call().
        self._concurrency: dict[int, int] = {o: c.concurrency for o, c in self.workers.items()}
        self._semaphores: dict[int, asyncio.Semaphore] = {}
        self._sem_loop_id: Optional[int] = None
        self._region_idx: dict[int, int] = {o: 0 for o in self.workers}
        self._clients: dict[str, object] = {}

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    def get_mapping(self) -> dict[int, str]:
        return {o: c.bedrock_id for o, c in self.workers.items()}

    def capability_lines(self) -> list[str]:
        """One line per worker for the metadata-rich router prompt (mask_style='full')."""
        return [
            f"ord_{o}: {c.capability_summary}"
            for o, c in sorted(self.workers.items())
        ]

    def _client(self, region: str):
        if region not in self._clients:
            self._clients[region] = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=BotoConfig(
                    read_timeout=60,
                    connect_timeout=10,
                    retries={"max_attempts": 2, "mode": "standard"},
                    max_pool_connections=50,
                ),
            )
        return self._clients[region]

    def _next_region(self, ord_: int) -> str:
        w = self.workers[ord_]
        regions = list(w.allowed_regions) if w.allowed_regions else AWS_REGIONS
        idx = self._region_idx[ord_] % len(regions)
        self._region_idx[ord_] = (self._region_idx[ord_] + 1) % max(len(regions), 1)
        return regions[idx]

    def _resolve_model_id(self, w: WorkerConfig) -> str:
        return f"{US_PREFIX}{w.bedrock_id}" if w.needs_us_prefix else w.bedrock_id

    def _get_semaphore(self, ord_: int) -> asyncio.Semaphore:
        """Get-or-create per-worker semaphore bound to the current event loop.
        Trainer calls asyncio.run() once per iter, which creates a new event loop;
        semaphores from a previous loop become unusable. Recreate when loop changes.
        """
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        if self._sem_loop_id != loop_id:
            self._semaphores = {
                o: asyncio.Semaphore(c) for o, c in self._concurrency.items()
            }
            self._sem_loop_id = loop_id
        return self._semaphores[ord_]

    async def call(
        self,
        ord_: int,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
    ) -> CallResult:
        if ord_ not in self.workers:
            return CallResult("", {}, 0, 0, 0.0, "", "", error=f"invalid ord {ord_}")
        w = self.workers[ord_]
        if max_tokens is None:
            max_tokens = w.output_tok_assumed + 128   # small headroom over budget
        async with self._get_semaphore(ord_):
            return await self._call_with_retry(ord_, w, prompt, max_tokens, temperature)

    async def _call_with_retry(self, ord_, w, prompt, max_tokens, temperature, attempts=3):
        last_err = None
        for attempt in range(attempts):
            region = self._next_region(ord_)
            start = time.monotonic()
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._invoke_sync, w, region, prompt, max_tokens, temperature,
                )
                result.latency_ms = (time.monotonic() - start) * 1000
                # Update stats
                s = self.stats[ord_]
                s.calls += 1
                s.total_input_tok += result.input_tok
                s.total_output_tok += result.output_tok
                s.total_latency_ms += result.latency_ms
                s.total_cost_dollars += (
                    result.input_tok * w.in_per_1m + result.output_tok * w.out_per_1m
                ) / 1_000_000.0
                return result
            except Exception as e:
                last_err = e
                msg = str(e)
                if "ThrottlingException" in msg or "TooManyRequests" in msg:
                    self.stats[ord_].throttled += 1
                    await asyncio.sleep(0.5 * (2**attempt) + random.random() * 0.3)
                    continue
                self.stats[ord_].errors += 1
                return CallResult("", {}, 0, 0, 0.0, region,
                                  self._resolve_model_id(w),
                                  error=f"{type(e).__name__}: {msg[:200]}")
        self.stats[ord_].errors += 1
        return CallResult("", {}, 0, 0, 0.0, "", "",
                          error=f"throttled after {attempts}: {last_err}")

    def _invoke_sync(self, w, region, prompt, max_tokens, temperature) -> CallResult:
        """Use Bedrock Converse API uniformly across all providers.

        Converse handles per-provider invocation format internally (Anthropic
        Messages, Nova, OpenAI-compat for Mistral/Kimi/MiniMax/GLM/etc.) and
        returns a normalized {role, content[]} message. Reasoning is split
        into its own `reasoningContent` block — we filter to `text` blocks
        only so extractors never see raw reasoning leaking in.
        """
        client = self._client(region)
        model_id = self._resolve_model_id(w)

        infer_cfg = {"maxTokens": max_tokens}
        if w.supports_temperature:
            infer_cfg["temperature"] = temperature

        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=infer_cfg,
        )

        # Filter to text blocks only (skip reasoningContent / toolUse / etc.)
        msg = resp.get("output", {}).get("message", {})
        content = msg.get("content", [])
        text_parts = [c["text"] for c in content if "text" in c]
        text = "\n".join(text_parts)

        usage = resp.get("usage", {})
        in_tok = usage.get("inputTokens", 0)
        out_tok = usage.get("outputTokens", 0)

        return CallResult(
            text=text,
            raw_response=resp,
            input_tok=in_tok,
            output_tok=out_tok,
            latency_ms=0.0,
            region=region,
            model_id=model_id,
        )

    def stats_summary(self) -> dict:
        out = {}
        for o, w in self.workers.items():
            s = self.stats[o]
            out[o] = {
                "name": w.name,
                "calls": s.calls,
                "errors": s.errors,
                "throttled": s.throttled,
                "avg_latency_ms": round(s.total_latency_ms / max(s.calls, 1), 1),
                "total_cost_$": round(s.total_cost_dollars, 4),
            }
        return out
