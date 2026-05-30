"""Worker proxy for RL Conductor training.

Routes model_id integers to backend LLM endpoints.
Tracks cost/latency/tokens per worker for analysis.
"""

import asyncio
import json
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

# Global socket timeout as safety net for all network calls
socket.setdefaulttimeout(45)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Bedrock config (preferred for Anthropic models — uses instance profile)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
USE_BEDROCK = os.environ.get("USE_BEDROCK", "1") == "1"


@dataclass
class WorkerConfig:
    model_id: int
    name: str
    backend: str  # "anthropic", "openai", "google", "vllm"
    model_name: str
    endpoint: Optional[str] = None
    cost_per_1m_output: float = 0.0


@dataclass
class WorkerStats:
    calls: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0


# Phase 1: 6-model pool via Bedrock (no API keys, instance profile auth)
# Diverse capabilities: reasoning (Opus), general (Sonnet/Kimi/GLM), fast (Haiku), code (Qwen3-Coder)
PHASE1_WORKERS = [
    WorkerConfig(0, "opus-4.7", "bedrock", "us.anthropic.claude-opus-4-7", cost_per_1m_output=75.0),
    WorkerConfig(1, "sonnet-4.6", "bedrock", "us.anthropic.claude-sonnet-4-6", cost_per_1m_output=15.0),
    WorkerConfig(2, "haiku-4.5", "bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0", cost_per_1m_output=1.25),
    WorkerConfig(3, "kimi-k2.5", "bedrock", "moonshotai.kimi-k2.5", cost_per_1m_output=8.0),
    WorkerConfig(4, "glm-5", "bedrock", "zai.glm-5", cost_per_1m_output=5.0),
    WorkerConfig(5, "qwen3-coder", "bedrock", "qwen.qwen3-coder-30b-a3b-v1:0", cost_per_1m_output=2.0),
]

# Phase 2: hybrid pool
PHASE2_WORKERS = [
    WorkerConfig(0, "qwen3.5-122b", "vllm", "Qwen/Qwen3.5-122B-A10B-FP8", endpoint="http://g7e-endpoint:8000/v1"),
    WorkerConfig(1, "glm-5", "vllm", "THUDM/GLM-5-FP8", endpoint="http://b200-endpoint:8000/v1"),
    WorkerConfig(2, "kimi-k2.6", "vllm", "moonshotai/Kimi-K2.6", endpoint="http://b300-endpoint:8000/v1"),
    WorkerConfig(3, "devstral-small-2", "vllm", "mistralai/Devstral-Small-2-2506", endpoint="http://g7e-12xl:8000/v1"),
    WorkerConfig(4, "haiku-4.5", "bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0", cost_per_1m_output=1.25),
    WorkerConfig(5, "sonnet-4.6", "bedrock", "us.anthropic.claude-sonnet-4-6", cost_per_1m_output=15.0),
    WorkerConfig(6, "opus-4.7", "bedrock", "us.anthropic.claude-opus-4-7", cost_per_1m_output=75.0),
]


class WorkerPool:
    def __init__(self, phase: int = 1):
        workers = PHASE1_WORKERS if phase == 1 else PHASE2_WORKERS
        self.workers = {w.model_id: w for w in workers}
        self.stats = {w.model_id: WorkerStats() for w in workers}
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        return self._session

    def call_worker_sync(self, model_id: int, prompt: str, max_tokens: int = 2048) -> str:
        """Synchronous worker call — fresh client per call to avoid pool deadlocks."""
        if model_id not in self.workers:
            return f"[ERROR: invalid model_id {model_id}]"

        worker = self.workers[model_id]
        start = time.monotonic()

        try:
            if worker.backend == "bedrock":
                result = self._call_bedrock_fresh(worker, prompt, max_tokens)
            else:
                result = f"[ERROR: sync call only supports bedrock backend, got {worker.backend}]"

            elapsed = (time.monotonic() - start) * 1000
            self.stats[model_id].calls += 1
            self.stats[model_id].total_latency_ms += elapsed
            self.stats[model_id].total_tokens += len(result.split())
            return result

        except Exception as e:
            self.stats[model_id].errors += 1
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _call_bedrock_fresh(self, worker: WorkerConfig, prompt: str, max_tokens: int) -> str:
        """Call Bedrock with a fresh client (no connection pooling)."""
        import boto3
        from botocore.config import Config as BotoConfig

        client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            config=BotoConfig(
                read_timeout=30,
                connect_timeout=10,
                retries={"max_attempts": 1},
                max_pool_connections=1,
            ),
        )

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        })

        resp = client.invoke_model(
            modelId=worker.model_name,
            body=body,
            contentType="application/json",
        )
        data = json.loads(resp["body"].read())

        if "content" in data and isinstance(data["content"], list):
            return data["content"][0]["text"]
        elif "choices" in data:
            return data["choices"][0]["message"]["content"]
        else:
            return str(data)

    async def call_worker(self, model_id: int, prompt: str, max_tokens: int = 2048) -> str:
        if model_id not in self.workers:
            return f"[ERROR: invalid model_id {model_id}]"

        worker = self.workers[model_id]
        start = time.monotonic()

        try:
            if worker.backend == "bedrock":
                result = await self._call_bedrock(worker, prompt, max_tokens)
            elif worker.backend == "anthropic":
                result = await self._call_anthropic(worker, prompt, max_tokens)
            elif worker.backend == "openai":
                result = await self._call_openai(worker, prompt, max_tokens)
            elif worker.backend == "google":
                result = await self._call_google(worker, prompt, max_tokens)
            elif worker.backend == "vllm":
                result = await self._call_vllm(worker, prompt, max_tokens)
            else:
                result = f"[ERROR: unknown backend {worker.backend}]"

            elapsed = (time.monotonic() - start) * 1000
            self.stats[model_id].calls += 1
            self.stats[model_id].total_latency_ms += elapsed
            self.stats[model_id].total_tokens += len(result.split())
            return result

        except Exception as e:
            self.stats[model_id].errors += 1
            return f"[ERROR: {type(e).__name__}: {e}]"

    async def _call_bedrock(self, worker: WorkerConfig, prompt: str, max_tokens: int) -> str:
        """Call Anthropic models via AWS Bedrock (uses instance profile credentials)."""
        import boto3

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._call_bedrock_sync, worker, prompt, max_tokens
        )

    def _call_bedrock_sync(self, worker: WorkerConfig, prompt: str, max_tokens: int) -> str:
        import boto3
        import threading

        from botocore.config import Config as BotoConfig

        # Thread-local client with timeout to avoid hung requests
        if not hasattr(self, "_thread_local"):
            self._thread_local = threading.local()
        if not hasattr(self._thread_local, "bedrock_client"):
            self._thread_local.bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=AWS_REGION,
                config=BotoConfig(
                    read_timeout=30,
                    connect_timeout=10,
                    retries={"max_attempts": 2},
                ),
            )

        # Anthropic models use Messages API; Kimi, GLM, Qwen use OpenAI-style
        is_anthropic = "anthropic" in worker.model_name

        if is_anthropic:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            })
        else:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            })

        resp = self._thread_local.bedrock_client.invoke_model(
            modelId=worker.model_name,
            body=body,
            contentType="application/json",
        )
        result = json.loads(resp["body"].read())

        # Handle both response formats
        if "content" in result and isinstance(result["content"], list):
            return result["content"][0]["text"]
        elif "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return str(result)

    async def _call_anthropic(self, worker: WorkerConfig, prompt: str, max_tokens: int) -> str:
        session = await self._get_session()
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": worker.model_name,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                return f"[ERROR: {data.get('error', {}).get('message', resp.status)}]"
            return data["content"][0]["text"]

    async def _call_openai(self, worker: WorkerConfig, prompt: str, max_tokens: int) -> str:
        session = await self._get_session()
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": worker.model_name,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                return f"[ERROR: {data.get('error', {}).get('message', resp.status)}]"
            return data["choices"][0]["message"]["content"]

    async def _call_google(self, worker: WorkerConfig, prompt: str, max_tokens: int) -> str:
        session = await self._get_session()
        async with session.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{worker.model_name}:generateContent?key={GOOGLE_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            },
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                return f"[ERROR: {data.get('error', {}).get('message', resp.status)}]"
            return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_vllm(self, worker: WorkerConfig, prompt: str, max_tokens: int) -> str:
        session = await self._get_session()
        async with session.post(
            f"{worker.endpoint}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": worker.model_name,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                return f"[ERROR: {data.get('error', {}).get('message', resp.status)}]"
            return data["choices"][0]["message"]["content"]

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def get_stats_summary(self) -> dict:
        summary = {}
        for mid, stats in self.stats.items():
            worker = self.workers[mid]
            avg_latency = stats.total_latency_ms / max(stats.calls, 1)
            summary[worker.name] = {
                "calls": stats.calls,
                "tokens": stats.total_tokens,
                "avg_latency_ms": round(avg_latency, 1),
                "errors": stats.errors,
                "est_cost": round(stats.total_tokens * worker.cost_per_1m_output / 1_000_000, 4),
            }
        return summary
