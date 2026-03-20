#!/usr/bin/env python3
"""
Phase 2b: ThunderAgent Scheduling Proxy.

Program-aware proxy that sits between coding agents and vLLM,
filling GPU bubbles during tool execution with backfill requests.

Architecture:
  Agents → :9000 (this proxy) → :8000 (vLLM, Qwen3.5 TP4, max-num-seqs=4)

Key idea: When agent A finishes an inference call and starts executing tools
(ACTING state), the GPU slot holding A's KV cache is "reclaimable". The proxy
admits a queued request from agent B to backfill that slot. When A returns
with a new inference request, B's in-flight request is allowed to finish
(vLLM handles queuing internally), but A gets priority in the next scheduling
round.

This is a simplified ThunderAgent (arxiv 2602.13692) — no preemption or KV
eviction, just admission control + priority reordering.

Usage:
  python3 thunder_proxy.py --backend http://localhost:8000 --port 9000 \
      --max-active 4 --tool-coefficient 1.5
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import aiohttp
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [thunder] %(message)s",
)
log = logging.getLogger("thunder")


class ProgramState(str, Enum):
    IDLE = "idle"           # No active request, no recent activity
    REASONING = "reasoning"  # Inference request in flight (on GPU)
    ACTING = "acting"       # Tool execution in progress (off GPU, slot reclaimable)


@dataclass
class Program:
    """Tracks one agent session (one 'LLM Program')."""
    program_id: str
    state: ProgramState = ProgramState.IDLE
    last_request_time: float = 0.0
    last_response_time: float = 0.0
    total_reasoning_s: float = 0.0
    total_acting_s: float = 0.0
    request_count: int = 0
    backfill_count: int = 0  # times this program was a backfill
    priority: int = 0        # lower = higher priority (original > backfill)


@dataclass
class ProxyStats:
    """Aggregated proxy metrics."""
    total_requests: int = 0
    direct_admits: int = 0       # admitted immediately (slot available)
    queued_requests: int = 0     # had to wait in queue
    backfill_admits: int = 0     # admitted as backfill into reclaimable slot
    active_programs: int = 0
    acting_programs: int = 0     # programs in tool execution (reclaimable slots)
    reasoning_programs: int = 0  # programs with active inference
    queue_depth: int = 0
    avg_queue_wait_s: float = 0.0
    gpu_utilization_pct: float = 0.0  # reasoning / (reasoning + acting) among active
    start_time: float = field(default_factory=time.monotonic)


class ThunderProxy:
    def __init__(self, backend_url: str, max_active: int, tool_coefficient: float):
        self.backend_url = backend_url.rstrip("/")
        self.max_active = max_active  # vLLM max-num-seqs
        self.tool_coefficient = tool_coefficient  # oversubscription factor
        self.effective_slots = int(max_active * tool_coefficient)

        self.programs: Dict[str, Program] = {}
        self.stats = ProxyStats()

        # Admission control
        self._reasoning_count = 0  # currently in-flight inference requests
        self._admit_event = asyncio.Event()
        self._admit_event.set()  # start open

        # Queue for waiting requests
        self._queue: asyncio.Queue = asyncio.Queue()

        # HTTP session (created on startup)
        self._session: Optional[aiohttp.ClientSession] = None

        # Metrics snapshot interval
        self._metrics_history: List[dict] = []

    def _get_or_create_program(self, program_id: str) -> Program:
        if program_id not in self.programs:
            self.programs[program_id] = Program(program_id=program_id)
        return self.programs[program_id]

    def _extract_program_id(self, request: web.Request) -> str:
        """Extract program_id from request headers or infer from connection.

        Priority:
        1. X-Program-ID header (set by ThunderAgent-aware clients)
        2. Authorization header with "Bearer prog-*" prefix (set by adapter
           via OPENCODE_API_KEY=prog-<worker_id>)
        3. Stable hash of first user message (fingerprints the conversation)
        4. Fallback: per-request unique ID (no program tracking)
        """
        pid = request.headers.get("X-Program-ID")
        if pid:
            return pid
        # Check if API key encodes a program ID (adapter sets OPENCODE_API_KEY=prog-*)
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer prog-"):
            return auth[7:]  # strip "Bearer "
        # Hash first user message as stable fingerprint
        if hasattr(request, '_thunder_body'):
            try:
                body = json.loads(request._thunder_body)
                msgs = body.get("messages", [])
                for m in msgs:
                    if m.get("role") == "user":
                        content = str(m.get("content", ""))[:200]
                        import hashlib
                        return f"msg-{hashlib.md5(content.encode()).hexdigest()[:12]}"
            except (json.JSONDecodeError, KeyError):
                pass
        return f"anon-{uuid.uuid4().hex[:8]}"

    def _update_utilization(self):
        """Recalculate GPU utilization estimate."""
        reasoning = sum(1 for p in self.programs.values() if p.state == ProgramState.REASONING)
        acting = sum(1 for p in self.programs.values() if p.state == ProgramState.ACTING)
        self._reasoning_count = reasoning
        self.stats.reasoning_programs = reasoning
        self.stats.acting_programs = acting
        self.stats.active_programs = reasoning + acting
        total_active = reasoning + acting
        if total_active > 0:
            self.stats.gpu_utilization_pct = round(100 * reasoning / self.max_active, 1)
        else:
            self.stats.gpu_utilization_pct = 0.0

    async def _try_admit(self, program_id: str) -> str:
        """
        Admission control. Returns 'direct', 'backfill', or 'queued'.

        Rules:
        1. If reasoning_count < max_active → direct admit
        2. If reasoning_count >= max_active but acting_count > 0 → backfill admit
           (an acting program's slot is reclaimable)
        3. Otherwise → queue
        """
        self._update_utilization()

        # Rule 1: Direct admit
        if self._reasoning_count < self.max_active:
            return "direct"

        # Rule 2: Backfill — there's a program in ACTING whose slot we can opportunistically use
        # Only allow up to effective_slots total reasoning
        if self._reasoning_count < self.effective_slots and self.stats.acting_programs > 0:
            return "backfill"

        # Rule 3: Queue
        return "queued"

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """Proxy /v1/chat/completions with ThunderAgent scheduling."""
        # Read body early so _extract_program_id can use message fingerprinting
        body = await request.read()
        request._thunder_body = body

        program_id = self._extract_program_id(request)
        program = self._get_or_create_program(program_id)
        self.stats.total_requests += 1

        # Mark program as wanting to reason
        if program.state == ProgramState.ACTING:
            # Returning from tool execution — record acting duration
            program.total_acting_s += time.monotonic() - program.last_response_time

        request_start = time.monotonic()
        program.last_request_time = request_start

        # Admission control
        admit_type = await self._try_admit(program_id)

        if admit_type == "queued":
            self.stats.queued_requests += 1
            self.stats.queue_depth += 1
            log.info(f"  [{program_id[:8]}] QUEUED (reasoning={self._reasoning_count}, "
                     f"acting={self.stats.acting_programs}, queue={self.stats.queue_depth})")

            # Wait for a slot to open
            my_event = asyncio.Event()
            await self._queue.put((program_id, my_event))
            await my_event.wait()
            self.stats.queue_depth -= 1

            # Re-check admission type after wait
            self._update_utilization()
            if self._reasoning_count < self.max_active:
                admit_type = "direct"
            else:
                admit_type = "backfill"

        if admit_type == "direct":
            self.stats.direct_admits += 1
        elif admit_type == "backfill":
            self.stats.backfill_admits += 1
            program.backfill_count += 1

        # Transition to REASONING
        program.state = ProgramState.REASONING
        program.request_count += 1
        queue_wait = time.monotonic() - request_start
        self._update_utilization()

        log.info(f"  [{program_id[:8]}] REASONING ({admit_type}, "
                 f"wait={queue_wait:.1f}s, active={self._reasoning_count}/{self.max_active})")

        # Body was already read for program ID extraction
        headers = {
            "Content-Type": "application/json",
        }
        # Pass through auth
        if "Authorization" in request.headers:
            headers["Authorization"] = request.headers["Authorization"]

        # Determine if streaming
        try:
            req_json = json.loads(body)
            is_stream = req_json.get("stream", False)
        except (json.JSONDecodeError, UnicodeDecodeError):
            is_stream = False

        # Proxy to backend
        backend_path = request.path
        backend_url = f"{self.backend_url}{backend_path}"

        try:
            if is_stream:
                response = await self._proxy_streaming(
                    request, backend_url, headers, body, program, program_id
                )
            else:
                response = await self._proxy_non_streaming(
                    backend_url, headers, body, program, program_id
                )
        except Exception as e:
            # On error, release slot
            program.state = ProgramState.IDLE
            self._update_utilization()
            self._release_slot()
            log.warning(f"  [{program_id[:8]}] ERROR: {e}")
            return web.json_response(
                {"error": {"message": str(e), "type": "proxy_error"}},
                status=502,
            )

        return response

    async def _proxy_non_streaming(
        self, url: str, headers: dict, body: bytes, program: Program, pid: str
    ) -> web.Response:
        """Proxy a non-streaming request."""
        async with self._session.post(url, headers=headers, data=body) as resp:
            resp_body = await resp.read()
            resp_headers = {
                k: v for k, v in resp.headers.items()
                if k.lower() not in ("transfer-encoding", "content-encoding")
            }

            # Transition to ACTING (tool execution expected)
            reasoning_time = time.monotonic() - program.last_request_time
            program.total_reasoning_s += reasoning_time
            program.last_response_time = time.monotonic()
            program.state = ProgramState.ACTING
            self._update_utilization()
            self._release_slot()

            log.info(f"  [{pid[:8]}] ACTING (inference={reasoning_time:.1f}s, "
                     f"reasoning_now={self._reasoning_count}/{self.max_active})")

            return web.Response(
                body=resp_body,
                status=resp.status,
                headers=resp_headers,
            )

    async def _proxy_streaming(
        self, orig_request: web.Request, url: str, headers: dict,
        body: bytes, program: Program, pid: str,
    ) -> web.StreamResponse:
        """Proxy a streaming request, detecting end-of-stream for state transition."""
        async with self._session.post(url, headers=headers, data=body) as resp:
            response = web.StreamResponse(
                status=resp.status,
                headers={
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in ("transfer-encoding", "content-length")
                },
            )
            response.content_type = resp.content_type
            await response.prepare(orig_request)

            async for chunk in resp.content.iter_any():
                await response.write(chunk)

            # Stream complete — transition to ACTING
            reasoning_time = time.monotonic() - program.last_request_time
            program.total_reasoning_s += reasoning_time
            program.last_response_time = time.monotonic()
            program.state = ProgramState.ACTING
            self._update_utilization()
            self._release_slot()

            log.info(f"  [{pid[:8]}] ACTING (stream done, inference={reasoning_time:.1f}s)")

            await response.write_eof()
            return response

    def _release_slot(self):
        """Release a reasoning slot, unblock a queued request if any."""
        # Try to unblock a queued request
        try:
            while not self._queue.empty():
                pid, event = self._queue.get_nowait()
                if not event.is_set():
                    event.set()
                    return
        except asyncio.QueueEmpty:
            pass

    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Return current proxy metrics."""
        self._update_utilization()
        metrics = {
            "timestamp": time.time(),
            "uptime_s": round(time.monotonic() - self.stats.start_time, 1),
            "total_requests": self.stats.total_requests,
            "direct_admits": self.stats.direct_admits,
            "backfill_admits": self.stats.backfill_admits,
            "queued_requests": self.stats.queued_requests,
            "current_reasoning": self.stats.reasoning_programs,
            "current_acting": self.stats.acting_programs,
            "current_queue_depth": self.stats.queue_depth,
            "gpu_utilization_pct": self.stats.gpu_utilization_pct,
            "max_active": self.max_active,
            "effective_slots": self.effective_slots,
            "tool_coefficient": self.tool_coefficient,
            "programs": {
                pid: {
                    "state": p.state.value,
                    "requests": p.request_count,
                    "reasoning_s": round(p.total_reasoning_s, 1),
                    "acting_s": round(p.total_acting_s, 1),
                    "backfills": p.backfill_count,
                }
                for pid, p in self.programs.items()
            },
        }
        return web.json_response(metrics)

    async def handle_passthrough(self, request: web.Request) -> web.Response:
        """Pass through non-chat endpoints (models, health, etc.) without scheduling."""
        body = await request.read()
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "transfer-encoding")
        }
        url = f"{self.backend_url}{request.path}"

        async with self._session.request(
            request.method, url, headers=headers, data=body
        ) as resp:
            resp_body = await resp.read()
            return web.Response(
                body=resp_body,
                status=resp.status,
                content_type=resp.content_type,
            )

    async def start(self, port: int):
        """Start the proxy server."""
        timeout = aiohttp.ClientTimeout(total=600, connect=10)
        self._session = aiohttp.ClientSession(timeout=timeout)

        app = web.Application()
        app.router.add_post("/v1/chat/completions", self.handle_chat_completions)
        app.router.add_get("/thunder/metrics", self.handle_metrics)
        # Pass through everything else
        app.router.add_route("*", "/{path:.*}", self.handle_passthrough)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()

        log.info(f"ThunderAgent proxy listening on :{port}")
        log.info(f"Backend: {self.backend_url}")
        log.info(f"Max active (vLLM max-num-seqs): {self.max_active}")
        log.info(f"Effective slots (with tool_coefficient={self.tool_coefficient}): {self.effective_slots}")
        log.info(f"Admission: direct < {self.max_active}, backfill < {self.effective_slots}")

        # Periodic metrics logging
        while True:
            await asyncio.sleep(30)
            self._update_utilization()
            log.info(
                f"  [metrics] reasoning={self.stats.reasoning_programs} "
                f"acting={self.stats.acting_programs} "
                f"queue={self.stats.queue_depth} "
                f"util={self.stats.gpu_utilization_pct}% "
                f"total={self.stats.total_requests} "
                f"backfills={self.stats.backfill_admits}"
            )


async def main():
    parser = argparse.ArgumentParser(description="ThunderAgent Scheduling Proxy")
    parser.add_argument("--backend", default="http://localhost:8000", help="vLLM backend URL")
    parser.add_argument("--port", type=int, default=9000, help="Proxy listen port")
    parser.add_argument("--max-active", type=int, default=4, help="vLLM max-num-seqs")
    parser.add_argument(
        "--tool-coefficient", type=float, default=1.5,
        help="Oversubscription factor (1.0 = no oversubscription, 2.0 = 2x slots)"
    )
    args = parser.parse_args()

    proxy = ThunderProxy(
        backend_url=args.backend,
        max_active=args.max_active,
        tool_coefficient=args.tool_coefficient,
    )
    await proxy.start(args.port)


if __name__ == "__main__":
    asyncio.run(main())
