"""
Bedrock Converse dispatch — drop-in replacement for fugu/llm_clients.py's
per-provider dispatch (query_oai / query_gemini / query_deepseek /
query_locally_hosted_model), folded into one Converse-based path.

Why a monkeypatch and not an edit: the spec fixes the vendored algorithm/loop
(es.py, core.py, head_modules.py) as upstream-reference; we adapt only the
*clients*. fugu/utils.py:_query_llm is the single dispatch seam — every worker
call in core.py funnels through it. install() rebinds that seam to route through
Bedrock Converse, keyed on the per-worker payload that worker_pool_bedrock.py
stashes in fugu's AGENT_CONFIGS registry.

Throttle handling at CMA-ES population scale (spec §throttle handling):
  - per-(worker, region) asyncio semaphore so one model can't saturate a region
  - round-robin across REGIONS on ThrottlingException, advancing region per retry
  - exponential backoff + jitter (0.5s * 2^attempt + U(0,0.5), cap 20s), ~8 attempts
  - after max attempts the call returns "" -> fugu marks the episode an
    infrastructure_failure (reward 0, logged) instead of crashing the iteration
  - THROTTLE_TELEMETRY accumulates per-iter counts; run_trinity_agent reads it to
    enforce the >2% dropped-episode rule (rate-limited, not compute-bound)

GPT-5.5 optional path (off by default): a worker with transport="openai_compat"
is invoked via the Bedrock OpenAI-compatible endpoint with the operator's bearer
token (BEDROCK_BEARER_TOKEN), NOT SigV4/IRSA. Only ord 0 ever uses this, only when
the Gate-0.0 conditional enabled it. All other workers stay on Converse/IRSA.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from worker_pool_bedrock import REGIONS, by_friendly_name


# ---------------------------------------------------------------------------
# Telemetry — read + reset per CMA-ES iter by run_trinity_agent.py
# ---------------------------------------------------------------------------
class _Telemetry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.throttles = 0
        self.dropped = 0          # calls that exhausted retries -> "" -> infra failure
        self.region_advances = 0

    def reset(self) -> None:
        with self._lock:
            self.calls = self.throttles = self.dropped = self.region_advances = 0

    def snapshot(self) -> dict:
        with self._lock:
            dropped_rate = self.dropped / self.calls if self.calls else 0.0
            return {
                "calls": self.calls,
                "throttles": self.throttles,
                "dropped": self.dropped,
                "region_advances": self.region_advances,
                "dropped_rate": dropped_rate,
            }

    def _inc(self, field: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + n)


THROTTLE_TELEMETRY = _Telemetry()


# ---------------------------------------------------------------------------
# Per-(region) Bedrock clients (thread-safe lazy cache) + per-(worker, region)
# semaphores. We use threading primitives because fugu drives worker calls from a
# ThreadPoolExecutor (batch_completion / es.py), not an asyncio loop.
# ---------------------------------------------------------------------------
_CLIENTS: Dict[str, "boto3.client"] = {}
_CLIENTS_LOCK = threading.Lock()

_SEMAPHORES: Dict[tuple, threading.BoundedSemaphore] = {}
_SEM_LOCK = threading.Lock()

_SDK_CONFIG = Config(
    connect_timeout=5,
    read_timeout=600,
    retries={"total_max_attempts": 3, "mode": "adaptive"},
)


def _client(region: str) -> "boto3.client":
    with _CLIENTS_LOCK:
        c = _CLIENTS.get(region)
        if c is None:
            c = boto3.client("bedrock-runtime", region_name=region, config=_SDK_CONFIG)
            _CLIENTS[region] = c
        return c


def _semaphore(friendly_name: str, region: str, concurrency: int) -> threading.BoundedSemaphore:
    key = (friendly_name, region)
    with _SEM_LOCK:
        s = _SEMAPHORES.get(key)
        if s is None:
            s = threading.BoundedSemaphore(concurrency)
            _SEMAPHORES[key] = s
        return s


# ---------------------------------------------------------------------------
# Message conversion (OpenAI-style dicts -> Converse schema)
# ---------------------------------------------------------------------------
def _to_converse(messages: List[Dict]) -> tuple[Optional[list], list]:
    system_blocks: Optional[list] = None
    converted: list = []
    for m in messages:
        if m["role"] == "system":
            system_blocks = [{"text": m["content"]}]
            continue
        converted.append({"role": m["role"], "content": [{"text": m["content"]}]})
    # Converse requires the first non-system message to be 'user'; fugu always
    # leads with a user turn for workers, but guard defensively.
    if converted and converted[0]["role"] != "user":
        converted.insert(0, {"role": "user", "content": [{"text": ""}]})
    return system_blocks, converted


def _extract_text(content_blocks: list) -> str:
    """Join visible text, dropping reasoningContent (thinking) blocks.

    Bedrock returns reasoning as a separate {"reasoningContent": {...}} block
    (confirmed live for Qwen3 reasoning + DeepSeek-R1). Trinity's role parsers
    (<suggestion>/<suggested_role>, ACCEPT/REJECT, <answer>) operate on the
    visible answer only, so we strip thinking exactly like cost-aware-routing's
    worker_proxy did. Any literal <think>...</think> the model emits inline is
    left for fugu/core.py's tag handling, which already strips it.
    """
    out: list[str] = []
    for b in content_blocks:
        if not isinstance(b, dict):
            continue
        if "text" in b:
            out.append(b["text"])
        # reasoningContent / toolUse / etc. intentionally ignored
    return "".join(out)


# ---------------------------------------------------------------------------
# Converse path (default — all 6 IRSA workers + Opus 4.8)
# ---------------------------------------------------------------------------
def _query_converse(
    model_id: str,
    friendly_name: str,
    concurrency: int,
    reasoning_effort: Optional[str],
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    max_attempts: int = 8,
) -> str:
    system_blocks, converted = _to_converse(messages)

    base_params: dict = {
        "modelId": model_id,
        "messages": converted,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system_blocks is not None:
        base_params["system"] = system_blocks
    if reasoning_effort is not None:
        # Qwen3-32B / DeepSeek-R1 on Bedrock use OpenAI-style reasoning_effort
        # (NOT Anthropic reasoning_config — that raises a validation error here).
        base_params["additionalModelRequestFields"] = {"reasoning_effort": reasoning_effort}

    THROTTLE_TELEMETRY._inc("calls")
    n_regions = len(REGIONS)

    for attempt in range(max_attempts):
        region = REGIONS[attempt % n_regions]
        if attempt > 0 and (attempt % n_regions) == 0:
            pass  # wrapped around regions; backoff below still applies
        sem = _semaphore(friendly_name, region, concurrency)
        with sem:
            try:
                resp = _client(region).converse(**base_params)
                return _extract_text(resp["output"]["message"]["content"])
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in ("ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException"):
                    THROTTLE_TELEMETRY._inc("throttles")
                    THROTTLE_TELEMETRY._inc("region_advances")
                    _backoff(attempt)
                    continue
                # Non-throttle ClientError: one cross-region retry then give up.
                if attempt < n_regions:
                    _backoff(attempt)
                    continue
                THROTTLE_TELEMETRY._inc("dropped")
                return ""
            except Exception:
                if attempt < max_attempts - 1:
                    _backoff(attempt)
                    continue
                THROTTLE_TELEMETRY._inc("dropped")
                return ""

    THROTTLE_TELEMETRY._inc("dropped")
    return ""


def _backoff(attempt: int) -> None:
    sleep_for = min(0.5 * (2 ** attempt) + random.uniform(0, 0.5), 20.0)
    time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# OpenAI-compatible path (GPT-5.5 optional ord-0 swap only)
# ---------------------------------------------------------------------------
def _query_openai_compat(
    model_id: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    max_attempts: int = 8,
) -> str:
    """Bedrock OpenAI-compatible endpoint, bearer-token auth.

    Used ONLY for a worker whose transport == "openai_compat" (GPT-5.5 at ord 0,
    enabled at Gate 0.0). Reaches the grant via the operator's bearer token, not
    IRSA SigV4. The request shape mirrors the upstream query_oai chat schema.
    """
    import urllib.request
    import urllib.error

    token = os.environ.get("BEDROCK_BEARER_TOKEN")
    if not token:
        # Misconfiguration: openai_compat selected but no token mounted.
        THROTTLE_TELEMETRY._inc("dropped")
        return ""
    region = os.environ.get("BEDROCK_OPENAI_REGION", REGIONS[0])
    url = f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/chat/completions"
    payload = {
        "model": model_id,                       # e.g. "openai.gpt-5.5"
        "messages": messages,                    # OpenAI chat schema (role/content strings)
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    THROTTLE_TELEMETRY._inc("calls")
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=600) as r:
                body = json.loads(r.read())
            return body["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                THROTTLE_TELEMETRY._inc("throttles")
                _backoff(attempt)
                continue
            if attempt < 2:
                _backoff(attempt)
                continue
            THROTTLE_TELEMETRY._inc("dropped")
            return ""
        except Exception:
            if attempt < max_attempts - 1:
                _backoff(attempt)
                continue
            THROTTLE_TELEMETRY._inc("dropped")
            return ""
    THROTTLE_TELEMETRY._inc("dropped")
    return ""


# ---------------------------------------------------------------------------
# The dispatch entry point that replaces fugu.utils._query_llm internals.
# ---------------------------------------------------------------------------
def query_bedrock_dispatch(
    model: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    server: Optional[str] = None,
    port: Optional[int] = None,
    debug: bool = False,
    together: bool = True,
    **kwargs,
) -> str:
    """Route a fugu worker call to Bedrock.

    `model` here is fugu's already-resolved actual_model. With our AGENT_CONFIGS,
    that's the friendly_name, and the per-worker Bedrock knobs ride in kwargs as
    _bedrock_model_id / _transport / _reasoning_effort (injected by fugu's
    payload-merge). We pop them so they never reach the Bedrock API.
    """
    model_id = kwargs.pop("_bedrock_model_id", None)
    transport = kwargs.pop("_transport", "converse")
    reasoning_effort = kwargs.pop("_reasoning_effort", None)
    kwargs.pop("_ord", None)
    # Upstream vLLM-only knobs that may ride along in payloads — drop them.
    for k in ("top_k", "top_p", "presence_penalty", "chat_template_kwargs"):
        kwargs.pop(k, None)

    if model_id is None:
        # Fall back to friendly-name lookup if payload wasn't merged for some path.
        w = by_friendly_name.get(model)
        if w is None:
            if debug:
                print(f"[bedrock_dispatch] unknown worker '{model}'")
            return ""
        model_id = w.model_id
        transport = w.transport
        from worker_pool_bedrock import reasoning_effort_for
        reasoning_effort = reasoning_effort_for(w)

    w = by_friendly_name.get(model)
    concurrency = w.concurrency if w else 10

    if transport == "openai_compat":
        return _query_openai_compat(model_id, messages, max_tokens, temperature)
    return _query_converse(
        model_id, model, concurrency, reasoning_effort,
        messages, max_tokens, temperature,
    )


def install() -> None:
    """Rebind fugu's dispatch seam to Bedrock.

    fugu.utils._query_llm is wrapped by log_api_call_wrapper and also retried by a
    tenacity decorator; we replace the *inner* provider dispatch by swapping the
    module-level per-provider functions it calls. Cleanest single seam: rebind
    fugu.utils._query_llm's body via a thin shim that ignores server/port and
    always calls Bedrock. We preserve fugu's cost tracking by letting it run after.
    """
    import fugu.utils as U

    # Replace the provider-dispatch functions so _query_llm routes to Bedrock for
    # every branch (oai/anthropic/gemini/deepseek), regardless of name heuristics.
    def _bedrock_any(model, messages, max_tokens, temperature, *a, **kw):
        return query_bedrock_dispatch(model, messages, max_tokens, temperature, **kw)

    U.query_oai = _bedrock_any
    U.query_anthropic = _bedrock_any
    U.query_gemini = _bedrock_any
    # query_deepseek has a use_together kwarg; absorb it.
    U.query_deepseek = lambda model, messages, max_tokens, temperature, *a, **kw: (
        _bedrock_any(model, messages, max_tokens, temperature,
                     **{k: v for k, v in kw.items() if k != "use_together"})
    )
    # Local vLLM path should never be taken (no server/port in AGENT_CONFIGS), but
    # rebind it too so a stray config can't escape to HTTP.
    U.query_locally_hosted_model = lambda model, messages, max_tokens, temperature, server, port, *a, **kw: (
        _bedrock_any(model, messages, max_tokens, temperature, **kw)
    )

    # Force the cloud-API branch in _query_llm (server/port None) by ensuring our
    # AGENT_CONFIGS carry no port. Nothing to patch here — documented invariant.


if __name__ == "__main__":
    # Liveness smoke (Gate 0.2): 1-token Converse ping per worker. Requires boto3
    # + Bedrock creds; run inside the Job or an environment with both.
    from worker_pool_bedrock import POOL, reasoning_effort_for

    print("Gate 0.2 — worker liveness via Converse:")
    for w in POOL:
        if w.transport == "openai_compat":
            print(f"  ord {w.ord} {w.friendly_name}: openai_compat (skipped — needs bearer token)")
            continue
        txt = _query_converse(
            w.model_id, w.friendly_name, w.concurrency, reasoning_effort_for(w),
            [{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=16, temperature=0.0,
        )
        ok = "ok" in txt.lower()
        print(f"  ord {w.ord} {w.friendly_name:24s} -> {'✅' if ok else '❌'} {txt[:40]!r}")
    print("telemetry:", THROTTLE_TELEMETRY.snapshot())
