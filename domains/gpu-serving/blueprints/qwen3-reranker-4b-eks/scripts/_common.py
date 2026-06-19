"""Shared helpers for Qwen3-Reranker-4B Stage 6 benchmark runners.

Conventions (classifier / cross-encoder, non-streaming):
  - api.type = "score", streaming = false
  - Endpoint: /v1/score, body {"model": ..., "text_1": query, "text_2": [candidates]}
  - One request = one (query, k candidates) → k float scores returned
  - Headline metric: per-request e2e latency (time to score all k pairs in a batch)
  - output_toks_per_s is meaningless for a classifier — core metrics.output_toks_per_s = 0.0
  - True reranker throughput unit is pairs_per_s; surfaced at extensions.reranker.pairs_per_s
  - Only prompt_tokens reported by vLLM (no completion tokens — pooling model)

Substrate caveat: this blueprint runs on g6e.2xlarge (L40S 48GB) for session-reuse
reasons. The spec-preferred instance is g6.xlarge (L4 24GB). Numbers are a valid
per-stream upper bound for L40S but are NOT a valid cost claim for the L4 row.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import random
import statistics
import uuid
from dataclasses import dataclass
from pathlib import Path

ENDPOINT = "http://localhost:8000"
MODEL_ID = "Qwen/Qwen3-Reranker-4B"
SCHEMA_VERSION = "1.0.0"
ENRICHMENT_VERSION = "1.0.0"
SOURCE_TOOL_NAME = "custom"

MODEL_BLOCK = {
    "name": "Qwen3-Reranker-4B",
    "id": MODEL_ID,
    "architecture": "cross-encoder",
    "parameters_total": "4B",
    "quantization": "bf16",
    "max_model_len": 4096,
}

ENGINE_BLOCK = {
    "name": "vllm",
    "version": "0.19.1",
    "container_image": "vllm/vllm-openai:v0.19.1",
    "base_image": None,
    "dockerfile": None,
    "tensor_parallel": 1,
    "pipeline_parallel": 1,
    "data_parallel": None,
    "expert_parallel": None,
    "replicas": 1,
    "reasoning": False,
    "kv_cache_dtype": "auto",
    "attention_backend": "flash-attn",
    "speculative_decode": None,
    "extra_args": {
        "runner": "pooling",
        "convert": "classify",
        "trust-remote-code": True,
        "max-num-seqs": 64,
        "gpu-memory-utilization": 0.90,
        "hf_overrides": {
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    },
}

# NOTE: substrate is g6e.2xlarge (L40S 48GB) for session reuse, NOT the
# spec-preferred g6.xlarge (L4 24GB). This is an upper-bound perf measurement.
INFRA_BLOCK = {
    "substrate": "eks",
    "instance_type": "g6e.2xlarge",
    "region": "us-east-2",
    "gpu": {
        "name": "L40S",
        "arch": "sm_89",
        "count": 1,
        "vram_gb": 48,
        "interconnect": "none",
    },
    "substrate_deviation": {
        "spec_preferred_instance": "g6.xlarge",
        "spec_preferred_gpu": "L4 24GB",
        "reason": (
            "Session reuse — g6e.2xlarge already provisioned for deepseek-ocr. "
            "Cost-efficiency row is NOT valid for L4; per-stream latency is an "
            "upper bound (L40S has ~33% more TFLOPS and 2x VRAM headroom)."
        ),
    },
}


@dataclass
class CandidateCorpus:
    query: str
    candidates: list[str]  # length k
    pair_length_target: int  # target per-pair tokens
    seed: int


# ------------------------------ helpers ---------------------------------
_LOREM = (
    "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim "
    "veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
    "commodo consequat Duis aute irure dolor in reprehenderit in voluptate "
    "velit esse cillum dolore eu fugiat nulla pariatur Excepteur sint "
    "occaecat cupidatat non proident sunt in culpa qui officia deserunt "
    "mollit anim id est laborum "
).split()


def _lorem_chunk(word_count: int, rng: random.Random) -> str:
    # Sample with wraparound; shuffle a local copy each time so candidates differ.
    words = list(_LOREM)
    rng.shuffle(words)
    out = []
    i = 0
    while len(out) < word_count:
        out.append(words[i % len(words)])
        i += 1
    return " ".join(out)


def build_corpus(
    k: int = 50,
    pair_length: int = 1024,
    seed: int = 42,
    query_words: int = 16,
) -> CandidateCorpus:
    """Build a fixed (query, k candidates) corpus.

    `pair_length` is the approximate per-pair token budget; candidates are
    sized to roughly that many tokens (~0.75 tokens/word lorem). We subtract
    the query word count + a small template overhead so the total stays under
    the requested budget.
    """
    rng = random.Random(seed)
    query = _lorem_chunk(query_words, rng) + " what is the primary topic"
    # token ~= 0.75 words for lorem; we want pair_length tokens total.
    # Subtract query overhead + reranker template tokens (~40).
    per_cand_words = max(16, int(pair_length / 0.75) - query_words - 40)
    candidates = [_lorem_chunk(per_cand_words, rng) for _ in range(k)]
    return CandidateCorpus(
        query=query,
        candidates=candidates,
        pair_length_target=pair_length,
        seed=seed,
    )


def build_request_body(corpus: CandidateCorpus) -> dict:
    return {
        "model": MODEL_ID,
        "text_1": corpus.query,
        "text_2": corpus.candidates,
    }


def compute_percentiles(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    xs = sorted(latencies_ms)

    def pct(p: float) -> float:
        k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
        return xs[k]

    return {
        "mean": statistics.fmean(xs),
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
    }


def envelope() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": str(uuid.uuid4()),
        "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_tool": {
            "name": SOURCE_TOOL_NAME,
            "version": "0.1.0-qwen3-reranker",
            "enrichment_version": ENRICHMENT_VERSION,
        },
    }


NULL_LATENCY = {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None}


def write_artifact(out_path: Path, doc: dict) -> Path:
    """Write the artifact JSON with ttft/tpot/itl nulled (classifier, non-streaming)."""
    metrics = doc.setdefault("metrics", {})
    for k in ("ttft_ms", "tpot_ms", "itl_ms"):
        metrics.setdefault(k, dict(NULL_LATENCY))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    return out_path
