"""Shared helpers for DeepSeek-OCR-2 Stage 6 benchmark runners.

Conventions (VLM, non-streaming):
  - api.type = "chat", streaming = false
  - e2e_ms only (no ttft/tpot/itl)
  - usage tokens come from response.usage.{completion,prompt}_tokens
  - source_tool.name = "custom" (schema enum gate)

Iteration 5 additions:
  - load_corpus() for the 6-doc stratified corpus
  - compute_equivalent_pages() geometric-mean normalizer
  - write_artifact() accepts per_doc_type dict -> extensions.stratification.*
"""
from __future__ import annotations

import base64
import datetime
import json
import math
import statistics
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ENDPOINT = "http://localhost:8000"
MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "
SCHEMA_VERSION = "1.0.0"
ENRICHMENT_VERSION = "1.0.0"
SOURCE_TOOL_NAME = "custom"

MODEL_BLOCK = {
    "name": "DeepSeek-OCR-2",
    "id": MODEL_ID,
    "architecture": "vlm",
    "parameters_total": "8B",
    "quantization": "bf16",
    "max_model_len": 8192,
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
        "trust-remote-code": True,
        "max-num-seqs": 32,
        "gpu-memory-utilization": 0.90,
    },
}

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
}

# Doc-type ordering used everywhere downstream (artifact keys + plotting).
DOC_TYPES = ["receipt", "article", "table", "formula", "dense", "handwritten"]


# ------------------------------ data model ------------------------------
@dataclass
class CorpusItem:
    doc_type: str
    path: Path
    base64_cached: str  # "data:image/png;base64,..."
    metadata: dict = field(default_factory=dict)


# ------------------------------ helpers ---------------------------------
def load_image_data_url(asset_path: Path) -> str:
    raw = asset_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def load_corpus(assets_dir: Path) -> list[CorpusItem]:
    """Load the 6-doc stratified corpus. Raises if any file is missing."""
    items: list[CorpusItem] = []
    for dt in DOC_TYPES:
        p = assets_dir / f"{dt}.png"
        if not p.is_file():
            raise FileNotFoundError(f"corpus image missing: {p}")
        items.append(
            CorpusItem(
                doc_type=dt,
                path=p,
                base64_cached=load_image_data_url(p),
                metadata={
                    "file_name": p.name,
                    "file_size_bytes": p.stat().st_size,
                },
            )
        )
    return items


def build_request_body(data_url: str, prompt: str = PROMPT, max_tokens: int = 256) -> dict:
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
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


def compute_equivalent_pages(
    input_tokens: float,
    output_tokens: float,
    std_input: float = 1200.0,
    std_output: float = 300.0,
) -> float:
    """Geometric-mean page-equivalence normalizer.

    One "equivalent page" is defined by (std_input, std_output). A request
    that has 2x the input tokens and 2x the output tokens of the standard
    counts as sqrt(2*2) = 2 pages. A request with 4x input and 1x output
    counts as sqrt(4*1) = 2 pages. Geometric mean is chosen because prefill
    and decode cost roughly multiply under continuous batching (both have
    to happen for the request to complete), so geometric mean is closer to
    "equivalent compute" than arithmetic mean.

    Returns 0.0 for degenerate inputs.
    """
    if input_tokens <= 0 or output_tokens <= 0 or std_input <= 0 or std_output <= 0:
        return 0.0
    return math.sqrt((input_tokens / std_input) * (output_tokens / std_output))


def envelope() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": str(uuid.uuid4()),
        "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_tool": {
            "name": SOURCE_TOOL_NAME,
            "version": "0.1.0-deepseek-ocr",
            "enrichment_version": ENRICHMENT_VERSION,
        },
    }


NULL_LATENCY = {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None}


def write_artifact(out_path: Path, doc: dict, per_doc_type: dict | None = None) -> Path:
    """Write the artifact JSON.

    If `per_doc_type` is provided, it is embedded at
    `extensions.stratification.per_doc_type[]` (list form, keyed by doc_type).
    The schema's `extensions` block is an open object so this is compliant.
    The top-level `metrics` block is untouched; the schema has no
    `image_toks_per_s` core metric, so that value is surfaced in
    `extensions.image_toks_per_s` instead (callers should set it directly
    on the `doc` before calling write_artifact).
    """
    # Schema requires ttft_ms / tpot_ms / itl_ms even for non-streaming VLM.
    # Fill with null percentiles when absent.
    metrics = doc.setdefault("metrics", {})
    for k in ("ttft_ms", "tpot_ms", "itl_ms"):
        metrics.setdefault(k, dict(NULL_LATENCY))
    if per_doc_type is not None:
        doc.setdefault("extensions", {}).setdefault("stratification", {})
        items = []
        for dt in DOC_TYPES:
            if dt in per_doc_type:
                entry = {"doc_type": dt}
                entry.update(per_doc_type[dt])
                items.append(entry)
        doc["extensions"]["stratification"]["per_doc_type"] = items
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    return out_path


def summarize_per_doc_type(buckets: dict[str, list[dict]]) -> dict[str, dict]:
    """Given a map of doc_type -> list of per-request records with keys
    {e2e_ms, completion_tokens, prompt_tokens, success}, return a map of
    doc_type -> summary stats dict suitable for embedding in
    extensions.stratification.per_doc_type[].
    """
    out: dict[str, dict] = {}
    for dt, recs in buckets.items():
        ok = [r for r in recs if r.get("success")]
        failed = len(recs) - len(ok)
        if not ok:
            out[dt] = {
                "completed": 0,
                "failed": failed,
                "image_tokens_p50": 0,
                "output_tokens_p50": 0,
                "e2e_ms_p50": 0.0,
                "e2e_ms_p99": 0.0,
                "request_throughput": 0.0,
                "output_toks_per_s": 0.0,
                "image_toks_per_s": 0.0,
                "equivalent_pages_per_s": 0.0,
            }
            continue
        e2e = sorted(r["e2e_ms"] for r in ok)
        comp = sorted(r["completion_tokens"] for r in ok)
        prompt = sorted(r["prompt_tokens"] for r in ok)
        def pct(xs, p):
            k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
            return xs[k]
        # bucket throughput = #ok / span_of_bucket_requests
        # We compute against total sum of e2e latencies divided by concurrency
        # isn't right either; simpler: bucket req/s = len(ok) / (total bucket
        # wall time). We don't track wall time per bucket, so derive from the
        # per-bucket sum of latencies / mean concurrency observed. To keep it
        # simple and honest, we use: bucket_req_throughput is computed by the
        # caller who knows the overall duration & bucket share. We only emit
        # tokens-per-request and latency percentiles here plus totals; the
        # caller divides by duration.
        total_comp = sum(r["completion_tokens"] for r in ok)
        total_prompt = sum(r["prompt_tokens"] for r in ok)
        out[dt] = {
            "completed": len(ok),
            "failed": failed,
            "image_tokens_p50": pct(prompt, 50),
            "output_tokens_p50": pct(comp, 50),
            "e2e_ms_p50": pct(e2e, 50),
            "e2e_ms_p99": pct(e2e, 99),
            "total_input_tokens": total_prompt,
            "total_output_tokens": total_comp,
            "mean_input_tokens": statistics.fmean(prompt),
            "mean_output_tokens": statistics.fmean(comp),
        }
    return out


def attach_throughput(
    per_doc_summary: dict[str, dict],
    duration_s: float,
    std_input: float = 1200.0,
    std_output: float = 300.0,
) -> dict[str, dict]:
    """Given the per_doc_summary from summarize_per_doc_type() plus the wall
    duration, compute request_throughput, output_toks_per_s, image_toks_per_s,
    equivalent_pages_per_s for each bucket. Mutates and returns the same dict.
    """
    if duration_s <= 0:
        return per_doc_summary
    for dt, s in per_doc_summary.items():
        if s["completed"] == 0:
            continue
        s["request_throughput"] = s["completed"] / duration_s
        s["output_toks_per_s"] = s["total_output_tokens"] / duration_s
        s["image_toks_per_s"] = s["total_input_tokens"] / duration_s
        # equivalent pages per second uses mean per-request input/output
        eq_per_req = compute_equivalent_pages(
            s["mean_input_tokens"], s["mean_output_tokens"], std_input, std_output
        )
        s["equivalent_pages_per_s"] = eq_per_req * s["request_throughput"]
    return per_doc_summary
