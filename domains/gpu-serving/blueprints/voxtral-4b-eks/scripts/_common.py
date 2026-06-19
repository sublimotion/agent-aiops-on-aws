"""Shared helpers for Voxtral-Mini-3B Stage 6 benchmark runners.

Audio (transcription) conventions:
  - api.type = "transcription", streaming = false, endpoint = "/v1/audio/transcriptions"
  - modality = "audio"
  - workload payload is multipart/form-data (NOT JSON chat)
  - usage tokens are typically not returned by the transcription endpoint, so
    `output_toks_per_s` is derived from word count of the response text divided
    by wall time. We surface the raw word count in extensions.audio.* as well.

Audio extensions block (5 metrics from research):
  - rtfx_p50, rtfx_p99       : audio_duration / wall_duration   (higher is better)
  - ttfw_ms                   : Time-To-First-Word; null for non-streaming runs
  - audio_seconds_processed   : aggregate audio seconds across requests
  - audio_minutes_per_dollar  : derived from rtfx and on-demand price/hr

Substrate caveat: this run is on g6e.2xlarge (L40S 48GB), spec primary is
g6.xlarge (L4 24GB). All artifacts MUST set extensions.substrate_caveat.
"""
from __future__ import annotations

import datetime
import statistics
import uuid
from dataclasses import dataclass
from pathlib import Path

ENDPOINT = "http://localhost:8000"
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
SCHEMA_VERSION = "1.0.0"
ENRICHMENT_VERSION = "1.0.0"
SOURCE_TOOL_NAME = "custom"

# On-demand price for g6e.2xlarge in us-east-2 (sidecar value)
ON_DEMAND_PRICE_PER_HR = 2.24

MODEL_BLOCK = {
    "name": "Voxtral-Mini-3B",
    "id": MODEL_ID,
    "architecture": "VoxtralForConditionalGeneration",
    "parameters_total": "3B",
    "quantization": "bf16",
    "max_model_len": 32768,
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
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "trust-remote-code": True,
        "max-num-seqs": 16,
        "gpu-memory-utilization": 0.85,
    },
}

INFRA_BLOCK = {
    "substrate": "eks",
    "instance_type": "g6e.2xlarge",
    "substrate_deviation": (
        "spec-alt: L40S 48GB instead of spec-preferred L4 24GB. "
        "Per-stream perf is upper bound; cost-row $/audio-min reflects "
        "L40S on-demand price ($2.24/hr) — L4 (~$0.80/hr) would be "
        "~2.8x more cost-efficient if the model is L4-fittable (it is)."
    ),
    "region": "us-east-2",
    "gpu": {
        "name": "L40S",
        "arch": "sm_89",
        "count": 1,
        "vram_gb": 48,
        "interconnect": "none",
    },
}


# ------------------------------ data model ------------------------------
@dataclass
class AudioItem:
    bucket: str          # "short-3s" / "medium-10s" / "long-30s"
    duration_s: float
    path: Path
    size_bytes: int


def load_audio_corpus(assets_dir: Path) -> list[AudioItem]:
    """Load 3 chirp WAVs (3s/10s/30s)."""
    items: list[AudioItem] = []
    for bucket, dur in (("short-3s", 3.0), ("medium-10s", 10.0), ("long-30s", 30.0)):
        p = assets_dir / f"{bucket}.wav"
        if not p.is_file():
            raise FileNotFoundError(f"audio asset missing: {p}")
        items.append(
            AudioItem(
                bucket=bucket,
                duration_s=dur,
                path=p,
                size_bytes=p.stat().st_size,
            )
        )
    return items


def compute_percentiles(xs: list[float]) -> dict:
    if not xs:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(xs)
    def pct(p: float) -> float:
        k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
        return s[k]
    return {
        "mean": statistics.fmean(s),
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
            "version": "0.1.0-voxtral",
            "enrichment_version": ENRICHMENT_VERSION,
        },
    }


NULL_LATENCY = {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None}


def write_artifact(out_path: Path, doc: dict) -> Path:
    """Write the artifact JSON. Schema requires ttft/tpot/itl percentile keys
    even for non-streaming non-token workloads — fill with null."""
    import json
    metrics = doc.setdefault("metrics", {})
    for k in ("ttft_ms", "tpot_ms", "itl_ms"):
        metrics.setdefault(k, dict(NULL_LATENCY))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    return out_path


def audio_minutes_per_dollar(rtfx: float, price_per_hr: float = ON_DEMAND_PRICE_PER_HR) -> float:
    """rtfx = audio_seconds processed per wall_second.
    Audio minutes per dollar = (rtfx * 3600s/hr) / price_per_hr / 60s/min
                              = rtfx * 60 / price_per_hr.
    """
    if price_per_hr <= 0:
        return 0.0
    return (rtfx * 60.0) / price_per_hr
