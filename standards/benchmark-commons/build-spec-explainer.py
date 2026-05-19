#!/usr/bin/env python3
"""build-spec-explainer.py — Generate standard-explainer.html from the schema + workload cards + hero examples.

Self-contained output: everything inlined so the page works from file://.
Run this whenever the schema, workload cards, or hero examples change.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA = ROOT / "container" / "schema" / "enriched-artifact.json"
WORKLOADS_DIR = ROOT / "workloads"
OUT = ROOT / "standard-explainer.html"

# Hero examples (3 — one per model shape)
REPO_ROOT = ROOT.parent.parent
HEROES = [
    {
        "label": "Qwen3-Embedding-8B",
        "short": "Embedding model — native v1 emitter",
        "description": (
            "Reference producer. Written natively as v1 envelope — no enrichment step. "
            "Demonstrates embedding-workload shape: <code>output_toks_per_s = 0</code> is valid "
            "(vectors don't have tokens); <code>request_throughput</code> is the meaningful metric."
        ),
        "path": REPO_ROOT / "domains/gpu-serving/blueprints/qwen3-embedding-8b-hyperpod/results/artifacts/qwen3-embedding-8b_hyperpod-eks_g5-4xl_vllm_concurrency-sweep_20260513T000000Z.json",
        "badges": ["native v1", "embedding", "small GPU (A10G)"],
    },
    {
        "label": "Kimi K2.6 EAGLE3 winner",
        "short": "Speculative decode — `extensions.speculative_decode_stats`",
        "description": (
            "Shows how the <code>speculative_decode</code> engine sub-block and the "
            "<code>speculative_decode_stats</code> extension work together. Caveat: "
            "<code>ttft_ms</code> is <code>null</code> — the bench driver didn't scrape Prometheus. "
            "Every v1 field is populated honestly; missing data is marked missing rather than invented."
        ),
        "path": REPO_ROOT / "domains/gpu-serving/blueprints/kimi-k2.6-speculative/results/standard/kimi-k2.6_ec2-spot_p6-b300_sglang-eagle3-s4d4k1-hicache200_concurrency-sweep_c128.json",
        "badges": ["enriched", "EAGLE3 spec decode", "B300", "TTFT missing — honest gap"],
    },
    {
        "label": "Nemotron-3-Super disagg",
        "short": "Disaggregated P/D via Dynamo — `framework` block",
        "description": (
            "Demonstrates the <code>framework</code> block (Dynamo with 4 prefill + 4 decode workers), "
            "SLO evaluation against real thresholds, and cost derivation. Enriched from legacy custbench flat format."
        ),
        "path": REPO_ROOT / "domains/gpu-serving/blueprints/nemotron-super/results/standard/nemotron-3-super-120b-a12b_eks_p6-b200_sglang-disagg-4p4d_concurrency-sweep_c1.json",
        "badges": ["enriched", "disagg P/D", "Dynamo framework", "B200"],
    },
]

# Annotation hints — key path → plain-English explanation shown on hover
ANNOTATIONS = {
    "schema_version": "MAJOR.MINOR.PATCH. v1.x envelopes are backwards-compatible; v2.x requires producer update.",
    "artifact_id": "UUIDv4. MUST be unique per run. Consumers use this as a primary key.",
    "created_at": "ISO 8601 UTC. Time the benchmark completed (not when the file was written).",
    "source_tool.name": "Producer identity. Known values: aiperf, vllm-bench-serve, sglang-bench-serving, custom, bench-standard.py.",
    "source_tool.enrichment_version": "Version of the enrichment layer that hoisted source-tool output into the v1 envelope.",
    "model.id": "HuggingFace-style id (org/model). Used by consumers to match across sessions.",
    "model.architecture": "Free-form architecture tag. Common values: dense, moe, mla-moe, hybrid-mamba-moe, eagle3.",
    "model.parameters_active": "Params touched per forward pass. For MoE = active experts only. For dense = equals parameters_total.",
    "model.quantization": "fp16, bf16, fp8, int8, int4, awq, gptq, etc. Affects HBM footprint.",
    "engine.tensor_parallel": "TP size. For disagg, applies within each prefill/decode group.",
    "engine.speculative_decode": "null if not used. Non-null object with algorithm/draft/num_steps/etc.",
    "engine.kv_cache_dtype": "fp16 / fp8 / int8. Halves KV memory per step at fp8.",
    "engine.extra_args": "Free-form key/value from the engine CLI. Always include parser flags.",
    "framework.name": "Orchestration layer above engines. null for single-engine serving. dynamo/ray-serve/llm-d/etc for disagg or fleet.",
    "framework.config.mode": "aggregated = one engine handles prefill+decode. disaggregated = separate prefill/decode workers.",
    "infrastructure.substrate": "eks / hyperpod / ec2-spot / local. Dictates the security + scaling story.",
    "infrastructure.gpu.arch": "sm_XX compute capability. sm_80=A100, sm_90=H100/H200, sm_100=B200, sm_103=B300.",
    "infrastructure.gpu.interconnect": "nvswitch vs pcie. Affects NCCL collective BW ceiling by 10-20×.",
    "workload.catalog_id": "References workloads/*.yaml. If null, an ad-hoc workload — describe in use_case + dataset.",
    "workload.load.type": "constant (fixed rate) / concurrency (fixed in-flight) / concurrency-sweep (multi-level).",
    "workload.load.warmup_requests": "Not included in metrics. MUST populate engine caches before measurement.",
    "metrics.ttft_ms": "Time from request submission to first output token. Critical for streaming UX. Captured from engine histogram.",
    "metrics.tpot_ms": "Time per output token (inter-token latency after first). Capturing point: decode-path only.",
    "metrics.itl_ms": "Inter-token latency — superset of TPOT. For streaming APIs, approximately equals TPOT.",
    "metrics.e2e_ms": "End-to-end request latency. Includes queue + prefill + decode. SLO target field.",
    "metrics.output_toks_per_s": "Aggregate output tokens generated per second across all in-flight requests. 0 is VALID for embedding/classifier workloads.",
    "metrics.request_throughput": "Requests completed per second. Primary metric for embedding/classifier workloads.",
    "metrics.total_toks_per_s": "Input + output tokens per second. Used for cost-per-token derivation.",
    "metrics.error_rate": "failed / (completed + failed). MUST be 0 for a valid SLO pass.",
    "slo.targets": "Thresholds set by the customer or spec. SHOULD come from the workload card's SLO block.",
    "slo.results.ttft_p99_ms.pass": "true only if actual ≤ target AND actual is not null.",
    "slo.overall_pass": "true only if every required SLO is pass=true. A null actual DOES NOT pass.",
    "extensions.gpu_telemetry": "DCGM-derived. GPU util, HBM BW util, tensor-core util, power, temp. Required for roofline analysis.",
    "extensions.cache_stats": "Engine /metrics: kv_utilization_pct, prefix_hit_rate, preemption_count.",
    "extensions.speculative_decode_stats": "accept_rate × accept_length = effective tokens per decode step. 2-6 is typical for good drafts.",
    "extensions.cost": "Instance $/hr and derived $/M output tokens. formula string documents the calc.",
    "extensions.reconciliation": "Client-counted requests vs engine-counted. reconciled=false flags silent failures (requests client never saw).",
}


def load_workload_cards():
    cards = []
    for yf in sorted(WORKLOADS_DIR.glob("*.yaml")):
        text = yf.read_text()

        # Strip inline comments (# ...) but not the ones inside multiline description blocks
        # For field-extraction purposes we just strip trailing "# ..." from lines.
        def clean_line(line):
            # Remove trailing comment preserving inline strings (good enough for these cards)
            return re.sub(r"\s*#.*$", "", line).rstrip()

        lines = [clean_line(ln) for ln in text.splitlines()]

        # Parse indentation-based sections into shallow maps
        def section_range(name):
            """Return (start_idx, end_idx) inclusive of the YAML top-level block `name`."""
            start = None
            for i, ln in enumerate(lines):
                if re.match(rf"^{re.escape(name)}:\s*$", ln) or re.match(rf"^{re.escape(name)}:\s+\S", ln):
                    start = i
                    break
            if start is None:
                return None
            end = len(lines) - 1
            for j in range(start + 1, len(lines)):
                if lines[j] and not lines[j].startswith((" ", "\t")) and not lines[j].startswith("#"):
                    end = j - 1
                    break
            return (start, end)

        def extract_value_from_block(block_lines, key):
            """Find `  key: ...` inside the block, return the raw RHS after `:`."""
            for ln in block_lines:
                m = re.match(rf"^\s+{re.escape(key)}:\s*(.*)$", ln)
                if m:
                    return m.group(1).strip()
            return None

        def extract_mean(rhs):
            """Given a RHS like `{mean: 2048, std_dev: 0}` or `2048`, return the scalar (string)."""
            if rhs is None:
                return None
            rhs = rhs.strip()
            if not rhs:
                return None
            if rhs.startswith("{"):
                m = re.search(r"mean:\s*([0-9.]+)", rhs)
                return m.group(1) if m else None
            if rhs.startswith("["):
                m = re.search(r"\[\s*([0-9.]+)", rhs)
                return m.group(1) if m else None
            # Scalar — strip trailing comma or quotes
            return rhs.strip(",").strip("\"'")

        # Top-level scalars
        def top_value(key):
            for ln in lines:
                m = re.match(rf"^{re.escape(key)}:\s*(.*)$", ln)
                if m:
                    v = m.group(1).strip().strip("\"'")
                    return v if v else None
            return None

        # Multiline description block (`description: >`)
        desc_m = re.search(r"^description:\s*>\s*\n((?:[ \t]+.*\n)+)", text, re.MULTILINE)
        description = ""
        if desc_m:
            dlines = [ln.strip() for ln in desc_m.group(1).splitlines()]
            description = " ".join(dlines).strip()[:240]

        # Extract dataset / load / slo sections
        dataset_range = section_range("dataset")
        load_range = section_range("load")
        slo_range = section_range("slo")

        dataset_lines = lines[dataset_range[0]:dataset_range[1] + 1] if dataset_range else []
        load_lines = lines[load_range[0]:load_range[1] + 1] if load_range else []
        slo_lines = lines[slo_range[0]:slo_range[1] + 1] if slo_range else []

        dataset_type = extract_mean(extract_value_from_block(dataset_lines, "type"))

        isl_mean = (
            extract_mean(extract_value_from_block(dataset_lines, "input_tokens"))
            or extract_mean(extract_value_from_block(dataset_lines, "input_tokens_first_turn"))
            or extract_mean(extract_value_from_block(dataset_lines, "shared_prefix_tokens"))
            or extract_mean(extract_value_from_block(dataset_lines, "system_prompt_tokens"))
            or extract_mean(extract_value_from_block(dataset_lines, "context_lengths"))
        )

        osl_mean = extract_mean(extract_value_from_block(dataset_lines, "output_tokens"))

        load_type = extract_mean(extract_value_from_block(load_lines, "type"))
        request_rate = extract_mean(extract_value_from_block(load_lines, "request_rate"))
        concurrency = (
            extract_mean(extract_value_from_block(load_lines, "max_concurrency"))
            or extract_mean(extract_value_from_block(load_lines, "concurrent_sessions"))
            or extract_mean(extract_value_from_block(load_lines, "concurrency"))
        )
        num_prompts = (
            extract_mean(extract_value_from_block(load_lines, "num_prompts"))
            or extract_mean(extract_value_from_block(load_lines, "num_sessions"))
        )

        ttft_p99 = (
            extract_mean(extract_value_from_block(slo_lines, "ttft_p99_ms"))
            or extract_mean(extract_value_from_block(slo_lines, "ttft_warm_p99_ms"))
            or extract_mean(extract_value_from_block(slo_lines, "ttft_cold_p99_ms"))
        )
        tpot_p99 = extract_mean(extract_value_from_block(slo_lines, "tpot_p99_ms"))
        e2e_p99 = extract_mean(extract_value_from_block(slo_lines, "e2e_p99_ms"))

        cards.append({
            "catalog_id": top_value("catalog_id") or yf.stem,
            "use_case": top_value("use_case") or "",
            "modality": top_value("modality") or "text",
            "description": description or "No description",
            "dataset_type": dataset_type,
            "isl": isl_mean,
            "osl": osl_mean,
            "load_type": load_type,
            "request_rate": request_rate,
            "concurrency": concurrency,
            "num_prompts": num_prompts,
            "ttft_p99_ms": ttft_p99,
            "tpot_p99_ms": tpot_p99,
            "e2e_p99_ms": e2e_p99,
            "file": yf.name,
        })
    return cards


def jsonify(obj):
    return json.dumps(obj, ensure_ascii=False)


def main():
    schema = json.loads(SCHEMA.read_text())
    heroes = []
    for h in HEROES:
        try:
            heroes.append({**h, "json": json.loads(h["path"].read_text())})
        except Exception as e:
            print(f"WARN: could not load hero {h['label']}: {e}")
    workloads = load_workload_cards()

    # Build page
    payload = {
        "schema": schema,
        "heroes": [{k: v for k, v in h.items() if k != "path"} for h in heroes],
        "annotations": ANNOTATIONS,
        "workloads": workloads,
    }
    inline = jsonify(payload)
    html = TEMPLATE.replace("__PAYLOAD__", inline)
    OUT.write_text(html)
    print(f"Wrote {OUT} ({len(html)//1024} KB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Benchmark Artifact — Spec, Schema & Hero Examples</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f1117;
      --surface: #1a1d2e;
      --surface-2: #242740;
      --border: #2e3151;
      --text: #e2e8f0;
      --text-muted: #8892a4;
      --accent: #6366f1;
      --accent-soft: rgba(99,102,241,0.15);
      --green: #22c55e;
      --green-soft: rgba(34,197,94,0.12);
      --red: #ef4444;
      --red-soft: rgba(239,68,68,0.12);
      --yellow: #f59e0b;
      --yellow-soft: rgba(245,158,11,0.12);
      --blue: #3b82f6;
      --blue-soft: rgba(59,130,246,0.12);
      --purple: #8b5cf6;
      --json-string: #a5f3a0;
      --json-number: #fbbf24;
      --json-bool: #c084fc;
      --json-null: #71717a;
      --json-key: #60a5fa;
    }
    [data-theme="light"] {
      --bg: #f8fafc; --surface: #ffffff; --surface-2: #f1f5f9;
      --border: #e2e8f0; --text: #0f172a; --text-muted: #64748b;
      --accent-soft: rgba(99,102,241,0.08);
      --json-string: #15803d; --json-number: #b45309; --json-bool: #7c3aed;
      --json-null: #94a3b8; --json-key: #2563eb;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); line-height: 1.55; }
    header {
      background: var(--surface); border-bottom: 1px solid var(--border);
      padding: 1.25rem 2rem; display: flex; align-items: center; justify-content: space-between;
      position: sticky; top: 0; z-index: 100;
    }
    header h1 { font-size: 1.1rem; font-weight: 600; }
    header .meta { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.2rem; }
    .theme-toggle {
      background: var(--surface-2); border: 1px solid var(--border); color: var(--text);
      padding: 0.4rem 0.9rem; border-radius: 6px; cursor: pointer;
      font-size: 0.8rem; font-family: inherit;
    }
    main { max-width: 1280px; margin: 0 auto; padding: 2rem; }

    /* Hero banner */
    .hero-banner {
      background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
      padding: 1.5rem 1.75rem; margin-bottom: 2rem;
    }
    .hero-banner h2 { font-size: 1.2rem; margin-bottom: 0.35rem; }
    .hero-banner p { font-size: 0.88rem; color: var(--text-muted); }
    .hero-banner .pills { display: flex; gap: 0.5rem; margin-top: 0.75rem; flex-wrap: wrap; }
    .pill {
      background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px;
      padding: 0.28rem 0.6rem; font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;
      color: var(--text-muted);
    }
    .pill.accent { background: var(--accent-soft); color: var(--accent); border-color: var(--accent); }
    .code {
      background: var(--surface-2); padding: 0.6rem 0.8rem; border-radius: 6px;
      font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; margin-top: 0.75rem;
      overflow-x: auto; border: 1px solid var(--border);
    }

    /* Envelope tree */
    .envelope-diagram {
      background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
      padding: 1.5rem 1.75rem; margin-bottom: 2rem;
    }
    .envelope-diagram h2 { font-size: 1rem; margin-bottom: 0.35rem; }
    .env-subtitle { font-size: 0.82rem; color: var(--text-muted); margin-bottom: 1.5rem; }
    .env-tree {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 0.75rem;
    }
    .env-block {
      background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
      padding: 0.85rem 1rem; cursor: pointer; transition: border-color 0.15s;
    }
    .env-block:hover { border-color: var(--accent); }
    .env-block .block-name {
      font-family: 'JetBrains Mono', monospace; font-weight: 600;
      font-size: 0.85rem; color: var(--json-key); margin-bottom: 0.2rem;
    }
    .env-block .block-required {
      font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--red); margin-left: 0.4rem; font-weight: 600;
    }
    .env-block .block-optional {
      font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--text-muted); margin-left: 0.4rem;
    }
    .env-block .block-desc { font-size: 0.78rem; color: var(--text-muted); }
    .env-block .block-fields {
      margin-top: 0.4rem; display: flex; flex-wrap: wrap; gap: 3px;
    }
    .env-block .block-fields .f {
      font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
      color: var(--text); background: var(--bg); padding: 1px 6px; border-radius: 3px;
    }

    /* Hero example carousel */
    .tabs { display: flex; gap: 2px; background: var(--surface-2); border-radius: 8px 8px 0 0;
      padding: 0.4rem 0.4rem 0; border: 1px solid var(--border); border-bottom: none;
    }
    .tab {
      background: transparent; color: var(--text-muted); border: none; padding: 0.55rem 0.95rem;
      cursor: pointer; font-family: inherit; font-size: 0.82rem; font-weight: 500;
      border-radius: 6px 6px 0 0; border-bottom: 2px solid transparent;
    }
    .tab:hover { color: var(--text); }
    .tab.active {
      color: var(--accent); background: var(--surface);
      border-bottom-color: var(--accent);
    }
    .hero-panel {
      background: var(--surface); border: 1px solid var(--border); border-radius: 0 0 12px 12px;
      padding: 1.25rem 1.5rem; margin-bottom: 2rem;
    }
    .hero-panel h3 { font-size: 0.95rem; margin-bottom: 0.3rem; }
    .hero-panel .hero-desc { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.75rem; }
    .hero-panel .hero-desc code {
      font-family: 'JetBrains Mono', monospace; background: var(--surface-2);
      padding: 0 4px; border-radius: 3px; color: var(--json-key);
    }

    /* JSON render with hover annotations */
    .json-view {
      font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; line-height: 1.55;
      background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
      padding: 1rem 1.25rem; overflow-x: auto; max-height: 620px; overflow-y: auto;
      position: relative;
    }
    .json-view .tok-str { color: var(--json-string); }
    .json-view .tok-num { color: var(--json-number); }
    .json-view .tok-bool { color: var(--json-bool); }
    .json-view .tok-null { color: var(--json-null); font-style: italic; }
    .json-view .tok-key { color: var(--json-key); }
    .json-view .tok-path {
      cursor: help; border-bottom: 1px dotted currentColor; transition: background 0.1s;
    }
    .json-view .tok-path:hover { background: var(--accent-soft); }
    .json-view .toggle {
      cursor: pointer; user-select: none; color: var(--text-muted); display: inline-block;
      width: 1em; text-align: center;
    }
    .json-view .collapsed > .children { display: none; }
    .json-view .collapsed > .toggle::before { content: '▶'; }
    .json-view .toggle::before { content: '▼'; }

    /* Tooltip */
    .tooltip {
      position: fixed; background: var(--surface-2); border: 1px solid var(--accent);
      border-radius: 6px; padding: 0.6rem 0.8rem; font-family: 'Inter', sans-serif;
      font-size: 0.82rem; color: var(--text); max-width: 360px; z-index: 200;
      pointer-events: none; box-shadow: 0 8px 24px rgba(0,0,0,0.3);
      display: none;
    }
    .tooltip.visible { display: block; }
    .tooltip .tt-path {
      font-family: 'JetBrains Mono', monospace; font-size: 0.74rem;
      color: var(--accent); margin-bottom: 0.3rem; font-weight: 600;
    }

    /* Schema explorer */
    .schema-tree { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; line-height: 1.65; }
    .schema-node { padding-left: 1.2rem; border-left: 1px solid var(--border); margin-left: 0.25rem; }
    .schema-node.root { padding-left: 0; border-left: none; margin-left: 0; }
    .schema-name { color: var(--json-key); font-weight: 600; cursor: pointer; user-select: none; }
    .schema-name:hover { text-decoration: underline; }
    .schema-type { color: var(--text-muted); font-size: 0.74rem; margin-left: 0.4rem; }
    .schema-req { color: var(--red); font-size: 0.64rem; margin-left: 0.35rem; font-weight: 600; text-transform: uppercase; }
    .schema-enum { color: var(--json-string); font-size: 0.74rem; margin-left: 0.35rem; }
    .schema-desc { color: var(--text-muted); font-size: 0.76rem; font-family: 'Inter', sans-serif; margin-left: 1.2rem; margin-bottom: 0.35rem; }
    .collapsed > .schema-children { display: none; }
    .schema-caret { color: var(--text-muted); margin-right: 0.25rem; }

    /* Workload cards grid */
    .workload-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 0.75rem;
    }
    .wc {
      background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
      padding: 0.9rem 1rem;
    }
    .wc .wc-id {
      font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
      color: var(--accent); font-weight: 600; margin-bottom: 0.25rem;
    }
    .wc .wc-meta {
      font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
      color: var(--text-muted); margin-bottom: 0.5rem;
    }
    .wc .wc-desc { font-size: 0.78rem; color: var(--text-muted); margin-bottom: 0.65rem; }
    .wc .wc-spec {
      display: grid; grid-template-columns: repeat(2, 1fr); gap: 3px 10px;
      background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
      padding: 0.5rem 0.65rem; font-family: 'JetBrains Mono', monospace;
      font-size: 0.7rem;
    }
    .wc .wc-spec .sp-label {
      color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; font-size: 0.62rem;
    }
    .wc .wc-spec .sp-value { color: var(--text); }
    .wc .wc-spec .sp-value.null { color: var(--json-null); font-style: italic; }
    .wc .wc-slo {
      margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 4px;
    }
    .wc .wc-slo .slo-pill {
      background: var(--surface-2); border: 1px solid var(--border); border-radius: 4px;
      padding: 2px 7px; font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; color: var(--text-muted);
    }
    .wc .wc-slo .slo-pill strong { color: var(--accent); font-weight: 600; }

    /* Producer checklist */
    .checklist {
      background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
      padding: 1.25rem 1.5rem; margin-bottom: 2rem;
    }
    .checklist h2 { font-size: 1rem; margin-bottom: 0.75rem; }
    .checklist ul { list-style: none; padding: 0; }
    .checklist li {
      padding: 0.4rem 0; border-bottom: 1px solid var(--border);
      font-size: 0.85rem; display: flex; align-items: flex-start; gap: 0.5rem;
    }
    .checklist li:last-child { border-bottom: none; }
    .checklist li::before {
      content: '▸'; color: var(--accent); flex-shrink: 0; margin-top: 0.1rem;
    }
    .checklist li code {
      font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
      background: var(--surface-2); padding: 0 5px; border-radius: 3px; color: var(--json-key);
    }

    section { margin-bottom: 2rem; }
    section h2.section-title {
      font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem;
      padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
    }
    footer { text-align: center; padding: 2rem; color: var(--text-muted); font-size: 0.72rem; border-top: 1px solid var(--border); }
    a { color: var(--accent); }
  </style>
</head>
<body>

<header>
  <div>
    <h1>Benchmark Artifact — Spec, Schema &amp; Hero Examples</h1>
    <div class="meta">v1 envelope reference · single-page explainer · schema <span id="schema-id"></span></div>
  </div>
  <button class="theme-toggle" onclick="toggleTheme()">Toggle theme</button>
</header>

<main>
  <!-- Hero banner -->
  <div class="hero-banner">
    <h2>One format, every producer</h2>
    <p>
      The <strong>v1 benchmark-commons envelope</strong> is the shared output contract for LLM inference benchmarks in this repo.
      Producers (vLLM bench-serve, SGLang, AIPerf, GenAI-Perf, guidellm, LLMPerf, recon-perf, <code>bench-standard.py</code>) emit the same structure.
      Consumers (<code>compare.py</code>, the results-vault dashboard, benchmark-analyst agent, Grafana/Athena/CloudWatch) read one format.
      The envelope captures model identity, engine config, deployment context, workload card, full latency distributions, SLO evaluation,
      and optional extensions (GPU telemetry, cost, speculative-decode stats, reconciliation).
    </p>
    <p style="margin-top: 0.75rem; font-weight: 600; color: var(--accent);">Keep your tool. Standardize the output.</p>
    <div class="pills">
      <span class="pill accent">schema v1.1</span>
      <span class="pill">JSON Schema Draft 2020-12</span>
      <span class="pill">EKS · HyperPod · EC2 spot · local</span>
      <span class="pill">17 workload cards</span>
      <span class="pill">9 supported tools</span>
    </div>
    <div class="code"># Validate any artifact against the schema
python3 standards/benchmark-commons/container/validate-artifact.py path/to/artifact.json</div>
  </div>

  <!-- What is required -->
  <div class="envelope-diagram" style="margin-bottom: 2rem;">
    <h2>What's required vs optional</h2>
    <div class="env-subtitle">Top-level blocks and metric fields that every conformant artifact MUST contain.</div>
    <div class="summary-grid" style="margin-bottom: 0;">
      <div class="summary-card">
        <div class="label">Required top-level blocks</div>
        <div class="value" style="font-size: 0.92rem; color: var(--red);" id="req-top"></div>
        <div class="sub">Producer MUST populate all of these.</div>
      </div>
      <div class="summary-card">
        <div class="label">Required metric fields</div>
        <div class="value" style="font-size: 0.92rem; color: var(--red);" id="req-metrics"></div>
        <div class="sub">Latency percentiles MAY be <code>null</code> when not captured — do NOT invent values.</div>
      </div>
      <div class="summary-card">
        <div class="label">Optional blocks</div>
        <div class="value" style="font-size: 0.88rem; color: var(--text-muted);">framework · slo · power · quality · stability · cold_start · cost · extensions</div>
        <div class="sub">Populate what the producer can measure.</div>
      </div>
      <div class="summary-card">
        <div class="label">Contract rules</div>
        <div class="value" style="font-size: 0.88rem;">5 hard rules</div>
        <div class="sub">null &gt; invented · same shape for every latency · filename convention · semver on schema · catalog_id = comparability</div>
      </div>
    </div>
  </div>

  <!-- Envelope structure diagram -->
  <section>
    <h2 class="section-title">Envelope structure (click a block to scroll to schema detail)</h2>
    <div class="envelope-diagram">
      <h2>The shape at a glance</h2>
      <div class="env-subtitle">Top-level blocks. Required = field must be present. Optional = producer MAY omit.</div>
      <div class="env-tree" id="env-tree"></div>
    </div>
  </section>

  <!-- Hero examples carousel -->
  <section>
    <h2 class="section-title">Hero examples — three real artifacts</h2>
    <div class="tabs" id="hero-tabs"></div>
    <div class="hero-panel" id="hero-panel"></div>
  </section>

  <!-- Schema explorer -->
  <section>
    <h2 class="section-title">Schema explorer (click keys to expand/collapse)</h2>
    <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem 1.5rem;">
      <div id="schema-root" class="schema-tree"></div>
    </div>
  </section>

  <!-- Workload cards -->
  <section>
    <h2 class="section-title">Workload cards (<span id="wc-count"></span>)</h2>
    <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem 1.5rem;">
      <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
        Each artifact references one workload card via <code>workload.catalog_id</code>. Cards live in
        <code>standards/benchmark-commons/workloads/</code>. Multiple engines and hardware targets produce
        comparable results <em>only when they share a catalog_id</em>.
      </p>
      <div class="workload-grid" id="workload-grid"></div>
    </div>
  </section>

  <!-- Metric definitions table -->
  <section>
    <h2 class="section-title">Core metric contract — definitions</h2>
    <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;">
      <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.75rem;">
        Every latency metric uses the same shape: <code>{mean, p50, p90, p95, p99}</code>. All in <strong>milliseconds</strong>. Use <code>null</code> for percentiles the producer did not capture — do NOT invent values. <code>min</code>/<code>max</code>/<code>std</code> belong in <code>extensions.latency_detail</code>.
      </p>
      <table id="metric-table" style="width: 100%; border-collapse: collapse; font-size: 0.82rem;">
        <thead>
          <tr style="background: var(--surface-2);">
            <th style="padding: 0.5rem 0.75rem; text-align: left; font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);">Metric</th>
            <th style="padding: 0.5rem 0.75rem; text-align: left; font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);">Definition</th>
            <th style="padding: 0.5rem 0.75rem; text-align: left; font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted);">Unit</th>
          </tr>
        </thead>
        <tbody style="font-family: 'JetBrains Mono', monospace;">
          <tr><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); color: var(--json-key);">ttft_ms</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); font-family: Inter, sans-serif;">Time from request send to first token received.</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border);">ms</td></tr>
          <tr><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); color: var(--json-key);">tpot_ms</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); font-family: Inter, sans-serif;"><code>(e2e - ttft) / (output_tokens - 1)</code> — excludes first token.</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border);">ms</td></tr>
          <tr><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); color: var(--json-key);">itl_ms</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); font-family: Inter, sans-serif;">Mean time between consecutive token arrivals.</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border);">ms</td></tr>
          <tr><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); color: var(--json-key);">e2e_ms</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); font-family: Inter, sans-serif;">Total request latency from send to last token.</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border);">ms</td></tr>
          <tr><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); color: var(--json-key);">output_toks_per_s</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); font-family: Inter, sans-serif;"><code>total_output_tokens / duration_s</code>. Legitimately <code>0</code> for embedding/classifier workloads.</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border);">tok/s</td></tr>
          <tr><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); color: var(--json-key);">request_throughput</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); font-family: Inter, sans-serif;"><code>completed / duration_s</code>. Primary metric for embedding workloads.</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border);">req/s</td></tr>
          <tr><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); color: var(--json-key);">total_toks_per_s</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border); font-family: Inter, sans-serif;"><code>(total_input + total_output) / duration_s</code>.</td><td style="padding: 0.5rem 0.75rem; border-top: 1px solid var(--border);">tok/s</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- Producer checklist -->
  <section>
    <h2 class="section-title">Producer checklist — minimum viable v1 envelope</h2>
    <div class="checklist">
      <h2>If you're writing a producer or consumer, get these right first</h2>
      <ul>
        <li>Emit <code>schema_version: "1.0.0"</code> and a fresh UUIDv4 <code>artifact_id</code> per run.</li>
        <li>Populate <strong>all required top-level blocks</strong>: <code>source_tool</code>, <code>model</code>, <code>engine</code>, <code>infrastructure</code>, <code>workload</code>, <code>metrics</code>.</li>
        <li>Under <code>metrics</code>, <strong>every one</strong> of <code>ttft_ms</code>, <code>tpot_ms</code>, <code>itl_ms</code>, <code>e2e_ms</code> is required with <code>{mean, p50, p90, p95, p99}</code> — all ms.</li>
        <li>If a latency percentile is <strong>not measured</strong>, emit <code>null</code> rather than inventing a value. Consumers detect and honour <code>null</code>. Invented values corrupt cross-team comparisons.</li>
        <li>Populate <code>output_toks_per_s</code> <em>and</em> <code>request_throughput</code> — both are required. For embedding / classifier workloads, <code>output_toks_per_s</code> is legitimately <code>0</code>; use <code>request_throughput</code> as the primary metric.</li>
        <li>Set <code>workload.catalog_id</code> to a known card (see grid) — this is what makes artifacts comparable across teams. Use <code>catalog_id: null</code> only for custom workloads (see Appendix A in PROPOSAL.md).</li>
        <li>Compute <code>slo.results[*].pass</code> as <code>actual ≤ target</code> strictly. A <code>null</code> <code>actual</code> means <code>pass: null</code>, never <code>true</code>.</li>
        <li>Name output files as <code>{model}_{substrate}_{instance}_{engine}_{workload}_{timestamp}.json</code> — every field separated by underscores.</li>
        <li>Scrape engine <code>/metrics</code> to populate <code>extensions.gpu_telemetry</code> and <code>extensions.cache_stats</code>. Client-side timing alone loses TTFT, KV cache, and queue-wait info. <code>bench-standard.py</code> does this for you.</li>
        <li>Validate with <code>python3 container/validate-artifact.py artifact.json</code> before committing or publishing.</li>
      </ul>
    </div>
  </section>
</main>

<div class="tooltip" id="tooltip"></div>

<footer>
  Generated from <code>container/schema/enriched-artifact.json</code> + <code>workloads/*.yaml</code> + hero artifacts · regenerate with
  <code>python3 standards/benchmark-commons/build-spec-explainer.py</code>
</footer>

<script>
const DATA = __PAYLOAD__;
const ANN = DATA.annotations;
const SCHEMA = DATA.schema;

document.getElementById('schema-id').textContent = SCHEMA['$id'] || 'enriched-artifact.json';

// Populate required-field summary cards from the schema
(function populateRequired() {
  const top = (SCHEMA.required || []).filter(x => !['schema_version','artifact_id','created_at','source_tool'].includes(x));
  document.getElementById('req-top').textContent = ['schema_version','artifact_id','created_at','source_tool'].concat(top).join(' · ');
  const metricsReq = ((SCHEMA.properties && SCHEMA.properties.metrics && SCHEMA.properties.metrics.required) || []);
  document.getElementById('req-metrics').textContent = metricsReq.join(' · ');
})();

// ---------- Envelope tree ----------
(function renderEnvelope() {
  const topRequired = (SCHEMA.required || []);
  const metricsRequired = ((SCHEMA.properties && SCHEMA.properties.metrics && SCHEMA.properties.metrics.required) || []);
  const blocks = [
    ['schema_version', 'required', 'SemVer of the envelope. v1.x is backwards-compatible.', ['1.0.0']],
    ['artifact_id', 'required', 'UUIDv4, unique per run.', ['uuid']],
    ['created_at', 'required', 'ISO 8601 UTC when the run completed.', ['2026-05-13T...']],
    ['source_tool', 'required', 'Producer identity.', ['name', 'version', 'enrichment_version']],
    ['model', 'required', 'Model identity + architecture + quantization.', ['name', 'id', 'architecture', 'parameters_total', 'parameters_active', 'quantization', 'max_model_len']],
    ['engine', 'required', 'Engine config (TP/PP/DP, spec decode, extra args).', ['name', 'container_image', 'tensor_parallel', 'kv_cache_dtype', 'speculative_decode', 'extra_args']],
    ['framework', 'optional', 'Orchestration above engine (Dynamo, Ray Serve, llm-d).', ['name', 'version', 'config']],
    ['infrastructure', 'required', 'Substrate, instance, GPU type, interconnect.', ['substrate', 'instance_type', 'region', 'gpu.*']],
    ['workload', 'required', 'Catalog_id + dataset + load pattern.', ['use_case', 'catalog_id', 'dataset', 'load', 'api']],
    ['metrics', 'required', 'TTFT, TPOT, ITL, E2E all required (null percentiles OK). Throughput required.', metricsRequired.filter(k => !['duration_s','completed','failed','error_rate'].includes(k))],
    ['slo', 'optional', 'Targets + per-SLO pass/fail + overall_pass.', ['targets', 'results', 'overall_pass']],
    ['extensions', 'optional', 'Well-known keys: gpu_telemetry, cache_stats, speculative_decode_stats, cost, reconciliation, per_request, raw_tool_output.', ['gpu_telemetry', 'cache_stats', 'cost', 'speculative_decode_stats', 'reconciliation']],
  ];
  const el = document.getElementById('env-tree');
  el.innerHTML = blocks.map(([name, req, desc, fields]) => `
    <div class="env-block" onclick="scrollToSchema('${name}')">
      <div class="block-name">${name}<span class="${req === 'required' ? 'block-required' : 'block-optional'}">${req}</span></div>
      <div class="block-desc">${desc}</div>
      <div class="block-fields">${fields.map(f => `<span class="f">${f}</span>`).join('')}</div>
    </div>
  `).join('');
})();

// ---------- Hero tabs ----------
let activeHero = 0;
(function renderHeroTabs() {
  const tabs = document.getElementById('hero-tabs');
  tabs.innerHTML = DATA.heroes.map((h, i) => `
    <button class="tab ${i === 0 ? 'active' : ''}" data-idx="${i}">${h.label}</button>
  `).join('');
  tabs.querySelectorAll('.tab').forEach(t => {
    t.onclick = () => {
      activeHero = parseInt(t.dataset.idx, 10);
      tabs.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
      renderHeroPanel();
    };
  });
  renderHeroPanel();
})();

function renderHeroPanel() {
  const h = DATA.heroes[activeHero];
  const el = document.getElementById('hero-panel');
  el.innerHTML = `
    <h3>${h.label}</h3>
    <div class="hero-desc">${h.short}</div>
    <div class="hero-desc">${h.description}</div>
    <div class="pills" style="margin-bottom: 0.75rem;">
      ${h.badges.map(b => `<span class="pill">${b}</span>`).join('')}
    </div>
    <div class="json-view" id="json-view-${activeHero}"></div>
  `;
  renderJSON(document.getElementById(`json-view-${activeHero}`), h.json, '');
}

// ---------- JSON renderer with hover annotations ----------
function renderJSON(container, obj, path) {
  container.innerHTML = '';
  container.appendChild(jsonNode(obj, path, true));
}

function jsonNode(val, path, isRoot) {
  if (val === null) return span('tok-null', 'null', path);
  if (typeof val === 'boolean') return span('tok-bool', String(val), path);
  if (typeof val === 'number') return span('tok-num', String(val), path);
  if (typeof val === 'string') return span('tok-str', JSON.stringify(val), path);
  if (Array.isArray(val)) return arrayNode(val, path);
  if (typeof val === 'object') return objectNode(val, path, isRoot);
  return span('', String(val), path);
}

function span(cls, text, path) {
  const s = document.createElement('span');
  s.className = cls;
  s.textContent = text;
  return s;
}

function objectNode(obj, path, open) {
  const wrap = document.createElement('div');
  wrap.className = 'json-obj' + (open ? '' : ' collapsed');

  const toggle = document.createElement('span');
  toggle.className = 'toggle';
  toggle.onclick = () => wrap.classList.toggle('collapsed');
  wrap.appendChild(toggle);

  const openBrace = document.createElement('span');
  openBrace.textContent = '{';
  wrap.appendChild(openBrace);

  const children = document.createElement('div');
  children.className = 'children';
  children.style.paddingLeft = '1.2rem';

  const keys = Object.keys(obj);
  keys.forEach((k, i) => {
    const line = document.createElement('div');
    const childPath = path ? `${path}.${k}` : k;
    const key = document.createElement('span');
    key.className = 'tok-key';
    key.textContent = JSON.stringify(k);
    // Annotation hover target
    if (ANN[childPath]) {
      key.classList.add('tok-path');
      key.dataset.path = childPath;
      key.dataset.desc = ANN[childPath];
    }
    line.appendChild(key);
    line.appendChild(document.createTextNode(': '));
    line.appendChild(jsonNode(obj[k], childPath, false));
    if (i < keys.length - 1) line.appendChild(document.createTextNode(','));
    children.appendChild(line);
  });
  wrap.appendChild(children);

  const closeBrace = document.createElement('span');
  closeBrace.textContent = '}';
  wrap.appendChild(closeBrace);

  return wrap;
}

function arrayNode(arr, path) {
  const wrap = document.createElement('span');
  if (arr.length === 0) { wrap.textContent = '[]'; return wrap; }
  if (arr.length <= 8 && arr.every(x => x === null || ['string','number','boolean'].includes(typeof x))) {
    // inline short arrays
    wrap.appendChild(document.createTextNode('['));
    arr.forEach((v, i) => {
      wrap.appendChild(jsonNode(v, `${path}[${i}]`, false));
      if (i < arr.length - 1) wrap.appendChild(document.createTextNode(', '));
    });
    wrap.appendChild(document.createTextNode(']'));
    return wrap;
  }
  wrap.appendChild(document.createTextNode('['));
  arr.forEach((v, i) => {
    const line = document.createElement('div');
    line.style.paddingLeft = '1.2rem';
    line.appendChild(jsonNode(v, `${path}[${i}]`, false));
    if (i < arr.length - 1) line.appendChild(document.createTextNode(','));
    wrap.appendChild(line);
  });
  wrap.appendChild(document.createTextNode(']'));
  return wrap;
}

// Tooltip on hover over annotated keys
const tt = document.getElementById('tooltip');
document.addEventListener('mouseover', e => {
  const t = e.target.closest('.tok-path');
  if (!t) return;
  tt.innerHTML = `<div class="tt-path">${t.dataset.path}</div>${t.dataset.desc}`;
  tt.classList.add('visible');
});
document.addEventListener('mousemove', e => {
  if (!tt.classList.contains('visible')) return;
  const rect = tt.getBoundingClientRect();
  let x = e.clientX + 14;
  let y = e.clientY + 14;
  if (x + rect.width > window.innerWidth - 10) x = e.clientX - rect.width - 14;
  if (y + rect.height > window.innerHeight - 10) y = e.clientY - rect.height - 14;
  tt.style.left = x + 'px';
  tt.style.top = y + 'px';
});
document.addEventListener('mouseout', e => {
  if (e.target.closest('.tok-path')) tt.classList.remove('visible');
});

// ---------- Schema explorer ----------
(function renderSchema() {
  const root = document.getElementById('schema-root');
  root.appendChild(schemaNode('(root)', SCHEMA, SCHEMA.required || [], true));
})();

function schemaNode(name, node, parentRequired, isRoot) {
  const wrap = document.createElement('div');
  wrap.className = 'schema-node' + (isRoot ? ' root' : '');

  const header = document.createElement('div');
  const props = node.properties || {};
  const hasChildren = Object.keys(props).length > 0;

  if (hasChildren) {
    const caret = document.createElement('span');
    caret.className = 'schema-caret';
    caret.textContent = '▼';
    header.appendChild(caret);
    const nameEl = document.createElement('span');
    nameEl.className = 'schema-name';
    nameEl.textContent = name;
    nameEl.onclick = () => {
      wrap.classList.toggle('collapsed');
      caret.textContent = wrap.classList.contains('collapsed') ? '▶' : '▼';
    };
    header.appendChild(nameEl);
  } else {
    const nameEl = document.createElement('span');
    nameEl.className = 'schema-name';
    nameEl.textContent = name;
    header.appendChild(nameEl);
  }
  // Type
  const typeEl = document.createElement('span');
  typeEl.className = 'schema-type';
  let typeStr = node.type || (node.enum ? 'enum' : 'any');
  if (Array.isArray(typeStr)) typeStr = typeStr.join(' | ');
  if (node.format) typeStr += ` <${node.format}>`;
  typeEl.textContent = typeStr;
  header.appendChild(typeEl);

  // Required marker
  if (parentRequired.includes(name)) {
    const req = document.createElement('span');
    req.className = 'schema-req';
    req.textContent = 'REQ';
    header.appendChild(req);
  }

  // Enum
  if (node.enum) {
    const en = document.createElement('span');
    en.className = 'schema-enum';
    en.textContent = '= [' + node.enum.join(', ') + ']';
    header.appendChild(en);
  }

  wrap.appendChild(header);

  // Description (from schema if present, else from ANN if matches a known path)
  if (node.description) {
    const d = document.createElement('div');
    d.className = 'schema-desc';
    d.textContent = node.description;
    wrap.appendChild(d);
  }

  if (hasChildren) {
    const childrenWrap = document.createElement('div');
    childrenWrap.className = 'schema-children';
    const required = node.required || [];
    Object.entries(props).forEach(([k, v]) => {
      childrenWrap.appendChild(schemaNode(k, v, required, false));
    });
    wrap.appendChild(childrenWrap);
  }
  return wrap;
}

function scrollToSchema(topLevelKey) {
  // Find the schema node whose name matches
  const nodes = document.querySelectorAll('.schema-node');
  for (const n of nodes) {
    const name = n.querySelector('.schema-name');
    if (name && name.textContent === topLevelKey) {
      n.scrollIntoView({behavior: 'smooth', block: 'center'});
      name.style.background = 'var(--accent-soft)';
      setTimeout(() => name.style.background = '', 1200);
      return;
    }
  }
}

// ---------- Workload cards ----------
(function renderWorkloads() {
  const el = document.getElementById('workload-grid');
  const nullish = v => (v === null || v === undefined || v === '' || v === 'null')
    ? '<span class="sp-value null">—</span>'
    : `<span class="sp-value">${v}</span>`;
  const sloPill = (label, v) => v
    ? `<span class="slo-pill">${label} <strong>${v}ms</strong></span>`
    : '';
  el.innerHTML = DATA.workloads.map(w => `
    <div class="wc">
      <div class="wc-id">${w.catalog_id}</div>
      <div class="wc-meta">${w.use_case || '—'} · ${w.modality}</div>
      <div class="wc-desc">${w.description}</div>
      <div class="wc-spec">
        <span class="sp-label">ISL</span>${nullish(w.isl ? w.isl + ' tok' : null)}
        <span class="sp-label">OSL</span>${nullish(w.osl ? w.osl + ' tok' : null)}
        <span class="sp-label">Dataset</span>${nullish(w.dataset_type)}
        <span class="sp-label">Load</span>${nullish(w.load_type)}
        <span class="sp-label">Rate / Conc</span>${nullish(w.request_rate ? w.request_rate + ' QPS' : (w.concurrency ? 'c=' + w.concurrency : null))}
        <span class="sp-label">Prompts</span>${nullish(w.num_prompts)}
      </div>
      <div class="wc-slo">
        ${sloPill('TTFT p99', w.ttft_p99_ms)}
        ${sloPill('TPOT p99', w.tpot_p99_ms)}
        ${sloPill('E2E p99', w.e2e_p99_ms)}
      </div>
    </div>
  `).join('');
  document.getElementById('wc-count').textContent = DATA.workloads.length;
})();

// ---------- Theme ----------
function toggleTheme() {
  const html = document.documentElement;
  html.setAttribute('data-theme', html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
