#!/usr/bin/env python3
"""build-dashboard.py — Generate dashboard.html from index.json.

Inlines the manifest so the dashboard works via file:// without fetch.
Re-run after rebuild-index.py when new artifacts land.
"""
from __future__ import annotations
import json
from pathlib import Path

VAULT = Path(__file__).resolve().parent
INDEX = VAULT / "index.json"
OUT = VAULT / "dashboard.html"

# Relative path from this dashboard (in results-vault/) to docs/reports/.
REPORTS_REL = "../../../docs/reports"

# model_name -> detailed per-blueprint report filename in docs/reports/.
# Used for row drill-down. Models without a bespoke report are omitted (no link).
REPORT_MAP = {
    "DeepSeek-V4-Flash": "benchmark-visual-deepseek-v4-flash-20260519.html",
    "Kimi-K2.6": "benchmark-visual-kimi-k2.6-speculative.html",
    "Qwen3-235B-A22B-FP8": "benchmark-visual-qwen3-235b-speculative.html",
    "Nemotron-3-Super-120B-A12B": "benchmark-report-nemotron-super.html",
    "DeepSeek-OCR-2": "deepseek-ocr-2-eks-visual.html",
    "Qwen3-Reranker-4B": "qwen3-reranker-4b-eks-visual.html",
    "Qwen3-Embedding-8B": "qwen3-embedding-8b-hyperpod-visual.html",
    "Voxtral-Mini-3B": "voxtral-4b-eks-visual.html",
}


def main():
    index = json.loads(INDEX.read_text())
    # Minify by dropping None fields to keep payload small
    rows = []
    for r in index["artifacts"]:
        rows.append({k: v for k, v in r.items() if v is not None})
    payload = {
        "generated_at": index["generated_at"],
        "artifact_count": index["artifact_count"],
        "artifacts": rows,
        "reports_rel": REPORTS_REL,
        "report_map": REPORT_MAP,
    }
    inline = json.dumps(payload)

    html = DASHBOARD_TEMPLATE.replace("__INDEX_JSON__", inline)
    OUT.write_text(html)
    print(f"Wrote {OUT} ({len(html)//1024} KB, {len(rows)} artifacts)")


DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GPU Serving — Results Vault Dashboard</title>
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
    }
    [data-theme="light"] {
      --bg: #f8fafc; --surface: #ffffff; --surface-2: #f1f5f9;
      --border: #e2e8f0; --text: #0f172a; --text-muted: #64748b;
      --accent-soft: rgba(99,102,241,0.08); --green-soft: rgba(34,197,94,0.08);
      --red-soft: rgba(239,68,68,0.08); --yellow-soft: rgba(245,158,11,0.08);
      --blue-soft: rgba(59,130,246,0.08);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }
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
    .theme-toggle:hover { background: var(--border); }
    main { max-width: 1500px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }
    h2 { font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem; }
    .section { margin-bottom: 2rem; }

    .summary-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.8rem; margin-bottom: 1.5rem;
    }
    .summary-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; padding: 1rem 1.1rem;
    }
    .summary-card .label {
      font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--text-muted); margin-bottom: 0.3rem;
    }
    .summary-card .value {
      font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 600; color: var(--accent);
    }
    .summary-card .sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.3rem; }

    .filter-bar {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;
    }
    .filter-group label {
      display: block; font-size: 0.7rem; text-transform: uppercase;
      letter-spacing: 0.06em; color: var(--text-muted); margin-bottom: 0.4rem;
    }
    .filter-group .search {
      width: 100%; padding: 0.4rem 0.6rem; background: var(--surface-2); border: 1px solid var(--border);
      color: var(--text); border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    }
    .filter-group .search:focus { outline: none; border-color: var(--accent); }

    /* ---- multi-select dropdown ---- */
    .ms { position: relative; }
    .ms-btn {
      width: 100%; text-align: left; padding: 0.45rem 0.6rem; background: var(--surface-2);
      border: 1px solid var(--border); color: var(--text); border-radius: 6px; cursor: pointer;
      font-family: 'Inter', sans-serif; font-size: 0.8rem; display: flex;
      align-items: center; justify-content: space-between; gap: 0.5rem;
    }
    .ms-btn:hover { border-color: var(--accent); }
    .ms-btn.partial { border-color: var(--accent); color: var(--accent); }
    .ms-btn .caret { color: var(--text-muted); font-size: 0.7rem; }
    .ms-panel {
      display: none; position: absolute; z-index: 200; top: calc(100% + 4px); left: 0; right: 0;
      background: var(--surface); border: 1px solid var(--accent); border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.35); max-height: 280px; overflow: auto; padding: 0.4rem;
    }
    .ms.open .ms-panel { display: block; }
    .ms-tools { display: flex; gap: 0.4rem; padding: 0.2rem 0.3rem 0.45rem; border-bottom: 1px solid var(--border); margin-bottom: 0.3rem; }
    .ms-tools button {
      flex: 1; background: var(--surface-2); border: 1px solid var(--border); color: var(--text-muted);
      border-radius: 5px; padding: 0.25rem; font-size: 0.7rem; cursor: pointer; font-family: inherit;
    }
    .ms-tools button:hover { color: var(--accent); border-color: var(--accent); }
    .ms-opt {
      display: flex; align-items: center; gap: 0.5rem; padding: 0.32rem 0.4rem; border-radius: 5px;
      cursor: pointer; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace;
    }
    .ms-opt:hover { background: var(--accent-soft); }
    .ms-opt input { accent-color: var(--accent); cursor: pointer; }

    /* ---- SLO / workload control panel ---- */
    .slo-bar {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;
    }
    .slo-bar .slo-head {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 0.85rem; flex-wrap: wrap; gap: 0.5rem;
    }
    .slo-bar .slo-head h2 { margin: 0; }
    .preset-row { display: flex; flex-wrap: wrap; gap: 6px; }
    .preset {
      padding: 0.3rem 0.7rem; border-radius: 6px; font-size: 0.74rem;
      border: 1px solid var(--border); background: var(--surface-2);
      color: var(--text-muted); cursor: pointer; user-select: none;
      font-family: 'Inter', sans-serif; font-weight: 500;
    }
    .preset:hover { border-color: var(--accent); }
    .preset.active { background: var(--accent-soft); color: var(--accent); border-color: var(--accent); }
    .slo-inputs {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.85rem;
    }
    .slo-inputs label {
      display: block; font-size: 0.68rem; text-transform: uppercase;
      letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 0.3rem;
    }
    .slo-inputs input {
      width: 100%; padding: 0.4rem 0.55rem; background: var(--surface-2);
      border: 1px solid var(--border); color: var(--text); border-radius: 6px;
      font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    }
    .slo-inputs input:focus { outline: none; border-color: var(--accent); }
    .slo-inputs input.dirty { border-color: var(--accent); }
    .slo-actions { margin-top: 0.85rem; display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
    .slo-actions button {
      background: var(--accent); border: none; color: #fff; padding: 0.45rem 1rem;
      border-radius: 6px; cursor: pointer; font-family: inherit; font-size: 0.8rem; font-weight: 600;
    }
    .slo-actions button.ghost { background: var(--surface-2); color: var(--text-muted); border: 1px solid var(--border); }
    .slo-actions button:hover { filter: brightness(1.1); }
    .slo-verdict { font-size: 0.82rem; color: var(--text-muted); }
    .slo-verdict b { color: var(--green); font-family: 'JetBrains Mono', monospace; }

    /* winner card highlight */
    .summary-card.winner { border-color: var(--green); background: var(--green-soft); }
    .summary-card.winner .value { color: var(--green); }

    .chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; margin-bottom: 1.5rem; position: relative; }
    .pareto-tip {
      position: absolute; pointer-events: none; display: none; z-index: 20;
      background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px;
      padding: 0.4rem 0.55rem; font-size: 0.72rem; font-family: 'Inter', sans-serif;
      color: var(--text); box-shadow: 0 4px 12px rgba(0,0,0,0.25); max-width: 280px;
    }
    .pareto-tip b { color: var(--accent); }
    .chart-row { display: block; }

    canvas { width: 100%; height: 420px; }

    .table-wrap {
      background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
      overflow: auto; max-height: 620px;
    }
    table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
    thead { background: var(--surface-2); position: sticky; top: 0; z-index: 10; }
    th {
      padding: 0.6rem 0.75rem; text-align: left; font-weight: 600; font-size: 0.7rem;
      text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted);
      cursor: pointer; user-select: none; white-space: nowrap;
      border-bottom: 1px solid var(--border);
    }
    th:hover { color: var(--text); }
    th.sorted-asc::after { content: ' ↑'; color: var(--accent); }
    th.sorted-desc::after { content: ' ↓'; color: var(--accent); }
    td {
      padding: 0.5rem 0.75rem; border-top: 1px solid var(--border);
      font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; white-space: nowrap;
    }
    td.name { font-family: 'Inter', sans-serif; font-weight: 500; }
    tr:hover td { background: var(--accent-soft); }
    tr.row-hl td { background: var(--accent-soft); box-shadow: inset 3px 0 0 var(--accent); }
    .slo-toggle { display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.78rem; color: var(--text-muted); cursor: pointer; user-select: none; }
    .slo-toggle input { cursor: pointer; }
    canvas#paretoChart { cursor: crosshair; }
    .badge {
      display: inline-block; padding: 0.12rem 0.45rem; border-radius: 4px; font-size: 0.68rem;
      font-weight: 600; font-family: 'Inter', sans-serif;
    }
    .badge-pass { background: var(--green-soft); color: var(--green); }
    .badge-fail { background: var(--red-soft); color: var(--red); }
    .badge-null { background: var(--surface-2); color: var(--text-muted); }

    .muted { color: var(--text-muted); }
    .legend { display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 0.75rem; font-size: 0.78rem; color: var(--text-muted); }
    .legend-item { display: flex; align-items: center; gap: 0.4rem; }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; }

    footer { text-align: center; padding: 2rem; color: var(--text-muted); font-size: 0.72rem; border-top: 1px solid var(--border); margin-top: 2rem; }
  </style>
</head>
<body>

<header>
  <div>
    <h1>GPU Serving — Results Vault Dashboard</h1>
    <div class="meta" id="header-meta">loading…</div>
  </div>
  <button class="theme-toggle" onclick="toggleTheme()">Toggle theme</button>
</header>

<main>
  <!-- Summary cards -->
  <div class="summary-grid" id="summary-grid"></div>

  <!-- Filters -->
  <div class="filter-bar">
    <div class="filter-group">
      <label>Model</label>
      <div class="ms" id="ms-model"></div>
    </div>
    <div class="filter-group">
      <label>GPU</label>
      <div class="ms" id="ms-gpu"></div>
    </div>
    <div class="filter-group">
      <label>Engine</label>
      <div class="ms" id="ms-engine"></div>
    </div>
    <div class="filter-group">
      <label>Workload</label>
      <div class="ms" id="ms-workload"></div>
    </div>
    <div class="filter-group">
      <label>ISL — input tokens (auto)</label>
      <input class="search" id="isl-display" readonly placeholder="—" style="cursor:default" />
    </div>
    <div class="filter-group">
      <label>OSL — output tokens (auto)</label>
      <input class="search" id="osl-display" readonly placeholder="—" style="cursor:default" />
    </div>
    <div class="filter-group">
      <label>Search (filename / tag)</label>
      <input class="search" id="filter-search" placeholder="e.g. s4d4 or c=256 or tp4dp2" />
    </div>
  </div>

  <!-- SLO / workload control panel -->
  <div class="slo-bar">
    <div class="slo-head">
      <h2>SLO target &amp; workload</h2>
      <div class="preset-row" id="preset-row"></div>
    </div>
    <div class="slo-inputs">
      <div><label>Max TTFT p99 (ms)</label><input type="number" id="slo-ttft" placeholder="∞" min="0" step="50"></div>
      <div><label>Max TPOT p99 (ms/tok)</label><input type="number" id="slo-tpot" placeholder="∞" min="0" step="1"></div>
      <div><label>Max E2E p99 (ms)</label><input type="number" id="slo-e2e" placeholder="∞" min="0" step="100"></div>
      <div><label>Min throughput (tok/s)</label><input type="number" id="slo-tput" placeholder="0" min="0" step="100"></div>
      <div><label>Min req/s (embed/rerank)</label><input type="number" id="slo-reqs" placeholder="0" min="0" step="1"></div>
      <div><label>Max $ / 1M out tok</label><input type="number" id="slo-cost" placeholder="∞" min="0" step="1"></div>
      <div><label>Max instance $/hr</label><input type="number" id="slo-hr" placeholder="∞" min="0" step="1"></div>
    </div>
    <div class="slo-actions">
      <button onclick="applySLO()">Apply SLO</button>
      <button class="ghost" onclick="clearSLO()">Clear</button>
      <label class="slo-toggle"><input type="checkbox" id="slo-table-filter" onchange="applyFilters()"> Show only passing rows in table</label>
      <span class="slo-verdict" id="slo-verdict">No SLO set — all configs shown at full opacity.</span>
    </div>
  </div>

  <!-- Charts -->
  <div class="chart-row">
    <div class="chart-wrap">
      <h2>Aggregate throughput vs concurrency</h2>
      <canvas id="aggChart"></canvas>
      <div class="legend" id="aggLegend"></div>
      <div class="muted" style="margin-top:0.5rem; font-size:0.78rem;">
        One line per (model, engine_config_tag). Use filters to scope. Missing concurrency = gap in data (e.g. embedding runs don't sweep 1..512).
      </div>
    </div>
    <div class="chart-wrap">
      <h2>Latency-throughput Pareto</h2>
      <canvas id="paretoChart"></canvas>
      <div class="pareto-tip" id="paretoTip"></div>
      <div class="legend" id="paretoLegend"></div>
      <div class="muted" style="margin-top:0.5rem; font-size:0.78rem;">
        X = aggregate tok/s · Y = decode latency (ms/token from TPOT). Missing points = TTFT/TPOT not captured (see vault README).
      </div>
    </div>
  </div>

  <!-- Table -->
  <div class="section">
    <h2>Artifacts (<span id="row-count">0</span>)</h2>
    <div class="table-wrap">
      <table id="artifact-table">
        <thead>
          <tr id="table-header"></tr>
        </thead>
        <tbody id="table-body"></tbody>
      </table>
    </div>
  </div>
</main>

<footer>
  Generated from <code>index.json</code> · data snapshot <span id="gen-ts"></span> · regenerate with <code>python3 build-dashboard.py</code>
</footer>

<script>
const DATA = __INDEX_JSON__;

// ------- Columns shown in table -------
const COLUMNS = [
  { key: 'model_name',            label: 'Model',          type: 'str', width: 180 },
  { key: 'gpu_type',              label: 'GPU',            type: 'str' },
  { key: 'instance_type',         label: 'Instance',       type: 'str' },
  { key: 'engine_config_tag',     label: 'Engine config',  type: 'str', width: 260 },
  { key: 'workload_catalog_id',   label: 'Workload',       type: 'str' },
  { key: 'concurrency',           label: 'c',              type: 'num' },
  { key: 'agg_tok_per_s',         label: 'Agg tok/s',      type: 'num' },
  { key: 'request_throughput_per_s', label: 'Req/s',       type: 'num' },
  { key: 'ttft_p50_ms',           label: 'TTFT p50',       type: 'num' },
  { key: 'ttft_p99_ms',           label: 'TTFT p99',       type: 'num' },
  { key: 'tpot_p99_ms',           label: 'TPOT p99',       type: 'num' },
  { key: 'e2e_p99_ms',            label: 'E2E p99',        type: 'num' },
  { key: 'spec_accept_length',    label: 'Accept len',     type: 'num' },
  { key: 'dollars_per_1m_output_tokens', label: '$/M tok', type: 'num' },
  { key: 'slo_overall_pass',      label: 'SLO',            type: 'bool' },
];

// ------- Filters -------
const filters = {
  models: new Set(),
  gpus: new Set(),
  engines: new Set(),      // stored as engine_name, not tag
  workloads: new Set(),
  search: '',
};

// ------- SLO state (null = unconstrained) -------
const slo = { ttft: null, tpot: null, e2e: null, tput: null, reqs: null, cost: null, hr: null };

// Non-generative architectures: no token decode, so TPOT/TTFT/tok-s are
// undefined. These are judged on E2E latency + req/s instead.
const NONGEN_ARCH = new Set(['embedding', 'cross-encoder']);
function isGenerative(r) { return !NONGEN_ARCH.has(r.model_architecture); }

// ------- Workload presets: map a label to the catalog_ids it selects -------
const PRESETS = {
  'All workloads':   null,
  'Long context':    ['rag-long-context', 'rag-1m-context', 'shared-prefix-multitenant'],
  'Concurrency':     ['concurrency-sweep'],
  'QPS sweep':       ['qps-sweep'],
  'Production mix':  ['production-mix', 'chatbot-short'],
  'Throughput/batch':['batch-throughput', 'burn-in'],
};
let activePreset = 'All workloads';

// Does a row satisfy the current SLO? Unset thresholds are ignored.
// Latency/throughput SLOs are modality-aware:
//   generative     → judged on TTFT/TPOT (token decode) + tok/s
//   embed/reranker → judged on E2E latency + req/s (no token decode)
// A token SLO never rejects a non-generative row, and a req/s SLO never
// rejects a generative one — the metric simply doesn't apply. Cost and $/hr
// apply to all. Missing a constrained, applicable metric = fail (can't prove pass).
function sloPass(r) {
  const gen = isGenerative(r);
  if (gen) {
    if (slo.ttft != null) { if (typeof r.ttft_p99_ms !== 'number' || r.ttft_p99_ms > slo.ttft) return false; }
    if (slo.tpot != null) { if (typeof r.tpot_p99_ms !== 'number' || r.tpot_p99_ms > slo.tpot) return false; }
    if (slo.tput != null) { if (typeof r.agg_tok_per_s !== 'number' || r.agg_tok_per_s < slo.tput) return false; }
  } else {
    if (slo.reqs != null) { if (typeof r.request_throughput_per_s !== 'number' || r.request_throughput_per_s < slo.reqs) return false; }
  }
  // E2E latency applies to both; for embedders it's the primary latency SLO.
  if (slo.e2e  != null) { if (typeof r.e2e_p99_ms !== 'number' || r.e2e_p99_ms > slo.e2e) return false; }
  if (slo.cost != null) { if (typeof r.dollars_per_1m_output_tokens !== 'number' || r.dollars_per_1m_output_tokens > slo.cost) return false; }
  if (slo.hr   != null) { if (typeof r.instance_cost_per_hr !== 'number' || r.instance_cost_per_hr > slo.hr) return false; }
  return true;
}
function sloActive() { return Object.values(slo).some(v => v != null); }

// ------- State -------
let sortKey = 'agg_tok_per_s';
let sortDir = 'desc';

// Registry of dropdowns so we can refresh button labels after All/None etc.
// Declared before bootstrap (buildFilterChips) — const is not hoisted.
const DROPDOWNS = [];

// table <tr> elements keyed by artifact file (for point↔row hover linking)
let rowEls = {};
// Pareto scatter point hit-boxes: { x, y, r, file } in CSS px (rebuilt each draw)
let paretoPoints = [];
let hlFile = null;
// Last filtered+sorted rows, keyed lookup for tooltip content
let lastRowsByFile = {};
// Rows last passed to the Pareto chart, so highlight can redraw without refilter
let lastParetoRows = null;

// ------- Bootstrap -------
document.getElementById('gen-ts').textContent = DATA.generated_at || 'unknown';
document.getElementById('header-meta').textContent =
  `${DATA.artifact_count} artifacts · ${uniq('model_name').length} models · ${uniq('gpu_type').length} GPUs · generated ${DATA.generated_at}`;

buildFilterChips();
buildPresets();
wireSLOInputs();
wireParetoHover();
renderHeader();
// Wait one frame so CSS grid has laid out canvas widths
requestAnimationFrame(() => requestAnimationFrame(applyFilters));

function buildPresets() {
  const row = document.getElementById('preset-row');
  row.innerHTML = '';
  Object.keys(PRESETS).forEach(name => {
    const el = document.createElement('div');
    el.className = 'preset' + (name === activePreset ? ' active' : '');
    el.textContent = name;
    el.onclick = () => {
      activePreset = name;
      document.querySelectorAll('#preset-row .preset').forEach(p =>
        p.classList.toggle('active', p.textContent === name));
      applyFilters();
    };
    row.appendChild(el);
  });
}

function wireSLOInputs() {
  ['ttft','tpot','e2e','tput','reqs','cost','hr'].forEach(k => {
    const el = document.getElementById('slo-' + k);
    el.addEventListener('input', () => el.classList.toggle('dirty', el.value !== ''));
    el.addEventListener('keydown', e => { if (e.key === 'Enter') applySLO(); });
  });
}

// Hover over a Pareto point → highlight matching table row + show tooltip.
// Hover over a table row → highlight matching Pareto point.
function wireParetoHover() {
  const canvas = document.getElementById('paretoChart');
  const tip = document.getElementById('paretoTip');

  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let hit = null, best = 1e9;
    for (const p of paretoPoints) {
      const d = (p.x - mx) ** 2 + (p.y - my) ** 2;
      if (d <= (p.r + 5) ** 2 && d < best) { best = d; hit = p; }
    }
    if (hit) {
      setHighlight(hit.file, false);
      const r = lastRowsByFile[hit.file];
      if (r) {
        tip.innerHTML = `<b>${r.model_name}</b> ${r.engine_config_tag || r.engine_name || ''}<br>` +
          `${r.gpu_type || '?'} · c=${r.concurrency ?? '?'} · ${r.workload_catalog_id || '?'}<br>` +
          `${Math.round(r.agg_tok_per_s || 0).toLocaleString()} tok/s · TPOT ${(r.tpot_p50_ms ?? 0).toFixed(1)}ms` +
          (typeof r.dollars_per_1m_output_tokens === 'number' ? ` · $${r.dollars_per_1m_output_tokens}/1M` : '');
        tip.style.display = 'block';
        // Position within the chart-wrap, nudged away from the cursor.
        const wrap = canvas.parentElement.getBoundingClientRect();
        let lx = e.clientX - wrap.left + 12, ly = e.clientY - wrap.top + 12;
        if (lx + tip.offsetWidth > wrap.width) lx = e.clientX - wrap.left - tip.offsetWidth - 12;
        tip.style.left = lx + 'px'; tip.style.top = ly + 'px';
      }
      canvas.style.cursor = 'pointer';
    } else {
      setHighlight(null, false);
      tip.style.display = 'none';
      canvas.style.cursor = 'crosshair';
    }
  });
  canvas.addEventListener('mouseleave', () => {
    setHighlight(null, false);
    tip.style.display = 'none';
  });

  // Reverse direction: row hover → point. Delegated so it survives re-render.
  const body = document.getElementById('table-body');
  body.addEventListener('mouseover', e => {
    const tr = e.target.closest('tr');
    if (tr && tr.dataset.file) setHighlight(tr.dataset.file, true);
  });
  body.addEventListener('mouseleave', () => setHighlight(null, true));
}

// Set the highlighted artifact and reflect it in both views.
// fromRow=true means the hover originated in the table (don't re-highlight rows,
// just redraw the chart); fromRow=false means it came from the chart.
function setHighlight(file, fromRow) {
  if (file === hlFile) return;
  hlFile = file;
  if (!fromRow) {
    Object.values(rowEls).forEach(tr => tr.classList.remove('row-hl'));
    if (file && rowEls[file]) {
      rowEls[file].classList.add('row-hl');
      rowEls[file].scrollIntoView({ block: 'nearest' });
    }
  }
  if (lastParetoRows) drawParetoChart(lastParetoRows);
}

function applySLO() {
  const num = id => { const v = document.getElementById(id).value; return v === '' ? null : parseFloat(v); };
  slo.ttft = num('slo-ttft'); slo.tpot = num('slo-tpot'); slo.e2e = num('slo-e2e');
  slo.tput = num('slo-tput'); slo.reqs = num('slo-reqs'); slo.cost = num('slo-cost'); slo.hr = num('slo-hr');
  applyFilters();
}

function clearSLO() {
  Object.keys(slo).forEach(k => slo[k] = null);
  ['ttft','tpot','e2e','tput','reqs','cost','hr'].forEach(k => {
    const el = document.getElementById('slo-' + k); el.value = ''; el.classList.remove('dirty');
  });
  applyFilters();
}

function uniq(key) {
  return Array.from(new Set(DATA.artifacts.map(r => r[key]).filter(v => v !== undefined && v !== null))).sort();
}

function buildFilterChips() {
  buildDropdown('ms-model',    'model',    uniq('model_name'),         filters.models,   'Models');
  buildDropdown('ms-gpu',      'gpu',      uniq('gpu_type'),           filters.gpus,     'GPUs');
  buildDropdown('ms-engine',   'engine',   uniq('engine_name'),        filters.engines,  'Engines');
  buildDropdown('ms-workload', 'workload', uniq('workload_catalog_id'),filters.workloads,'Workloads');

  document.getElementById('filter-search').addEventListener('input', e => {
    filters.search = e.target.value.toLowerCase();
    applyFilters();
  });

  // Close any open panel when clicking outside.
  document.addEventListener('click', e => {
    if (!e.target.closest('.ms')) DROPDOWNS.forEach(d => d.root.classList.remove('open'));
  });
}

// Build one multi-select dropdown. stateSet starts empty = "all" (no filter).
function buildDropdown(mountId, name, values, stateSet, noun) {
  const root = document.getElementById(mountId);
  root.innerHTML = '';
  const btn = document.createElement('button');
  btn.className = 'ms-btn';
  btn.innerHTML = `<span class="ms-label"></span><span class="caret">▾</span>`;
  const panel = document.createElement('div');
  panel.className = 'ms-panel';

  const tools = document.createElement('div');
  tools.className = 'ms-tools';
  const allBtn = document.createElement('button'); allBtn.textContent = 'All';
  const noneBtn = document.createElement('button'); noneBtn.textContent = 'None';
  tools.append(allBtn, noneBtn);
  panel.appendChild(tools);

  const boxes = [];
  values.forEach(v => {
    const opt = document.createElement('label');
    opt.className = 'ms-opt';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.value = v; cb.checked = true; // default: all on
    cb.onchange = syncFromBoxes;
    const span = document.createElement('span'); span.textContent = v;
    opt.append(cb, span);
    panel.appendChild(opt);
    boxes.push(cb);
  });

  // We model selection as "checked = included". filters.<set> stores the
  // INCLUDED values when not all are checked; empty set = all (no filter),
  // matching applyFilters' `if (set.size && !set.has(...))` semantics.
  function syncFromBoxes() {
    const checked = boxes.filter(b => b.checked).map(b => b.value);
    stateSet.clear();
    if (checked.length !== boxes.length) checked.forEach(v => stateSet.add(v));
    updateLabel();
    applyFilters();
  }
  function updateLabel() {
    const checked = boxes.filter(b => b.checked).length;
    const lbl = btn.querySelector('.ms-label');
    if (checked === boxes.length) { lbl.textContent = `All ${noun}`; btn.classList.remove('partial'); }
    else if (checked === 0)       { lbl.textContent = `No ${noun}`;  btn.classList.add('partial'); }
    else                          { lbl.textContent = `${checked} of ${boxes.length} ${noun}`; btn.classList.add('partial'); }
  }

  allBtn.onclick = () => { boxes.forEach(b => b.checked = true); syncFromBoxes(); };
  noneBtn.onclick = () => { boxes.forEach(b => b.checked = false); syncFromBoxes(); };
  btn.onclick = () => {
    const wasOpen = root.classList.contains('open');
    DROPDOWNS.forEach(d => d.root.classList.remove('open'));
    root.classList.toggle('open', !wasOpen);
  };

  root.append(btn, panel);
  DROPDOWNS.push({ root, updateLabel });
  updateLabel();
}

function renderHeader() {
  const tr = document.getElementById('table-header');
  tr.innerHTML = '';
  COLUMNS.forEach(col => {
    const th = document.createElement('th');
    th.textContent = col.label;
    th.onclick = () => {
      if (sortKey === col.key) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      else { sortKey = col.key; sortDir = col.type === 'str' ? 'asc' : 'desc'; }
      applyFilters();
    };
    tr.appendChild(th);
  });
  // Static, non-sortable columns appended after the data columns.
  ['vs SLO', 'Detail'].forEach(label => {
    const th = document.createElement('th');
    th.textContent = label;
    tr.appendChild(th);
  });
}

function applyFilters() {
  const presetIds = PRESETS[activePreset];
  const rows = DATA.artifacts.filter(r => {
    if (filters.models.size && !filters.models.has(r.model_name)) return false;
    if (filters.gpus.size && !filters.gpus.has(r.gpu_type)) return false;
    if (filters.engines.size && !filters.engines.has(r.engine_name)) return false;
    if (filters.workloads.size && !filters.workloads.has(r.workload_catalog_id)) return false;
    if (presetIds && !presetIds.includes(r.workload_catalog_id)) return false;
    if (filters.search) {
      const hay = [r.file, r.engine_config_tag, r.model_name].filter(Boolean).join(' ').toLowerCase();
      // Support "c=NN" shorthand
      const s = filters.search;
      if (s.match(/^c\s*=\s*\d+$/)) {
        const n = parseInt(s.replace(/\D/g, ''), 10);
        if (r.concurrency !== n) return false;
      } else if (!hay.includes(s)) {
        return false;
      }
    }
    return true;
  });

  // Sort
  rows.sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (av === undefined && bv === undefined) return 0;
    if (av === undefined) return 1;
    if (bv === undefined) return -1;
    if (typeof av === 'number' && typeof bv === 'number') {
      return sortDir === 'asc' ? av - bv : bv - av;
    }
    const as = String(av), bs = String(bv);
    return sortDir === 'asc' ? as.localeCompare(bs) : bs.localeCompare(as);
  });

  lastRowsByFile = {};
  rows.forEach(r => { if (r.file) lastRowsByFile[r.file] = r; });

  renderTable(rows);
  renderSummary(rows);
  renderVerdict(rows);
  renderISLOSL(rows);
  drawAggChart(rows);
  drawParetoChart(rows);

  document.querySelectorAll('th').forEach((th, i) => {
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (COLUMNS[i] && COLUMNS[i].key === sortKey) th.classList.add(sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
  });
}

function reportHref(r) {
  const rel = DATA.reports_rel, file = (DATA.report_map || {})[r.model_name];
  return file ? `${rel}/${file}` : null;
}

function renderTable(rows) {
  const sloOn = sloActive();
  const onlyPass = sloOn && document.getElementById('slo-table-filter').checked;
  const shown = onlyPass ? rows.filter(sloPass) : rows;
  document.getElementById('row-count').textContent = shown.length;
  const body = document.getElementById('table-body');
  body.innerHTML = '';
  rowEls = {};
  shown.forEach(r => {
    const tr = document.createElement('tr');
    if (r.file) { tr.dataset.file = r.file; rowEls[r.file] = tr; }
    const pass = sloPass(r);
    if (sloOn && !pass) tr.style.opacity = '0.38';
    COLUMNS.forEach(col => {
      const td = document.createElement('td');
      const v = r[col.key];
      if (col.type === 'bool') {
        td.innerHTML = v === true ? '<span class="badge badge-pass">pass</span>' :
                       v === false ? '<span class="badge badge-fail">fail</span>' :
                       '<span class="badge badge-null">—</span>';
      } else if (v === undefined || v === null) {
        td.innerHTML = '<span class="muted">—</span>';
      } else if (col.type === 'num') {
        td.textContent = typeof v === 'number' ? (v >= 100 ? v.toFixed(0) : v.toFixed(2)) : v;
      } else {
        if (col.key === 'model_name') td.className = 'name';
        td.textContent = v;
      }
      tr.appendChild(td);
    });
    // SLO-vs-target column
    const sloTd = document.createElement('td');
    if (sloOn) sloTd.innerHTML = pass ? '<span class="badge badge-pass">meets</span>'
                                      : '<span class="badge badge-fail">misses</span>';
    else sloTd.innerHTML = '<span class="muted">—</span>';
    tr.appendChild(sloTd);
    // Drill-down report link
    const repTd = document.createElement('td');
    const href = reportHref(r);
    repTd.innerHTML = href ? `<a href="${href}" target="_blank" style="color:var(--accent)">report ↗</a>`
                           : '<span class="muted">—</span>';
    tr.appendChild(repTd);
    body.appendChild(tr);
  });
}

function renderSummary(rows) {
  // Top agg throughput + winner config
  const withAgg = rows.filter(r => typeof r.agg_tok_per_s === 'number' && r.agg_tok_per_s > 0);
  const maxAgg = withAgg.length ? withAgg.reduce((a, b) => a.agg_tok_per_s > b.agg_tok_per_s ? a : b) : null;
  // Best req/s (for embedding-style)
  const withReq = rows.filter(r => typeof r.request_throughput_per_s === 'number' && r.request_throughput_per_s > 0);
  const maxReq = withReq.length ? withReq.reduce((a, b) => a.request_throughput_per_s > b.request_throughput_per_s ? a : b) : null;
  // Cheapest $/M tok
  const withCost = rows.filter(r => typeof r.dollars_per_1m_output_tokens === 'number' && r.dollars_per_1m_output_tokens > 0);
  const cheapest = withCost.length ? withCost.reduce((a, b) => a.dollars_per_1m_output_tokens < b.dollars_per_1m_output_tokens ? a : b) : null;
  // Unique engine configs
  const configs = new Set(rows.map(r => r.engine_config_tag).filter(Boolean));

  const cards = [
    { label: 'Artifacts (filtered)', value: rows.length, sub: `of ${DATA.artifact_count} total` },
    { label: 'Peak aggregate', value: maxAgg ? `${Math.round(maxAgg.agg_tok_per_s).toLocaleString()} tok/s` : '—',
      sub: maxAgg ? `${maxAgg.engine_config_tag} @ c=${maxAgg.concurrency}` : 'no data' },
    { label: 'Peak request rate', value: maxReq ? `${maxReq.request_throughput_per_s.toFixed(1)} req/s` : '—',
      sub: maxReq ? `${maxReq.engine_config_tag || maxReq.workload_catalog_id}` : 'no data' },
    { label: 'Cheapest $/M out tok', value: cheapest ? `$${cheapest.dollars_per_1m_output_tokens}` : '—',
      sub: cheapest ? `${cheapest.engine_config_tag} @ c=${cheapest.concurrency}` : 'no data' },
    { label: 'Unique engine configs', value: configs.size, sub: '' },
  ];
  document.getElementById('summary-grid').innerHTML = cards.map(c => `
    <div class="summary-card">
      <div class="label">${c.label}</div>
      <div class="value">${c.value}</div>
      <div class="sub">${c.sub}</div>
    </div>
  `).join('');
}

function renderVerdict(rows) {
  const el = document.getElementById('slo-verdict');
  if (!sloActive()) {
    el.innerHTML = 'No SLO set — all configs shown at full opacity.';
    return;
  }
  const passing = rows.filter(sloPass);
  const parts = [];
  if (slo.ttft != null) parts.push(`TTFT p99 ≤ ${slo.ttft}ms`);
  if (slo.tpot != null) parts.push(`TPOT p99 ≤ ${slo.tpot}ms`);
  if (slo.e2e  != null) parts.push(`E2E p99 ≤ ${slo.e2e}ms`);
  if (slo.tput != null) parts.push(`≥ ${slo.tput} tok/s`);
  if (slo.reqs != null) parts.push(`≥ ${slo.reqs} req/s`);
  if (slo.cost != null) parts.push(`≤ $${slo.cost}/1M`);
  if (slo.hr   != null) parts.push(`≤ $${slo.hr}/hr`);
  if (!passing.length) {
    el.innerHTML = `<b style="color:var(--red)">0</b> of ${rows.length} configs meet [${parts.join(', ')}] — relax the SLO or widen filters.`;
    return;
  }
  // Cheapest passing config (by $/1M out tok), else highest throughput.
  const withCost = passing.filter(r => typeof r.dollars_per_1m_output_tokens === 'number');
  const winner = withCost.length
    ? withCost.reduce((a, b) => a.dollars_per_1m_output_tokens < b.dollars_per_1m_output_tokens ? a : b)
    : passing.reduce((a, b) => (a.agg_tok_per_s || 0) > (b.agg_tok_per_s || 0) ? a : b);
  const cost = typeof winner.dollars_per_1m_output_tokens === 'number'
    ? `$${winner.dollars_per_1m_output_tokens.toFixed(2)}/1M` : 'cost n/a';
  el.innerHTML = `<b>${passing.length}</b> of ${rows.length} meet [${parts.join(', ')}] · ` +
    `best: <b>${winner.model_name}</b> ${winner.engine_config_tag || ''} on ${winner.instance_type || '?'} ` +
    `@ c=${winner.concurrency ?? '?'} → ${cost}, ${Math.round(winner.agg_tok_per_s || 0).toLocaleString()} tok/s`;
}

// Auto-populate ISL/OSL display from the current selection. Shows the
// min..max span of input/output tokens across filtered rows (a single value
// if they all match). Reflects whatever workload/model/etc. is selected.
function renderISLOSL(rows) {
  const fmt = (key, unit) => {
    const vs = rows.map(r => r[key]).filter(v => typeof v === 'number');
    if (!vs.length) return '—';
    const lo = Math.min(...vs), hi = Math.max(...vs);
    const r = n => n >= 1000 ? (n / 1000).toFixed(n % 1000 ? 1 : 0) + 'K' : String(Math.round(n));
    return lo === hi ? `${r(lo)} ${unit}` : `${r(lo)}–${r(hi)} ${unit}`;
  };
  document.getElementById('isl-display').value = fmt('input_tokens_mean', 'tok');
  document.getElementById('osl-display').value = fmt('output_tokens_mean', 'tok');
}

// ------- Color palette per engine_config_tag -------
const PALETTE = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#3b82f6', '#ec4899', '#14b8a6', '#f97316', '#a855f7', '#06b6d4', '#84cc16'];
const colorCache = {};
function colorFor(key) {
  if (!colorCache[key]) colorCache[key] = PALETTE[Object.keys(colorCache).length % PALETTE.length];
  return colorCache[key];
}

function groupByConfig(rows) {
  const groups = {};
  rows.forEach(r => {
    const key = `${r.model_name} · ${r.engine_config_tag || r.engine_name || 'unknown'}`;
    (groups[key] = groups[key] || []).push(r);
  });
  // Sort each group's points by concurrency for proper line rendering
  Object.values(groups).forEach(arr => arr.sort((a, b) => (a.concurrency || 0) - (b.concurrency || 0)));
  return groups;
}

function drawCanvas(canvasId, draw) {
  const canvas = document.getElementById(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  // Guard against pre-layout zero-width on first render
  const W = Math.max(320, rect.width);
  const H = Math.max(240, rect.height || 360);
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);  // reset any prior scale
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  draw(ctx, W, H);
}

function drawAggChart(rows) {
  drawCanvas('aggChart', (ctx, W, H) => {
    const pad = { top: 20, right: 12, bottom: 40, left: 60 };
    const cW = W - pad.left - pad.right, cH = H - pad.top - pad.bottom;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const axis = isDark ? '#475569' : '#94a3b8';
    const grid = isDark ? '#1e293b' : '#e2e8f0';
    const txt = isDark ? '#e2e8f0' : '#0f172a';
    const mut = isDark ? '#64748b' : '#475569';

    const xLabels = [1, 8, 32, 64, 128, 256, 512];
    const groups = groupByConfig(rows.filter(r => typeof r.concurrency === 'number' && typeof r.agg_tok_per_s === 'number' && r.agg_tok_per_s > 0));
    const allVals = Object.values(groups).flat().map(r => r.agg_tok_per_s);
    const maxY = Math.max(1000, ...allVals) * 1.05;

    const xPos = c => {
      const i = xLabels.indexOf(c);
      if (i >= 0) return pad.left + (i / (xLabels.length - 1)) * cW;
      // interpolate for off-spec concurrencies
      const lo = Math.max(...xLabels.filter(x => x <= c).concat([1]));
      const hi = Math.min(...xLabels.filter(x => x >= c).concat([512]));
      if (lo === hi) return pad.left + (xLabels.indexOf(lo) / (xLabels.length - 1)) * cW;
      const li = xLabels.indexOf(lo), hi_i = xLabels.indexOf(hi);
      const frac = (Math.log(c) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
      return pad.left + ((li + frac * (hi_i - li)) / (xLabels.length - 1)) * cW;
    };
    const yPos = v => pad.top + (1 - v / maxY) * cH;

    // Grid
    ctx.strokeStyle = grid; ctx.lineWidth = 0.5;
    ctx.fillStyle = mut; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
    xLabels.forEach(c => {
      ctx.beginPath(); ctx.moveTo(xPos(c), pad.top); ctx.lineTo(xPos(c), H - pad.bottom); ctx.stroke();
      ctx.fillText('c=' + c, xPos(c), H - pad.bottom + 14);
    });
    for (let i = 0; i <= 5; i++) {
      const v = (maxY / 5) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, yPos(v)); ctx.lineTo(W - pad.right, yPos(v)); ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(v >= 1000 ? (v / 1000).toFixed(1) + 'K' : v.toFixed(0), pad.left - 5, yPos(v) + 3);
    }
    ctx.strokeStyle = axis; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, H - pad.bottom); ctx.lineTo(W - pad.right, H - pad.bottom); ctx.stroke();
    ctx.fillStyle = txt; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText('Concurrency', (pad.left + W - pad.right) / 2, H - 6);
    ctx.save(); ctx.translate(14, (pad.top + H - pad.bottom) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('Aggregate tok/s', 0, 0); ctx.restore();

    // Lines (limit to top 12 by peak to avoid clutter)
    const groupPeaks = Object.entries(groups).map(([k, vs]) => [k, vs, Math.max(...vs.map(r => r.agg_tok_per_s))]);
    groupPeaks.sort((a, b) => b[2] - a[2]);
    const show = groupPeaks.slice(0, 12);

    show.forEach(([label, vs]) => {
      const color = colorFor(label);
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.beginPath();
      vs.forEach((r, i) => {
        const x = xPos(r.concurrency), y = yPos(r.agg_tok_per_s);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      vs.forEach(r => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(xPos(r.concurrency), yPos(r.agg_tok_per_s), 3, 0, Math.PI * 2);
        ctx.fill();
      });
    });

    // Legend
    const legend = document.getElementById('aggLegend');
    legend.innerHTML = show.map(([label]) => `
      <div class="legend-item"><span class="legend-dot" style="background:${colorFor(label)}"></span>${label}</div>
    `).join('') + (groupPeaks.length > 12 ? `<div class="legend-item muted">+${groupPeaks.length - 12} more (filter to see)</div>` : '');
  });
}

function drawParetoChart(rows) {
  lastParetoRows = rows;
  drawCanvas('paretoChart', (ctx, W, H) => {
    const pad = { top: 20, right: 12, bottom: 40, left: 60 };
    const cW = W - pad.left - pad.right, cH = H - pad.top - pad.bottom;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const axis = isDark ? '#475569' : '#94a3b8';
    const grid = isDark ? '#1e293b' : '#e2e8f0';
    const txt = isDark ? '#e2e8f0' : '#0f172a';
    const mut = isDark ? '#64748b' : '#475569';

    const pts = rows.filter(r => typeof r.agg_tok_per_s === 'number' && r.agg_tok_per_s > 0
                              && typeof r.tpot_p50_ms === 'number' && r.tpot_p50_ms > 0);
    if (!pts.length) {
      paretoPoints = [];
      ctx.fillStyle = mut; ctx.font = '12px Inter'; ctx.textAlign = 'center';
      ctx.fillText('No (throughput, TPOT) points in current filter', W / 2, H / 2);
      ctx.fillText('(Kimi K2.6-spec TTFT/TPOT are null — see vault README)', W / 2, H / 2 + 16);
      document.getElementById('paretoLegend').innerHTML = '';
      return;
    }

    const maxX = Math.max(1000, ...pts.map(r => r.agg_tok_per_s)) * 1.05;
    const maxY = Math.max(10, ...pts.map(r => r.tpot_p50_ms)) * 1.15;
    const xPos = v => pad.left + (v / maxX) * cW;
    const yPos = v => pad.top + (v / maxY) * cH;

    ctx.strokeStyle = grid; ctx.lineWidth = 0.5;
    ctx.fillStyle = mut; ctx.font = '10px Inter';
    for (let i = 0; i <= 5; i++) {
      const v = (maxX / 5) * i;
      ctx.beginPath(); ctx.moveTo(xPos(v), pad.top); ctx.lineTo(xPos(v), H - pad.bottom); ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText(v >= 1000 ? (v / 1000).toFixed(1) + 'K' : v.toFixed(0), xPos(v), H - pad.bottom + 14);
    }
    for (let i = 0; i <= 5; i++) {
      const v = (maxY / 5) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, yPos(v)); ctx.lineTo(W - pad.right, yPos(v)); ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(v.toFixed(0), pad.left - 5, yPos(v) + 3);
    }
    // ---- SLO feasibility box (drawn under points) ----
    // Feasible = right of throughput-min (X) and below TPOT max (Y).
    // X uses agg_tok_per_s (exact axis match). The TPOT line is a p99 target
    // shown against the p50 axis as a visual guide; point dimming below uses
    // the full multi-metric SLO (incl. p99, cost, $/hr), which is authoritative.
    // Only shade when at least one axis-relevant bound is set — otherwise the
    // box would span the whole plot and read as "everything passes".
    if (slo.tput != null || slo.tpot != null) {
      const xMin = slo.tput != null ? Math.min(xPos(slo.tput), W - pad.right) : pad.left;
      const yMax = slo.tpot != null ? Math.min(yPos(slo.tpot), H - pad.bottom) : H - pad.bottom;
      ctx.fillStyle = isDark ? 'rgba(34,197,94,0.10)' : 'rgba(34,197,94,0.12)';
      ctx.fillRect(xMin, pad.top, (W - pad.right) - xMin, yMax - pad.top);
      ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
      if (slo.tput != null) { ctx.beginPath(); ctx.moveTo(xMin, pad.top); ctx.lineTo(xMin, H - pad.bottom); ctx.stroke(); }
      if (slo.tpot != null) { ctx.beginPath(); ctx.moveTo(pad.left, yMax); ctx.lineTo(W - pad.right, yMax); ctx.stroke(); }
      ctx.setLineDash([]);
      ctx.fillStyle = '#22c55e'; ctx.font = '10px Inter'; ctx.textAlign = 'left';
      ctx.fillText('SLO-feasible', xMin + 6, pad.top + 14);
    }

    ctx.strokeStyle = axis; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, H - pad.bottom); ctx.lineTo(W - pad.right, H - pad.bottom); ctx.stroke();
    ctx.fillStyle = txt; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText('Aggregate tok/s', (pad.left + W - pad.right) / 2, H - 6);
    ctx.save(); ctx.translate(14, (pad.top + H - pad.bottom) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('TPOT p50 (ms/token)', 0, 0); ctx.restore();

    // Group by config → scatter by color. Dim points that miss the full SLO.
    const sloOn = sloActive();
    const groups = groupByConfig(pts);
    const groupEntries = Object.entries(groups);

    paretoPoints = [];
    let hlHit = null;
    groupEntries.forEach(([label, vs]) => {
      const color = colorFor(label);
      vs.forEach(r => {
        const pass = sloPass(r);
        const px = xPos(r.agg_tok_per_s), py = yPos(r.tpot_p50_ms);
        const rad = (sloOn && pass) ? 5 : 4;
        if (r.file) paretoPoints.push({ x: px, y: py, r: rad, file: r.file });
        const isHl = hlFile && r.file === hlFile;
        if (isHl) { hlHit = { x: px, y: py, color, r: rad }; }
        ctx.globalAlpha = (sloOn && !pass && !isHl) ? 0.18 : 1.0;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(px, py, rad, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = (sloOn && pass) ? '#22c55e' : (isDark ? '#0f1117' : '#ffffff');
        ctx.lineWidth = (sloOn && pass) ? 1.5 : 1;
        ctx.stroke();
      });
    });
    // Draw the hovered point's emphasis ring last so it sits on top.
    if (hlHit) {
      ctx.globalAlpha = 1.0;
      ctx.beginPath();
      ctx.arc(hlHit.x, hlHit.y, hlHit.r + 4, 0, Math.PI * 2);
      ctx.strokeStyle = isDark ? '#f8fafc' : '#0f172a';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;

    document.getElementById('paretoLegend').innerHTML = groupEntries.slice(0, 10).map(([label]) => `
      <div class="legend-item"><span class="legend-dot" style="background:${colorFor(label)}"></span>${label}</div>
    `).join('');
  });
}

// Theme toggle
function toggleTheme() {
  const html = document.documentElement;
  const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  applyFilters(); // redraw with new colors
}

// Redraw on resize
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(applyFilters, 100);
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
