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
    .chip-row { display: flex; flex-wrap: wrap; gap: 4px; }
    .chip {
      padding: 0.22rem 0.55rem; border-radius: 999px; font-size: 0.72rem;
      border: 1px solid var(--border); background: var(--surface-2);
      color: var(--text-muted); cursor: pointer; user-select: none;
      font-family: 'JetBrains Mono', monospace;
    }
    .chip:hover { border-color: var(--accent); }
    .chip.active { background: var(--accent-soft); color: var(--accent); border-color: var(--accent); }
    .filter-group .search {
      width: 100%; padding: 0.4rem 0.6rem; background: var(--surface-2); border: 1px solid var(--border);
      color: var(--text); border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    }
    .filter-group .search:focus { outline: none; border-color: var(--accent); }

    .chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; }
    .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }
    @media (max-width: 1000px) { .chart-row { grid-template-columns: 1fr; } }

    canvas { width: 100%; height: 360px; }

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
      <div class="chip-row" id="filter-model"></div>
    </div>
    <div class="filter-group">
      <label>GPU</label>
      <div class="chip-row" id="filter-gpu"></div>
    </div>
    <div class="filter-group">
      <label>Engine</label>
      <div class="chip-row" id="filter-engine"></div>
    </div>
    <div class="filter-group">
      <label>Workload</label>
      <div class="chip-row" id="filter-workload"></div>
    </div>
    <div class="filter-group">
      <label>Search (filename / tag)</label>
      <input class="search" id="filter-search" placeholder="e.g. s4d4 or c=256 or tp4dp2" />
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

// ------- State -------
let sortKey = 'agg_tok_per_s';
let sortDir = 'desc';

// ------- Bootstrap -------
document.getElementById('gen-ts').textContent = DATA.generated_at || 'unknown';
document.getElementById('header-meta').textContent =
  `${DATA.artifact_count} artifacts · ${uniq('model_name').length} models · ${uniq('gpu_type').length} GPUs · generated ${DATA.generated_at}`;

buildFilterChips();
renderHeader();
// Wait one frame so CSS grid has laid out canvas widths
requestAnimationFrame(() => requestAnimationFrame(applyFilters));

function uniq(key) {
  return Array.from(new Set(DATA.artifacts.map(r => r[key]).filter(v => v !== undefined && v !== null))).sort();
}

function buildFilterChips() {
  const models = uniq('model_name');
  const gpus = uniq('gpu_type');
  const engines = uniq('engine_name');
  const workloads = uniq('workload_catalog_id');

  makeChips('filter-model', models, filters.models);
  makeChips('filter-gpu', gpus, filters.gpus);
  makeChips('filter-engine', engines, filters.engines);
  makeChips('filter-workload', workloads, filters.workloads);

  // Default: all selected
  models.forEach(m => filters.models.add(m));
  gpus.forEach(m => filters.gpus.add(m));
  engines.forEach(m => filters.engines.add(m));
  workloads.forEach(m => filters.workloads.add(m));
  refreshChips();

  document.getElementById('filter-search').addEventListener('input', e => {
    filters.search = e.target.value.toLowerCase();
    applyFilters();
  });
}

function makeChips(containerId, values, stateSet) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  values.forEach(v => {
    const c = document.createElement('div');
    c.className = 'chip';
    c.textContent = v;
    c.dataset.value = v;
    c.onclick = () => {
      if (stateSet.has(v)) stateSet.delete(v); else stateSet.add(v);
      refreshChips();
      applyFilters();
    };
    el.appendChild(c);
  });
}

function refreshChips() {
  document.querySelectorAll('#filter-model .chip').forEach(c => c.classList.toggle('active', filters.models.has(c.dataset.value)));
  document.querySelectorAll('#filter-gpu .chip').forEach(c => c.classList.toggle('active', filters.gpus.has(c.dataset.value)));
  document.querySelectorAll('#filter-engine .chip').forEach(c => c.classList.toggle('active', filters.engines.has(c.dataset.value)));
  document.querySelectorAll('#filter-workload .chip').forEach(c => c.classList.toggle('active', filters.workloads.has(c.dataset.value)));
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
}

function applyFilters() {
  const rows = DATA.artifacts.filter(r => {
    if (filters.models.size && !filters.models.has(r.model_name)) return false;
    if (filters.gpus.size && !filters.gpus.has(r.gpu_type)) return false;
    if (filters.engines.size && !filters.engines.has(r.engine_name)) return false;
    if (filters.workloads.size && !filters.workloads.has(r.workload_catalog_id)) return false;
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

  renderTable(rows);
  renderSummary(rows);
  drawAggChart(rows);
  drawParetoChart(rows);

  document.querySelectorAll('th').forEach((th, i) => {
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (COLUMNS[i] && COLUMNS[i].key === sortKey) th.classList.add(sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
  });
}

function renderTable(rows) {
  document.getElementById('row-count').textContent = rows.length;
  const body = document.getElementById('table-body');
  body.innerHTML = '';
  rows.forEach(r => {
    const tr = document.createElement('tr');
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
    ctx.strokeStyle = axis; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, H - pad.bottom); ctx.lineTo(W - pad.right, H - pad.bottom); ctx.stroke();
    ctx.fillStyle = txt; ctx.font = '11px Inter'; ctx.textAlign = 'center';
    ctx.fillText('Aggregate tok/s', (pad.left + W - pad.right) / 2, H - 6);
    ctx.save(); ctx.translate(14, (pad.top + H - pad.bottom) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('TPOT p50 (ms/token)', 0, 0); ctx.restore();

    // Group by config → scatter by color
    const groups = groupByConfig(pts);
    const groupEntries = Object.entries(groups);

    groupEntries.forEach(([label, vs]) => {
      const color = colorFor(label);
      ctx.fillStyle = color;
      vs.forEach(r => {
        ctx.beginPath();
        ctx.arc(xPos(r.agg_tok_per_s), yPos(r.tpot_p50_ms), 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = isDark ? '#0f1117' : '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();
      });
    });

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
