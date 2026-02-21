# CSS Patterns and Design Reference

Design system used across all visual-explainer templates. Follow these patterns when populating templates or creating new ones.

## Color Palette

### Dark mode (default)

| Variable | Hex | Usage |
|----------|-----|-------|
| `--bg` | `#0f1117` | Page background |
| `--surface` | `#1a1d2e` | Cards, header, panels |
| `--surface-2` | `#242740` | Table headers, secondary surfaces |
| `--border` | `#2e3151` | All borders and dividers |
| `--text` | `#e2e8f0` | Primary text |
| `--text-muted` | `#8892a4` | Labels, metadata, secondary text |
| `--accent` | `#6366f1` | Interactive elements, active states, sort indicators |
| `--green` | `#22c55e` | PASS status, positive metrics |
| `--red` | `#ef4444` | FAIL status, error states, worst values |
| `--yellow` | `#f59e0b` | PENDING status, warnings |
| `--blue` | `#3b82f6` | Informational highlights |

### Light mode overrides

Only override background and text variables. Status colors stay the same across modes (they desaturate slightly via `-soft` variants).

## Typography

| Use | Font | Size | Weight |
|-----|------|------|--------|
| Body text | Inter | 0.875rem | 400 |
| Labels (uppercase) | Inter | 0.7–0.75rem | 500–600, letter-spacing 0.06–0.08em |
| Card headings | Inter | 1rem–1.1rem | 600 |
| Metric values | JetBrains Mono | 0.82–0.85rem | 400 |
| Large stats | JetBrains Mono | 1.4rem | 600 |
| Page title | Inter | 1.1rem | 600 |

## Status indicators

### Check cards (audit-report.html)

Use `border-left: 3px solid <color>` for category color. Use a circular dot with background from the `-soft` variants:

```html
<div class="check-card pass">   <!-- border-left green -->
  <div class="status-dot">✓</div>
  <div class="check-content">...</div>
</div>
```

Status classes: `pass`, `fail`, `pending`, `skip`

### Verdict banner

Three variants: `pass` (green), `conditional` (yellow), `fail` (red).
Icon recommendations: ✅ for pass, ⚠️ for conditional, ❌ for fail.

## Mermaid configuration

### Initialization (dark mode)

```js
mermaid.initialize({
  startOnLoad: true,
  theme: 'dark',
  themeVariables: {
    primaryColor: '#6366f1',
    primaryTextColor: '#e2e8f0',
    primaryBorderColor: '#2e3151',
    lineColor: '#8892a4',
    background: '#1a1d2e',
    mainBkg: '#1a1d2e',
  }
});
```

### Chart types for AIOps data

**Throughput comparison** → `xychart-beta` bar chart
```
xychart-beta
  title "Throughput by Config (tokens/s)"
  x-axis ["baseline", "lmcache", "dynamo"]
  y-axis "tokens/s" 0 --> 2000
  bar [1240, 1620, 1890]
```

**TTFT/ITL latency** → `xychart-beta` with bar + line (p50 bar, p90 line)
```
xychart-beta
  title "TTFT p50 / p90 (ms)"
  x-axis ["baseline", "lmcache"]
  y-axis "ms" 0 --> 200
  bar [42, 38]
  line [68, 61]
```

**Architecture** → `graph TB` or `graph LR` with subgraphs for VPC/subnet boundaries
```
graph TB
  subgraph VPC["VPC (10.0.0.0/16)"]
    subgraph Private["Private Subnets"]
      ...
    end
  end
```

**AgentCore flow** → `sequenceDiagram` for auth + message flow
```
sequenceDiagram
  Client->>Cognito: InitiateAuth
  Cognito-->>Client: ID Token
  Client->>Proxy: WS connect + Bearer token
  Proxy->>AgentCore: InvokeAgent
  AgentCore-->>Proxy: Response stream
  Proxy-->>Client: WS message
```

## Layout conventions

- Max content width: 1200px (tables) or 1400px (diagrams)
- Page padding: 2rem
- Card border-radius: 10px (cards), 8px (compact cards), 6px (buttons/badges)
- Section spacing: `margin-bottom: 2.5rem`
- Sticky header with `z-index: 100`

## Table conventions

- `border-collapse: collapse` with `border-top` on `td` (not border on `tr`)
- Column headers: uppercase, letter-spacing, `var(--text-muted)` color
- Hover state: `rgba(99,102,241,0.05)` background on `tr:hover td`
- Metric columns: JetBrains Mono, 0.82rem
- Config name column: Inter, 0.85rem, font-weight 500

## Hardware config card (benchmark-comparison.html)

Use CSS Grid `repeat(auto-fit, minmax(160px, 1fr))` for responsive multi-column layout.
Fields: Instance, GPUs, Model, Workload, Concurrency.

## Responsive breakpoints

Templates use CSS Grid `auto-fill` / `auto-fit` with `minmax()` — no explicit breakpoints needed. The grid collapses gracefully on narrow viewports.
