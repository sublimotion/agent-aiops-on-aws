# Visual Explainer Skill

Converts dense AIOps terminal output and benchmark data into self-contained, interactive HTML pages. No build step required — all output is single-file HTML with CDN-only dependencies.

## Commands

| Command | Input | Output |
|---------|-------|--------|
| `/generate-web-diagram` | Mermaid diagram spec or architecture description | Interactive diagram HTML |
| `/benchmark-visual` | `results/benchmark-report.md` or raw JSON results | Sortable comparison table + bar chart HTML |
| `/audit-visual` | Readiness audit markdown | Color-coded PASS/FAIL status grid HTML |
| `/compound-recap` | Compound summary markdown | Visual compound summary HTML |
| `/fact-check` | Blueprint review report markdown | Structured findings HTML |

## Workflow

1. **Read the source** — Read the input file (benchmark report, audit, compound summary, or review).
2. **Select template** — Choose the appropriate template from `.claude/skills/visual-explainer/templates/`.
3. **Populate the template** — Replace all `<!-- DATA: ... -->` placeholders with actual data. For Mermaid charts, generate the diagram spec from the data.
4. **Write the output** — Save to `results/<output-name>-<date>.html` alongside the source markdown.
5. **Open in browser** — Run `open <path>` (macOS) or `xdg-open <path>` (Linux) to display immediately.

## Template selection

| Command | Template |
|---------|----------|
| `/benchmark-visual` | `templates/benchmark-comparison.html` |
| `/generate-web-diagram` or architecture | `templates/architecture.html` |
| `/audit-visual` or `/fact-check` | `templates/audit-report.html` |
| `/compound-recap` | `templates/audit-report.html` (adapted) |

## Output naming

```
results/benchmark-visual-<YYYYMMDD>.html
results/audit-visual-<YYYYMMDD>.html
results/architecture-<name>-<YYYYMMDD>.html
results/compound-recap-<YYYYMMDD>.html
```

## AIOps use cases

### benchmark-analyst → `/benchmark-visual`

After `benchmark-analyst` writes `results/benchmark-report.md`:
- Extract TTFT (p50/p90/p99), ITL (p50/p90/p99), throughput, error rate per config
- Populate benchmark-comparison.html: sortable table rows = configs, columns = metrics
- Generate Mermaid bar chart for throughput comparison
- Include hardware config header (instance type, GPU count, model)

### blueprint-reviewer → `/fact-check`

After blueprint-reviewer produces its findings:
- Map PASS/FAIL/PENDING items to audit-report.html status grid
- Color-code by severity (FAIL = red, PASS = green, PENDING = yellow)
- Render action items table with file paths and line numbers

### compound-learner → `/compound-recap`

After compound-learner writes `results/compound-<date>.md`:
- Show elevated vs. kept-local lessons as a two-panel summary
- Highlight which steering files were updated

## Design constraints

- All templates are self-contained (no external runtime, no build step)
- CDN dependencies: Mermaid.js (jsdelivr), Inter font (Google Fonts)
- Dark/light mode toggle using CSS custom properties
- Mobile-responsive layout
- Accessible color palette (WCAG AA contrast ratios)
