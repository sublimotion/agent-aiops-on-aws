# Visual Explainer Troubleshooting

## Common Issues

### Template placeholder not replaced

**Error**: Output HTML shows `<!-- DATA: ... -->` comments instead of actual data.
**Cause**: The skill read the template but didn't substitute all placeholders.
**Solution**: After populating, grep the output file for `<!-- DATA:` — if any remain, re-read the source data and fill them. Every `<!-- DATA: ... -->` comment must be replaced with actual HTML content.

### Mermaid chart not rendering

**Error**: Blank area where chart should appear, or raw mermaid syntax visible.
**Cause**: Mermaid.js CDN failed to load, or syntax error in the diagram spec.
**Solution**:
1. Check that the HTML includes `<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>`
2. Validate mermaid syntax at https://mermaid.live — common issues: unquoted labels with special chars, missing axis bounds in `xychart-beta`
3. For `xychart-beta` bar charts, ensure x-axis labels and bar values have the same count

### Dark/light toggle not working

**Error**: Toggle button exists but clicking doesn't change colors.
**Cause**: CSS custom properties not defined for both modes.
**Solution**: Verify `[data-theme="light"]` overrides exist in the `<style>` block. The toggle JS must set `document.documentElement.setAttribute('data-theme', ...)`.

### Table sorting breaks after data update

**Error**: Clicking column headers sorts incorrectly or does nothing.
**Cause**: Sort comparator doesn't handle the metric format (e.g., "1,240 tok/s").
**Solution**: The sort function must strip units and parse numbers. For metric columns, use: `parseFloat(cell.replace(/[^0-9.-]/g, ''))`.

### Benchmark data missing configs

**Error**: HTML report has fewer rows than the markdown report.
**Cause**: Config names in benchmark JSON don't match the names extracted from markdown.
**Solution**: Cross-reference `SERVING_CONFIGS` in `run-benchmarks.py` with the configs listed in `results/benchmark-report.md`. Use exact config names as row identifiers.

### Output file not opening in browser

**Error**: `open` command fails or opens wrong application.
**Cause**: macOS `open` command not available (Linux), or file extension not associated with browser.
**Solution**: Use `open <path>` on macOS, `xdg-open <path>` on Linux. Verify the file has `.html` extension.
