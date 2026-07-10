# trace-effectiveness

Observability over Claude Code agent traces (local JSONL). Phase 1 of the
two-loop observe→improve system in `domains/ai-infra/specs/trace-effectiveness.md`.

## Tools

### `analyze.py` — skill/subagent effectiveness report
Scope-aware dead-skill detection, subagent invocation table, tool error rates,
user-correction topic clustering, token economy, thrash signals.

```bash
python3 analyze.py                    # scan ~/.claude/projects, markdown to stdout
python3 analyze.py --project <substr> # scope to one repo's sessions
python3 analyze.py --json out.json    # + raw stats for visual-explainer
python3 analyze.py --all-plugins      # count other-project plugins as in-scope (noisy)
```

Dead-skill detection is **scope-aware by default**: a plugin skill installed
scoped to a *different* project is not counted as "dead here" (that false
positive inflated the count 3→43). Reads `~/.claude/plugins/installed_plugins.json`.

### `lineage.py` — artifact lineage + consistency-drift audit
Per-session graph of file interactions (read→write→edit, ordered + timestamped)
for the **main agent**, plus drift flags: a file mutated, then a related file
changed later, and never revisited before session end ("forgot to update X").

```bash
python3 lineage.py --project <substr> --last 1   # newest matching session
python3 lineage.py --session PATH.jsonl           # explicit session
python3 lineage.py --session PATH --json out.json # graph+drift JSON
```

## Known limitations

- **Main-agent only.** Subagent-internal file ops are not in JSONL on current
  Claude Code versions (0% sidechain in recent sessions). A file a subagent
  edited won't appear. Deep subagent lineage needs the Phase-2 OTel trace path.
- **`lineage.py` coupling is content-based (heuristic).** A drift edge B→A
  fires only when B's *current content references A* (A's basename; import/stem
  for code). Generic basenames (`README.md`, `main.tf`, `lessons.md`, …) require
  a `parentdir/basename` suffix to match, so we don't couple every README to
  every other. This is direction-aware (index→target order is fine) and ignores
  unrelated new neighbors. Residual limits: a reference via a renamed/aliased
  path won't match; a file changed *outside* the session (not in the trace)
  can't be a drift source. Treat flags as high-signal candidates, still worth an
  eyeball.
- **Token fields subject to Claude Code bug #25941** — treat `analyze.py`
  token economy as order-of-magnitude trend, not exact accounting (carried over
  from the vault `friction.py`).

## Baseline / relationship to friction.py
`analyze.py` is a superset of the vault `06_Metadata/automation/trace-analytics/friction.py`
(keeps its correction + tool-error logic, adds inventory-aware skill/subagent
effectiveness, token economy, thrash-by-session). Keep the correction-detection
regexes in sync between the two.
