# Agent Trace Effectiveness — Skill/Subagent Observability

## Status: RUNNING

Phase 1 (JSONL effectiveness report) is built and running. Phases 2–3 (OTel pipe, verifier-as-online-scorer, ADP export) are designed but not built.

## Hypothesis

We can measure how well our agentic *framework* (skills + subagents + tools) is working directly from Claude Code trace data, and the measurement will reveal concrete, actionable waste: dead skills the agent never selects, subagents that dominate delegation, tool classes with pathological error rates, and recurring correction topics that map to skill gaps.

Threshold for "useful": the report identifies ≥1 dead skill worth deleting/retriggering, ≥1 tool with an error rate >10% worth fixing, and clusters user corrections into a topic distribution — all from local JSONL with zero infra.

## Falsification criteria

- Falsified as a *standalone JSONL tool* if the signals it needs (skill invocations, subagent spawns, tool errors, corrections) are too sparse or too noisy to act on — e.g. if correction clustering is dominated by false positives (synthetic subagent prompts) to the point of being misleading.
- The **subagent-trajectory** ambition is already falsified for JSONL: current Claude Code versions write ~0 sidechain lines, so per-subagent token cost and internal tool-use quality are **not** recoverable from JSONL. That capability is deferred to the OTel path (Phase 2), not claimed here.

## Why this matters

We author skills, subagents, and steering rules but have no feedback on whether the agent actually *uses* them or whether they help. This is the observability gap for the framework itself, distinct from GPU/serving observability. A dead skill is wasted authoring effort and prompt-context bloat; a high-error tool is silent thrash; a correction cluster is a soft negative-reward signal pointing at a missing guardrail. Closing this loop lets us prune, retrigger, and eventually feed graded trajectories back into training (Phase 3, ADP).

## The two-loop architecture (design context)

This blueprint is Phase 1 of a larger observe→improve system. Three schemas, three jobs, no conflict:

| Schema | Job | Standard | Phase |
|---|---|---|---|
| OpenTelemetry GenAI semconv (`gen_ai.*`, `mcp.*`, `invoke_agent`/`execute_tool`) | **Capture** live telemetry | CNCF, `Development` stage (pin version) | 2 |
| OpenInference / Opik span kinds (AGENT/TOOL/EVALUATOR) | **Analyze + score** traces | Apache-2.0, self-hostable | 2 |
| Agent Data Protocol (ADP, arXiv 2510.24702) | **Train** — SFT/RL-ready trajectories (Action: API/Code/Message + Observation: Text/Web) | released code+data | 3 |

```
OFFLINE loop (optimize framework)         ONLINE loop (observe live inference)
──────────────────────────────────       ────────────────────────────────────
trace corpus → golden/regression set  ←→  live agent+serving traces (OTel gen_ai.*)
  ↓ LLM-judge + code metrics                ↓ verifier (v009, prec 0.92) as online scorer
skill/subagent routing eval                 ↓ sentiment/frustration = SOFT signal only
  ↓ regression gate before merge            ↓ dashboards + tripwires
improve skills / agent defs               → graded winners → ADP → SFT corpus → back
```

Key design decision: **buy the pipe, build the judge.** No vendor answers "which skills/subagents are effective." Our calibrated verifier is the differentiator; sentiment/frustration is a diagnostic tripwire correlated against verifier ground-truth, **never a direct optimization target** (reward-hacking/sycophancy risk — arXiv 2402.06627; see `memory/` reward-hacking notes).

## Stage-budget claim

N/A — this is a trace-analytics experiment, not a cold-start/serving technique. The cold-start stage budget does not apply.

## Matrix

| Axis | Values |
|------|--------|
| Trace source | local JSONL (`~/.claude/projects`) — Phase 1 |
| Scope filter | all projects, or `--project <substr>` for one repo |
| Report dimensions | token economy, skill adoption+dead-skills, subagent invocations, tool error rates, correction topics, thrash signals |

## Baseline

"Off" = the existing `friction.py` (vault `06_Metadata/automation/trace-analytics/`), which reports corrections, tool errors, and retry loops but has **no skill/subagent/dead-skill/token analysis**. This blueprint's `analyze.py` is a superset: it keeps friction.py's correction + tool-error logic (kept in sync) and adds inventory-aware skill/subagent effectiveness + token economy + thrash-by-session.

## Measurement

- **Tool**: `analyze.py` (this blueprint). Pure stdlib Python 3, no deps.
- **Inventory discovery**: scans `.claude/skills` (project + user + plugins) and `.claude/agents` to know what *could* be invoked, so it can flag never-invoked = dead.
- **Primary output**: markdown report (skill adoption %, dead-skill list, subagent spawn table, tool error-rate table with normalized patterns, correction topic distribution, thrash signals).
- **Secondary**: `--json` dump for downstream/visual-explainer.
- **Sample**: all sessions in root (~309 at time of writing). No sampling.

### Findings from first run (309 sessions, 2026-07-02)

- **Skills are barely used**: 29 invocations, only 7% of sessions used any skill. `visual-explainer` (8) dominates. Dead-skill count is **scope-dependent**: 43 of 47 with `--all-plugins` (counts other projects' plugins), but only **3 of 6** in-scope for this repo (`benchmark-runner`, `deployment-orchestrator`, `terraform-automation`) — and those 3 are load-bearing consistency contracts wired into AGENTS.md, kept deliberately. The scary "43" was a measurement artifact; `analyze.py` is scope-aware by default.
- **Subagents carry delegation**: 133 spawns vs 29 skill invocations. `general-purpose` (57), `carryover-auditor` (21), `infra-deployer` (21). 4 project agents never spawned (`agentcore-deployer`, `autoresearch-runner`, `benchmark-analyst`, `blueprint-reviewer`).
- **Tool error hotspots**: `AskUserQuestion` 16.0% (mostly user-rejected, not tool failure — expected), `Edit` 3.4% (38× "file not read yet", 9× "No such tool: Edit" from deferred-tool loading, 7× "modified since read"), `Read` 2.1% (28× "file does not exist"). MCP tools with 100% rate are low-N (`sift_insights_fetchById` 14/14).
- **Corrections**: 153 turns, 89 "read-before-acting" — **but** top samples are synthetic subagent prompts leaking the filter (known limitation, see below).
- **Thrash**: 80 consecutive same-tool error loops. Highest-token sessions are fintech-agent (4.1M out-tok) and this repo.

## Known limitations

1. **Subagent internals invisible in JSONL** — deferred to OTel (Phase 2). Section 2 is invocation-level only.
2. **Correction false positives** — synthetic subagent/system prompts (e.g. "You are summarizing a Claude Code session… Read the …") match `read-before-acting` patterns. The `is_synthetic()` filter catches `<`-prefixed and short turns but not injected agent prompts. Fix: also skip user turns that are sidechain-origin or match known agent-prompt preambles.
3. **`AskUserQuestion` "errors" are mostly user rejections**, not failures — treat that row as a signal of plan/direction mismatches, not tool bugs.
4. **Token-per-session** attributes all output tokens to the session's main project dir; subagent tokens fold into the parent.
5. **Token-accounting bug (#25941)** — JSONL `usage` fields are subject to the same Claude Code token bug that `friction.py` deliberately steers around (its signals use only bug-immune fields). `analyze.py` reports a token economy section anyway, flagged in-report as order-of-magnitude trend, not exact accounting. The skill/subagent/error/correction signals are unaffected. Carried over from `friction.py` docstring.

## Rule the experiment would produce

Not a steering *tech* rule — an **operational cadence rule** for the framework:

> Run `domains/ai-infra/blueprints/trace-effectiveness/analyze.py` on a periodic cadence (e.g. weekly, via the vault launchd runner). Act on: dead skills (delete or fix trigger description), tools >10% error rate (root-cause), and the top correction topic (add a steering guardrail). Feed the token/thrash outliers to context-hygiene review.

Longer term (Phase 3): high-verifier-score trajectories export to ADP format as SFT data — codified when the OTel pipe + verifier scorer land.

## Out of scope

- OTel pipe, Opik/Phoenix self-host, verifier-as-online-scorer (Phase 2 — separate PR).
- ADP export / training-data generation (Phase 3).
- GPU/serving inference observability (DCGM/vLLM via `gpu-infra`) — pairs with this on the same OTel backbone but is a separate concern.
- Sentiment scoring as a *reward* (explicitly rejected; tripwire only).

## Carryover audit (spec-design gate)

- [x] This is a trace-analytics tool, not a GPU/serving deploy — no `domains/**/lessons.md` stack overlap applies. The relevant prior knowledge is `memory/` reward-hacking notes (sentiment-as-reward → sycophancy), reflected in the "soft signal only" design constraint.

## Cost estimate

Phase 1: ~$0 (local stdlib script over existing JSONL). Phase 2 (OTel→Opik self-host): compute for a small always-on container + LLM-judge calls on sampled traces (author of the EDD methodology caps online eval ~$2k/mo — we sample, not all-traffic). Phase 3: SFT compute, separate budget.

## References

- Agent Data Protocol (ADP) — arXiv 2510.24702 (training-data interlingua; 13 datasets unified; SWE-Bench Verified 2.2%→40.3% on 32B via SFT).
- OpenTelemetry GenAI semconv — `github.com/open-telemetry/semantic-conventions-genai` (agent spans, `mcp.*`, Anthropic attributes).
- Claude Code telemetry — `code.claude.com/docs/en/monitoring-usage` (`CLAUDE_CODE_ENABLE_TELEMETRY`, `CLAUDE_CODE_ENHANCED_TELEMETRY_BETA`, `skill_activated`/`tool_decision` events).
- Opik — `github.com/comet-ml/opik` (Apache-2.0; Agent Tool Correctness, Trajectory Accuracy, User Frustration judges; online scoring rules).
- Evaluation-Driven Development — decodingai.com/p/how-evaluation-driven-development-works (offline eval-before-merge loop; simulate inputs not outputs).
- Reward-hacking risk of sentiment-as-reward — arXiv 2402.06627; Anthropic reward-hacking→sycophancy.
- Baseline: vault `06_Metadata/automation/trace-analytics/friction.py`.
