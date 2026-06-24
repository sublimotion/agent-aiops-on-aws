# GLM-5.2 Agent-Trace Analysis vs Opus / Sonnet / Haiku

Same harness (OpenCode), same 46 SWE-bench Lite issues. GLM-5.2 from the run summary (per-issue
metrics); Claude tiers from the full event JSONLs (verifier-reward/results/events/, matched-46 only).

> **Data caveat (operator error, not a capture gap)**: the driver DID write per-issue event JSONLs
> (`_save_trace()` → `opencode_glm52_<id>.jsonl` on the node's NVMe, with OpenCode `tool_use` events
> carrying `part.tool` = the bash/read/edit/grep type). But at teardown I pulled only `summary_glm52.json`
> and committed that as if it were the complete artifact set — the raw JSONLs were **never exfiltrated by
> ANY path**: not committed to git, no local copy on the laptop, and not in S3 (the node's IAM role lacked
> S3 write — kimi L10 — and the runner had no S3-sync step). Scaling the node to 0 wiped the instance-store
> NVMe. **Verified dead: git ✗, laptop fs ✗, all bench S3 buckets ✗.** So the summary metrics
> (effort/token/outcome) are clean, but **tool-type mix, turns-to-first-edit, exploration-vs-edit sequencing,
> and error-recovery patterns are permanently lost — re-runnable only, not recoverable.** Root cause: no
> durable sink existed for this run (ephemeral NVMe + no S3 write + raw traces never committed). See the
> teardown-artifact-durability rule added to `.claude/steering/tech-stack.md`.

## 1. Effort efficiency — GLM-5.2 ≈ Opus, more efficient than Haiku
| model | median turns | median tool calls | edit rate |
|-------|-------------|-------------------|-----------|
| Sonnet | 16 | 15 | 46/46 (100%) |
| Opus | 21 | 21 | 46/46 (100%) |
| **GLM-5.2** | **22** | **22** | **41/46 (89%)** |
| Haiku | 25 | 24 | 31/46 (67%) |

GLM-5.2's tool-loop efficiency is essentially **Opus-class** (22 vs 21 turns) — it resolves the agent
loop in a comparable number of steps, well clear of Haiku's 25. Sonnet is the most economical (16). The
89% edit rate (vs Opus/Sonnet 100%) is the gap: on 5/46 it never lands an edit.

## 2. Effort-vs-outcome — GLM-5.2 FAILS FAST, doesn't thrash
| | median turns |
|--|--|
| issues GLM-5.2 fixed | **23** |
| issues GLM-5.2 did NOT fix | **13** |
Failed attempts are SHORTER, not longer — GLM-5.2 gives up / hits a wall early rather than burning turns
going in circles. This is a healthy signal (no pathological loops); the not-fixed cases are early
bailouts (incl. 1 context-overflow error), not exhaustion.

## 3. Token economy — the reasoning-first tax
| metric (median/issue) | value |
|-----------------------|-------|
| input tokens | 396,687 |
| output tokens | 6,851 |
| **total** | **403,538** |
| wall-clock/issue | 98s |

GLM-5.2 is **reasoning-first** → ~404K tokens/issue, dominated by input (context re-sent each turn). The
harness recorded 0 client-side cache_read, but SGLang's **server-side radix cache absorbed it** (our
serving sweep measured 92% prefix-cache hit) — so the *served* cost is far below the raw token count.
Output is small (6.8K) — the model reasons heavily then emits concise edits. For reference, our internal
Codex×Devstral runs hit ~1.4M tokens/issue (140× OpenCode); GLM-5.2's 404K is mid-pack for a reasoning model.

## 4. What this says about GLM-5.2 as an agent
- **Capability**: drives the tool loop like a frontier model (Opus-class turn count), 89% fix rate.
- **Discipline**: fails fast (13 vs 23 turns) — no thrashing; the 11% non-fix is early bailout.
- **Cost**: reasoning-first token tax (~404K/issue) is the price; server-side prefix caching is essential
  to make it economical (92% hit → most of the 397K input is cache-served, not recomputed).

## 5. Follow-up that needs trace re-capture (node down)
Re-run capturing per-issue event JSONLs (and, per the LiteLLM correction, all 3 harnesses via **native**
SGLang `/v1/messages` + `/v1/responses`) to unlock:
- **Tool-type mix** (bash/read/edit/grep ratio) vs the Claude tiers — exploration-vs-action profile.
- **Turns-to-first-edit** (Parkinson's-Law-for-agents axis from our agent-harness work).
- **Error-recovery patterns** — how GLM-5.2 reacts to failed tool calls.
- Gold-pass under a full Docker test harness (current 13% is a test-env-limited lower bound).
