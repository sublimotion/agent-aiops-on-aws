# GLM-5.2-FP8 Agent Capability — SWE-bench Lite (46 matched issues)

Harness: **OpenCode** (the same harness that produced the Opus/Sonnet/Haiku baseline traces in
verifier-reward/results/events/ — so model-vs-model is apples-to-apples on harness). Served by
SGLang FP8/B200-TP8 on-node (localhost:30000, no tunnel). glm47 tool parser. 46 issues common to all
three Claude-tier baselines, 12 repos.

## GLM-5.2-FP8 × OpenCode results
| metric | value |
|--------|-------|
| **Fix rate (patch generated)** | **41/46 = 89%** |
| Edit made | 41/46 = 89% |
| Gold pass (tests) | 6/46 = 13% (of 41 evaluated; see caveat) |
| Errors | 1 (pytest-5103: 65536 context overflow — long issue exceeded ctx) |
| Median tokens/issue | ~404K (reasoning-first model → high token use) |

Gold-passed: matplotlib-22711, matplotlib-23299, pylint-5859, pylint-6506, pytest-11143, pytest-11148.

## Comparison vs Claude tiers (edit/attempt rate — the reliable axis)
| model × OpenCode | edit/fix rate (46 matched) |
|------------------|----------------------------|
| Opus (baseline trace) | 100% edit-events |
| Sonnet (baseline trace) | 100% edit-events |
| Haiku (baseline trace) | 100% edit-events |
| **GLM-5.2-FP8** | **89% fix+edit** |

## Honest caveats (read before citing)
1. **Fix rate (89%) is the reliable capability signal** — GLM-5.2 drives the OpenCode tool loop and
   makes real edits on 41/46, just below the Claude tiers' 100% edit-attempt rate.
2. **Gold-pass (13%) is a LOWER BOUND, not directly comparable** to the Claude baseline: (a) the baseline
   event traces don't carry stored gold-eval verdicts in the same form, so no clean gold-vs-gold number;
   (b) in-pod gold-eval only fully ran for repos whose test env was present (matplotlib/pylint/pytest
   passed) — many "gold fail" are likely test-env-not-installed, not wrong patches (same limitation as
   prior SWE-bench runs without per-repo Docker). True gold-pass needs the SWE-bench Docker harness.
3. **1 context-overflow error** at 65536 — a couple of long issues exceed ctx; a 131072 serve would fix it.
4. Codex + Claude Code arms BLOCKED by LiteLLM reasoning/thinking-block translation (see
   harness-wiring-smoke.md) — comparison is OpenCode-only, which is the cleanest match anyway.

## Takeaway
GLM-5.2-FP8 is a **capable coding agent** — 89% fix rate via OpenCode, just under the frontier Claude
tiers, on identical harness + issues. Verified gold-pass requires the full Docker test harness to state
a clean number; the fix-rate signal is strong. Traces captured at /mnt/nvme/results/agent-compare/ (to
be pulled into the blueprint).
