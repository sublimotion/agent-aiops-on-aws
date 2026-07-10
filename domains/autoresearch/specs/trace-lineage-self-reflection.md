# Trace-Lineage Self-Reflection — Harness Reliability Layer Experiment

## Status: DRAFT

> Spec authored from the ai-infra `_template.md` conventions + AGENTS.md structure (the autoresearch `_template.md` was unreadable at authoring time due to a transient sandbox read-block). Reconcile section order against `domains/autoresearch/specs/_template.md` before running.

## Hypothesis

Feeding a **mechanical, trace-derived drift signal** back to an autonomous coding agent mid-run improves cross-artifact consistency, and the improvement **grows with trajectory length**.

Concretely: on seeded tasks that require propagating a change to K coupled sites, an agent that receives injected drift flags ("you edited A; B references A and was never revisited") achieves higher consistency-completion than an agent that does not — with a measurable, non-trivial gap (≥15pp absolute on completion) at long trajectories, and the gap widens vs. short ones.

A hypothesis without a threshold is a wish. Threshold: **≥15pp** consistency-completion lift (reflect vs control) at the long-trajectory tier, **without** a net increase in wall-clock turns-to-done that exceeds the value of the fix (thrash guard, see Measurement).

## Why this matters

As agents run longer and more autonomously, derived representations of history — compaction summaries, `MEMORY.md`, the agent's own recollection — drift from the immutable trace of what actually happened. The agent then acts on stale facts. This is measured, not hypothetical: "context rot" (Chroma 2025) shows recall degrades with context length; "The Illusion of Diminishing Returns" (arXiv:2509.09677, ICLR 2026) shows task success ≈ p^H — ~100% single-step accuracy still collapses below 50% within ~15 dependent steps, worsened by self-conditioning (injecting the model's own prior errors into context). The raw JSONL trace is ground truth; every derived representation is lossy. A lineage index over the trace lets the agent **detect and rectify** consistency breaches it would otherwise carry forward — a **harness reliability layer** whose value is proportional to how autonomous the run is. A positive result unlocks a second prize (out of scope here, see below): JIT fact-retrieval from the trace so context can stay *small*, attacking inference cost at the root.

## Novelty / prior art (why this isn't reinventing the wheel)

Two research passes (2025-2026 landscape) established:
- **The data model exists** — Graphiti/Zep (Apache-2.0, self-hostable) is a bi-temporal knowledge graph grounding every derived fact in a source "episode" with `valid_at`/`invalid_at` invalidation. Evaluate before building any temporal store. We are **not** rebuilding memory storage.
- **The efficiency thesis is Anthropic's own** — the context-engineering guidance (JIT retrieval, context offloading, context rot) validates the small-context direction. Native features (memory tool `memory_20250818`, context-editing `clear_tool_uses_20250919`, compaction, `/rewind`) attack *token pressure*, **not drift** — none diff the summary/MEMORY.md against the raw trace.
- **The self-correction evidence is decisive** — "LLMs Cannot Self-Correct Reasoning Yet" (ICLR 2024, arXiv:2310.01798): *intrinsic* self-correction degrades reasoning. CRITIC (arXiv:2305.11738) and Reflexion (arXiv:2303.11366): *external* feedback works. Our signal is **mechanical external feedback**, not LLM self-judgment — the class that works. This also reconciles our own history: verifier-reward T5 (self-critique in generation HURT, 54%→30%) vs verification-primitives (external checkpoint injection WORKED, 95% fix). This experiment sits on the winning side of that line by construction.

**The unoccupied seam (our contribution):** mechanical drift detection between *derived context* and *immutable trace*, **fed back to the agent** for self-correction, **objectively validated**. Every surveyed provenance-for-agent-memory system (Graphiti, ContextNest arXiv:2607.02116, Episodic-to-Semantic arXiv:2607.01988) stops at *audit for humans* — none close the loop to the agent. No seeded mechanical-oracle benchmark for "propagate a change to all coupled sites" exists — we build it.

## Falsification criteria

Drop / rethink the layer if any of:
1. **No lift:** reflect-informational consistency-completion is within noise (<15pp, CI overlapping) of control at the long tier. The signal doesn't help even when injected.
2. **Net thrash:** reflect-informational improves completion but at a turns/token cost exceeding the value of the fix — i.e. the agent chases false positives more than it fixes real drift (`false_positive_action_rate` high, see Measurement). A reliability layer that costs more than it saves is falsified.
3. **No length scaling:** the reflect-informational−control gap does NOT widen from short→long trajectories. If drift feedback helps equally regardless of run length, the "grows with autonomy" thesis (the core motivation) is wrong even if the tool is mildly useful.
4. **Detector precision floor unmet:** mechanically-scored drift-flag precision (see Measurement) < 0.7 on the seeded corpus. Below this, in-loop feedback trains the agent on noise; stay advisory-only (do not ship the reflect path).

## Design

### Arms (the independent variable = what happens with a drift flag)
| Arm | Behavior |
|-----|----------|
| **control** | No drift detection. Agent runs the task as-is. Baseline. |
| **advisory** | Drift flags surfaced as a `systemMessage` at Stop (human-visible, not injected into the agent loop). Isolates "does surfacing help at all". |
| **reflect-informational** | Drift flags injected back into the agent loop at Stop as *informational context* — "these files reference something that changed and weren't revisited; review whether they're still consistent." Re-enters the loop, but with **neutral, non-coercive** framing. This is the arm modeled on the verification-primitives *winning* condition. |
| **reflect-mandatory** | Same flags, same loop re-entry, but with **coercive** framing — "you MUST reconcile each flagged file before finishing." Included specifically to *test* whether coercion helps or hurts here. |

**Why the two reflect arms (carryover from verification-primitives):** VP found mandatory language *hurt* (40% adoption, lower fix rate) while *informational* checkpoint injection *worked* (83% adoption, first significant gold-pass lift). Baking "forced/MUST" into the reflect arm would repeat that mistake. So the primary self-correction arm is **reflect-informational**; **reflect-mandatory** is a deliberate A/B on framing, not the default. If mandatory ≥ informational here (contradicting VP), that's itself a finding worth recording.

**Single-variable caveat:** advisory (systemMessage, no loop re-entry) vs reflect-* (loop re-entry) differ in *both* visibility and re-entry; reflect-informational vs reflect-mandatory isolate *framing* alone (same re-entry). The clean framing comparison is the reflect pair; advisory is the "surfacing without acting" reference.

### Seeded task corpus + mechanical oracle
- **Task shape:** a small seeded repo where a "source of truth" token appears **verbatim** at K coupled sites (a value in a config, its uses in code, an assertion in a test, a line in a README). Task prompt: "change the token to X." Consistency **objectively requires** propagation to all K sites.
- **Oracle:** `grep` the K sites for the new value. Zero LLM judgment. `consistency_completion = updated_sites / K`. This is the missing benchmark — build it; bootstrap coupling ground-truth from CrossCodeEval-style (arXiv:2310.11248) static cross-file references if we want realism.
- **Reward-hacking guard:** the drift flag says *"B references A which changed"* — it **never reveals the correct value**. Pointer, not answer. The oracle target stays held-out (same discipline as the optimization-loop quality gate). An agent cannot game completion by reading the flag.

### Trajectory-length tiers (tests the scaling claim)
Run each arm at ≥2 tiers: **short** (task alone, ~few turns) and **long**. The primary hypothesis is about the *interaction* (arm × length), not just the arm main effect.

> **Pilot correction (2026-07-06, see blueprint lessons.md):** the v1 synthetic value-propagation task has **no drift headroom** — both Sonnet 4.6 and Haiku 4.5 hit 1.0 completion at every cell (K≤8, short and long), because "change X everywhere" on toy files doesn't compound. Light filler does NOT induce forgetting. **The long tier must use a documented drift-induction protocol**, not hand-rolled filler:
> - **Primary — self-conditioning (arXiv:2509.09677, `github.com/long-horizon-execution/measuring-execution`):** non-thinking model, 100+ dependent steps, controlled injection of its own prior errors. Dials (horizon × injected-error rate) put control in the 30-70% band on demand; the injected errors are exactly what the detector must catch.
> - **Real-code track — PyMigBench (arXiv:2504.13272):** documented per-change 94% → composite 64% (unit-test graded) — ecologically-valid coupled-site propagation, fits value-drift mode.
> Skip LLM-judge substrates (LongMemEval, LoCoMo) — mechanical oracle only.

## Matrix

| Axis | Values |
|------|--------|
| Arm | control, advisory, reflect-informational, reflect-mandatory |
| Trajectory tier | short (no compaction), long (≥1 compaction triggered) |
| Coupling K | small (K=3), larger (K=6–8) — does harder propagation widen the gap |
| Model | **Sonnet 4.6** (`claude-sonnet-4-6`) via Bedrock, held fixed for v1. Chosen for headroom: Opus 4.8 (agent-runner's default) risks a **ceiling effect** — if the model self-corrects without the drift flag, reflect−control collapses to noise (the trinity-coordinator saturation pattern: strong model + task → no headroom → signal invisible). Haiku risks the opposite (task failure noise drowns the drift signal). Sonnet is the altitude where a capable-but-unsaturated agent leaves room for the flag to move the metric. Override agent-runner's Opus default via `ANTHROPIC_MODEL`. **v2 axis:** sweep Haiku/Sonnet/Opus to *measure* where the layer helps most (hypothesis: more on weaker/unsaturated models) — deferred from v1. |
| Trials per cell | ≥20 seeded tasks (mechanical oracle → cheap to scale; power for a 15pp effect) |

State which cells run vs exploratory before launch. Main result = arm × length at fixed K=3, then K as a secondary sweep.

## Baseline

"Off" = the **control** arm: the agent as it runs today, no drift detection, on the identical seeded corpus and identical harness. Advisory and reflect arms differ **only** in what happens to a detected flag — same detector, same tasks, same model, same turn budget. No cross-harness comparison.

## Measurement

- **Primary metric:** `consistency_completion` = mean(updated_sites / K) at task end, per arm × length tier. Mechanical (grep), no judgment.
- **Primary secondary:** `acted_on_stale_rate` — fraction of runs where the final artifacts contain a stale (un-propagated) coupled site. The correctness failure we care about.
- **Thrash / cost guard (go/no-go for reflect):**
  - `false_positive_action_rate` = re-touches of files that were already consistent (cost of imperfect coupling precision). This is the metric that decides whether reflect ships.
  - turns-to-done and total tokens per task, reflect vs control. Lift must not be bought with runaway thrash.
- **Detector precision (falsification #4):** scored **mechanically, not by subjective judgment** — on the seeded corpus the coupled sites are *known ground truth*, so a flag is a true positive iff the flagged file literally still contains the old token (verifiable by grep) AND references a site that changed. This keeps the precision gate on the same mechanical footing as the completion oracle; it does **not** reintroduce the verifier-reward semantic-mismatch problem because "does file B still contain the stale token" is a string check, not a judgment. (For the optional real-repo/LongMemEval extension where ground-truth coupling isn't seeded, precision would need labeling — that is explicitly out of scope for the v1 falsification gate.)
- **Scaling:** the reflect−control completion gap at short vs long tiers (tests hypothesis's length interaction).
- **Sample / stats:** the ≥20/cell figure is a *pilot floor*, not the powered n. **Run a power calc before the full run**: to detect a 15pp difference in a proportion (completion) with 80% power at α=0.05, two-proportion sizing needs ~n≈50–70/arm depending on baseline (VP used n=50–300 for comparable effects — 20 is under-powered for a confident 15pp claim). Pilot at 20 to confirm the effect is *measurable and in the right direction*, then size the confirmatory run from the pilot's observed variance. Report means + bootstrap CIs; the 15pp threshold is on the long-tier reflect-informational−control difference with non-overlapping CIs.
- **Threshold grounding (not arbitrary):** *15pp* — set so the effect must clear VP's demonstrated checkpoint-injection magnitude (VP moved fix-rate double digits); a smaller lift isn't worth shipping an always-on layer. *0.7 precision* — tied to the false-positive-action cost: below ~0.7, >30% of flags are noise the agent acts on, and at typical flag counts per long run that thrash plausibly overwhelms the correctness gain. Both are **pre-registered and revisable from the pilot** — if the pilot shows fp-actions are near-free (agent dismisses false flags cheaply), the precision floor can drop; record the revision, don't move it post-hoc to fit results.
- **Output:** enriched JSON per run (arm, tier, K, completion, stale_rate, fp_actions, turns, tokens, flag precision) under the blueprint `results/`; a summary the `benchmark-analyst`/`visual-explainer` path can render.

## Fixtures / substrate

- Detector: `domains/ai-infra/blueprints/trace-effectiveness/lineage.py` (already built; content-based coupling, drift flags). This experiment **consumes** it — do not duplicate detection logic.
- Injection: the Stop-hook pattern. `.claude/hooks/lineage-drift-check.sh` already emits a `systemMessage` (= **advisory** arm). The **reflect** arms re-enter the loop via a Stop hook returning `decision: block` + `reason` (block = "keep working", per the hooks schema); **reflect-informational** puts neutral review-framing in `reason`, **reflect-mandatory** puts coercive "you MUST" framing in `reason`. Same detector (`lineage.py`), only the `reason` text differs between the two reflect arms.
- Optional temporal-store evaluation: Graphiti (only if the seeded oracle needs fact-versioning beyond file-reference coupling — likely NOT needed for v1).
- Optional external check: LongMemEval (ICLR 2025, arXiv:2410.10813) "knowledge updates" split, to show the result isn't purely our bespoke corpus.

## Execution substrate — agent-runtime (agent-runner on EKS)

This experiment is a strong fit for the detached-agent runtime (`../agent-runner`, `agent-runtime` skill): it's massively parallel (arms × tiers × K × trials = hundreds of independent runs) and needs **only** the run role's existing perms (S3/DynamoDB/ECR/Bedrock/KMS) — the no-infra-perms limit that blocks GPU-serving is a non-issue here. Verified against the repo (2026-07):

- **Launch:** `agent-runner launch <spec> --harness claude-code --cluster <name>`; runs headless `claude -p ... --output-format stream-json --max-turns 200` (default Opus via Bedrock IRSA); outputs to `s3://<bucket>/runs/<id>/{run.log,report.html}` + an `agent-run/<id>` git branch. 24h default deadline → long tier + compaction achievable.
- **TRACE-FORMAT ADAPTER REQUIRED (prerequisite):** agent-runner captures **stream-json** in `run.log` (S3), NOT `~/.claude/projects/**/*.jsonl`. `lineage.py` currently parses the projects-JSONL shape. The file events (tool_use `file_path`) are all present in stream-json (see `agent-runner report.py`), so this is a **parser variant, not a data gap** — add a `--stream-json` input mode to `lineage.py`. This is the one real build prerequisite.
- **Stop-hook feasibility (advisory + reflect):** settings.json hooks load in `-p` mode (repo `.claude/settings.json` is checked out into the worktree). Per Claude Code headless docs, a Stop hook's `decision: block` "prevents Claude from stopping, continues the conversation" and injects `reason`/`additionalContext` — so **reflect arms are feasible headless** (this corrects an earlier over-firm "not supported" read). **One documented unknown:** the `decision: block` × `--max-turns` interaction (does a block extend past the turn cap, or does the cap terminate anyway?). **Gate with a 5-turn smoke test** (`claude -p --max-turns 5` + a blocking Stop hook; confirm it exceeds 5 turns) before committing the reflect arms to a full cluster run.
- **Phasing:** control + advisory are runnable as soon as the stream-json adapter lands; reflect arms after the smoke test passes.

## Rule the experiment would produce

If the hypothesis holds:
> Ship the trace-lineage reliability layer with the **reflect** path enabled for long autonomous runs (RALPH loops), advisory-only below a trajectory-length threshold. Detector precision must stay ≥ 0.7 (monitored). Steering note in `project-structure.md`/`tech-stack.md`: "Long autonomous runs enable in-loop drift reflection; the drift flag is external mechanical feedback (per arXiv:2310.01798, do not replace with LLM self-judgment)."

If falsified: keep the detector as a **human-facing advisory** (the Stop-hook systemMessage already built) and record why in-loop feedback didn't pay.

## Out of scope

- **JIT context-shrink / inference-efficiency** (the "small context + retrieval" prize). This experiment establishes the **precision + correctness precondition** first; the efficiency study (tokens-per-task at equal quality via trace retrieval instead of fat context) is a **follow-on spec**, gated on this one's detector-precision result.
- Building a memory graph / temporal store (evaluate Graphiti; don't rebuild).
- Compaction-boundary re-grounding of MEMORY.md as a *mechanism* (the narrower MEMORY.md-integrity study) — a sibling spec; this one uses the general coupled-site oracle.
- Subagent-internal lineage (needs the OTel path; JSONL is main-agent only).
- LLM-as-judge scoring anywhere (mechanical oracle only — avoids the verifier-reward semantic-mismatch recall ceiling).

## Carryover audit (spec-design gate)

- [ ] Run `carryover-auditor` before executing.
- [x] Reward-hacking / sentiment-as-reward → sycophancy lessons: honored — no reward-shaped-by-model-feedback loop; the oracle is mechanical and held-out; the flag never reveals the answer.
- [x] verifier-reward T5 (self-critique in generation HURTS) + verification-primitives (external checkpoint WORKS): this experiment is designed on the winning side — external mechanical signal, injected at a checkpoint, not intrinsic self-critique. Cited in Novelty.
- [x] Token-accounting bug #25941: token cost is a *secondary* metric here; treat as order-of-magnitude (carried from `friction.py`/`trace-effectiveness`).
- [ ] Local sklearn / python3.13 lesson: applies only if analysis uses sklearn; note at run time.

## Cost estimate

Cheap by design: seeded tasks are tiny, the oracle is grep, and the model can be Haiku or the local vLLM endpoint. Dominant cost = number of cells × trials × turn budget × (long-tier compaction overhead). Cap before launch; estimate ≪ the verifier-reward experiment. The long tier costs more per run (must reach compaction) — budget accordingly.

## References

- Detector + prior blueprint: `domains/ai-infra/blueprints/trace-effectiveness/` (analyze.py, lineage.py, README.md, spec `domains/ai-infra/specs/trace-effectiveness.md`).
- Self-correction: arXiv:2310.01798 (LLMs Cannot Self-Correct Reasoning Yet, ICLR 2024); arXiv:2305.11738 (CRITIC); arXiv:2303.11366 (Reflexion).
- Provenance-for-agent-memory (audit-only prior art we differentiate from): ContextNest arXiv:2607.02116; Episodic-to-Semantic arXiv:2607.01988.
- Long-horizon drift (measured, reproducible): "The Illusion of Diminishing Returns" arXiv:2509.09677 (ICLR 2026, success≈p^H, self-conditioning, public code github.com/long-horizon-execution/measuring-execution — the recommended drift-induction protocol); Context Rot (Chroma, 2025, github.com/chroma-core/context-rot); "LLMs Get Lost in Multi-Turn" arXiv:2505.06120 (instruction-sharding, ~39% drop); Lost in the Middle arXiv:2307.03172. (NOTE: arXiv:2602.06413 previously cited here is theory-only with no code — replaced by the empirical 2509.09677.)
- Drift-headroom benchmarks for substrate reuse (mechanically graded): τ-bench arXiv:2406.12045 (built-in `pass^k` drift metric, DB-state oracle, pass^8<25% retail); PyMigBench LLM-migration arXiv:2504.13272 (real code coupled-site propagation, per-change 94% → composite 64%); Aider refactor benchmark (AST-graded partial-edit detector); MQuAKE arXiv:2305.14795 (multi-hop edit propagation → 7-8%, knowledge analog). Skip LLM-judge-graded evals (LongMemEval, LoCoMo) to keep the oracle mechanical.
- Memory-consistency benchmarks: LongMemEval arXiv:2410.10813 (ICLR 2025); LoCoMo arXiv:2402.17753. Cross-file coupling: CrossCodeEval arXiv:2310.11248.
- Temporal-memory data model (build-vs-borrow): Graphiti — github.com/getzep/graphiti (Apache-2.0).
- Native context features: Anthropic context-engineering (anthropic.com/engineering/effective-context-engineering-for-ai-agents); memory tool + context-editing (platform.claude.com/docs/en/docs/build-with-claude/context-editing).
- Related internal memory: reward-hacking notes, verifier-reward + verification-primitives blueprints.
