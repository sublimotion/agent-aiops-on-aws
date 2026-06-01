# Contributing a serving rule

A rule encodes one hard-won serving fact that can be checked from a *declared*
config (the benchmark.yaml sidecar + optional mdc card). Rules are pure functions
in `resolver/registry.py`. The compiler runs them all and fails CLOSED on any
FAIL. Follow these steps — the conformance test enforces most of them.

## When something belongs here

Add a rule when the fact is **deterministic** (arithmetic or a compatibility
matrix) and **checkable from the declared config** — not a judgment call.

- ✅ `moe_intermediate_size / TP` must be divisible by 128 (arithmetic).
- ✅ B200 NVL5+ needs the AL2023 AMI (compatibility fact).
- ✅ `max_model_len <= max_position_embeddings` (arithmetic).
- ❌ "TP4 is usually faster than TP8 here" — a benchmark result, not a rule.
  That belongs in a blueprint's `lessons.md` and surfaces via the corpus, not a
  hard rule.

If the fact is empirical (it *happened* in a deployment) rather than provable
from config, you don't write a rule — you record it as a `failure_category` in the
blueprint's `lessons.md` frontmatter, and the corpus replays it. Add a new
category → rule mapping in `corpus.CATEGORY_TO_RULE` only once a codified rule
exists to catch it.

## Steps to add a rule

1. **Find the steering source.** The rule must already be documented in
   `.claude/steering/tech-stack.md` (or another steering file). If it isn't,
   document it there first — a rule with no auditable source fails the conformance
   test. Copy the consequence sentence **verbatim**; do not paraphrase. The
   operator should read the identical text in the report and in steering.

2. **Write `_chk_<name>(cfg: ServingConfig) -> Optional[Finding]`** in
   `registry.py`:
   - Return `None` when the rule does not apply to this config (e.g. an FP8-MoE
     rule on a dense bf16 model). Returning None is the common case — most rules
     are silent on most configs.
   - Return a `Finding` with:
     - `rule`: a short stable kebab-case id (e.g. `fp8-moe-tp-divisibility`).
     - `verdict`: `"fail"` (violates a hard rule — must not deploy as-is),
       `"warn"` (likely-wrong, operator must read, deploy may proceed), or
       `"info"` (an applicable note worth surfacing).
     - `reason`: the verbatim consequence from steering.
     - `source`: `f"{TECH_STACK} §'<section title>'"` plus the blueprint lessons
       reference if one exists (e.g. `(qwen3-235b-b300 lessons L1)`).
     - `fix`: the concrete deterministic remediation, when there is one.

3. **Append it to `CHECKS`** at the bottom of `registry.py`. Order: hard fails
   first, then warnings, then info (cosmetic — the compiler runs all of them).

4. **Add a fixture** to `FIXTURES` in
   `resolver/tests/test_serving_conformance.py`: a tuple of
   `(passing_cfg, failing_cfg, expected_failing_verdict)`. The conformance test
   asserts the passing config does NOT fail the rule and the failing config emits
   the expected verdict. A rule without a fixture fails the suite.

   > Note: the test builds `ModelSpec`/`HardwareSpec` directly, which bypasses the
   > keyword-derivation in `from_sidecar`. If your rule keys off `is_mla` /
   > `is_mamba_hybrid` / `is_moe`, set those flags explicitly in the fixture
   > (e.g. `_model(architecture="hybrid-mamba-moe", is_mamba_hybrid=True)`).

5. **If the sidecar lacks a fact your rule needs** (like `moe_intermediate_size`),
   add the field to `ModelSpec`/`EngineSpec` in `model.py` and wire it in
   `from_sidecar` — reading it from the sidecar first, then falling back to the
   `card` dict. The sidecar wins on conflict (it is the deployment's own truth).
   A rule that needs an absent fact should WARN ("cannot verify — add X to the
   sidecar"), never silently pass.

6. **Run the suite:**
   ```bash
   python3 -m unittest discover -s resolver/tests -v
   ```

## Verbatim-reason discipline

The `reason` text is the contract surface. If a steering rule changes (e.g. a new
vLLM version relaxes a constraint), update the `reason` here **and** the citation,
in the same change. The version-refresh protocol (`compound-learner.md`) scans for
stale steering rules; keep registry reasons in lockstep so the report never tells
an operator something steering no longer says.
