# serving-commons

A deterministic, fail-closed resolver for LLM **serving configs** — the
serving-side analog of `benchmark-commons`. It answers one question before a
deployment burns capacity: *given this model, engine, and hardware, is this
config known to break?*

```
                 ┌─────────────────────────────────────────────┐
   benchmark.yaml│  model.py    parse sidecar → ServingConfig    │  (pure)
   (+ mdc card) ─┤  registry.py deterministic rule table (CHECKS)│  (pure)
                 │  compiler.py run all rules → ValidationReport │  (pure)
                 │  corpus.py   harvest blueprint lessons.md     │  (I/O)
                 └─────────────────────────────────────────────┘
                                  │
                                  ▼
              validate-serving-config.py  (CLI gate, exit≠0 on FAIL)
```

## Why this exists

Specs say *what to deploy* ("Qwen3-235B FP8, TP8"). Blueprints record *what
actually happened* ("TP8 fails the FP8 block-size check; TP4 works"). That delta
is hard-won knowledge that currently lives only in prose. This package codifies
it two ways:

1. **Static rules** (`registry.py`) — deterministic arithmetic / compatibility
   facts lifted verbatim from `.claude/steering/tech-stack.md`. A FAIL blocks the
   deploy. An LLM cannot reinterpret these.
2. **Empirical corpus** (`corpus.py`) — harvests `failure_categories` from every
   blueprint's `lessons.md` field-note frontmatter and replays them: "a prior
   deployment of this model/engine recorded failure_category=X — see blueprint Y."

The compiler is **PURE** (no I/O, no network, no GPU): the same config always
yields the same report. `corpus.py` is the only I/O boundary; it loads the corpus
once and hands the pure compiler a frozen snapshot.

## Fail-closed contract

If any check returns `verdict="fail"`, `report.ok` is `False` and
`raise_if_invalid()` raises `InvalidServingConfig` with every failure and its
fix. Callers that gate a deployment must refuse to proceed. This mirrors
benchmark-commons' `compile_card` raising `UnsupportedWorkload` rather than
silently degrading. Warnings and info never block — they surface the caveat so an
operator (or an agent reading the report) sees it without having to remember it.

## Usage

```bash
# Gate a blueprint's serving config before capacity reservation / Stage 0.
python3 resolver/validate-serving-config.py \
  --sidecar domains/gpu-serving/blueprints/<name>/benchmark.yaml \
  --card    <model-deployment-card>.json \      # optional: supplies moe_intermediate_size etc.
  --corpus-root .                               # optional: harvest prior lessons.md failures

# Treat warnings as blocking too (stricter pre-prod gate):
python3 resolver/validate-serving-config.py --sidecar ... --warnings-as-errors

# Machine-readable:
python3 resolver/validate-serving-config.py --sidecar ... --json
```

Exit codes: `0` clean, `2` hard-rule FAIL, `3` warnings-as-errors triggered.

### From Python

```python
from compiler import validate_sidecar
from corpus import load_corpus

corpus = load_corpus(".")                       # harvest blueprint field notes
report = validate_sidecar(sidecar_dict, card=card_dict, corpus=corpus)
report.raise_if_invalid()                       # fail closed
```

## The corpus ↔ rule link

`corpus.CATEGORY_TO_RULE` maps each `failure_category` (recorded in lessons.md) to
the deterministic rule that would now catch it — e.g.
`fp8_block_size_mismatch → fp8-moe-tp-divisibility`. A recorded category with a
codified rule prompts "confirm that check fired"; one without (e.g. `nccl`, `oom`)
is surfaced as a bare warning because no deterministic check guards it yet. That
gap list is the backlog of rules worth codifying next.

## Tests

```bash
python3 -m unittest discover -s resolver/tests -v
```

The conformance suite asserts: every rule cites a source; every rule has a
passing + failing fixture; the canonical Qwen3-235B TP8-fails/TP4-works case;
every real `benchmark.yaml` in the repo validates (or is waived); and the corpus
harvests and replays blueprint lessons.

See `CONTRIBUTING.md` to add a rule.
