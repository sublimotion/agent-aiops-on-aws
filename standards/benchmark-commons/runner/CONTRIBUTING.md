# Benchmark Runner — Contributor & Agent Guide

This guide is written for **a human or an AI agent extending the benchmark
framework**. It is the navigation map: read this first, then touch code. Every
extension point below is exercised by the conformance test, so if you follow the
recipe and the test passes, your change is correct by construction.

## The one rule

**The runner never guesses.** A workload card declares *what* it wants; the
compiler resolves that to *exactly one* execution path. Anything unmapped
**raises** (`UnsupportedWorkload`) — it never falls back to a degenerate
default. This is deliberate: the framework once silently dropped dataset flags
and produced misleading results (the Kimi K2.6 TTFT-loss incident). Fail closed,
always.

If you find yourself about to hand-write a one-off benchmark script, STOP. Add a
registry handler instead.

## Architecture (who owns what)

```
workloads/*.yaml          # WHAT to measure (declarative cards) — 21 of them
        │
        ▼
runner/registry.py        # (dataset.type, load.type) -> resolution. THE contract.
        │                 #   DATASET_HANDLERS  + LOAD_EXPANDERS
        ▼
runner/compiler.py        # compile_card(card, sidecar, tool) -> ExecutionPlan
        │                 #   PURE function: overrides, sweeps, goodput. No I/O.
        ├──────────────► vendor plan: concrete argv per step
        └──────────────► orchestrated plan: named executor + reason
        │
        ▼
runner/platforms/*.py     # WHERE to run (local / eks / hyperpod). Thin.
runner/orchestrators.py   # bespoke executors for non-vendor cards (when present)
        │
        ▼
runner/adapters/*.py      # vendor raw JSON -> v1 envelope
container/schema/         # the v1 envelope JSON Schema (source of truth for output)
```

**Mental model:** `registry.py` is a lookup table. `compiler.py` is a pure
transform over that table. Platforms and orchestrators are the only things that
do I/O. Keep it that way.

## Decision: is my new card vendor or orchestrated?

A card is **vendor-executable** (runs via `vllm bench serve` /
`sglang.bench_serving`) only if it is a sequence of independent, single-shot
requests with a fixed-ish shape. Use vendor when:
- requests are independent (no conversation state carried server-side),
- the dataset is random / sonnet / sharegpt / prefix_repetition / hf,
- load is constant / sweep / open-loop / poisson / qps-constrained / concurrency-sweep.

A card is **orchestrated** (needs `orchestrators.py`) when it requires logic the
vendor tools cannot express:
- stateful multi-turn sessions (`coding-agent`, `multi-turn-chat`),
- multiple concurrent models / noisy-neighbour rotation (`cohost-isolation`, `mig-partitioning`),
- long sliced soaks with drift analysis (`burn-in`),
- phase-timed probes (`cold-start`),
- non-text modalities needing special payloads (`video-summary`, `transcription-sweep`),
- load shapes derived from a prior measurement (`power-efficiency` ceiling-fraction).

When unsure, grep `registry.py` for the closest existing `dataset.type` and copy
its pattern.

## Recipe: add a new workload card

1. Write `workloads/<id>.yaml`. Reuse an existing `dataset.type` and `load.type`
   if one fits — then **you write zero code**, the compiler already handles it.
2. Run the conformance test:
   ```
   python3 -m unittest discover -s runner/tests -v
   ```
   - If `test_all_cards_resolve` passes → done.
   - If it fails with "dataset.type X has no handler" → go to the next recipe.
3. Bump the `21` count in `tests/test_card_conformance.py::test_all_cards_resolve`
   (`assertEqual(len(CARDS), 21)`) to the new total.
4. Dry-run to eyeball the argv:
   ```
   bash runner/run-benchmark.sh --platform local --endpoint http://x \
     --workload <id> --sidecar <any-sidecar>.yaml --tool vllm --dry-run
   ```

## Recipe: add a new `dataset.type`

Edit `runner/registry.py`, `DATASET_HANDLERS`:

```python
def _ds_myhandler(dataset: dict, tool: str, modality: str) -> DatasetResolution:
    # Read fields off `dataset`; emit vendor flags for `tool`.
    in_len = _mean(dataset.get("input_tokens"), 2048)
    if tool == "vllm":
        argv = ["--dataset-name", "random", "--random-input-len", str(in_len), ...]
    elif tool == "sglang":
        argv = [...]
    else:
        raise UnsupportedWorkload(f"mytype: unknown tool {tool!r}")
    return DatasetResolution(kind="vendor", argv=argv,
                             summary={"type": "mytype", ...})

DATASET_HANDLERS["mytype"] = _ds_myhandler
```

If the type can't be done with vendor tools, register it as orchestrated:

```python
DATASET_HANDLERS["mytype"] = _orchestrated(
    "my_runner", "one-sentence reason it needs bespoke logic", "mytype")
```

Then implement `my_runner(plan, endpoint, model, output, sidecar)` in
`orchestrators.py`. Until you do, the runner exits with a clear message — it does
**not** silently produce a wrong result.

## Recipe: add a new `load.type`

Edit `runner/registry.py`, `LOAD_EXPANDERS`. An expander turns the `load` block
into a list of concrete `LoadStep`s (one per sweep point):

```python
def _lx_mytype(load: dict) -> list[LoadStep]:
    return [LoadStep(label="...", argv=_load_argv("_", request_rate=r, ...),
                     request_rate=r, num_prompts=np) for r in load["rates"]]

LOAD_EXPANDERS["mytype"] = _lx_mytype
```

For a load shape that itself demands orchestration regardless of dataset:

```python
LOAD_EXPANDERS["mytype"] = _lx_orchestrated("my_runner", "why")
```

## Recipe: change the output envelope

The v1 envelope is defined by `container/schema/enriched-artifact.json`. The
adapters (`adapters/vllm.py`, `adapters/sglang.py`) and the Prometheus-first
driver (`../.claude/skills/benchmark-runner/scripts/bench-standard.py`) all emit
it. Change the schema first, then the producers, then re-validate with
`container/validate-artifact.py`. TTFT/TPOT/ITL/E2E are mandatory; use `null`
percentiles, never invented numbers.

## Invariants the conformance test enforces (don't break these)

- Every `dataset.type` in any card has a `DATASET_HANDLERS` entry.
- Every `load.type` in any card has a `LOAD_EXPANDERS` entry.
- Every card compiles for both `vllm` and `sglang` to a vendor plan (with a
  `--dataset-name` flag on every step) or an orchestrated plan (with an executor
  name and a reason).
- Sweep cards expand to the declared number of steps; multi-tier cards
  (`rag-1m-context`) carry a distinct swept value per step.
- The goodput SLO propagates into vendor argv.

## Files at a glance

| File | Purpose | Touch when |
|------|---------|-----------|
| `registry.py` | dataset/load resolution table | new dataset.type or load.type |
| `compiler.py` | pure card→plan transform | new structural sweep axis or override rule |
| `platforms/local.py` | localhost/bare-metal executor | rarely |
| `platforms/eks.py`, `hyperpod.py` | k8s / SSM executors | rarely |
| `orchestrators.py` | bespoke non-vendor executors | implementing an orchestrated card |
| `adapters/*.py` | vendor JSON → v1 envelope | vendor output format changes |
| `tests/test_card_conformance.py` | the CI gate | every card/handler change |
| `../workloads/*.yaml` | the cards | new use case |
| `../container/schema/enriched-artifact.json` | output contract | envelope changes |
