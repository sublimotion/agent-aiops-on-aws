# trace-lineage-self-reflection

Experiment harness for the drift-detection **harness reliability layer**: does
feeding a mechanical trace-drift signal back to an autonomous agent improve
cross-artifact consistency? Spec: `domains/autoresearch/specs/trace-lineage-self-reflection.md`.

**Status: harness BUILT + VALIDATED; experiment BLOCKED on drift headroom.** See
`lessons.md` — neither Sonnet 4.6 nor Haiku 4.5 naturally drift on the current
seeded tasks, so there is nothing for the reflect arm to fix. Do not fan out to
the cluster until a drifting task regime is constructed.

## Pieces

| File | Role |
|------|------|
| `scripts/gen_task.py` | Generates a seeded repo where a value token appears at K coupled sites + a `task.json` manifest. Deterministic (seed-indexed, no RNG). |
| `scripts/grade.py` | **Mechanical grep oracle.** `consistency_completion = updated_sites / K`; flags `acted_on_stale`. Zero LLM judgment. |
| `scripts/run_cell.sh` | Runs one cell: gen task → headless `claude -p` under an arm's Stop hook → grade → append result row. |
| `hooks/reflect-hook.sh` | The arm behaviors (`DRIFT_ARM`): `control` (off), `advisory` (systemMessage), `reflect-informational` (block + neutral reason), `reflect-mandatory` (block + coercive reason). Value-drift only by default. |

Detector is `domains/ai-infra/blueprints/trace-effectiveness/lineage.py` (consumed, not duplicated). This experiment added **value-drift** to it (`detect_value_drift`, `--no-reference-drift`, `--stream-json`).

## Run a cell (local)

```bash
export ANTHROPIC_MODEL="us.anthropic.claude-sonnet-4-6"   # inference-profile ID; bare alias is rejected under Bedrock
bash scripts/run_cell.sh --arm reflect-informational --seed 2 --k 6 --tier short --max-turns 60 --out results/
cat results/results.jsonl        # one JSON row per run
```

Arms: `control` | `advisory` | `reflect-informational` | `reflect-mandatory`.
Tiers: `short` | `long`. K: 2..8.

## Validated (pilot, 2026-07-06)

- End-to-end pipeline runs; `decision: block` **does** continue a headless agent (spec's open unknown — resolved YES).
- `lineage.py --stream-json` parses agent-runner `run.log` (the cluster path).
- **Value-drift is precise**: 0 false positives on a completed task, 2/2 true positives on an incomplete one.
- Reference-drift false-positives on completed value-tasks → experiment uses `--no-reference-drift`.

## The blocker (why it's not launched)

Control-arm completion is **1.0 at every tested cell** (K=3/6/8, short and long, both Sonnet and Haiku). No natural drift = no headroom = reflect cannot show a lift. The current "long tier" never reaches compaction (tasks finish in 26–46 turns). **Next step: construct a genuinely long / implicitly-coupled task where the control agent forgets** — see `lessons.md` for the lever list. Launching to the cluster before that would measure a null effect.

## Cluster path (when headroom exists)

`agent-runner launch <this-spec> --harness claude-code` — the run role's S3/DynamoDB/ECR/Bedrock/KMS perms suffice (no infra perms needed). Traces land as `s3://<bucket>/runs/<id>/run.log` (stream-json → use `lineage.py --stream-json`).
