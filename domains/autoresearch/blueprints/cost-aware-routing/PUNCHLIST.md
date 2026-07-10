# Phase 1 Launch — Punchlist

What's left for the operator. Costs in USD; commands assume cwd is the blueprint dir.

## Status snapshot (2026-05-25)

| Item | Status | Cost |
|------|:------:|------|
| Pool YAML verified via AWS Pricing API | ✅ | — |
| Reward function + extractors + graders (89 unit tests) | ✅ | — |
| Reward landscape gate (min adjacent gap 0.0102) | ✅ | — |
| CheckpointManager (RNG + S3 + resume) | ✅ | — |
| Worker proxy (Anthropic-thinking-block aware) | ✅ | — |
| `metadata_prompt.py` neutral codes router prompt | ✅ | — |
| Judge (Haiku-as-judge wrapper + Gate 0.3 runner) | ✅ | — |
| Phase 0 gate orchestrator (`run_gates.py`) | ✅ | — |
| Pareto eval harness (`pareto_eval.py`) | ✅ | — |
| Data build script (12 datasets, license-gated) | ✅ | — |
| `bootstrap.sh` for p5 spot provisioning | ✅ | — |
| All 14 scripts compile cleanly | ✅ | — |
| **Dataset fetch (HF download)** | 🟡 in progress | $0 |
| Gate 0.2b — per-worker parser audit | ⏳ blocked | ~$5 |
| Gate 0.3 — judge calibration (needs hand-graded JSONL) | ⏳ blocked | ~$5 |
| Gate 0.5 — S3 smoke (needs IAM creds) | ⏳ blocked | $0 |
| **Phase 1a launch on p5.48xlarge spot** | ⏳ blocked | ~$840 |

---

## Step 1 — finish dataset fetch (no $)

Backgrounded by Claude:
```bash
# Watch progress
tail -f /tmp/build_data.log    # or check the agent log
ls -la data/                    # train.jsonl + eval/*.jsonl + licenses.json
```

If it failed (HF rate limits, license walls), re-run:
```bash
uv run --with pyyaml --with datasets --with pyarrow --python 3.12 \
    python -m scripts.build_data --out-dir data/ --seed 17 --include-phase1b
```

By default substitutes LongBench-v2 for QuALITY (license). Pass `--accept-quality-license` to opt in.

---

## Step 2 — Gate 0.3 calibration data (manual, ~30 min, ~$5)

Hand-grade 30 math items so we can compare Haiku-judge agreement to a human.

```bash
# Sample 30 random math problems with worker answers (e.g., from a dry-run)
# and write a human-graded JSONL: {question, predicted, gold, human_label: bool}
# Then:
uv run --with pyyaml --with boto3 --python 3.12 \
    python -m scripts.judge --calibrate data/judge_calibration.jsonl --judge-model haiku
```

Gate passes if Haiku-judge ↔ human agreement ≥ 90%. If not, retry with `--judge-model sonnet` (~$25, ≥95% expected).

---

## Step 3 — Gate 0.2b parser audit (~$5, 5 min)

Validates output extraction works on every (worker, dataset) cell. CRITICAL — the rl-conductor reward bugs hid for weeks because this gate didn't exist.

```bash
# Build a small per-dataset eval subset (10 q each, will use ~20 calls per cell)
# data/eval_subsets.json must exist with format: {"math500": [...], "mmlu": [...], ...}
uv run --with pyyaml --with boto3 --python 3.12 \
    python -m scripts.run_gates --pool configs/pool.yaml \
    --gates 0.2b_parser_audit --include-bedrock-gates
```

Failures here are blocking. Common findings: Kimi K2 Thinking emits intermediate `\boxed`, Opus extended-thinking content blocks, GLM 5 nested `<think>` tags. All handled by `strip_reasoning()` but verify per-worker.

---

## Step 4 — IAM + S3 smoke (~$0, ~30s)

```bash
# Configure AWS profile if not set
aws configure   # us-east-1; access to agent-aiops-research bucket

# Run S3 write smoke
uv run --with pyyaml --with boto3 --python 3.12 \
    python -m scripts.run_gates --gates 0.5_artifact_capture --include-s3-gates \
    --s3-prefix s3://agent-aiops-research/cost-aware-routing
```

If your IAM role lacks `s3:PutObject` on this prefix, fix that before launching training (CheckpointManager will silently lose data).

---

## Step 5 — full pre-flight (free)

```bash
uv run --with pytest --with pyyaml --with sympy --with boto3 --python 3.12 \
    python -m scripts.run_gates --pool configs/pool.yaml \
    --include-bedrock-gates --include-s3-gates
```

Should print `[PASS] all blocking gates green — Phase 1 launch unblocked` before you provision the p5.

---

## Step 6 — provision p5 spot (~$9.63/hr)

```bash
export ALPHA=1.0
export S3_BUCKET=agent-aiops-research
./scripts/bootstrap.sh launch
# wait for fulfillment + tagging
export INSTANCE_IP=$(cat .last_instance_ip)
./scripts/bootstrap.sh provision
```

`provision` clones the repo onto NVMe, sets up `uv venv`, installs torch/transformers/etc.

---

## Step 7 — kick off Phase 1a, α=1.0 first

```bash
ALPHA=1.0 ./scripts/bootstrap.sh resume
# This launches train.py inside tmux. SSH in and reattach:
ssh -i ~/.ssh/g7e-bench.pem ec2-user@$INSTANCE_IP
tmux a -t train-1.0
```

Watch for the iter-0 brand-bias diagnostic in train.py output. If `histogram_entropy < 1.8` nats, halt — it's not an emergency, but sign that neutral_codes alone isn't fixing brand bias and you may need anti-bias SFT warmup as a Phase 1.5 prep.

Expected runtime: ~12h per α. ~$115 each. Run α=1.0 first; if it converges to >5× quality/$ over Always-Opus on the eval subset, queue the other 4 αs in parallel (need 1 instance per α — they're independent).

---

## Step 8 — Phase 1a eval + Pareto curve

```bash
# After all 5 α runs complete, sync rollouts back from S3:
aws s3 sync s3://agent-aiops-research/cost-aware-routing/rollouts/ data/rollouts/
# Build the always-X baselines (each one is a Bedrock-only eval, ~$5/baseline × 8):
# (TODO: write build_baselines.py — one Bedrock pass per always-X policy on the eval set)

# Then:
uv run --with pyyaml --python 3.12 \
    python -m scripts.pareto_eval \
    --rollouts data/rollouts/all_alphas.jsonl \
    --baselines-dir data/baselines/ \
    --out-json results/pareto_phase1.json
```

Headline-pass criterion: `ratio ≥ 5.0` between best router quality/$ and Always-Opus quality/$.

---

## Decision points / risks

1. **Gate 0.3 fails** → escalate to Sonnet judge for math; budget +$200 for full Phase 1.
2. **Gate 0.2b fails** → fix `strip_reasoning` for the failing worker; re-run audit. Don't lower the 90/95% threshold.
3. **iter-0 brand bias > 25% on any single worker** → halt, anti-bias SFT warmup (Phase 1.5 backup).
4. **Spot reclaim** → CheckpointManager auto-resumes from last full ckpt (every 25 iters). Same-AZ resume only (Gate 0.7).
5. **Format failure rate >5%** in first 10 iters → halt, recheck Gate 0.1 (chat template). Don't make the parser more lenient.
6. **Reward variance collapse at α=5** → use Lagrangian-CMDP backup variant pre-registered in spec § "Reward function".

---

## Cost ledger (target)

| Item | Estimate |
|------|---------:|
| Phase 1a — 5 α × $115 train + $265 Bedrock + judge | ~$840 |
| Phase 1b — stratified retrain + 1α ablation + Bedrock | ~$1,107 |
| Phase 1.5 transition + Phase 2 multi-step | ~$1,500 |
| **Total Phase 1 + Phase 2** | **~$3,450** |

Phase 0 Bedrock-billed gates: ~$10 total.

