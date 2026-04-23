# g7e Session Plan: Verifier Transferability T6 + T6b

## Goal

Test whether adversarial framing transfers to Devstral as generator, using two approaches:
- **T6**: Self-critique in the generation prompt (zero extra cost)
- **T6b**: Verifier-as-skill post-generation loop (best-of-N with v001∩v009)

Both produce Devstral diffs with gold labels, which we then feed to the verifier ensemble post-session to also answer T1-T3 (does the verifier transfer to non-Claude-generated patches?).

## Session Budget

- **Instance**: g7e.24xlarge (4x RTX PRO 6000 Blackwell)
- **Estimated time**: 3-4 hours
- **Compute cost**: ~$30-40 (g7e on-demand)
- **Verifier API cost**: ~$2 (T6b only, ~50 × 3 attempts × $0.038/call × ~46% fix rate)

## Baselines (from existing agent-harness data)

| Config | Fix Rate | Pass Rate | Source |
|--------|----------|-----------|--------|
| SERA × Devstral (control) | 46% (23/50) | 16% (8/50) | agent-harness Phase 2 |
| OpenCode × Devstral | 88% (44/50) | 22% (11/50) | agent-harness Phase 2 |

## Session Steps

### Phase 0: Infrastructure (20 min)

```bash
ssh -i ~/.ssh/g7e-bench.pem ec2-user@35.94.217.100

# Verify GPU
nvidia-smi

# Start Devstral Small 2 on vLLM
sudo nerdctl run -d --name devstral --gpus '"device=0"' \
  --network host \
  -v /mnt/nvme/models:/models \
  vllm/vllm-openai:latest \
  --model /models/devstral-small-2 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser mistral

# Verify serving
curl http://localhost:8000/v1/models

# Sync scripts from local
# (scp the updated multi_harness_eval.py, harness_eval.py, run_verifier_loop.py)
```

### Phase 1: T6 — SERA × Devstral + Self-Critique (1.5 hrs)

```bash
cd /mnt/nvme/agent-harness

python3 scripts/multi_harness_eval.py \
  --harness sera \
  --endpoint http://localhost:8000 \
  --model devstral-small-2 \
  --prompt-variant self-critique-strong \
  --output-dir results/t6_sera_selfcritique
```

**Outputs**:
- `results/t6_sera_selfcritique/phase2_sera_self-critique-strong.jsonl` — metrics
- `results/t6_sera_selfcritique/diffs/sera/*.diff` — patch diffs

### Phase 2: T6b — SERA × Devstral + Verifier Loop (2 hrs)

```bash
# Requires AWS creds for Bedrock (Claude Haiku verifier)
export AWS_REGION=us-east-1

python3 /path/to/verifier-reward/scripts/run_verifier_loop.py \
  --endpoint http://localhost:8000 \
  --model devstral-small-2 \
  --max-attempts 3 \
  --output-dir results/t6b_verifier_loop \
  --resume
```

**Outputs**:
- `results/t6b_verifier_loop/t6b_results.jsonl` — per-issue results with attempt details
- `results/t6b_verifier_loop/diffs/sera_verifier_loop/*.diff` — final selected diffs
- `results/t6b_verifier_loop/diffs/candidates/*.diff` — all candidate diffs

### Phase 3: Gold Evaluation (30 min)

```bash
# Gold eval T6 patches
python3 /path/to/verifier-reward/scripts/gold_eval.py \
  --diffs-dir results/t6_sera_selfcritique/diffs/sera/ \
  --output /path/to/verifier-reward/results/gold_devstral_sera_selfcritique.jsonl

# Gold eval T6b patches (final selected)
python3 /path/to/verifier-reward/scripts/gold_eval.py \
  --diffs-dir results/t6b_verifier_loop/diffs/sera_verifier_loop/ \
  --output /path/to/verifier-reward/results/gold_devstral_sera_verifier_loop.jsonl
```

### Phase 4: Download Results

```bash
# From local machine
VR=domains/autoresearch/blueprints/verifier-reward/results

# T6 diffs + gold
scp -i ~/.ssh/g7e-bench.pem -r \
  ec2-user@35.94.217.100:/mnt/nvme/agent-harness/results/t6_sera_selfcritique/diffs/sera/ \
  $VR/diffs/devstral_sera_selfcritique/

scp -i ~/.ssh/g7e-bench.pem \
  ec2-user@35.94.217.100:$VR/gold_devstral_sera_selfcritique.jsonl \
  $VR/

# T6b diffs + gold + results
scp -i ~/.ssh/g7e-bench.pem -r \
  ec2-user@35.94.217.100:/mnt/nvme/agent-harness/results/t6b_verifier_loop/diffs/sera_verifier_loop/ \
  $VR/diffs/devstral_sera_verifier_loop/

scp -i ~/.ssh/g7e-bench.pem \
  ec2-user@35.94.217.100:$VR/gold_devstral_sera_verifier_loop.jsonl \
  $VR/

scp -i ~/.ssh/g7e-bench.pem \
  ec2-user@35.94.217.100:/mnt/nvme/agent-harness/results/t6b_verifier_loop/t6b_results.jsonl \
  $VR/
```

## Post-Session (local, API-only)

### Run Verifier Ensemble on T6 Patches (~$1)

```bash
cd domains/autoresearch/blueprints/verifier-reward

# T6: Does verifier transfer to Devstral self-critique patches?
python3 scripts/run_cross_verifier.py \
  --verifier haiku \
  --patch-source devstral_sera_selfcritique
```

T6b patches already have verifier verdicts from the loop — just need gold eval comparison.

### Analysis

```python
# Compare pass rates
# Control: SERA × Devstral = 16% (from agent-harness)
# T6:  SERA × Devstral + self-critique = ?
# T6b: SERA × Devstral + verifier loop = ?
```

## Expected Outcomes

### T6: Self-Critique Transfer

| Scenario | T6 Pass Rate | Conclusion |
|----------|-------------|------------|
| > 19% (+3pp) | Self-critique transfers to Devstral |
| 13-19% (~control) | No effect — Devstral can't self-critique |
| < 13% | Self-critique hurts (overthinking) |

### T6b: Verifier-as-Skill

| Scenario | T6b Pass Rate | Conclusion |
|----------|-------------|------------|
| > 25% | Verifier loop is a viable production pattern |
| 16-25% | Marginal improvement, not worth 3x generation cost |
| < 16% | Verifier rejects too aggressively (recall problem) |

### Verifier Transfer (T1-T3 answered from T6 data)

| Scenario | Precision on Devstral Patches | Conclusion |
|----------|------------------------------|------------|
| = 1.00 | Verifier is generator-agnostic |
| 0.50-0.99 | Partial transfer, Devstral patches have different error signatures |
| < 0.50 | Verifier overfit to Claude-style patches |

## Scripts Modified

| File | Change |
|------|--------|
| `agent-harness/scripts/harness_eval.py` | `_get_git_diff()` cap 50K→100K; `_prompt_suffix` support in `run_instrumented_loop` |
| `agent-harness/scripts/multi_harness_eval.py` | Added `--prompt-variant`, `PROMPT_VARIANTS`, wired through SERA + CLI paths |
| `verifier-reward/scripts/run_verifier_loop.py` | **NEW** — T6b verifier-in-the-loop script |
| `verifier-reward/scripts/run_cross_verifier.py` | Added devstral patch sources, flexible gold label + diff directory resolution |
