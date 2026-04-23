# g7e Session Plan: T9 Qwen3.5 Verifier Transfer

## Execution Summary (2026-03-24)

- **Pivoted**: 397B-GPTQ-Int4 garbage output on vLLM 0.18 → used **122B-A10B-FP8** instead
- **Serving**: vLLM 0.18.0, TP4, `--tool-call-parser qwen3_xml --reasoning-parser qwen3`
- **Fix rate**: 43/50 (86%), **Gold pass**: 4/43 (9%)
- **Verifier**: Prec=0.50, Rec=0.50, F₀.₅=0.50 (2 TP, 2 FP, 2 FN, 37 TN)
- **Verdict**: Hypothesis partially confirmed — harness helps but model still matters
- **Runtime**: 97 min generation + 10 min gold eval. Compute cost ~$40.

## Goal

Test the "harness determines transfer" hypothesis: if OpenCode-style diffs (large, multi-hunk) are the key to verifier precision, then Qwen3.5 397B × OpenCode should achieve precision 0.70-1.00 — similar to Claude × OpenCode — despite being a completely different generator model.

## Hypothesis

| Factor | Devstral × SERA (T6b) | Qwen3.5 × OpenCode (T9) |
|--------|----------------------|--------------------------|
| Diff style | Surgical, 2-7 lines | Multi-file, 10-15K chars |
| Verifier surface area | Minimal | Rich |
| Expected precision | 0.33 (observed) | 0.70-1.00 (predicted) |
| Expected recall | 0.25 | 0.33+ |

If confirmed, this proves the verifier is **harness-agnostic** (works on any model's diffs as long as the harness produces rich patches), and the T6b precision drop was a SERA problem, not a Devstral problem.

## Baselines

| Config | Fix Rate | Pass Rate | Source |
|--------|----------|-----------|--------|
| Qwen3.5 × OpenCode | 88% (44/50) | — | agent-swarm Phase 1 |
| Qwen3.5 × SERA | 72% (36/50) | — | agent-swarm Phase 1 |
| Claude × OpenCode (verifier ref) | — | Prec=1.00 | verifier-reward T1-T5 |
| Devstral × SERA (verifier ref) | — | Prec=0.33 | verifier-reward T6b |

Note: Phase 1 swarm data had no gold eval (no diffs captured). This session captures diffs + runs gold eval.

## Session Budget

- **Instance**: g7e.24xlarge (4x RTX PRO 6000 Blackwell, 384 GB GDDR7 total)
- **Model**: Qwen3.5-397B (MoE, ~200GB weights → TP4 across 4 GPUs)
- **Estimated time**: 4-5 hours (Qwen3.5 is slower than Devstral 24B)
- **Compute cost**: ~$40-50 (g7e on-demand)
- **Verifier API cost**: ~$2 (post-session, 50 × 5 calls × $0.008/call)

## Prerequisites

- [ ] Qwen3.5-397B weights downloaded to `/mnt/nvme/models/` (or HF cache)
- [ ] vLLM image with Qwen3.5 support (hermes tool parser works)
- [ ] OpenCode harness scripts synced with diff capture enabled
- [ ] AWS credentials for Bedrock (verifier calls)

## Session Steps

### Phase 0: Infrastructure (30 min)

```bash
ssh -i ~/.ssh/g7e-bench.pem ec2-user@<IP>

# Verify GPUs
nvidia-smi

# Download Qwen3.5-397B if not cached
# ~200GB, takes ~20 min on g7e NVMe
huggingface-cli download Qwen/Qwen3.5-397B-A35B \
  --local-dir /mnt/nvme/models/Qwen3.5-397B-A35B

# Start Qwen3.5 on vLLM with TP4
sudo nerdctl run -d --name qwen35 --gpus all \
  --network host \
  -v /mnt/nvme/models:/models \
  -v /mnt/nvme/hf-cache:/root/.cache/huggingface \
  --shm-size 16g \
  vllm/vllm-openai:latest \
  --model /models/Qwen3.5-397B-A35B \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 4 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# Wait for model load (~5-10 min for 397B)
# Watch logs
sudo nerdctl logs -f qwen35

# Verify serving
curl http://localhost:8000/v1/models
```

**Key config notes:**
- TP4 required — 397B MoE with ~35B active params, but full weights ~200GB
- `--tool-call-parser hermes` — Qwen3.5 outputs proper `tool_calls` with hermes format
- `--max-num-seqs 4` — conservative; Qwen3.5 at 65K context is memory-heavy
- `--shm-size 16g` — needed for TP4 shared memory

### Phase 1: OpenCode × Qwen3.5 with Diff Capture (3 hrs)

```bash
cd /mnt/nvme/agent-harness

# Run OpenCode harness with diff capture
python3 scripts/multi_harness_eval.py \
  --harness opencode \
  --endpoint http://localhost:8000 \
  --model Qwen3.5-397B-A35B \
  --output-dir results/t9_opencode_qwen35 \
  --save-diffs
```

**Critical**: Ensure `--save-diffs` captures full diffs to `results/t9_opencode_qwen35/diffs/opencode/*.diff`. Previous Phase 1 swarm data lost diffs because they were on ephemeral NVMe without capture.

**Expected runtime**: ~3 hrs (777K tokens/issue avg from Phase 1, but TP4 throughput higher than single GPU).

**Outputs**:
- `results/t9_opencode_qwen35/phase2_opencode.jsonl` — per-issue metrics
- `results/t9_opencode_qwen35/diffs/opencode/*.diff` — patch diffs (THE KEY ARTIFACT)

### Phase 2: Gold Evaluation (30 min)

```bash
# Run gold eval on captured diffs
python3 scripts/gold_eval.py \
  --diffs-dir results/t9_opencode_qwen35/diffs/opencode/ \
  --output results/t9_opencode_qwen35/gold_qwen35_opencode.jsonl
```

**Expected**: ~88% fix rate (44/50 produce diffs), ~20-25% pass rate based on Devstral OpenCode baseline.

### Phase 3: Download Results

```bash
# From local machine
G7E=ec2-user@<IP>
VR=domains/autoresearch/blueprints/verifier-reward/results

# Diffs (most important)
scp -i ~/.ssh/g7e-bench.pem -r \
  $G7E:/mnt/nvme/agent-harness/results/t9_opencode_qwen35/diffs/opencode/ \
  $VR/diffs/qwen35_opencode/

# Gold eval
scp -i ~/.ssh/g7e-bench.pem \
  $G7E:/mnt/nvme/agent-harness/results/t9_opencode_qwen35/gold_qwen35_opencode.jsonl \
  $VR/

# Per-issue metrics
scp -i ~/.ssh/g7e-bench.pem \
  $G7E:/mnt/nvme/agent-harness/results/t9_opencode_qwen35/phase2_opencode.jsonl \
  $VR/
```

## Post-Session (local, API-only)

### Run Verifier Ensemble on Qwen3.5 Diffs (~$2)

```bash
cd domains/autoresearch/blueprints/verifier-reward

# Run v001∩v009 ensemble on Qwen3.5 × OpenCode diffs
python3 scripts/run_cross_verifier.py \
  --verifier haiku \
  --patch-source qwen35_opencode \
  --gold-file results/gold_qwen35_opencode.jsonl
```

### Analysis

```python
# Compare verifier precision across generators + harnesses:
#
# Generator × Harness        | Diff Size | Precision | Recall
# Claude × OpenCode          | ~12K      | 1.00      | 0.33
# Devstral × SERA            | ~1.5K     | 0.33      | 0.25
# Qwen3.5 × OpenCode (T9)   | ~10-15K?  | ???       | ???
#
# If Qwen3.5 precision >= 0.70: harness determines transfer (CONFIRMED)
# If Qwen3.5 precision ~= 0.33: model determines transfer (REFUTED)
```

### Optional: SERA × Qwen3.5 Control

If time permits during the g7e session, also run SERA harness for a direct harness comparison on the same model:

```bash
python3 scripts/multi_harness_eval.py \
  --harness sera \
  --endpoint http://localhost:8000 \
  --model Qwen3.5-397B-A35B \
  --output-dir results/t9_sera_qwen35 \
  --save-diffs
```

This gives us a 2×2 matrix (Devstral/Qwen3.5 × SERA/OpenCode) to fully disentangle model vs harness effects.

## Expected Outcomes

### Primary: Verifier Transfer

| Scenario | Precision on Qwen3.5 Diffs | Conclusion |
|----------|-----------------------------|------------|
| >= 0.70 | Harness determines transfer. Verifier works on any model with OpenCode. |
| 0.40-0.69 | Partial — model-specific patterns matter but harness helps. |
| < 0.40 | Model determines transfer. Verifier overfit to Claude error signatures. |

### Secondary: Qwen3.5 Gold Pass Rate

| Scenario | Pass Rate | Conclusion |
|----------|-----------|------------|
| > 25% | Qwen3.5 × OpenCode competitive with Claude |
| 15-25% | On par with Devstral × OpenCode (22%) |
| < 15% | Token overhead (777K) doesn't translate to quality |

### Tertiary: Diff Size Confirmation

| Prediction | Measurement | |
|------------|-------------|---|
| Median diff ~10-15K chars | ? | Confirms OpenCode produces similar diff sizes regardless of model |
| Median diff ~1-3K chars | ? | Would mean Qwen3.5 produces smaller patches even through OpenCode |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Qwen3.5 397B OOM on 4x 96GB | Reduce `--max-model-len` to 32768 or `--max-num-seqs 2` |
| vLLM Qwen3.5 tool calling broken | Fall back to `--tool-call-parser` auto or try `--chat-template` override |
| OpenCode diff capture not working | Verify on first issue, check `_get_git_diff()` in harness_eval.py |
| g7e spot termination | Use on-demand; checkpoint every 10 issues via `--resume` |
| SSH flaky under load (seen in T6b) | Use `nohup` + `screen`, check via `tail -f` after reconnect |

## Diff from T6 Session Plan

| Aspect | T6/T6b | T9 |
|--------|--------|-----|
| Generator model | Devstral 24B | Qwen3.5 397B |
| Serving config | 1 GPU, TP1 | 4 GPUs, TP4 |
| Primary harness | SERA | OpenCode |
| Key question | Does verifier transfer to non-Claude? | Does harness determine transfer? |
| Context window | 65K | 65K |
| Token consumption | ~10K/issue (SERA) | ~777K/issue (OpenCode) |
