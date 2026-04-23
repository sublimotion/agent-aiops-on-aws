# Agent Swarm - Phase 1 Execution Plan

**Status**: COMPLETE
**Started**: 2026-03-18
**Updated**: 2026-03-19

## Configuration Matrix (12 runs)

| # | Model | Harness | Status | Fix Rate | Notes |
|---|-------|---------|--------|----------|-------|
| 1 | Devstral Small 2 24B | OpenCode | DONE (reused) | 44/50 (88%) | From Phase 2b |
| 2 | Devstral Small 2 24B | Claude Code | DONE (reused) | 19/50 (38%) | From Phase 2b |
| 3 | Devstral Small 2 24B | SERA | DONE (reused) | 23/50 (46%) | From Phase 2a |
| 4 | Qwen 2.5 Coder 32B | OpenCode | FAILED (0%) | 0/45 (0%) | Bare JSON, hermes parser can't extract |
| 5 | Qwen 2.5 Coder 32B | Claude Code | SKIPPED | — | Anthropic API incompatible with Qwen |
| 6 | Qwen 2.5 Coder 32B | SERA | **DONE** | **24/50 (48%)** | Bare JSON fallback patch |
| 7 | SWE-agent-LM 32B | OpenCode | FAILED (0%) | 0/46 (0%) | Bare JSON, hermes parser can't extract |
| 8 | SWE-agent-LM 32B | Claude Code | SKIPPED | — | Anthropic API incompatible with Qwen |
| 9 | SWE-agent-LM 32B | SERA | **DONE** | **9/50 (18%)** | SWE-smith SFT -30pp vs base Qwen 2.5 |
| 10 | Qwen3.5-397B-A17B | OpenCode | **DONE** | **44/50 (88%)** | Best config tied with #1 |
| 11 | Qwen3.5-397B-A17B | Claude Code | FAILED | — | Anthropic API doesn't translate tools to Qwen chat template |
| 12 | Qwen3.5-397B-A17B | SERA | **DONE** | **36/50 (72%)** | Best SERA result |

## Phase 2: Allen AI SERA-32B (PLANNED)

**Goal**: Compare Allen AI's SVG-trained `allenai/SERA-32B` against Princeton's SWE-smith SFT (`SWE-agent-LM-32B`) and the base Qwen models. This is the critical finetuning comparison — SVG (verification-guided training) vs SWE-smith (trajectory SFT) vs no finetuning.

**Why this matters**: SERA-32B reports 49.5% on SWE-bench Verified. It uses Qwen 3-32B as base (vs SWE-agent-LM's Qwen 2.5 Coder 32B). The SVG training methodology — soft verification via line-level recall, no test execution — is directly relevant to the Learned Verifier Experiment. Allen AI baked verification into the training loop; we use it at inference time. Comparing both approaches on the same harnesses answers: is it better to verify during training or at inference?

| # | Model | Harness | Status | Fix Rate | Notes |
|---|-------|---------|--------|----------|-------|
| 13 | allenai/SERA-32B | SERA | PLANNED | — | SVG-trained, Qwen 3-32B base, 49.5% published |
| 14 | allenai/SERA-32B | OpenCode | PLANNED | — | Check tool calling format compatibility first |
| 15 | allenai/SERA-32B | Claude Code | PLANNED | — | Likely incompatible (Anthropic API) |

### Setup Notes

- **HF ID**: `allenai/SERA-32B`
- **Base model**: Qwen 3-32B (newer than SWE-agent-LM's Qwen 2.5)
- **Size**: 32B dense, TP1 — fits on single GPU same as other 32B models
- **Training**: SVG on 25K synthetic trajectories, GLM-4.6 teacher, ~$2,000 total cost
- **Tool calling**: Unknown — need to check if it uses `<tool_call>` tags, bare JSON, or Qwen 3 native format. This determines harness compatibility.
- **Weights**: Need to download to `/mnt/nvme/models/sera-32b-allenai/`

### Key Comparisons

| Comparison | What It Tests |
|-----------|---------------|
| SERA-32B vs Qwen 2.5 Coder 32B | SVG finetuning vs no finetuning (different base, so confounded) |
| SERA-32B vs SWE-agent-LM 32B | SVG vs SWE-smith SFT (different base + different method) |
| SERA-32B vs Qwen3.5 397B | Can 32B + SVG training compete with 397B scale? |
| SERA-32B vs Devstral 24B | 32B SVG-trained vs 24B Mistral-native on same harnesses |

### Open Questions

- Does SERA-32B's Qwen 3 base have native tool calling that works with vLLM hermes parser? If yes, it may be compatible with OpenCode (unlike Qwen 2.5 models).
- The published 49.5% is on SWE-bench Verified (full 500). Our subset is 50 issues — expect variance.
- SERA-32B was trained with its own agent scaffold. Does it generalize to SERA harness / OpenCode, or is it scaffold-specific like SWE-agent-LM was?

## Key Findings So Far

1. **Qwen-family models incompatible with OpenAI tool calling API**: Both Qwen 2.5 Coder 32B and SWE-agent-LM 32B output bare JSON (`{"name": "...", "arguments": {...}}`) in content instead of proper `<tool_call>` XML tags. vLLM's hermes parser can't extract these. OpenCode and Claude Code both rely on OpenAI-format `tool_calls` → 0% fix rate.

2. **SERA harness works with bare JSON patch**: Added `_BARE_JSON_TOOL_RE` fallback to `extract_tool_calls_from_msg()` in `harness_eval.py`. SERA extracts tool calls from content text, executes tools, and builds proper `tool_calls` messages for the conversation history. Qwen's chat template handles this correctly.

3. **32K context required, 16K too small**: 16K context causes broken pipe errors by turn 5. File reads (especially Django source files) consume 5-10K tokens each. With 32K + `enforce_eager` + `gpu_memory_utilization=0.95`, fits on single GPU (91.6 GB / 97.9 GB).

4. **Context management critical**: Even with 32K, conversations exceed limit by turn 5-7. Required three patches:
   - Tool output truncation: 4000 chars max per tool result
   - `_trim_context()`: Trim old tool results when total chars > 20K
   - `max_tokens`: Reduced from 4096 → 1024 (leaves 31744 for input)

5. **Broken pipe = context overflow, not connection issue**: aiohttp `ClientOSError: [Errno 32] Broken pipe` means vLLM rejected the request (400) or the request payload is too large for TCP send buffer. `force_close=True` doesn't help — the fix is context management.

6. **Only SERA harness viable for Qwen models**: OpenCode and Claude Code both require OpenAI-format tool calls. This reduces the experiment from 3 harnesses to 1 for Qwen-family models, limiting harness spread measurement.

## Execution Order (Revised)

**Phase 1a — TP1 models (parallel on GPU 0 + GPU 1)** ← CURRENT
1. Qwen 2.5 Coder 32B × SERA (GPU 0, port 8000)
2. SWE-agent-LM 32B × SERA (GPU 1, port 8001)

**Phase 1b — TP4 model (needs all 4 GPUs)**
3. Qwen3.5-397B-A17B × {OpenCode, Claude Code, SERA}

Note: Qwen3.5-397B may have different tool calling behavior (MoE, different training). Need to test OpenCode/Claude Code compatibility before assuming SERA-only.

## Infrastructure

- **Instance**: g7e.24xlarge (4x RTX PRO 6000 Blackwell, 96 GB each)
- **vLLM**: v0.16.0 container (`vllm/vllm-openai:v0.16.0`)
- **GPU 0**: Qwen 2.5 Coder 32B (TP1, 32K context, port 8000) — `vllm-qwen25-coder`
- **GPU 1**: SWE-agent-LM 32B (TP1, 32K context, port 8001) — `vllm-sweagent-lm`
- **GPU 2-3**: Free (reserved for Qwen3.5-397B TP4 later)
- **vLLM flags**: `--max-model-len 32768 --max-num-seqs 2 --gpu-memory-utilization 0.95 --enforce-eager --tool-call-parser hermes --enable-auto-tool-choice --enable-prefix-caching`

## Model Weights

| Model | Path | Status |
|-------|------|--------|
| Devstral Small 2 FP8 | `/mnt/nvme/models/devstral-small-2-fp8` | Ready |
| Qwen 2.5 Coder 32B | `/mnt/nvme/models/qwen25-coder-32b` | Ready (14 shards) |
| SWE-agent-LM 32B | `/mnt/nvme/models/swe-agent-lm-32b` | Ready (14 shards) |
| Qwen3.5-397B-A17B FP8 | `/mnt/nvme/models/qwen3-next-fp8` | Ready |
| allenai/SERA-32B | `/mnt/nvme/models/sera-32b-allenai` | **TODO: download** |

## Patches Applied to harness_eval.py

1. **Bare JSON tool call extraction** (`_BARE_JSON_TOOL_RE`): Regex fallback matching `{"name": "...", "arguments": {...}}` in content after `<tool_call>` regex fails
2. **Context trimming** (`_trim_context`): Trims old tool results to 500 chars when total conversation > 20K chars
3. **Tool output truncation**: All tool outputs capped at 4000 chars
4. **max_tokens**: Reduced to 1024 (from 4096) to leave more room for input
5. **HTTP timeout**: Increased to 600s (from 180s)
6. **TCP force_close**: `aiohttp.TCPConnector(force_close=True)` to prevent stale connections
7. **Error logging**: Added exception type and traceback to turn error logs

## Monitoring

```bash
# SSH to g7e
ssh -i ~/.ssh/g7e-bench.pem ec2-user@35.94.217.100

# Check progress
wc -l /mnt/nvme/agent-harness/results/swarm/swarm_phase1_*.jsonl

# Check fix rates
cat /mnt/nvme/agent-harness/results/swarm/swarm_phase1_qwen25-coder-32b_sera.jsonl | python3 -c "
import sys, json
fixes = sum(1 for l in sys.stdin if json.loads(l)['fix_generated'])
print(f'Fixes: {fixes}')
"

# Watch logs
tail -f /mnt/nvme/agent-harness/swarm_qwen25_sera.log
tail -f /mnt/nvme/agent-harness/swarm_swesmith_sera.log

# GPU utilization
nvidia-smi -l 5
```

## Results Location

All results in: `/mnt/nvme/agent-harness/results/swarm/`
Format: `swarm_phase1_{model}_{harness}.jsonl`
