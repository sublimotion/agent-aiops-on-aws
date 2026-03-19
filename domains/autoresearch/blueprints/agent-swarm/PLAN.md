# Agent Swarm - Phase 1 Execution Plan

**Status**: IN PROGRESS
**Started**: 2026-03-18
**Updated**: 2026-03-19

## Configuration Matrix (12 runs)

| # | Model | Harness | Status | Pass Rate | Notes |
|---|-------|---------|--------|-----------|-------|
| 1 | Devstral Small 2 24B | OpenCode | DONE (reused) | 11/50 (22%) | From Phase 2b |
| 2 | Devstral Small 2 24B | Claude Code | DONE (reused) | 10/50 (20%) | From Phase 2b |
| 3 | Devstral Small 2 24B | SERA | DONE (reused) | 8/50 (16%) | From Phase 2a |
| 4 | Qwen 2.5 Coder 32B | OpenCode | FAILED (0%) | 0/50 (0%) | Model outputs bare JSON, not `<tool_call>` — hermes parser can't extract |
| 5 | Qwen 2.5 Coder 32B | Claude Code | SKIPPED | - | Same tool calling incompatibility |
| 6 | Qwen 2.5 Coder 32B | SERA | RUNNING (~5/50) | 2/5 fix rate so far | Bare JSON fallback patch, 32K context, context trimming |
| 7 | SWE-agent-LM 32B | OpenCode | FAILED (0%) | 0/50 (0%) | Bare JSON output, no XML tags |
| 8 | SWE-agent-LM 32B | Claude Code | SKIPPED | - | Same tool calling incompatibility |
| 9 | SWE-agent-LM 32B | SERA | RUNNING (~3/50) | 0/3 fix rate so far | Same patches as #6 |
| 10 | Qwen3.5-397B-A17B | OpenCode | QUEUED | - | TP4, all GPUs |
| 11 | Qwen3.5-397B-A17B | Claude Code | QUEUED | - | |
| 12 | Qwen3.5-397B-A17B | SERA | QUEUED | - | |

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
