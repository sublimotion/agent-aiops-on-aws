# Nemotron-3-Ultra-550B-A55B-NVFP4 Lessons Learned

Blueprint scaffolded from `domains/gpu-serving/specs/nemotron-ultra.md` on 2026-06-05,
using sibling `nemotron-super/` as the structural template (same Mamba-2 + LatentMoE
family). Target: p6-b300.48xlarge spot in us-west-2b (usw2-az2), vLLM v0.22.0-cu130, TP4
single-replica smoke test toward the Stage 5 / P0 gate.

## CORRECTION (2026-06-06): vLLM/B200 re-test overturns the "acceptance ceiling" verdict

> **READ THIS FIRST.** The Run-1 SGLang conclusions below (lines tagged "acceptance is
> the gate", single-stream ~117-177 tok/s, "300 tok/s unreachable, inherent to the model")
> were **measurement artifacts of a broken SGLang MTP path**, not properties of the model.
> They are retained verbatim as an accurate record of what SGLang produced, but the
> root-cause verdict was wrong. The corrected picture:

### root-cause: SGLang NemotronH MTP is broken (#21138) — vLLM gets 1.7× the single-stream speed on half the GPUs
<!-- captured: 2026-06-06 | stage: 6 | engine: vllm -->

First run (SGLang v0.5.12.post1-cu130, B300 TP8 wide-tree) measured single-stream decode
~177 tok/s, accept_len ~2.4, and concluded acceptance was an inherent EAGLE/MTP ceiling on
this model. **That was SGLang bug [#21138](https://github.com/sgl-project/sglang/issues/21138):
NemotronH MTP reports ~0% real acceptance (accept_len 2.4 ≈ the always-accepted base token
only).** NVIDIA's tech report (Table 6) publishes accept_len 4.9-5.5 for this exact model.

Re-tested on **vLLM v0.22.0, B200 p6-b200.48xlarge TP4** (us-east-2 az2, NVIDIA min-latency
recipe, `nemotron_h_mtp` k=5), same diverse real prompts, temp=1.0, 512 out:
- **single-stream decode median 297.7 tok/s** (vs SGLang 177 — 1.7×), wall 267.6 tok/s.
  **6/12 prompts clear 300 decode.** DeepInfra's 300 tok/s SLA is essentially MATCHED (0.99×).
- **accept_len 3.54** (accept_rate 0.508) on diverse prompts — vs SGLang's bugged 2.4.
  Holds flat ~3.56 across the entire concurrency sweep (c=1→256). NVIDIA's 4.9-5.5 suggests
  more headroom (lower temp, draft-length tuning).
- Peak aggregate **~1883 tok/s @ c=256** (SGLang 1040 — 1.8×); sustained ~1847 tok/s, 0 errors.
- Every P1 workload 2.4-4.9× the SGLang figure; long-context 468-514 tok/s @ 64k-256k (3.7-4.3×).
- All on **4 GPUs (TP4) vs SGLang's 8 (TP8)**.

**Takeaway**: for NemotronH + native MTP, **engine choice dominates hardware**. Use vLLM
`nemotron_h_mtp`, not SGLang EAGLE/MTP, until #21138 is fixed. The "acceptance is the
inherent gate" lesson below is engine-specific to SGLang and must NOT be generalized.
Cost: self-host ~$6.00/M output @ $40.57/hr B200 spot, still 2.4× DeepInfra's $2.50/M retail.

### serving: vLLM v0.22.0 dropped `--disable-log-requests` — remove it from the recipe
<!-- captured: 2026-06-06 | stage: 5 | engine: vllm -->

NVIDIA's published vLLM recipe (and older cards) include `--disable-log-requests`. vLLM
v0.22.0 removed that flag — `vllm serve` exits immediately with
`error: unrecognized arguments: --disable-log-requests`. Drop the flag (request logging is
off by default at INFO anyway, or use `--otlp-traces-endpoint` for structured logging).
Pod otherwise boots clean: multithread shard load ~4.5 shards/s (113 shards in ~25s from
NVMe), healthy in ~8 min total incl. CUDA-graph capture + MTP warmup.

### serving: B200/us-east-2 is the vLLM surface — the cu130 blocker was B300-specific
<!-- captured: 2026-06-06 | stage: 5 | engine: vllm -->

The Run-1 blocker "vLLM v0.22.0-cu130 does not exist" is real but **only constrains B300
(sm_103, needs cu130)**. B200 is **sm_100** and runs the standard **`vllm/vllm-openai:v0.22.0`**
(cu12.x) image perfectly. So the vLLM throughput leg belongs on B200/us-east-2, not B300.
Operational notes for a fresh B200 node group (`ai-infra-use2-b200-spot`,
cluster `qwen3-next-bench-eks-cluster`):
- nvidia-device-plugin is gated on the `nvidia.com/gpu.present=true` node label — a fresh
  NG may lack it; `kubectl label node <n> nvidia.com/gpu.present=true --overwrite` to advertise GPUs.
- Instance-store NVMe is NOT pre-RAIDed on this NG (unlike the B300 NG): mdadm RAID0 the
  8×3.5T devices → /dev/md0 → mkfs.ext4 → mount /mnt/nvme (28T) via SSM before staging.
- hostNetwork pods need `dnsPolicy: Default` (NOT ClusterFirstWithHostNet) for external DNS
  (HF download, etc.) — CoreDNS can't resolve external names for hostNetwork pods.
- Single-quote inline-JSON vLLM args (`--speculative-config`, `--model-loader-extra-config`)
  in YAML or the `{...}:` breaks the YAML parser.
- Direct HF→NVMe `snapshot_download` in a python:3.12-slim pod is the fast staging path
  (329G/113 shards); the host has no pip/HF tooling.

## Pre-Deployment

### blocker: vLLM `v0.22.0-cu130` image does NOT exist — B300 smoke must use SGLang cu130
<!-- captured: 2026-06-05 | stage: 5 -->

The spec's image table (and `variables.tf` default `vllm/vllm-openai:v0.22.0-cu130`) assumes
a tag that is not published on Docker Hub. Verified against the live `vllm/vllm-openai`
registry this session:
- The newest **cu130** vLLM stable tag is **`v0.20.0-cu130`** (too old for `nemotron_h_mtp` +
  NVFP4, which need >= 0.22).
- `v0.22.1` exists but **only on cu129** (`v0.22.1-cu129`, CUDA 12.9). B300 is sm_103 and
  requires a CUDA-13 (cu130) runtime — a cu129 image will not load on sm_103.
- So there is no vLLM image that is simultaneously >= 0.22 AND cu130. The vLLM B300 leg is
  blocked on upstream publishing a cu130 build at >= 0.22 (or building one ourselves).

By contrast SGLang **`lmsysorg/sglang:v0.5.12.post1-cu130` exists exactly as the spec lists**
and pulls fine (verified ~185 MiB/s on-node). SGLang is the documented B300 path and carries
EAGLE MTP for this model.

**Fix**: Pivot the B300 Stage 5 smoke test to SGLang (`v0.5.12.post1-cu130`,
`--reasoning-parser nemotron_3`, EAGLE MTP, `SGLANG_DISABLE_DEEP_GEMM=1`). Keep the vLLM
config as the primary throughput leg but defer it to B200 (us-east-2, where standard cu12.x
`v0.22.0` runs on sm_100) OR until a vLLM cu130 >= 0.22 image is available. Update
`variables.tf` / the spec image table to reflect that vLLM cu130 >= 0.22 is unavailable as
of 2026-06-05.



### serving: SGLang launch — three card-command corrections needed for nemotron_h on cu130
<!-- captured: 2026-06-05 | stage: 5 -->

The HF card's SGLang command (transcribed verbatim into the spec) failed at arg-parse /
init three times on `lmsysorg/sglang:v0.5.12.post1-cu130`. Resolved each:

1. **`--kv-cache-dtype fp8` → `fp8_e4m3`.** SGLang rejects bare `fp8`
   (`invalid choice: 'fp8' (choose from auto, fp8_e5m2, fp8_e4m3, bf16, fp4_e2m1)`). `fp8`
   is the *vLLM* spelling; the card mixed the two engines' values. Use `fp8_e4m3` for SGLang.

2. **MTP + radix cache + `--mamba-scheduler-strategy no_buffer` is rejected.** With the
   verbatim flags SGLang errors:
   `ValueError: Speculative decoding for NemotronHForCausalLM is not compatible with radix
   cache when using --mamba-scheduler-strategy no_buffer. To use radix cache with
   speculative decoding, please use --mamba-scheduler-strategy extra_buffer and set
   SGLANG_ENABLE_SPEC_V2=1.`
   This is the mamba-hybrid + MTP + prefix-cache conflict the Stage 0c WARN predicted.

3. **`extra_buffer` is ALSO unsupported for this model.** Following the engine's own
   suggestion (`extra_buffer` + `SGLANG_ENABLE_SPEC_V2=1`) then fails with
   `AssertionError: mamba extra_buffer is not supported for NemotronHForCausalLM model`.

**Resolution that boots**: keep `--mamba-scheduler-strategy no_buffer` + EAGLE MTP, and
**`--disable-radix-cache`** (drop prefix caching). This matches the lessons.md Stage 0c
fallback exactly: on a mamba-hybrid you cannot have MTP *and* prefix/radix caching at once —
MTP (the 300 tok/s lever) wins, prefix caching is sacrificed. Net effect for benchmarking:
no cross-request prefix cache on SGLang, so the warm-cache TTFT win in rag-1m-context won't
appear on this engine/config — note when reporting long-context results.

### benchmark: P4 "1M context" is impossible — model max is 262144 (256k); feasible tiers are 64k/128k/256k
<!-- captured: 2026-06-05 | stage: 6 -->

The spec's P4 (and the `rag-1m-context` card) sweep 64k/128k/256k/512k/**1m** prefix tiers. This
model's `max_position_embeddings = max_model_len = 262144` (256k) — confirmed in benchmark.yaml
and the on-node config.json. The 512k and 1m tiers **exceed the architectural context limit** and
cannot run (requests >262k tokens 400 with Bad Request). Only 64k/128k/256k are valid.

Measured on TP8 (temp=1.0, `ignore_eos`, 256 output tokens), long-context decode is **stable**:
~131 tok/s @ 118k context, ~119 tok/s @ 236k context (near the 262k ceiling) — no decode collapse,
no OOM. The 58.6M-token KV pool is nowhere near saturated by a single long-context request.
TTFT/e2e scales with prefill as expected: 4.7s @ 64k → 16.9s @ 236k. **Fix**: cap the P4 sweep at
256k for this model; do not report 1M tiers. (`decode_throughput` in `meta_info` occasionally
returns a corrupt huge value under spec-v1 / wide-tree — cross-check with completion/e2e.)

### benchmark: TP8 vs TP4 + the 300 tok/s verdict — acceptance is the gate, not decode hops
<!-- captured: 2026-06-05 | stage: 6 -->

The spec's primary hypothesis for beating DeepInfra's 300 tok/s single-stream was TP8 (fewer
cross-GPU decode hops). Tested both layouts, same real prompts / temp=1.0 / 512 out:
- **TP4 + EAGLE (topk=1, steps=5)**: single-stream median **117 tok/s** decode (range 103-246).
  Weights 80 GB/GPU, KV pool 39.75M tokens.
- **TP8 + EAGLE (topk=1, steps=5)**: single-stream median **152 tok/s** decode (mean 165,
  range 104-338, 1/10 prompts ≥300). Weights **42 GB/GPU** (halved), KV pool 58.6M tokens,
  draft-graph capture adds a step (~"up to several minutes" log, actual ~30-60s).

TP8 is ~+30% on the single-stream median — real but **not the 2.5× needed**. The 539/338 tok/s
peaks happened only on prompts where EAGLE `accept_len` spiked to 3.1-3.4; median accept_len is
flat ~2.0 across both layouts. So **single-stream speed is gated by EAGLE acceptance on real
prompts, not by TP topology**. Decode-hop reduction helps a constant factor; acceptance variance
dominates. To chase 300 tok/s the lever is the draft tree (topk/num_draft_tokens) and/or the
native MTP head, not more TP. Concurrency-aggregate is unaffected: TP4 already saturates the
node at ~1040 tok/s (c≥64), so TP8's value is single-stream latency + per-GPU memory headroom,
not aggregate throughput.

### benchmark: single-stream is ~117 tok/s on REAL prompts (not 188-244) — smoke number was synthetic-inflated
<!-- captured: 2026-06-05 | stage: 6 -->

The P0 smoke reported 188-244 tok/s single-stream warm decode. That number is **not
representative** — it was measured on repetitive/synthetic prompts, which inflate EAGLE/MTP
acceptance (the exact trap in the `feedback_synthetic_specdec_repetition` working preference).

Re-measured at the **required temperature=1.0** with 5 diverse, realistic prompts (explain
transformer attention, write binary search, training parallelism tradeoffs, French Revolution,
debug a microservice), 512 output tokens each, single-stream (c=1), warm:
- **median decode_throughput ≈ 117 tok/s** (SGLang `meta_info.decode_throughput`), range 103-246.
- median wall-clock tok/s ≈ 94 (includes prefill + HTTP).
- **EAGLE acceptance collapses on real prompts**: accept_len 1.68-2.35 (smoke claimed 1.8-4.47),
  accept_rate 0.13-0.27 (smoke 0.16-0.69). The 246 tok/s max was a single short high-acceptance
  generation (101 tok, accept_len 2.35) — not steady state.

**Implication for "beat DeepInfra 300 tok/s":** at the as-deployed EAGLE config (steps=5,
topk=1, draft=5) we are at ~117 tok/s single-stream — **~2.5x short** of the 300 tok/s target,
NOT "close pre-tuning" as the smoke note implied. Acceptance is the bottleneck; the next levers
are EAGLE tree width/depth tuning (topk, num_steps, num_draft_tokens) and verifying the draft
(MTP) head is the model's native one. Use `meta_info.decode_throughput` + real prompts + temp=1.0
as the authoritative single-stream metric — never the synthetic smoke number.

### smoke: STAGE 5 / P0 GATE PASSED — all 6 items green, MTP accepting, ~244 tok/s warm peak
<!-- captured: 2026-06-05 | stage: 5 -->

SGLang TP4 + flashinfer_trtllm + EAGLE MTP on 4×B300 (us-west-2b). Smoke results:
1. **Health**: `/health` → 200.
2. **Model registered**: `/v1/models` → `nvidia/nemotron-3-ultra`.
3. **Basic completion**: returns exactly `SMOKE_OK`, finish_reason=stop.
4. **Reasoning toggle**: `chat_template_kwargs.enable_thinking=true` populates
   `reasoning_content` separately from `content` (parser `nemotron_3` works). With
   `enable_thinking=false`, reasoning is empty.
5. **Tool call**: `finish_reason=tool_calls`, clean
   `get_weather({"city":"Paris"})` (parser `qwen3_coder` works). **Must** pass
   `chat_template_kwargs={"enable_thinking":true,"force_nonempty_content":true}` per card.
6. **MTP acceptance > 0 CONFIRMED**: SGLang decode logs show `accept len` 1.8–4.47 (of 5
   draft tokens) and `accept rate` 0.16–0.69 across requests. **gen throughput 188–244 tok/s
   single-stream** in warm steady-state (peak 243.7 tok/s). End-to-end /v1 measure was ~98
   tok/s for a 230-tok generation (2.35s incl. prefill) — the steady-state decode rate is the
   relevant single-stream number and it's already near the 300 tok/s target pre-tuning.

**Metrics gotcha**: the Prometheus `/metrics` endpoint returns 404 because this manifest does
NOT set `--enable-metrics`. MTP acceptance was read from the scheduler decode logs instead.
For P1-P4, add `--enable-metrics` so AIPerf/Prometheus can scrape acceptance + KV usage.

**Minor warning (non-blocking)**: "Tokenizer for /model is still TokenizersBackend after
retries with --trust-remote-code. Model-specific tokenizer attributes may be missing." —
generation/tool/reasoning all work, so this is cosmetic; revisit only if chat-template edge
cases appear.

STOP POINT reached per deployment instruction — gate is green, P1-P4 benchmarks NOT run.

### serving: NVFP4 MoE needs `--moe-runner-backend flashinfer_trtllm` + NO expert parallel
<!-- captured: 2026-06-05 | stage: 5 -->

The card's `--ep-size 4 --moe-runner-backend triton` does NOT serve this NVFP4 MoE on
`sglang:v0.5.12.post1-cu130`. Two separate kernel bugs:
- **`--ep-size 4`**: fails at *weight load* in `modelopt_quant.process_weights_after_loading`
  — `RuntimeError: size of tensor a (512) must match tensor b (128)`. `w13_input_scale` is
  global (512 = n_routed_experts) but `w13_weight_scale_2` is sharded to 128 per EP rank;
  the scale-2 tensor isn't sharded to match. Expert-parallel NVFP4 scale handling is broken.
- **`--moe-runner-backend triton` (with ep_size=1)**: loads fine but fails at *CUDA-graph
  capture* in `cutlass_moe_fp4` —
  `AssertionError: nx2_w1 == params.intermediate_size_per_partition * 2` (cutlass_moe.py:427).
  The triton MoE path still dispatches NVFP4 through the cutlass fp4 kernel, which asserts on
  the per-partition intermediate size.

**Working config (Stage 5 booted with this)**: `--tp-size 4` (NO `--ep-size`),
**`--moe-runner-backend flashinfer_trtllm`**, `--mamba-scheduler-strategy no_buffer`,
`--disable-radix-cache`, `--kv-cache-dtype fp8_e4m3`, EAGLE MTP (5 steps, topk 1, 6 draft).
flashinfer_trtllm is the NVIDIA-native NVFP4 Blackwell path; SGLang auto-logs "FlashInfer
TRTLLM MoE is enabled" + "Use flashinfer as attention backend on sm100" and auto-sets
`--disable-shared-experts-fusion` and disables piecewise CUDA graph (bypassed-topk
incompatible with torch.compile). Boot stats: weights 80.09 GB/GPU (cold load ~110s from
NVMe, ~64s warm), KV pool **39.75M tokens** (K/V 56.87 GB each — this model is extremely
KV-efficient), CUDA-graph capture 17s, ready in ~4-6 min total. `max_running_requests=48`,
`context_len=262144`, ~20 GB free/GPU after graphs.

### cost: B300 spot is ~$27/hr, NOT the ~$15/hr the spec estimates
<!-- captured: 2026-06-05 | stage: 0 -->

The spec Cost Considerations table lists p6-b300.48xlarge spot at ~$15/hr (and the
$170-185 session total derives from it). That number is STALE. Live spot for
p6-b300.48xlarge in us-west-2 is ~$27/hr (confirmed with the user, who set the target).
All cost math in this blueprint uses $27/hr. The spec's P3 $/1M-token break-even and the
"~$170-185 total" should be recomputed at $27/hr before any P3 cost claim is made.

**Fix**: Use $27/hr for B300 spot in all cost calculations. Flag the spec's $15/hr as
stale in the compound step so the spec gets corrected.

### infra: no mdc card and no gpu-infra b300 card exist for this model/hardware
<!-- captured: 2026-06-05 | stage: 0 -->

`mdc get nemotron-3-ultra --engine vllm` → "No card found". `mdc sync` then `mdc list`
shows only `nemotron-49b` (a different dense_nas model, not this 550B MoE). `mdc prs
nemotron-3-ultra` → no watch_prs/search_terms defined (nothing to track yet). On the GPU
side `gpu-infra card b300` → "No card found"; only `g7e`, `p5e`, `p6-b200`, `aicr` exist.
The `p6-b200` card is the closest reference (same AL2023/nodeadm/Fabric-Manager/NVSwitch
operational facts) but its `arch` field says `sm_120` which is imprecise — B200 is sm_100,
B300 is sm_103. Operationally the p6-b200 card's AMI + bootstrap + NCCL guidance still
applies.

**Fix**: Proceed from the HF model card transcribed verbatim in the spec. After this
deployment, `mdc learn` should create a nemotron-3-ultra card and `gpu-infra learn` should
seed a b300 card (compound step).

### infra: serving-config resolver does NOT model NVFP4 MoE TP divisibility
<!-- captured: 2026-06-05 | stage: 0c -->

The Stage 0c resolver's `fp8-moe-tp-divisibility` rule only fires when
`model.is_fp8` (quantization in fp8/fp8_e4m3/fp8_e5m2). This model is `nvfp4` (fp4), so
the rule is SKIPPED — the resolver will NOT verify NVFP4 MoE TP divisibility even though
`moe_intermediate_size` is declared in the sidecar. FP8 block quant uses block_n=128;
NVFP4 uses a different micro-block (16) so the 128-divisibility rule does not transfer.
The "verify rather than WARN" line in the spec's Stage 0c checklist is therefore only
partially satisfiable with the current resolver.

**Fix**: Declare `model.moe_intermediate_size` in the sidecar anyway (documents intent and
future-proofs if a fp4 rule lands), but do not rely on the resolver to catch an NVFP4 TP
mismatch. The real guard is: vLLM raises `output_size not divisible by block` at load if
the layout is wrong — the Stage 5 smoke test (weights load, no OOM/load error) is the
authoritative NVFP4-layout check. Recommend adding an `nvfp4-moe-tp-divisibility` rule to
serving-commons in the compound step.

**VALIDATED 2026-06-05 from the real on-node config.json**: `model_type=nemotron_h`,
`hidden_size=8192`, `moe_intermediate_size=5120`, `n_routed_experts=512`,
`num_attention_heads=64`, `num_key_value_heads=2`. NVFP4 `group_size=16` (from
`hf_quant_config.json`; quant_algo is `MIXED_PRECISION` — mamba in/out_proj are FP8, MoE
experts up/down_proj are NVFP4). TP-divisibility for NVFP4 is the micro-block (16), not 128:
`5120/4=1280` and `5120/8=640` are both divisible by 16 → **TP4 and TP8 are both layout-safe**.
Caveat: `num_key_value_heads=2` < TP(4/8), so KV heads replicate across ranks (standard GQA
behavior, engine-handled) — not a blocker, but means attention KV is duplicated, not sharded.

### serving-config: Stage 0c WARN — MTP + prefix-caching conflict on mamba-hybrid
<!-- captured: 2026-06-05 | stage: 0c -->

The Stage 0c resolver (exit 0, no FAILs) raised WARN `mamba-mtp-prefix-cache`: the verbatim
HF-card vLLM command enables BOTH `--enable-prefix-caching` and native MTP
(`nemotron_h_mtp`) on a mamba-hybrid (`model_type=nemotron_h`). The codified rule
(tech-stack §"Mamba hybrid architectures have different caching and speculative decoding
constraints") says MTP conflicts with vLLM's mamba 'align' mode and may require
`--no-enable-prefix-caching` to work *at all*. This is the #1 predicted failure mode for
the brand-new-model smoke test.

**Fix (fallback if engine fails at startup)**: If vLLM errors on startup with a
mamba/align/prefix-cache conflict, drop `--enable-prefix-caching` from the launch command
(or add `--no-enable-prefix-caching`) and retry. Keep MTP — it is the 300 tok/s lever and
the gate requires acceptance > 0. First attempt uses the verbatim card config as written
(card author presumably validated it); deviate only on observed failure.

Also INFO `specdec-acceptance-gate`: MTP acceptance not yet measured — smoke item #5 must
confirm it is present in /metrics and > 0 before P1-P4.

### staging: huggingface_hub v1.x removed local_dir_use_symlinks; control host too small
<!-- captured: 2026-06-05 | stage: 3 -->

Two staging gotchas discovered before bringing up the GPU node:
1. The control machine (macOS) has only ~58 GB free and no `/mnt/nvme` — it CANNOT stage
   the 352 GB / 113-shard NVFP4 repo. HF->S3->NVMe staging must run on the GPU node's NVMe
   (28 TB) or a dedicated build host, never the laptop.
2. `huggingface_hub` is v1.4.1 here. v1.11+ removed `local_dir_use_symlinks` from
   `snapshot_download` (raises TypeError) and removed `HfFolder` (renamed CLI
   huggingface-cli -> `hf`). Updated `scripts/stage-model.sh` to drop
   `local_dir_use_symlinks` (v1.x copies real files into local_dir by default).

HF repo verified accessible WITHOUT a token (OpenMDW public): sha 02c7d9e6, 113
safetensors shards, 352.4 GB total — consistent with the spec's ~335 GB estimate.

**Fix**: Run `scripts/stage-model.sh` on the B300 node (or build host). Do not pass
`local_dir_use_symlinks` on hub >= 1.x.

### blocker: Stage 4a GPU pre-flight not executable from this environment — stopped BEFORE burn
<!-- captured: 2026-06-05 | stage: 4a -->

Before scaling the ~$27/hr B300 node, confirmed two access gaps that make the MANDATORY
Stage 4a GPU health gate (and node-side staging) impossible from this control environment:

1. **No SSH key**: `~/.ssh/gpu-cluster-key` (the path the gpu-infra MCP server is
   configured to use in `.mcp.json`) does NOT exist; no `*.pem` present either. The
   gpu-infra MCP live-diagnostic tools (`discover_cluster`, `check_gpu_health`,
   `run_nccl_test`) SSH to the target node and therefore cannot connect.
2. **Node group has no remote-access path**: `ai-infra-b300-spot` has `remoteAccess: null`
   and `launchTemplate: null` — fresh B300 nodes get no SSH key pair and no remote-access
   SG. No direct SSH. Only the 2 existing system nodes are SSM-managed; the b300 NG's SSM
   posture is unverified.
3. The gpu-infra MCP tools are also not present in the current callable tool set (only
   shell/file tools), so even with a node I could not invoke the Stage 4a checks as
   designed.

Decision (cost discipline): did NOT scale the node group up. Burning $27/hr while unable to
run the mandatory GPU pre-flight gate or drive node-side staging is wasteful. Node group
left at desiredSize=0 ($0/hr). All scaffolding + gates + staging tooling are checkpointed
to disk, so nothing is lost.

**Fix / unblock (for the next session, with node access)**:
- Add an SSH key to the `ai-infra-b300-spot` node group (recreate NG with a launch
  template that sets a key pair + remote-access SG), OR confirm the SSM agent + an
  instance profile with `AmazonSSMManagedInstanceCore` so the gpu-infra MCP server can
  target it via SSM, OR place `~/.ssh/gpu-cluster-key` matching the NG's key pair.
- Then: `scripts/scale-node.sh 1`; once Ready, run `scripts/stage-model.sh` on the node
  (HF->S3->NVMe, ~352 GB); run Stage 4a (discover_cluster / check_gpu_health /
  run_nccl_test — expect 8x B300 sm_103, 0 ECC uncorrected, all_reduce >450 GB/s);
  then `terraform apply -var-file=nemotron-ultra-b300.tfvars` and `scripts/smoke-test.sh`.

### unblock: Stage 4a reachable via SSM (node role has AmazonSSMManagedInstanceCore) — prior SSH blocker resolved
<!-- captured: 2026-06-05 | stage: 4a -->

The prior session blocked Stage 4a on "no SSH key / NG remoteAccess=null". That is the
wrong access path for this node group. Verified this session:
`aws iam list-attached-role-policies --role-name ai-infra-b300-node` returns
`AmazonSSMManagedInstanceCore` (plus the 3 standard EKS worker policies). So once a b300
node boots and registers, it is reachable KEYLESSLY two ways, neither needing
`~/.ssh/gpu-cluster-key` or the gpu-infra MCP SSH tools:
1. `kubectl exec` into the vLLM pod (preferred for Stage 4a nvidia-smi/NCCL — runs in the
   CUDA container with the GPUs attached), and
2. `aws ssm start-session --region us-west-2 --target <instance-id>` to the host.

**Fix**: Do NOT recreate the node group with an SSH key. Run Stage 4a GPU checks
(nvidia-smi topo, ECC/thermal, NCCL all-reduce) as plain commands over kubectl exec / SSM.
The gpu-infra MCP SSH tooling is out of scope and not required.

### staging: stream HF->S3 one file at a time from the control host (no GPU burn, no 352GB scratch)
<!-- captured: 2026-06-05 | stage: 3 -->

Prior session marked Stage 3 blocked because the control macOS host (58 GB free, no
/mnt/nvme) and the system nodes (m6i.xlarge, ~47 GB ephemeral) cannot hold the full 352 GB
repo. But staging does NOT require holding the whole repo: download one file, `aws s3 cp`
it, delete it, repeat. Largest single shard is 8.4 GB, so peak local disk stays < 10 GB —
fits the control host trivially and burns $0 GPU. Idempotent on S3 size match so it resumes
after interruption.

**Fix**: `/tmp/stage-nemotron-stream.py` (this session) streams all 113 shards + config +
tokenizer + per-shard json + `ultra_v3_reasoning_parser.py` to
`s3://qn-sglang-models-...7/nemotron-3-ultra-nvfp4/` from the laptop, no node required.
Stage the weights BEFORE scaling the $27/hr node.
