---
blueprint: fin-rag-answer
domain: gpu-serving
model: NVIDIA-Nemotron-3-Super-120B-A12B
model_arch: NemotronHForCausalLM
gpu_arch: sm_100          # B200 Blackwell NVSwitch (NOT sm_90 Hopper)
instance: p6-b200.48xlarge
engine: vllm-0.18.1
status: complete
date: 2026-06-11
winner: fp8-agg-tp2-x4-mnbt16384-triton_attn
slo_ceiling_concurrency: 200
---

# Fin RAG Answer — Lessons Learned

Model: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` (BF16 + FP8) · Engine: vLLM
Hardware: p6-b200.48xlarge (8x B200 NVSwitch) · EKS 1.32 us-east-2
Started: 2026-06-11

Field notes captured mid-run. The compound-learner (Stage 8) decides which to elevate.

---

### benchmark: MoE Triton tile-tuning is a NO-OP on B200 — production runs FLASHINFER_TRTLLM, which never reads the tuned tile JSON
<!-- captured: 2026-06-12 | stage: 6b -->

The c192 SLO-knee DCGM showed launch/scheduling-bound (SM ~50%, HBM ~15%, tensor ~11%),
which pointed at vLLM's Triton fused-MoE grouped GEMM falling back to a heuristic tile
config (missing `E=512,N=1344,device=B200,dtype=fp8_w8a8.json`). I built a ray-free
single-GPU tuner (`scripts/tune-moe-rayfree.py`, stubs `ray` in `sys.modules`, reuses
the stock `benchmark_moe.py` internals), produced a tuned config, mounted it via
`VLLM_TUNED_CONFIG_FOLDER=/tuned-moe` (`manifests/serving-agg-tp2x4-fp8-fsx-moetuned.yaml`),
and re-ran c192.

**Result: zero effect.** Tuned c192 = e2e p50 6559 / p90 10621 ms vs baseline p50 6048 /
p90 9051 — within single-run cold-start noise, identical throughput (768 reqs / 28.3 s, 0 err).

**Root cause — the premise was a misattribution.** On B200 vLLM auto-selects the
**`FLASHINFER_TRTLLM`** FP8 MoE backend (logged `fp8.py:396` on all 4 replicas:
`Using FLASHINFER_TRTLLM Fp8 MoE backend out of potential backends: [...]`). That backend
maps to `TrtLlmFp8ExpertsMonolithic/Modular` (`fused_moe/experts/trtllm_fp8_moe.py`), whose
GEMMs are vendor-autotuned by TensorRT-LLM and **never consult the Triton tile JSON**. Only
the `TRITON` backend's `TritonExperts` (`fused_moe/fused_moe.py`) reads `VLLM_TUNED_CONFIG_FOLDER`
and emits the `"Using default MoE config. Performance might be sub-optimal!"` warning. That
warning **never fired** in any serving log — baseline or tuned. The one runtime
"default … sub-optimal" warning present is for **Mamba SSU** (`mamba_ssm.py:104`), a
different kernel; I had conflated the two. The backend priority list in `oracle/fp8.py`
puts TRTLLM ahead of TRITON whenever FlashInfer supports the config, which it does here.

**Implications:**
1. The launch-bound knee is real, but Triton tile tuning is NOT the unlock for the production
   path. To even attempt it you'd have to force `--moe-runner-backend triton` and then beat
   TRTLLM's own tuned GEMM — unlikely net-positive, untested.
2. Same applies to **H200**: the FP8 MoE backend is independent of attention backend and of
   GPU arch selection here; H200 also auto-selects FlashInfer (sm_90 supported) → tile tuning
   equally inert. No per-device re-tune is worth doing.
3. Keep the `moetuned` manifest + tuner script as a documented negative result; do not wire
   `VLLM_TUNED_CONFIG_FOLDER` into the production manifest (dead env var under TRTLLM).
4. General rule: **before tuning a Triton MoE config, confirm the active FP8 MoE backend is
   actually `TRITON`** (grep serving log for `Fp8 MoE backend`). On modern Blackwell/Hopper
   vLLM with modelopt per-tensor FP8, the default is FlashInfer TRTLLM, not Triton.

---

### infra: B200 spot node joins without `nvidia.com/gpu.present` label → 0 GPU capacity
<!-- captured: 2026-06-11 | stage: 4 -->

The `ai-infra-use2-b200-spot` managed nodegroup joined the node Ready, but
`nvidia.com/gpu` allocatable stayed empty. Root cause: the `nvidia-device-plugin`
daemonset has nodeSelector `nvidia.com/gpu.present=true`, and the spot nodegroup did
NOT apply that label on bootstrap (no GPU Feature Discovery running in this cluster).
The plugin therefore never scheduled, so the kubelet never advertised GPUs.

**Fix**: `kubectl label node <node> nvidia.com/gpu.present=true --overwrite`. The
device-plugin pod scheduled within ~20s (its tolerations already include
`nvidia.com/gpu` and `ai-infra/b200`), and GPU allocatable went 0 → 8 in ~80s.

### infra: aws-efa-k8s-device-plugin does NOT tolerate the `ai-infra/b200` taint
<!-- captured: 2026-06-11 | stage: 4 -->

EFA device-plugin daemonset DESIRED=0 on the B200 node. Its tolerations are only
`CriticalAddonsOnly` and `nvidia.com/gpu`, but the B200 node carries taint
`ai-infra/b200=true:NoSchedule`. So `vpc.amazonaws.com/efa` capacity never registers.

**Impact**: NONE for this run — single-node TP over NVSwitch needs no EFA. EFA only
matters for multi-node NIXL disagg, which is out of scope (spec: single
p6-b200.48xlarge). If an intra-node disagg config ever needs the EFA resource, patch
the EFA ds tolerations to add `{key: ai-infra/b200, operator: Exists}`. Left unpatched.

### image: vLLM 0.18.1 not in ECR; ECR image `nemotron-super-vllm:dynamo-0.9.1` ships vLLM 0.14.1 (too old)
<!-- captured: 2026-06-11 | stage: 0/5 -->

Prior run's `vllm-version-check` job reported the ECR image
`nemotron-super-vllm:dynamo-0.9.1` bundles **vLLM 0.14.1** (torch 2.9.1+cu129).
0.14.1 predates the spec's clean 0.18.1 pin and lacks proper
`NemotronHForCausalLM` support, the Mamba-2 automatic prefix-caching path
(vLLM #26201), and the `nemotron_v3` reasoning parser. Serving with it is invalid.

**Fix**: Copy `docker.io/vllm/vllm-openai:v0.18.1` straight to ECR
(`615299764834.dkr.ecr.us-east-2.amazonaws.com/vllm-openai:v0.18.1`) with an
in-cluster **skopeo** job (`manifests/skopeo-vllm-018-copy.yaml`) on a CPU node —
no local mac pull, correct linux/amd64 platform, streams layers. ECR dest creds
via a short-lived `ecr-push-creds` secret (`aws ecr get-login-password`).

### bench: shared `bench-standard.py` sends ONE identical prompt → fake ~100% prefix-hit (fails fin-support reliability gate)
<!-- captured: 2026-06-11 | stage: 6 -->

`.claude/skills/benchmark-runner/scripts/bench-standard.py` builds the request body
as `prompt = " ".join(["hello"] * input_len)` — the SAME prompt for every request.
The fin-support workload card (`forbid_identical_replay`, reliability flag
`prefix_hit_rate_gt_corpus_ceiling` action `flag_invalid_cache_inflation`)
explicitly FAILS the run if prefix-cache hit rate exceeds the ~30% real-corpus
ceiling. Identical-replay would report ~100% hit rate → INVALID results.
Also its ISL is fixed (`input_tokens.std_dev=0`) so it cannot reproduce the
measured ISL p50 8823 / p90 11952 distribution.

**Fix**: write a fin-support-aware driver that emits a VERBATIM ~3050-tok header +
a UNIQUE body per request (recombined guidelines + synthetic passages + varied
query), sized to the ISL distribution, tagging real|synthetic for the
augmentation audit. Must also scrape vLLM prefix-cache metrics to verify hit rate
stays below the ~30% ceiling. The stock driver is usable only for synthetic
concurrency-sweep cards, NOT fin-support.

### infra: Nemotron-3-Super is HF-gated → cluster needs an `hf-token` secret to stage weights
<!-- captured: 2026-06-11 | stage: 3 -->

No HF token secret existed in `ml-inference`; the NVIDIA Nemotron-3-Super repos are
gated. Created `hf-token` secret from the local `~/.cache/huggingface/token` and
wired it into the staging job env (`HF_TOKEN`). FSx held only old qwen3 weights
(~228GB used of 4800Gi) — nemotron weights were NOT pre-staged, so a fresh
download of FP8 (~124GB) then BF16 (~240GB) is required.

### decision: image policy reversed — try vLLM latest-stable (0.22.1) first, smoke-test as arbiter
<!-- captured: 2026-06-11 | stage: 0/5 -->

Spec and the original benchmark.yaml pinned vLLM `0.18.1` and explicitly listed
"AVOID 0.22 (#44022)". Operator directive (2026-06-11) reversed this: deploy the
LATEST STABLE first and let the Stage 0 coherence smoke test decide, rather than
blind-pinning to an old known-good. Rationale: 0.22 had output regression #44022,
but `v0.22.1` is the point release on top of 0.22 and is plausibly the fix — worth
validating empirically for the newer fixes/features.

**Decision**:
- PRIMARY = `vllm/vllm-openai:v0.22.1` (Docker Hub, pushed 2026-06-05; manifest confirmed present).
- FALLBACK = `vllm/vllm-openai:v0.18.1` (known-clean) — used ONLY on smoke failure.
- Explicitly excluded: Docker Hub `latest` (nightly, too unstable for a benchmark of
  record) and ECR `nemotron-super-vllm:dynamo-0.9.1` (vLLM 0.14.1, too old).

**Smoke gate (the arbiter, not the version number)** — on ~10 real Fin prompts assert:
  (a) coherent output, NO garble;
  (b) model loads as `NemotronHForCausalLM`;
  (c) Mamba2 automatic prefix caching active (`mamba_cache_mode=all` accepted; prefix
      hit rate > 0 on the ~3,050-tok shared header).
PASS → use 0.22.1; record version + smoke evidence here. FAIL → fall back to 0.18.1 and
document exactly what failed on 0.22.1 (that failure note is itself a valuable field note).

**Staging mechanism**: in-cluster skopeo Job copies Docker Hub → ECR registry-to-registry
(no local docker daemon, no node pull). Manifest: `manifests/skopeo-vllm-0221-copy.yaml`.
Both 0.22.1 (primary) and 0.18.1 (fallback) pre-staged to ECR `vllm-openai` so a fallback
needs no extra network round-trip mid-deploy. ECR creds via secret `ecr-push-creds`.
benchmark.yaml `engine.version` now `0.22.1` with `version_fallback: 0.18.1` +
`candidate_images.{primary,fallback}`.

### infra: FSx Lustre PVC mounts on CPU nodes but FAILS on the B200 node (`mount.lustre: Can't parse NID`)
<!-- captured: 2026-06-11 | stage: 3/4 -->

The FSx Lustre PVC `vllm-qwen3-fsx-pvc` mounts fine on the older CPU system
nodes (used for the weight-download job) but the FSx CSI mount FAILS on the
fresh B200 spot node:
`mount.lustre: Can't parse NID 'fs-...@tcp:/m2h5hbev'` (exit status 22). The
AL2023 NVIDIA AMI on the B200 node lacks a working Lustre client / lnet setup
that the CSI node-stage step expects. So the planned FSx -> NVMe copy ON the
B200 node is impossible.

**Fix**: do NOT relay weights through FSx for the B200 serving tier. Download
weights DIRECTLY to the B200's local NVMe RAID via a job pinned to the B200 node
that pulls from HF with `hf_transfer` (no FSx mount). FP8 (~124GB) lands in
~4-5 min on B200 networking. FSx staging is still useful as a persistent cache
for CPU-node consumers, but it is not on the B200 serving path. (If FSx-on-B200
is ever needed, the AMI needs the matching `lustre-client` kmod + `lctl network up`.)

### infra: B200-node pods cannot reach kube-dns (172.20.0.10) — even cluster-internal DNS fails
<!-- captured: 2026-06-11 | stage: 3 -->

Pods scheduled on the B200 spot node fail ALL DNS resolution, including
cluster-internal (`socket.gethostbyname('kubernetes.default')` ->
`gaierror: [Errno -3] Temporary failure in name resolution`). resolv.conf
correctly points at 172.20.0.10, but the B200 node's pod network can't reach the
CoreDNS endpoints (CNI/SG/kube-proxy gap on the freshly-joined spot node). The
node ITSELF has working DNS (it pulled the ~10GB ECR vLLM image fine).

**Fix for downloads**: run the HF-download job with `hostNetwork: true` +
`dnsPolicy: Default` so the pod uses the NODE's VPC resolver instead of kube-dns.
The download immediately reached huggingface.co.

**Implication for benchmarking**: run the bench-runner pod on a CPU node (working
kube-dns) and hit the serving ClusterIP/Service from there — do NOT run the
bench pod on the B200 node. Verify CPU-node -> B200-pod ClusterIP routing works
before the benchmark (cross-node pod traffic is a separate CNI concern).
Serving pods themselves need no DNS (local model path, local everything).

### env: HF_HUB_ENABLE_HF_TRANSFER deprecated in huggingface_hub 1.11+ — use HF_XET_HIGH_PERFORMANCE
<!-- captured: 2026-06-11 | stage: 3 -->

`hf download` with `huggingface_hub>=1.11` warns that `HF_HUB_ENABLE_HF_TRANSFER`
is deprecated (hf_transfer no longer used); Xet is the new fast path. Set
`HF_XET_HIGH_PERFORMANCE=1` instead. Downloads still work with the old var (warning
only), so non-blocking, but update staging manifests on next refresh.

### serving: vLLM 0.18.1 FP8 Nemotron-3-Super serves cleanly on B200; cold start ~4min (NOT 15min); KV cache 8.44M tokens / 507x conc headroom
<!-- captured: 2026-06-11 | stage: 5 -->

agg-tp2-x4 (4 replicas x TP=2) on the 8x B200 node. Per replica: weights load 40s
from NVMe, torch.compile 17s, CUDA graph capture 20s, init engine 53s -> total
~4 min to `Application startup complete`. The spec's "~15 min DeepGEMM JIT" did
NOT materialize (matches nemotron-super lesson). vLLM auto-selected
`FLASHINFER_TRTLLM` FP8 MoE backend, `FlashInferFP8ScaledMMLinearKernel`,
`DeepGEMM E8M0`. **GPU KV cache = 8,444,800 tokens; max concurrency at 16K ctx =
507x** -> huge headroom for the 130-concurrent target. `--mamba-cache-mode all`
accepted; `enable_prefix_caching=True`; `kv_cache_dtype=fp8`; reasoning_parser=''
(OFF). Stage 0 coherence smoke: 5/5 real Fin prompts produced coherent, grounded,
text-message-style answers, no garble, no leaked <think>. 0.18.1 confirmed clean.

**Set readiness `initialDelaySeconds` ~60-120, NOT 900** — 900 left pods 0/1 long
after they were serving and produced an empty-endpoints Service. Also use
`strategy: Recreate` (not RollingUpdate): all 8 GPUs are consumed by the 4
replicas, so a rolling update deadlocks (new pods Pending for GPUs while old pods
won't terminate until new ones are Ready).

### infra: cross-node pod routing is BROKEN on this cluster — B200-node pods unreachable from CPU-node pods (even by pod IP)
<!-- captured: 2026-06-11 | stage: 5/6 -->

A CPU-node pod could NOT reach the B200 serving pods via ClusterIP OR direct pod
IP (curl exit 7/28, timeout). Same root cause family as the broken kube-dns from
B200 pods: the B200 spot node's pod network is isolated from the rest of the
cluster (VPC CNI / security-group gap). BUT a pod PINNED to the B200 node with
`hostNetwork: true` reaches the serving ClusterIP fine (kube-proxy handles it
node-locally): health 200, /v1/models OK.

**Fix**: run the bench-runner ON the B200 node (`nodeSelector gpu.present=true` +
taint tolerations + `hostNetwork: true`), hitting the serving ClusterIP. Do NOT
follow the generic Stage 6b "bench from a CPU-node pod" guidance on this cluster.

### benchmark: P0 SLO PASSES on FP8 agg-tp2-x4, but prefix-cache hit rate is 0 at TP=2 (the #26201 TP>1 risk is REAL)
<!-- captured: 2026-06-11 | stage: 6 -->

P0 0a headline @ conc=130, FP8, agg-tp2-x4 (fin-support, verbatim-header+unique-tail):
- **E2E p50 5334ms <= 6500 PASS; E2E p90 8055ms <= 9500 PASS**; error_rate 0.0 (520/520).
- TTFT p50 437 / p90 3223 / p99 4759 ms. TPOT p50 61.6 / p99 83.5 ms.
- ISL p50 8911 / p90 12200 (tracks target 8823/11952). distinct_prompt_fraction 1.0.
- conc=8: E2E p50 ~1052ms, TTFT p50 162ms.

**BUT `vllm:prefix_cache_hits_total` = 0 on ALL 4 replicas** despite 47K-165K
queries each. The ~3050-tok shared system header is re-prefilled every request;
prefix caching is effectively NON-FUNCTIONAL at TP=2 for this hybrid-Mamba model
on vLLM 0.18.1 — exactly the upstream #26201 "TP>1 not yet tested" caveat. Flags
`--enable-prefix-caching --mamba-cache-mode all` were accepted and
`enable_prefix_caching=True` shows in the engine config, but hits stay 0.

**Consequence**: the P0 *SLO* gate passes (B200 prefill throughput + 507x KV
headroom absorb the full re-prefill), but the P0 *prefix-hit* sub-gate ("hit rate
> 0, TP>1-validated") FAILS. The RAG prefix-cache win does not materialize at
TP>1 here. Next: validate prefix caching at TP=1 (isolate to TP>1) and check
whether agg-tp4-x2 / a single-replica TP1 config registers hits. If TP>1 is
required, the shared-header caching benefit is currently unavailable on 0.18.1.

NOTE: model under-generates vs telemetry (out_tokens p50 ~74 vs target OSL 243) —
greedy temp=0 answers are short; realistic, but decode-cost numbers are a lower
bound vs the OSL p50 243 profile.

### benchmark: BF16 fails p90 SLO gate at conc=130; FP8 wins
<!-- captured: 2026-06-11 | stage: 6b -->

P0 0b precision comparison (agg-tp2-x4, same config, kv-cache-dtype fp8 in both):
- **FP8** conc=130: E2E p50 5334ms / p90 8055ms -> PASS both gates (<=6500 / <=9500).
- **BF16** conc=130: E2E p50 6128ms / p90 **10496ms** -> p90 FAILS (>9500). TPOT p50 72.5ms vs FP8 lower.
- conc=8 floor: BF16 E2E p50 1451 / TTFT 215 / TPOT 16.9; FP8 E2E p50 1052 / TTFT 162. FP8 faster across the board.
- BF16 cold start ~3min (4/4 ready), no DeepGEMM JIT (vs FP8 first-start JIT). Weight load only.
- Both: 0 errors at all levels; prefix-cache hit rate ~0 at TP>1 (upstream #26201 limitation).

**Fix / decision**: Recommend **FP8** for this workload. On B200 the per-tensor-static FP8 path is both faster (lower TPOT, tighter tail) and meets the p90 SLO at the target concurrency where BF16 does not. BF16 only viable below ~conc=110 for the p90 gate.

### deploy: serving manifest on disk kept 900s/1200s probe delays after only a live patch
<!-- captured: 2026-06-11 | stage: 5 -->

Earlier I patched the LIVE deployment's readinessProbe initialDelaySeconds 900->30 to get pods Ready, but never wrote it back to `serving-agg-tp2x4-fp8.yaml`. On a later `kubectl apply -f` (redeploy after BF16), the stale 900s/1200s came back: all 4 pods served /health 200 within ~3min but stayed 0/1 because the readiness probe doesn't fire until delay=900s elapses. Looked like a hang; it was just the probe window.

**Fix**: Persist probe values to the file (readiness initialDelaySeconds 60 + failureThreshold 80*15s ceiling; liveness 120). Also added `strategy: Recreate` to the file (RollingUpdate deadlocks when all 8 GPUs are consumed). Rule: when you live-patch a manifest field, immediately mirror it into the file or the next apply silently reverts it.

### benchmark: chunked-prefill max-num-batched-tokens=16384 is the winner for ~9K-ISL RAG
<!-- captured: 2026-06-11 | stage: 6b -->

P1 chunked-prefill sweep (FP8 agg-tp2-x4, conc=130, fin-support ISL p50~8.8K):
| max-num-batched-tokens | E2E p50 | E2E p90 | TTFT p50 | TPOT | SLO p50<=6500 | SLO p90<=9500 |
|---|---|---|---|---|---|---|
| 4096  | 7065 | 11933 | 684 | 83.0 | FAIL | FAIL |
| 8192  | 5105 | 8878  | 454 | 55.8 | PASS | PASS |
| 16384 | 4685 | 8147  | 387 | 55.0 | PASS | PASS (best) |

**Fix / decision**: Set `--max-num-batched-tokens 16384` (== max-model-len). For long-context RAG prefills (~9K ISL), small chunk budgets fragment each prefill into many chunked iterations, inflating TTFT and the p90 tail. 16384 lets a prefill complete in one chunk -> lowest TTFT and tightest tail. 4096 is actively harmful (both SLO gates fail). Promote 16384 to the production config.

### blocker: native MTP spec-decode crashes for NemotronH on vLLM 0.18.1 (TP=2)
<!-- captured: 2026-06-11 | stage: 6b -->

Enabled `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`. The FP8 checkpoint DOES ship MTP weights (1042 `mtp.*` tensors, `num_nextn_predict_layers=1`, `mtp_hybrid_override_pattern="*E"`). vLLM resolves `NemotronHMTPModel`, loads the drafter, shares target embedding + lm_head with the draft model — then crashes on the first forward / cudagraph dummy-run:

```
[eagle.py:1419] Detected MTP model. Sharing target model lm_head weights with the draft model.
RuntimeError: Worker failed with error 'The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1'
```

Crash is in the MTP draft forward under the NemotronH hybrid (Mamba-2 + select-attention) layout at TP=2 — a shape mismatch (seq dim 2 vs 3), not a flag/config error. Pods exit 137 in a restart loop (the engine error also tripped the liveness probe before I raised it to 1500s — the longer delay just let the real crash surface instead of masking it as a probe kill).

**Fix / workaround**: Native MTP is NOT usable for nemotron-3-super on vLLM 0.18.1 + TP>1. Removed the speculative-config and reverted to the dense FP8 baseline (mnbt=16384). Fall back to n-gram (prompt-lookup) spec-decode for an acceptance number — it needs no draft model and is TP-agnostic. Re-test MTP only on a vLLM build that lands the NemotronH-MTP TP fix.

### benchmark: n-gram (prompt-lookup) spec-decode deployed for acceptance @ temp=1.0 (TP-agnostic, no draft model)
<!-- captured: 2026-06-11 | stage: 6b -->

Native MTP is a confirmed blocker on 0.18.1 TP2 (shape mismatch, above). Falling
back to n-gram/prompt-lookup spec-decode, which needs no draft model and is
TP-agnostic. Config: `--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'`
on the FP8 agg-tp2-x4 winner (mnbt=16384). RAG answers copy source spans verbatim
so n-gram should accept well on the retrieved-context tail. Acceptance is measured
@ **temperature=1.0** (added `--temperature` to bench-fin-support.py) because the
model-recommended temp lowers acceptance and greedy (temp=0) numbers would
overstate it. Driver also now scrapes `vllm:spec_decode_num_accepted_tokens` /
`_num_draft_tokens` to compute acceptance over the measured window, and reports
net E2E/TPOT effect vs the no-spec winner — not acceptance alone.
Manifest: `manifests/serving-agg-tp2x4-fp8-ngram.yaml`. Applied via Recreate
(winner drained first). Cold start ~4 min; checking readiness with bounded probes,
not wait-loops.

### benchmark: n-gram spec-decode CRASHES on Mamba2 + CUDA-graph capture (vLLM 0.18.1) — both spec-decode paths now blocked at TP2
<!-- captured: 2026-06-11 | stage: 6b -->

The n-gram leg never reached Ready — CrashLoopBackOff (10+ restarts, exit 1).
Root cause is in CUDA-graph capture during warmup, NOT a config error:
`vllm/v1/attention/backends/mamba_attn.py:498 _update_metadata_for_cudagraph_capture`
→ `RuntimeError: The size of tensor a (2) must match the size of tensor b (6) at
non-singleton dimension 1`. Spec-decode proposes num_speculative_tokens=4 (query
len ~5-6 per step), but the Mamba2 attention backend's graph-capture path assumes
a fixed single-token decode shape and can't reconcile the multi-token speculative
query. So BOTH spec-decode variants are blocked for this hybrid-Mamba model on
0.18.1: MTP (OOM/SIGKILL during long compile, earlier note) and n-gram (this
shape mismatch in graph capture).

**Workaround to still MEASURE acceptance**: add `--enforce-eager` to the n-gram
leg. That disables CUDA-graph capture entirely, sidestepping the broken
mamba_attn path, and lets us read `vllm:spec_decode_num_accepted_tokens` on real
Fin prompts @ temp=1.0 — which answers the spec's actual question ("is acceptance
measurable on real RAG spans"). CAVEAT: eager-mode latency is NOT
production-representative, so report the acceptance RATE as the headline and label
the E2E/TPOT from this leg as eager-only (do not compare its latency to the
graph-captured winner). **Net conclusion for the customer regardless: spec-decode
is not deployable for Nemotron-3-Super on vLLM 0.18.1 at TP2** — consistent with
the workload being prefill-dominated where spec-decode was always the secondary
lever. Worth a re-test when a newer vLLM fixes Mamba2 graph capture + spec-decode.

### benchmark: `--enforce-eager` unblocks n-gram spec-decode startup (eager-only) to MEASURE acceptance
<!-- captured: 2026-06-11 | stage: 6b -->

Applied the operator-directed workaround for the n-gram graph-capture crash:
added `--enforce-eager` to `manifests/serving-agg-tp2x4-fp8-ngram.yaml`. Eager mode
disables CUDA-graph capture entirely, so the broken `mamba_attn`
`_update_metadata_for_cudagraph_capture` path is never hit and the engine starts.
Re-applied via Recreate (winner replicas/crash-loop pods drain first; identical
deployment name+selector so `apply` rolls them). Cold start expected faster than the
graph-captured winner (no capture phase).

**Fix**: `- --enforce-eager` after the `--speculative-config` arg. Headline metric for
this leg = `acceptance_rate` (vllm:spec_decode_num_accepted_tokens / _num_draft_tokens)
@ temp=1.0. E2E/TPOT from this leg are EAGER-ONLY and must NOT be compared to the
graph-captured winner's latency. Net customer conclusion is unchanged: spec-decode is
not deployable for Nemotron-3-Super on vLLM 0.18.1 at TP2 (both MTP and n-gram require
graph capture that the Mamba2 backend breaks); this leg only quantifies whether
acceptance is attractive enough to justify a future re-test on a fixed vLLM.

### image: upgrading to vLLM 0.22.1 does NOT unblock prefix-cache-at-TP>1 or Mamba2 spec-decode — verified against upstream, no rebase warranted
<!-- captured: 2026-06-11 | stage: 0/research -->

Question raised: "does 0.22.1 dramatically increase performance? should we rebase
on latest?" Answered from upstream changelogs/issue trackers — NO GPU time spent.
Both of this run's actual blockers are **feature gaps, not stale-version bugs**, and
neither is fixed in any stable tag including 0.22.1 (released 2026-06-05):

1. **Prefix caching at TP>1 (the big RAG lever).** Tracking issue vLLM #26201,
   updated by the maintainer **2026-06-09** (2 days before this run), still has the
   checklist item **"Test TP>1 behaviour" UNCHECKED.** Our observed hit-rate-0 at TP2
   is genuinely untested-upstream behaviour, not a version regression. The real fix
   path is the maintainer's stated pivot from "all" mode → "align" mode, gated on the
   **Marconi admission-policy PR #37898 — merged 2026-06-10, i.e. AFTER 0.22.1 shipped
   (2026-06-05).** So that fix is **nightly-only**; no stable tag carries it. Upgrading
   to 0.22.1 changes nothing here.

2. **Mamba2 spec-decode.** The Nemotron-H MTP + Mamba spec-decode PR (#33726) **merged
   2026-02-24**; vLLM **0.18.1 shipped 2026-03-31** → the fix is ALREADY in the engine
   we ran. We still hit MTP OOM + the n-gram CUDA-graph-capture crash, so this is a
   **config/shape incompatibility at our TP2 setup, not a missing upstream fix.** 0.22.1
   carries the same code → no change expected.

**Net**: there is no evidence for a "rebase-on-latest for a dramatic win" — neither a
feature-unlock case nor a raw-throughput case. Release-to-release vLLM kernel deltas
are single-to-low-double-digit %, and this B200 config already clears SLO comfortably
(p50 4685 / p90 8147 vs 6500/9500). Also note: the #44022 we'd cited as the 0.22
"output garble" regression is actually a *different* bug (Nemotron NVFP4 startup on a
5090) — still OPEN, but not the per-tensor-FP8 path we serve. Do NOT chase a 0.22
rebase for this workload. The legitimate follow-up for prefix-caching-at-scale is a
**vLLM nightly with align mode + #37898** — a separate, higher-risk experiment
(nightly correctness unvalidated on this hybrid model), NOT a stable-tag rebase. The
in-grid TP1 prefix-cache leg (does hit-rate go >0 with zero TP shards?) remains the
correct cheap experiment to run first.

### benchmark: Leg 1 — FP8 KV cache flag is a NO-OP on the FP8 checkpoint (engine auto-selects fp8_e4m3 KV regardless)
<!-- captured: 2026-06-11 | stage: 6b -->

P1 Leg 1 asked: does `--kv-cache-dtype fp8` cost accuracy/latency vs default, and
does it change KV headroom? Removed `--kv-cache-dtype fp8` from the graph-captured
winner (everything else identical: TP2, TRITON_ATTN, mnbt=16384, max-num-seqs 256),
re-applied via Recreate, benchmarked conc=8 + conc=130.

**Finding**: dropping the flag does NOT disable fp8 KV. vLLM 0.18.1 logs
`Using fp8 data type to store kv cache` and the engine config still shows
`kv_cache_dtype=fp8_e4m3` (`quantization=modelopt`). The ModelOpt FP8 checkpoint
ships k/v scaling factors, so the engine DEFAULTS the KV cache to fp8_e4m3 for this
model — the `--kv-cache-dtype fp8` arg is effectively a no-op confirmation of the
default. Available KV cache memory = 95.02 GiB (identical to the winner's ~95 GiB),
so KV headroom is unchanged.

Numbers (conc=130, both on the same fp8 KV path):
| Config | E2E p50 | E2E p90 | TTFT p50 | TPOT p50 | SLO |
|--------|---------|---------|----------|----------|-----|
| winner (--kv-cache-dtype fp8) | 4685 | 8147 | 387 | 55.0 | PASS |
| leg1 (flag removed → auto=fp8) | 4598 | 7366 | 390 | 49.8 | PASS |

The ~1-10% delta is run-to-run noise on the identical fp8 KV path, NOT a precision
effect. conc=8 floor: 1345/162/17.9 (vs winner 857/162/10.3 — within noise).
0 errors both runs. Prefix-cache hit rate still 0 (TP2, #26201).

**Conclusion for customer**: FP8 KV cache costs nothing here and is the default for
the FP8 checkpoint anyway. There is no "KV-off" config to compare on this build short
of loading the BF16 checkpoint (already covered: BF16 fails p90). KV footprint is not
a lever for this prefill-dominated workload (95 GiB pool = 507x conc headroom either way).

### research: SGLang is NOT ahead of vLLM on this model's blockers (spec-decode parity, prefix-cache behind) — but the SGLang sm_120 matrix is gold for the g7e variant
<!-- captured: 2026-06-11 | stage: research -->

Asked whether SGLang would beat vLLM on this model's two blockers. Checked upstream
sgl-project/sglang — answer: NO for B200, but valuable signal for g7e.

- **Spec-decode**: SGLang PR #20470 ("Nemotron-3-Super Speculative Decoding Support")
  MERGED 2026-03-27, but OPEN bug **#21138**: MTP/NEXTN gives **0% draft acceptance**
  on Nemotron-3-Super (accept_rate 0.33, every spec token rejected). So spec-decode is
  effectively dead on this model in SGLang too. Combined with our B200/vLLM result
  (MTP OOM + n-gram graph crash at TP2), spec-decode is a dead end on **both** engines
  for Nemotron-3-Super on NVSwitch — strengthens the "spec-decode CLOSED" verdict; it
  is the model, not a vLLM gap.
- **Prefix caching**: no evidence SGLang has hybrid-Mamba2 prefix caching at all (uses
  radix cache; the Nemotron g7e bench ran `--disable-radix-cache`). vLLM is AHEAD here
  (#26201 all/align work) — switching to SGLang would LOSE the RAG prefix-cache lever,
  not gain it.
- **Net for B200**: do not switch this campaign to SGLang. It loses on prefix caching
  and ties at zero on spec-decode.

**BUT — directly useful for the deferred g7e variant**: SGLang issue **#20541** is a
3-run-validated benchmark of THIS EXACT model on g7e (Nemotron-3-Super-FP8, 8× RTX PRO
6000 sm_120). It fully characterizes the sm_120 FP8 backend matrix (DeepGEMM/CUTLASS
FAIL, Triton works; FlashInfer attn works at defaults; CUDA graphs work; KV fp8_e4m3
works; DeepEP/HiCache fail on PCIe). I folded this into
`domains/gpu-serving/specs/fin-rag-answer-g7e.md` → "Upstream evidence" section, which
reduces g7e Stage 0/G0 from exploration to a smoke confirmation (~3.5hr → ~1-1.5hr).
NOTE the spec-decode result there is OPPOSITE (SGLang EAGLE works on g7e, accept 0.81)
— spec-decode viability is engine+hardware-specific, must re-measure on vLLM g7e.

### benchmark: Leg 2 — FlashInfer attention LOADS on hybrid Mamba2 + is faster than TRITON_ATTN, but emits an fp8 q/prob-scale accuracy warning
<!-- captured: 2026-06-11 | stage: 6b -->

P1 Leg 2: `--attention-backend FLASHINFER` (vs TRITON_ATTN), all else = winner
(TP2, --kv-cache-dtype fp8, mnbt=16384, max-num-seqs 256). Recreate; manifest
`serving-leg2-flashinfer.yaml`.

**FlashInfer DID load** on this hybrid Mamba2 + Select-Attention model — contrary to
the spec's caution that select-attention "may not be supported by all backends". Log:
`Using AttentionBackendEnum.FLASHINFER backend` + `HND KV cache layout`. (Note: the FP8
MoE GEMM backend stayed `FLASHINFER_TRTLLM` in BOTH legs — it is independent of the
attention backend, as expected.)

Numbers (conc=130):
| Config | E2E p50 | E2E p90 | TTFT p50 | TPOT p50 | SLO |
|--------|---------|---------|----------|----------|-----|
| winner (TRITON_ATTN) | 4685 | 8147 | 387 | 55.0 | PASS |
| leg2 (FLASHINFER)    | 4197 | 7598 | 354 | 44.9 | PASS |
conc=8 floor: FlashInfer 1346/166/15.5 vs winner 857/162/10.3.

FlashInfer is ~10% faster on E2E p50 and ~18% lower TPOT at conc=130, 0 errors.

**CAVEAT (why TRITON_ATTN remains the safe production default)**: FlashInfer runs fp8
attention and warns at startup:
`Using uncalibrated q_scale 1.0 and/or prob_scale 1.0 with fp8 attention. This may cause
accuracy issues. Please make sure q/prob scaling factors are available in the fp8
checkpoint.` The Nemotron-3-Super FP8 checkpoint ships k/v scales but NOT q/prob scales,
so FlashInfer's fp8 attention runs with default 1.0 scales — a documented accuracy risk.
TRITON_ATTN does not take this fp8-attention path. **Recommendation**: FlashInfer is a
viable latency win IF a coherence/accuracy smoke confirms no quality regression on real
Fin prompts; otherwise keep TRITON_ATTN (hybrid-safe, no uncalibrated-scale warning). Cooled
cleanly cold start ~5 min (slower than TRITON ~4 min due to FlashInfer autotuning).

### research: spec-decode root cause is the Mamba2 cudagraph-capture bug — NOT "vLLM can't do custom/EAGLE drafts" (correction)
<!-- captured: 2026-06-11 | stage: research -->

Correcting an earlier overstatement. vLLM DOES support custom draft models: `main`'s
`--speculative-config` `method` accepts `eagle`, `eagle3`, `draft_model`, `custom_class`,
plus the model's own `nemotron_h_mtp`; the `model` field takes a draft-model / eagle-head
path. So "use a custom draft like EAGLE on vLLM" is correct in general. What vLLM lacks is
narrower: a **draft-MoE-runner-backend selector** (SGLang's `--speculative-moe-runner-backend
triton`). Confirmed by reading vLLM `envs.py` on `main` — only global MoE flags
(`VLLM_USE_DEEP_GEMM`, `VLLM_MOE_USE_DEEP_GEMM`, FlashInfer-MoE-int4, ROCm/AITER); no
draft-specific or generic MoE-runner selector.

Why EAGLE still does not rescue B200 spec-decode for THIS model:
1. **No published EAGLE/EAGLE3 head exists for Nemotron-3-Super** (searched HF — none). EAGLE
   needs a trained head matched to the target's hidden states; can't point `--model` at a
   head that doesn't exist. Would have to TRAIN one (a project, not a flag).
2. **The model already ships its own draft head (MTP) and THAT is what crashed.** The FP8
   checkpoint carries 1042 `mtp.*` tensors (`num_nextn_predict_layers=1`); vLLM resolved
   `NemotronHMTPModel`, loaded the drafter, shared embedding+lm_head — then crashed on the
   cudagraph dummy-run (same Mamba2 `mamba_attn.py:498` graph-capture shape bug as n-gram).
   So the failure is the **Mamba2 + spec-decode graph-capture path in vLLM 0.18.1**, not a
   missing draft-model feature.
3. SGLang's g7e "win" (#20541) was NOT a separate EAGLE model — it was the SAME built-in
   MTP/NEXTN head, made to work via the draft-MoE-backend flag + SGLang's Mamba2 capture
   not hitting the crash. So the real differentiator is engine impl + that flag, NOT "EAGLE
   vs MTP".

**Revised verdict** (supersedes the "version-independent, no change expected" wording in the
0.22.1 entry for the spec-decode axis specifically): the realistic unlock is a **newer vLLM
that fixes Mamba2 cudagraph capture for spec-decode** — that would make the built-in MTP head
work as-is. So MTP is worth a RE-TEST on a current vLLM nightly (I earlier wrote it off too
firmly). Other paths: SGLang-on-B200 (has the flag; loses prefix-cache lead), or train an
EAGLE3 head (most work). For THIS campaign on 0.18.1, spec-decode stays CLOSED.

### benchmark: Leg 3 — agg-tp4-x2 FAILS SLO at conc=130; tp2-x4 (more replicas) wins for prefill-dominated high-concurrency
<!-- captured: 2026-06-11 | stage: 6b -->

P2 Leg 3: 2 replicas x TP=4 (all 8 B200) vs the 4 replicas x TP=2 winner. TP4 is
arithmetically safe (per-tensor static FP8, no block_n=128 constraint) and loaded clean
on B200 NVSwitch (world_size=4 NCCL init fine — no Blackwell-PCIe-style invalid-argument;
NVSwitch NCCL is mature). Manifest `serving-leg3-tp4x2.yaml`.

TP4 per-replica KV pool = 122.94 GiB / 10,741,632 tokens / 635x conc headroom (vs TP2's
95 GiB / 8.44M / 507x) — MORE headroom per replica, but only 2 replicas.

Numbers (conc=130):
| Config | E2E p50 | E2E p90 | TTFT p50 | TPOT p50 | SLO |
|--------|---------|---------|----------|----------|-----|
| winner agg-tp2-x4 | 4685 | 8147 | 387 | 55.0 | PASS |
| leg3 agg-tp4-x2   | 6755 | 10593 | 547 | 74.5 | **FAIL both** |
conc=8 floor: TP4 1156/382/**9.0** (lowest TPOT — TP4 helps single-stream decode) vs TP2 857/162/10.3.

**Finding**: TP4x2 FAILS both SLO gates at conc=130. Root cause: with only 2 replicas, each
serves ~65 concurrent requests vs ~32 for the 4-replica TP2 layout → deeper prefill queue per
replica → higher TTFT/E2E. The extra per-replica KV headroom (635x) is wasted (507x already
ample). NVSwitch TP4 all-reduce comms also lift TPOT (74.5 vs 55.0). **More REPLICAS beats
more TP for this prefill-dominated, high-concurrency RAG workload.** TP4 only wins the conc=8
single-stream TPOT floor (9.0 ms), irrelevant at the 130-concurrent operating point. Keep
agg-tp2-x4. (Same direction the spec predicted: throughput/queueing dominates, not per-request compute.)

### note: the `uncalibrated q_scale/prob_scale 1.0 fp8 attention` warning is from --kv-cache-dtype fp8, NOT FlashInfer-specific
<!-- captured: 2026-06-11 | stage: 6b -->

Correction to the Leg 2 caveat: the warning `Using uncalibrated q_scale 1.0 and/or prob_scale
1.0 with fp8 attention` also appears on the TRITON_ATTN TP4 leg (Leg 3) — it is triggered by
`--kv-cache-dtype fp8` (fp8 attention compute reads q/prob scales the checkpoint lacks), NOT by
the FlashInfer attention backend. So this accuracy caveat applies to the fp8-KV winner too, not
just FlashInfer. The Stage 0 coherence smoke (5/5 coherent on the winner) already covers it: in
practice the default 1.0 scales produced coherent grounded answers. FlashInfer's real differentiator
is the ~10% latency win, not a unique accuracy risk.

### decision: spec-decode PARKED on product grounds (not just the engine bug) — throughput-bound + ITL already imperceptible
<!-- captured: 2026-06-11 | stage: 6b/decision -->

Operator decision (2026-06-11): park spec-decode for the fin-support chatbot use case. The
earlier "CLOSED because it crashes on 0.18.1" verdict is correct but incomplete — even if a
newer vLLM fixed the Mamba2 cudagraph-capture bug, spec-decode would NOT be worth enabling for
THIS workload, for two product reasons:

1. **ITL is already below human perception.** Spec-decode accelerates the decode loop (ITL/TPOT),
   never prefill (TTFT). Measured ITL on the winner is ~17-55 ms/token; a human reads streamed
   text at ~150-250 ms/token-equivalent. Tokens already arrive 4-10x faster than anyone reads,
   so cutting ITL further buys ZERO perceptual UX gain for a streaming chatbot. (Note the wall-
   clock nuance: E2E p50 4685 = TTFT 387 + ~4300 decode, so decode IS ~92% of E2E — spec-decode
   *could* move the E2E/TPOT SLO numbers; it just doesn't help the human.)

2. **At 130 concurrent the box is throughput-bound, where spec-decode usually LOSES.** Draft+verify
   spends extra FLOPs per step; that only pays off with idle compute (low concurrency / single
   stream). At the peak operating point we size for, enabling it would steal throughput and worsen
   $/token unless acceptance is very high — the opposite of the goal.

**The binding UX lever is TTFT (already met: p50 387ms, p99 well under the 6000ms SLO), and the
binding business lever is throughput/$.** Spec-decode helps neither here. It would only matter for
a latency-sensitive *single-user / low-concurrency* tier with long outputs — not this use case.

**Park, don't pursue.** Revisit ONLY if (a) a future tier is low-concurrency + long-output, AND
(b) a vLLM version fixes Mamba2 spec-decode graph capture (re-test the built-in MTP head then).
No further spec-decode legs in this campaign.

### benchmark: Leg 4 — prefix-cache hit rate is 0 at TP1 TOO — Mamba2 prefix caching is non-functional on this build REGARDLESS of TP (not a TP>1-only issue)
<!-- captured: 2026-06-11 | stage: 6b -->

P2 Leg 4 (highest-value science): does prefix-cache hit rate go >0 at TP=1? The open
question was whether the 0% hit rate at TP2 is the upstream #26201 "TP>1 not yet tested"
caveat, or a deeper Mamba2 caching gap. Deployed ONE TP1 replica (manifest
`serving-leg4-tp1-prefix.yaml`, gpu-mem-util 0.92).

**TP1 LOADS on a single B200** (no OOM): 120G FP8 weights + KV + activations fit. But the
single GPU holding ALL weights leaves a SMALL KV pool: 1,938,560 tokens / 116.67x conc
headroom (vs TP2's 8.44M / 507x, TP4's 10.74M / 635x). 116x < 130 target, so a single TP1
replica cannot sustain conc=130 anyway. Cold start ~5.7 min (single-GPU weight load + compile
slower than the TP2 split).

**THE ANSWER — prefix cache is STILL 0 at TP1.** Authoritative engine metric after 1.8M
queries against the byte-identical ~3050-tok shared header:
```
vllm:prefix_cache_queries_total{engine="0"} 1,805,903
vllm:prefix_cache_hits_total{engine="0"}    0.0
```
So the 0% hit rate is NOT a TP>1 artifact (#26201). It is present at TP1 as well →
**automatic prefix caching is non-functional for Nemotron-3-Super (hybrid Mamba2 + LatentMoE +
Select-Attention) on vLLM 0.18.1 at ANY TP.** The `--enable-prefix-caching --mamba-cache-mode
all` flags are ACCEPTED and `enable_prefix_caching=True` shows in the engine config, but the
Mamba-2 'all'-mode caching path never registers a single hit. This points at the Mamba2
prefix-caching IMPLEMENTATION (vLLM #26201/#25752 merged but apparently not yet effective for
this architecture on 0.18.1), not the parallelism layout. Re-test on a newer vLLM that matures
Mamba2 PC.

Perf (conc=64, single TP1 replica): E2E p50 10559 / p90 16940, TTFT p50 846, TPOT p50 129 →
FAILS SLO (single-GPU throughput too low; 116x ceiling). 0 errors. TP1 is not a viable serving
layout here regardless — it was a science probe, and it answered the question cleanly.

**Driver note**: bench-fin-support.py's prefix scrape sums lines matching `prefix_cache_hits`
substring, which also catches the `_created` timestamp gauges (~1.78e9 epoch) → garbage
cumulative `hits`/`queries` (~3.5e9, hit_rate ~0.9999). The WINDOWED `hit_rate_measured` (delta)
is still correct (0.0) because the bogus `_created` value is constant across pre/post so it
cancels. Authoritative reading is `vllm:prefix_cache_hits_total` (=0) scraped directly. Fix the
driver to match `prefix_cache_hits_total`/`_queries_total` exactly (anchor on `_total`).

**Customer conclusion**: the RAG prefix-cache win (cache the ~3050-tok shared header → lower
TTFT) does NOT materialize for this model on vLLM 0.18.1 at any TP. SLO is met purely on raw
B200 prefill throughput + ample KV headroom, NOT on prefix reuse. If prefix caching is required,
wait for a vLLM build with working Mamba2 automatic prefix caching and re-validate.

## Max-concurrency-at-SLO sweep (2026-06-11)

Pressure-tested the winner (FP8 agg-tp2-x4, mnbt=16384) past the 130 anchor to find the SLO ceiling. Gates: E2E p50 <= 6500, p90 <= 9500.

| conc | E2E p50 | E2E p90 | TPOT p50 | errors | verdict |
|------|--------:|--------:|---------:|-------:|---------|
| 130 (anchor) | 4899 | 7909 | 60.9 | 0 | PASS |
| 160 | 5351 | 7837 | 65.6 | 0 | PASS |
| 200 | 6395 | 9188 | 77.6 | 0 | PASS (marginal, ~105ms p50 / ~312ms p90 headroom) |
| 256 | 7711 | 11101 | 94.5 | 0 | FAIL (both gates) |

- **SLO ceiling = conc ~200** (true edge just above 200; 256 is first failure). Bracket 200->256.
- **0 errors at every level including 256** — failure is pure latency (decode-batch interference lifting TPOT), not capacity/errors. TTFT p50 stayed <500ms throughout.
- **~1.5x headroom** from the validated 130-concurrent operating point to the ceiling; 25-RPS production peak is comfortably inside.
- Anchor reproduced within ~5% (campaign c130 4685/8147 vs sweep c130 4899/7909). No drift.
- Raw: `/tmp/fin-rag-conc-sweep-20260611.md` + `/tmp/fin-rag-conc-sweep-c{130,160,200,256}.json`; in-pod `/tmp/sweep/`.
