# Qwen3-235B Speculative Decode — Lessons Learned

## Session: (not started — scaffolded 2026-05-14)

**Hardware (planned)**: p6-b300.48xlarge spot in us-west-2b (usw2-az2), 8× B300 SXM6 AC, NVSwitch NV18
**Engine under test**: SGLang v0.5.10-cu130
**Baseline**: qwen3-235b-b300 — vLLM v0.19.1 TP4 peak 11,820 tok/s @ c=512 / TP2+DP4+EP peak 13,877
**Target model**: `Qwen/Qwen3-235B-A22B-Instruct-FP8`
**Draft model**: `lmsys/Qwen3-235B-A22B-EAGLE3` (accept length 3.0-3.5 per model card)
**Goal**: Phases 0, 1, 1b, 4, 5 — EAGLE3 + HiCache + TP2+DP2 optimization tier

## Carried-over lessons from related sessions

Reviewed before session start — apply these from day one:

### L14 (Kimi K2.6-spec) — VPC endpoint interference in custom subnets
If us-west-2 subnet turns out to block ECR/SSM endpoints, fall back to public images + SCP for secrets. Pre-test with `curl -m 5 https://api.ecr.us-west-2.amazonaws.com/` in userdata.

### L15 (Kimi K2.6-spec) — Explicit HF_TOKEN export required
Always `export HF_TOKEN="$(cat ~/.cache/huggingface/token)"` in the subshell that runs `hf download`. Also set `HF_HUB_ENABLE_HF_TRANSFER=1` for Rust-based fast downloader.

### L16 (Kimi K2.6-spec) — FlashInfer cubin symlink race on EVERY cold start
NOT just on config sweeps. Pre-clear inside the container at launch:
```bash
docker run ... "$IMAGE" bash -c "find /usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins -name trtllmGen_bmm_export -exec rm -rf {} + 2>/dev/null; exec python3 -m sglang.launch_server ..."
```

### L19 (Kimi K2.6-spec) — CUDA graphs are worth 6.4× on spec decode
Never benchmark spec decode with `--disable-cuda-graph` as a "vanilla" comparison. That config is pathological, not baseline. Expect similar ratio on Qwen3 GQA architecture.

### L20 (Kimi K2.6-spec) — TP reduction helps at high concurrency with spec decode
Kimi saw TP4+DP2 beat TP8 by +14% at c=256. For Qwen3-235B on TP4 (already smaller than Kimi's TP8), test TP2+DP2 — may be similar lever but the baseline TP4 already has VRAM headroom, so expected win smaller.

### Baseline session lessons (qwen3-235b-b300)

From `../qwen3-235b-b300/lessons.md`:

**L#1** — FP8 block_n=128: `moe_intermediate_size=1536 / TP_SIZE % 128 == 0` required. TP4 (1536/4=384 ✓), TP2 (768 ✓), TP1/TP3 valid. **TP8 INVALID** (192, not divisible by 128). This is why the plan uses TP4 primary, TP2+DP2 for variant — NOT TP8.

**L#2** — `max_position_embeddings=40960`, NOT 131072. FP8 variant does not include YaRN. Do not set `--max-model-len > 40960`.

**L#3** — Tool-call parser: **`--tool-call-parser hermes`** (NOT `qwen3_xml` — doesn't parse `<tool_call>` tags in vLLM v0.19.1). SGLang may differ — verify at first launch.

**L#4** — Reasoning parser: `--reasoning-parser deepseek_r1` per official Qwen3 docs. Thinking mode on by default; disable with `/no_think` in system prompt. Validate on first SGLang launch — SGLang may use a different name.

**L#5** — TP4 FP8 on B300: 55 GiB/GPU weights, 210 GiB KV headroom. Baseline peak 11,820 tok/s @ c=512 (before spec decode).

## To be captured mid-flight

### L4 — EKS NVIDIA device plugin needs `nvidia.com/gpu.present=true` label
**Severity**: HIGH · **Category**: eks
Pods requesting `nvidia.com/gpu` stayed Pending with `Insufficient nvidia.com/gpu` even though `nvidia-smi` showed 8 B300s. The `kube-system/nvidia-device-plugin` DaemonSet has `nodeSelector: nvidia.com/gpu.present=true`. New nodes don't get this label automatically — add it manually or via nodeadm `--node-labels`.
**Fix**: `kubectl label node <name> nvidia.com/gpu.present=true`. Also add to nodeadm config `flags: --node-labels=nvidia.com/gpu.present=true` so future nodes auto-register.

### L6 — SGLang uses `--context-length`; vLLM uses `--max-model-len`
**Severity**: LOW · **Category**: engine-flags
vLLM baseline manifest used `--max-model-len 40960` — SGLang doesn't recognize this. Equivalent flag is `--context-length 40960`. Adapter tables between engines should note this.
**Rule**: SGLang flag reference is at `python3 -m sglang.launch_server --help`. Always verify flags for the specific SGLang version, not vLLM equivalents.

### L12 — SGLang `DP+EP+EAGLE3` combo is broken in 0.5.10 (LMSYS-warned)
**Severity**: HIGH · **Category**: engine-bug
Phase 6b on Qwen3-235B with `--tp 2 --dp 2 --expert-parallel-size 2 --moe-a2a-backend deepep --speculative-algorithm EAGLE3` crashed at scheduler init with `RuntimeError: Rank 0 scheduler died during initialization (exit code: -3)`. The LMSYS Wide-EP blog explicitly warns: *"SGLang supports MTP but lacks full integration with DP attention, reducing efficiency in mixed parallelism configurations."* Our crash confirms this generalizes to EAGLE3.
**Workaround**: Don't combine all three. Either (a) DP+EP no spec, (b) TP-only + EP + EAGLE3, or (c) DP+EAGLE3 no EP.
**Rule**: Before combining EP + DP + speculative decode, check the SGLang release notes for explicit support. As of v0.5.10 the combo is unsupported.

### L18 — vLLM stock NVFP4 ≈ SGLang FP8+EAGLE3 on real workloads — both ~2× behind CoreWeave
**Severity**: CRITICAL · **Category**: cross-provider methodology
Direct ShareGPT comparison (200 conversations, c=16, decode tok/s per request):
- **vLLM TP4 + NVFP4 (no spec decode)**: 63.3 tok/s
- **SGLang TP8 + FP8 + EAGLE3 + HiCache**: 63.8 tok/s
- **CoreWeave (NVFP4 + custom kernels)**: 128.2 tok/s (Artificial Analysis)

The two stock OSS configurations land within 1% of each other. NVFP4's HBM advantage and EAGLE3's spec-decode advantage roughly cancel out — neither alone is enough to close the gap to CoreWeave's 128.2 tok/s. This means CoreWeave's 2× lead is NOT primarily about NVFP4 quantization (we have that). It comes from:
- Custom NVFP4 cutlass 3.x kernels that exploit B200/B300 FP4 tensor cores (vLLM v0.19's compressed-tensors path is more generic)
- Likely a custom-trained draft model on production traffic (would boost accept rate from 0.35 → 0.7+)
- Possibly Kimi-specific batching / MoE expert routing tuned for the 384-expert architecture

**Rule for customer comparisons**: Stock OSS on B300 in mid-2026 ≈ 60-65 tok/s decode on Kimi K2.6 ShareGPT, regardless of engine choice (SGLang+EAGLE3 vs vLLM+NVFP4). Closing the gap to CoreWeave requires non-stock work: custom kernels OR custom drafts. Neither is cheap. If a customer doesn't need that last 2× speedup and just wants OSS + reproducibility, our stack is fine.

### L17 — Kimi K2.6 was NOT trained with MTP — `model_type: kimi_k25`, no nextn weights
**Severity**: CRITICAL · **Category**: model-feature-assumption
Despite community assumption that Kimi K2.6 ships MTP weights (analogous to DeepSeek V3/R1), inspection of both the FP8 base (`moonshotai/Kimi-K2.6`) and the NVFP4 quant (`nvidia/Kimi-K2.6-NVFP4`) `config.json` shows: `architectures: ["KimiK25ForConditionalGeneration"]`, `model_type: kimi_k25`, no `num_nextn_predict_layers`, no MTP fields. vLLM's `--speculative-config '{"method":"mtp"}'` rejects with `NotImplementedError: Unsupported speculative method: 'mtp'` because there's nothing to attach to.
**Implication**: Spec decode on Kimi K2.6 requires an external draft model (EAGLE3 via SGLang per L8 of Kimi-spec lessons) — there's no native MTP path. Commercial providers serving Kimi K2.6 with high decode tok/s (CoreWeave 128.2) are doing one of: (a) NVFP4 with no spec decode + custom kernels, (b) custom EAGLE/MTP draft they trained themselves, or (c) custom-trained MTP retrofit. Option (a) is most likely given that NVIDIA's NVFP4 release card mentions no spec decode.
**Rule**: Don't assume MoE models ship MTP. Always verify via `config.json` keys before configuring vLLM speculative-config. DeepSeek V3/R1 ship MTP; Kimi K2.6, Qwen3-235B, and Llama-4 do NOT.

### L16 — NVFP4 vs FP8 quantization is a 2× HBM bandwidth advantage
**Severity**: HIGH · **Category**: methodology / cross-provider comparison
Commercial providers (CoreWeave 128.2 tok/s, Azure 143.7 tok/s on Kimi K2.6 per Artificial Analysis) serve **NVFP4** (4-bit, microscaled). Our SGLang 0.5.10 setup serves **FP8** (8-bit, block-scaled). Per the Phase 0 roofline, decode is 84% BW-headroom-bound — halving the bytes-per-token via 4-bit weights gives close to 2× decode throughput on the same hardware before any other optimization.
**Comparison rule**: When comparing our SGLang FP8 numbers vs commercial provider numbers, treat anything within **0.5× to 1.0×** of the provider's reported tok/s as competitive at the FP8 tier. To genuinely beat NVFP4 providers, we'd need either NVFP4 kernels (cutlass 3.x — not in SGLang 0.5.10, see L21 in Kimi report) OR a workload regime where their FP4 numerical floor hurts (long-context where BW per token is dominated by KV not weights).
**Phase 5d extension**: this strengthens the case for FP4 once cutlass 3.x kernels land in SGLang. Expected gain on Kimi K2.6: 1.6-2× single-stream and aggregate decode.

### L15 — Synthetic prompts overstate EAGLE3 gains by 3-5× vs real ShareGPT traffic
**Severity**: CRITICAL · **Category**: methodology
On synthetic uniform `"hello hello hello..."` prompts (the c=1 test in Phase 1b), EAGLE3 accept rate = **1.0** and accept length = **5.0** — draft trivially predicts everything. On real ShareGPT production conversations:
- Accept rate drops to **0.156** (15.6%, vs 100% synthetic)
- Accept length drops to **1.62** (barely better than no spec decode)
- Per-request throughput drops from 325 tok/s (synthetic) → **54 tok/s** (ShareGPT)
**Implication**: All synthetic-prompt benchmark numbers in this report (and the Kimi K2.6-spec report) overstate real-world EAGLE3 throughput by 3-5×. Production traffic accept rate depends entirely on the draft model's training distribution match to real prompts.
**Rule for future benchmarks**: Always run `production-mix` (ShareGPT replay) alongside any synthetic concurrency-sweep. The synthetic numbers are useful for engine-vs-engine comparison at fixed prompt distribution, but they are NOT predictive of production throughput. Customer-facing reports MUST include the ShareGPT number.

### L14 — SGLang EP path BROKEN for Kimi K2.6 mxint4, with OR without EAGLE3
**Severity**: BLOCKING · **Category**: engine-bug
Kimi K2.6 (mxint4 quantization) with `--expert-parallel-size 8 --moe-a2a-backend deepep` crashes at first decode step with `AssertionError: forward_deepgemm_masked is deprecated` in `sglang/srt/layers/moe/ep_moe/layer.py:248`. **Tested with AND without EAGLE3** — same crash both times. The deprecated forward path is hit during the EP-MoE dispatch regardless of speculative decoding. Affects mxint4 MoE on EP path specifically (Qwen3-235B FP8 EP doesn't hit this — different code path).
**Workaround**: Don't use EP with Kimi K2.6 in SGLang 0.5.10. Wait for upstream fix or downgrade to 0.5.9 (which had the non-deprecated path).
**Rule**: When testing EP on a new model+quantization combo, run a single-request smoke without spec decode first to isolate engine bugs from spec-decode integration issues. EP code paths are model-quantization-specific in SGLang.

### L13 — SGLang single-node EP+EAGLE3 hurts (Phase 6a measurement)
**Severity**: HIGH · **Category**: performance
On Qwen3-235B (235B/22B active, 128 experts, 8 active per token) with `TP4+EP4+EAGLE3+HiCache` on B300×8: throughput regressed by 14% at high concurrency and 39% at c=1 vs the same config without EP. LMSYS's Wide-EP blog reported 5.2× decode speedup on DeepSeek-V3 (671B, 256 experts, EP72 across 9 nodes) — the gain is from multi-node EP at scale, not single-node EP. Single-node DeepEP all-to-all + per-step EAGLE3 multiplication overhead exceeds the per-rank weight-read savings when only ~2 experts are active per rank per step.
**Rule**: Single-node EP for MoE serving is rarely worth it on B300/B200 NVSwitch. Reserve EP for ≥2-node deployments where the topology forces it. EAGLE3 + EP is anti-synergistic on single-node.

### L11 — SGLang EP flags use full names: `--expert-parallel-size N` + `--moe-a2a-backend deepep`
**Severity**: HIGH · **Category**: engine-flags
The shorthand `--ep-size` and the boolean `--enable-ep-moe` (commonly seen in older docs) do not exist in SGLang 0.5.10. Correct flags are:
- `--expert-parallel-size N` — sets EP world size
- `--moe-a2a-backend {deepep,mooncake,nixl,mori,flashinfer,...}` — selects the all-to-all backend (DeepEP recommended, matches LMSYS Wide-EP blog)
- (No need for an `--enable-*` flag — EP is implicit when EP size > 1)
**Rule**: Always check `python3 -m sglang.launch_server --help | grep -iE "expert|ep-|moe"` against the running version, not blog posts.

### L10 — `hf download --exclude '*.md' '*.txt'` is a footgun in hf-CLI ≥1.14
**Severity**: HIGH · **Category**: tooling
hf-CLI 1.14 prints `UserWarning: Ignoring --exclude since filenames have being explicitly set` and then treats `*.md` and `*.txt` as literal positional filename args. It tries to download a file literally named `*.txt`, fails with `File not found in repository`, and the entire download exits non-zero. The hf-CLI no longer accepts the old positional include/exclude syntax — `--exclude` requires the new keyword form, but documentation didn't catch up.
**Fix**: Drop `--exclude` entirely for benchmark staging (the few KB of READMEs are noise). If you must filter, use `--allow-patterns '*.safetensors' '*.json'` instead.
**Rule**: For production weight staging, just do `hf download <repo> --local-dir <dir> --max-workers 16` — no filter args. The model card files are <100 KB total.

### L9 — HiCache allocates `hicache_size × TP` GB of HOST memory; pod memory limit must accommodate
**Severity**: HIGH · **Category**: memory-budgeting
SGLang `--hicache-size 200` allocates 200 GB of **host memory** per TP rank. For TP4, total = **800 GB** host memory. The default pod limit (256 GiB) OOM-killed the pod at "Allocating 200.00 GB host memory for hierarchical KV cache" on the first rank. Node has 4 TB so capacity is fine, but pod limit must match the total.
**Fix**: Set pod `resources.limits.memory: 1200Gi` (800 GB HiCache + ~400 GB headroom for weight shards, KV buffers, CUDA graphs). For Kimi K2.6 TP8 × 200 GB = 1.6 TB, so pod limit should be 1800Gi+ there.
**Rule**: `pod_memory_limit >= hicache_size_gb * tp_size + 400` (GB). Or use a reasonable uplift like 4× hicache total.

### L8 — SGLang `/metrics` endpoint requires explicit `--enable-metrics` flag
**Severity**: HIGH · **Category**: observability
Without `--enable-metrics`, SGLang returns 404 on `/metrics` — Prometheus cannot scrape histograms. Our smoke test caught this (`up=0` for sglang target) before any bench ran. If we'd skipped the smoke and gone straight to benchmarks, we would have lost TTFT data (same shape as the Kimi-spec loss).
**Fix**: Add `--enable-metrics` to every SGLang launch command.
**Rule**: Never assume engines expose metrics by default. Always smoke-test Prometheus `up{job=...}` before running benchmarks.

### L7 — SGLang tool-call-parser enum for Qwen3 is `qwen3_coder`, not `qwen3`
**Severity**: LOW · **Category**: engine-flags
SGLang 0.5.10 tool-call-parser enum: `deepseekv3, deepseekv31, deepseekv32, glm, glm45, glm47, gpt-oss, kimi_k2, lfm2, llama3, mimo, mistral, pythonic, qwen, qwen25, qwen3_coder, step3, step3p5, minimax-m2, trinity, interns1, hermes, gigachat3`. For Qwen3-235B, `qwen3_coder` is the right choice. `qwen` (older Qwen) and `qwen25` also exist.
**Rule**: Engine flag enums drift between versions — always read the error message (SGLang prints the enum on invalid choice).

### L5 — SGLang 0.5.10 doesn't accept `deepseek_r1` (underscore); uses dashes
**Severity**: LOW · **Category**: engine-flags
The baseline (vLLM v0.19.1) used `--reasoning-parser deepseek_r1`. SGLang 0.5.10 expects dash form: `deepseek-r1`, `qwen3`, `qwen3-thinking`, `kimi_k2`, etc. For Qwen3-235B, **`qwen3-thinking`** is the right reasoning parser (thinking mode on by default) and **`qwen3`** for tool calls (not `hermes`).
**Fix applied**: `--tool-call-parser qwen3 --reasoning-parser qwen3-thinking` in k8s/sglang-eagle3-phase1.yaml.
**Rule**: SGLang parser names ≠ vLLM parser names. Check `sglang.launch_server --help | grep parser` on the target SGLang version.

### L2 — AL2023 DLAMI lacks nodeadm (must use EKS-optimized AMI for node-join)
**Severity**: HIGH · **Category**: ami-selection
The "Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)" (`ami-0edf8a2b20d08b539` at time of writing) does NOT include `nodeadm`. Kimi K2.6-spec used this AMI successfully because it ran as a standalone EC2 (no EKS join). For EKS cluster membership the EKS-optimized AL2023+NVIDIA AMI is required:
  `aws ssm get-parameter --name /aws/service/eks/optimized-ami/1.32/amazon-linux-2023/x86_64/nvidia/recommended/image_id`
  → `ami-0d868cc255a3e103a` (1.32, us-west-2, as of 2026-05-14)
This AMI ships with nodeadm, kubelet, and containerd pre-configured; the MIME `application/node.eks.aws` userdata part works natively.
**Rule**: When joining a pod-on-EKS (Option A), always use the EKS-optimized AMI. Standalone EC2 (Option B) can use the generic DLAMI.

### L3 — `set -x` leaks secrets to cloud-init-output.log
**Severity**: MEDIUM · **Category**: security
With `set -eux`, every shell line including `HF_TOKEN_VAL=hf_...` from `aws ssm get-parameter` printed to `/var/log/cloud-init-output.log`, which is readable by anyone with EC2 console access or SSM. First spot had this leak in its log.
**Fix**: Use `set -eu` (no `-x`) in userdata scripts, or wrap secret-handling in `set +x; ...; set -x`.
**Rule**: Userdata that fetches SSM SecureString parameters must not trace commands.

### L1 — EKS cluster SG vs node SG (us-west-2 qn-sglang-eks-cluster)
**Severity**: HIGH · **Category**: networking
The `aws eks describe-cluster` output shows one SG under `resourcesVpcConfig.securityGroupIds` — that is the **control-plane** SG (empty egress, for control-plane ENIs only). Launching a node into that SG means no egress to docker/HF/S3/SSM; cloud-init dnf timed out for 15 min before failing.
**Fix**: Use the auto-generated **cluster SG** (`kubernetes.io/cluster/<name>: owned` tag, name prefix `eks-cluster-sg-<name>-`). In qn-sglang-eks-cluster that's `sg-070da338e3796648d`. Existing EKS nodes use that SG.
**Rule**: Before launching EC2 into an EKS cluster, inspect an existing node's SGs (`kubectl get nodes` → `describe-instances`) rather than trusting `describe-cluster.resourcesVpcConfig.securityGroupIds`.

## Next phases queued

Per spec:
- **Phase 0**: roofline (NCCL + DeepGEMM, confirm BW matches Kimi session)
- **Phase 1**: SGLang EAGLE3 defaults (num_steps=3, num_draft=4, topk=1)
- **Phase 1b**: 13-config sweep
- **Phase 4**: winner + HiCache 200 GB/rank
- **Phase 5**: default stack / no-cuda-graph / TP2+DP2 / FP4 probe

Expected winner per model card: `num_steps=2` or `3` (accept length caps at 3.5, so num_steps=4 unlikely to help).
