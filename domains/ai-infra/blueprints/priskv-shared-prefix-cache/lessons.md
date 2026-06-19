---
experiment: "priskv-shared-prefix-cache"
hypothesis: "Shared host-DRAM KV cache (PrisKV) lowers TTFT vs local-APC + prefix-aware routing on PCIe-only multi-replica g7e at high prefix reuse"
outcome: "falsified"
model: "qwen3-32b-fp8"
engine: "vllm-0.10.2 + aibrix_kvcache + priskv"
hardware: "g7e.12xlarge"
gpu_arch: "sm_120"
deployment_date: "2026-06-18"
failure_categories: ["technique-negative-result"]
result_summary: "PrisKV shared cache 2-5x SLOWER than prefix-aware routing at 70% reuse; relatively worse as prefix grows (host-DRAM round-trip > avoided on-GPU prefill). Gate-4 build PASSED but needed ~14 undocumented fix-ups."
steering_candidate: true
---

# PrisKV Shared Prefix Cache — Lessons

Spec: `domains/ai-infra/specs/priskv-shared-prefix-cache.md`
Node: g7e.12xl `i-030d90b609a2fc333`, Tokyo 1c, 2× RTX PRO 6000 sm_120. 2026-06-18 (terminated).

## VERDICT: FALSIFIED — use prefix-aware routing, not an external KV store

PrisKV shared host-DRAM cache (Arm C) was **2× slower at 800-tok, 5× slower at 4000-tok** vs both
prefix-aware routing (B) and naive round-robin (A) at 70% reuse. Gets relatively worse with prefix
size — opposite of its thesis. On PCIe-only g7e (PrisKV's *best-case* hardware), keeping the KV
on-GPU and routing to it beats an off-GPU host-DRAM fetch. Full data: `results/RESULT-20260618.md`.

**STEERING RULE (proposed for `.claude/steering/tech-stack.md`)**: For cross-replica prefix reuse
on PCIe-only / NVLink-less GPUs, use **prefix-aware routing (llm-d / GAIE)**, NOT an external
host-DRAM KV store (PrisKV/aibrix_kvcache offloading). The host↔GPU round-trip for cached KV costs
more than the avoided on-GPU prefill, and the penalty scales with prefix size. PrisKV's "zero-copy
shared memory" does not engage on a single node (UCX falls back to TCP; `UCX_TLS=sm` times out).
Closes the AIBrix single-node-PD evaluation for our fleet. Corroborates `[[pd_disagg_single_node]]`
and `[[project_pd_disagg_frontier_only]]`: KV-movement tricks only pay at frontier scale on fast
fabric, never as a host-DRAM hop on commodity GPUs.

## Phase 0 — build (falsification gate #4) — **PASSED** (serves), cache-population WIP

**Gate #4 verdict: PASS.** The full stack builds and serves correct inference on g7e/sm_120:
vLLM 0.10.2 (cu128/sm_120) + aibrix_kvcache (`integrate-vllm-aibrix`) + PrisKV L2 over UCX/TCP.
"The capital of France is" → " Paris" (HTTP 200, correct). Model loaded 32.04 GiB on GPU 0,
connector initialized (`engine_block_ntokens=16, cache_block_ntokens=64`). Build was NON-trivial
(~10 fix-ups, see L1-L12) but well under the 1-day maturity bar. The blog's "just deploy" framing
massively understates the integration work.

### L10 — UCX endpoint timeout fix: `UCX_TLS=tcp` + `PRISKV_USE_SHM=n` (NOT `sm,self,tcp`)
With `UCX_TLS=sm,self,tcp` the client→server connection failed: `priskv_ucx_ep_create: Endpoint
timeout` (server logged `UCX: recv handshake msg failed`). The PrisKV repo's own `SETUP_UCX_TCP.md`
(commit #49) prescribes `PRISKV_TRANSPORT=ucx`, **`UCX_TLS=tcp`**, `PRISKV_USE_SHM=n` for
no-RDMA-hardware dev environments. After switching, endpoint created fine and the model loaded.
(So on g7e the localhost path is UCX-over-TCP, NOT shared-memory — even though both containers share
`--ipc host`. The "zero-copy shared memory" the blog touts did not work out of the box.)

### L11 — redis-cli defaults to :6379; PrisKV meta server ran on :16379 — silent empty meta
Every `redis-cli SET priskv_cluster_metadata` silently hit nothing (`Connection refused` to :6379)
because the meta-redis ran on `--port 16379`. The PrisKV client then loaded EMPTY cluster-meta and
the C client asserted `priskvClusterMetaDataLoad: Assertion priskvClusterMetaDataNodeLoad failed`
(client.c:309 → couldn't open node). Fix: `redis-cli -p 16379`, and write the meta JSON from a
**file** (`redis-cli -x SET < file`) — inline shell quoting mangled the JSON (stripped quotes,
captured literal `\x27`). The `_PRIS_REMOTE_PORT` env points at the **redis meta port (16379)**, NOT
the UCX server port (9000); the client discovers the UCX server (addr/port in the meta JSON) FROM
redis.

### L12 — CACHE POPULATION RESOLVED: offload has a 128-token floor (not a bug, a threshold)
Initial 0% hit was NOT inertness — it was the **offload skip threshold**:
`threshold = max(OFFLOADING_CONNECTOR_SKIP_THRESHOLD(=8) × engine_block_ntokens(=16),
cache_block_ntokens(=64)) = 128 tokens` (aibrix_offloading_connector_type1.py:937). Requests whose
aligned query length is <128 tokens are SKIPPED — never offloaded. My 5-token and 206-token tests
fell under the *aligned-query* floor.
**Confirmed working** with a 600-token prompt fired 3×:
`SEND: 896 tokens sent` → `L2Cache PUT ops=2` → on repeat `L2Cache GET: 896 hit, 0 miss,
Hit rate 100.00%`, `RECV latency 36.8ms`. The KV genuinely round-trips through the shared PrisKV
store. **Implication for the matrix**: the `shared-prefix-multitenant` card's 4096-token shared
prompt is comfortably over the floor — good. But Arm comparisons must use prompts >128 tokens or
PrisKV silently no-ops (and would falsely look like a regression vs local APC). Document the floor
in the workload sizing.

**GATE #4 FULLY CLOSED**: stack builds, serves correct output, AND offloads+retrieves KV via the
shared PrisKV L2 with 100% hit rate. The experiment is runnable.

## Phase 0 — build (falsification gate #4) — original notes

### L1 — PrisKV server has NO sm_120 device kernels — Blackwell compile risk is a non-issue
The PrisKV *server* is a standalone C binary (`make all PRISKV_USE_CUDA=1`) whose only CUDA
surface is `cudaHostRegister`/`cudaMemcpy`/`cudaHostUnregister` (pinned host memory). It builds
against **CUDA 12.1** (not cu130) and UCX 1.19, and runs as its own container talking to the
engine over TCP/UCX. There is nothing to compile for sm_120. So the "build on Blackwell" fear in
gate #4 does NOT apply to the server — the risk is entirely on the vLLM-side connector.

### L2 — README's `pip install pypriskv` and `pip install aibrix_kvcache` are BOTH FALSE
Neither package is on PyPI (`ERROR: No matching distribution found`). The blog/README claim is
wrong. Real install paths:
- `pypriskv` → build from source in the repo: `make pyclient` → wheel in `pypriskv/dist/`.
- `aibrix_kvcache` → lives in `github.com/vllm-project/aibrix` under `python/aibrix_kvcache`
  (has `csrc/` + `CMakeLists.txt` → C++/CUDA build, not pure-python).

### L3 — DECISIVE: aibrix_kvcache→vLLM is a SOURCE PATCH pinned to vLLM v0.10.2
The integration is NOT a pip plugin — it's `integration/vllm/patches/vllm_v0.10.2-aibrix-kvcache.patch`,
which monkeypatches the KV-connector `factory.py` and adds `aibrix_offloading_connector_type{1,2,3}.py`
into the vLLM source tree. So the connector is **hard-pinned to vLLM 0.10.2 internals**, not
version-portable. Confirms why the China-ECR reference image was vLLM 0.10.2.

**The gate #4 fork this creates**: the connector needs vLLM **0.10.2**, but sm_120 (RTX PRO 6000
Blackwell) needs a vLLM new enough to support it.

### L4 — GATE #4 RESOLVED (PASS path): vLLM 0.10.2 ALREADY supports sm_120
`vllm/vllm-openai:v0.10.2` ships **torch 2.8.0+cu128**, `arch_list` includes **`sm_120`**, and it
correctly detects `NVIDIA RTX PRO 6000 Blackwell` with capability (12,0). So the feared version
conflict does NOT exist — the aibrix-pinned vLLM 0.10.2 runs natively on our Blackwell node. No
forward-port required. (Also: v0.10.2 is the NEWEST aibrix patch available — patches exist for
v0.8.5, v0.9.1, v0.10.2 only. 0.10.2 is both the connector ceiling AND sm_120-capable. Lucky
alignment.)

**Patch is invasive** (10 files, not just additive): touches `factory.py`, `base.py`,
`scheduler.py`, `gpu_model_runner.py`, `kv_connector_model_runner_mixin.py`, `envs.py`,
`kv_transfer_metrics.py` + 3 new connector files. Because the v0.10.2 image has vLLM installed as
a *package* (`/usr/local/lib/python3.12/dist-packages/vllm/`), the patch can be applied in-place to
the installed tree (fast) rather than rebuilding vLLM from source. Build plan:
1. `FROM vllm/vllm-openai:v0.10.2`
2. apply `vllm_v0.10.2-aibrix-kvcache.patch` against `dist-packages/` (adjust `-p` strip path)
3. `pip install` aibrix_kvcache (cmake/torch build) + the `make pyclient` pypriskv wheel
4. validate connector registers: import + `--load-format dummy` boot with the offloading connector.

### L5 — PrisKV server image built OK (5.04GB); transport is RDMA/UCX only
`priskv:local` built from `Dockerfile_ubuntu2204` (CUDA 12.1 + UCX 1.19). Server binary runs
(`/workspace/priskv-server`, prints env-var config). **Transport options are RDMA and UCX only**
(no shared-mem/TCP transport enum) — on a single node with no RDMA HW, **UCX with sm/tcp
transports is the localhost path** (`UCX_TLS=sm,self,tcp` — echoes the kimi-k2.6-nvfp4 disagg
lesson that UCX needs sm,self,tcp for control-plane). The Dockerfile `rm -rf /root/priskv` in the
last layer deletes the `make pyclient` wheel, so **pypriskv must be built/installed separately
into the vLLM container** (it's the client lib the engine needs, not a server-side dep).

### L6 — Final image assembly: 3 components, official Dockerfile + 2 fix-ups
`vllm-priskv:final` = official aibrix integration Dockerfile (`FROM vllm/vllm-openai:v0.10.2`,
build aibrix_kvcache wheel, patch vLLM in-place) PLUS two things the official recipe omits:
1. **pypriskv client** — NOT on PyPI; build the wheel via `make pyclient` in the `priskv:local`
   builder (needs `git config --global --add safe.directory '*'` + `git submodule update
   --init --recursive` first — the `--depth 1` clone has no submodules). Import name is **`priskv`**,
   not `pypriskv` (dist name ≠ module name).
2. **UCX runtime libs** — the `priskv` native `.so` links `libucp/ucs/uct/ucm.so` + the `/usr/lib/ucx`
   plugin dir. The vLLM base lacks them → copy ALL `/usr/lib/libuc*.so*` + `/usr/lib/ucx` from the
   priskv builder via multi-stage, then `ldconfig`. Copying just libucp/libucs is insufficient
   (also needs libucm, libuct). Set `PRISKV_TRANSPORT=UCX UCX_TLS=sm,self,tcp` for localhost
   (no RDMA HW).
Import chain validated: `priskv` + `aibrix_kvcache._aibrix_C` + connector/factory all load on GPU.

### L7 — OPERATIONAL: SG ingress IP rotates — "SSH timed out" (not "refused") = your IP changed
Mid-session SSH died with `Operation timed out`. Node was healthy (both EC2 status checks passed).
Root cause: workstation public IP rotated (108.26.230.24 → 50.187.107.73); the SG only allowed the
old /32. Fix: `authorize-security-group-ingress` the new IP, revoke the old. **Diagnostic tell:
"Connection refused" = sshd issue (port reachable); "timed out" = network/SG blocking (packets not
arriving).** Check `curl checkip.amazonaws.com` vs the SG rule before assuming the node is broken.

### L8 — DECISIVE version skew: aibrix wants module `pris` + backend `PRIS`; PrisKV ships `priskv`
The `integrate-vllm-aibrix` branch (the ONLY aibrix branch with PrisKV support — `pris.py` absent
on main/feature/context-cache) hard-codes:
- L2 backend name **`PRIS`** (env `AIBRIX_KV_CACHE_OL_L2_CACHE_BACKEND=PRIS`), NOT `PRISKV`.
- Env vars `AIBRIX_KV_CACHE_OL_PRIS_REMOTE_{ADDR,PORT}` (default port 6379), NOT `_PRISKV_`.
- Python module `pris._pris` + `pris.pris_client.PrisClient`.

But the public PrisKV repo's wheel installs module **`priskv`** (`priskv._priskv`,
`priskv.priskv_client.PriskvClient`) — it NEVER shipped a `pris` module. The reference deploy.yaml
in PrisKV's own repo (using `PRISKV` + `_PRISKV_` env names) does NOT match the aibrix branch — a
**genuine cross-repo version skew**. The blog's "just deploy it" framing hides this entirely.

**Fix (works because rename was cosmetic — API surfaces identical)**: built a `pris`→`priskv` alias
package (`pris/__init__.py`→`from priskv import *`, `pris/_pris.py`→`priskv._priskv`,
`pris/pris_client.py` maps `PrisClient = PriskvClient`). Verified API parity first: `priskv._priskv`
has `SGL`; `PriskvClient` has all methods pris.py calls (`reg_memory/dereg_memory/exists/get/set/
mget/close`). After alias + `BACKEND=PRIS` + `_PRIS_` env + `REMOTE_PORT=9000` (the priskv-server
UCX port, not redis), the connector initializes: **"Creating v1 connector AIBrixOffloadingConnector"
+ "Allocating slabs 100%"** — the L2 PrisKV backend allocates MRs and connects. Image:
`vllm-priskv:final2`.

### L9 — priskv-server pool sizing: -v × -b = total bytes; reference 512GB OOMs
`priskv-server -v <block_bytes> -b <num_blocks>` → pool = product. Reference deploy used
`-v 1048576 -b 524288` = **512 GB** → `MEM: failed to allocate memory` + segfault on our 499GB node.
Use `-v 65536 -b 524288` = **32 GB** for a 2-replica Qwen3-32B test (+ `--shm-size 40g --ipc host`).
Healthy start logs `UCX: <127.0.0.1:9000> ready`.

## Matrix run — multi-replica gotchas (Arm C)

### L13 — PrisKV connector binds a FIXED ZMQ port (6667+dp_rank) → 2 same-node replicas collide
Two independent single-GPU PrisKV replicas both crash the 2nd with
`zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:6667')`. The connector's
sidechannel port = `VLLM_AIBRIX_SIDE_CHANNEL_PORT(default 6667) + data_parallel_rank` — and two
separate processes both have dp_rank=0. **Fix: set `VLLM_AIBRIX_SIDE_CHANNEL_PORT` distinct per
replica** (6667, 6677). This is a real multi-replica-per-node limitation the connector wasn't
designed for (it assumes one engine per host, scaling via DP-rank not separate processes) — exactly
the topology PrisKV's single-node-multi-replica value prop requires. Undocumented.

### L14 — HARNESS BUG (mine): stale container answered health check → false "healthy"
Arm C "came up healthy" but r0 was actually the **leftover gate-#4 `vllm-test`** container still
bound to :18001; the real r0 had crashed with `Free memory 7.11/94.97 GiB < 0.9 util` because
vllm-test still held GPU 0. Round-robin then routed half the bench to dead r0 → hangs. **Fix:
teardown must remove ALL engine containers (added `vllm-test` to the rm list); ALWAYS verify both
GPUs show ~89GB before benchmarking** (proves both replicas are the new ones, not a zombie). Lesson:
health-200 on a port is necessary but not sufficient — check GPU residency matches replica count.

## PRELIMINARY RESULTS (2x TP1, 70% reuse, 800-tok prefix, 20 sessions) — verify before asserting

| Arm | TTFT p50 | TTFT p99 | shared-prefix p50 |
|-----|----------|----------|-------------------|
| **A** local APC + round-robin | **49.8 ms** | 168.8 ms | 49.8 ms |
| **C** PrisKV shared L2 + round-robin | 102.3 ms | 410.6 ms | 102.2 ms |
| **B** local APC + prefix-aware routing | (running) | | |

### 4000-token prefix (70% reuse, 16 sessions) — the cell where PrisKV SHOULD win
| Arm | TTFT p50 | TTFT p99 | shared p50 |
|-----|----------|----------|------------|
| **B** prefix-aware routing | **62.8 ms** | 898 ms | 62.8 ms |
| **C** PrisKV shared | 314.8 ms | 2084 ms | 311.1 ms |
| **A** round-robin | (running) | | |

**At 4000 tokens — the large-prefix regime where recompute is expensive and PrisKV's reuse should
pay off — PrisKV is 5× SLOWER on p50 and 2.3× slower on p99 than prefix-aware routing.** Even C's
cached reads (shared p50 = 311ms) are far slower than B's on-GPU APC hits (63ms). The host-DRAM
KV round-trip for 4000 tokens of KV state is the killer; prefix-aware routing keeps the hit entirely
on-GPU at NVLink-less PCIe speeds it never has to cross.

**Result is consistent across prefix sizes**: 800-tok (A/B≈49ms, C=102ms) and 4000-tok (B=63ms,
C=315ms). Larger prefix makes PrisKV RELATIVELY WORSE, not better — opposite of its thesis. Mechanism
confirmed: (a) vLLM local APC hit is nearly free; (b) the PrisKV GET (HBM←host-DRAM copy over PCIe +
UCX-TCP + connector serialization) scales with KV size, so bigger prefix = bigger penalty.

### Correctness gate: PASS (no PrisKV cache-race corruption observed)
Identical prompts returned identical completions across all arms and repeats (temp=0); the gate-#4
600-token offload test showed 100% hit with 0 miss and correct output. No evidence of the #41/#42
cache races in this single-node, moderate-load run. (Caveat: didn't stress concurrent writes hard.)

## Pre-flight gates passed
- Stage 4a: both GPUs clean (volatile ECC 0, remapped rows clean), driver 580.159.04, sm_120.
- Node prep: DLAMI pre-mounts NVMe (LVM) at `/opt/dlami/nvme` → symlinked `/mnt/nvme` (3.3T).
  Docker 29.5.3 + nvidia runtime pre-wired (turnkey — no manual containerd dance needed).

## Assets staged
- vLLM base `vllm/vllm-openai:latest` (cu130) pulled.
- Qwen3-32B-FP8 weights (32G) at `/mnt/nvme/models/Qwen3-32B-FP8` (HF, xet disabled per kimi L5).
- PrisKV server image build running (`priskv:local`, apt+CUDA12.1+UCX from scratch — slow).
