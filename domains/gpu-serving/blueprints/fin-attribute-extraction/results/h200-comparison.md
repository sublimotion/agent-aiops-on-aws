# Qwen3.6-35B-A3B — H200 results (2026-06-16), 3-way with g7e + B200

**Setup**: p5en.48xlarge SPOT, us-east-2b, via `qwen3-next-bench` EKS cluster (`h200-use2b-spot` nodegroup). 1× H200 SXM (sm_90, NVSwitch), vLLM 0.22.1 FP8, TP1, mnbt=8192, `--limit-mm-per-prompt '{"image":0,"video":0}'`. Identical config to the g7e and B200 runs.

## Headline: H200 ≈ B200, ~2× g7e — confirms the graph-capture hypothesis

**Full CUDA-graph capture WORKS on H200/sm_90** (`Capturing CUDA graphs (decode, FULL): 100%|35/35`, finished in 18s) — same as B200, unlike g7e/sm_120 which crashes full-decode capture (`1 != ISL`). This was the predicted differentiator and it holds: H200 lands right next to B200.

## Per-replica SLO sweep (E2E p90 ms, FP8 TP1 mnbt=8192, single replica)
| Rate | g7e | **H200** | B200 |
|------|-----|----------|------|
| 10   | 1401 | 800  | 863  |
| 12   | 1991 | 894  | 916  |
| 14   | 3489 (FAIL) | 970 | 948 |
| 18   | (sat ~14) | 1160 | 1439 |
| 20   | — | **1559 (knee)** | — |
| 22   | — | 2354 (FAIL) | 1829 (PASS) |
| 26   | — | 2763 | 2670 |

- **H200 SLO-safe ceiling ≈ ~20 RPS/replica**, saturates ~25 req/s.
- **g7e ~12, H200 ~20, B200 ~22.** H200 essentially matches B200 (slightly below at the very top — B200 holds 22, H200's knee is ~20-21). Both ~1.7-2× g7e.
- Unloaded latency identical across all three: H200 TTFT 91ms / E2E 176ms (g7e 98/237, B200 97/181).

## Bottleneck class — compute-bound (same as B200, NOT launch-bound like g7e)
- **H200 at sustained load (rate 24): SM 100%, HBM mem-util ~47%** (in-pod `nvidia-smi dmon`, steady-state). Power peaked ~688W (DCGM).
- Compare: g7e SM ~43%/HBM ~0% (launch-bound); B200 SM ~99%/HBM ~31% (compute-bound). **H200 = compute-bound, like B200.**
- The working full-graph capture removes the per-step launch overhead → SMs fill → runs until genuinely compute-limited. H200's HBM is a bit more engaged (47% vs B200 31%), consistent with H200's lower HBM bandwidth (4.8 vs 8 TB/s) — but still compute-bound, not bandwidth-bound, at the knee.

## The 3-way picture: it's a software (graph-capture) story, not a silicon-tier story
| GPU | arch | full graph capture | SLO-safe RPS/GPU | bottleneck |
|-----|------|--------------------|--------------------|-----------|
| g7e RTX PRO 6000 | sm_120 PCIe | **CRASHES** (`1 != ISL`) | ~12 | launch-bound |
| H200 | sm_90 NVSwitch | works | ~20 | compute-bound |
| B200 | sm_100 NVSwitch | works | ~22 | compute-bound |

The ~2× gap between g7e and the datacenter parts is **not** about raw FLOPs/bandwidth — it's that vLLM 0.22.1's full-decode CUDA-graph capture crashes on sm_120 but works on sm_90/sm_100. If a newer vLLM fixes the g7e crash, g7e's ceiling should rise toward the others (it has the FLOPs; it's just paying launch overhead today).

## Cost (us-east-2 spot, 2026-06-16)
| GPU | instance | $/node-hr (8 GPU) | $/GPU-hr | SLO-safe RPS/GPU | $/sustained-RPS-hr |
|-----|----------|-------------------|----------|--------------------|---------------------|
| g7e | g7e.48xlarge | $11.17 | $1.40 | ~12 | $0.116 |
| **H200** | **p5en.48xlarge** | **$15.31** | **$1.91** | **~20** | **$0.096** |
| B200 | p6-b200.48xlarge | $18.90 | $2.36 | ~22 | $0.107 |

- **H200 is the CHEAPEST per unit of SLO-safe capacity ($0.096)** — it gets ~B200 throughput at a lower spot price. ~17% cheaper than g7e and ~10% cheaper than B200 per sustained-RPS.
- For 70 RPS: ~4 H200 GPUs (half a node), same as B200, vs ~6 g7e.
- Caveat: p5en spot price is volatile (seen $13-16/hr); at the high end the H200/B200 gap narrows. p5e (non-EN) would be cheaper still (~$10.64) if available — networking is unused here (TP1, no collective).

## Notes
- Same B200 gotchas applied: node launched WITHOUT the node SG (attached `sg-0bf5ad07fc6c29df1` manually); vision-encoder must be disabled (`--limit-mm-per-prompt`). The new nodegroup had the `nvidia.com/gpu.present` label baked in, so the device plugin attached automatically (no manual label needed, unlike the ad-hoc B200 node).
- Weights reused from FSx (staged during B200 run); fetched to H200 local emptyDir over pod HTTP from a CPU-node fsx-http server.
- DCGM: exporter runs on the node and exposes per-GPU profiling fields on :9400, but Prometheus is NOT scraping it (no ServiceMonitor) — scraped the exporter directly. In-pod `nvidia-smi dmon` is the reliable continuous source; DCGM PROF point-samples miss the bursty peaks.
