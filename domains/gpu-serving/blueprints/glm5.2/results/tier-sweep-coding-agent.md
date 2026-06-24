# GLM-5.2-FP8 B200 TP8 — Tier sweep (coding-agent workload)

SGLang v0.5.13.post1-cu130, single 8-GPU B200, coding-agent shape (12K shared system prompt).

| Tier | c=1 | c=4 | c=16 | c=32 | c=64 | notes |
|------|-----|-----|------|------|------|-------|
| T0 (FP8 floor, radix OFF) agg tok/s | 98.5 | 314.7 | 883.4 | 1305.3 | 1708.2 | cache=0 |
| T0 TTFT p99 (s) | 0.89 | 2.25 | 7.84 | 14.85 | 29.69 | |
| T2 (+prefix cache) agg tok/s | 99.3 | 317.6 | 1101.6 | 1884.6 | 2950.1 | cache=0.92 |
| T2 TTFT p99 (s) | 0.88 | 0.82 | 1.53 | 1.98 | 3.50 | |
| **T2 vs T0 throughput** | +1% | +1% | +25% | +44% | **+73%** | |
| **T2 vs T0 TTFT p99** | — | 2.7x | 5.1x | 7.5x | **8.5x** | |

T2 (prefix cache) is the dominant coding-agent lever — the 12K shared system prompt caches at 92%,
collapsing TTFT and lifting throughput. 0 errors all levels. Next: T1 fp8-KV, T3 EAGLE, T5 full stack.
