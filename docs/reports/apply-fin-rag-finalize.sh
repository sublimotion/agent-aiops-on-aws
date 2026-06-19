#!/usr/bin/env bash
# Finalize fin-rag-answer campaign — applies the 3 existing-file edits that the
# Claude Code process could not perform (EDR tamper-protection blocked writes to
# pre-existing inodes from the agent process tree; new-file creation was allowed,
# which is how this script and the visual report got written).
#
# RUN THIS FROM YOUR INTERACTIVE TERMINAL (your process is allowlisted by the EDR).
#   bash docs/reports/apply-fin-rag-finalize.sh
#
# Idempotent: safe to re-run. Each edit checks for its marker before applying.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
BP=domains/gpu-serving/blueprints/fin-rag-answer
PROG=$BP/results/progress.md
LESS=$BP/lessons.md
REPORT=$BP/results/final-report-20260611.md

say(){ printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[skip]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. progress.md — flip stage flags to complete
# ---------------------------------------------------------------------------
if [ -f "$PROG" ]; then
  say "Updating $PROG stage flags"
  # macOS/BSD sed -i needs an explicit backup suffix; use '' for none.
  sed -i '' \
    -e 's/^status:.*/status: complete/' \
    -e 's/^last_stage:.*/last_stage: 7/' \
    -e 's/^stage6b_benchmark:.*/stage6b_benchmark: complete/' \
    -e 's/^stage7_readiness:.*/stage7_readiness: complete/' \
    "$PROG"
  say "  done (stage8_compound left as-is — see note in §3 about manual learn commands)"
else
  warn "$PROG not found"
fi

# ---------------------------------------------------------------------------
# 2. lessons.md — prepend compound frontmatter IF not already present.
#    NOTE: gpu_arch is sm_100 (B200 is Blackwell sm_100). The compound-learner
#    sub-agent emitted sm_90 (Hopper/H100) — that was WRONG and is corrected here.
# ---------------------------------------------------------------------------
if [ -f "$LESS" ]; then
  if head -1 "$LESS" | grep -q '^---$'; then
    warn "$LESS already has frontmatter — leaving untouched (edit manually if needed)"
  else
    say "Prepending compound frontmatter to $LESS"
    TMP=$(mktemp)
    cat > "$TMP" <<'FM'
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

FM
    cat "$LESS" >> "$TMP"
    cat "$TMP" > "$LESS"   # write back via redirect into existing inode (your process is allowed)
    rm -f "$TMP"
    say "  done"
  fi
else
  warn "$LESS not found"
fi

# ---------------------------------------------------------------------------
# 3. Append the max-concurrency sweep ceiling to lessons.md + final report
# ---------------------------------------------------------------------------
SWEEP_MARK='## Max-concurrency-at-SLO sweep (2026-06-11)'
SWEEP_BLOCK=$(cat <<'MD'

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
MD
)

for f in "$LESS" "$REPORT"; do
  if [ -f "$f" ]; then
    if grep -qF "$SWEEP_MARK" "$f"; then
      warn "sweep block already in $f"
    else
      say "Appending sweep ceiling to $f"
      printf '%s\n' "$SWEEP_BLOCK" >> "$f"
    fi
  fi
done

# ---------------------------------------------------------------------------
say "All edits applied. Review with: git diff --stat && git diff"
echo
echo "Compound-step learn commands for the operator (run if you use fe/mdc):"
echo "  # frontmatter already written above with the sm_100 correction."
echo "  # If your steering tooling re-ingests lessons, point it at:"
echo "  #   $LESS"
echo "  # 5 cross-cutting rules to elevate to .claude/steering/tech-stack.md:"
echo "  #   1. ModelOpt FP8 checkpoints auto-select fp8 KV -> --kv-cache-dtype fp8 is a no-op."
echo "  #   2. Mamba-2 automatic prefix caching is non-functional on vLLM 0.18.1 at ANY TP (verify per-arch, not per-TP)."
echo "  #   3. More replicas > more TP for prefill-dominated high-concurrency RAG (queue depth dominates)."
echo "  #   4. Spec-decode is engine+draft-MoE-backend specific, NOT a hardware story (g7e/SGLang works, B200/vLLM 0.18.1 doesn't)."
echo "  #   5. B200 is sm_100 (Blackwell NVSwitch), NOT sm_90 (Hopper) — tag gpu_arch correctly in frontmatter."
