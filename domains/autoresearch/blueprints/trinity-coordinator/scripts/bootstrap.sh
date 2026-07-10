#!/usr/bin/env bash
# Trinity coordinator — fresh-box bootstrap for a single-GPU g6e (L40S, AL2023).
#
# Idempotent: safe to re-run. Encodes every environment fix discovered during the
# first bring-up (see lessons.md):
#   - py3.13 venv via uv
#   - datasets==3.6.0 (5.0.0 dropped remote-script loading → LiveCodeBench version_tag breaks)
#   - SVD decompose in fp32 (transformers 5.x loads bf16; torch.svd has no bf16 kernel)
#   - SVD weights cached to S3 so re-staging after a spot reclaim is ~30s, not ~2min
#
# Usage (run ON the box, repo already rsync'd to ~/trinity-coordinator):
#   bash ~/trinity-coordinator/scripts/bootstrap.sh
#
# Env knobs:
#   SVD_S3=s3://agent-aiops-bench-us-east-2/trinity-coordinator/_cache/svd_weights_qwen3-0.6b.pt
set -euo pipefail

ROOT="$HOME/trinity-coordinator"
VENDOR="$ROOT/vendor/trinity-upstream"
SVD_DST="$VENDOR/decomposed_models/Qwen_Qwen3-0.6B/svd_weights.pt"
SVD_S3="${SVD_S3:-s3://agent-aiops-bench-us-east-2/trinity-coordinator/_cache/svd_weights_qwen3-0.6b.pt}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-2}"

echo "[bootstrap] GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- venv -----------------------------------------------------------------
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "[bootstrap] creating py3.13 venv"
  uv venv --python 3.13 "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

echo "[bootstrap] installing deps (datasets pinned to 3.6.0 — see lessons)"
uv pip install -q \
  torch transformers accelerate cma boto3 scipy numpy \
  openai google-genai tiktoken huggingface_hub fire hf_transfer \
  "datasets==3.6.0" "fsspec<=2025.3.0"

# --- SVD weights (cache-or-build) ----------------------------------------
mkdir -p "$(dirname "$SVD_DST")"
if [ -f "$SVD_DST" ]; then
  echo "[bootstrap] SVD weights already present: $SVD_DST"
elif aws s3 ls "$SVD_S3" >/dev/null 2>&1; then
  echo "[bootstrap] restoring SVD weights from S3 cache"
  aws s3 cp "$SVD_S3" "$SVD_DST"
else
  echo "[bootstrap] decomposing Qwen3-0.6B (fp32) — first time, ~90s"
  ( cd "$VENDOR" && python decompose_model.py --model_name Qwen/Qwen3-0.6B --output_dir ./decomposed_models )
  echo "[bootstrap] caching SVD weights to S3 for next reclaim"
  aws s3 cp "$SVD_DST" "$SVD_S3" || echo "[bootstrap] WARN: S3 cache upload failed (non-fatal)"
fi

echo "[bootstrap] verifying import chain + CUDA"
( cd "$VENDOR" && PYTHONPATH=".:$ROOT/scripts" python -c "
import torch; assert torch.cuda.is_available(), 'no CUDA'
import fugu.trainer, fugu.utils, fugu.job_manager  # noqa
from cma_train import CMATrainingLoop  # noqa
import bedrock_clients as bc; bc.install()
import fugu.utils as U
from worker_pool_bedrock import LLM_NAMES
bad=[n for n in LLM_NAMES if not (U._is_oai_model(n) or U._is_anthropic_model(n) or U._is_deepseek_model(n))]
assert not bad, f'these names route to Unsupported (retry-hang): {bad}'
print('[bootstrap] OK — CUDA', torch.cuda.get_device_name(0), '| all', len(LLM_NAMES), 'workers route to Bedrock')
" )

echo "[bootstrap] DONE. Launch smoke with:"
echo "  cd $VENDOR && AWS_DEFAULT_REGION=us-east-2 AWS_REGION=us-east-2 \\"
echo "    HF_DATASETS_TRUST_REMOTE_CODE=1 PYTHONPATH=\$HOME/trinity-coordinator/scripts:. \\"
echo "    TRINITY_S3_URI=s3://agent-aiops-bench-us-east-2/trinity-coordinator/phase05-smoke-\$(date +%Y%m%d-%H%M%S) \\"
echo "    nohup python \$HOME/trinity-coordinator/scripts/run_trinity_agent.py \\"
echo "      --phase smoke --vendor-root . --iters 3 --num-workers 8 --cost-cap-usd 250 > smoke.log 2>&1 &"
