#!/usr/bin/env bash
#
# Launch CoderForge-Preview SFT on Qwen3.5-122B-A10B MoE
# Hardware: 8x B200 (p6-b200.48xlarge), NVSwitch interconnect
# Framework: FSDP2 + ZeRO-3 via HF Trainer + accelerate
#
# Usage:
#   ./launch_training.sh                    # fresh start
#   ./launch_training.sh --resume           # resume from S3 checkpoint
#   ./launch_training.sh --dry-run          # print config, don't launch
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Activate training virtualenv
# ---------------------------------------------------------------------------
VENV_DIR="/mnt/nvme/train-env"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: ${VENV_DIR}"
else
    echo "ERROR: Training venv not found at ${VENV_DIR}"
    echo "Create with: python3 -m venv ${VENV_DIR} && pip install torch transformers accelerate"
    exit 1
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_coderforge.py"

# Model & data
MODEL_NAME="/mnt/nvme/models/qwen35-122b-a10b-bf16"
DATASET_PATH="/mnt/nvme/coderforge-raw/"
OUTPUT_DIR="/mnt/nvme/coderforge-output/"
MAX_SEQ_LEN=8192    # 8K — 248K vocab makes logits huge; 32K OOMs at cross_entropy

# Hardware
NUM_GPUS=8

# Training hyperparameters (SERA reference: LR=1e-5)
LEARNING_RATE="1e-5"
NUM_EPOCHS=1
PER_DEVICE_BATCH_SIZE=4          # 8K context + LoRA — GPUs have headroom (44-68GB/178GB used)
GRADIENT_ACCUMULATION_STEPS=8    # effective batch = 4 × 8 = 32
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# Checkpointing
SAVE_STEPS=500
S3_BUCKET="s3://agent-aiops-checkpoints/coderforge-eval"

# Logging
LOGGING_STEPS=10
REPORT_TO="none"  # Use "wandb" if wandb is configured; "none" for tensorboard-only
RUN_NAME="coderforge-qwen35-122b-a10b-$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# NCCL tuning for B200 NVSwitch
# ---------------------------------------------------------------------------
# B200 NVSwitch topology: all 8 GPUs fully connected via NVSwitch
# NCCL 2.26.2+ required for Blackwell (fixes sm_120 shared memory bug)
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1                    # No InfiniBand on single-node
export NCCL_NET=Socket                      # Force socket transport — skip OFI plugin (double-free on B200)
export NCCL_P2P_LEVEL=NVL                   # Use NVLink/NVSwitch for P2P
export NCCL_NET_GDR_LEVEL=0                 # No GPUDirect RDMA (single-node)
export NCCL_SOCKET_IFNAME=lo                # Loopback for socket fallback
export NCCL_ALGO=Ring,Tree                  # Both algorithms available
export NCCL_PROTO=Simple,LL,LL128           # All protocols
export NCCL_SHM_USE_CUDA_MEMCPY=0           # Disable CUDA memcpy for SHM (Blackwell fix)
export NCCL_CROSS_NIC=0                     # Single-node, no cross-NIC
export LD_PRELOAD=""                        # Prevent aws-ofi-nccl from loading via LD_PRELOAD

# ---------------------------------------------------------------------------
# CUDA / PyTorch tuning
# ---------------------------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1        # Serialize CUDA streams for FSDP
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Detect NCCL hangs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"  # Reduce fragmentation
export OMP_NUM_THREADS=4                    # CPU threads per worker

# Disable torch.compile for training stability (can re-enable after validating)
export TORCH_COMPILE_DISABLE=1

# ---------------------------------------------------------------------------
# LoRA configuration (replaces FSDP — full fine-tune OOMs on 256-expert MoE)
# ---------------------------------------------------------------------------
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

echo "LoRA config: r=${LORA_RANK}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "Target modules: ${LORA_TARGET_MODULES}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
RESUME_FLAG=""
DRY_RUN=false

for arg in "$@"; do
    case "${arg}" in
        --resume)
            RESUME_FLAG="--resume_from_s3 True"
            echo ">>> Will resume from latest S3 checkpoint"
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        *)
            echo "Unknown argument: ${arg}"
            echo "Usage: $0 [--resume] [--dry-run]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "============================================"
echo "CoderForge SFT — Qwen3.5-122B-A10B MoE"
echo "============================================"
echo ""
echo "Model:           ${MODEL_NAME}"
echo "Dataset:         ${DATASET_PATH}"
echo "Output:          ${OUTPUT_DIR}"
echo "GPUs:            ${NUM_GPUS}x B200"
echo "Max seq length:  ${MAX_SEQ_LEN} (128K)"
echo "Batch size:      ${PER_DEVICE_BATCH_SIZE}/GPU × ${GRADIENT_ACCUMULATION_STEPS} accum = $((NUM_GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) effective"
echo "LR:              ${LEARNING_RATE}"
echo "Epochs:          ${NUM_EPOCHS}"
echo "Save steps:      ${SAVE_STEPS}"
echo "S3 bucket:       ${S3_BUCKET}"
echo "Run name:        ${RUN_NAME}"
echo ""

# Check NVIDIA driver
if command -v nvidia-smi &>/dev/null; then
    echo "GPU status:"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    echo ""
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "${GPU_COUNT}" -lt "${NUM_GPUS}" ]; then
        echo "ERROR: Expected ${NUM_GPUS} GPUs but found ${GPU_COUNT}"
        exit 1
    fi
else
    echo "WARNING: nvidia-smi not found — cannot verify GPU configuration"
fi

# Check dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    echo "Download with: huggingface-cli download togethercomputer/CoderForge-Preview --local-dir ${DATASET_PATH}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

if [ "${DRY_RUN}" = true ]; then
    echo ">>> DRY RUN — not launching training"
    exit 0
fi

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
echo "Launching LoRA training with device_map=auto (${NUM_GPUS} GPUs)..."
echo ""

# Single-process launch — device_map="auto" distributes model across GPUs
# No torchrun/FSDP needed. gradient_accumulation gives effective batch size.
exec python "${TRAIN_SCRIPT}" \
    --model_name_or_path "${MODEL_NAME}" \
    --dataset_path "${DATASET_PATH}" \
    --max_seq_length "${MAX_SEQ_LEN}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --lr_scheduler_type "cosine" \
    --logging_steps "${LOGGING_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 3 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --remove_unused_columns False \
    --report_to "${REPORT_TO}" \
    --run_name "${RUN_NAME}" \
    --seed 42 \
    --s3_checkpoint_bucket "${S3_BUCKET}" \
    --s3_checkpoint_interval "${SAVE_STEPS}" \
    --load_balance_loss_weight 0.01 \
    --router_entropy_threshold 0.5 \
    --expert_collapse_threshold 3 \
    --enable_expert_monitoring True \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    ${RESUME_FLAG}
