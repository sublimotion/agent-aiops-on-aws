#!/usr/bin/env bash
# SERA QLoRA Training for Devstral Small 2
#
# Launches QLoRA fine-tuning with trl SFTTrainer + DeepSpeed ZeRO-2 across 4 GPUs.
#
# Prerequisites:
#   - Training container running with trl, peft, bitsandbytes, deepspeed installed
#   - BF16 base weights at /mnt/nvme/models/devstral-small-2-bf16
#   - Formatted training data at /mnt/nvme/sera-data/formatted/train_trl.jsonl
#
# Usage:
#   bash sera-train.sh [--dry-run]

set -euo pipefail

# --- Configuration ---
MODEL_PATH="/mnt/nvme/models/devstral-small-2-bf16"
TRAIN_DATA="/mnt/nvme/sera-data/mixed/train_trl.jsonl"
VAL_DATA="/mnt/nvme/sera-data/mixed/val_trl.jsonl"
OUTPUT_DIR="/mnt/nvme/sera-checkpoints/mixed-run-001"
DS_CONFIG="/mnt/nvme/sera-scripts/ds_config_zero2.json"

LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Mixed dataset: 180K examples (AI2 SERA 18% + CoderForge 82%)
# batch_size=1 for long sequences, grad_accum=16 for effective batch=64
# 1 epoch = ~2,819 steps; save/eval every 500
BATCH_SIZE=1
GRAD_ACCUM=16
EPOCHS=1
LR="5e-5"
MAX_SEQ_LEN=16384
WARMUP_RATIO=0.03

NUM_GPUS=4

# --- Parse args ---
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

# --- Validate ---
echo "Checking prerequisites..."

for f in "$MODEL_PATH/config.json" "$TRAIN_DATA" "$DS_CONFIG"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

echo "  Model:      $MODEL_PATH"
echo "  Train data: $TRAIN_DATA"
echo "  Output:     $OUTPUT_DIR"
echo "  GPUs:       $NUM_GPUS"
echo "  LoRA:       r=$LORA_R alpha=$LORA_ALPHA"
echo "  Batch:      $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS)) effective"
echo "  Epochs:     $EPOCHS"
echo "  LR:         $LR"
echo "  Max seq:    $MAX_SEQ_LEN"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run complete. Remove --dry-run to start training."
    exit 0
fi

# --- Create output dir ---
mkdir -p "$OUTPUT_DIR"

# --- Launch training ---
echo ""
echo "Starting QLoRA training at $(date -Iseconds)..."

accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file "$DS_CONFIG" \
    $(which trl) sft \
    --model_name_or_path "$MODEL_PATH" \
    --dataset_name "$TRAIN_DATA" \
    --eval_dataset_name "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --load_in_4bit \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --use_peft \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules "$LORA_TARGETS" \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_epochs $EPOCHS \
    --learning_rate $LR \
    --lr_scheduler_type cosine \
    --warmup_ratio $WARMUP_RATIO \
    --max_seq_length $MAX_SEQ_LEN \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --report_to none \
    --trust_remote_code \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "Training complete at $(date -Iseconds)"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Log: $OUTPUT_DIR/train.log"
