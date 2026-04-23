#!/usr/bin/env python3
"""
CoderForge-Preview SFT on Qwen3.5-122B-A10B MoE via Unsloth.

v2: MoE-stabilized training (fast path, no masking).
  - r=16, alpha=16 (Unsloth recommended for MoE)
  - LR=2e-6 (5x lower to protect router)
  - adamw_8bit optimizer
  - Router aux loss enabled (coef=0.01)
  - Async S3 checkpoint sync every save
  - Spot interruption handler (2-min metadata warning)

Usage:
    python train_unsloth.py
"""

import json
import logging
import os
import subprocess
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional

import datasets
import torch
from unsloth import FastModel
from transformers import TrainingArguments, set_seed
from transformers.trainer_callback import TrainerCallback
from trl import SFTTrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S3_BUCKET = "s3://agent-aiops-checkpoints/coderforge-eval"
MODEL_PATH = "/mnt/nvme/models/qwen35-122b-a10b-bf16"
DATASET_PATH = "/mnt/nvme/coderforge-raw/"
OUTPUT_DIR = "/mnt/nvme/coderforge-unsloth-output-v2/"
MAX_SEQ_LEN = 8192


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def load_dataset_from_path(dataset_path: str) -> datasets.Dataset:
    """Load CoderForge dataset from local parquet files.

    Supports two layouts:
    1. trajectories/ — raw messages (needs format_trajectory)
    2. trajectories-tokenized_qwencoder/ — pre-formatted text in chat_template_applied
    """
    import pathlib

    # Prefer raw trajectories
    trajectories_dir = os.path.join(dataset_path, "trajectories")
    if os.path.isdir(trajectories_dir):
        parquet_files = sorted(str(p) for p in pathlib.Path(trajectories_dir).glob("**/*.parquet"))
        if parquet_files:
            logger.info(f"Loading {len(parquet_files)} parquet files from {trajectories_dir}")
            return datasets.Dataset.from_parquet(parquet_files)

    # Fall back to tokenized (has chat_template_applied text column)
    tokenized_dir = os.path.join(dataset_path, "trajectories-tokenized_qwencoder")
    if os.path.isdir(tokenized_dir):
        parquet_files = sorted(str(p) for p in pathlib.Path(tokenized_dir).glob("**/*.parquet"))
        if parquet_files:
            logger.info(f"Loading {len(parquet_files)} tokenized parquet files (using chat_template_applied)")
            ds = datasets.Dataset.from_parquet(parquet_files)
            # Rename chat_template_applied → text, drop tokenized columns
            if "chat_template_applied" in ds.column_names:
                ds = ds.rename_column("chat_template_applied", "text")
                cols_to_drop = [c for c in ds.column_names if c not in ("text", "reward", "trajectory_id")]
                if cols_to_drop:
                    ds = ds.remove_columns(cols_to_drop)
            return ds

    return datasets.load_dataset(dataset_path, split="train")


def filter_successful_trajectories(ds: datasets.Dataset) -> datasets.Dataset:
    """Filter to reward >= 1.0 trajectories."""
    if "reward" in ds.column_names:
        original_len = len(ds)
        ds = ds.filter(lambda x: x.get("reward", 0) >= 1.0, num_proc=8)
        logger.info(f"Filtered: {original_len} -> {len(ds)} (reward >= 1.0)")
    return ds


def format_trajectory(example: Dict[str, Any], tokenizer) -> Optional[str]:
    """Convert a CoderForge trajectory to a chat-formatted string."""
    messages_raw = example.get("messages") or example.get("trajectory")
    if not messages_raw:
        return None

    if isinstance(messages_raw, str):
        try:
            messages = json.loads(messages_raw)
        except json.JSONDecodeError:
            return None
    else:
        messages = messages_raw

    if not isinstance(messages, list):
        return None

    cleaned = []
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            cleaned.append({"role": msg["role"], "content": str(msg["content"])})

    if len(cleaned) < 2:
        return None
    if not any(m["role"] == "user" for m in cleaned):
        return None

    # Unwrap Processor if needed
    _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    try:
        text = _tok.apply_chat_template(
            cleaned, tokenize=False, add_generation_prompt=False,
        )
        return text
    except Exception:
        return None


# ---------------------------------------------------------------------------
# S3 checkpoint callback (async)
# ---------------------------------------------------------------------------
_upload_threads: list[threading.Thread] = []


def _s3_sync(local_path: str, s3_path: str, blocking: bool = False):
    """Upload checkpoint to S3. Runs in background thread unless blocking=True."""
    cmd = ["aws", "s3", "sync", local_path, s3_path, "--quiet"]

    def _run():
        logger.info(f"S3 upload started: {local_path} -> {s3_path}")
        try:
            subprocess.run(cmd, check=True, timeout=600)
            logger.info(f"S3 upload done: {s3_path}")
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")

    if blocking:
        _run()
    else:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        _upload_threads.append(t)


def _wait_pending_uploads(timeout: float = 90):
    """Wait for all background S3 uploads to finish (used on spot interruption)."""
    for t in _upload_threads:
        t.join(timeout=timeout)
    _upload_threads.clear()


class S3CheckpointCallback(TrainerCallback):
    """Upload every checkpoint to S3 asynchronously."""

    def __init__(self, s3_bucket: str):
        self.s3_bucket = s3_bucket

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(checkpoint_dir):
            _s3_sync(
                checkpoint_dir,
                f"{self.s3_bucket}/unsloth-v2-checkpoint-{state.global_step}/",
            )


# ---------------------------------------------------------------------------
# Spot interruption handler
# ---------------------------------------------------------------------------
SPOT_METADATA_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"


class SpotInterruptionHandler(TrainerCallback):
    """Poll EC2 spot metadata for termination notice. On 2-min warning:
    1. Set should_save + should_training_stop on trainer
    2. The save triggers S3CheckpointCallback
    3. Wait for uploads to finish before exit
    """

    def __init__(self, poll_interval: float = 5.0):
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._interrupted = False
        self._thread: threading.Thread | None = None
        self._control = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._control = control
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Spot interruption handler started")

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                req = urllib.request.Request(SPOT_METADATA_URL, method="GET")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    body = resp.read().decode()
                    logger.warning(f"SPOT INTERRUPTION NOTICE: {body}")
                    self._interrupted = True
                    return
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    pass  # No interruption pending
                else:
                    logger.debug(f"Spot metadata check error: {e}")
            except Exception:
                pass  # Network error, IMDSv2 token issue, etc.
            self._stop_event.wait(self.poll_interval)

    def on_step_end(self, args, state, control, **kwargs):
        if self._interrupted:
            logger.warning(
                f"Spot reclaim detected at step {state.global_step}. "
                f"Triggering emergency save..."
            )
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        self._stop_event.set()
        if self._interrupted:
            logger.warning("Waiting for S3 uploads to complete before exit...")
            _wait_pending_uploads(timeout=90)


# ---------------------------------------------------------------------------
# Loss collapse detector
# ---------------------------------------------------------------------------
class LossCollapseDetector(TrainerCallback):
    """Alert if loss drops suspiciously fast (MoE router collapse)."""

    def __init__(self, threshold: float = 0.01, min_steps: int = 500):
        self.threshold = threshold
        self.min_steps = min_steps
        self.alerted = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and not self.alerted:
            loss = logs.get("loss")
            step = state.global_step
            if loss is not None and step < self.min_steps and loss < self.threshold:
                logger.warning(
                    f"LOSS COLLAPSE ALERT: loss={loss:.6f} at step {step} "
                    f"(< {self.threshold} before step {self.min_steps}). "
                    f"Router may be collapsing. Consider reducing rank or LR."
                )
                self.alerted = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/mnt/nvme/unsloth_train_v2.metrics.log"),
        ],
    )
    set_seed(3407)

    # --------------- Model via Unsloth ---------------
    logger.info(f"Loading model via Unsloth: {MODEL_PATH}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        device_map="balanced",
    )

    # --------------- Re-enable router aux loss ---------------
    for cfg in [model.config, getattr(model.config, "text_config", None)]:
        if cfg and hasattr(cfg, "router_aux_loss_coef"):
            cfg.router_aux_loss_coef = 0.01
            cfg.output_router_logits = True
            logger.info(f"Router aux loss enabled: coef=0.01 on {type(cfg).__name__}")

    # --------------- LoRA ---------------
    model = FastModel.get_peft_model(
        model,
        r=16,  # Unsloth recommended for MoE (r=64 caused 7.3B params → loss collapse)
        lora_alpha=16,
        lora_dropout=0,  # Must be 0 for Unsloth's ParamWrapper
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: {total_params / 1e9:.1f}B total, "
        f"{trainable_params / 1e6:.1f}M trainable ({trainable_params/total_params*100:.3f}%)"
    )

    # --------------- Dataset ---------------
    if hasattr(tokenizer, "tokenizer"):
        _tok = tokenizer.tokenizer
    else:
        _tok = tokenizer
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
        _tok.pad_token_id = _tok.eos_token_id

    raw_ds = load_dataset_from_path(DATASET_PATH)
    raw_ds = filter_successful_trajectories(raw_ds)
    logger.info(f"Dataset: {len(raw_ds)} trajectories after filtering")

    if "text" not in raw_ds.column_names:
        # Raw trajectories — need to format with chat template
        def _add_text_column(example):
            text = format_trajectory(example, tokenizer)
            return {"text": text if text else ""}

        raw_ds = raw_ds.map(_add_text_column, num_proc=8, desc="Formatting trajectories")
        before_filter = len(raw_ds)
        raw_ds = raw_ds.filter(lambda x: len(x["text"]) > 0, num_proc=8)
        logger.info(f"Formatted: {before_filter} -> {len(raw_ds)} valid trajectories")
    else:
        # Pre-formatted (from tokenized dataset) — just filter empty
        before_filter = len(raw_ds)
        raw_ds = raw_ds.filter(lambda x: len(x["text"]) > 0, num_proc=8)
        logger.info(f"Pre-formatted text: {before_filter} -> {len(raw_ds)} valid")

    # Remove all columns except "text" for SFTTrainer
    cols_to_remove = [c for c in raw_ds.column_names if c != "text"]
    if cols_to_remove:
        raw_ds = raw_ds.remove_columns(cols_to_remove)

    # --------------- Training ---------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # effective batch = 32
        learning_rate=2e-6,  # Lower for MoE stability (was 1e-5)
        warmup_steps=200,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=10,
        save_steps=200,  # Save frequently for spot reclaim resilience
        save_total_limit=3,
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        run_name=f"coderforge-unsloth-v2-{time.strftime('%Y%m%d_%H%M%S')}",
        seed=3407,
    )

    # Use SFTTrainer with dataset_text_field — Unsloth's fast tokenization path
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=raw_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        callbacks=[
            S3CheckpointCallback(s3_bucket=S3_BUCKET),
            SpotInterruptionHandler(poll_interval=5.0),
            LossCollapseDetector(threshold=0.01, min_steps=500),
        ],
    )

    logger.info("Starting Unsloth training v2 (r=16, LR=2e-6, fast path)...")
    train_result = trainer.train()

    # Save
    logger.info("Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Wait for any pending async uploads, then upload final model (blocking)
    _wait_pending_uploads(timeout=120)
    _s3_sync(OUTPUT_DIR, f"{S3_BUCKET}/unsloth-v2-final/", blocking=True)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
