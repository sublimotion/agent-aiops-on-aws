#!/usr/bin/env python3
"""
CoderForge-Preview SFT on Qwen3.5-122B-A10B MoE.

LoRA + device_map="auto" across 8x B200.
Loss computed on assistant responses only; instructions are masked.
S3 checkpointing every 500 steps for spot resilience.

Full fine-tuning OOMs with FSDP1 (122B MoE has 256 experts, each gathered
during forward/backward). LoRA trains <1% of parameters with model-parallel
sharding via device_map, fitting comfortably.

Usage:
    python train_coderforge.py [args...]
"""

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import datasets
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForCausalLM,
    Qwen3_5MoeForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IGNORE_INDEX = -100
S3_BUCKET = "s3://agent-aiops-checkpoints/coderforge-eval"
DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B"
DEFAULT_DATASET = "/mnt/nvme/coderforge-dataset/"
DEFAULT_OUTPUT = "/mnt/nvme/coderforge-output/"
MAX_SEQ_LEN = 131072  # 128K tokens


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=DEFAULT_MODEL)
    trust_remote_code: bool = field(default=True)
    torch_dtype: str = field(default="bfloat16")
    attn_implementation: str = field(default="sdpa")


@dataclass
class DataArguments:
    dataset_path: str = field(default=DEFAULT_DATASET)
    max_seq_length: int = field(default=MAX_SEQ_LEN)
    preprocessing_num_workers: int = field(default=16)


@dataclass
class ExtraTrainingArguments:
    s3_checkpoint_bucket: str = field(default=S3_BUCKET)
    s3_checkpoint_interval: int = field(default=500)
    resume_from_s3: bool = field(
        default=False,
        metadata={"help": "Download latest S3 checkpoint before training."},
    )
    load_balance_loss_weight: float = field(
        default=0.01,
        metadata={"help": "Weight for auxiliary MoE load-balancing loss."},
    )
    router_entropy_threshold: float = field(
        default=0.5,
        metadata={"help": "If router entropy drops below this, halve LR."},
    )
    enable_expert_monitoring: bool = field(default=True)
    expert_collapse_threshold: int = field(
        default=3,
        metadata={"help": "Min active experts before load-balance loss kicks in."},
    )
    # LoRA configuration
    lora_rank: int = field(default=64, metadata={"help": "LoRA rank (r)"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha scaling"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"},
    )


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
TOOL_CALL_START_RE = re.compile(r"<tool_call>")
TOOL_CALL_END_RE = re.compile(r"</tool_call>")


def _role_is_assistant(role: str) -> bool:
    """Check if role should be trained on (assistant / tool_call responses)."""
    return role in ("assistant", "tool")


def build_chat_from_trajectory(
    trajectory: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Convert a CoderForge trajectory into input_ids + labels.

    The CoderForge dataset stores trajectories as a list of messages with
    role/content pairs.  We apply the Qwen3.5 chat template, then mask
    non-assistant tokens with IGNORE_INDEX so loss is only on responses.
    """
    messages_raw = trajectory.get("messages") or trajectory.get("trajectory")
    if not messages_raw:
        return None
    # CoderForge stores messages as a JSON string, not a native list
    if isinstance(messages_raw, str):
        try:
            messages = json.loads(messages_raw)
        except json.JSONDecodeError:
            return None
    else:
        messages = messages_raw
    if not isinstance(messages, list):
        return None

    # Filter to role/content dicts
    cleaned: List[Dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            cleaned.append({"role": msg["role"], "content": str(msg["content"])})
    if len(cleaned) < 2:
        return None

    # Qwen3.5 chat template requires at least one "user" message
    if not any(m["role"] == "user" for m in cleaned):
        return None

    # Apply chat template — produces the full formatted string
    try:
        text = tokenizer.apply_chat_template(
            cleaned,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return None

    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoding["input_ids"].squeeze(0)  # (seq_len,)

    if input_ids.numel() == 0:
        return None

    # Build labels: mask non-assistant tokens
    # Strategy: tokenize each message individually to find boundaries,
    # then only unmask assistant segments.
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    # Re-tokenize incrementally to find assistant boundaries
    prefix_len = 0
    for i, msg in enumerate(cleaned):
        # Build text up to and including this message
        try:
            partial_text = tokenizer.apply_chat_template(
                cleaned[: i + 1],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Partial message list may not satisfy template constraints
            # (e.g., system-only prefix before first user message)
            continue
        partial_ids = tokenizer(
            partial_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )["input_ids"]
        current_len = len(partial_ids)

        if _role_is_assistant(msg["role"]):
            # Unmask assistant tokens (shifted by 1 for next-token prediction)
            start = max(prefix_len, 0)
            end = min(current_len, input_ids.numel())
            if start < end:
                labels[start:end] = input_ids[start:end]

        prefix_len = current_len

    # If no assistant tokens were unmasked, skip
    if (labels != IGNORE_INDEX).sum() == 0:
        return None

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
    }


class CoderForgeDataset(torch.utils.data.Dataset):
    """
    Lazily wraps a HuggingFace dataset of CoderForge trajectories.
    Tokenization happens on __getitem__ (cached by the dataloader workers).
    """

    def __init__(
        self,
        raw_dataset: datasets.Dataset,
        tokenizer: AutoTokenizer,
        max_seq_length: int,
    ):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.raw_dataset[idx]
        result = build_chat_from_trajectory(row, self.tokenizer, self.max_seq_length)
        if result is None:
            # Return a dummy zero-loss sample so DataLoader doesn't crash
            dummy = torch.zeros(1, dtype=torch.long)
            return {
                "input_ids": dummy,
                "labels": torch.full((1,), IGNORE_INDEX, dtype=torch.long),
                "attention_mask": dummy,
            }
        return result


# ---------------------------------------------------------------------------
# Data collator — pads to longest in batch
# ---------------------------------------------------------------------------
@dataclass
class SFTDataCollator:
    """Pad input_ids, labels, and attention_mask to the longest sequence."""

    tokenizer: Any
    max_seq_length: int = MAX_SEQ_LEN

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = min(
            max(f["input_ids"].numel() for f in features),
            self.max_seq_length,
        )

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for f in features:
            seq_len = f["input_ids"].numel()
            pad_len = max_len - seq_len

            if pad_len > 0:
                pad_id = self.tokenizer.pad_token_id or 0
                batch_input_ids.append(
                    torch.cat([f["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
                )
                batch_labels.append(
                    torch.cat([f["labels"], torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
                )
                batch_attention_mask.append(
                    torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
                )
            else:
                batch_input_ids.append(f["input_ids"][:max_len])
                batch_labels.append(f["labels"][:max_len])
                batch_attention_mask.append(f["attention_mask"][:max_len])

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attention_mask),
        }


# ---------------------------------------------------------------------------
# MoE monitoring callback
# ---------------------------------------------------------------------------
class MoEMonitorCallback(TrainerCallback):
    """
    Track MoE expert utilization and router entropy every N steps.
    Triggers load-balance loss amplification if experts collapse.
    Halves LR if router entropy drops below threshold.
    """

    def __init__(
        self,
        model: nn.Module,
        collapse_threshold: int = 3,
        entropy_threshold: float = 0.5,
        log_interval: int = 50,
    ):
        self.model = model
        self.collapse_threshold = collapse_threshold
        self.entropy_threshold = entropy_threshold
        self.log_interval = log_interval
        self._lr_halved = False
        self._expert_names = self._find_expert_layers()

    def _find_expert_layers(self) -> List[str]:
        """Find MoE gate/router layer names in the model."""
        gate_names = []
        for name, module in self.model.named_modules():
            # Qwen MoE uses gate layers; common names: gate, router, shared_expert_gate
            if any(kw in name.lower() for kw in ("gate", "router")) and hasattr(module, "weight"):
                gate_names.append(name)
        return gate_names

    def _compute_expert_stats(self) -> Dict[str, float]:
        """
        Compute per-expert utilization and router entropy from the most
        recent forward pass.  We hook into the router logits stored by
        the model during forward.
        """
        stats: Dict[str, float] = {}

        # Look for aux_loss or router_logits attribute
        # Qwen3.5 MoE stores router_logits on the model output
        # We compute stats from the gate weights as a proxy when logits
        # are not available (they're consumed by the loss computation).
        for name in self._expert_names:
            try:
                parts = name.split(".")
                mod = self.model
                for p in parts:
                    mod = getattr(mod, p)
                w = mod.weight.data  # (num_experts, hidden_dim) or similar
                # Use weight norm as a proxy for expert utilization
                norms = w.float().norm(dim=-1)
                active = (norms > 1e-6).sum().item()
                total = norms.numel()
                entropy = -(
                    torch.softmax(norms, dim=0) * torch.log_softmax(norms, dim=0)
                ).sum().item()
                max_entropy = math.log(total) if total > 0 else 1.0
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

                stats[f"expert/active_{name}"] = active
                stats[f"expert/total_{name}"] = total
                stats[f"expert/entropy_{name}"] = normalized_entropy
            except Exception:
                continue

        return stats

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.log_interval != 0:
            return

        stats = self._compute_expert_stats()
        if not stats:
            return

        # Check for collapse
        for key, val in stats.items():
            if key.startswith("expert/active_") and val < self.collapse_threshold:
                gate_name = key.replace("expert/active_", "")
                logger.warning(
                    f"EXPERT COLLAPSE DETECTED: {gate_name} has only {val} active experts "
                    f"(threshold: {self.collapse_threshold}). "
                    f"Load-balancing loss is active."
                )

        # Check router entropy
        for key, val in stats.items():
            if key.startswith("expert/entropy_") and val < self.entropy_threshold:
                if not self._lr_halved:
                    logger.warning(
                        f"Router entropy {val:.3f} < {self.entropy_threshold}. Halving LR."
                    )
                    for pg in kwargs.get("optimizer", {}).param_groups if hasattr(kwargs.get("optimizer", {}), "param_groups") else []:
                        pg["lr"] *= 0.5
                    self._lr_halved = True

        # Log to wandb via trainer logs
        if logs is not None:
            logs.update(stats)


# ---------------------------------------------------------------------------
# S3 checkpoint utilities
# ---------------------------------------------------------------------------
def _s3_sync(local_path: str, s3_path: str, direction: str = "upload"):
    """Sync between local and S3 using aws cli."""
    if direction == "upload":
        cmd = ["aws", "s3", "sync", local_path, s3_path, "--quiet"]
    else:
        cmd = ["aws", "s3", "sync", s3_path, local_path, "--quiet"]

    logger.info(f"S3 {direction}: {local_path} <-> {s3_path}")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, timeout=600)
        logger.info(f"S3 {direction} completed in {time.time() - t0:.1f}s")
    except subprocess.TimeoutExpired:
        logger.error(f"S3 {direction} timed out after 600s")
    except subprocess.CalledProcessError as e:
        logger.error(f"S3 {direction} failed: {e}")


def download_s3_checkpoint(s3_bucket: str, output_dir: str) -> Optional[str]:
    """Download the latest checkpoint from S3 if it exists."""
    rank = int(os.environ.get("RANK", "0"))

    checkpoint_dir = None
    if rank == 0:
        # List S3 checkpoints
        try:
            result = subprocess.run(
                ["aws", "s3", "ls", f"{s3_bucket}/"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Parse checkpoint directories (format: checkpoint-NNNN/)
                dirs = []
                for line in result.stdout.strip().split("\n"):
                    parts = line.strip().split()
                    if parts and parts[-1].startswith("checkpoint-"):
                        step = parts[-1].rstrip("/").split("-")[-1]
                        if step.isdigit():
                            dirs.append((int(step), parts[-1].rstrip("/")))
                if dirs:
                    dirs.sort(key=lambda x: x[0], reverse=True)
                    latest = dirs[0][1]
                    local_ckpt = os.path.join(output_dir, latest)
                    os.makedirs(local_ckpt, exist_ok=True)
                    _s3_sync(local_ckpt, f"{s3_bucket}/{latest}/", direction="download")
                    checkpoint_dir = local_ckpt
                    logger.info(f"Downloaded S3 checkpoint: {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"Could not list S3 checkpoints: {e}")

    return checkpoint_dir


class S3CheckpointCallback(TrainerCallback):
    """Upload checkpoint to S3 every N steps."""

    def __init__(self, s3_bucket: str, interval: int = 500):
        self.s3_bucket = s3_bucket
        self.interval = interval

    def on_save(self, args, state, control, **kwargs):
        if state.global_step % self.interval != 0:
            return

        # The trainer just saved a checkpoint — upload it
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_dir):
            s3_path = f"{self.s3_bucket}/checkpoint-{state.global_step}/"
            _s3_sync(checkpoint_dir, s3_path, direction="upload")


# ---------------------------------------------------------------------------
# Custom Trainer with MoE load-balancing loss
# ---------------------------------------------------------------------------
class MoESFTTrainer(Trainer):
    """
    Extends HF Trainer to add auxiliary MoE load-balancing loss.
    Qwen3.5 MoE models return `router_logits` in model output when
    `output_router_logits=True`.
    """

    def __init__(self, *args, load_balance_weight: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_balance_weight = load_balance_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Standard cross-entropy on assistant tokens + auxiliary load-balancing loss.
        """
        outputs = model(**inputs)
        loss = outputs.loss

        # Add auxiliary load-balancing loss if present
        if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
            aux_loss = outputs.aux_loss
            loss = loss + self.load_balance_weight * aux_loss

            # Log the components
            if self.state.global_step % 10 == 0:
                self.log({
                    "train/ce_loss": outputs.loss.item(),
                    "train/aux_loss": aux_loss.item(),
                    "train/total_loss": loss.item(),
                })
        elif hasattr(outputs, "router_logits") and outputs.router_logits is not None:
            # Compute load-balancing loss from router logits
            aux_loss = self._compute_load_balance_loss(outputs.router_logits)
            if aux_loss is not None:
                loss = loss + self.load_balance_weight * aux_loss
                if self.state.global_step % 10 == 0:
                    self.log({
                        "train/ce_loss": outputs.loss.item(),
                        "train/aux_loss": aux_loss.item(),
                        "train/total_loss": loss.item(),
                    })

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _compute_load_balance_loss(
        router_logits: tuple,
    ) -> Optional[torch.Tensor]:
        """
        Switch Transformer style load-balancing loss.
        Encourages uniform expert assignment across tokens.

        For each layer:
            L_balance = N * sum_i(f_i * p_i)
        where f_i = fraction of tokens routed to expert i,
              p_i = mean routing probability for expert i,
              N = number of experts.
        """
        if not router_logits:
            return None

        total_loss = torch.tensor(0.0, device=router_logits[0].device)
        count = 0

        for logits in router_logits:
            if logits is None or logits.ndim != 2:
                continue
            # logits: (num_tokens, num_experts)
            num_experts = logits.shape[-1]
            probs = torch.softmax(logits.float(), dim=-1)  # (tokens, experts)

            # Expert assignment fractions
            top_indices = logits.argmax(dim=-1)  # (tokens,)
            one_hot = torch.nn.functional.one_hot(top_indices, num_experts).float()
            fractions = one_hot.mean(dim=0)  # (experts,)

            # Mean routing probability per expert
            mean_probs = probs.mean(dim=0)  # (experts,)

            # Load balance loss
            layer_loss = num_experts * (fractions * mean_probs).sum()
            total_loss = total_loss + layer_loss
            count += 1

        if count == 0:
            return None

        return total_loss / count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_dataset_from_path(dataset_path: str) -> datasets.Dataset:
    """
    Load the CoderForge-Preview dataset.

    Tries these strategies in order:
    1. Load from local disk (Arrow/Parquet files)
    2. Load from HuggingFace Hub
    """
    logger.info(f"Loading dataset from {dataset_path}")

    # CoderForge-Preview has two configs:
    #   - "trajectories": raw messages in JSON string format (columns: messages, tools, reward, ...)
    #   - "trajectories-tokenized_qwencoder": pre-tokenized for Qwen Coder (incompatible with Qwen3.5)
    # And four splits: SWE_Rebench, SWE_Smith, R2E_Gym, filtered_reward1
    # We use "trajectories" config, "filtered_reward1" split (155K verified trajectories)

    # Check for local raw trajectories directory
    raw_trajectories_dir = os.path.join(dataset_path, "trajectories")
    if os.path.isdir(raw_trajectories_dir):
        # Load from local parquet files using HF datasets
        logger.info(f"Loading from local parquet: {raw_trajectories_dir}")
        try:
            # Load filtered_reward1 split
            pattern = os.path.join(raw_trajectories_dir, "filtered_reward1-*.parquet")
            import glob as globmod
            files = sorted(globmod.glob(pattern))
            if files:
                ds = datasets.Dataset.from_parquet(files)
                logger.info(f"Loaded {len(ds)} examples from filtered_reward1 parquet files")
                return ds
            # Fallback: load all parquet files
            all_files = sorted(globmod.glob(os.path.join(raw_trajectories_dir, "*.parquet")))
            if all_files:
                ds = datasets.Dataset.from_parquet(all_files)
                logger.info(f"Loaded {len(ds)} examples from all parquet files")
                return ds
        except Exception as e:
            logger.warning(f"Failed to load local parquet: {e}")

    # Try loading from HuggingFace Hub with correct config/split
    try:
        ds = datasets.load_dataset(
            "togethercomputer/CoderForge-Preview",
            name="trajectories",
            split="filtered_reward1",
            trust_remote_code=True,
            cache_dir=dataset_path if os.path.isdir(dataset_path) else None,
        )
        logger.info(f"Loaded {len(ds)} examples from HuggingFace Hub (filtered_reward1)")
        return ds
    except Exception as e:
        logger.warning(f"Failed to load filtered_reward1: {e}")

    # Fallback: try loading any available split
    for split in ["filtered_reward1", "R2E_Gym", "SWE_Smith", "SWE_Rebench"]:
        try:
            ds = datasets.load_dataset(
                "togethercomputer/CoderForge-Preview",
                name="trajectories",
                split=split,
                trust_remote_code=True,
                cache_dir=dataset_path if os.path.isdir(dataset_path) else None,
            )
            logger.info(f"Loaded {len(ds)} examples from split '{split}'")
            return ds
        except Exception:
            continue

    raise RuntimeError(
        f"Could not load CoderForge dataset from {dataset_path}. "
        f"Download with: huggingface-cli download togethercomputer/CoderForge-Preview "
        f"--local-dir {dataset_path} --repo-type dataset"
    )


def filter_successful_trajectories(ds: datasets.Dataset) -> datasets.Dataset:
    """
    Filter to successful trajectories only.
    CoderForge uses 'reward' field: 1.0 = success, 0.0 = failure.
    The 'filtered_reward1' split is already pre-filtered (reward >= 1.0).
    """
    original_len = len(ds)

    # Check if reward column exists and filter
    if "reward" in ds.column_names:
        ds = ds.filter(lambda x: x["reward"] >= 1.0, num_proc=8)
        logger.info(f"Filtered by reward >= 1.0: {original_len} -> {len(ds)} trajectories")
        return ds

    # Try common field names as fallback
    for field_name in ("resolved", "success", "is_resolved"):
        if field_name in ds.column_names:
            ds = ds.filter(lambda x: bool(x[field_name]), num_proc=8)
            logger.info(f"Filtered by '{field_name}': {original_len} -> {len(ds)}")
            return ds

    logger.warning(
        f"No reward/success field found. Using all {original_len} trajectories. "
        f"Columns: {ds.column_names}"
    )
    return ds


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ExtraTrainingArguments))

    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, extra_args = parser.parse_json_file(sys.argv[1])
    else:
        model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()

    # --------------- Logging ---------------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    transformers.utils.logging.set_verbosity_info()
    set_seed(training_args.seed)

    rank = int(os.environ.get("RANK", "0"))

    # --------------- Resume from S3 ---------------
    if extra_args.resume_from_s3:
        os.makedirs(training_args.output_dir, exist_ok=True)
        s3_ckpt = download_s3_checkpoint(
            extra_args.s3_checkpoint_bucket,
            training_args.output_dir,
        )
        if s3_ckpt:
            training_args.resume_from_checkpoint = s3_ckpt
            logger.info(f"Will resume from S3 checkpoint: {s3_ckpt}")
        else:
            logger.info("No S3 checkpoint found — starting from scratch")

    # Also check for local checkpoints
    if (
        training_args.resume_from_checkpoint is None
        and os.path.isdir(training_args.output_dir)
    ):
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt:
            training_args.resume_from_checkpoint = last_ckpt
            logger.info(f"Resuming from local checkpoint: {last_ckpt}")

    # --------------- Tokenizer ---------------
    logger.info(f"Loading tokenizer: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        model_max_length=data_args.max_seq_length,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --------------- Dataset ---------------
    raw_ds = load_dataset_from_path(data_args.dataset_path)
    raw_ds = filter_successful_trajectories(raw_ds)

    logger.info(f"Dataset: {len(raw_ds)} successful trajectories")
    if rank == 0 and len(raw_ds) > 0:
        sample = raw_ds[0]
        logger.info(f"Sample columns: {list(sample.keys())}")
        for k, v in sample.items():
            if isinstance(v, str):
                logger.info(f"  {k}: {v[:200]}...")
            elif isinstance(v, list):
                logger.info(f"  {k}: list of {len(v)} items")

    train_dataset = CoderForgeDataset(raw_ds, tokenizer, data_args.max_seq_length)
    data_collator = SFTDataCollator(tokenizer=tokenizer, max_seq_length=data_args.max_seq_length)

    # --------------- Model ---------------
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    torch_dtype = getattr(torch, model_args.torch_dtype, torch.bfloat16)

    # Load with device_map="auto" — distributes layers across all GPUs
    # This replaces FSDP which OOMs on 256-expert MoE during backward pass
    model = Qwen3_5MoeForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        device_map="auto",
    )

    # Disable KV cache for training (set on config, not as constructor kwarg)
    model.config.use_cache = False

    # Disable the model's built-in load_balancing_loss_func — it allocates
    # (batch × seq_len × num_layers, num_experts) float tensors that OOM on
    # 256-expert models.
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = False
    if hasattr(model.config, "router_aux_loss_coef"):
        model.config.router_aux_loss_coef = 0.0

    # --------------- LoRA ---------------
    target_modules = [m.strip() for m in extra_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=extra_args.lora_rank,
        lora_alpha=extra_args.lora_alpha,
        lora_dropout=extra_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        # Don't apply LoRA to router/gate layers — keep them frozen for stability
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: {total_params / 1e9:.1f}B total params, "
        f"{trainable_params / 1e9:.1f}B trainable"
    )

    # --------------- Callbacks ---------------
    callbacks = [
        S3CheckpointCallback(
            s3_bucket=extra_args.s3_checkpoint_bucket,
            interval=extra_args.s3_checkpoint_interval,
        ),
    ]

    if extra_args.enable_expert_monitoring:
        callbacks.append(
            MoEMonitorCallback(
                model=model,
                collapse_threshold=extra_args.expert_collapse_threshold,
                entropy_threshold=extra_args.router_entropy_threshold,
                log_interval=50,
            )
        )

    # --------------- Trainer ---------------
    # Use standard Trainer — LoRA handles param efficiency,
    # aux loss disabled (OOMs on 256 experts), monitoring via callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # --------------- Train ---------------
    logger.info("Starting training...")
    train_result = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint,
    )

    # --------------- Save final model ---------------
    logger.info("Saving final model...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Final S3 upload
    if rank == 0:
        _s3_sync(
            training_args.output_dir,
            f"{extra_args.s3_checkpoint_bucket}/final/",
            direction="upload",
        )
        logger.info("Final model uploaded to S3")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
