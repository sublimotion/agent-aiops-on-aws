"""Checkpoint + state persistence for GRPO training.

Per Gate 0.7 (rl-conductor lesson `feedback_grpo_resume_state.md`): on resume,
weights + optimizer.pt is NOT enough. Must persist EVERY piece of state that
influences trajectory selection or rollout generation, or the policy will
diverge at the resume seam.

What we save per checkpoint (s3://.../checkpoints/{alpha}/iter-{n}/):

    model/                       — HuggingFace save_pretrained() of policy weights
    tokenizer/                   — tokenizer.save_pretrained() (idempotent; saves once)
    optimizer.pt                 — AdamW state
    scheduler.pt                 — LR scheduler state
    rng.pt                       — python.random + torch + cuda + numpy RNG states
    train_state.json             — iter idx, samples seen, alpha, batch idx, seed, replay buffer hash
    pool_mapping.json            — neutral_code → (ord, real_model_id) mapping in effect for THIS run
    config.json                  — full TrainConfig (lr, batch_size, KL coef schedule, etc.)
    rollouts.jsonl.zst           — compressed full rollouts captured during the iters since last ckpt
    metrics.jsonl                — per-iter aggregates (correct_rate, mean_reward, format_fail_rate)

Restore order on resume:
    1. Load config.json — bail if alpha or pool_mapping don't match the requested run
    2. Restore RNG before any data loading or rollout generation
    3. Load model + optimizer + scheduler
    4. Restore train_state.json (iter idx, etc.)
    5. Resume training with LR÷10 warmup for first 5 iters + KL β>0 for first 5 iters

Sync cadence:
    - rollouts.jsonl: appended in memory per iter, flushed + uploaded every 5 iters
    - full checkpoint (model+opt+rng+state): every 25 iters AND on graceful shutdown
    - metrics.jsonl: appended after every iter, uploaded every 5 iters

Spot-reclaim safety:
    - All writes go to local NVMe FIRST, then async S3 sync
    - The trainer registers a SIGTERM handler that triggers final flush before exit
    - On boot, restore from latest S3 checkpoint, NOT local NVMe (NVMe may be wiped)
"""
from __future__ import annotations

import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    """Everything that must persist across restarts beyond model/optimizer."""
    iter_idx: int
    samples_seen: int
    alpha: float
    seed: int
    pool_seed: int                          # for code→ord mapping shuffle
    config_hash: str                        # hash of TrainConfig for safety check
    rollout_count: int = 0
    last_lr: float = 0.0


@dataclass
class TrainConfig:
    """Hyperparameters; serialized to config.json. ANY change to this between
    a save and a resume MUST fail loudly — Gate 0.7."""
    base_model: str = "Qwen/Qwen3-8B"
    alpha: float = 1.0
    lr: float = 1e-6
    batch_size: int = 4                     # 4 questions × 64 rollouts = 256
    rollouts_per_question: int = 64
    max_iters: int = 200
    kl_coef: float = 0.0                    # paper-faithful default
    kl_warmup_iters: int = 5                # post-resume: KL>0 for first 5 iters
    kl_warmup_coef: float = 0.01
    lr_warmup_post_resume: float = 0.1      # ÷ 10 for first 5 iters
    lr_warmup_iters_post_resume: int = 5
    max_router_tokens: int = 1024     # Qwen3 reasons inside <think> before answering; 1024 allows ~700 tok of reasoning + Answer line
    seed: int = 17
    pool_seed: int = 17

    def hash(self) -> str:
        import hashlib
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# RNG state — the ACTUAL gap from rl-conductor
# ---------------------------------------------------------------------------

def capture_rng_state() -> dict:
    """Return everything needed to reproduce the next pseudo-random sample."""
    state = {
        "python_random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ---------------------------------------------------------------------------
# S3 sync helpers
# ---------------------------------------------------------------------------

def _bucket_region(bucket: str) -> Optional[str]:
    """Look up bucket's region. Cached per-process via lru_cache so we don't
    spam GetBucketLocation."""
    import functools
    @functools.lru_cache(maxsize=16)
    def _lookup(b):
        try:
            out = subprocess.check_output(
                ["aws", "s3api", "get-bucket-location", "--bucket", b,
                 "--query", "LocationConstraint", "--output", "text"],
                text=True, timeout=15,
            ).strip()
            return out if out and out != "None" else "us-east-1"
        except Exception:
            return None
    return _lookup(bucket)


def _parse_s3_uri(uri: str) -> tuple[Optional[str], str]:
    """Returns (bucket, key) from s3://bucket/key style URI."""
    if uri.startswith("s3://"):
        rest = uri[5:]
        bucket, _, key = rest.partition("/")
        return bucket, key
    return None, uri


def s3_sync(local: Path, remote: str, exclude: Optional[list[str]] = None) -> int:
    """Wrapper around `aws s3 sync`. Returns process exit code; 0 = ok.
    Auto-detects bucket region to avoid PermanentRedirect when AWS_DEFAULT_REGION
    differs from the bucket's region (common when training in us-east-2 with a
    bucket created in us-east-1, or vice versa)."""
    cmd = ["aws", "s3", "sync", str(local), remote, "--only-show-errors"]
    bucket, _ = _parse_s3_uri(remote)
    if bucket:
        region = _bucket_region(bucket)
        if region:
            cmd.extend(["--region", region])
    for pat in (exclude or []):
        cmd.extend(["--exclude", pat])
    try:
        result = subprocess.run(cmd, check=False, timeout=600,
                                capture_output=True, text=True)
        if result.returncode != 0:
            # Surface what actually went wrong
            log.warning("s3 sync rc=%d: %s",
                        result.returncode, (result.stderr or result.stdout)[:500])
        return result.returncode
    except subprocess.TimeoutExpired:
        log.warning("s3 sync timed out for %s -> %s", local, remote)
        return -1


def s3_pull(remote: str, local: Path) -> int:
    """Pull a remote S3 prefix into a local dir. Used on cold-resume."""
    local.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "sync", remote, str(local), "--only-show-errors"]
    bucket, _ = _parse_s3_uri(remote)
    if bucket:
        region = _bucket_region(bucket)
        if region:
            cmd.extend(["--region", region])
    result = subprocess.run(cmd, check=False, timeout=600,
                            capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("s3 pull rc=%d: %s",
                    result.returncode, (result.stderr or result.stdout)[:500])
    return result.returncode


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Persists training state to local NVMe + async-syncs to S3.

    Spot-reclaim semantics:
        local_dir (NVMe) is ephemeral — lost on instance termination.
        s3_prefix is the durable source of truth.
        On boot: pull latest checkpoint from S3 to local_dir, then resume.
        During training: write to local_dir first, then sync to S3.
    """

    def __init__(
        self,
        local_dir: Path,
        s3_prefix: str,
        config: TrainConfig,
        full_ckpt_every: int = 10,    # was 25; lowered after 2026-05-26 spot reclaim
        sync_every: int = 1,          # was 5; rollouts.jsonl is small, sync every iter
    ):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.s3_prefix = s3_prefix.rstrip("/")
        self.config = config
        self.full_ckpt_every = full_ckpt_every
        self.sync_every = sync_every

        # Buffers flushed at sync_every cadence
        self._rollouts_buffer: list[dict] = []
        self._metrics_buffer: list[dict] = []

        # Files that grow append-only across iters
        self.rollouts_path = self.local_dir / "rollouts.jsonl"
        self.metrics_path = self.local_dir / "metrics.jsonl"

        # Persist config + pool mapping ONCE per run; never overwrite
        self._write_immutable_config()

        # Install SIGTERM handler for spot reclaim
        self._final_state: Optional[TrainState] = None
        signal.signal(signal.SIGTERM, self._on_sigterm)
        signal.signal(signal.SIGINT, self._on_sigterm)

    # ------------------------------------------------------------------
    # Cold start / resume
    # ------------------------------------------------------------------

    @classmethod
    def find_latest_iter(cls, s3_prefix: str) -> Optional[int]:
        """Query S3 for the highest iter-N/ key. Used on boot."""
        prefix = s3_prefix.rstrip("/") + "/"
        try:
            out = subprocess.check_output(
                ["aws", "s3", "ls", prefix], text=True, timeout=60,
            )
        except subprocess.CalledProcessError:
            return None
        iters = []
        for line in out.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            name = parts[-1].rstrip("/")
            if name.startswith("iter-"):
                try:
                    iters.append(int(name.split("-", 1)[1]))
                except (ValueError, IndexError):
                    continue
        return max(iters) if iters else None

    def restore_from_s3(self, iter_idx: int) -> tuple[TrainState, dict]:
        """Pull a specific checkpoint from S3 to local_dir, restore, and
        return (TrainState, rng_state). Caller is responsible for loading
        model/optimizer from self.local_dir / 'iter-{n}'."""
        remote = f"{self.s3_prefix}/iter-{iter_idx}/"
        local = self.local_dir / f"iter-{iter_idx}"
        rc = s3_pull(remote, local)
        if rc != 0:
            raise RuntimeError(f"failed to pull {remote}; exit {rc}")

        cfg_on_disk = json.loads((local / "config.json").read_text())
        if cfg_on_disk["hash"] != self.config.hash():
            raise RuntimeError(
                f"checkpoint config hash {cfg_on_disk['hash']} != current "
                f"{self.config.hash()} — refusing to resume across config changes"
            )

        train_state = TrainState(**json.loads((local / "train_state.json").read_text()))
        rng_state = torch.load(local / "rng.pt", map_location="cpu", weights_only=False)
        return train_state, rng_state

    # ------------------------------------------------------------------
    # Per-iter calls
    # ------------------------------------------------------------------

    def append_rollouts(self, rollouts: list[dict]) -> None:
        """Buffer raw rollouts. Flushed every sync_every iters (Gate 0.5)."""
        self._rollouts_buffer.extend(rollouts)

    def append_metric(self, iter_idx: int, **metrics) -> None:
        self._metrics_buffer.append({"iter": iter_idx, "ts": time.time(), **metrics})

    def maybe_sync(self, iter_idx: int) -> None:
        """Called by trainer after every iter. Flushes buffers + S3 syncs every sync_every."""
        if iter_idx % self.sync_every != 0:
            return
        self._flush_append_buffers()
        # Sync only the small files; full ckpt sync handled separately
        rc = s3_sync(
            self.local_dir,
            self.s3_prefix + "/",
            exclude=["iter-*/*", "iter-*"],   # don't re-upload completed iter dirs
        )
        if rc != 0:
            log.warning("sync_every=%d s3 sync returned %d", self.sync_every, rc)

    def maybe_full_checkpoint(
        self,
        iter_idx: int,
        model,
        optimizer,
        scheduler,
        train_state: TrainState,
        pool_mapping: dict,
    ) -> Optional[Path]:
        """Save full checkpoint (model+opt+sched+rng+state+mapping) every
        full_ckpt_every iters. Returns local checkpoint path if saved.

        Always ckpt at iter 0 — banks the first iter of work against spot reclaim.
        (Skipping iter-0 was a mistake that cost us the entire first run on
         2026-05-26 when spot reclaim hit at iter ~80 with no checkpoints saved.)
        """
        if iter_idx % self.full_ckpt_every != 0 and iter_idx != self.config.max_iters:
            return None
        return self.full_checkpoint(iter_idx, model, optimizer, scheduler, train_state, pool_mapping)

    def full_checkpoint(
        self,
        iter_idx: int,
        model,
        optimizer,
        scheduler,
        train_state: TrainState,
        pool_mapping: dict,
    ) -> Path:
        ckpt_dir = self.local_dir / f"iter-{iter_idx}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log.info("[ckpt] iter %d → %s", iter_idx, ckpt_dir)
        t0 = time.monotonic()

        # Save model + tokenizer
        model.save_pretrained(ckpt_dir / "model", safe_serialization=True)
        if hasattr(model, "tokenizer") and model.tokenizer is not None:
            model.tokenizer.save_pretrained(ckpt_dir / "tokenizer")

        # Save optimizer + scheduler
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

        # CRITICAL: RNG state — this was the actual rl-conductor gap
        torch.save(capture_rng_state(), ckpt_dir / "rng.pt")

        # Train state + config + pool mapping
        (ckpt_dir / "train_state.json").write_text(json.dumps(asdict(train_state), indent=2))
        (ckpt_dir / "config.json").write_text(json.dumps(
            {**asdict(self.config), "hash": self.config.hash()}, indent=2
        ))
        (ckpt_dir / "pool_mapping.json").write_text(json.dumps(pool_mapping, indent=2))

        elapsed = time.monotonic() - t0
        log.info("[ckpt] saved locally in %.1fs", elapsed)

        # Async S3 sync (block here so partial ckpt isn't left on disk if interrupted)
        rc = s3_sync(ckpt_dir, f"{self.s3_prefix}/iter-{iter_idx}/")
        if rc != 0:
            log.error("[ckpt] S3 sync FAILED rc=%d — local ckpt at %s; manual sync required", rc, ckpt_dir)
        else:
            log.info("[ckpt] synced to %s/iter-%d/", self.s3_prefix, iter_idx)

        return ckpt_dir

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _flush_append_buffers(self) -> None:
        if self._rollouts_buffer:
            with self.rollouts_path.open("a") as f:
                for r in self._rollouts_buffer:
                    f.write(json.dumps(r) + "\n")
            self._rollouts_buffer.clear()
        if self._metrics_buffer:
            with self.metrics_path.open("a") as f:
                for m in self._metrics_buffer:
                    f.write(json.dumps(m) + "\n")
            self._metrics_buffer.clear()

    def _write_immutable_config(self) -> None:
        cfg_path = self.local_dir / "run_config.json"
        if cfg_path.exists():
            existing = json.loads(cfg_path.read_text())
            if existing.get("hash") != self.config.hash():
                raise RuntimeError(
                    f"local_dir {self.local_dir} already has a different config "
                    f"({existing.get('hash')} != {self.config.hash()}). "
                    f"Resume into a fresh dir or fix the config."
                )
            return
        cfg_path.write_text(json.dumps(
            {**asdict(self.config), "hash": self.config.hash()}, indent=2
        ))

    def _on_sigterm(self, signum, frame):
        log.warning("[ckpt] caught signal %d — flushing buffers", signum)
        try:
            self._flush_append_buffers()
            # Sync the small files; full ckpt only happens on iter cadence
            s3_sync(self.local_dir, self.s3_prefix + "/", exclude=["iter-*/*", "iter-*"])
        except Exception:
            log.exception("[ckpt] flush on SIGTERM failed")
        sys.exit(143)
