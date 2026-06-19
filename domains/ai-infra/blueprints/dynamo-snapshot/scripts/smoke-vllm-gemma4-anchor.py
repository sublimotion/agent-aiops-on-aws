#!/usr/bin/env python3
"""
Anchor-cell harness for dynamo-snapshot — Gemma-4-26B-A4B-it on H200.

Two-path design:
  PATH=A (correctness, default): enforce_eager=True, sleep+wake gates
  PATH=B (anchor compare):       enforce_eager=False (CUDA graphs + compile),
                                 sleep+wake; measures restore TTFT vs Modal 22s

Same orchestration contract as smoke-vllm-sleep.py:
  1. Construct LLM, generate once (warm up), hash 64-token IDs.
  2. llm.sleep(level=1).
  3. Print READY_FOR_CHECKPOINT pid=<pid>, block on SIGUSR1.
  4. After SIGUSR1: llm.wake_up(), measure t_first_token, regenerate, hash.
  5. Compare hashes; exit 0 on match else 1.

Outputs into $SMOKE_OUT_DIR (default /mnt/nvme/smoke):
  pre.json, post.json (token IDs + sha256 + timing)
  ttft.json (post-restore wake_up + first-token wallclock)
"""

import json
import os
import signal
import sys
import time
import threading
import hashlib

WAKE_EVENT = threading.Event()


def _on_sigusr1(_signum, _frame):
    WAKE_EVENT.set()


def main() -> int:
    signal.signal(signal.SIGUSR1, _on_sigusr1)

    os.environ.setdefault("HF_HOME", "/mnt/nvme/hf")
    os.environ.setdefault("UV_USE_IO_URING", "0")
    # Match Modal: TRITON_ATTN forced for heterogeneous head dims on Gemma4.
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN_VLLM_V1")

    path_mode = os.environ.get("PATH_MODE", "A")
    use_eager = (path_mode == "A")

    from vllm import LLM, SamplingParams

    out_dir = os.environ.get("SMOKE_OUT_DIR", "/mnt/nvme/smoke")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[harness] PATH_MODE={path_mode} use_eager={use_eager}", flush=True)
    print(f"[harness] constructing vLLM LLM(Gemma-4-26B-A4B-it) ...", flush=True)

    t_load_start = time.monotonic()
    llm_kwargs = dict(
        model="google/gemma-4-26B-A4B-it",
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enable_sleep_mode=True,
        max_model_len=8192,
        seed=42,
        tensor_parallel_size=1,
    )
    if use_eager:
        llm_kwargs["enforce_eager"] = True
    else:
        llm_kwargs["enforce_eager"] = False
        # Match Modal's args where possible for the offline path
        llm_kwargs["compilation_config"] = {"compile_sizes": [1, 2, 4, 8, 16, 32]}

    llm = LLM(**llm_kwargs)
    t_load_end = time.monotonic()
    print(f"[harness] LLM constructed in {t_load_end - t_load_start:.2f}s", flush=True)

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=64,
        n=1,
        seed=42,
    )

    prompt = "Explain in one sentence why Kubernetes pods are ephemeral."

    # 1) Pre-checkpoint generate (warmup)
    t_warm_start = time.monotonic()
    pre = llm.generate([prompt], sp, use_tqdm=False)
    t_warm_end = time.monotonic()
    pre_token_ids = list(pre[0].outputs[0].token_ids)
    pre_hash = hashlib.sha256(
        b"".join(t.to_bytes(4, "little") for t in pre_token_ids)
    ).hexdigest()
    pre_text = pre[0].outputs[0].text
    with open(os.path.join(out_dir, "pre.json"), "w") as f:
        json.dump({
            "token_ids": pre_token_ids,
            "sha256": pre_hash,
            "text": pre_text,
            "warm_seconds": t_warm_end - t_warm_start,
            "load_seconds": t_load_end - t_load_start,
            "path_mode": path_mode,
        }, f, indent=2)
    print(f"[harness] pre  hash={pre_hash} warm={t_warm_end - t_warm_start:.2f}s "
          f"ntok={len(pre_token_ids)}", flush=True)

    # 2) Sleep
    print(f"[harness] llm.sleep(level=1) ...", flush=True)
    t_sleep_start = time.monotonic()
    llm.sleep(level=1)
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    t_sleep_end = time.monotonic()
    print(f"[harness] sleep done in {t_sleep_end - t_sleep_start:.2f}s", flush=True)

    # 3) Block for orchestrator
    pid = os.getpid()
    print(f"READY_FOR_CHECKPOINT pid={pid}", flush=True)
    sys.stdout.flush()
    WAKE_EVENT.wait()
    print(f"[harness] SIGUSR1 received", flush=True)

    # 4) wake_up + first-token timing
    t_wake_start = time.monotonic()
    llm.wake_up()
    t_wake_end = time.monotonic()
    print(f"[harness] wake_up done in {t_wake_end - t_wake_start:.2f}s", flush=True)

    # First-token TTFT: use streaming to get true TTFT for the anchor cell
    sp_ttft = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1, n=1, seed=42)
    t_ft_start = time.monotonic()
    _ = llm.generate([prompt], sp_ttft, use_tqdm=False)
    t_ft_end = time.monotonic()
    ttft = t_ft_end - t_ft_start
    # Total restore-to-first-token: measured by the orchestrator from criu restore
    # exit time, but harness-side wake_up + 1-token gen is reported here.

    # Full regen for token equality
    post = llm.generate([prompt], sp, use_tqdm=False)
    post_token_ids = list(post[0].outputs[0].token_ids)
    post_hash = hashlib.sha256(
        b"".join(t.to_bytes(4, "little") for t in post_token_ids)
    ).hexdigest()
    post_text = post[0].outputs[0].text
    with open(os.path.join(out_dir, "post.json"), "w") as f:
        json.dump({
            "token_ids": post_token_ids,
            "sha256": post_hash,
            "text": post_text,
            "wake_seconds": t_wake_end - t_wake_start,
            "ttft_after_wake_seconds": ttft,
            "path_mode": path_mode,
        }, f, indent=2)
    with open(os.path.join(out_dir, "ttft.json"), "w") as f:
        json.dump({
            "wake_seconds": t_wake_end - t_wake_start,
            "ttft_after_wake_seconds": ttft,
            "path_mode": path_mode,
        }, f, indent=2)
    print(f"[harness] post hash={post_hash} wake={t_wake_end - t_wake_start:.2f}s "
          f"ttft={ttft:.3f}s ntok={len(post_token_ids)}", flush=True)

    if pre_hash == post_hash:
        print("[harness] GATE_1_PASS: token IDs byte-identical", flush=True)
        return 0
    print("[harness] GATE_1_FAIL: token-ID hashes differ", flush=True)
    print(f"[harness] pre_text=\"{pre_text}\"", flush=True)
    print(f"[harness] post_text=\"{post_text}\"", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
