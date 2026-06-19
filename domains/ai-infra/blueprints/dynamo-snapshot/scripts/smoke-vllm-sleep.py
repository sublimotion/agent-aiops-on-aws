#!/usr/bin/env python3
"""
Bridging-cell smoke harness for dynamo-snapshot.

Mirrors the Dynamo snapshot-agent C/R protocol, but driven from a shell
orchestrator on the host (no K8s):

  1. Construct vLLM offline LLM with enable_sleep_mode=True.
  2. Generate once with deterministic settings (warms CUDA graphs, kernels).
  3. Hash first 64 token IDs as the byte-equality reference.
  4. llm.sleep(level=1)  -> KV cache + activations released from GPU memory.
                            Weights stay (else snapshot has nothing to restore).
  5. Print READY_FOR_CHECKPOINT pid=<pid>, block on SIGUSR1.
     Orchestrator runs cuda-checkpoint lock/checkpoint + criu dump here.
  6. After SIGUSR1 (post-restore), call llm.wake_up().
  7. Re-generate with the same prompt+seed, hash the first 64 tokens.
  8. Compare hashes; exit 0 on match else 1.

Determinism: temperature=0, seed=42, n=1, max_tokens=64, no concurrency.
"""

import json
import os
import signal
import sys
import threading
import hashlib

WAKE_EVENT = threading.Event()


def _on_sigusr1(_signum, _frame):
    WAKE_EVENT.set()


def main() -> int:
    signal.signal(signal.SIGUSR1, _on_sigusr1)

    os.environ.setdefault("HF_HOME", "/mnt/nvme/hf")
    # Defensive: keep libuv off io_uring even if the workload spawns threads
    # that touch a libuv loop. Dynamo achieves the same via seccomp.
    os.environ.setdefault("UV_USE_IO_URING", "0")
    # vLLM defaults to uvloop in API server paths; the offline LLM class is
    # synchronous and shouldn't, but pin it just in case any background
    # threads pick it up.
    os.environ.setdefault("VLLM_USE_TRITON_FLASH_ATTN", "0")

    from vllm import LLM, SamplingParams

    out_dir = os.environ.get("SMOKE_OUT_DIR", "/mnt/nvme/smoke")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[harness] constructing vLLM with enable_sleep_mode=True ...", flush=True)
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        dtype="float16",
        gpu_memory_utilization=0.40,  # ~9.2 GiB on 23 GiB L4 — fits in 15 GiB host RAM
        enforce_eager=True,           # avoid CUDA graph capture surprises
        enable_sleep_mode=True,
        max_model_len=2048,
        seed=42,
    )

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=64,
        n=1,
        seed=42,
    )

    prompt = "Explain in one sentence why Kubernetes pods are ephemeral."

    # 1) Pre-checkpoint generate
    pre = llm.generate([prompt], sp, use_tqdm=False)
    pre_token_ids = list(pre[0].outputs[0].token_ids)
    pre_hash = hashlib.sha256(
        b"".join(t.to_bytes(4, "little") for t in pre_token_ids)
    ).hexdigest()
    pre_text = pre[0].outputs[0].text
    with open(os.path.join(out_dir, "pre.json"), "w") as f:
        json.dump(
            {"token_ids": pre_token_ids, "sha256": pre_hash, "text": pre_text},
            f,
            indent=2,
        )
    print(f"[harness] pre  hash={pre_hash} ntok={len(pre_token_ids)}", flush=True)

    # 2) Sleep — the gate-2 lever
    print(f"[harness] llm.sleep(level=1) ...", flush=True)
    llm.sleep(level=1)
    # Encourage Python/PyTorch to release any cached allocator blocks
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    # 3) Block for orchestrator
    pid = os.getpid()
    print(f"READY_FOR_CHECKPOINT pid={pid}", flush=True)
    sys.stdout.flush()
    WAKE_EVENT.wait()
    print(f"[harness] SIGUSR1 received, llm.wake_up() ...", flush=True)

    # 4) Wake + re-generate
    llm.wake_up()
    post = llm.generate([prompt], sp, use_tqdm=False)
    post_token_ids = list(post[0].outputs[0].token_ids)
    post_hash = hashlib.sha256(
        b"".join(t.to_bytes(4, "little") for t in post_token_ids)
    ).hexdigest()
    post_text = post[0].outputs[0].text
    with open(os.path.join(out_dir, "post.json"), "w") as f:
        json.dump(
            {"token_ids": post_token_ids, "sha256": post_hash, "text": post_text},
            f,
            indent=2,
        )
    print(f"[harness] post hash={post_hash} ntok={len(post_token_ids)}", flush=True)

    if pre_hash == post_hash:
        print("[harness] GATE_1_PASS: token IDs byte-identical", flush=True)
        return 0
    print("[harness] GATE_1_FAIL: token-ID hashes differ", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
