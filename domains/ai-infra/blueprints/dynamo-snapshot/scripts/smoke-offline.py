#!/usr/bin/env python3
"""Stage 0 smoke harness — single-process vLLM offline mode.

Designed to be checkpointed by an external orchestrator. Flow:

1. Boot vLLM, generate a deterministic completion, hash token IDs, write to
   /tmp/dynamo-snapshot/pre.json.
2. Print "READY_FOR_CHECKPOINT <pid>" and wait for SIGUSR1.
3. On SIGUSR1, generate again, hash, write post.json, exit 0.

The orchestrator (run-smoke-offline.sh) does the cuda-checkpoint + criu
dump/restore between steps 1 and 2.
"""
import json, os, signal, sys, time, hashlib, pathlib

WORK = pathlib.Path("/tmp/dynamo-snapshot")
WORK.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
PROMPT = "The capital of France is"
SEED = 42
MAX_TOKENS = 64

print(f"[smoke] loading {MODEL}", flush=True)
from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    enforce_eager=True,            # avoid CUDA-graph capture (simpler state)
    enable_sleep_mode=True,        # required by spec for sleep()/wake_up()
)
sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=SEED, n=1)

def gen():
    outs = llm.generate([PROMPT], sp)
    o = outs[0].outputs[0]
    tids = list(o.token_ids)
    text = o.text
    h = hashlib.sha256(",".join(map(str, tids)).encode()).hexdigest()
    return {"text": text, "token_ids": tids, "sha256": h}

print("[smoke] warm-gen + pre-snapshot", flush=True)
pre = gen()
(WORK / "pre.json").write_text(json.dumps(pre, indent=2))
print(f"[smoke] PRE  sha256={pre['sha256']}", flush=True)
print(f"[smoke] PRE  text={pre['text']!r}", flush=True)

# Signal handler for resume
_resume = {"hit": False}
def _on_sigusr1(signum, frame):
    _resume["hit"] = True
signal.signal(signal.SIGUSR1, _on_sigusr1)

print(f"READY_FOR_CHECKPOINT pid={os.getpid()}", flush=True)
sys.stdout.flush()

# Wait for resume signal (after restore)
while not _resume["hit"]:
    time.sleep(0.5)

print("[smoke] post-snapshot regen", flush=True)
post = gen()
(WORK / "post.json").write_text(json.dumps(post, indent=2))
print(f"[smoke] POST sha256={post['sha256']}", flush=True)
print(f"[smoke] POST text={post['text']!r}", flush=True)

if pre["token_ids"] == post["token_ids"]:
    print("GATE_1_TOKEN_EQUALITY: PASS", flush=True)
    sys.exit(0)
else:
    print("GATE_1_TOKEN_EQUALITY: FAIL", flush=True)
    sys.exit(1)
