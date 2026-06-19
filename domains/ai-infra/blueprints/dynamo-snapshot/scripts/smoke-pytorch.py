#!/usr/bin/env python3
"""Minimal pure-pytorch smoke for cuda-checkpoint primitive validation.

Avoids vLLM/uvloop/io_uring (which CRIU 4.2 cannot dump). Loads Qwen3-0.6B
with HuggingFace transformers, does a deterministic greedy generation,
prints READY, waits for SIGUSR1, regenerates, checks token equality.
"""
import json, os, signal, sys, time, hashlib, pathlib

os.environ.setdefault("HF_HOME", "/mnt/nvme/hf-cache")
WORK = pathlib.Path("/tmp/dynamo-snapshot")
WORK.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
PROMPT = "The capital of France is"
SEED = 42
MAX_TOKENS = 64

print(f"[smoke] loading {MODEL}", flush=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
mdl.eval()

@torch.inference_mode()
def gen():
    ids = tok(PROMPT, return_tensors="pt").to("cuda")
    out = mdl.generate(
        **ids,
        max_new_tokens=MAX_TOKENS,
        do_sample=False,        # greedy → deterministic
        temperature=1.0,
        num_beams=1,
        pad_token_id=tok.eos_token_id,
    )
    new = out[0, ids["input_ids"].shape[1]:].tolist()
    text = tok.decode(new, skip_special_tokens=False)
    h = hashlib.sha256(",".join(map(str, new)).encode()).hexdigest()
    return {"token_ids": new, "text": text, "sha256": h}

print("[smoke] warm-gen", flush=True)
pre = gen()
(WORK / "pre.json").write_text(json.dumps(pre, indent=2))
print(f"[smoke] PRE  sha256={pre['sha256']}  text={pre['text']!r}", flush=True)

_resume = {"hit": False}
def _on_sigusr1(signum, frame):
    _resume["hit"] = True
signal.signal(signal.SIGUSR1, _on_sigusr1)

print(f"READY_FOR_CHECKPOINT pid={os.getpid()}", flush=True)
sys.stdout.flush()

while not _resume["hit"]:
    time.sleep(0.5)

print("[smoke] post-restore regen", flush=True)
post = gen()
(WORK / "post.json").write_text(json.dumps(post, indent=2))
print(f"[smoke] POST sha256={post['sha256']}  text={post['text']!r}", flush=True)

if pre["token_ids"] == post["token_ids"]:
    print("GATE_1_TOKEN_EQUALITY: PASS", flush=True)
    sys.exit(0)
else:
    print("GATE_1_TOKEN_EQUALITY: FAIL", flush=True)
    sys.exit(1)
