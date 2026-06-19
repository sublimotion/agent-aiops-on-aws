#!/usr/bin/env python3
"""round_runner.py — single-round orchestrator for the continuous-calibration RLVR loop.

For round N:
  1. Load base model (Qwen3.5-27B) + prior adapter (Gen0 for N=1, Gen_{N-1} for N>1).
  2. SFT on round_N_train's resolved=1 trajectories → Gen_N LoRA adapter.
  3. Generate patches on round_N_control (never-seen) with Gen_N model.
  4. Generate patches on drift_audit_300 (never-seen, same across all rounds) with Gen_N model.
  5. Emit predictions.jsonl files for docker_gold_eval to consume.

The actual gold eval and verifier recalibration happen in separate scripts (run_round.sh glues).
Keeping this script pure-generation means it's GPU-bound only — CPU eval runs in parallel elsewhere.

Why we split generation from eval:
  - GPU (p4de) is the expensive resource; don't waste it waiting on CPU Docker runs.
  - Eval (m7i Docker) can happen concurrently with next round's training start-up.
  - Eval results write to a known path; verifier_recalibrate.py picks them up.
"""

import argparse
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


def load_base_model(model_id: str, **kwargs):
    """Load either a pure-text Causal LM or a VLM (e.g. Qwen3.5-27B is VLM).

    Detects model_type from config and picks the right AutoClass.
    VLMs use AutoModelForImageTextToText; LoRA-wrapping then applies to the
    language model projections (target_modules like q_proj/k_proj exist on both).
    """
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=kwargs.get("trust_remote_code", True))
    archs = getattr(cfg, "architectures", []) or []
    is_vlm = any("ConditionalGeneration" in a or "ImageTextToText" in a for a in archs) \
             or hasattr(cfg, "vision_config")
    if is_vlm:
        print(f"[load_base_model] {model_id} detected as VLM ({archs}); using AutoModelForImageTextToText")
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


_CHECKPOINT_REQUESTED = False


def _sigusr1(*_):
    global _CHECKPOINT_REQUESTED
    _CHECKPOINT_REQUESTED = True
    print("[round_runner] SIGUSR1 received — checkpoint at next step", flush=True)


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def _to_plain(v):
    """Recursively convert numpy/pandas containers to plain Python lists/dicts."""
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        return [_to_plain(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {k: _to_plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_to_plain(x) for x in v]
    return v


def _parse_tool_call_args(tcs):
    """Qwen3-Coder chat template expects tool_call.function.arguments to be a dict
    (it does `.items`), but Nebius stored it as a JSON string. Parse it here.
    """
    import json as _json
    out = []
    for tc in tcs or []:
        tc = dict(tc)
        fn = tc.get("function")
        if fn:
            fn = dict(fn)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = _json.loads(args)
                except Exception:
                    fn["arguments"] = {"_raw": args}
            tc["function"] = fn
        out.append(tc)
    return out


def format_for_sft(record: dict, tokenizer) -> str:
    traj = record.get("trajectory")
    if traj is None:
        return ""
    traj = _to_plain(traj)  # ndarray -> list-of-dict
    messages = []
    for turn in traj:
        role = turn.get("role")
        content = turn.get("content", "") or ""
        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            msg = {"role": "assistant", "content": content}
            tcs = turn.get("tool_calls")
            if tcs:
                msg["tool_calls"] = _parse_tool_call_args(tcs)
            messages.append(msg)
        elif role == "tool":
            messages.append({"role": "tool", "content": content,
                             "tool_call_id": turn.get("tool_call_id", "")})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def train_round(args, round_num: int, out_dir: Path) -> Path:
    """SFT on round_N_train. Returns path to final LoRA adapter."""
    train_path = Path(args.splits_dir) / f"round_{round_num}_train.jsonl"
    print(f"[round {round_num}] train set: {train_path}")
    records = load_jsonl(train_path)
    # If the split files don't carry `trajectory` (they didn't in the first build),
    # join in the trajectory column from the Nebius parquet. Splits are pinned by
    # trajectory_id so this is safe.
    if records and records[0].get("trajectory") is None and args.nebius_parquet:
        import pandas as pd
        print(f"[round {round_num}] trajectory missing from split; joining from {args.nebius_parquet}")
        wanted = {r["trajectory_id"] for r in records}
        traj_df = pd.read_parquet(args.nebius_parquet, columns=["trajectory_id", "trajectory"])
        traj_df = traj_df[traj_df["trajectory_id"].isin(wanted)]
        tmap = dict(zip(traj_df["trajectory_id"], traj_df["trajectory"]))
        for r in records:
            r["trajectory"] = tmap.get(r["trajectory_id"])
    print(f"[round {round_num}] {len(records)} training trajectories")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prev_adapter = None
    if round_num == 1:
        # Round 1 may start from a prior Gen0 adapter, or from bare base model.
        # Only reuse if adapter_config.json's base_model_name_or_path matches our base.
        if args.gen0_adapter and Path(args.gen0_adapter).exists() and \
           (Path(args.gen0_adapter) / "adapter_model.safetensors").exists():
            try:
                import json as _json
                with open(Path(args.gen0_adapter) / "adapter_config.json") as _f:
                    ac = _json.load(_f)
                adapter_base = ac.get("base_model_name_or_path", "")
                if adapter_base == args.base_model:
                    prev_adapter = args.gen0_adapter
                else:
                    print(f"[round {round_num}] Gen0 adapter is for {adapter_base!r}, "
                          f"not our base {args.base_model!r} — starting fresh LoRA")
            except Exception as e:
                print(f"[round {round_num}] Gen0 adapter config unreadable: {e} — starting fresh LoRA")
    else:
        prev_adapter = str(Path(args.output_root) / f"round_{round_num - 1}" / "adapter")
    print(f"[round {round_num}] base={args.base_model}, prev_adapter={prev_adapter or 'NONE (fresh LoRA on raw base)'}")

    base = load_base_model(
        args.base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto",
    )
    if prev_adapter:
        model = PeftModel.from_pretrained(base, str(prev_adapter), is_trainable=True)
    else:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(base, lora_cfg)
        model.print_trainable_parameters()

    texts = []
    skipped = 0
    first_error = None
    for r in records:
        try:
            s = format_for_sft(r, tokenizer)
            if s:
                texts.append(s)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            if first_error is None:
                first_error = f"{type(e).__name__}: {e}"
    if first_error:
        print(f"[round {round_num}] first format error: {first_error}")
    print(f"[round {round_num}] formatted {len(texts)} trajectories, skipped {skipped}")
    if not texts:
        raise SystemExit(f"No trajectories formatted — check format_for_sft. First error: {first_error}")
    ds = Dataset.from_dict({"text": texts})

    adapter_out = out_dir / "adapter"
    cfg = SFTConfig(
        output_dir=str(out_dir / "train_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_seq,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tokenizer)

    signal.signal(signal.SIGUSR1, _sigusr1)

    class CkptOnSignal(TrainerCallback):
        def on_step_end(self, _args, state, control, **_kw):
            global _CHECKPOINT_REQUESTED
            if _CHECKPOINT_REQUESTED:
                control.should_save = True
                _CHECKPOINT_REQUESTED = False
            return control
    trainer.add_callback(CkptOnSignal())

    trainer.train()
    trainer.save_model(str(adapter_out))
    print(f"[round {round_num}] adapter saved to {adapter_out}")

    del model, base, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_out


def generate_patches(adapter_path: Path, base_model: str, instances_path: Path,
                     output_path: Path, vllm_host: str = "localhost", vllm_port: int = 8000):
    """Call OpenHands batch runner with vLLM serving (base+adapter).

    Same logic as gen0_rebaseline.py; extracted into run_openhands_batch-compatible shape.
    Writes output_path in SWE-bench predictions.jsonl format.
    """
    import subprocess
    # Launch vLLM
    # Give GPU memory from prior SFT a chance to release fully.
    # (race with vLLM's 92% claim — lower to 0.80 and delay for headroom.)
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    time.sleep(30)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--tensor-parallel-size", "8",
        "--max-model-len", "65536",
        "--gpu-memory-utilization", "0.80",
        "--enable-lora",
        "--lora-modules", f"gen={adapter_path}",
        "--port", str(vllm_port),
        "--host", vllm_host,
    ]
    print(f"[generate_patches] launching vLLM: {' '.join(cmd)}")
    vllm_proc = subprocess.Popen(cmd)
    import urllib.request, urllib.error
    for i in range(120):
        try:
            urllib.request.urlopen(f"http://{vllm_host}:{vllm_port}/v1/models", timeout=3)
            print(f"[generate_patches] vLLM ready after {i*5}s")
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(5)
    else:
        vllm_proc.terminate()
        raise RuntimeError("vLLM failed to start within 10 minutes")

    try:
        # Load instances (one row per instance_id)
        instances = load_jsonl(instances_path)
        instance_ids = [r["instance_id"] for r in instances]
        print(f"[generate_patches] generating on {len(instance_ids)} instances")

        # Write a SWE-bench-compatible task list file for the OpenHands runner
        task_file = output_path.parent / f"{output_path.stem}_tasks.jsonl"
        with open(task_file, "w") as f:
            for r in instances:
                f.write(json.dumps({"instance_id": r["instance_id"]}) + "\n")

        # Invoke OpenHands eval runner
        # This requires OpenHands v0.54.0 to be installed in the venv, with SWE-bench adapter
        openhands_cmd = [
            sys.executable, "-m", "openhands.core.main",
            "--agent", "CodeActAgent",
            "--llm-base-url", f"http://{vllm_host}:{vllm_port}/v1",
            "--llm-model", "gen",
            "--llm-api-key", "dummy",
            "--eval-dataset", "princeton-nlp/SWE-bench_Lite",
            "--eval-task-ids-file", str(task_file),
            "--eval-output", str(output_path),
            "--eval-max-iterations", "30",
        ]
        print(f"[generate_patches] openhands: {' '.join(openhands_cmd)}")
        subprocess.run(openhands_cmd, check=False)
    finally:
        vllm_proc.send_signal(signal.SIGINT)
        try: vllm_proc.wait(timeout=30)
        except subprocess.TimeoutExpired: vllm_proc.kill()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True, help="Which round (1-5)")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--gen0-adapter", required=True, help="Path to Gen0 LoRA adapter")
    p.add_argument("--splits-dir", required=True, help="Directory containing round_N_*.jsonl")
    p.add_argument("--output-root", required=True, help="Parent dir for round_N/ outputs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--nebius-parquet", default=None,
                   help="Path to trajectories.parquet; joined to split records if trajectory column is missing")
    p.add_argument("--skip-train", action="store_true", help="Reuse existing adapter (for resume)")
    p.add_argument("--skip-generate", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_root) / f"round_{args.round}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # Step 1: SFT
    if args.skip_train and (out_dir / "adapter").exists():
        adapter_path = out_dir / "adapter"
        print(f"[round {args.round}] reusing existing adapter: {adapter_path}")
    else:
        adapter_path = train_round(args, args.round, out_dir)
    train_s = time.time() - t0

    # Step 2: Generate on round_N_control
    t1 = time.time()
    control_out = out_dir / "control_predictions.jsonl"
    if args.skip_generate and control_out.exists():
        print(f"[round {args.round}] skipping control gen, output exists")
    else:
        generate_patches(
            adapter_path, args.base_model,
            Path(args.splits_dir) / f"round_{args.round}_control.jsonl",
            control_out,
        )
    control_gen_s = time.time() - t1

    # Step 3: Generate on drift_audit_300 (same across all rounds — this is the drift trajectory signal)
    t2 = time.time()
    drift_out = out_dir / "drift_audit_predictions.jsonl"
    if args.skip_generate and drift_out.exists():
        print(f"[round {args.round}] skipping drift gen, output exists")
    else:
        generate_patches(
            adapter_path, args.base_model,
            Path(args.splits_dir) / "drift_audit_300.jsonl",
            drift_out,
        )
    drift_gen_s = time.time() - t2

    summary = {
        "round": args.round,
        "adapter_path": str(adapter_path),
        "control_predictions": str(control_out),
        "drift_audit_predictions": str(drift_out),
        "elapsed_s": {
            "train": train_s,
            "control_gen": control_gen_s,
            "drift_gen": drift_gen_s,
            "total": time.time() - t0,
        },
        "next_step": "run docker_gold_eval.py on both predictions files, then verifier_recalibrate.py",
    }
    with open(out_dir / "round_runner_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
