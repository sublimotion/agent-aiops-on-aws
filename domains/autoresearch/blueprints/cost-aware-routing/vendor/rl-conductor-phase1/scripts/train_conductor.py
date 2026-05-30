"""GRPO training loop for RL Conductor (Phase 1) — optimized for throughput.

Trains Qwen2.5-7B to output workflow orchestration steps using GRPO.
Key optimizations over naive version:
  - Batched generation (all rollouts per question in one forward pass)
  - Concurrent Bedrock API calls via asyncio.gather
  - Separate inference model (no_grad) and training model (grad)
"""

import json
import os
import random
import time
from pathlib import Path

import boto3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import compute_reward, parse_conductor_output
from worker_proxy import WorkerPool
from workflow_runtime import execute_workflow_sync

# === Config ===
MODEL_NAME = "/opt/dlami/nvme/models/Qwen2.5-7B"
DATA_PATH = "/opt/dlami/nvme/rl-conductor/data/train.jsonl"
CHECKPOINT_DIR = "/opt/dlami/nvme/rl-conductor/checkpoints"
S3_BUCKET = "agent-aiops-research"
S3_PREFIX = "rl-conductor/checkpoints/phase1"
LOG_DIR = "/opt/dlami/nvme/rl-conductor/logs"

NUM_ITERATIONS = 200
QUESTIONS_PER_BATCH = 8
ROLLOUTS_PER_QUESTION = 8
BATCH_SIZE = QUESTIONS_PER_BATCH * ROLLOUTS_PER_QUESTION  # 64
LR = 1e-6
MAX_WORKFLOW_DEPTH = 5
MAX_GEN_TOKENS = 512
CHECKPOINT_EVERY = 25
PHASE = 1
GENERATION_BATCH_SIZE = 16  # Generate 16 at a time (memory-bound)

# === Conductor system prompt ===
CONDUCTOR_SYSTEM_PROMPT = """You are a Conductor that orchestrates a team of {num_workers} AI workers to solve problems.

Workers: IDs 0 to {max_id}. Each has different strengths.

For each workflow step, output exactly three Python lists:
subtasks = ["description of task for worker 1", "description for worker 2", ...]
model_id = [worker_id_1, worker_id_2, ...]
access_list = [[], [0], ...]

Rules:
- All three lists must have the same length
- model_id values must be integers in [0, {max_id}]
- access_list[i] contains indices of previous outputs worker i can see (empty [] for first step)
- You may output multiple steps (Step 1, Step 2, etc.)
- After workflow, write: FINAL ANSWER: <your answer>

Example:
subtasks = ["Solve the equation step by step", "Verify the solution"]
model_id = [0, 2]
access_list = [[], [0]]
FINAL ANSWER: x = 42"""


def load_training_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_prompt(question: str, num_workers: int) -> str:
    system = CONDUCTOR_SYSTEM_PROMPT.format(
        num_workers=num_workers,
        max_id=num_workers - 1,
    )
    return f"{system}\n\nProblem: {question}\n\nYour workflow:"


@torch.no_grad()
def batch_generate(model, tokenizer, prompts: list[str], batch_size: int = 16) -> list[str]:
    """Generate completions for multiple prompts in batches."""
    all_outputs = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_GEN_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        for j, output in enumerate(outputs):
            prompt_len = inputs["attention_mask"][j].sum().item()
            generated = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            all_outputs.append(generated)

    return all_outputs


MAX_EXEC_TIME = 180  # Hard time limit for execution phase (3 min)


def _exec_worker(generated_texts, questions, rollouts_per_q, worker_pool, result_file):
    """Run in a forked subprocess — can be killed on timeout."""
    results = []
    deadline = time.time() + MAX_EXEC_TIME

    for idx, gen_text in enumerate(generated_texts):
        q_idx = idx // rollouts_per_q
        question = questions[q_idx]

        parsed = parse_conductor_output(gen_text)
        if parsed is None or time.time() > deadline:
            final_answer = extract_final_answer_quick(gen_text)
            reward = 0.0 if parsed is None else 0.5
        else:
            try:
                final_answer = execute_workflow_sync(gen_text, worker_pool, max_depth=MAX_WORKFLOW_DEPTH)
                reward = compute_reward(gen_text, final_answer, question["answer"], question.get("type", "math"))
            except Exception:
                final_answer = extract_final_answer_quick(gen_text)
                reward = 0.5

        results.append({
            "response": gen_text,
            "final_answer": final_answer,
            "reward": reward,
            "gold_answer": question["answer"],
        })

    with open(result_file, "w") as f:
        json.dump(results, f)


def execute_rollouts_sync(
    generated_texts: list[str],
    questions: list[dict],
    rollouts_per_q: int,
    worker_pool: WorkerPool,
) -> list[dict]:
    """Execute workflows in a subprocess with hard kill timeout."""
    import subprocess
    import tempfile
    import pickle

    result_file = "/tmp/rl_conductor_exec_results.json"
    input_file = "/tmp/rl_conductor_exec_input.pkl"

    # Remove stale results from previous iteration
    if os.path.exists(result_file):
        os.remove(result_file)

    # Serialize inputs
    with open(input_file, "wb") as f:
        pickle.dump((generated_texts, questions, rollouts_per_q), f)

    # Run in subprocess that can be killed
    script = f'''
import sys, json, pickle, time
sys.path.insert(0, "/opt/dlami/nvme/rl-conductor")
from reward import compute_reward, parse_conductor_output
from worker_proxy import WorkerPool
from workflow_runtime import execute_workflow_sync

with open("{input_file}", "rb") as f:
    generated_texts, questions, rollouts_per_q = pickle.load(f)

worker_pool = WorkerPool(phase=1)
MAX_WORKFLOW_DEPTH = 5
deadline = time.time() + {MAX_EXEC_TIME}
results = []

for idx, gen_text in enumerate(generated_texts):
    q_idx = idx // rollouts_per_q
    question = questions[q_idx]
    parsed = parse_conductor_output(gen_text)
    if parsed is None or time.time() > deadline:
        from train_conductor import extract_final_answer_quick
        final_answer = extract_final_answer_quick(gen_text)
        reward = 0.0 if parsed is None else 0.5
    else:
        try:
            final_answer = execute_workflow_sync(gen_text, worker_pool, max_depth=MAX_WORKFLOW_DEPTH)
            reward = compute_reward(gen_text, final_answer, question["answer"], question.get("type", "math"))
        except Exception:
            from train_conductor import extract_final_answer_quick
            final_answer = extract_final_answer_quick(gen_text)
            reward = 0.5
    results.append({{"response": gen_text, "final_answer": final_answer, "reward": reward, "gold_answer": question["answer"]}})

with open("{result_file}", "w") as f:
    json.dump(results, f)
'''

    proc = subprocess.Popen(
        ["/opt/pytorch/bin/python3", "-c", script],
        env={**os.environ, "PYTHONPATH": "/opt/dlami/nvme/rl-conductor"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        proc.wait(timeout=MAX_EXEC_TIME + 30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # Read results or generate fallback
    if os.path.exists(result_file):
        try:
            with open(result_file) as f:
                results = json.load(f)
            if len(results) == len(generated_texts):
                return results
        except (json.JSONDecodeError, IOError):
            pass

    # Fallback: assign rewards without API calls
    results = []
    for idx, gen_text in enumerate(generated_texts):
        q_idx = idx // rollouts_per_q
        question = questions[q_idx]
        parsed = parse_conductor_output(gen_text)
        final_answer = extract_final_answer_quick(gen_text)
        reward = 0.0 if parsed is None else 0.5
        results.append({
            "response": gen_text,
            "final_answer": final_answer,
            "reward": reward,
            "gold_answer": question["answer"],
        })
    return results


def extract_final_answer_quick(text: str) -> str:
    """Quick extraction without API calls."""
    import re
    match = re.search(r'FINAL\s+ANSWER\s*:\s*(.*?)(?:\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    boxed = re.search(r'\\boxed\{(.*?)\}', text)
    if boxed:
        return boxed.group(1)
    return ""


def compute_grpo_loss(
    model,
    tokenizer,
    prompts: list[str],
    rollout_results: list[dict],
    questions: list[dict],
    rollouts_per_q: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute GRPO loss with gradient accumulation (one forward pass at a time)."""
    n_contributing = 0
    accumulated_loss = 0.0

    for q_idx in range(len(questions)):
        start = q_idx * rollouts_per_q
        end = start + rollouts_per_q
        group_results = rollout_results[start:end]
        prompt = prompts[start]

        rewards = torch.tensor([r["reward"] for r in group_results], dtype=torch.float32)
        mean_r = rewards.mean()
        std_r = rewards.std()

        if std_r < 1e-6:
            continue

        advantages = (rewards - mean_r) / std_r

        for i, (result, adv) in enumerate(zip(group_results, advantages)):
            if abs(adv.item()) < 0.01:
                continue

            full_text = prompt + result["response"]
            encoded = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(device)

            outputs = model(**encoded, labels=encoded["input_ids"])
            prompt_tokens = len(tokenizer(prompt, truncation=True, max_length=1536)["input_ids"])
            response_tokens = encoded["input_ids"].shape[1] - prompt_tokens

            if response_tokens > 0:
                scaled_loss = (-adv.to(device) * outputs.loss) / 4
                scaled_loss.backward()
                accumulated_loss += scaled_loss.item()
                n_contributing += 1

            del outputs, encoded, scaled_loss
            torch.cuda.empty_cache()

            if n_contributing >= 4:
                break

        if n_contributing >= 4:
            break

    return torch.tensor(accumulated_loss * 4 if n_contributing > 0 else 0.0)


def save_checkpoint(model, tokenizer, optimizer, iteration: int, stats: dict):
    """Save checkpoint locally and to S3."""
    ckpt_path = Path(CHECKPOINT_DIR) / f"iter-{iteration:04d}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")

    with open(ckpt_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Upload to S3 (async, don't block training)
    try:
        s3 = boto3.client("s3")
        for file in ckpt_path.rglob("*"):
            if file.is_file():
                key = f"{S3_PREFIX}/iter-{iteration:04d}/{file.relative_to(ckpt_path)}"
                s3.upload_file(str(file), S3_BUCKET, key)
        print(f"  Checkpoint saved: {ckpt_path} + S3")
    except Exception as e:
        print(f"  Checkpoint saved locally only (S3 error: {e})")


def train():
    print("=== RL Conductor Training (Phase 1) ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Config: {NUM_ITERATIONS} iters, batch {BATCH_SIZE} ({QUESTIONS_PER_BATCH}q × {ROLLOUTS_PER_QUESTION}r), lr {LR}")
    print(f"Generation batch size: {GENERATION_BATCH_SIZE}")
    print()

    device = torch.device("cuda:0")

    # Load model on single GPU (7B bf16 = ~14GB, fits easily in 80GB A100)
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.gradient_checkpointing_enable()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS)

    # Data
    train_data = load_training_data(DATA_PATH)
    print(f"Loaded {len(train_data)} training problems")

    # Worker pool
    worker_pool = WorkerPool(phase=PHASE)
    print(f"Worker pool: {worker_pool.num_workers} workers (Phase {PHASE})")
    print()

    # Log file — resume from last completed iteration
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    log_path = Path(LOG_DIR) / "training.jsonl"
    start_iteration = 0
    if log_path.exists():
        with open(log_path) as f:
            start_iteration = sum(1 for _ in f)
        if start_iteration > 0:
            print(f"Resuming from iteration {start_iteration}")
    log_file = open(log_path, "a")

    # Training loop
    for iteration in range(start_iteration, NUM_ITERATIONS):
        iter_start = time.time()

        # Sample questions
        questions = random.sample(train_data, min(QUESTIONS_PER_BATCH, len(train_data)))

        # Build prompts (same prompt repeated for each rollout)
        prompts = []
        for q in questions:
            prompt = format_prompt(q["question"], worker_pool.num_workers)
            prompts.extend([prompt] * ROLLOUTS_PER_QUESTION)

        # === GENERATE (batched, no_grad) ===
        gen_start = time.time()
        model.eval()
        generated_texts = batch_generate(model, tokenizer, prompts, batch_size=GENERATION_BATCH_SIZE)
        gen_time = time.time() - gen_start

        # === EXECUTE WORKFLOWS + COMPUTE REWARDS (thread pool) ===
        exec_start = time.time()
        rollout_results = execute_rollouts_sync(generated_texts, questions, ROLLOUTS_PER_QUESTION, worker_pool)
        exec_time = time.time() - exec_start

        # === COMPUTE GRPO LOSS + BACKPROP (gradient accumulated inside) ===
        train_start = time.time()
        model.train()
        optimizer.zero_grad()

        loss = compute_grpo_loss(
            model, tokenizer, prompts, rollout_results,
            questions, ROLLOUTS_PER_QUESTION, device,
        )

        if loss.item() != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        train_time = time.time() - train_start

        # === STATS ===
        rewards = [r["reward"] for r in rollout_results]
        format_failures = sum(1 for r in rollout_results if r["reward"] == 0.0)
        correct = sum(1 for r in rollout_results if r["reward"] == 1.0)
        parseable = sum(1 for r in rollout_results if r["reward"] >= 0.5)

        depths = []
        worker_usage = {}
        for r in rollout_results:
            parsed = parse_conductor_output(r["response"])
            if parsed:
                depths.append(len(parsed["subtasks"]))
                for mid in parsed["model_id"]:
                    if not isinstance(mid, int): continue
                    worker_usage[mid] = worker_usage.get(mid, 0) + 1

        iter_stats = {
            "iteration": iteration,
            "loss": loss.item() if hasattr(loss, 'item') else 0.0,
            "mean_reward": sum(rewards) / len(rewards),
            "reward_std": (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)) ** 0.5,
            "format_failure_rate": format_failures / len(rollout_results),
            "parseable_rate": parseable / len(rollout_results),
            "correct_rate": correct / len(rollout_results),
            "mean_depth": sum(depths) / max(len(depths), 1),
            "worker_usage": {str(k): v for k, v in worker_usage.items()},
            "lr": scheduler.get_last_lr()[0],
            "gen_time_s": gen_time,
            "exec_time_s": exec_time,
            "train_time_s": train_time,
            "wall_time_s": time.time() - iter_start,
        }

        log_file.write(json.dumps(iter_stats) + "\n")
        log_file.flush()

        print(f"[Iter {iteration:03d}] loss={iter_stats['loss']:.4f} "
              f"reward={iter_stats['mean_reward']:.3f}±{iter_stats['reward_std']:.3f} "
              f"fmt_fail={iter_stats['format_failure_rate']:.0%} "
              f"correct={iter_stats['correct_rate']:.0%} "
              f"depth={iter_stats['mean_depth']:.1f} "
              f"[gen={gen_time:.0f}s exec={exec_time:.0f}s train={train_time:.0f}s total={iter_stats['wall_time_s']:.0f}s]")

        if worker_usage:
            usage_str = " ".join(
                f"{worker_pool.workers[k].name}:{v}" for k, v in sorted(worker_usage.items())
                if k in worker_pool.workers
            )
            print(f"         workers: {usage_str}")

        # Checkpoint
        if (iteration + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, tokenizer, optimizer, iteration, iter_stats)

    # Final checkpoint
    save_checkpoint(model, tokenizer, optimizer, NUM_ITERATIONS - 1, iter_stats)
    log_file.close()

    print("\n=== Training complete ===")
    print(f"Final: reward={iter_stats['mean_reward']:.3f}, correct={iter_stats['correct_rate']:.1%}")
    # Auto-sync to S3 after training completes
    import subprocess
    print("
[Auto-sync] Uploading results to S3...")
    subprocess.run(["/bin/bash", "/opt/dlami/nvme/rl-conductor/sync_to_s3.sh"], check=False)
    print(f"Worker usage: {json.dumps(worker_pool.get_stats_summary(), indent=2)}")


if __name__ == "__main__":
    train()
