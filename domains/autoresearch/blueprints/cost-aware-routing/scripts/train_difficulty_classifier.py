"""Train a per-question difficulty classifier on baseline n_workers_correct.

Two variants:
  --mode binary    -> {easy: n_correct >= 6, hard: n_correct < 6}
  --mode 3class    -> {easy: >=7, medium: 4-6, hard: <4}

Same ModernBERT-base, same train/eval split (seed=17, 80/20). Outputs to
artifacts/difficulty_classifier_<mode>/.

The labels come from the baseline rollouts in:
  results/baselines/always_x_math500.json
  results/baselines/always_x_aime25_n30.json
  results/baselines/always_x_wildchat_n50.json
  results/baselines/always_x_augmented.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def short_id(category: str, text: str) -> str:
    return f"{category}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def load_questions_with_difficulty(mode: str) -> list[dict]:
    """Load all questions + their difficulty label.

    Returns list of {id, question, category, n_correct, label}.
    """
    src_to_cat = {"math500": "math", "aime25": "math", "wildchat": "open-domain"}
    by_qid: dict[str, dict] = {}

    # Existing 130-q baselines
    for path, cor_key in [
        ("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_math500.json", "is_correct"),
        ("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_aime25_n30.json", "is_correct"),
        ("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_wildchat_n50.json", "acceptable"),
    ]:
        d = json.load(open(path))
        source = "math500" if "math500" in path else ("aime25" if "aime25" in path else "wildchat")
        cat = src_to_cat[source]
        for r in d["rollouts"]:
            qid = short_id(cat, r["question"])
            entry = by_qid.setdefault(qid, {
                "id": qid, "question": r["question"], "category": cat, "n_correct": 0,
            })
            entry["n_correct"] += int(bool(r[cor_key]))

    # Augmented
    aug = json.load(open("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_augmented.json"))
    # Need full question text — re-load from data file
    aug_data = {}
    with open("domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_500q.jsonl") as f:
        for line in f:
            r = json.loads(line)
            aug_data[r["id"]] = r
    for r in aug["rollouts"]:
        qid = r["id"]
        full_q = aug_data.get(qid, {}).get("question", r.get("question", ""))
        entry = by_qid.setdefault(qid, {
            "id": qid, "question": full_q, "category": r["category"], "n_correct": 0,
        })
        entry["n_correct"] += int(bool(r["is_correct"]))

    # Apply label
    out = []
    for q in by_qid.values():
        n = q["n_correct"]
        if mode == "binary":
            q["label_str"] = "hard" if n < 6 else "easy"
        elif mode == "3class":
            q["label_str"] = "hard" if n < 4 else ("medium" if n < 7 else "easy")
        else:
            raise ValueError(f"Unknown mode: {mode}")
        out.append(q)
    return out


class QDataset(Dataset):
    def __init__(self, rows, tokenizer, label_to_id, max_length=512):
        self.rows = rows
        self.tok = tokenizer
        self.l2i = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(
            r["question"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.l2i[r["label_str"]], dtype=torch.long),
        }


def evaluate(model, loader, device, labels: list[str]):
    from collections import defaultdict
    model.eval()
    n = 0; correct = 0
    per_label_total = defaultdict(int)
    per_label_correct = defaultdict(int)
    confusion = [[0] * len(labels) for _ in labels]
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            preds = out.logits.argmax(-1)
            n += y.size(0)
            correct += (preds == y).sum().item()
            for true_l, pred_l in zip(y.tolist(), preds.tolist()):
                per_label_total[labels[true_l]] += 1
                if pred_l == true_l:
                    per_label_correct[labels[true_l]] += 1
                confusion[true_l][pred_l] += 1
    return {
        "accuracy": correct / max(n, 1),
        "n": n,
        "per_label_accuracy": {l: round(per_label_correct[l] / max(per_label_total[l], 1), 4) for l in labels},
        "confusion": confusion,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "3class"], required=True)
    ap.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    if args.mode == "binary":
        labels = ["easy", "hard"]
    else:
        labels = ["easy", "medium", "hard"]
    l2i = {l: i for i, l in enumerate(labels)}

    rows = load_questions_with_difficulty(args.mode)
    print(f"Loaded {len(rows)} questions with difficulty labels (mode={args.mode})")
    from collections import Counter
    print(f"Label distribution: {dict(Counter(r['label_str'] for r in rows))}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_eval = int(len(rows) * 0.20)
    train_rows = rows[n_eval:]
    eval_rows = rows[:n_eval]
    print(f"Train: {len(train_rows)}  Eval: {len(eval_rows)}")
    print(f"Eval label balance: {dict(Counter(r['label_str'] for r in eval_rows))}")

    device = torch.device(args.device)
    print(f"Loading {args.model_name}...")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels),
    ).to(device)

    train_ds = QDataset(train_rows, tok, l2i)
    eval_ds = QDataset(eval_rows, tok, l2i)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    n_steps = args.epochs * len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_frac * n_steps), num_training_steps=n_steps,
    )

    print(f"\nTraining {args.epochs} epochs ({n_steps} steps)...")
    import time
    t_start = time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=y)
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += out.loss.item()
            if step == 0:
                print(f"    epoch {epoch+1} step 1 loss={out.loss.item():.4f} t={time.time()-t_start:.0f}s")
        avg_loss = epoch_loss / len(train_loader)
        eval_metrics = evaluate(model, eval_loader, device, labels)
        history.append({
            "epoch": epoch, "train_loss": round(avg_loss, 4),
            "eval_accuracy": round(eval_metrics["accuracy"], 4),
            "per_label": eval_metrics["per_label_accuracy"],
        })
        print(f"  epoch {epoch+1}/{args.epochs}  train_loss={avg_loss:.4f}  "
              f"eval_acc={eval_metrics['accuracy']:.1%}  "
              f"per_label={eval_metrics['per_label_accuracy']}")

    out_dir = Path(f"domains/autoresearch/blueprints/cost-aware-routing/artifacts/difficulty_classifier_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    final_eval = evaluate(model, eval_loader, device, labels)
    summary = {
        "mode": args.mode, "labels": labels, "model_name": args.model_name,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "n_train": len(train_rows), "n_eval": len(eval_rows),
        "history": history, "final_eval": final_eval,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_dir}")
    print(f"Final eval: {final_eval['accuracy']:.1%}  per-label: {final_eval['per_label_accuracy']}")


if __name__ == "__main__":
    main()
