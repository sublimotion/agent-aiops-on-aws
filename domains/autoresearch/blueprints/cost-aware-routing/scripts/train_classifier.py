"""Fine-tune ModernBERT-base on category labels.

Categories: math, code, factual, reasoning, open-domain (5 classes).
Data: 480-question augmented baseline. Split 80/20 train/eval.
Targets: ≥85% accuracy on eval.

Runs locally on Apple Silicon MPS (~149M params fit in unified memory).
~10-15 minutes wall on M-series.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Stop the OpenMP duplicate-init crash on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Line-buffered stdout so background runs show progress
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


CATEGORIES = ["math", "code", "factual", "reasoning", "open-domain"]
CAT_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}


def load_data(path: str, seed: int = 17, eval_frac: float = 0.20):
    """Load augmented_baseline_500q.jsonl, split train/eval."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "question": r["question"],
                "label": CAT_TO_ID[r["category"]],
                "category": r["category"],
                "id": r["id"],
            })
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_eval = int(len(rows) * eval_frac)
    return rows[n_eval:], rows[:n_eval]


class QDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int = 512):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(
            r["question"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(r["label"], dtype=torch.long),
        }


def evaluate(model, loader, device):
    model.eval()
    n = 0
    correct = 0
    per_cat_correct = [0] * len(CATEGORIES)
    per_cat_total = [0] * len(CATEGORIES)
    confusion = [[0] * len(CATEGORIES) for _ in CATEGORIES]
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            preds = out.logits.argmax(-1)
            n += labels.size(0)
            correct += (preds == labels).sum().item()
            for true_l, pred_l in zip(labels.tolist(), preds.tolist()):
                per_cat_total[true_l] += 1
                if pred_l == true_l:
                    per_cat_correct[true_l] += 1
                confusion[true_l][pred_l] += 1
    return {
        "accuracy": correct / max(n, 1),
        "n": n,
        "per_category_accuracy": {
            CATEGORIES[i]: round(per_cat_correct[i] / max(per_cat_total[i], 1), 4)
            for i in range(len(CATEGORIES))
        },
        "confusion_matrix": confusion,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_500q.jsonl")
    ap.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    ap.add_argument("--out-dir", default="domains/autoresearch/blueprints/cost-aware-routing/artifacts/classifier")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="auto", help="auto, mps, cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    train_rows, eval_rows = load_data(args.data, seed=args.seed)
    print(f"Train: {len(train_rows)}, Eval: {len(eval_rows)}")

    # Class balance check
    from collections import Counter
    train_balance = Counter(CATEGORIES[r["label"]] for r in train_rows)
    eval_balance = Counter(CATEGORIES[r["label"]] for r in eval_rows)
    print(f"Train balance: {dict(train_balance)}")
    print(f"Eval balance:  {dict(eval_balance)}")

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"\nLoading {args.model_name}...")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(CATEGORIES),
    ).to(device)

    train_ds = QDataset(train_rows, tok)
    eval_ds = QDataset(eval_rows, tok)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    n_steps = args.epochs * len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_frac * n_steps),
        num_training_steps=n_steps,
    )

    print(f"\nTraining for {args.epochs} epochs ({n_steps} total steps)...")
    import time
    t_start = time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            if step == 0:
                print(f"    epoch {epoch+1} step 1 loss={loss.item():.4f} t={time.time()-t_start:.0f}s")
        avg_loss = epoch_loss / len(train_loader)

        eval_metrics = evaluate(model, eval_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "eval_accuracy": round(eval_metrics["accuracy"], 4),
            "per_category": eval_metrics["per_category_accuracy"],
        })
        print(f"  epoch {epoch+1}/{args.epochs}  train_loss={avg_loss:.4f}  "
              f"eval_acc={eval_metrics['accuracy']:.1%}  "
              f"per_cat={eval_metrics['per_category_accuracy']}")

    # Save model
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    final_eval = evaluate(model, eval_loader, device)
    summary = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "categories": CATEGORIES,
        "history": history,
        "final_eval": final_eval,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== FINAL ===")
    print(f"Eval accuracy: {final_eval['accuracy']:.1%}")
    print(f"Per-category: {final_eval['per_category_accuracy']}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
