"""Train ModernBERT-base to predict the cheapest correct worker.

Label = cheapest worker (ord 0..8) that gets the question right; if no
worker is correct, label as ord_8 (Opus, the strongest fallback).

This is the most direct cost-aware routing target: at inference, the
model's prediction is the worker to route to. No intermediate "category"
or "difficulty" label.

Class imbalance is heavy (66% Gemma, 7% Opus, single-digit elsewhere).
We train without re-weighting first; if Opus recall is too low, retry
with class weights.
"""
from __future__ import annotations

import argparse
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


WORKER_NAMES = ["gemma", "gpt-oss", "qwen3-32b", "qwen-coder", "mistral",
                "deepseek", "haiku", "sonnet", "opus"]


class QDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=512):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(r["question"], truncation=True, max_length=self.max_length,
                       padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(r["label"]), dtype=torch.long),
        }


def evaluate(model, loader, device, n_classes):
    from collections import defaultdict
    model.eval()
    n = 0; correct = 0
    per_label_total = defaultdict(int)
    per_label_correct = defaultdict(int)
    confusion = [[0] * n_classes for _ in range(n_classes)]
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
                per_label_total[true_l] += 1
                if pred_l == true_l:
                    per_label_correct[true_l] += 1
                confusion[true_l][pred_l] += 1
    return {
        "accuracy": correct / max(n, 1),
        "n": n,
        "per_label_recall": {WORKER_NAMES[i]: round(per_label_correct[i] / max(per_label_total[i], 1), 4)
                              for i in range(n_classes) if per_label_total[i] > 0},
        "confusion": confusion,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="domains/autoresearch/blueprints/cost-aware-routing/data/cheapest_correct_labels.jsonl")
    ap.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--class-weights", action="store_true",
                    help="Apply inverse-frequency weights to the loss.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    rows = []
    with open(args.data) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} (question, cheapest-correct-worker) labels")
    from collections import Counter
    label_counts = Counter(r["label"] for r in rows)
    print(f"Label distribution: {dict(sorted(label_counts.items()))}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_eval = int(len(rows) * 0.20)
    train_rows = rows[n_eval:]
    eval_rows = rows[:n_eval]
    print(f"Train: {len(train_rows)}  Eval: {len(eval_rows)}")
    print(f"Eval label balance: {dict(Counter(r['label'] for r in eval_rows))}")

    device = torch.device(args.device)

    print(f"Loading {args.model_name}...")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=9,
    ).to(device)

    train_ds = QDataset(train_rows, tok)
    eval_ds = QDataset(eval_rows, tok)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    n_steps = args.epochs * len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_frac * n_steps), num_training_steps=n_steps,
    )

    # Optional class weights
    loss_weights = None
    if args.class_weights:
        # Inverse-frequency weights, capped to avoid extremes
        train_counts = Counter(r["label"] for r in train_rows)
        weights = torch.zeros(9)
        for c in range(9):
            weights[c] = (len(train_rows) / max(9 * train_counts.get(c, 1), 1))
        weights = weights.clamp(min=0.5, max=10.0).to(device)
        loss_weights = weights
        print(f"Class weights: {weights.cpu().tolist()}")

    print(f"\nTraining {args.epochs} epochs ({n_steps} steps)...")
    import time
    t0 = time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            if loss_weights is not None:
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = torch.nn.functional.cross_entropy(logits, y, weight=loss_weights)
            else:
                loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            if step == 0:
                print(f"    epoch {epoch+1} step 1 loss={loss.item():.4f} t={time.time()-t0:.0f}s")
        avg_loss = epoch_loss / len(train_loader)
        eval_metrics = evaluate(model, eval_loader, device, 9)
        history.append({"epoch": epoch, "train_loss": round(avg_loss, 4),
                        "eval_accuracy": round(eval_metrics["accuracy"], 4),
                        "per_label_recall": eval_metrics["per_label_recall"]})
        print(f"  epoch {epoch+1}/{args.epochs}  train_loss={avg_loss:.4f}  "
              f"eval_acc={eval_metrics['accuracy']:.1%}  "
              f"per_label={eval_metrics['per_label_recall']}")

    out_dir = Path("domains/autoresearch/blueprints/cost-aware-routing/artifacts/cheapest_correct_classifier"
                   + ("_weighted" if args.class_weights else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    final_eval = evaluate(model, eval_loader, device, 9)
    summary = {
        "model_name": args.model_name, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr,
        "n_train": len(train_rows), "n_eval": len(eval_rows),
        "class_weights_used": args.class_weights,
        "history": history, "final_eval": final_eval,
        "worker_names": WORKER_NAMES,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_dir}")
    print(f"Final eval: {final_eval['accuracy']:.1%}  per-label: {final_eval['per_label_recall']}")


if __name__ == "__main__":
    main()
