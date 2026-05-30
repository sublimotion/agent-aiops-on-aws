"""Build cheapest-correct-worker labels from the merged 5K + 480 corpus.

Reads:
  results/baselines/always_x_math500.json
  results/baselines/always_x_aime25_n30.json
  results/baselines/always_x_wildchat_n50.json
  results/baselines/always_x_augmented.json     (480q baseline)
  results/baselines/always_x_5000q.json         (3,587 new questions)
  data/augmented_baseline_500q.jsonl
  data/augmented_baseline_5000q.jsonl

Output:
  data/cheapest_correct_labels_5k.jsonl  with rows {id, question, category, label}
  where label = cheapest worker (0..8) that's correct, fallback ord_8 if all wrong.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter, defaultdict


def short_id(category: str, text: str) -> str:
    return f"{category}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="domains/autoresearch/blueprints/cost-aware-routing/data/cheapest_correct_labels_5k.jsonl")
    args = ap.parse_args()

    src_to_cat = {"math500": "math", "aime25": "math", "wildchat": "open-domain"}
    by_qid: dict[str, dict] = {}

    # Existing 130-q (legacy)
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
            e = by_qid.setdefault(qid, {"question": r["question"], "category": cat, "workers": {}})
            e["workers"][r["ord"]] = bool(r[cor_key])

    # Augmented 480q (the original phase-1 augmentation)
    aug = json.load(open("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_augmented.json"))
    aug_data = {}
    for line in open("domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_500q.jsonl"):
        r = json.loads(line)
        aug_data[r["id"]] = r["question"]
    for r in aug["rollouts"]:
        qid = r["id"]
        text = aug_data.get(qid, r.get("question", ""))
        e = by_qid.setdefault(qid, {"question": text, "category": r["category"], "workers": {}})
        e["workers"][r["ord"]] = bool(r["is_correct"])

    # 5K augmentation
    p_5k = "domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_5000q.json"
    if pathlib.Path(p_5k).exists():
        big = json.load(open(p_5k))
        big_data = {}
        for line in open("domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_5000q.jsonl"):
            r = json.loads(line)
            big_data[r["id"]] = r["question"]
        for r in big["rollouts"]:
            qid = r["id"]
            text = big_data.get(qid, r.get("question", ""))
            e = by_qid.setdefault(qid, {"question": text, "category": r["category"], "workers": {}})
            e["workers"][r["ord"]] = bool(r["is_correct"])
    else:
        print(f"WARN: 5K baseline not yet at {p_5k} — output will only include 480q-scale data.")

    # Compute cheapest-correct label
    rows = []
    for qid, e in by_qid.items():
        if not e["workers"]:
            continue
        cheapest = None
        for w in range(9):
            if e["workers"].get(w, False):
                cheapest = w
                break
        if cheapest is None:
            cheapest = 8
        rows.append({
            "id": qid,
            "question": e["question"],
            "category": e["category"],
            "label": cheapest,
        })

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")
    label_dist = Counter(r["label"] for r in rows)
    NAMES = ["gemma","gpt-oss","qwen3-32b","qwen-coder","mistral","deepseek","haiku","sonnet","opus"]
    print(f"Label distribution:")
    for w in sorted(label_dist):
        print(f"  ord_{w} {NAMES[w]:12s}: {label_dist[w]:>4d} ({label_dist[w]*100//len(rows)}%)")
    cat_dist = Counter(r["category"] for r in rows)
    print(f"By category: {dict(cat_dist)}")


if __name__ == "__main__":
    main()
