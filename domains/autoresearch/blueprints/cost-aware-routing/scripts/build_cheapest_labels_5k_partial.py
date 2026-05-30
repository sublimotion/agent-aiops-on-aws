"""Build cheapest-correct labels using only the workers that completed in the 5K run.

Workers 5-8 (deepseek/haiku/sonnet/opus) hit network errors mid-run. Rather than
re-running them, we restrict the label space to ord 0..4 (gemma, gpt-oss,
qwen3-32b, qwen-coder-480b, mistral-large-3). For questions where none of those
workers were correct we fall back to ord 4 (mistral-large-3, the strongest
available) and tag the row as `fallback=True` so training can mask or down-weight.

Output:
  data/cheapest_correct_labels_5k_partial.jsonl
  rows: {id, question, category, label, fallback}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter


MAX_ORD = 4  # only ord 0..4 have valid rollouts on the 5K run
NAMES = ["gemma", "gpt-oss", "qwen3-32b", "qwen-coder", "mistral",
         "deepseek", "haiku", "sonnet", "opus"]


def short_id(category: str, text: str) -> str:
    return f"{category}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="domains/autoresearch/blueprints/cost-aware-routing/data/cheapest_correct_labels_5k_partial.jsonl",
    )
    args = ap.parse_args()

    by_qid: dict[str, dict] = {}

    # 130-q legacy baselines (full 9 workers)
    src_to_cat = {"math500": "math", "aime25": "math", "wildchat": "open-domain"}
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
            if r["ord"] <= MAX_ORD:
                e["workers"][r["ord"]] = bool(r[cor_key])

    # 480-q augmented (full 9 workers)
    aug = json.load(open("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_augmented.json"))
    aug_data = {}
    for line in open("domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_500q.jsonl"):
        r = json.loads(line)
        aug_data[r["id"]] = r["question"]
    for r in aug["rollouts"]:
        qid = r["id"]
        text = aug_data.get(qid, r.get("question", ""))
        e = by_qid.setdefault(qid, {"question": text, "category": r["category"], "workers": {}})
        if r["ord"] <= MAX_ORD:
            e["workers"][r["ord"]] = bool(r["is_correct"])

    # 5K augmentation (only ord 0..4 valid)
    big = json.load(open("domains/autoresearch/blueprints/cost-aware-routing/results/baselines/always_x_5000q.json"))
    big_data = {}
    for line in open("domains/autoresearch/blueprints/cost-aware-routing/data/augmented_baseline_5000q.jsonl"):
        r = json.loads(line)
        big_data[r["id"]] = r["question"]
    for r in big["rollouts"]:
        if r["ord"] > MAX_ORD:
            continue
        qid = r["id"]
        text = big_data.get(qid, r.get("question", ""))
        e = by_qid.setdefault(qid, {"question": text, "category": r["category"], "workers": {}})
        e["workers"][r["ord"]] = bool(r["is_correct"])

    rows = []
    for qid, e in by_qid.items():
        if not e["workers"]:
            continue
        cheapest = None
        for w in range(MAX_ORD + 1):
            if e["workers"].get(w, False):
                cheapest = w
                break
        fallback = cheapest is None
        if fallback:
            cheapest = MAX_ORD  # mistral-large-3 as the strongest available
        rows.append({
            "id": qid,
            "question": e["question"],
            "category": e["category"],
            "label": cheapest,
            "fallback": fallback,
        })

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")
    label_dist = Counter(r["label"] for r in rows)
    fb = sum(1 for r in rows if r["fallback"])
    print(f"Label distribution (search space ord 0..{MAX_ORD}):")
    for w in sorted(label_dist):
        print(f"  ord_{w} {NAMES[w]:12s}: {label_dist[w]:>4d} ({label_dist[w]*100//len(rows)}%)")
    print(f"Fallback (no worker 0..{MAX_ORD} correct): {fb} ({fb*100//len(rows)}%)")
    cat_dist = Counter(r["category"] for r in rows)
    print(f"By category: {dict(cat_dist)}")


if __name__ == "__main__":
    main()
