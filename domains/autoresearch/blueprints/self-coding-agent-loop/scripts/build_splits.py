#!/usr/bin/env python3
"""Build instance-level rolling-rounds data splits.

Reads: trajectories.parquet (Nebius OpenHands 67K trajectories, 6,306 unique instances)
Writes:
  splits_manifest.json   — SHA-pinned instance_id lists for every split
  final_stress_500.jsonl — 500 instances, write-locked through all 5 rounds + Phase 1
  drift_audit_300.jsonl  — 300 instances, re-eval'd every round
  round_1_control.jsonl .. round_5_control.jsonl — 300 each, becomes training next round
  round_1_train.jsonl    .. round_5_train.jsonl  — 800 each, trajectories filtered to resolved=1
  v1b_bootstrap_200.jsonl — 100 pos + 100 neg, subset of round_1_train (no leak)

Invariants enforced (asserted before write):
  - final_stress ∩ drift_audit ∩ all rounds_control = ∅
  - round_N_control ∩ round_N_train (at round N start) = ∅
  - v1b_bootstrap_200 ⊂ round_1_train
  - 500 + 300 + 5×300 + 5×800 = 6300 ≤ 6306 unique instances
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd


SPLITS = {
    "final_stress_500": 500,
    "drift_audit_300": 300,
    "round_1_control": 300, "round_2_control": 300, "round_3_control": 300,
    "round_4_control": 300, "round_5_control": 300,
    "round_1_train": 800, "round_2_train": 800, "round_3_train": 800,
    "round_4_train": 800, "round_5_train": 800,
}


def sha_of_ids(ids: list[str]) -> str:
    canon = "\n".join(sorted(ids)).encode()
    return hashlib.sha256(canon).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="Path to trajectories.parquet")
    p.add_argument("--out-dir", required=True, help="Where to write split jsonl files + manifest")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[build_splits] loading {args.parquet}")
    df = pd.read_parquet(args.parquet, columns=[
        "trajectory_id", "instance_id", "repo", "model_patch",
        "resolved", "gen_tests_correct", "pred_passes_gen_tests",
        "trajectory",  # needed for SFT training input
    ])
    print(f"  trajectories: {len(df)}")
    unique_instances = sorted(df["instance_id"].unique())
    print(f"  unique instances: {len(unique_instances)}")

    required = sum(SPLITS.values())
    assert len(unique_instances) >= required, (
        f"need {required} instances, have {len(unique_instances)}"
    )

    rng = random.Random(args.seed)
    shuffled = list(unique_instances)
    rng.shuffle(shuffled)

    # Assign in declared order
    cursor = 0
    split_to_ids = {}
    for name, n in SPLITS.items():
        split_to_ids[name] = shuffled[cursor:cursor + n]
        cursor += n
    slack = shuffled[cursor:]
    print(f"  slack (unused instances): {len(slack)}")

    # Sanity — all disjoint
    all_ids = set()
    for name, ids in split_to_ids.items():
        s = set(ids)
        overlap = all_ids & s
        assert not overlap, f"split {name} overlaps with prior splits: {list(overlap)[:3]}"
        all_ids |= s
    print(f"  disjointness verified across {len(all_ids)} assigned instances")

    # v1b_bootstrap_200: 100 pos + 100 neg from round_1_train
    r1_train_ids = set(split_to_ids["round_1_train"])
    r1_df = df[df["instance_id"].isin(r1_train_ids)]
    any_resolved = r1_df.groupby("instance_id")["resolved"].max()
    r1_pos = [i for i, v in any_resolved.items() if v == 1]
    r1_neg = [i for i, v in any_resolved.items() if v == 0]
    print(f"  round_1_train: {len(r1_pos)} pos instances, {len(r1_neg)} neg instances")
    rng.shuffle(r1_pos); rng.shuffle(r1_neg)
    bootstrap_pos = r1_pos[:100]
    bootstrap_neg = r1_neg[: min(100, len(r1_neg))]
    split_to_ids["v1b_bootstrap_200"] = bootstrap_pos + bootstrap_neg
    print(f"  v1b_bootstrap_200: {len(bootstrap_pos)} pos + {len(bootstrap_neg)} neg = {len(split_to_ids['v1b_bootstrap_200'])}")

    # Emit jsonl for each split
    for name, ids in split_to_ids.items():
        id_set = set(ids)
        if name.endswith("_train"):
            # Training: filter to gold-passing trajectories only
            sub = df[(df["instance_id"].isin(id_set)) & (df["resolved"] == 1) &
                     (df["model_patch"].str.len() > 0)]
        elif name == "v1b_bootstrap_200":
            # Pick one trajectory per instance, prefer resolved=1 for pos, resolved=0 for neg
            rows = []
            for inst in ids:
                inst_df = df[df["instance_id"] == inst]
                if inst in set(bootstrap_pos):
                    inst_df = inst_df[inst_df["resolved"] == 1].sort_values("trajectory_id")
                else:
                    inst_df = inst_df[inst_df["resolved"] == 0].sort_values("trajectory_id")
                if not inst_df.empty:
                    rows.append(inst_df.iloc[0])
            sub = pd.DataFrame(rows)
        else:
            # Eval splits: one canonical row per instance (smallest trajectory_id for determinism)
            sub = df[df["instance_id"].isin(id_set)].sort_values(["instance_id", "trajectory_id"])
            sub = sub.drop_duplicates(subset=["instance_id"], keep="first")

        path = out / f"{name}.jsonl"
        sub.to_json(path, orient="records", lines=True)
        n_traj = len(sub)
        n_inst = sub["instance_id"].nunique() if len(sub) else 0
        size_mb = path.stat().st_size / 1e6
        print(f"  wrote {name}: {n_inst} instances, {n_traj} trajectories, {size_mb:.1f} MB")

    # Invariant checks
    for a, b in [
        ("final_stress_500", "drift_audit_300"),
        ("final_stress_500", "round_1_train"),
        ("final_stress_500", "round_5_train"),
        ("drift_audit_300", "round_1_train"),
        ("round_1_control", "round_1_train"),
        ("round_2_control", "round_1_train"),
        ("round_5_control", "round_1_train"),
    ]:
        overlap = set(split_to_ids[a]) & set(split_to_ids[b])
        assert not overlap, f"INVARIANT BROKEN: {a} ∩ {b} = {list(overlap)[:5]}"
    # v1b_bootstrap ⊂ round_1_train
    boot_set = set(split_to_ids["v1b_bootstrap_200"])
    r1_train_set = set(split_to_ids["round_1_train"])
    assert boot_set <= r1_train_set, (
        f"INVARIANT BROKEN: v1b_bootstrap ⊄ round_1_train ({len(boot_set - r1_train_set)} leak)"
    )
    print("[build_splits] all invariants verified")

    # Manifest
    manifest = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "seed": args.seed,
        "source_parquet": args.parquet,
        "source_rows": len(df),
        "source_unique_instances": len(unique_instances),
        "slack_unused_instances": len(slack),
        "splits": {
            name: {
                "n_instances": len(ids),
                "sha256_of_sorted_instance_ids": sha_of_ids(ids),
            }
            for name, ids in split_to_ids.items()
        },
        "invariants_verified": [
            "final_stress ∩ drift_audit = ∅",
            "final_stress ∩ any round_N_train = ∅",
            "drift_audit ∩ any round_N_train = ∅",
            "round_N_control ∩ round_N_train = ∅ for all N",
            "v1b_bootstrap ⊂ round_1_train",
        ],
        "usage_rules": {
            "final_stress_500": "read ONLY at end-of-Phase-1 and in Phase 2. NEVER during training.",
            "drift_audit_300": "read-only every round; re-eval'd under every Gen-N for drift trajectory.",
            "round_N_control": "held-out ONLY during round N. After round N completes, becomes training data for round N+1.",
            "round_N_train": "training data for round N. Cumulative across rounds.",
            "v1b_bootstrap_200": "for V1b_bootstrap RF recalibration. Week 1 only.",
        },
    }
    manifest_path = out / "splits_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build_splits] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
