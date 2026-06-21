#!/usr/bin/env python3
"""RQ2 calibration check: run the adversarial ensemble with a NON-Claude
verifier (default nova-pro) on a subset, to test whether the adversarial lift
is Claude-specific (coding-domain T4 finding) or domain-specific.

Same adversarial rubric + 4-call ensemble + 4/4-unanimous rule as
run_verifiers.py, but with --model nova-pro. Output is scored by analyze.py.
"""
import argparse
import json
import sys

from finqa_common import build_context
from run_verifiers import run_adversarial, run_confirmatory, load_rubric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="nova-pro")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--adv-rubric",
                    default="../skills/finqa-verifier/versions/adversarial.md")
    ap.add_argument("--conf-rubric",
                    default="../skills/finqa-verifier/versions/confirmatory.md")
    args = ap.parse_args()

    adv_rubric = load_rubric(args.adv_rubric)
    conf_rubric = load_rubric(args.conf_rubric)
    data = json.load(open(args.data))
    by_id = {ex.get("id", f"idx_{i}"): ex for i, ex in enumerate(data)}
    recs = [json.loads(l) for l in open(args.answers) if l.strip()][:args.limit]

    with open(args.out, "w") as out:
        for i, rec in enumerate(recs):
            ex = by_id.get(rec["id"])
            if ex is None:
                continue
            ctx = build_context(ex)
            try:
                conf = run_confirmatory(conf_rubric, ctx, rec, args.model)
                adv = run_adversarial(adv_rubric, ctx, rec, args.model)
            except Exception as e:
                print(f"[{i}] ERROR: {e}", file=sys.stderr)
                continue
            merged = {**rec, **conf, **adv}
            out.write(json.dumps(merged) + "\n")
            out.flush()
            print(f"[{i+1}/{len(recs)}] match={rec['match']} "
                  f"conf={conf['conf_verdict']} adv={adv['adv_verdicts']} "
                  f"confident={adv['adv_confident']}")
    print(f"\ncross-verifier ({args.model}) done, n={len(recs)}")


if __name__ == "__main__":
    main()
