"""
Pre-flight: Haiku vs Sonnet agreement on MATH500 judging.

Spec line 344 flags that the cited 96% Haiku-Sonnet agreement is from code
patches (verifier-reward T4), not math. Phase 1 reward signal is `is_correct`
judged by Haiku, so we need to validate calibration on math BEFORE training.

Input: 50 (question, gold, response) tuples from rl-conductor v4 iter-074.
Output: per-item Haiku verdict, Sonnet verdict, parser verdict;
        agreement matrices and confusion stats.

Cost: ~50 questions x 2 judges x ~$0.003/q (Haiku) + ~$0.012/q (Sonnet) ~= $0.75.
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import sys
import time

import boto3

REGION = "us-west-2"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET = "us.anthropic.claude-sonnet-4-6"

JUDGE_PROMPT = """You are grading a math problem.

Question:
{question}

Gold answer:
{gold}

Student's final answer (extracted): {extracted}

Student's reasoning (tail of full response):
{response_tail}

Decide whether the student's final answer is mathematically equivalent to the gold answer.
Equivalent means: same value after simplification, regardless of formatting (e.g. `\\frac{{1}}{{2}}` and `0.5` and `1/2` are all equivalent).

Reply with EXACTLY one token on the first line: CORRECT or INCORRECT
Then optionally one short sentence of justification."""


def call_judge(client, model_id: str, item: dict) -> dict:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": JUDGE_PROMPT.format(
                            question=item["question"],
                            gold=item["gold"],
                            extracted=item.get("extracted", "(none)"),
                            response_tail=str(item.get("last_worker_response_tail", ""))[
                                -1500:
                            ],
                        ),
                    }
                ],
            }
        ],
    }
    t0 = time.time()
    resp = client.invoke_model(modelId=model_id, body=json.dumps(body))
    dur = time.time() - t0
    payload = json.loads(resp["body"].read())
    text = payload["content"][0]["text"].strip()
    first = text.splitlines()[0].strip().upper() if text else ""
    verdict = (
        True
        if first.startswith("CORRECT")
        else False
        if first.startswith("INCORRECT")
        else None
    )
    usage = payload.get("usage", {})
    return {
        "verdict": verdict,
        "raw": text[:300],
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "elapsed_s": round(dur, 2),
    }


def grade_one(client, item):
    out = {
        "q_idx": item["q_idx"],
        "question_id": item.get("question_id"),
        "gold": item["gold"],
        "extracted": item.get("extracted"),
        "parser_correct": item["correct"],
    }
    out["haiku"] = call_judge(client, HAIKU, item)
    out["sonnet"] = call_judge(client, SONNET, item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="domains/autoresearch/blueprints/rl-conductor/results/greedy_eval/eval_greedy_v4_iter-0074_n50_paperfaithful.json",
    )
    ap.add_argument(
        "--output",
        default="domains/autoresearch/blueprints/cost-aware-routing/results/preflight/judge_agreement_n50.json",
    )
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    src = json.load(open(args.input))
    items = src["results"][: args.limit]
    client = boto3.client("bedrock-runtime", region_name=REGION)

    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(grade_one, client, it): it["q_idx"] for it in items}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as e:
                row = {"q_idx": futures[fut], "error": repr(e)}
            rows.append(row)
            print(
                f"[{i}/{len(items)}] q_idx={row.get('q_idx')} "
                f"parser={row.get('parser_correct')} "
                f"haiku={row.get('haiku', {}).get('verdict')} "
                f"sonnet={row.get('sonnet', {}).get('verdict')}",
                flush=True,
            )

    rows.sort(key=lambda r: r.get("q_idx", -1))

    # Aggregate
    def vec(field):
        return [r.get(field, {}).get("verdict") for r in rows]

    parser_v = [r.get("parser_correct") for r in rows]
    haiku_v = vec("haiku")
    sonnet_v = vec("sonnet")

    def agree(a, b):
        n = sum(1 for x, y in zip(a, b) if x is not None and y is not None)
        m = sum(1 for x, y in zip(a, b) if x is not None and y is not None and x == y)
        return m, n, (m / n if n else None)

    haiku_sonnet = agree(haiku_v, sonnet_v)
    haiku_parser = agree(haiku_v, parser_v)
    sonnet_parser = agree(sonnet_v, parser_v)

    def confusion(a, b):
        # rows = a (judge), cols = b (truth)
        out = {"TT": 0, "TF": 0, "FT": 0, "FF": 0, "AbsentA": 0, "AbsentB": 0}
        for x, y in zip(a, b):
            if x is None:
                out["AbsentA"] += 1
                continue
            if y is None:
                out["AbsentB"] += 1
                continue
            out[("T" if x else "F") + ("T" if y else "F")] += 1
        return out

    summary = {
        "n": len(rows),
        "haiku_correct_rate": sum(1 for v in haiku_v if v) / len(rows),
        "sonnet_correct_rate": sum(1 for v in sonnet_v if v) / len(rows),
        "parser_correct_rate": sum(1 for v in parser_v if v) / len(rows),
        "agreement": {
            "haiku_vs_sonnet": haiku_sonnet,
            "haiku_vs_parser": haiku_parser,
            "sonnet_vs_parser": sonnet_parser,
        },
        "confusion_haiku_vs_sonnet": confusion(haiku_v, sonnet_v),
        "haiku_total_input_tokens": sum(
            r.get("haiku", {}).get("input_tokens", 0) for r in rows
        ),
        "haiku_total_output_tokens": sum(
            r.get("haiku", {}).get("output_tokens", 0) for r in rows
        ),
        "sonnet_total_input_tokens": sum(
            r.get("sonnet", {}).get("input_tokens", 0) for r in rows
        ),
        "sonnet_total_output_tokens": sum(
            r.get("sonnet", {}).get("output_tokens", 0) for r in rows
        ),
    }

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
