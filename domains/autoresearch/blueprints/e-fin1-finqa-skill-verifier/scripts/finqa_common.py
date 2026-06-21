#!/usr/bin/env python3
"""
Shared utilities for E_fin1 FinQA skill-verifier replication.

- Bedrock call via the `aws bedrock-runtime converse` CLI (boto3 is NOT
  installed in this environment; the AWS CLI is). Returns text + token usage.
- FinQA context serialization (question + table + gold supporting text).
- exact_match(): numeric ground-truth scorer with rounding/units tolerance,
  handling FinQA's inconsistent percent representation in qa.exe_ans.
- cost_usd(): token-based cost using Haiku-4.5 list pricing.

Why CLI not boto3: this host has aws-cli/2.x but no pip/boto3. The CLI
`converse` op returns usage.inputTokens/outputTokens, which is all we need.
"""

import json
import os
import re
import subprocess
import tempfile
import time

# ---------------------------------------------------------------------------
# Bedrock model IDs (verified callable in us-east-2 on 2026-06-21)
# ---------------------------------------------------------------------------
BEDROCK_MODELS = {
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
    # Non-Claude verifier for the RQ2 calibration check. Coding-domain T4
    # found Nova Pro precision=0.14 (failed to calibrate); re-tested here.
    "nova-pro": "us.amazon.nova-pro-v1:0",
}

# List pricing per 1M tokens (input, output), USD.
# Haiku 4.5 = $1.00 / $5.00. We log raw token counts too so cost can be
# recomputed under any pricing. The verifier-reward $0.03 coding ceiling was
# computed at (0.80, 4.00) for Haiku; we report under both when summarizing.
PRICING = {
    "haiku": (1.00, 5.00),
    "sonnet": (3.00, 15.00),
    "nova-pro": (0.80, 3.20),
}

REGION = os.environ.get("AWS_REGION", "us-east-2")


def cost_usd(input_tokens, output_tokens, model_key="haiku", pricing=None):
    pin, pout = (pricing or PRICING)[model_key]
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


def call_bedrock(prompt, model_key="haiku", temperature=0.0, max_tokens=1024,
                 max_retries=5):
    """Call Claude via `aws bedrock-runtime converse`. Returns dict with
    text, input_tokens, output_tokens, latency_ms. Retries on throttling."""
    model_id = BEDROCK_MODELS[model_key]
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    inf = {"maxTokens": max_tokens, "temperature": temperature}

    last_err = None
    for attempt in range(max_retries):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as mf:
            json.dump(messages, mf)
            msg_path = mf.name
        try:
            start = time.monotonic()
            proc = subprocess.run(
                [
                    "aws", "bedrock-runtime", "converse",
                    "--region", REGION,
                    "--model-id", model_id,
                    "--messages", f"file://{msg_path}",
                    "--inference-config", json.dumps(inf),
                ],
                capture_output=True, text=True, timeout=180,
            )
            latency_ms = int((time.monotonic() - start) * 1000)
        finally:
            try:
                os.unlink(msg_path)
            except OSError:
                pass

        if proc.returncode != 0:
            err = proc.stderr.strip()
            last_err = err
            # Backoff on throttling / transient errors
            if any(t in err for t in ("Throttling", "TooManyRequests",
                                      "ServiceUnavailable", "timed out",
                                      "InternalServerException")):
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Bedrock error: {err}")

        resp = json.loads(proc.stdout)
        text = resp["output"]["message"]["content"][0]["text"]
        usage = resp.get("usage", {})
        return {
            "text": text,
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "latency_ms": latency_ms,
        }

    raise RuntimeError(f"Bedrock failed after {max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# FinQA context serialization
# ---------------------------------------------------------------------------
def serialize_table(table):
    """Render a FinQA table (list of rows) as pipe-delimited markdown."""
    if not table:
        return ""
    lines = []
    for row in table:
        lines.append(" | ".join(str(c) for c in row))
    return "\n".join(lines)


def gold_text(qa):
    """Concatenate the gold supporting sentences (qa.gold_inds values)."""
    inds = qa.get("gold_inds", {})
    return "\n".join(str(v) for v in inds.values())


def build_context(example, use_gold_inds=True):
    """Build the financial context block for an agent prompt.

    Uses gold_inds (the annotated supporting evidence) to keep context tight
    and token cost low — this is the retrieval-oracle setting, which isolates
    the verification question from the retrieval question (E_fin1 is about
    verification, not retrieval)."""
    qa = example["qa"]
    parts = []
    table = example.get("table", [])
    if table:
        parts.append("TABLE:\n" + serialize_table(table))
    if use_gold_inds:
        gt = gold_text(qa)
        if gt:
            parts.append("RELEVANT TEXT:\n" + gt)
    else:
        pre = " ".join(example.get("pre_text", []))
        post = " ".join(example.get("post_text", []))
        parts.append("TEXT:\n" + pre + "\n" + post)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Numeric ground-truth scorer
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\$?\(?\d[\d,]*\.?\d*\)?%?")


def _to_float(s):
    """Parse a numeric string allowing $ , ( ) % and stray text.
    Returns (value, was_percent) or (None, False)."""
    if s is None:
        return None, False
    if isinstance(s, (int, float)):
        return float(s), False
    s = str(s).strip()
    was_pct = "%" in s
    neg = s.strip().startswith("(") and s.strip().endswith(")")
    cleaned = s.replace(",", "").replace("$", "").replace("%", "")
    cleaned = cleaned.replace("(", "").replace(")", "").strip()
    try:
        v = float(cleaned)
    except ValueError:
        return None, was_pct
    if neg:
        v = -abs(v)
    return v, was_pct


def _close(a, b, rel=0.01, absol=1e-6):
    """Match within relative tolerance (default 1%), with a tiny absolute
    floor only to catch exact-zero golds. 1% relative covers FinQA's
    round-to-5-decimals gold vs an agent's 2-3 sig-fig stated answer, while
    still rejecting genuinely different numbers (>1% apart). A larger
    absolute floor was rejected in the Stage-0 smoke test: for ratio-form
    percents (gold 0.935) it created a multi-point match window."""
    if a is None or b is None:
        return False
    if abs(a - b) <= absol:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= rel


def exact_match(pred, gold_exe_ans):
    """Score a predicted answer against FinQA qa.exe_ans.

    Handles:
      - yes/no string answers (exact, case-insensitive)
      - numeric with $ , ( ) tolerance
      - FinQA percent inconsistency: exe_ans stores some percents as ratios
        (0.935 for "93.5%") and some as already-scaled (24.69 for "24.69%").
        So we accept a match if pred equals gold, gold*100, or gold/100.
      - rounding: 1% relative / 0.02 absolute tolerance.
    Returns bool.
    """
    # Categorical (yes/no)
    if isinstance(gold_exe_ans, str) and gold_exe_ans.strip().lower() in ("yes", "no"):
        if pred is None:
            return False
        return str(pred).strip().lower().startswith(gold_exe_ans.strip().lower())

    gv, _ = _to_float(gold_exe_ans)
    pv, p_pct = _to_float(pred)
    if gv is None or pv is None:
        return False

    # Direct, and the two percent-scaling reconciliations.
    candidates = [gv, gv * 100.0, gv / 100.0]
    for c in candidates:
        if _close(pv, c):
            return True
    return False


# ---------------------------------------------------------------------------
# Robust JSON extraction from model output
# ---------------------------------------------------------------------------
def extract_json(text):
    """Pull a JSON object from a model response (handles ```json fences)."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1)
    # Find first balanced {...}
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
