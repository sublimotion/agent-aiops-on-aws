#!/usr/bin/env python3
"""
Shared utilities for E_harness2 — DBBench JIT-vs-offline harness-authoring ablation.

Design decisions (grounded in carryover lessons):

* **Bedrock via `aws bedrock-runtime converse` CLI** with native tool-use.
  No pip/boto3/SDK is installable in this environment, exactly as E_fin1 found.
  The OpenAI Agents SDK path named in the spec is therefore replaced by the
  native Bedrock function-calling equivalent — the smoke test + L0→L1
  replication gate are the guards against misreading any SDK-path artifact
  (spec "Known Limitations"). Exponential backoff on throttling (E_fin1 lesson:
  "throttling is real at sustained call rates").

* **Oracle = SELECT-family tasks with a VERIFIED self-contained label.**
  DBBench's INSERT/UPDATE/DELETE tasks score via MySQL `md5()`/`group_concat`
  table-hashing, which the official Life-Harness task code *explicitly leaves
  unimplemented for SQLite* (task.py:607 "Table hash calculation for SQLite not
  implemented"). Replicating it would violate the verification-primitives
  lesson "never assume the eval harness works". We instead use the SELECT-family
  path (`label` + `DBResultProcessor.compare_results`), which ports faithfully,
  AND we keep only tasks whose GOLD sql, run in our SQLite env, reproduces the
  gold `label`. That gives a deterministic, self-verified oracle (the analog of
  the FinQA `exe_ans` exact-match gate / the Docker test oracle).

* **L1 harness is the VENDORED Life-Harness module, used frozen** — not a
  reimplementation. We import `vendor/life_harness_dbbench.py` (their
  DBBenchHarnessRuntime: H2 SQL gate, H3 prompt/tool patch, H4 post-step
  monitor, H5 BM25 skills) and drive it through the same hooks task.py uses.
"""

import contextlib
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VENDOR = os.path.join(ROOT, "vendor")
sys.path.insert(0, VENDOR)

import life_harness_dbbench as LH  # noqa: E402  (vendored, frozen)

# ---------------------------------------------------------------------------
# Vendored DBBench scorer (strip the mysql .interaction import)
# ---------------------------------------------------------------------------
_rp_src = open(os.path.join(VENDOR, "dbbench_result_processor_raw.py")).read()
_rp_src = _rp_src.replace("from .interaction import Database", "Database=object")
_rp_ns = {}
exec(compile(_rp_src, "dbbench_result_processor", "exec"), _rp_ns)
DBResultProcessor = _rp_ns["DBResultProcessor"]


def compare_results(answer, label, query_type):
    """Official DBBench comparison, with its debug prints silenced."""
    with contextlib.redirect_stdout(io.StringIO()):
        return DBResultProcessor.compare_results(answer, label, query_type)


# ---------------------------------------------------------------------------
# Bedrock model IDs (verified callable in us-east-2 on 2026-06-21, E_fin1)
# ---------------------------------------------------------------------------
BEDROCK_MODELS = {
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
}
# List pricing per 1M tokens (input, output), USD.
PRICING = {
    "haiku": (1.00, 5.00),
    "sonnet": (3.00, 15.00),
}
REGION = os.environ.get("AWS_REGION", "us-east-2")


def cost_usd(input_tokens, output_tokens, model_key="haiku"):
    pin, pout = PRICING[model_key]
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


def converse(messages, model_key="haiku", system=None, tools=None,
             temperature=0.0, max_tokens=1024, max_retries=6):
    """Call Bedrock `converse` with optional tool-config. Returns dict:
    {stop_reason, text, tool_uses:[{id,name,input}], input_tokens,
     output_tokens, latency_ms}. Exponential backoff on throttling."""
    model_id = BEDROCK_MODELS[model_key]
    inf = {"maxTokens": max_tokens, "temperature": temperature}

    last_err = None
    for attempt in range(max_retries):
        tmpfiles = []
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as mf:
                json.dump(messages, mf)
                tmpfiles.append(mf.name)
            cmd = ["aws", "bedrock-runtime", "converse",
                   "--region", REGION, "--model-id", model_id,
                   "--messages", f"file://{mf.name}",
                   "--inference-config", json.dumps(inf)]
            if system:
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as sf:
                    json.dump([{"text": system}], sf)
                    tmpfiles.append(sf.name)
                cmd += ["--system", f"file://{sf.name}"]
            if tools:
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
                    json.dump({"tools": tools}, tf)
                    tmpfiles.append(tf.name)
                cmd += ["--tool-config", f"file://{tf.name}"]

            start = time.monotonic()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            latency_ms = int((time.monotonic() - start) * 1000)
        finally:
            for p in tmpfiles:
                try:
                    os.unlink(p)
                except OSError:
                    pass

        if proc.returncode != 0:
            err = proc.stderr.strip()
            last_err = err
            if any(t in err for t in ("Throttling", "TooManyRequests",
                                      "ServiceUnavailable", "timed out",
                                      "InternalServerException", "ModelTimeout")):
                time.sleep(2 ** attempt + 0.5)
                continue
            raise RuntimeError(f"Bedrock error: {err}")

        resp = json.loads(proc.stdout)
        msg = resp["output"]["message"]
        text_parts, tool_uses = [], []
        for block in msg.get("content", []):
            if "text" in block:
                text_parts.append(block["text"])
            if "toolUse" in block:
                tu = block["toolUse"]
                tool_uses.append({"id": tu["toolUseId"], "name": tu["name"],
                                  "input": tu.get("input", {})})
        usage = resp.get("usage", {})
        return {
            "stop_reason": resp.get("stopReason"),
            "text": "\n".join(text_parts),
            "tool_uses": tool_uses,
            "assistant_content": msg.get("content", []),
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "latency_ms": latency_ms,
        }

    raise RuntimeError(f"Bedrock failed after {max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# DBBench SQLite environment (deterministic oracle)
# ---------------------------------------------------------------------------
def _coldefs(columns):
    return ", ".join('"' + c["name"] + '" TEXT' for c in columns)


def build_sqlite(entry):
    """Build an in-memory SQLite DB from the entry's inline table(s).
    Returns (conn, [table_names]). All columns are TEXT, matching DBBench."""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    tables = entry["table"] if isinstance(entry["table"], list) else [entry["table"]]
    names = []
    for t in tables:
        ti = t["table_info"]
        name = t["table_name"]
        names.append(name)
        cur.execute(f'CREATE TABLE "{name}" ({_coldefs(ti["columns"])})')
        ph = ",".join("?" * len(ti["columns"]))
        cur.executemany(f'INSERT INTO "{name}" VALUES ({ph})',
                        [tuple(str(x) for x in row) for row in ti["rows"]])
    conn.commit()
    return conn, names


def run_sql(conn, sql):
    """Execute one SQL statement; return DBBench-style result string.
    Mirrors the SQLite executor in interaction.py (str(fetchall), 800-char cap)."""
    cur = conn.cursor()
    try:
        cur.execute(sql)
        try:
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            conn.commit()
            rows = []
        result_str = str(rows)
        conn.commit()
    except Exception as e:  # noqa: BLE001
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        result_str = f"SQLite execution error: {e}"
    finally:
        cur.close()
    if len(result_str) > 800:
        result_str = result_str[:800] + "[TRUNCATED]"
    return result_str


def answers_from_result(result_str):
    """Convert a raw SQLite fetchall-string into the answers list the agent
    would submit / the scorer would compare (scalar col -> [v], else repr)."""
    try:
        rows = eval(result_str)  # noqa: S307 — trusted: our own run_sql output
    except Exception:  # noqa: BLE001
        return [result_str]
    if not rows:
        return []
    if len(rows[0]) == 1:
        return [str(r[0]) for r in rows]
    return [str(r) for r in rows]


def gold_label_matches(entry):
    """True if the gold SQL, run in our SQLite env, reproduces the gold label.
    This is the eval-harness-trust gate: only tasks that PASS define the set."""
    try:
        conn, _ = build_sqlite(entry)
        res = run_sql(conn, entry["sql"]["query"])
        conn.close()
        if res.startswith("SQLite execution error"):
            return False
        ans = answers_from_result(res)
        return compare_results(str(ans) if len(ans) != 1 else ans[0],
                               entry["label"], entry["type"][0])
    except Exception:  # noqa: BLE001
        return False


SELECT_FAMILY = {"SELECT", "other", "counting", "comparison", "ranking",
                 "aggregation-MIN", "aggregation-MAX", "aggregation-AVG",
                 "aggregation-SUM", "aggregation-COUNT"}


def is_select_family(entry):
    return entry["type"][0] in SELECT_FAMILY
