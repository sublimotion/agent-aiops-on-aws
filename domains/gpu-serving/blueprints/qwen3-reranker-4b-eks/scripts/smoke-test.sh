#!/usr/bin/env bash
# Qwen3-Reranker-4B smoke test — winning shape is OpenAI-style /v1/score.
#
# Winning shape (iteration 1):
#   POST /v1/score
#   body: {"model": ..., "text_1": "<query>", "text_2": ["cand1", "cand2", ...]}
#   response: {"data": [{"index": 0, "score": 0.98...}, ...]}
#
# The /v1/rerank (Cohere-style) shape also works and returns results sorted
# with `relevance_score`, but /v1/score is simpler for benchmark plumbing
# because the response is in request-order. Both are validated below.
#
# Prereq:
#   kubectl -n cto-reranker-g6e-2xlarge port-forward svc/qwen3-reranker-g6e-2xlarge-svc 8000:8000
#
# Exits non-zero if either endpoint returns non-200 or the top-scored doc
# is not the obvious positive. This guards against the score-head being
# broken by a later vLLM upgrade.

set -euo pipefail

ENDPOINT="${ENDPOINT:-http://localhost:8000}"
MODEL="${MODEL:-Qwen/Qwen3-Reranker-4B}"

for bin in curl jq; do
  command -v "$bin" >/dev/null 2>&1 || { echo "ERROR: '$bin' missing" >&2; exit 127; }
done

# --- health -----------------------------------------------------------------
HTTP_CODE="$(curl -sS -o /dev/null -w '%{http_code}' "$ENDPOINT/health")"
[[ "$HTTP_CODE" == "200" ]] || { echo "FAIL: /health $HTTP_CODE" >&2; exit 1; }
echo "[smoke] /health OK"

# --- /v1/score --------------------------------------------------------------
SCORE_BODY='{"model":"'"$MODEL"'","text_1":"What is the capital of France?","text_2":["The capital of France is Paris.","Berlin is the capital of Germany.","Paris is a beautiful city."]}'
SCORE_RES="$(mktemp)"
trap 'rm -f "$SCORE_RES" "$RERANK_RES" 2>/dev/null || true' EXIT
HTTP_CODE="$(curl -sS -o "$SCORE_RES" -w '%{http_code}' -X POST "$ENDPOINT/v1/score" \
  -H 'Content-Type: application/json' --data-binary "$SCORE_BODY")"
[[ "$HTTP_CODE" == "200" ]] || { echo "FAIL: /v1/score HTTP $HTTP_CODE" >&2; cat "$SCORE_RES" >&2; exit 2; }
S0="$(jq -r '.data[0].score' < "$SCORE_RES")"
S1="$(jq -r '.data[1].score' < "$SCORE_RES")"
S2="$(jq -r '.data[2].score' < "$SCORE_RES")"
echo "[smoke] /v1/score scores: paris=$S0 berlin=$S1 beautiful=$S2"
# Positive docs (idx 0 and 2) should both beat the Berlin doc.
awk -v s0="$S0" -v s1="$S1" -v s2="$S2" 'BEGIN{exit !(s0>s1 && s2>s1)}' \
  || { echo "FAIL: wrong ordering on /v1/score" >&2; exit 3; }

# --- /v1/rerank -------------------------------------------------------------
RERANK_BODY='{"model":"'"$MODEL"'","query":"What is the capital of France?","documents":["The capital of France is Paris.","Berlin is the capital of Germany.","Paris is a beautiful city."]}'
RERANK_RES="$(mktemp)"
HTTP_CODE="$(curl -sS -o "$RERANK_RES" -w '%{http_code}' -X POST "$ENDPOINT/v1/rerank" \
  -H 'Content-Type: application/json' --data-binary "$RERANK_BODY")"
[[ "$HTTP_CODE" == "200" ]] || { echo "FAIL: /v1/rerank HTTP $HTTP_CODE" >&2; cat "$RERANK_RES" >&2; exit 4; }
TOP_IDX="$(jq -r '.results[0].index' < "$RERANK_RES")"
echo "[smoke] /v1/rerank top doc index: $TOP_IDX"
[[ "$TOP_IDX" == "0" || "$TOP_IDX" == "2" ]] \
  || { echo "FAIL: /v1/rerank top doc should be 0 or 2, got $TOP_IDX" >&2; exit 5; }

echo "PASS"
