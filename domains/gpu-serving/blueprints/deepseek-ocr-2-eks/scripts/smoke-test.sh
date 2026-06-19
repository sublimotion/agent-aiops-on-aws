#!/usr/bin/env bash
# DeepSeek-OCR-2 smoke test (Path B: /v1/chat/completions with grounding prompt).
#
# Sends ONE request using the winning prompt shape from iteration 2:
#   user.content[0].text       = "<image>\n<|grounding|>Convert the document to markdown. "
#   user.content[1].image_url  = data:image/png;base64,...   (bundled local asset)
#
# Prereq (run in another shell, or this script starts one itself):
#   kubectl -n cto-ocr-g6e-2xlarge port-forward svc/deepseek-ocr-g6e-2xlarge-svc 8000:8000
#
# Exits non-zero if:
#   - curl fails / non-200
#   - assistant content is empty
#   - assistant content matches the known-bad degenerate pattern "1. 1. 1. ..."

set -euo pipefail

ENDPOINT="${ENDPOINT:-http://localhost:8000}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-OCR-2}"
PROMPT="${PROMPT:-<image>\\n<|grounding|>Convert the document to markdown. }"
MAX_TOKENS="${MAX_TOKENS:-256}"
ASSET_DIR="$(cd "$(dirname "$0")" && pwd)/test-assets"
ASSET_PATH="${ASSET_PATH:-$ASSET_DIR/sample-doc.png}"

# --- dependency checks ------------------------------------------------------
for bin in jq curl base64; do
  command -v "$bin" >/dev/null 2>&1 || { echo "ERROR: '$bin' not found in PATH" >&2; exit 127; }
done

[[ -f "$ASSET_PATH" ]] || { echo "ERROR: test asset missing: $ASSET_PATH" >&2; exit 2; }

# --- build data URL (base64-encode bundled PNG) -----------------------------
B64="$(base64 < "$ASSET_PATH" | tr -d '\n')"
DATA_URL="data:image/png;base64,${B64}"

# --- build request body via jq (safe JSON encoding) -------------------------
BODY="$(jq -nc \
  --arg model "$MODEL" \
  --arg prompt "$(printf '%b' "$PROMPT")" \
  --arg url   "$DATA_URL" \
  --argjson max_tokens "$MAX_TOKENS" \
  '{model:$model, messages:[{role:"user", content:[
     {type:"text", text:$prompt},
     {type:"image_url", image_url:{url:$url}}
   ]}], max_tokens:$max_tokens, temperature:0.0}')"

# --- fire --------------------------------------------------------------------
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

HTTP_CODE="$(curl -sS -o "$TMP" -w '%{http_code}' \
  -X POST "$ENDPOINT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "$BODY")"

echo "HTTP $HTTP_CODE"
if [[ "$HTTP_CODE" != "200" ]]; then
  echo "--- response body ---" >&2
  cat "$TMP" >&2
  exit 1
fi

CONTENT="$(jq -r '.choices[0].message.content // ""' < "$TMP")"
USAGE="$(jq -c '.usage // {}' < "$TMP")"

echo "--- assistant content ---"
printf '%s\n' "$CONTENT"
echo "--- usage ---"
echo "$USAGE"

# Empty check
if [[ -z "${CONTENT//[[:space:]]/}" ]]; then
  echo "FAIL: empty assistant content" >&2
  exit 3
fi

# Degenerate pattern check: content that is essentially "1. 1. 1. ..." loop
# (strip whitespace, see if it reduces to repeating "1." tokens)
STRIPPED="$(printf '%s' "$CONTENT" | tr -d '[:space:]')"
if [[ "$STRIPPED" =~ ^(1\.){5,}$ ]]; then
  echo "FAIL: degenerate '1. 1. 1. ...' loop detected" >&2
  exit 4
fi

echo "PASS"
