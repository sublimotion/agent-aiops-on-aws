#!/usr/bin/env bash
# Stage 5 / P0 smoke gate for Nemotron-3-Ultra-550B-A55B-NVFP4.
# Confirms ALL 6 items the deployment instruction requires before committing bench hours.
#
# Usage: BASE=http://localhost:8000 MODEL=nvidia/nemotron-3-ultra ./smoke-test.sh
# Sampling is model-mandated: temperature=1.0, top_p=0.95 (NO greedy).
set -uo pipefail

BASE="${BASE:-http://localhost:8000}"
MODEL="${MODEL:-nvidia/nemotron-3-ultra}"
PASS=0; FAIL=0
ok(){ echo "  PASS: $1"; PASS=$((PASS+1)); }
no(){ echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

echo "=== Item 1+2: /health 200 + coherent 1K/512 completion ==="
code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")
[ "$code" = "200" ] && ok "/health -> 200" || no "/health -> $code"

RESP=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\":\"user\",\"content\":\"Explain why the sky is blue in about 300 words.\"}],
  \"max_tokens\": 512, \"temperature\": 1.0, \"top_p\": 0.95
}")
echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); t=d['choices'][0]['message']['content']; print('--- sample ---'); print(t[:500]); assert len(t.split())>50, 'too short'; import collections; w=t.split(); assert max(collections.Counter(w).values())<len(w)*0.4, 'degenerate repetition'" \
  && ok "coherent non-degenerate output" || no "completion invalid/degenerate"

echo "=== Item 4a: Reasoning ON -> </think> trace present ==="
RON=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\":\"user\",\"content\":\"What is 17*23? Think step by step.\"}],
  \"max_tokens\": 512, \"temperature\": 1.0, \"top_p\": 0.95,
  \"chat_template_kwargs\": {\"enable_thinking\": true}
}")
echo "$RON" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d['choices'][0]['message']; rc=m.get('reasoning_content') or ''; ct=m.get('content') or ''; assert ('</think>' in ct) or rc, 'no reasoning trace'; print('reasoning_content len:', len(rc), '| </think> in content:', '</think>' in ct)" \
  && ok "enable_thinking=True produces trace" || no "no </think> trace with thinking ON"

echo "=== Item 4b: Reasoning OFF -> no trace ==="
ROFF=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\":\"user\",\"content\":\"What is 17*23?\"}],
  \"max_tokens\": 256, \"temperature\": 1.0, \"top_p\": 0.95,
  \"chat_template_kwargs\": {\"enable_thinking\": false}
}")
echo "$ROFF" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d['choices'][0]['message']; rc=m.get('reasoning_content') or ''; ct=m.get('content') or ''; assert ('</think>' not in ct) and not rc.strip(), 'unexpected trace with thinking OFF'; print('no trace as expected')" \
  && ok "enable_thinking=False produces no trace" || no "trace leaked with thinking OFF"

echo "=== Item 6: Tool call via qwen3_coder parser ==="
TOOL=$(curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\":\"user\",\"content\":\"What is the weather in San Francisco?\"}],
  \"temperature\": 1.0, \"top_p\": 0.95, \"max_tokens\": 256,
  \"tools\": [{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]
}")
echo "$TOOL" | python3 -c "import sys,json; d=json.load(sys.stdin); tc=d['choices'][0]['message'].get('tool_calls'); assert tc and tc[0]['function']['name']=='get_weather', 'no valid tool_calls'; print('tool_calls:', json.dumps(tc)[:300])" \
  && ok "qwen3_coder tool_calls valid" || no "tool call parse failed"

echo "=== Item 5: MTP spec-decode acceptance > 0 in /metrics ==="
METRICS=$(curl -s "$BASE/metrics")
echo "$METRICS" | grep -E 'spec_decode|accepted_tokens|num_draft' | head -10
ACC=$(echo "$METRICS" | python3 -c "
import sys,re
txt=sys.stdin.read()
def val(name):
    m=re.findall(r'^'+re.escape(name)+r'(?:\{[^}]*\})?\s+([0-9eE+.\-]+)', txt, re.M)
    return sum(float(x) for x in m) if m else 0.0
acc=val('vllm:spec_decode_num_accepted_tokens_total')
draft=val('vllm:spec_decode_num_draft_tokens_total') or val('vllm:spec_decode_num_drafts_total')
print(f'accepted={acc} draft={draft}', file=sys.stderr)
print(acc)
")
python3 -c "import sys; acc=float('$ACC' or 0); sys.exit(0 if acc>0 else 1)" \
  && ok "MTP acceptance > 0 (accepted=$ACC)" || no "MTP acceptance not present or 0 (run a few requests first)"

echo
echo "=============================="
echo "SMOKE GATE: PASS=$PASS FAIL=$FAIL"
echo "Gate requires all green AND BFCL>=75% (run separately) before P1-P4."
[ "$FAIL" -eq 0 ] && echo "RESULT: GATE PASS" || echo "RESULT: GATE FAIL — STOP and diagnose"
