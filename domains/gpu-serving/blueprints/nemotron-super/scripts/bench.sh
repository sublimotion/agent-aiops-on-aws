#!/usr/bin/env bash
# Nemotron-3-Super benchmark runner
# Usage: ./bench.sh <config> <concurrency> [output_tokens] [input_tokens]
# Example: ./bench.sh agg-vllm-tp2x4 16 1024 4096

set -euo pipefail

CONFIG="${1:?Usage: $0 <config> <concurrency> [output_tokens] [input_tokens]}"
CONCURRENCY="${2:?Usage: $0 <config> <concurrency> [output_tokens] [input_tokens]}"
OUTPUT_TOKENS="${3:-1024}"
INPUT_TOKENS="${4:-4096}"
ENDPOINT="${ENDPOINT:-http://localhost:8000}"
MODEL="${MODEL:-nemotron-3-super}"
RESULTS_DIR="${RESULTS_DIR:-results}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REASONING="${REASONING:-false}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULT_FILE="${RESULTS_DIR}/${CONFIG}_c${CONCURRENCY}_in${INPUT_TOKENS}_out${OUTPUT_TOKENS}_${TIMESTAMP}.json"

mkdir -p "${RESULTS_DIR}"

echo "=== Nemotron-3-Super Benchmark ==="
echo "Config:       ${CONFIG}"
echo "Endpoint:     ${ENDPOINT}"
echo "Model:        ${MODEL}"
echo "Concurrency:  ${CONCURRENCY}"
echo "Input tokens: ${INPUT_TOKENS}"
echo "Output tokens:${OUTPUT_TOKENS}"
echo "Num prompts:  ${NUM_PROMPTS}"
echo "Reasoning:    ${REASONING}"
echo "Output:       ${RESULT_FILE}"
echo "=================================="

# Build extra body for reasoning control
EXTRA_BODY=""
if [ "${REASONING}" = "true" ]; then
  EXTRA_BODY=',"extra_body":{"chat_template_kwargs":{"enable_thinking":true}}'
else
  EXTRA_BODY=',"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}'
fi

# Health check
echo "Checking endpoint health..."
if ! curl -sf "${ENDPOINT}/health" > /dev/null 2>&1; then
  echo "ERROR: ${ENDPOINT}/health not reachable"
  exit 1
fi
echo "Endpoint healthy."

# Run benchmark with python openai client
python3 - <<'PYTHON_SCRIPT' \
  "${ENDPOINT}" "${MODEL}" "${CONCURRENCY}" "${INPUT_TOKENS}" "${OUTPUT_TOKENS}" \
  "${NUM_PROMPTS}" "${REASONING}" "${RESULT_FILE}"
import sys, json, time, random, string, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

endpoint, model, concurrency, input_tokens, output_tokens, num_prompts, reasoning, result_file = sys.argv[1:]
concurrency = int(concurrency)
input_tokens = int(input_tokens)
output_tokens = int(output_tokens)
num_prompts = int(num_prompts)

client = OpenAI(base_url=f"{endpoint}/v1", api_key="none")

# Generate random prompts of approximate token count
def make_prompt(n_tokens):
    words = [random.choice(string.ascii_lowercase) * random.randint(3, 8) for _ in range(n_tokens // 2)]
    return "Continue this text: " + " ".join(words)

prompts = [make_prompt(input_tokens) for _ in range(num_prompts)]

extra_body = {"chat_template_kwargs": {"enable_thinking": reasoning == "true"}}

results = []

def run_one(prompt):
    t0 = time.perf_counter()
    first_token_time = None
    token_times = []
    output_text = ""

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=output_tokens,
            temperature=1.0,
            top_p=0.95,
            stream=True,
            extra_body=extra_body,
        )
        for chunk in stream:
            now = time.perf_counter()
            if chunk.choices:
                delta = chunk.choices[0].delta
                # Check both content and reasoning_content (Nemotron reasoning parser)
                text = getattr(delta, 'content', None) or getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
                if text:
                    if first_token_time is None:
                        first_token_time = now
                    else:
                        token_times.append(now)
                    output_text += text

        t_end = time.perf_counter()
        n_output = len(output_text.split())  # approximate
        ttft = (first_token_time - t0) * 1000 if first_token_time else None
        itls = [(token_times[i] - token_times[i-1]) * 1000 for i in range(1, len(token_times))] if len(token_times) > 1 else []
        e2e = (t_end - t0) * 1000

        return {
            "success": True,
            "ttft_ms": ttft,
            "e2e_ms": e2e,
            "itl_ms": itls,
            "output_tokens_approx": n_output,
            "output_len_chars": len(output_text),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

print(f"Running {num_prompts} prompts at concurrency {concurrency}...")
t_start = time.perf_counter()

with ThreadPoolExecutor(max_workers=concurrency) as pool:
    futures = [pool.submit(run_one, p) for p in prompts]
    for f in as_completed(futures):
        results.append(f.result())

t_total = time.perf_counter() - t_start

successes = [r for r in results if r["success"]]
failures = [r for r in results if not r["success"]]

ttfts = [r["ttft_ms"] for r in successes if r["ttft_ms"] is not None]
e2es = [r["e2e_ms"] for r in successes]
all_itls = [itl for r in successes for itl in r.get("itl_ms", [])]
total_output_chars = sum(r["output_len_chars"] for r in successes)

def pcts(data, ps=[50, 90, 99]):
    if not data:
        return {f"p{p}": None for p in ps}
    data_s = sorted(data)
    return {f"p{p}": data_s[min(int(len(data_s) * p / 100), len(data_s) - 1)] for p in ps}

summary = {
    "config": sys.argv[0],
    "concurrency": concurrency,
    "input_tokens": input_tokens,
    "output_tokens_target": output_tokens,
    "num_prompts": num_prompts,
    "reasoning": reasoning,
    "total_time_s": round(t_total, 2),
    "successes": len(successes),
    "failures": len(failures),
    "ttft_ms": pcts(ttfts),
    "e2e_ms": pcts(e2es),
    "itl_ms": pcts(all_itls),
    "total_output_chars": total_output_chars,
    "approx_output_toks_per_sec": round(total_output_chars / 4 / t_total, 1) if t_total > 0 else 0,
    "requests_per_sec": round(len(successes) / t_total, 2) if t_total > 0 else 0,
}

print(f"\n=== Results ===")
print(f"Successes: {len(successes)}/{num_prompts}")
print(f"Total time: {t_total:.1f}s")
print(f"TTFT: {summary['ttft_ms']}")
print(f"ITL:  {summary['itl_ms']}")
print(f"E2E:  {summary['e2e_ms']}")
print(f"Approx throughput: {summary['approx_output_toks_per_sec']} tok/s")
print(f"Requests/sec: {summary['requests_per_sec']}")

with open(result_file, "w") as f:
    json.dump({"summary": summary, "raw": results}, f, indent=2)
print(f"\nResults saved to {result_file}")
PYTHON_SCRIPT

echo "Benchmark complete: ${RESULT_FILE}"
