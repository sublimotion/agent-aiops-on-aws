#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request
from pathlib import Path


BASE_URL = "http://127.0.0.1:18000"
METRICS_URL = "http://127.0.0.1:18081/metrics"
MODEL = "Qwen/Qwen3-0.6B"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def request(method, url, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read().decode()


def selected_metrics():
    _, text = request("GET", METRICS_URL, timeout=10)
    prefixes = (
        "dynamo_component_requests_total",
        "vllm:external_prefix_cache",
        "vllm:prefix_cache",
        "vllm:request_success_total",
        "lmcache:num_",
        "lmcache:lookup",
        "lmcache:retrieve",
        "lmcache:remote",
    )
    return [line for line in text.splitlines() if line.startswith(prefixes)]


def build_prompt():
    phrase = "SageMaker HyperPod Dynamo LMCache shorter frontend L2 replay stable key 20260709. "
    prefix = phrase * 45
    return prefix + "\nAnswer in one short sentence: what cache path is being validated?", len(prefix)


def run(phase):
    prompt, common_prefix_chars = build_prompt()
    health_status, health = request("GET", f"{BASE_URL}/health")
    models_status, models = request("GET", f"{BASE_URL}/v1/models")
    before = selected_metrics()

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }
    start = time.time()
    status, raw_response = request("POST", f"{BASE_URL}/v1/chat/completions", body=body, timeout=180)
    latency_ms = round((time.time() - start) * 1000, 2)
    response = json.loads(raw_response)

    time.sleep(2)
    after = selected_metrics()

    artifact = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": f"dynamo-frontend-short-{phase}",
        "endpoint": "dynamo-lmcache-frontend service via localhost:18000 port-forward",
        "prompt_tag": "dynamo-frontend-short-l2-replay-20260709",
        "common_prefix_chars": common_prefix_chars,
        "health_status": health_status,
        "health": json.loads(health),
        "models_status": models_status,
        "models": json.loads(models),
        "request": {
            "status": status,
            "latency_ms": latency_ms,
            "usage": response.get("usage"),
            "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
            "content": response.get("choices", [{}])[0].get("message", {}).get("content", "")[:240],
        },
        "selected_metrics_before": before,
        "selected_metrics_after": after,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"e2e-telemetry-dynamo-frontend-short-{phase}-20260709.json"
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"path": str(path), "request": artifact["request"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["store", "replay"])
    args = parser.parse_args()
    run(args.phase)


if __name__ == "__main__":
    main()
