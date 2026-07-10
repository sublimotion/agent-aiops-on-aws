#!/usr/bin/env python3
"""Store -> restart -> replay L2 probe for llm-d on HyperPod.

Proves that a KV cache hit survives a vLLM replica restart, so the hit can only
come from the HyperPod managed ai-toolkit L2 daemon (L0 GPU + L1 CPU are wiped
by the restart).

Usage:
    # port-forward the gateway first, e.g.:
    #   kubectl port-forward -n llmd-hp-lmcache svc/<gateway-svc> 8080:80
    python3 llmd_l2_probe.py store
    kubectl rollout restart deployment -n llmd-hp-lmcache -l llm-d.ai/model=Qwen3-0.6B
    kubectl rollout status  deployment -n llmd-hp-lmcache -l llm-d.ai/model=Qwen3-0.6B --timeout=15m
    python3 llmd_l2_probe.py replay

Requests go through the gateway (not the pod) so the proof also exercises the
EPP routing path. Replica metrics are read separately via --metrics-url.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
MODEL = "Qwen/Qwen3-0.6B"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Deterministic shared prefix. Keep the phrase stable across store/replay so the
# LMCache key matches (PYTHONHASHSEED=0 on the replica keeps keys stable too).
PREFIX_PHRASE = "SageMaker HyperPod llm-d LMCache L2 replay stable-key probe. "
PREFIX_REPEATS = 45


def request(method, url, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    headers = {} if body is None else {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read().decode()


def selected_metrics(metrics_url):
    if not metrics_url:
        return []
    try:
        _, text = request("GET", metrics_url, timeout=10)
    except Exception as exc:  # metrics are best-effort; the artifact records the failure
        return [f"# metrics fetch failed: {exc}"]
    prefixes = (
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
    prefix = PREFIX_PHRASE * PREFIX_REPEATS
    prompt = prefix + "\nAnswer in one short sentence: what cache path is being validated?"
    return prompt, len(prefix)


def run(phase, base_url, metrics_url):
    prompt, common_prefix_chars = build_prompt()
    health_status = models_status = None
    try:
        health_status, _ = request("GET", f"{base_url}/health")
    except Exception:
        pass
    try:
        models_status, _ = request("GET", f"{base_url}/v1/models")
    except Exception:
        pass

    before = selected_metrics(metrics_url)

    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": False,
    }
    start = time.time()
    status, raw_response = request("POST", f"{base_url}/v1/chat/completions", body=body, timeout=180)
    latency_ms = round((time.time() - start) * 1000, 2)
    response = json.loads(raw_response)

    time.sleep(2)
    after = selected_metrics(metrics_url)

    artifact = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": f"llmd-l2-{phase}",
        "orchestrator": "llm-d",
        "endpoint": base_url,
        "model": MODEL,
        "common_prefix_chars": common_prefix_chars,
        "health_status": health_status,
        "models_status": models_status,
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
    stamp = time.strftime("%Y%m%d", time.gmtime())
    path = RESULTS_DIR / f"e2e-telemetry-llmd-{phase}-{stamp}.json"
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"path": str(path), "request": artifact["request"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["store", "replay"])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="gateway base URL (default: %(default)s)")
    parser.add_argument("--metrics-url", default=None,
                        help="vLLM replica /metrics URL (e.g. http://127.0.0.1:8000/metrics via a second port-forward)")
    args = parser.parse_args()
    run(args.phase, args.base_url.rstrip("/"), args.metrics_url)


if __name__ == "__main__":
    main()
