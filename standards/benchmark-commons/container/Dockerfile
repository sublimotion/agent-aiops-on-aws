# AIPerf Benchmark Runner — standardized container for all GPU serving benchmarks
# Includes: AIPerf CLI, enrichment wrapper, workload catalog, schema validation
#
# Usage:
#   docker build -t aiperf-runner:latest -f standards/benchmark-commons/container/Dockerfile .
#   # or push to ECR:
#   aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REPO
#   docker tag aiperf-runner:latest $ECR_REPO/aiperf-runner:latest
#   docker push $ECR_REPO/aiperf-runner:latest

FROM python:3.11-slim AS base

# System deps for aiohttp, prometheus client, and AIPerf's ZMQ pipeline
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# AIPerf + dependencies
RUN pip install --no-cache-dir \
    aiperf>=0.6.0 \
    pyyaml>=6.0 \
    jsonschema>=4.20 \
    prometheus-client>=0.20 \
    requests>=2.31 \
    aiohttp>=3.9

# Enrichment tooling
COPY standards/benchmark-commons/container/enrich-benchmark.py /usr/local/bin/enrich-benchmark
COPY standards/benchmark-commons/container/scrape-engine-metrics.py /usr/local/bin/scrape-engine-metrics
COPY standards/benchmark-commons/container/validate-artifact.py /usr/local/bin/validate-artifact
RUN chmod +x /usr/local/bin/enrich-benchmark /usr/local/bin/scrape-engine-metrics /usr/local/bin/validate-artifact

# Workload catalog (standard definitions)
COPY standards/benchmark-commons/workloads/ /etc/aiperf/workloads/

# JSON Schema for artifact validation
COPY standards/benchmark-commons/container/schema/ /etc/aiperf/schema/

# Working directory for results
WORKDIR /results

# Default: run AIPerf with enrichment
ENTRYPOINT ["/usr/local/bin/enrich-benchmark"]
CMD ["--help"]
