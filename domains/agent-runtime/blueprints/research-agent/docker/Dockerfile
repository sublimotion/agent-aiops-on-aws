# Research Agent — AgentCore Runtime container
# Platform: linux/arm64 (Graviton / ECS Fargate ARM64)
# Build: docker buildx build --platform linux/arm64 -t <ecr-url>:latest .

FROM public.ecr.aws/docker/library/python:3.11-slim AS base

ARG DEBIAN_FRONTEND=noninteractive

# System dependencies for matplotlib, reportlab, and EFS/NFS client
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    unzip \
    nfs-common \
    libglib2.0-0 \
    libgl1 \
    fontconfig \
    fonts-dejavu \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2 (ARM64)
RUN curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws

WORKDIR /app

# Create non-root user for security (claude refuses --dangerously-skip-permissions as root)
RUN groupadd -r agent && useradd -r -g agent -d /app -s /bin/bash agent

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Source: clone original research-agent from GitHub to get utils + prompts
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/anthropics/claude-agent-sdk-demos /tmp/sdk-demos \
    && cp -r /tmp/sdk-demos/research-agent/research_agent /app/research_agent \
    && rm -rf /tmp/sdk-demos

# ---------------------------------------------------------------------------
# Overwrite agent.py with the Bedrock-adapted version
# ---------------------------------------------------------------------------
COPY research_agent/agent.py /app/research_agent/agent.py

# ---------------------------------------------------------------------------
# HTTP server entry point
# ---------------------------------------------------------------------------
COPY server.py /app/server.py
COPY mcp_search_server.py /app/mcp_search_server.py
COPY .claude/ /app/.claude/

# ---------------------------------------------------------------------------
# Runtime directories (EFS will be mounted at /app/files at task start)
# ---------------------------------------------------------------------------
RUN mkdir -p /app/files/research_notes \
              /app/files/data \
              /app/files/charts \
              /app/files/reports \
              /app/logs \
    && chown -R agent:agent /app

# ---------------------------------------------------------------------------
# Environment defaults (most values injected by ECS task definition at runtime)
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    FILES_BASE=/app/files \
    AWS_REGION=us-east-1 \
    CLAUDE_CODE_USE_BEDROCK=1

# S3_OUTPUT_BUCKET is injected by ECS task definition (see main.tf websocket_proxy module)

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run as non-root so claude --dangerously-skip-permissions works
USER agent

CMD ["python", "server.py"]
