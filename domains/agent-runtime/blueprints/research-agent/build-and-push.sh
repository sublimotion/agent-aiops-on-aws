#!/bin/bash
set -e

echo "Building Research Agent container image..."

# Variables
ECR_REPO="615299764834.dkr.ecr.us-east-1.amazonaws.com/research-agent"
IMAGE_TAG="${1:-latest}"
REGION="us-east-1"

# Authenticate to ECR
echo "Authenticating to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO

# Clone the agent SDK demos repo to get base code
echo "Cloning agent SDK demos..."
git clone --depth 1 https://github.com/anthropics/claude-agent-sdk-demos /tmp/sdk-demos

# Create a temporary build directory
BUILD_DIR="/tmp/build-context"
mkdir -p $BUILD_DIR

# Copy the Dockerfile and requirements
cat > $BUILD_DIR/Dockerfile << 'DOCKERFILE'
# Research Agent — AgentCore Runtime container
# Platform: linux/arm64 (Graviton / ECS Fargate ARM64)
# Build: docker buildx build --platform linux/arm64 -t <ecr-url>:latest .

FROM python:3.11-slim AS base

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
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2 (ARM64)
RUN curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws

WORKDIR /app

# Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Source: clone original research-agent from GitHub to get utils + prompts
RUN git clone --depth 1 https://github.com/anthropics/claude-agent-sdk-demos /tmp/sdk-demos \
    && cp -r /tmp/sdk-demos/research-agent/research_agent /app/research_agent \
    && rm -rf /tmp/sdk-demos

# Overwrite agent.py with the Bedrock-adapted version
COPY agent.py /app/research_agent/agent.py

# HTTP server entry point
COPY server.py /app/server.py

# Runtime directories (EFS will be mounted at /app/files at task start)
RUN mkdir -p /app/files/research_notes \
              /app/files/data \
              /app/files/charts \
              /app/files/reports \
              /app/logs

# Environment defaults (overridden by ECS task definition)
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    FILES_BASE=/app/files \
    AWS_BEDROCK_REGION=us-east-1 \
    ANTHROPIC_BEDROCK=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "server.py"]
DOCKERFILE

# Create requirements.txt
cat > $BUILD_DIR/requirements.txt << 'REQUIREMENTS'
# Core dependencies
claude-agent-sdk>=0.2.0
anthropic-bedrock>=0.1.0
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.0.0

# Data processing and visualization
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
reportlab>=4.0.0

# AWS SDK
boto3>=1.34.0

# Utilities
python-dotenv>=1.0.0
httpx>=0.25.0
REQUIREMENTS

# Create a modified agent.py with Bedrock support
cat > $BUILD_DIR/agent.py << 'AGENT_PY'
"""Research Agent - Modified for Bedrock"""
import os
import asyncio
from claude_agent_sdk import Agent, Task
from anthropic_bedrock import AnthropicBedrock

# Use AnthropicBedrock client when ANTHROPIC_BEDROCK env var is set
if os.getenv("ANTHROPIC_BEDROCK") == "1":
    client = AnthropicBedrock(
        aws_region=os.getenv("AWS_BEDROCK_REGION", "us-east-1")
    )
else:
    # Fallback to standard Anthropic client
    from anthropic import Anthropic
    client = Anthropic()

# Import the rest of the agent logic from the original
import sys
sys.path.append('/app')
from research_agent.agent import *

# Override the client in the main agent
if __name__ == "__main__":
    # Set file paths to use /app/files mount point
    os.environ["FILES_BASE"] = "/app/files"
AGENT_PY

# Create server.py
cat > $BUILD_DIR/server.py << 'SERVER_PY'
"""FastAPI server wrapper for Research Agent"""
import os
import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Import the research agent
from research_agent import agent

app = FastAPI(title="Research Agent API")

class InvokeRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "agent": "research-agent",
        "version": "1.0.0"
    }

@app.post("/invoke")
async def invoke(request: InvokeRequest):
    """Invoke the research agent with a query"""
    try:
        # Run the agent
        result = await agent.run(request.query)

        # Stream the response as Server-Sent Events
        async def generate():
            yield f"data: {json.dumps({'type': 'text', 'content': 'Starting research...'})}\n\n"

            # Send progress updates
            for update in result.get("updates", []):
                yield f"data: {json.dumps({'type': 'text', 'content': update})}\n\n"

            # Send final result
            yield f"data: {json.dumps({'type': 'result', 'content': result.get('report_url', 'Report generation complete')})}\n\n"
            yield f"data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
SERVER_PY

# Build the image
echo "Building Docker image..."
cd $BUILD_DIR
docker buildx create --use --name multiarch || true
docker buildx build --platform linux/arm64 --push -t $ECR_REPO:$IMAGE_TAG -t $ECR_REPO:latest .

# Verify the image was pushed
echo "Verifying image in ECR..."
aws ecr describe-images --repository-name research-agent --region $REGION --image-ids imageTag=$IMAGE_TAG

echo "Build complete! Image pushed to $ECR_REPO:$IMAGE_TAG"