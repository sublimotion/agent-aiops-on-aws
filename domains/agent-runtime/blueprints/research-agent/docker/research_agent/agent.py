"""
Research agent adapted for Amazon Bedrock AgentCore Runtime.

Changes from the source (anthropics/claude-agent-sdk-demos/research-agent):
  - model="haiku" → BEDROCK_MODEL_ID (Opus 4.6 inference profile)
  - Removed load_dotenv() and ANTHROPIC_API_KEY check
  - File paths use /app/files/ (EFS mount) via FILES_BASE env var
  - Exports run_query() for use by server.py (non-interactive)
  - load_dotenv is gone; all config comes from ECS task env vars

The SDK uses the Anthropic Python client internally. Set these env vars to
redirect API calls to Bedrock instead of api.anthropic.com:
  AWS_BEDROCK_REGION=us-east-1
  (The SDK resolves boto3 credentials from the ECS task IAM role automatically.)
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import AsyncIterator

from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
)

from research_agent.utils.subagent_tracker import SubagentTracker
from research_agent.utils.transcript import setup_session, TranscriptWriter
from research_agent.utils.message_handler import process_assistant_message

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-opus-4-6-v1",
)

# Working directory for file I/O — EFS is mounted here in the container.
# Falls back to a local ./files directory for local dev.
FILES_BASE = Path(os.environ.get("FILES_BASE", "/app/files"))

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    prompt_path = PROMPTS_DIR / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _stderr_logger(line: str) -> None:
    """Capture stderr from the claude subprocess for debugging."""
    logging.getLogger("claude.stderr").error("[claude stderr] %s", line.rstrip())


def _get_bedrock_env() -> dict:
    """Fetch AWS credentials from boto3 and build env dict for claude subprocess."""
    try:
        import boto3
        session = boto3.Session()
        creds = session.get_credentials()
        if creds:
            frozen = creds.get_frozen_credentials()
            env = {
                **os.environ,
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
                "AWS_ACCESS_KEY_ID": frozen.access_key,
                "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
            }
            if frozen.token:
                env["AWS_SESSION_TOKEN"] = frozen.token
            return env
    except Exception as exc:
        logging.getLogger("agent").warning("Failed to get boto3 credentials: %s", exc)
    return {**os.environ, "CLAUDE_CODE_USE_BEDROCK": "1"}


def _make_options(transcript: TranscriptWriter, tracker: SubagentTracker) -> ClaudeAgentOptions:
    """Build ClaudeAgentOptions with all subagents wired up."""
    lead_agent_prompt = load_prompt("lead_agent.txt")
    researcher_prompt = load_prompt("researcher.txt")
    data_analyst_prompt = load_prompt("data_analyst.txt")
    report_writer_prompt = load_prompt("report_writer.txt")

    agents = {
        "researcher": AgentDefinition(
            description=(
                "Use this agent when you need to gather research information on any topic. "
                "The researcher uses web search to find relevant information, articles, and sources "
                "from across the internet. Writes research findings to files/research_notes/ "
                "for later use by report writers."
            ),
            tools=["mcp__search__web_search", "Write"],
            prompt=researcher_prompt,
            model=BEDROCK_MODEL_ID,
        ),
        "data-analyst": AgentDefinition(
            description=(
                "Use this agent AFTER researchers have completed their work to generate quantitative "
                "analysis and visualizations. Reads research notes from files/research_notes/, "
                "extracts numerical data, and generates charts using Python/matplotlib via Bash. "
                "Saves charts to files/charts/ and writes a data summary to files/data/."
            ),
            tools=["Glob", "Read", "Bash", "Write"],
            prompt=data_analyst_prompt,
            model=BEDROCK_MODEL_ID,
        ),
        "report-writer": AgentDefinition(
            description=(
                "Use this agent when you need to create a formal research report document. "
                "Reads findings from files/research_notes/, data from files/data/, "
                "and charts from files/charts/, then synthesizes into a PDF report in "
                "files/reports/ using reportlab. After generating the PDF, uploads it to "
                "the S3 output bucket using the AWS CLI."
            ),
            tools=["Skill", "Write", "Glob", "Read", "Bash"],
            prompt=report_writer_prompt,
            model=BEDROCK_MODEL_ID,
        ),
    }

    hooks = {
        "PreToolUse": [
            HookMatcher(matcher=None, hooks=[tracker.pre_tool_use_hook])
        ],
        "PostToolUse": [
            HookMatcher(matcher=None, hooks=[tracker.post_tool_use_hook])
        ],
    }

    # Append instruction to produce a text summary for MCP callers
    summary_instruction = (
        "\n\nIMPORTANT: After all sub-agents have completed their work, "
        "write a comprehensive markdown summary of the research findings directly "
        "in your final response. Include key facts, data points, and conclusions. "
        "This text will be returned directly to the user."
    )

    return ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        setting_sources=["project"],
        system_prompt=lead_agent_prompt + summary_instruction,
        allowed_tools=["Task"],
        agents=agents,
        hooks=hooks,
        model=BEDROCK_MODEL_ID,
        stderr=_stderr_logger,
        env=_get_bedrock_env(),
    )


async def run_query(query: str, session_id: str | None = None) -> AsyncIterator[dict]:
    """
    Run a single research query and yield event dicts for the caller.

    Each yielded dict has the shape:
        {"type": "text" | "tool_use" | "result" | "done", "content": str}

    Used by server.py to stream SSE responses.
    """
    transcript_file, session_dir = setup_session()
    transcript = TranscriptWriter(transcript_file)
    tracker = SubagentTracker(transcript_writer=transcript, session_dir=session_dir)

    options = _make_options(transcript, tracker)

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt=query)

            async for msg in client.receive_response():
                if type(msg).__name__ == "AssistantMessage":
                    for block in getattr(msg, "content", []):
                        block_type = type(block).__name__
                        if block_type == "TextBlock":
                            yield {"type": "text", "content": block.text}
                        elif block_type == "ToolUseBlock":
                            yield {
                                "type": "tool_use",
                                "content": f"[{block.name}] {json.dumps(getattr(block, 'input', {}))[:200]}",
                            }
    finally:
        transcript.write("\n\nDone.\n")
        transcript.close()
        tracker.close()

    yield {"type": "done", "content": ""}


async def chat() -> None:
    """Interactive CLI entry point (local dev / debugging only)."""
    transcript_file, session_dir = setup_session()
    transcript = TranscriptWriter(transcript_file)
    tracker = SubagentTracker(transcript_writer=transcript, session_dir=session_dir)
    options = _make_options(transcript, tracker)

    print("\n" + "=" * 50)
    print("  Research Agent  (Bedrock / Opus 4.6)")
    print("=" * 50)
    print("\nType your research query, or 'exit' to quit.\n")

    try:
        async with ClaudeSDKClient(options=options) as client:
            while True:
                try:
                    user_input = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input or user_input.lower() in ("exit", "quit", "q"):
                    break

                transcript.write_to_file(f"\nYou: {user_input}\n")
                await client.query(prompt=user_input)
                transcript.write("\nAgent: ", end="")

                async for msg in client.receive_response():
                    if type(msg).__name__ == "AssistantMessage":
                        process_assistant_message(msg, tracker, transcript)

                transcript.write("\n")
    finally:
        transcript.write("\n\nGoodbye!\n")
        transcript.close()
        tracker.close()
        print(f"\nSession logs saved to: {session_dir}")


if __name__ == "__main__":
    asyncio.run(chat())
