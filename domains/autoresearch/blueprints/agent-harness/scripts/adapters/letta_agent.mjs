#!/usr/bin/env node
/**
 * Letta Code adapter for SWE-bench evaluation (self-hosted).
 *
 * Key differentiator: memory-first agent with persistent memory across sessions,
 * skill learning, and sleeptime background processing. Uses Letta server (Docker)
 * as intermediary to local vLLM.
 *
 * Requires:
 *   - Letta server running: docker run -p 8283:8283 letta/letta-server
 *   - LETTA_BASE_URL=http://localhost:8283
 *   - LLM provider configured on the server to point at local vLLM
 *
 * Output: JSON on last line: {"pass": bool, "turns": int, "tokens": int, "fix_generated": bool}
 */

import { execSync } from "child_process";
import { existsSync } from "fs";
import { join } from "path";

const args = process.argv.slice(2);
const workspace = args[args.indexOf("--workspace") + 1];
const model = args[args.indexOf("--model") + 1];
const issue = args[args.indexOf("--issue") + 1];
const testCmd = args[args.indexOf("--test-cmd") + 1];

const fail = (error) => {
  console.log(JSON.stringify({ pass: false, turns: 0, tokens: 0, fix_generated: false, error }));
  process.exit(0);
};

let prompt, createAgent, createSession;
try {
  ({ prompt, createAgent, createSession } = await import("@letta-ai/letta-code-sdk"));
} catch (e) {
  fail(`import failed: ${e.message}`);
}

const taskPrompt = [
  `You are a coding agent fixing a bug in this repository at ${workspace}.`,
  ``,
  `Fix this issue:`,
  issue,
  ``,
  `After fixing, verify with: ${testCmd}`,
  ``,
  `Use edit tools for targeted changes. Do not rewrite entire files.`,
  `Start by reading relevant files to understand the codebase.`,
].join("\n");

let turns = 0;
let tokens = 0;

try {
  // Create a fresh agent for this issue (no cross-issue memory leakage)
  const agentId = await createAgent({
    persona: "You are an expert software engineer who fixes bugs methodically.",
  });

  await using session = createSession(agentId);

  await session.send(taskPrompt);

  for await (const msg of session.stream()) {
    if (msg.type === "tool_call" || msg.type === "tool_use") {
      turns++;
    }
    if (msg.type === "usage" && msg.total_tokens) {
      tokens += msg.total_tokens;
    }
  }
} catch (e) {
  // Don't exit — still check for fix and tests below
  if (turns === 0) {
    fail(e.message?.slice(0, 200) || "unknown error");
  }
}

// Check for fix via git diff
let fixGenerated = false;
try {
  const diff = execSync("git diff", { cwd: workspace, encoding: "utf-8" });
  fixGenerated = diff.trim().length > 0;
} catch {}

// Check tests
let testsPass = false;
try {
  const venv = join(workspace, ".venv", "bin", "activate");
  const cmd = existsSync(venv) ? `source ${venv} && ${testCmd}` : testCmd;
  const result = execSync(cmd, {
    cwd: workspace,
    encoding: "utf-8",
    timeout: 120000,
    shell: "/bin/bash",
  });
  const lower = result.toLowerCase();
  if (lower.includes("passed") && !lower.split("passed")[0].includes("failed")) {
    testsPass = true;
  }
  if ((lower.includes("\nok") || lower.trimEnd().endsWith("ok")) && !lower.includes("fail")) {
    testsPass = true;
  }
} catch {}

console.log(JSON.stringify({
  pass: testsPass,
  turns,
  tokens,
  fix_generated: fixGenerated,
}));
