#!/usr/bin/env node
/**
 * OpenCode (anomalyco/opencode) adapter for SWE-bench evaluation.
 *
 * Uses the OpenCode SDK: starts a local server, creates a session,
 * and drives it programmatically. Supports 75+ LLM providers via
 * @ai-sdk/openai-compatible config.
 *
 * Output: JSON on last line: {"pass": bool, "turns": int, "tokens": int, "fix_generated": bool}
 */

import { execSync } from "child_process";
import { existsSync, writeFileSync, mkdirSync } from "fs";
import { join } from "path";

const args = process.argv.slice(2);
const workspace = args[args.indexOf("--workspace") + 1];
const model = args[args.indexOf("--model") + 1];
const issue = args[args.indexOf("--issue") + 1];
const testCmd = args[args.indexOf("--test-cmd") + 1];
const maxTurns = parseInt(args[args.indexOf("--max-turns") + 1] || "30");
const endpoint = process.env.OPENAI_BASE_URL || "http://localhost:9000/v1";
const apiKey = process.env.OPENAI_API_KEY || "dummy";

const fail = (error) => {
  console.log(JSON.stringify({ pass: false, turns: 0, tokens: 0, fix_generated: false, error }));
  process.exit(0);
};

// Write opencode.json config for local vLLM provider
const opencodeConfig = {
  provider: {
    "local-vllm": {
      npm: "@ai-sdk/openai-compatible",
      name: "Local vLLM",
      options: { baseURL: endpoint },
      models: {
        [model]: {
          name: model,
          limit: { context: 32768, output: 4096 },
        },
      },
    },
  },
};

const configDir = join(workspace, ".opencode");
try {
  mkdirSync(configDir, { recursive: true });
  writeFileSync(join(configDir, "config.json"), JSON.stringify(opencodeConfig, null, 2));
} catch (e) {
  fail(`config write failed: ${e.message}`);
}

let createOpencodeServer, createOpencodeClient;
try {
  ({ createOpencodeServer, createOpencodeClient } = await import("@opencode-ai/sdk"));
} catch (e) {
  fail(`import failed: ${e.message}`);
}

const prompt = [
  `You are a coding agent fixing a bug in this repository.`,
  ``,
  `Fix this issue:`,
  issue,
  ``,
  `After fixing, verify with: ${testCmd}`,
  ``,
  `Use edit tools for targeted changes. Do not rewrite entire files.`,
].join("\n");

let turns = 0;
let tokens = 0;

try {
  const server = await createOpencodeServer({ cwd: workspace });
  const client = createOpencodeClient({ baseUrl: server.url });

  const session = await client.session.create();

  await client.session.prompt({
    path: { id: session.data.id },
    body: {
      parts: [{ type: "text", text: prompt }],
      model: `local-vllm:${model}`,
    },
  });

  // Fetch session messages to count tool calls
  try {
    const messages = await client.session.messages({ path: { id: session.data.id } });
    if (messages.data) {
      for (const msg of messages.data) {
        if (msg.role === "assistant" && msg.tool_calls?.length > 0) {
          turns += msg.tool_calls.length;
        }
        if (msg.usage) {
          tokens += msg.usage.total_tokens || 0;
        }
      }
    }
  } catch {
    // Message counting is best-effort
  }

  // Shutdown server
  try { server.close?.(); } catch {}
} catch (e) {
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
