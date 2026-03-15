#!/usr/bin/env node
/**
 * Pi Agent (badlogic/pi-mono) adapter for SWE-bench evaluation.
 *
 * The base framework that oh-my-pi forks from. Uses str_replace-style
 * editing (no hashline), but has extensible skills system, sub-agents,
 * and context management. Useful as a control to isolate the effect of
 * oh-my-pi's hash-anchored edits vs the base Pi scaffolding.
 *
 * Drives Pi via RPC mode (stdin JSON -> stdout events).
 *
 * Output: JSON on last line: {"pass": bool, "turns": int, "tokens": int, "fix_generated": bool}
 */

import { spawn, execSync } from "child_process";
import { writeFileSync, existsSync, mkdirSync } from "fs";
import { join } from "path";

const args = process.argv.slice(2);
const workspace = args[args.indexOf("--workspace") + 1];
const model = args[args.indexOf("--model") + 1];
const issue = args[args.indexOf("--issue") + 1];
const testCmd = args[args.indexOf("--test-cmd") + 1];
const maxTurns = parseInt(args[args.indexOf("--max-turns") + 1] || "30");
const endpoint = process.env.OPENAI_BASE_URL || "http://localhost:9000/v1";
const apiKey = process.env.OPENAI_API_KEY || "dummy";

// Write models.json config for local vLLM provider
const modelsConfig = {
  providers: {
    "local-vllm": {
      type: "openai-compatible",
      baseURL: endpoint,
      apiKey: apiKey,
      models: {
        [model]: {
          name: model,
          limit: { context: 32768, output: 4096 },
        },
      },
    },
  },
};

const configDir = join(workspace, ".pi");
try {
  mkdirSync(configDir, { recursive: true });
  writeFileSync(join(configDir, "models.json"), JSON.stringify(modelsConfig, null, 2));
} catch (e) {
  console.log(JSON.stringify({ pass: false, turns: 0, tokens: 0, fix_generated: false, error: `config write failed: ${e.message}` }));
  process.exit(0);
}

const prompt = [
  `You are a coding agent fixing a bug in this repository.`,
  ``,
  `Fix this issue:`,
  issue,
  ``,
  `After fixing, verify with: ${testCmd}`,
  ``,
  `Use the edit tool for targeted changes. Do not rewrite entire files.`,
].join("\n");

// Launch Pi in RPC mode
const pi = spawn("pi", [
  "--mode", "rpc",
  "--no-session",
  "--model", `local-vllm:${model}`,
  "--cwd", workspace,
], {
  cwd: workspace,
  stdio: ["pipe", "pipe", "pipe"],
  env: { ...process.env, HOME: process.env.HOME },
});

let turns = 0;
let tokens = 0;

pi.stdout.on("data", (data) => {
  const lines = data.toString().split("\n").filter(Boolean);
  for (const line of lines) {
    try {
      const event = JSON.parse(line);
      if (event.type === "tool_call" || event.type === "tool_use") {
        turns++;
      }
      if (event.usage) {
        tokens += event.usage.total_tokens || 0;
      }
    } catch {
      // Non-JSON output, skip
    }
  }
});

pi.stderr.on("data", () => {}); // drain stderr

// Send the prompt
const request = JSON.stringify({ id: "swe-1", type: "prompt", message: prompt });
pi.stdin.write(request + "\n");

// Set timeout
const timeout = setTimeout(() => {
  pi.kill("SIGTERM");
}, maxTurns * 20 * 1000);

pi.on("close", () => {
  clearTimeout(timeout);

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
});
