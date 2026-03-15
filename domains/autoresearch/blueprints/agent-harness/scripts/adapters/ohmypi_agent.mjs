#!/usr/bin/env node
/**
 * oh-my-pi (can1357/oh-my-pi) adapter for SWE-bench evaluation.
 *
 * Key differentiator: hash-anchored LINE#ID edit tool that eliminates
 * exact-text-match failures. Also has LSP, ast_grep, Python kernel,
 * sub-agents, and context compaction.
 *
 * Drives omp via RPC mode (stdin JSON → stdout events).
 *
 * Output: JSON on last line: {"pass": bool, "turns": int, "tokens": int, "fix_generated": bool}
 */

import { spawn, execSync } from "child_process";
import { writeFileSync, existsSync } from "fs";
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

const configDir = join(workspace, ".omp");
const configPath = join(configDir, "models.json");
try {
  execSync(`mkdir -p ${configDir}`);
  writeFileSync(configPath, JSON.stringify(modelsConfig, null, 2));
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

// Launch omp in RPC mode
const omp = spawn("omp", [
  "--mode", "rpc",
  "--no-session",
  "--model", `local-vllm:${model}`,
  "--tools", "read,edit,write,find,grep,bash,python,task",
  "--cwd", workspace,
], {
  cwd: workspace,
  stdio: ["pipe", "pipe", "pipe"],
  env: { ...process.env, HOME: process.env.HOME },
});

let stdout = "";
let turns = 0;
let tokens = 0;
let done = false;

omp.stdout.on("data", (data) => {
  const lines = data.toString().split("\n").filter(Boolean);
  for (const line of lines) {
    try {
      const event = JSON.parse(line);
      // Count tool_use events as turns
      if (event.type === "tool_call" || event.type === "tool_use") {
        turns++;
      }
      // Track token usage if reported
      if (event.usage) {
        tokens += event.usage.total_tokens || 0;
      }
      // Check for completion
      if (event.type === "response" || event.type === "done" || event.type === "error") {
        done = true;
      }
    } catch {
      // Non-JSON output, skip
    }
  }
  stdout += data.toString();
});

let stderr = "";
omp.stderr.on("data", (data) => {
  stderr += data.toString();
});

// Send the prompt
const request = JSON.stringify({ id: "swe-1", type: "prompt", message: prompt });
omp.stdin.write(request + "\n");

// Set timeout
const timeout = setTimeout(() => {
  omp.kill("SIGTERM");
  done = true;
}, maxTurns * 20 * 1000); // ~20s per turn max

omp.on("close", (code) => {
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
