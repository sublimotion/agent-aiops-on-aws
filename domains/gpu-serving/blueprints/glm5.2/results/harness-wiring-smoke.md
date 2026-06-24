# GLM-5.2 agent-harness wiring smoke (2026-06-23)

Endpoint: SGLang v0.5.13.post1-cu130, GLM-5.2-FP8, TP8 B200, --context-length 131072,
prefix cache ON. Reached from laptop via kubectl port-forward localhost:30000.

| Harness | API path | Result | Notes |
|---------|----------|--------|-------|
| **OpenCode** | /v1/chat/completions | ✅ **WORKS** | vllm provider in opencode.json (@ai-sdk/openai-compatible). Read+edit tools, fixed the bug correctly. Event schema matches baseline traces. Needs context limit 131072 + output cap 8192 (OpenCode sys prompt ~65K tokens alone). |
| **Codex** | /v1/responses | ❌ blocked | codex-cli 0.136.0 dropped wire_api="chat"; now REQUIRES /v1/responses. SGLang registers /v1/responses but it returns EMPTY for GLM-5.2 (non-stream + stream both). Needs a responses↔chat proxy (old memory: 4 patches). |
| **Claude Code** | /v1/messages | ❌ not attempted | SGLang doesn't expose /v1/messages at all. Needs LiteLLM anthropic-passthrough shim. |

GLM-5.2 trait: reasoning-first — emits reasoning_content before content/tool_calls; tight max_tokens
starves the answer. Use generous budgets (harnesses already do).

DECISION: OpenCode is the clean, validated path. Codex + Claude Code each need a translation shim
(real build). Recommend: run the full 46-issue comparison on OpenCode (apples-to-apples with the
existing opus/sonnet/haiku OpenCode traces — SAME harness, cleanest comparison), add Codex/CC later
behind a LiteLLM proxy if multi-harness coverage is wanted.

## Update — LiteLLM shim attempt (2026-06-23)

LiteLLM 1.82.5 proxy (litellm-glm52.yaml) exposes GLM-5.2 under /v1/messages (Anthropic) + wildcard.
- Single-turn /v1/messages: ✅ works (returns thinking + content blocks).
- **Claude Code multi-turn: ❌ FAILS** with SGLang 400 (23 validation errors). Root cause: GLM-5.2 is
  reasoning-first → emits `thinking` content blocks. On turn 2+, LiteLLM's Anthropic→OpenAI adapter
  passes the assistant `{'type':'thinking',...}` block through VERBATIM as message content. SGLang's
  OpenAI schema rejects `thinking` parts (only text/image_url/etc allowed). LiteLLM also mis-tags the
  assistant turn role. This is a LiteLLM adapter gap × GLM-5.2 reasoning trait — needs an adapter patch
  or a normalizing proxy that strips/converts thinking blocks before the OpenAI call.
- Also: ~/.claude/settings.json pins CLAUDE_CODE_USE_BEDROCK=1 (overrides inline env); use
  `claude --settings <file>` with CLAUDE_CODE_USE_BEDROCK=0 to route to the proxy.

## Verdict
- OpenCode: ✅ clean, validated, SAME harness as the baseline Opus/Sonnet/Haiku traces → best comparison.
- Codex: ❌ needs /v1/responses (SGLang's returns empty) — proxy/patch required.
- Claude Code: ❌ LiteLLM thinking-block translation breaks multi-turn — adapter fix required.
RECOMMENDATION: run the 46-issue comparison on OpenCode now (the apples-to-apples path); treat Codex +
Claude Code as a separate "build the shims" task, not a blocker for the agent-capability result.

## Update 2 — root causes nailed (2026-06-23, after user push-back on Codex)

Re-tested with stable tunnel. The earlier "Codex blocked / responses empty" was CONTAMINATED by a dead
port-forward. Corrected findings:
- **SGLang /v1/responses WORKS** (curl, stream + non-stream). My "empty" reads were the dead tunnel.
- **Codex direct→SGLang /v1/responses FAILS** for a real reason: SGLang's responses endpoint only
  accepts tools of type web_search_preview/code_interpreter, REJECTS Codex's `function`-type tools (422).
  Fix: route Codex → LiteLLM (/v1/responses) → SGLang (/v1/chat/completions, glm47 tools). Needs
  env_key="OPENAI_API_KEY" in the codex provider so it forwards the bearer to LiteLLM's master_key.
- **Port-forward is the saboteur**: rock-solid for plain/non-streaming (OpenCode: 20/20 health over 40s),
  but DESTABILIZES under streaming-proxy SSE bursts (Codex /v1/responses, Claude Code). Both streaming
  harnesses repeatedly hit "Cannot connect to host 127.0.0.1:30000" mid-run as the pf reconnects.
- **DECISION: run the harness batch ON THE NODE** (runner pod, harnesses → localhost:30000 directly,
  LiteLLM colocated). Eliminates the tunnel — same approach that produced the verifier-reward baseline
  traces. Node IP 10.0.19.164 is private (no direct laptop path).

Config artifacts: ~/.config/opencode/opencode.json (vllm provider), ~/.codex/config.toml
([model_providers.glm52], wire_api=responses, env_key), litellm-glm52.yaml (proxy),
/tmp/cc-glm52-settings.json (Claude Code, Bedrock off). Driver copied to scripts/run_agent_compare.py.

## Update 3 — ON-NODE pilot (3 issues × 3 harnesses), 2026-06-24
Runner pod on B200 node (hostNetwork → localhost:30000 direct, NO tunnel). All 3 harnesses installed
(opencode 1.17.9, codex 0.142.0, claude 2.1.187), litellm in-pod.

| harness | pilot result | root cause |
|---------|-------------|------------|
| **OpenCode** | ✅ **3/3 fix + edit, 9-48 tools, traces captured** | direct chat-completions — works perfectly |
| Codex | ❌ all fail | LiteLLM /v1/responses → SGLang returns 500 (reasoning/thinking-block translation). Not stdin, not connectivity. |
| Claude Code | ❌ all fail | (1) --dangerously-skip-permissions blocked under root → FIXED with IS_SANDBOX=1. (2) multi-turn /v1/messages → 400 (thinking-block translation, same family as Codex). |

**ROOT CAUSE (both proxy harnesses):** LiteLLM's Anthropic-messages + OpenAI-responses adapters can't
translate GLM-5.2's `thinking`/reasoning content blocks into SGLang's OpenAI chat schema on multi-turn
→ 400/500.

### CORRECTION (2026-06-24) — the LiteLLM shim was UNNECESSARY; operator error, not a real block
The Codex/Claude-Code "block" above was a CONSEQUENCE OF INTRODUCING LiteLLM, not a genuine limitation.
SGLang serves the **Anthropic Messages API `/v1/messages` natively** (auto-registered, no flag —
docs.sglang.io/docs/basic_usage/anthropic_api), and the docs' own example is literally GLM-5.2-FP8 with
`--tool-call-parser glm47 --reasoning-parser glm45`. Claude Code should have pointed straight at SGLang:
- `ANTHROPIC_BASE_URL=http://127.0.0.1:30000` (server root, **NO `/v1`** — the Anthropic SDK appends
  `/v1/messages` itself). I wrongly pointed it at the LiteLLM proxy on :4000.
- No LiteLLM, no thinking-block translation layer, no shim. The 400/500s were LiteLLM's adapter, not SGLang
  or GLM-5.2.
- Codex `/v1/responses`: SGLang returned 200 directly earlier but rejected Codex's `function`-type tools
  (schema wants web_search_preview/code_interpreter) — that one may still need work, but is separate from
  the LiteLLM red herring. (vLLM also serves `/v1/messages` + `/v1/responses` natively as an alternative.)
- **Why the mistake:** anchored on a stale memory ("Claude Code needs the vLLM Anthropic patch; SGLang has
  no messages endpoint") from the Mistral/Qwen era; did NOT re-verify against current SGLang docs before
  building a shim — violating the project's own "re-verify blockers against the live tracker" rule.
- **Correct retry (node down, not re-run):** drop LiteLLM entirely; point Claude Code at SGLang
  `:30000` (no `/v1`); re-test Codex against SGLang `/v1/responses` directly. Likely both work native,
  making a clean 3-harness comparison achievable without any proxy.

**DECISION (still valid):** the 46-issue capability comparison ran on **OpenCode** — it works AND it's the
SAME harness as the Opus/Sonnet/Haiku baseline traces, so it's the cleanest apples-to-apples comparison.
The Codex/Claude-Code arms are a documented follow-up via NATIVE SGLang endpoints (not LiteLLM).
