# GLM-5 llm-d Lessons

## Session: 2026-03-07 (B200 p6-b200.48xlarge, vLLM + llm-d)

### #1: vLLM glm47 parser is required for GLM-5 structured tool calling
SGLang outputs tool calls as `<tool_call>...</tool_call>` XML in the content field. vLLM's `--tool-call-parser glm47` produces proper structured `tool_calls` array with `function.name` and `function.arguments`. This is critical for BFCL eval, agent frameworks, and llm-d routing.

### #2: DeepGEMM JIT cold start on B200 is 16 minutes
Breakdown: model load 77s, JIT compilation 200s (117 kernels for sm_100f), warmup 200s (2259 kernels), torch.compile 509s, CUDA graph capture 245s (51 graphs). Kernels cached at `/root/.cache/vllm/deep_gemm/cache/` and AOT functions at `/root/.cache/vllm/torch_aot_compile/`. Second start should be much faster.

### #3: vLLM reaches 91% of SGLang HiCache throughput
Peak: 2,374.6 tok/s at 64 concurrent vs SGLang HiCache 2,602 tok/s. The 9% gap is acceptable given vLLM's structured tool calling and reasoning support.

### #4: GLM-5 reasoning parser works on vLLM
`--reasoning-parser glm45` extracts chain-of-thought reasoning into the `reasoning` field of the response. Every response includes reasoning when tool calls are involved. This is a unique feature not available on SGLang.

### #5: InferencePool GA API (v1) has different schema than x-k8s.io (v1alpha2)
- v1 uses `endpointPickerRef` (not `extensionRef`)
- v1 uses `targetPorts: [{number: 8000}]` (not `targetPortNumber: 8000`)
- v1 uses `selector.matchLabels` (not flat `selector`)
- EPP v1.3.1 watches the GA group `inference.networking.k8s.io`, not the experimental group

### #6: EPP v1.3.1 requires explicit configuration
- `--config-file` or `--config-text` is mandatory (no defaults)
- Flags use kebab-case: `--pool-name`, `--grpc-port`, `--secure-serving`
- RBAC: service account needs list/watch on pods, inferencepools (GA group), inferencemodelrewrites, inferenceobjectives, inferencepoolimports (all in x-k8s.io group)
- Image: `registry.k8s.io/gateway-api-inference-extension/epp:v1.3.1`

### #7: Envoy Gateway needs explicit GatewayClass
When installing with `--skip-crds`, the GatewayClass is not created. Must apply:
```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: eg
spec:
  controllerName: gateway.envoyproxy.io/gatewayclass-controller
```
Also: Gateway `allowedRoutes.namespaces.from: All` for cross-namespace HTTPRoutes.

### #8: Redis can run on GPU nodes with taint toleration
System nodes (m5.xlarge) often lack CPU/memory for Redis. Adding `tolerations: [{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}]` lets Redis schedule on GPU nodes which have abundant CPU/RAM beyond what vLLM uses (80% CPU, 90% memory free on p6-b200).

### #9: BFCL accuracy is 100% with glm47 parser
50/50 scenarios passed across 10 categories: simple calls, multi-param, numeric, array, boolean, tool selection, no-tool, nested objects, date/time, complex multi-step. All with correct function name and argument extraction.

### #10: vLLM MTP speculative decode works with FlashMLASparse
`--speculative-config.method mtp --speculative-config.num_speculative_tokens 1` enables Multi-Token Prediction. Uses PIECEWISE CUDA graph mode (FULL_AND_PIECEWISE not supported with spec-decode for DeepseekV32IndexerBackend). KV cache block size forced to 64 for FlashMLASparse.

### #11: EnvoyExtensionPolicy requires Envoy Gateway CRD install + controller restart
The `envoyextensionpolicies.gateway.envoyproxy.io` CRD is not installed when Envoy Gateway is deployed with `--skip-crds`. Must install from chart: `helm show crds ... | kubectl apply -f -`. After installing the CRD, the Envoy Gateway controller must be restarted (`kubectl rollout restart`) to begin watching the new resource type. Without restart, policies are silently ignored.

### #12: ext-proc messageTimeout must be increased for LLM inference
Default `messageTimeout` is 200ms — way too short for LLM serving. Set to at least 30s in the EnvoyExtensionPolicy. Without this, every request gets `ext_proc_error_per-message_timeout_exceeded`. With `failOpen: true`, requests bypass EPP and go directly to backend.

### #13: EPP v1.3.1 ext-proc accepts connections but doesn't process with empty scheduler config
With an empty plugins list (no scorers), EPP's ext-proc handler accepts gRPC streams from Envoy but never responds. The `max-score-picker` default plugin can't pick endpoints without scores. `LoadAwareScorer` type is not registered in v1.3.1 — the available plugin types need to be discovered from the EPP source. Full ext-proc integration requires: (1) correct scorer plugin names, (2) possibly InferenceModel CRDs, (3) vLLM metrics integration for prefix-cache state reporting.
