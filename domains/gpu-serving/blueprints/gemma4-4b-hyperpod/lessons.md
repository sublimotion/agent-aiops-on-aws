---
model: gemma-4-E4B
engine: vllm
hardware: ml.g5.4xlarge (A10G 24GB)
outcome: success
failure_categories:
  - image_compatibility
  - disk_pressure
  - kubeconfig_context_switch

learn_commands:
  - 'mdc learn gemma-4-E4B vllm "Use vllm/vllm-openai:gemma4 image — standard tags lack gemma4 model_type support"'
  - 'mdc learn gemma-4-E4B vllm "Model is ungated (no HF_TOKEN needed). head_dim heterogeneity (256/512) forces TRITON_ATTN backend"'
  - 'mdc learn gemma-4-E4B vllm "tool_choice:required works perfectly (100% BFCL). auto mode outputs tools in content text instead of structured format"'
  - 'mdc learn gemma-4-E4B vllm "15.16 GiB GPU memory BF16, 53s load, 47s compile. 3.58 GiB KV cache, 39K tokens, ~6x concurrency at 32K"'
  - 'gpu-infra learn -c platform "HyperPod stale disk-pressure: restart kubelet via kubectl debug node to clear after large image pulls"'
  - 'gpu-infra learn -c platform "HyperPod GPU pods need 3 tolerations: RestrictedNode, nvidia.com/gpu, disk-pressure"'
---

# Gemma 4 E4B — Lessons Learned

## Lesson #1: Gemma 4 requires dedicated vLLM image
**Context**: Deploying google/gemma-4-E4B-it on vLLM
**Observation**: Both vLLM v0.19.0 (latest stable) and v0.8.5.post1 fail with `The checkpoint you are trying to load has model type 'gemma4' but Transformers does not recognize this architecture`. The bundled transformers library doesn't know about `gemma4` model_type.
**Rule**: Use `vllm/vllm-openai:gemma4` dedicated image for Gemma 4 models, not `latest` or versioned tags.
**Why**: Gemma 4 support requires a newer transformers build not yet in stable releases.

## Lesson #2: HyperPod stale disk-pressure taint
**Context**: Multiple vLLM image pulls (~10GB each) on 100GB EBS root volume
**Observation**: Node reported DiskPressure=True even with 50GB (50%) free space. Kubelet eviction threshold is 10%, so pressure shouldn't trigger. The taint persisted across pod cycles, causing an eviction storm (100+ evicted pods in 2 minutes).
**Rule**: If HyperPod GPU node shows disk-pressure with >20% free, restart kubelet: `kubectl debug node/<name> --image=busybox -- sh -c "chroot /host systemctl restart kubelet"`. This clears the stale condition.
**Why**: Kubelet's disk-pressure check can get stuck after transient disk spikes during large image pulls, even after containerd GC reclaims space.

## Lesson #3: HyperPod EKS requires 3 tolerations for GPU pods
**Context**: Scheduling vLLM pod on ml.g5.4xlarge HyperPod node
**Observation**: Pod stays Pending without tolerations. HyperPod GPU nodes have taints: `sagemaker.amazonaws.com/RestrictedNode:NoSchedule` and `nvidia.com/gpu:NoSchedule`. System nodes may also have `node.kubernetes.io/disk-pressure:NoSchedule`.
**Rule**: Always include all three tolerations in GPU pod specs on HyperPod EKS.
**Why**: HyperPod adds its own RestrictedNode taint beyond the standard nvidia.com/gpu taint.

## Lesson #4: Gemma 4 tool calling parser behavior
**Context**: Testing tool calling with `--tool-call-parser pythonic --enable-auto-tool-choice`
**Observation**: With `tool_choice: "auto"`, the model outputs tool calls in content text (`call:get_weather{city:San Francisco}`) instead of structured `tool_calls` JSON. With `tool_choice: "required"`, structured output works perfectly (10/10 accuracy).
**Rule**: Use `tool_choice: "required"` for structured tool call output with Gemma 4. The `pythonic` parser may not fully match Gemma 4's auto tool format.
**Why**: Gemma 4 uses a different tool calling format than Python-style function calls that the pythonic parser expects.

## Lesson #5: kubeconfig context switching in multi-cluster environments
**Context**: Working with both inference-eks-v132 (us-east-1) and gemma4-mistral4-eks (us-east-2)
**Observation**: kubectl context silently switched to us-east-2 cluster mid-session, causing "not found" errors. Some aws/kubectl commands update the default context as a side effect.
**Rule**: Always verify `kubectl config current-context` before critical operations. Pin context with `kubectl config use-context <arn>` after any AWS CLI command.
**Why**: `aws eks update-kubeconfig` and similar commands can change the active context.

## Lesson #6: Gemma 4 E4B is ungated
**Context**: Deploying google/gemma-4-E4B-it
**Observation**: Unlike Gemma 3 (gated, requires HF token), Gemma 4 models are fully ungated. `gated: False` in API metadata, config.json returns 307 redirect without auth.
**Rule**: No HF_TOKEN needed for Gemma 4 models. Can download directly in pod without secrets.
**Why**: Google changed the licensing approach for Gemma 4.

## Lesson #7: Gemma 4 E4B quality exceeds expectations for 4B model
**Context**: Benchmarking google/gemma-4-E4B-it on A10G
**Observation**: 100% BFCL tool calling (10/10), 100% code generation (5/5), 100% vision tasks (4/4). Throughput ~35-42 tok/s single request, TTFT p50 ~60-150ms across contexts. Model loads in 53s, uses 15.16 GiB GPU memory (BF16).
**Rule**: Gemma 4 E4B is a strong candidate for lightweight serving — tool calling quality matches much larger models.
**Why**: head_dim heterogeneity (256/512) and hybrid attention (sliding+global) are architectural improvements that maintain quality at small scale.
