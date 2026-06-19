# Autoresearch Spec: VLA CV-Tool Distillation

## Status: DRAFT

## Overview

Build a pipeline where a large VLM agent (Qwen2.5-VL-72B or Claude) equipped with specialized CV tools (detection, segmentation, depth estimation) solves robotic manipulation tasks in simulation, then distill those agentic traces into a compact Qwen-based VLA model (Qwen2.5-VL-3B + action head). The key hypothesis: intermediate CV-tool outputs as auxiliary supervision produce a better student VLA than standard behavioral cloning on actions alone.

**Core hypothesis**: A VLM teacher calling CV tools generates richer training signal than raw demonstrations. The student VLA trained on these traces — including intermediate perception outputs (bounding boxes, masks, depth, affordances) as auxiliary losses — will outperform a student trained only on (image, language) → action pairs.

**Motivation**: The VLA distillation landscape (VLA-OPD, VITA-VLA, Shallow-pi) focuses on compressing existing VLAs. The trace generation landscape (GenSim, ViVLA) focuses on scaling demonstrations. Nobody has combined tool-augmented reasoning traces with multi-task auxiliary distillation. Our inference optimization experience (KV cache, FP8, prefix caching, TP strategies) directly transfers to serving both the teacher and student models.

**Why CV tools matter**: VANGUARD (2603.04277) showed VLMs have >50% spatial estimation errors without tool augmentation. The teacher agent needs grounded perception, not hallucinated spatial reasoning. By recording which tools the teacher calls and what they return, we create structured intermediate representations that the student can learn to predict internally — absorbing the tool capability into the weights.

**Connection to our work**: This spec bridges two bodies of knowledge:
1. **Inference optimization** (gpu-serving domain): FP8 quantization, prefix caching, HiCache, TP strategies — all transfer to VLA serving (see analysis in MEMORY.md)
2. **Agentic trace distillation** (autoresearch domain): verification-primitives showed agents compose tools effectively with checkpoint guidance (83.7% adoption, 58.3% SWE-bench pass). Same principle: give the teacher structured tools, collect traces, train the student.

**Depends on**: gpu-serving infrastructure (g7e or B300 for model serving), agent-harness eval methodology (trace collection patterns)

## Research Questions

1. **Does tool-augmented trace generation produce better VLA training data than raw demonstrations?** Compare: student trained on teacher traces (with CV tool outputs) vs student trained on scripted demonstrations (same tasks, same action labels, no reasoning).

2. **Do auxiliary perception losses improve action prediction?** Compare: student with (image, lang) → action only vs student with (image, lang) → (bboxes, masks, depth, action). Multi-task auxiliary heads should provide richer gradient signal (supported by PALM 2601.07060: affordance distillation → 91.8% on LIBERO-LONG).

3. **What's the optimal teacher-student gap?** Compare: Qwen2.5-VL-72B teacher → 3B student vs Qwen2.5-VL-7B teacher → 3B student vs Claude teacher → 3B student. Larger teachers may generate better traces but at higher cost.

4. **Which CV tools provide the most value as auxiliary targets?** Ablation: remove one tool at a time from teacher and corresponding auxiliary loss from student. Rank: detection > segmentation > depth > affordance (or discover different ordering).

5. **Does our inference optimization stack improve teacher throughput enough for practical trace generation?** Target: 10K traces in <$500 and <48 hours on available hardware. Prefix caching for shared scene descriptions, FP8 for the teacher, batched inference.

## Phases

### Phase 1: Teacher Agent & CV Tool Integration (Week 1-2)

**Goal**: Build the tool-augmented VLM teacher and validate it can solve manipulation tasks in simulation with grounded perception.

**Steps**:

1. **Select simulation environment**:
   - Primary: MuJoCo + robosuite (established manipulation benchmark, 8 task suites)
   - Fallback: LIBERO (standardized, 130 tasks across 5 suites, used by NORA/AnchorVLA4D/PALM)
   - Render: RGB images at 256x256, 2 camera views (front + wrist)

2. **Implement CV tool suite** (callable by teacher agent):

   **Tool 1: `detect_objects`**
   - Backend: GroundingDINO-T (open-vocab detection, runs on CPU or single GPU)
   - Input: RGB image + text query (e.g., "red block", "gripper", "target zone")
   - Output: list of (bbox, confidence, label)
   - Latency target: <100ms/call

   **Tool 2: `segment_object`**
   - Backend: SAM 2 (prompted by bbox from detect_objects)
   - Input: RGB image + bbox prompt
   - Output: binary mask + mask confidence
   - Latency target: <200ms/call

   **Tool 3: `estimate_depth`**
   - Backend: Depth Anything V2 (monocular depth, ViT-S for speed)
   - Input: RGB image
   - Output: depth map (H x W float32, metric depth in meters)
   - Latency target: <50ms/call

   **Tool 4: `estimate_affordance`**
   - Backend: Custom prompt to teacher VLM (not a separate model)
   - Input: RGB image + object mask + task description
   - Output: structured affordance (grasp_point_2d, approach_direction, contact_type)
   - This is the "reasoning" tool — the teacher VLM itself provides affordance judgment

3. **Build teacher agent loop**:
   ```
   for each task episode:
     obs = env.reset()
     while not done:
       # Teacher perceives via tools
       objects = detect_objects(obs.rgb, task.objects)
       masks = [segment_object(obs.rgb, obj.bbox) for obj in objects]
       depth = estimate_depth(obs.rgb)
       affordance = estimate_affordance(obs.rgb, masks[target], task.instruction)

       # Teacher reasons and plans
       action = teacher_vlm.plan(
         obs.rgb, task.instruction,
         tool_results={objects, masks, depth, affordance}
       )

       # Execute and record
       obs, reward, done = env.step(action)
       trace.append(obs, objects, masks, depth, affordance, action, reward)
   ```

4. **Validate teacher on 10 LIBERO tasks**:
   - Success rate target: >50% (teacher doesn't need to be perfect, just better than random)
   - If teacher can't solve tasks: add more tools, increase context, or switch to scripted oracle + tool annotation overlay

**Exit criteria**:
- 4 CV tools implemented and tested
- Teacher agent completes at least 5/10 LIBERO tasks
- Trace format defined: (rgb, objects, masks, depth, affordance, action, reward, reasoning_text)
- CV tool latency measured and within targets
- Teacher serving config documented (which instance, TP, quantization)

**Estimated cost**: ~$50-100 (API calls for teacher VLM, compute for CV tools)

### Phase 2: Trace Collection at Scale (Week 3-4)

**Goal**: Generate 10K+ successful traces across diverse manipulation tasks using the teacher agent, with full CV-tool intermediate outputs recorded.

**Steps**:

1. **Task selection** (LIBERO benchmark):
   - LIBERO-Spatial: 10 tasks (spatial reasoning)
   - LIBERO-Object: 10 tasks (object generalization)
   - LIBERO-Goal: 10 tasks (goal-conditioned)
   - LIBERO-Long: 10 tasks (long-horizon, 5-7 subtasks)
   - Total: 40 tasks, target 250 successful traces per task = 10,000 traces

2. **Teacher serving optimization** (apply our inference stack):
   - **Hardware**: g7e.24xlarge (4x RTX PRO 6000) or B300 if available
   - **Model**: Qwen2.5-VL-72B-Instruct FP8 (fits TP4 on g7e with ~24 GB/GPU weights, 72 GB KV headroom)
   - **Engine**: vLLM 0.19+ or SGLang 0.5.10+
   - **Prefix caching**: Robot system prompt + task description + tool definitions (~4K tokens shared across all steps within an episode). Expected 90%+ cache hit rate within episodes.
   - **KV cache dtype**: FP8 (doubles concurrent episode capacity)
   - **Chunked prefill**: 4096 tokens (low-latency for interactive simulation loop)
   - **Batching**: Run 4 episodes in parallel (1 per GPU with TP=1 for 72B... or TP4 for single instance). If 72B doesn't fit TP1, use Qwen2.5-VL-7B with TP1 × 4 parallel episodes.

3. **CV tool serving**:
   - GroundingDINO + SAM 2 + Depth Anything V2: all fit on CPU or share 1 GPU
   - Run as local services alongside simulation
   - Batch CV inference across parallel episodes where possible

4. **Trace filtering**:
   - Keep only successful episodes (task completed within step budget)
   - Quality filter: discard traces where teacher called >20 tool invocations per step (degenerate behavior)
   - Target: 10K successful traces from ~15-20K total episodes (50-67% success rate)

5. **Trace format** (per timestep within episode):
   ```json
   {
     "episode_id": "libero_spatial_003_ep_127",
     "timestep": 5,
     "task_instruction": "pick up the red block and place it on the blue plate",
     "rgb_front": "base64_or_path",
     "rgb_wrist": "base64_or_path",
     "tool_outputs": {
       "objects": [{"bbox": [x1,y1,x2,y2], "label": "red_block", "conf": 0.94}],
       "masks": [{"object": "red_block", "mask": "rle_encoded"}],
       "depth": "path_to_depth_npy",
       "affordance": {"grasp_point": [px, py], "approach": "top_down", "contact": "pinch"}
     },
     "reasoning": "The red block is at (0.3, 0.5) on the table. Gripper is above and to the left. Need to move right and down to align with grasp point.",
     "action": [dx, dy, dz, droll, dpitch, dyaw, gripper],
     "reward": 0.0,
     "done": false
   }
   ```

6. **Cost estimation**:
   - Teacher VLM: ~2 images + 2K tokens input + 200 tokens output per step
   - Average episode: ~50 steps
   - 15K episodes × 50 steps = 750K VLM calls
   - At Qwen2.5-VL-72B self-hosted on g7e ($16.57/hr): ~$200-400 for 48 hours
   - CV tools: CPU-only, negligible cost
   - Total: ~$200-500

**Exit criteria**:
- 10K+ successful traces collected across 40 tasks
- Trace format validated (all fields populated, actions executable in replay)
- Teacher success rate measured per task suite
- Inference throughput measured (traces/hour, cost/trace)
- Traces stored on NVMe or S3 for student training

**Estimated cost**: ~$200-500

### Phase 3: Student VLA Training (Week 5-7)

**Goal**: Train a compact Qwen2.5-VL-3B VLA on the collected traces, comparing standard behavioral cloning against auxiliary-supervised distillation.

**Steps**:

1. **Student architecture** (based on NORA):
   - **Backbone**: Qwen2.5-VL-3B (frozen or LoRA-tuned vision encoder + language model)
   - **Action head**: FAST+ tokenizer (frequency-space action tokens, from NORA 2504.19854) OR lightweight diffusion head (from AnchorVLA4D 2603.12730)
   - **Auxiliary heads** (the key innovation):
     - Detection head: predict object bboxes from visual features (1-layer MLP → bbox regression)
     - Segmentation head: predict object masks (lightweight decoder → binary mask)
     - Depth head: predict depth map (1-layer conv → depth regression)
     - Affordance head: predict grasp point + approach (MLP → 2D point + categorical)
   - **Loss**: `L = L_action + λ_det * L_detection + λ_seg * L_segmentation + λ_depth * L_depth + λ_aff * L_affordance`
   - Start with equal λ = 0.1, tune in Phase 3b

2. **Training configurations** (comparison matrix):

   | Config | Action supervision | Auxiliary supervision | Data source |
   |--------|-------------------|----------------------|-------------|
   | **A: BC-scripted** | Scripted demos (no teacher) | None | LIBERO default demos |
   | **B: BC-teacher** | Teacher traces (actions only) | None | Phase 2 traces |
   | **C: Aux-all** | Teacher traces | All 4 aux heads | Phase 2 traces |
   | **D: Aux-detect+depth** | Teacher traces | Detection + depth only | Phase 2 traces |
   | **E: Aux-affordance** | Teacher traces | Affordance only | Phase 2 traces |
   | **F: Reasoning-distill** | Teacher traces | Reasoning text (predict CoT) | Phase 2 traces |

3. **Training setup**:
   - **Hardware**: g7e.24xlarge (4 GPUs for data-parallel training) or single B300 GPU
   - **Framework**: PyTorch + HuggingFace Transformers + custom action head
   - **LoRA**: r=16, alpha=32, target: q_proj, v_proj, o_proj (following Unsloth patterns from finetuning-recipes)
   - **Batch size**: 32 (8 per GPU × 4 GPUs) or 128 on B300
   - **Epochs**: 50 over 10K traces (standard for LIBERO, following NORA)
   - **Optimizer**: AdamW, lr=2e-5, cosine schedule, warmup 5%
   - **Mixed precision**: bf16 (B300) or fp16 (g7e)

4. **Evaluation** (LIBERO benchmark):
   - **LIBERO-Spatial**: 10 tasks, 20 episodes each = 200 rollouts
   - **LIBERO-Object**: 10 tasks, 20 episodes each = 200 rollouts
   - **LIBERO-Goal**: 10 tasks, 20 episodes each = 200 rollouts
   - **LIBERO-Long**: 10 tasks, 20 episodes each = 200 rollouts
   - **Metric**: Average success rate across all 800 rollouts
   - **Baselines from literature**: OpenVLA 7B (~70% LIBERO avg), NORA 3B (~75%), PALM (~91.8% LIBERO-Long)

5. **Inference optimization for student** (apply our stack):
   - **FP8 quantization**: Qwen2.5-VL-3B FP8 → ~3 GB weights, 93 GB KV headroom on g7e
   - **CUDA graphs**: Pre-compiled action generation graph (no dynamic shapes in robot control)
   - **Prefix caching**: Scene description + task instruction cached across control steps
   - **Target frequency**: 15-25 Hz (latency budget: 40-67ms per action)
   - **Auxiliary heads**: Detached at inference (zero overhead — they're training-only)

**Exit criteria**:
- All 6 configs trained and evaluated
- Config C (Aux-all) vs Config B (BC-teacher): measures auxiliary supervision value
- Config B vs Config A: measures teacher trace value over scripted demos
- Student inference frequency measured on g7e (target: >15 Hz)
- Results table with success rates + 95% CIs across all LIBERO suites

**Estimated cost**: ~$100-300 (GPU training time, 4-8 hours per config × 6 configs)

### Phase 4: Ablation & Analysis (Week 8-9)

**Goal**: Understand which components drive performance gains and characterize the cost-performance frontier.

**Prerequisites**: Phase 3 shows Config C or D beats Config A by >5pp.

**Steps**:

1. **Auxiliary loss ablation**:
   - Remove each auxiliary head one at a time from Config C
   - Rank contribution: which perception output matters most for action prediction?
   - Hypothesis: affordance > depth > detection > segmentation (affordance is closest to action)

2. **Teacher model ablation**:
   - Qwen2.5-VL-72B teacher traces → 3B student (Phase 3 baseline)
   - Qwen2.5-VL-7B teacher traces → 3B student (cheaper teacher)
   - Claude Sonnet teacher traces → 3B student (different architecture)
   - Scripted oracle + CV-tool annotation overlay → 3B student (perfect actions, tool outputs from CV models on oracle trajectory)

3. **Data scaling**:
   - Train Config C on 1K, 2.5K, 5K, 10K traces
   - Plot learning curve: where does auxiliary supervision plateau?
   - Compare to BC-scripted scaling (does auxiliary supervision improve data efficiency?)

4. **Inference latency profiling**:
   - Profile student end-to-end: vision encoder → LLM backbone → action head
   - Identify bottleneck (likely vision encoder token count)
   - Test visual token pruning (TIES: 78% reduction, 2603.24941): does it work on our student?
   - Test KERV speculative decoding (2603.01581): applicable to action token generation?
   - Target: 25-40 Hz with optimizations

5. **On-policy refinement** (optional, if time permits):
   - VLA-OPD style (2603.26666): student generates trajectories, teacher provides dense supervision
   - DAgger-style: student acts in sim, teacher corrects failures
   - 1-2 rounds of on-policy data collection + retraining

**Exit criteria**:
- Auxiliary loss ranking established
- Teacher model comparison completed
- Data scaling curve plotted
- Student inference at >20 Hz on target hardware
- Decision: proceed to real-robot transfer, more sim tasks, or iterate on architecture

**Estimated cost**: ~$200-500

### Phase 5: Generalization & Transfer (Week 10-12, contingent)

**Goal**: Test whether the distilled VLA generalizes beyond training tasks and transfers to new settings.

**Prerequisites**: Phase 3-4 confirm auxiliary-supervised student outperforms BC baselines.

**Steps**:

1. **Zero-shot generalization** (within LIBERO):
   - Hold out 5 tasks per suite during training
   - Evaluate on held-out tasks: does auxiliary supervision improve generalization?

2. **Cross-benchmark transfer**:
   - Evaluate on SimplerEnv (Google DeepMind, standardized real-to-sim benchmark)
   - Evaluate on RoboTwin 2.0 (bimanual tasks, used by RDT2-VQ)
   - No fine-tuning — pure zero-shot transfer

3. **Real robot pilot** (if hardware available):
   - Franka Emika or WidowX (most common in VLA papers)
   - 5 simple tasks: pick-place, stack, push, drawer open, bin sorting
   - Fine-tune student on 50 real demos + evaluate
   - Compare: student pretrained on teacher traces vs student trained from scratch on real demos

4. **Publish training data**:
   - Release 10K+ traces with full CV-tool annotations as open dataset
   - Format compatible with Open X-Embodiment and LIBERO standards
   - Include teacher reasoning text (unique to this dataset)

**Exit criteria**:
- Zero-shot generalization measured (held-out tasks)
- Cross-benchmark results on SimplerEnv and/or RoboTwin
- Real robot results (if hardware available)
- Dataset released (or ready for release)

**Estimated cost**: ~$100-300 (compute only, no new data collection)

## Components

### 1. Compute
- **Teacher serving**: g7e.24xlarge (4x RTX PRO 6000, 96 GB each) or B300 if available
  - Qwen2.5-VL-72B FP8: TP4 on g7e (~$16.57/hr) or TP2 on B300 (~$16/hr spot)
  - Alternative: Qwen2.5-VL-7B FP8: TP1, 4 parallel instances on g7e
- **CV tools**: CPU (GroundingDINO, Depth Anything V2) or shared GPU (SAM 2)
- **Simulation**: CPU-only (MuJoCo/robosuite render to offscreen buffer)
- **Student training**: g7e.24xlarge (4-GPU DDP) or single B300 GPU
- **Student inference**: Single RTX PRO 6000 (3B model fits easily)

### 2. Codebase
- **Source**: New blueprint at `domains/autoresearch/blueprints/vla-cv-distillation/`
- **Dependencies**:
  - robosuite / LIBERO (simulation + benchmark)
  - GroundingDINO (detection), SAM 2 (segmentation), Depth Anything V2 (depth)
  - transformers + PEFT (student model training)
  - vLLM or SGLang (teacher serving)
  - NORA codebase (reference for FAST+ action tokenizer)
- **Fixed files**:
  - LIBERO benchmark definitions and evaluation protocol
  - CV model weights (pretrained, not fine-tuned)
  - Simulation environment definitions
- **Agent-editable files**:
  - `scripts/teacher_agent.py` — teacher loop with tool calling
  - `scripts/cv_tools.py` — CV tool wrappers
  - `scripts/train_student.py` — student VLA training
  - `scripts/eval_student.py` — LIBERO evaluation
  - `configs/` — serving configs, training hyperparameters
- **Agent instructions**: `program.md`

### 3. Experiment Protocol
- **Primary metric**: Average success rate on LIBERO (4 suites × 10 tasks × 20 episodes = 800 rollouts)
- **Secondary metrics**:
  - Student inference frequency (Hz) on target hardware
  - Training cost (GPU-hours, $)
  - Trace generation cost ($/trace)
  - Auxiliary loss convergence (do perception heads learn useful representations?)
  - Per-suite success rates (spatial vs object vs goal vs long-horizon)
- **Eval protocol**: 20 evaluation episodes per task, random seeds, 3 training seeds for confidence intervals
- **Logging**: WandB or local TensorBoard, JSONL per evaluation episode

### 4. Networking
- **Teacher serving**: localhost (colocated on same instance as simulation)
- **CV tools**: localhost (colocated)
- **No external API calls required** (all self-hosted)

### 5. Storage
- **Traces**: ~500 GB for 10K episodes (RGB images + depth maps + masks)
  - NVMe on g7e: 7 TB available, sufficient
  - Alternatively: S3 for persistence
- **Model checkpoints**: ~20 GB per student config × 6 configs = 120 GB
- **Results**: `domains/autoresearch/blueprints/vla-cv-distillation/results/`

## Success Criteria

### Phase 1: Teacher Works
- CV tools return valid outputs (bboxes, masks, depth) on simulation images
- Teacher agent solves >50% of test tasks in LIBERO
- Traces are complete (all fields populated, actions replay correctly)

### Phase 2: Traces at Scale
- 10K+ successful traces collected
- Teacher throughput: >50 traces/hour on g7e
- Cost: <$500 total for 10K traces
- Trace quality: replayed actions achieve >90% of original success rate

### Phase 3: Auxiliary Supervision Helps
- Config C (Aux-all) beats Config A (BC-scripted) by >5pp average success rate
- Config C beats Config B (BC-teacher) by >3pp (isolates auxiliary supervision value)
- Student inference: >15 Hz on single RTX PRO 6000
- At least one auxiliary head provides statistically significant improvement (p < 0.05)

### Phase 4: Understanding What Matters
- Auxiliary loss ranking established with ablation
- Data efficiency: Aux-supervised student at 5K traces matches BC at 10K traces
- Student inference: >20 Hz with visual token pruning

### Phase 5: It Generalizes
- Zero-shot held-out task success: >40% (meaningful generalization)
- Cross-benchmark transfer shows positive results on at least one new benchmark

### Negative Results (Still Valuable)

- **Auxiliary supervision doesn't help**: The student learns action prediction equally well with or without CV-tool outputs. Would mean: intermediate representations aren't the bottleneck; action quality in the traces is what matters. Simplifies the pipeline.
- **Teacher can't solve sim tasks**: VLM + CV tools isn't sufficient for closed-loop robot control. Would validate that VLA pretraining on real demonstrations (NORA, OpenVLA) is necessary, and teacher-generated traces aren't a viable shortcut.
- **Small teacher ≈ large teacher**: Qwen2.5-VL-7B traces produce equally good students as 72B traces. Would mean: trace diversity matters more than trace quality, favoring cheaper data collection.
- **Affordance is the only useful auxiliary**: Only the affordance head (which uses teacher VLM reasoning) helps; geometric outputs (detection, depth) don't. Would narrow the approach to reasoning distillation rather than perception distillation.

## Non-Requirements
- Real robot hardware (simulation-only, real robot is optional Phase 5)
- Multi-agent orchestration (single teacher agent with tools)
- RL or online learning (pure supervised distillation, on-policy refinement is optional)
- Full Open X-Embodiment scale (focus on LIBERO, expand later)
- Production deployment (research prototype, not a serving system)
- Custom CV model training (use pretrained off-the-shelf models)

## Known Limitations
- Simulation-to-real gap: LIBERO results may not transfer to physical robots without sim-to-real adaptation
- Teacher VLM action quality: VLMs generate text-based plans, not precise motor commands. Action grounding via inverse kinematics or motion planning adds complexity.
- Depth Anything V2 on simulation images: monocular depth models trained on real images may produce noisy depth on rendered scenes (mitigate: use ground-truth sim depth as fallback)
- NORA's FAST+ tokenizer is architecture-specific: may need adaptation for our student architecture
- 10K traces may be insufficient for long-horizon tasks (LIBERO-Long has 5-7 subtasks per episode)

## Risk Register

- **Teacher action grounding fails** (HIGH probability, HIGH impact): VLM outputs high-level plans ("move gripper to red block") but can't produce precise delta actions. Mitigation: use motion planning (RRT/OMPL) to convert waypoints to trajectories, or fall back to scripted oracle with tool-output overlay.
- **Sim images fool CV tools** (MEDIUM, MEDIUM): GroundingDINO/SAM trained on real photos may fail on MuJoCo renders. Mitigation: use LIBERO's realistic rendering, or fine-tune GroundingDINO on sim images (small dataset sufficient).
- **Auxiliary heads don't converge** (MEDIUM, MEDIUM): Detection/segmentation losses dominate action loss. Mitigation: tune λ weights carefully, use loss magnitude normalization.
- **GPU memory pressure during training** (LOW, MEDIUM): 3B model + 4 auxiliary heads + LoRA may exceed single-GPU memory. Mitigation: gradient checkpointing, smaller batch size, or detach auxiliary heads after warmup epochs.
- **LIBERO environment setup complexity** (MEDIUM, LOW): LIBERO depends on specific robosuite/mujoco versions. Mitigation: use Docker container with pinned dependencies.

## Relationship to Other Specs

- **agent-harness**: Provides the agentic trace collection methodology (Phase 2a trace format, behavioral telemetry). This spec extends trace collection from code editing to robotic manipulation.
- **verification-primitives**: Proved that agents compose tools effectively with checkpoint guidance (83.7% adoption). Same principle applies: the teacher agent composes CV tools when given structured access. The two-stage checkpoint pattern (edit@40% + verify@55%) may inspire similar checkpoint patterns for the teacher (perceive@30% + plan@60% + act@80%).
- **verifier-reward**: Provides the adversarial verification methodology. Future work: use a verifier to score student VLA traces (is this trajectory likely to succeed?) for rejection sampling.
- **gpu-serving specs**: Infrastructure patterns (FP8, prefix caching, TP strategies, g7e/B300 configs) directly reused for teacher serving and student inference optimization.

## Key References

- NORA (2504.19854): Qwen2.5-VL-3B VLA, FAST+ action tokenizer, 970K demos
- AnchorVLA4D (2603.12730): Qwen2.5-VL + diffusion action head, 80% real-world success
- VLA-OPD (2603.26666): On-policy distillation, teacher supervises student trajectories
- VITA-VLA (2510.09607): Action expert distillation into VLMs, 97.3% LIBERO
- PALM (2601.07060): Affordance distillation, 91.8% LIBERO-Long
- GenSim (2310.01361): LLM generates sim tasks, 25% real transfer improvement
- VoxPoser (2307.05973): LLM + VLM compose 3D value maps for manipulation
- Code as Policies (2209.07753): LLM calls perception APIs for robot control
- VANGUARD (2603.04277): CV tools fix VLM spatial hallucinations (>50% error rate without tools)
- OpenVLA (2406.09246): 7B open-source VLA, baseline comparison
- Shallow-pi (2601.20262): VLA layer distillation, 2x faster with <1% drop
- ProbeFlow (2603.17850): Adaptive denoising steps (50 → 2.6), 14.8x action decode speedup
- OxyGen (2603.14371): Unified KV cache for VLA, 70 Hz + 200 tok/s
- KERV (2603.01581): Kinematic speculative decoding for VLA, 27-37% acceleration
- CogVLA (2508.21046): Visual token pruning, 2.8x faster than OpenVLA
- TIES (2603.24941): 78% visual token reduction with +6% success improvement
- Lang2Grasp (GitHub): Qwen2-VL-2B → GroundingDINO → SAM → grasp coordinates (closest existing work)
- ViVLA (2512.07582): 892K traces from video, 30%+ improvement

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
