# Failure Modes Catalog — Continuous-Calibration RLVR Pipeline

**Captured 2026-05-10 / 2026-05-11 during first p4de launch of the self-coding-agent-loop experiment.**
**These are real failures from real hours spent debugging, not theoretical concerns.**

Organized by layer — bottom-to-top corresponds roughly to failure order during standup.

---

## Layer 1 — AWS / Instance

### FM-1.1: p4de AZ confusion (zone name ≠ zone ID)
**Symptom**: Launch planning targets "us-east-1 az6" (zone ID). Script hardcoded `us-east-1f`.
**Cause**: AWS uses stable zone IDs (`use1-az6`) and per-account zone names (e.g. `us-east-1c`). They don't match across accounts.
**Fix**: `aws ec2 describe-availability-zones --region us-east-1 --query '[].{name:ZoneName,id:ZoneId}'` — in our account `use1-az6 = us-east-1c`.
**Impact**: Would have launched in wrong AZ. p4de is only offered in us-east-1b (~$21/hr) and us-east-1c (~$13/hr). us-east-1f has no p4de capacity.
**Lesson**: Always resolve zone ID → zone name in the target account before launching.

### FM-1.2: No S3-write IAM profile in account
**Symptom**: Existing `*-gpu-node-*` instance profiles are EKS-only (worker + CNI + ECR-read), no S3 rw.
**Fix**: Found reusable `g7e-bench-profile` with inline `s3-artifacts-rw` policy scoped to `agent-aiops-artifacts` bucket.
**Lesson**: Before creating new IAM resources, grep `list-instance-profiles` for anything with your target bucket in an inline policy. Reuse is almost always possible.

### FM-1.3: Spot instance type `persistent` vs `one-time`
**Symptom**: `persistent` auto-relaunches on interruption — but our reclaim handler already sync'd state to S3 and we don't want silent restart.
**Fix**: Use `SpotInstanceType=one-time` + `InstanceInterruptionBehavior=terminate`.
**Lesson**: `persistent` is for stateful services; ML training with S3-backed checkpointing wants `one-time`.

### FM-1.4: Root volume sized too small
**Symptom**: Original launch used 200GB EBS. Qwen3.5-27B weights alone are ~55GB, plus pip cache, plus vLLM compile cache.
**Fix**: Bumped to 500GB gp3.
**Lesson**: For any serving/training of 20B+ models: ≥500GB root vol; consider using /mnt/nvme (see FM-2.1) for weights instead.

---

## Layer 2 — AMI / OS / Storage

### FM-2.1: DL AMI pre-formats NVMe as LVM — don't mkfs
**Symptom**: User-data ran `mkfs.xfs /dev/nvme8n1` → `Device or resource busy`. Then `mount /dev/nvme8n1 /mnt/nvme` → `unknown filesystem type LVM2_member`.
**Cause**: Deep Learning Base AMI (Ubuntu 22.04 NVIDIA) stripes all 8 NVMe drives as a single 6.8TB LVM volume at `/opt/dlami/nvme`.
**Fix**: Don't mkfs. Symlink: `ln -s /opt/dlami/nvme /mnt/nvme`.
**Detection**: `lsblk | grep "lvm.*ephemeral"` or `df -h | grep dlami`.
**Lesson**: On any AWS DL AMI, NVMe is ready-to-use — check `mount` before trying to format.

### FM-2.2: `ec2-user` vs `ubuntu` default user
**Symptom**: Systemd units specified `User=ec2-user`, SSH commands targeted `ubuntu@`. Unit failed silently.
**Cause**: AL2/AL2023 AMIs use `ec2-user`, Ubuntu AMIs use `ubuntu`.
**Fix**: Detect AMI family at bootstrap, alias accordingly. Simplest: pick one AMI family per project and standardize.

### FM-2.3: Repo snapshot tarball too big
**Symptom**: First-launch snapshot 2.6GB, upload took 5 min.
**Cause**: Included `learned-verifier/data/features/*.csv` (some 5MB+), full experiment result JSONLs, etc.
**Fix**: Explicit excludes for `data/nebius`, `data/features/e6_*`, `*.tfstate*`. Or: don't tarball the repo; `aws s3 cp` only scripts + config.
**Lesson**: Keep repo snapshots < 500MB. Large datasets go to S3 separately and pull on demand.

---

## Layer 3 — Python / ML Library

### FM-3.1: transformers pre-built wheels don't know about new model types
**Symptom**: `KeyError: 'qwen3_5'` and `KeyError: 'qwen3_moe'` when loading configs.
**Cause**: `Qwen/Qwen3.5-27B` (model_type `qwen3_5`) and `Qwen/Qwen3-Coder-30B-A3B-Instruct` (model_type `qwen3_moe`) are newer than transformers 4.57.6 on PyPI.
**Fix**: `pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"` → gets 5.8.0.dev0.
**Lesson**: For any Qwen-family model released < 3 months ago: install transformers from git main, don't trust the PyPI wheel.

### FM-3.2: vLLM 0.18 pins transformers<5
**Symptom**: After upgrading transformers to 5.x, vLLM still "works" per `import` test but may be brittle.
**Fix**: Upgrade vLLM to 0.20.2+ (dropped the `<5` pin).
**Lesson**: Paired versions: transformers 5.x → vLLM 0.20.2+, trl 1.4+, peft 0.19+.

### FM-3.3: Qwen3.5-27B is a VLM, not a text model
**Symptom**: `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-27B")` → `ValueError: checkpoint has model type qwen3_5 but Transformers does not recognize this architecture`. After transformers upgrade: loads but has `vision_config`.
**Cause**: Qwen3.5-27B is `Qwen3_5ForConditionalGeneration` (vision-language). LoRA on `q_proj`/`k_proj` does partial-work but adapter keys mismatch.
**Fix**: Detect VLM via `config.architectures` containing `ConditionalGeneration` or `ImageTextToText`. Use `AutoModelForImageTextToText`. OR: pick a pure-text model (e.g. Qwen3-Coder-30B-A3B-Instruct) and avoid the complexity.
**Lesson**: Check `AutoConfig.from_pretrained(...).architectures` BEFORE building training code. One line of pre-flight saves hours.

### FM-3.4: trl 1.4 API rename — `max_seq_length` → `max_length`
**Symptom**: `SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`.
**Fix**: Use `max_length=...` in SFTConfig. Pinning trl version is the alternative.

### FM-3.5: trl 1.4 API rename — `tokenizer=` → `processing_class=`
**Symptom**: `SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'`.
**Fix**: `SFTTrainer(..., processing_class=tokenizer)`.

### FM-3.6: trl 1.4 Callback must inherit from TrainerCallback
**Symptom**: Training starts, `on_train_begin` lookup: `AttributeError: 'CkptOnSignal' object has no attribute 'on_train_begin'`.
**Cause**: Older trl was permissive about duck-typed callbacks; 1.4 dispatches all lifecycle events and bare objects don't have them.
**Fix**: `from transformers import TrainerCallback` and have custom callbacks inherit from it.

### FM-3.7: PEFT adapter key mismatch across model families
**Symptom**: `UserWarning: Found missing adapter keys: ['base_model.model.model.language_model.layers.0...']`.
**Cause**: Gen0 adapter trained against Qwen3.5-27B (VLM — keys under `model.language_model.layers`). Loaded onto Qwen3-Coder-30B-A3B (text — keys under `model.layers`).
**Fix**: Validate adapter's `base_model_name_or_path` against current target before loading. Fall back to fresh LoRA if mismatch.
**Detection**: `json.load(open("adapter_config.json"))["base_model_name_or_path"]` == current base.

---

## Layer 4 — Data / Format

### FM-4.1: Trajectory field dropped from split files
**Symptom**: `round_1_train.jsonl` records missing `trajectory` key → SFT format returns 0 usable examples.
**Cause**: `build_splits.py` read only 7 columns from parquet; didn't include `trajectory` (the actual training input).
**Fix**: Add `"trajectory"` to the read column list. For existing splits that don't have it, join from parquet on-demand.
**Lesson**: Write a split-validation script that opens one record and asserts all fields the trainer needs are present.

### FM-4.2: numpy ndarray inside trajectory breaks chat template
**Symptom**: `TypeError: 'ndarray' object is not subscriptable` or `Can only get item pairs from a mapping` during `apply_chat_template`.
**Cause**: Parquet deserializes list-of-struct columns as numpy arrays, not Python lists. Jinja templates assume dicts/lists.
**Fix**: Recursive `_to_plain()` converter: `hasattr(v, "tolist") → v.tolist()`; `isinstance(v, dict) → {k: _to_plain(val)}`; etc.
**Lesson**: Always assume parquet data needs normalization before feeding to templating libraries.

### FM-4.3: tool_call.function.arguments is a JSON string, not a dict
**Symptom**: After _to_plain, chat template still fails: `Can only get item pairs from a mapping` at index 2 (first assistant turn with tool_calls).
**Cause**: Qwen3-Coder chat template does `tool_call.arguments|items` — expects dict. Nebius stores `arguments` as serialized JSON string.
**Fix**: Parse `arguments` with `json.loads` before passing to apply_chat_template.
**Lesson**: Every LLM's chat template has its own assumptions about tool_call structure. Test with one trajectory end-to-end before paying for 4K.

### FM-4.4: Silent except in format loop hides root cause for hours
**Symptom**: `[round 1] formatted 0 trajectories, skipped 4258` with no indication WHY.
**Cause**: Classic `try: format(); except: skipped += 1` pattern ate the underlying TypeError.
**Fix**: Capture `first_error = f"{type(e).__name__}: {e}"` and log it. Raise SystemExit if 100% fail rate.
**Lesson**: Never silently swallow exceptions in format/parse loops during new pipeline development. Two lines of error capture saves hours.

---

## Layer 5 — Operational / Cost

### FM-5.1: p4de idle during Docker eval → burning $13.50/hr
**Symptom**: Sequential pipeline: Round N SFT → Round N generate → wait on Docker eval (12-24hr on m7i) → Round N+1.
**Fix**: Orchestrator starts Round N+1 SFT in parallel with Round N's Docker eval. Halves wall-clock for multi-round experiments.

### FM-5.2: Bootstrap restart loses SFT token cache
**Symptom**: Tokenizing the train dataset is ~9 minutes of CPU work; every time orchestrator restarts (for a fix), we re-tokenize from scratch.
**Fix**: Pre-tokenize once to `/opt/dlami/nvme/tokenized/round_N_formatted.jsonl`. Skip tokenization if file exists.
**Lesson**: On any pipeline where you iterate on trainer config frequently, cache the tokenized dataset to disk.

### FM-5.4: Pre-pulling 600 Docker images on a small eval box fills disk
**Symptom**: Tried to pre-pull all 600 SWE-rebench eval images on m7i.4xlarge (200GB root vol). Got 36 images (~130GB) before disk hit 100%. 564/600 failed with "no space left on device".
**Cause**: SWE-rebench eval images average ~3-5GB. 600 images = ~1.8TB total. m7i.4xlarge's default 200GB EBS is insufficient.
**Fix**: Don't pre-pull. SWE-rebench-V2 `eval.py` pulls+runs+`docker rmi` per instance by default — peak disk is ~5GB (one image at a time), not 1.8TB.
**Lesson**: Default to on-demand pull for Docker eval at scale. Pre-pulling only makes sense if (a) total corpus fits in disk, OR (b) eval reuses the same small set of images many times.

### FM-5.3: Debugging on $13.50/hr compute is expensive
**Symptom**: Spent ~$50 iterating through API-fit issues on p4de that could have been caught locally on a 1B model.
**Fix**: For NEXT pipeline: smoke-test the format loop + trainer config locally with TinyLlama-1.1B before paying for p4de.
**Lesson**: Every API-level bug costs $13.50/hr × minutes-to-diagnose on p4de. Most are reproducible on any Linux box with 8GB GPU.

---

## Layer 6 — Model family / recipe mismatches

### FM-6.1: Gen0 adapter trained against wrong base
**Symptom**: Gen0 LoRA at `blueprints/rejection-sampling-sft/models/run_d_Qwen3.5-27B/final/` targets Qwen3.5-27B VLM. We tried to load on Qwen3-Coder-30B-A3B.
**Fix**: Validate adapter `base_model_name_or_path` against current target. Fresh LoRA if mismatch.
**Lesson**: Every LoRA adapter should declare its base explicitly; pipelines should refuse to load silently-mismatched adapters.

### FM-6.3: SWE-ReBench instances aren't in public swebench Docker registry — but they ARE in swerebench
**Symptom**: `docker pull swebench/sweb.eval.x86_64.<instance_id>:latest` → `pull access denied` for 599/600.
**Cause chain**:
1. The public `swebench/` Docker Hub namespace contains SWE-bench Lite and Verified only.
2. The Nebius OpenHands trajectories dataset we train on comes from `nebius/SWE-rebench` (v1, 21K tasks). Not SWE-bench, not SWE-rebench-V2.
3. v1 task metadata lives at `nebius/SWE-rebench` HF dataset — has the exact field `image_name` per instance, pointing at `swerebench/sweb.eval.x86_64.<encoded_id>:latest` (note: `swerebench`, not `swebench`).
4. Instance ID encoding: `user__repo-N` → `user_1776_repo-N` (lowercase, `__` → `_1776_`). Also often has extra `:tag` suffix like `:latest` or a commit SHA.
**Resolution**:
- Load task metadata from `nebius/SWE-rebench` HF dataset.
- Extract `image_name` field per instance — it's already the full Docker reference.
- Use SWE-rebench-V2's `scripts/eval.py` from github.com/SWE-rebench/SWE-rebench-V2 with `--json` input (pass the task records + our model_patch overlay via `--patches`).
**Verification**: `docker pull docker.io/swerebenchv2/elastic-synthetics:316-f52f0bf` works. Pre-pull script updated.
**Lesson**: Three-dataset confusion — trajectories dataset (for training) ≠ task dataset (for eval metadata) ≠ benchmark version (v1 vs V2). Read the dataset card to find image references. Don't guess registry naming.

### FM-6.2: Nebius trajectories were generated by Qwen3-Coder-480B
**Context**: `nebius/SWE-rebench-openhands-trajectories` — 67K trajectories generated by the 480B teacher. If you train the 30B-A3B student on them, you're doing teacher-student distillation (expected good); if you train an unrelated model family, you're doing format-matching + vibes.
**Lesson**: Match the student to the teacher's family when using Nebius trajectories. Qwen3-Coder-30B-A3B is the natural student.

---

## Severity grid

| Severity | Criterion | Examples |
|---|---|---|
| P0 (halt) | Silent data loss, reward hacking, verifier-gold divergence | FM-4.4 hid real cause for 1.5hr |
| P1 (fix before next round) | Training bugs, API breakage | FM-3.4, FM-3.5, FM-3.6, FM-4.2, FM-4.3 |
| P2 (degrades performance) | Wrong base, wrong recipe | FM-6.1, FM-3.3 |
| P3 (operational) | Cost waste, slow wall-clock | FM-5.1, FM-5.2 |
| P4 (nuisance) | User confusion, AMI quirks | FM-1.1, FM-2.1, FM-2.2 |

---

## Debug time budget (empirical from this project)

| Category | Cumulative debug hours |
|---|---|
| Layer 1 (AWS) | 0.5 |
| Layer 2 (AMI/storage) | 0.8 |
| Layer 3 (Python/ML) | 2.5 |
| Layer 4 (data) | 1.5 |
| Layer 5 (ops) | ongoing |
| **Total to first successful SFT step** | **~5 hours, ~$50 burned on idle p4de** |

Budget 1 day of engineering time for first-run standup of this pipeline on a new team. The failures above account for ~90% of what will go wrong.
