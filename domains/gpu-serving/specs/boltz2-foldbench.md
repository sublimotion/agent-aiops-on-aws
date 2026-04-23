# Autoresearch Spec: Boltz2 FoldBench

## Status: DRAFT

## Overview

Benchmark protein structure prediction on FoldBench using Boltz2 and ESMFold across three tasks:

1. **ESMFold monomers** — Fold 330 protein monomers from FoldBench with ESMFold (single-sequence, no MSA)
2. **Boltz2 monomers** — Fold the same 330 monomers with Boltz2 (without MSA)
3. **Boltz2 antibody-antigen** — Fold 172 Ab-Ag complexes with Boltz2 (without MSA)

**Why**: Establish baseline structural accuracy for diffusion-based (Boltz2) vs language-model-based (ESMFold) prediction on a standardized low-homology benchmark. Published Boltz2 numbers on FoldBench: 70.5% protein-protein DockQ, but only 25.0% Ab-Ag (vs AlphaFold3 48.0%). This run validates those numbers on our infrastructure and establishes a reproducible pipeline for future model comparisons.

**Competitive context**: Luminal.com is an LLM inference compiler (not protein folding). TerraBind (Terray Therapeutics) claims 26x faster than Boltz2 by skipping coordinate generation — not applicable here since FoldBench evaluates 3D structure quality (LDDT, DockQ).

**Depends on**: FoldBench (BEAM-Labs/FoldBench), Boltz2 (jwohlwend/boltz), ESMFold (facebookresearch/esm)

## Components

### 1. Compute

- **Platform**: AWS EC2 (bare metal GPU instances)
- **Phase 1 (initial)**: 2x `p4d.24xlarge` (8x A100 80GB SXM per node = 16 GPUs total)
- **Phase 2 (scale)**: 16x `p4d.24xlarge` (128 GPUs total)
- **Cost estimate**: Phase 1 ~$200 on-demand (~$80 spot) for full run; Phase 2 ~$260 for 30-min run

### 2. Codebase

- **FoldBench**: `https://github.com/BEAM-Labs/FoldBench`
  - `targets/monomer_protein.csv` — 330 monomers (pdb_id, chain_id)
  - `targets/interface_antibody_antigen.csv` — 172 Ab-Ag complexes (pdb_id, chain pairs)
  - Ground truth CIF structures via Google Drive download
  - Evaluation: `evaluate.py` (LDDT for monomers, DockQ >= 0.23 for interfaces)

- **Boltz2**: `https://github.com/jwohlwend/boltz` (MIT license)
  - Install: `pip install boltz[cuda] -U`
  - Inference: `boltz predict <input_dir> --sampling_steps 50 --diffusion_samples 1`
  - No-MSA mode: `msa: empty` in YAML input

- **ESMFold**: `pip install "fair-esm[esmfold]"`
  - Monomers only (cannot predict complexes)
  - CLI: `esm-fold -i proteins.fasta -o output/ --max-tokens-per-batch 1024 --chunk-size 128`

- **FoldBench integration** (4 files per algorithm):
  - `algorithms/Boltz2/container.def` — Apptainer container definition
  - `algorithms/Boltz2/preprocess.py` — Convert alphafold3_inputs.json → Boltz YAML
  - `algorithms/Boltz2/make_predictions.sh` — Run `boltz predict`
  - `algorithms/Boltz2/postprocess.py` — Generate prediction_reference.csv
  - Same pattern for `algorithms/ESMFold/`

### 3. Experiment Protocol

#### Task 1: ESMFold Monomers (330 structures)

- **Input**: FASTA sequences extracted from FoldBench monomer_protein.csv
- **Model**: ESMFold v1 (ESM-2 3B backbone)
- **GPU**: Single A100 sufficient (seconds per protein)
- **Metric**: LDDT (per-residue, averaged)
- **Expected runtime**: ~5-15 min on 1 GPU

#### Task 2: Boltz2 Monomers (330 structures, no MSA)

- **Input**: YAML files with `msa: empty` for each monomer
- **Flags**: `--sampling_steps 50 --recycling_steps 3 --diffusion_samples 1`
- **GPU**: 1 A100 per prediction (embarrassingly parallel)
- **Metric**: LDDT
- **Expected runtime**: ~60 min on 16 GPUs (directory-mode batching, length-sorted bin packing)

#### Task 3: Boltz2 Ab-Ag Complexes (172 structures, no MSA)

- **Input**: YAML files with heavy chain, light chain, and antigen as separate protein entries, `msa: empty`
- **Flags**: same as Task 2
- **GPU**: 1 A100 per prediction
- **Metric**: DockQ (threshold >= 0.23 for success)
- **Expected runtime**: ~60 min on 16 GPUs

#### Optimization Stack

| Optimization | Speedup | Implementation |
|---|---|---|
| `--sampling_steps 50` (not 200) | 4x on diffusion phase | CLI flag (Boltz paper: 0.002 LDDT loss) |
| Directory-mode batching | ~5x vs individual calls | Split inputs into per-GPU directories |
| Length-sorted bin packing | 1.1-1.3x (tail latency) | Sort by residue count, greedy assignment to 16 buckets |
| 16-way job parallelism | 16x | One `boltz predict <dir>` per GPU via GNU parallel |
| `compile_structure=True` | ~1.3x (estimated, optional) | Code fork — enable hidden torch.compile flag |

**Not using**: BioNeMo NIM (proprietary, estimated ~1.3-1.8x delta not worth the dependency), Boltz-DAP/Fold-CP (memory tools for large proteins, FoldBench targets fit on single A100), Ray (no benefit for embarrassingly parallel single-GPU jobs).

### 4. Networking

- **SSH**: Direct SSH to p4d instances
- **Inter-node**: Not needed (no multi-GPU sharding per prediction). Nodes operate independently.
- **Data transfer**: FoldBench ground truth CIFs (~GB) downloaded once to shared storage

### 5. Storage

- **Data**: FoldBench targets + ground truth CIFs on NVMe local storage (`/mnt/nvme`)
- **Model weights**: Boltz2 (~5GB) and ESMFold (~6GB) cached in `~/.boltz` and `~/.cache/torch/hub`
- **Results**: Per-task output directories with CIF predictions + FoldBench evaluation CSVs
- **Logs**: Per-GPU prediction logs, timing data, GPU utilization metrics

## Execution Plan

### Phase 0: Setup (~1 hour)

1. Launch 2x p4d.24xlarge (spot if available)
2. Install Boltz2 (`pip install boltz[cuda]`) and ESMFold on each node
3. Clone FoldBench, download ground truth structures
4. Extract sequences from FoldBench CSVs, generate Boltz2 YAML inputs with `msa: empty`
5. Bin-pack inputs into 16 directories sorted by sequence length (longest-first)

### Phase 1: ESMFold Baseline (~15 min)

1. Run ESMFold on all 330 monomers (single GPU)
2. Convert PDB outputs to CIF format for FoldBench evaluation
3. Run `evaluate.py` for LDDT scores
4. Record: per-protein LDDT, mean LDDT, wall time, GPU utilization

### Phase 2: Boltz2 Monomers (~1 hour)

1. Run `boltz predict` on each of 16 GPU directories (GNU parallel)
2. Postprocess outputs for FoldBench evaluation format
3. Run `evaluate.py` for LDDT scores
4. Record: per-protein LDDT, mean LDDT, wall time per GPU, total wall time
5. Compare ESMFold vs Boltz2 on same 330 monomers

### Phase 3: Boltz2 Ab-Ag (~1 hour)

1. Generate YAML inputs for 172 complexes (multi-chain: heavy, light, antigen)
2. Bin-pack into 16 directories
3. Run `boltz predict` on each GPU
4. Run `evaluate.py` for DockQ scores
5. Record: per-complex DockQ, success rate (DockQ >= 0.23), wall time

### Phase 4: Analysis

1. Compare against published FoldBench numbers (Boltz2 paper Table)
2. Analyze failure modes: which proteins/complexes fail? Size correlation?
3. ESMFold vs Boltz2 monomer head-to-head (speed vs accuracy tradeoff)
4. Cost-per-structure analysis
5. Extrapolate Phase 2 (16-node) timing from Phase 1 results

## Success Criteria

1. **Reproduce published results**: Boltz2 monomer LDDT within 0.02 of published FoldBench numbers
2. **Ab-Ag baseline**: Establish DockQ success rate on 172 complexes (published: 25.0%)
3. **ESMFold comparison**: Quantify accuracy gap between single-sequence ESMFold and no-MSA Boltz2 on monomers
4. **Throughput**: Complete all 502 predictions in < 3 hours on 16 A100s
5. **Pipeline**: Reproducible end-to-end script from FoldBench CSVs to evaluation results

## Non-Requirements

- MSA generation (all runs use `msa: empty` / single-sequence mode)
- Multi-GPU sharding per prediction (all FoldBench targets fit on single A100 80GB)
- Binding affinity prediction (structure-only evaluation)
- NIM / TensorRT optimization (open-source stack only)
- Training or fine-tuning of any model
- Comparison with AlphaFold3 (no open-source access)

## Known Limitations

- **Boltz2 Ab-Ag accuracy is low**: Published 25.0% DockQ success vs AlphaFold3's 48.0%. This is a known model limitation, not an infrastructure issue.
- **No MSA reduces accuracy**: Running without MSA (`msa: empty`) will produce lower accuracy than MSA-augmented runs. This is intentional — establishes single-sequence baseline comparable to ESMFold.
- **A100 memory ceiling**: ~2,400 residues per monomer. FoldBench targets should be well under this, but any OOM failures will be logged and retried with `--sampling_steps 20`.
- **ESMFold is monomers-only**: Cannot evaluate on Ab-Ag complexes. Only covers Task 1.
- **FoldBench evaluation requires Apptainer**: May need containerized setup for standardized evaluation.

## Future Extensions

- **With MSA**: Re-run Boltz2 with `--use_msa_server` to quantify MSA improvement
- **Protenix-Mini**: Benchmark ByteDance's lightweight model for speed/accuracy tradeoff
- **Fold-CP at scale**: If Phase 2 reveals OOM targets, use NVIDIA's context parallelism ([NVIDIA-Digital-Bio/boltz-cp](https://github.com/NVIDIA-Digital-Bio/boltz-cp)) for 2D-tiled pair representation sharding
- **Boltz-DAP for large complexes**: Use [coqylight/boltz_dap](https://github.com/coqylight/boltz_dap) for single-node multi-GPU sharding of any oversized targets
- **Consistency distillation**: Research direction — distill Boltz2's diffusion into a consistency model for 5-10 step inference (cf. Together AI's CDLM approach, not yet applied to structure prediction)
- **16-node throughput**: Scale to 128 GPUs for sub-30-min full benchmark completion

---

> **Note**: Operational artifacts (lessons learned, experiment results, analysis)
> belong in the blueprint directory, not in this spec.
