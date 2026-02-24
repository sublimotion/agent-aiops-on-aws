# NVIDIA TensorRT-LLM: Supported Models -- Comprehensive Research

**Research Date:** February 2026
**TensorRT-LLM Latest Stable Version:** v1.3.x (as of early 2025)
**Sources:** Official GitHub repository (NVIDIA/TensorRT-LLM), NVIDIA documentation site

---

## 1. Overview

NVIDIA TensorRT-LLM is a high-performance inference framework for Large Language Models (LLMs), built on top of PyTorch and NVIDIA TensorRT. It provides optimized GPU inference with features including custom attention kernels, inflight batching, paged KV caching, multiple quantization formats (FP8, FP4, INT4 AWQ, INT8 SmoothQuant), speculative decoding, expert parallelism, guided decoding, and multi-GPU/multi-node deployment.

TensorRT-LLM operates with **two backends**:
- **PyTorch Backend**: The newer, actively developed backend with direct PyTorch integration
- **TensorRT Backend** (legacy/classic): Uses compiled TensorRT engines for inference

As of 2025, the framework supports **90+ distinct model architectures** across both backends, spanning language-only, multimodal (image, video, audio), encoder-decoder, and classification models.

---

## 2. Full List of Supported Model Architectures

### 2.1 PyTorch Backend -- Language Models

| HuggingFace Architecture Class | Model Family | Example Model |
|---|---|---|
| `BertForSequenceClassification` | BERT-based | textattack/bert-base-uncased-yelp-polarity |
| `DeciLMForCausalLM` | Nemotron | nvidia/Llama-3_1-Nemotron-51B-Instruct |
| `DeepseekV3ForCausalLM` | DeepSeek-V3 | deepseek-ai/DeepSeek-V3 |
| `DeepseekV32ForCausalLM` | DeepSeek-V3.2 | deepseek-ai/DeepSeek-V3.2 |
| `Exaone4ForCausalLM` | EXAONE 4.0 | LGAI-EXAONE/EXAONE-4.0-32B |
| `ExaoneMoEForCausalLM` | K-EXAONE | LGAI-EXAONE/K-EXAONE-236B-A23B |
| `Gemma3ForCausalLM` | Gemma 3 | google/gemma-3-1b-it |
| `Glm4MoeForCausalLM` | GLM-4.5 / 4.6 / 4.7 | THUDM/GLM-4-100B-A10B |
| `GptOssForCausalLM` | GPT-OSS | openai/gpt-oss-120b |
| `LlamaForCausalLM` | Llama 3.1, Llama 3, Llama 2, LLaMA | meta-llama/Meta-Llama-3.1-70B |
| `MiniMaxM2ForCausalLM` | MiniMax M2 / M2.1 | MiniMaxAI/MiniMax-M2 |
| `MistralForCausalLM` | Mistral, Bielik | mistralai/Mistral-7B-v0.1 |
| `MixtralForCausalLM` | Mixtral (MoE) | mistralai/Mixtral-8x7B-v0.1 |
| `NemotronForCausalLM` | Nemotron-3, Nemotron-4, Minitron | nvidia/Minitron-8B-Base |
| `NemotronHForCausalLM` | Nemotron-3-Nano | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 |
| `NemotronNASForCausalLM` | NemotronNAS (Super) | nvidia/Llama-3_3-Nemotron-Super-49B-v1 |
| `Phi3ForCausalLM` | Phi-4 | microsoft/Phi-4 |
| `Qwen2ForCausalLM` | QwQ, Qwen2 | Qwen/Qwen2-7B-Instruct |
| `Qwen2ForProcessRewardModel` | Qwen2-based Process Reward | Qwen/Qwen2.5-Math-PRM-7B |
| `Qwen2ForRewardModel` | Qwen2-based Reward Model | Qwen/Qwen2.5-Math-RM-72B |
| `Qwen3ForCausalLM` | Qwen3 | Qwen/Qwen3-8B |
| `Qwen3MoeForCausalLM` | Qwen3 MoE | Qwen/Qwen3-30B-A3B |
| `Qwen3NextForCausalLM` | Qwen3Next | Qwen/Qwen3-Next-80B-A3B-Thinking |

### 2.2 PyTorch Backend -- Multimodal Models

| HuggingFace Architecture Class | Model Family | Modalities |
|---|---|---|
| `Gemma3ForConditionalGeneration` | Gemma 3 Vision | Language + Image |
| `HCXVisionForCausalLM` | HyperCLOVAX-SEED-Vision | Language + Image |
| `LlavaLlamaModel` | VILA | Language + Image + Video |
| `LlavaNextForConditionalGeneration` | LLaVA-NeXT | Language + Image |
| `Llama4ForConditionalGeneration` | Llama 4 | Language + Image |
| `MllamaForConditionalGeneration` | Llama 3.2 Vision | Language + Image |
| `Mistral3ForConditionalGeneration` | Mistral 3 | Language + Image |
| `NemotronH_Nano_VL_V2` | Nemotron Nano Vision | Language + Image + Video |
| `Phi4MMForCausalLM` | Phi-4 Multimodal | Language + Image + Audio |
| `Qwen2VLForConditionalGeneration` | Qwen2-VL | Language + Image + Video |
| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL | Language + Image + Video |
| `Qwen3VLForConditionalGeneration` | Qwen3-VL | Language + Image + Video |
| `Qwen3VLMoeForConditionalGeneration` | Qwen3-VL MoE | Language + Image + Video |

### 2.3 TensorRT Backend -- LLM Models

The TensorRT (classic/compiled engine) backend supports the following architectures:

| Model | Notes |
|---|---|
| Arctic | Snowflake Arctic MoE model |
| Baichuan | Baichuan 7B/13B |
| Baichuan2 | Baichuan2 7B/13B |
| BART | Encoder-decoder seq2seq |
| BERT | Encoder-only, classification/QA |
| BLOOM | BigScience BLOOM family |
| ByT5 | Byte-level T5 variant |
| ChatGLM | THU/Zhipu ChatGLM |
| ChatGLM2 | Second generation |
| ChatGLM3 | Third generation |
| Code LLaMA | Meta's code-specialized LLaMA |
| DBRX | Databricks DBRX MoE |
| Exaone | LG AI EXAONE |
| FairSeq NMT | Facebook's neural machine translation |
| Falcon | TII Falcon (7B/40B/180B) |
| Flan-T5 | Google's instruction-tuned T5 |
| Gemma | Google Gemma 2B/7B |
| Gemma2 | Google Gemma 2 |
| GLM-4 | Fourth-gen GLM |
| GPT | GPT-2 style models |
| GPT-J | EleutherAI GPT-J-6B |
| GPT-Nemo | NVIDIA NeMo GPT |
| GPT-NeoX | EleutherAI GPT-NeoX-20B |
| Granite-3.0 | IBM Granite models |
| Grok-1 | xAI Grok-1 (314B MoE) |
| InternLM | Shanghai AI Lab InternLM |
| InternLM2 | InternLM2 (7B/20B) |
| LLaMA | Meta LLaMA 1/2/3 family |
| Mamba | State-space model (Mamba-1, Mamba-2) |
| mBART | Multilingual BART |
| Minitron | NVIDIA Minitron (compressed) |
| Mistral | Mistral AI (7B and variants) |
| Mistral NeMo | Mistral NeMo variant |
| Mixtral | Mistral MoE (8x7B, 8x22B) |
| MPT | MosaicML MPT (7B/30B) |
| mT5 | Multilingual T5 |
| Nemotron | NVIDIA Nemotron family |
| OPT | Meta OPT (125M to 175B) |
| Phi-1.5 | Microsoft Phi-1.5 |
| Phi-2 | Microsoft Phi-2 |
| Phi-3 | Microsoft Phi-3 (mini/small/medium) |
| Qwen | Alibaba Qwen 1.0 |
| Qwen1.5 | Alibaba Qwen 1.5 |
| Qwen-VL | Qwen Vision-Language |
| RecurrentGemma | Google's recurrent Gemma |
| Replit Code | Replit code generation |
| RoBERTa | Robustly Optimized BERT |
| SantaCoder | BigCode SantaCoder |
| Skywork | Skywork models |
| Smaug | Smaug-72B |
| StarCoder | BigCode StarCoder 1/2 |
| T5 | Google T5 family |
| Whisper | OpenAI Whisper (speech-to-text) |

### 2.4 TensorRT Backend -- Multi-Modal Models

| Model | Description |
|---|---|
| BLIP2 w/ OPT | Vision-language with OPT backbone |
| BLIP2 w/ T5 | Vision-language with T5 backbone |
| CogVLM | THU CogVLM |
| Deplot | Chart/plot understanding |
| Fuyu | Adept Fuyu multimodal |
| Kosmos | Microsoft Kosmos-2 |
| LLaVA-v1.5 | LLaVA version 1.5 |
| LLaVA-Next | LLaVA-NeXT (1.6+) |
| LLaVA-OneVision | LLaVA-OneVision |
| NeVA | NVIDIA NeVA |
| Nougat | Meta Nougat (document understanding) |
| Phi-3-vision | Microsoft Phi-3 with vision |
| Video NeVA | NVIDIA Video NeVA |
| VILA | NVIDIA VILA |
| MLLaMA | Meta multimodal LLaMA |
| Llama 3.2 VLM | Meta Llama 3.2 Vision |

---

## 3. Supported Model Sizes and Configurations

TensorRT-LLM does not impose hard limits on model sizes. It supports models ranging from small (sub-1B parameters) to extremely large (hundreds of billions of parameters) through parallelism strategies:

### Size Examples Across Model Families

| Model Family | Supported Sizes | Notes |
|---|---|---|
| LLaMA / Llama 2/3 | 7B, 8B, 13B, 70B, 405B | Full range from smallest to largest |
| Falcon | 7B, 40B, 180B | All publicly released sizes |
| BLOOM | 560M to 176B | Full BLOOM family |
| OPT | 125M to 175B | All size variants |
| GPT-J | 6B | Standard configuration |
| GPT-NeoX | 20B | Standard configuration |
| Mixtral | 8x7B, 8x22B | MoE configurations |
| DeepSeek-V3 | 671B (37B active) | Large MoE architecture |
| Grok-1 | 314B (MoE) | Large MoE architecture |
| Phi | 1.5B (Phi-1.5), 2.7B (Phi-2), 3.8B-14B (Phi-3/4) | Small model family |
| Qwen | 0.5B to 110B+ | Wide range of sizes |
| Gemma | 2B, 7B, 9B, 27B | Multiple generations |
| BERT / RoBERTa | Base (110M), Large (340M) | Encoder models |
| Whisper | Tiny to Large-v3 | All Whisper sizes |

### Parallelism Support

- **Tensor Parallelism (TP)**: Splits model layers across GPUs. Widely supported across all models.
- **Pipeline Parallelism (PP)**: Splits model layers sequentially across GPUs. Supported for many large models.
- **Expert Parallelism (EP)**: Distributes MoE experts across GPUs. Supported for Mixtral, DeepSeek-V3, Qwen3MoE, Arctic, DBRX, Grok-1, and other MoE models.
- **Context Parallelism (CP)**: For long-context scenarios.
- **Attention Data Parallelism**: Supported for select PyTorch backend models.

---

## 4. Mixture-of-Experts (MoE) Model Support

TensorRT-LLM has extensive MoE support:

| MoE Model | Architecture | Expert Config |
|---|---|---|
| Mixtral 8x7B | Sparse MoE | 8 experts, top-2 routing |
| Mixtral 8x22B | Sparse MoE | 8 experts, top-2 routing |
| DeepSeek-V3 | MoE + MLA | 256 experts, shared experts |
| Arctic | Dense + MoE hybrid | 128 experts |
| DBRX | Fine-grained MoE | 16 experts, top-4 routing |
| Grok-1 | Sparse MoE | 8 experts |
| Qwen3MoE | Sparse MoE | Various configs (e.g., 30B-A3B) |
| Qwen3-VL MoE | Multimodal MoE | Vision-Language MoE |
| GLM-4 MoE | Sparse MoE | GLM-4-100B-A10B |
| K-EXAONE | Sparse MoE | 236B-A23B |
| GPT-OSS | MoE | 120B variant |
| Qwen3Next | MoE | 80B-A3B-Thinking |
| MiniMax M2 | MoE | Large-scale MoE |

MoE-specific features include FP8 quantization for expert weights, expert parallelism across GPUs, and the TRTLLM MoE backend as the default on Blackwell GPUs.

---

## 5. Hugging Face Compatibility

### Direct Loading from Hugging Face Hub

TensorRT-LLM's high-level Python API (`LLM` class) supports direct loading from Hugging Face:

```python
from tensorrt_llm import LLM

# Load directly from Hugging Face Hub
llm = LLM(model="meta-llama/Meta-Llama-3.1-70B")

# Load from local directory (previously downloaded)
llm = LLM(model="/path/to/local/model")
```

Models are automatically downloaded when specified by their HuggingFace model ID.

### Checkpoint Conversion Workflow

For models requiring the TensorRT engine compilation path:

1. **Convert**: Transform weights from source framework (HuggingFace, NeMo, DeepSpeed, JAX, ModelOpt) into TensorRT-LLM checkpoint format
2. **Build**: Compile the checkpoint into a TensorRT engine using `trtllm-build`
3. **Deploy**: Load the engine for inference

### Supported Source Frameworks

- **HuggingFace Transformers** (primary and most common)
- **NVIDIA NeMo**
- **Microsoft DeepSpeed**
- **JAX**
- **NVIDIA ModelOpt** (for pre-quantized models)

### Checkpoint Format

TensorRT-LLM checkpoints consist of:
- A `config.json` with model hyperparameters (architecture, dtype, vocab size, layer count, quantization settings)
- One or more rank weight files in `safetensors` format (one per GPU rank in multi-GPU configurations)

### HuggingFace Quantized Models

NVIDIA provides pre-quantized models on HuggingFace Hub in FP4 and FP8 formats, ready for use with TensorRT-LLM. The `trust_remote_code` parameter is supported for custom HuggingFace model architectures (added in v0.13.0).

---

## 6. How New Models Are Added

### Four-Step Process

Adding a new decoder-only model to TensorRT-LLM follows a structured process:

**Step 1: Create Model Architecture**
- Create a new directory under `tensorrt_llm/models/` (e.g., `my_model/`)
- Implement `model.py` using TensorRT-LLM's API hierarchy:
  - Low-level functions: `concat`, `add`, `sum`
  - Basic layers: `Linear`, `LayerNorm`
  - High-level layers: `MLP`, `Attention`
  - Base class: `DecoderModelForCausalLM`
- Define decoder layers with `LayerNorm`, `Attention`, and `MLP` components

**Step 2: Implement Weight Conversion**
- Create a `from_hugging_face` classmethod that:
  - Creates a TensorRT-LLM model instance
  - Converts HuggingFace checkpoint weights to TensorRT-LLM format
  - Loads converted weights into the model
- Optionally create a standalone `convert_checkpoint.py` script under `examples/my_model/`

**Step 3: Register the Model**
- Register the new model class in `tensorrt_llm/models/__init__.py`

**Step 4: Verify Functionality**
- Run checkpoint conversion
- Build the engine with `trtllm-build`
- Test inference with sample prompts
- Run benchmarks (e.g., summarization)

### Community Contributions

The repository accepts community contributions for new model support. Notable community-contributed models include InternLM2 and distil-whisper.

---

## 7. Quantization Support

### Supported Quantization Methods

| Method | Description |
|---|---|
| FP8 | 8-bit floating point (Hopper/Ada Lovelace and newer) |
| FP4 | 4-bit floating point (Blackwell and newer) |
| INT8 SmoothQuant | Per-channel/per-token INT8 quantization |
| INT4 AWQ | Activation-aware Weight Quantization |
| INT4 GPTQ | GPTQ-based 4-bit quantization |
| W8A16 | 8-bit weights, 16-bit activations |
| W4A16 | 4-bit weights, 16-bit activations |
| W4A16 AWQ | AWQ variant |
| W4A8 AWQ | Mixed precision AWQ |
| W4A16 GPTQ | GPTQ variant |
| W8A8 SQ per-channel | SmoothQuant per-channel |

### KV Cache Quantization

- FP8 KV cache
- INT8 KV cache

### Precision by GPU Architecture

| GPU Architecture | Supported Precisions |
|---|---|
| Blackwell (SM100+) | FP32, FP16, BF16, FP8, FP4, INT8, INT4 |
| Hopper (SM90) | FP32, FP16, BF16, FP8, INT8, INT4 |
| Ada Lovelace (SM89) | FP32, FP16, BF16, FP8, INT8, INT4 |
| Ampere (SM80/SM86) | FP32, FP16, BF16, INT8, INT4 |

**Important limitation**: Not all quantization types are supported for all models. Users must consult model-specific documentation/examples to verify quantization compatibility.

---

## 8. Limitations on Model Support

### General Limitations

1. **Quantization coverage is not universal**: FP8 and quantized data types (INT8, INT4) are not implemented for all models. Each model's example directory documents which quantization methods are available.

2. **Backend disparity**: The PyTorch backend and TensorRT backend support different model sets. Some models are only available on one backend. The PyTorch backend is the actively developed path for new features.

3. **Encoder-decoder limitations**: Models like T5, BART, and mBART have more limited feature support compared to decoder-only models (e.g., paged KV cache for encoder-decoder was added later and initially limited to beam width 1).

4. **Linear weight transposition**: TensorRT-LLM checkpoints require linear weights in `(out_feature, in_feature)` shape, which may differ from some source frameworks. The build process handles transposition automatically.

5. **Hardware requirements**: FP8 requires Hopper (H100) or Ada Lovelace (L40/RTX 4090) or newer GPUs. FP4 requires Blackwell (B100/B200) GPUs.

6. **License compliance**: Some models (e.g., Llama family) require accepting license agreements on HuggingFace before downloading.

7. **Feature availability varies by model**: Advanced features like speculative decoding (EAGLE-3, Medusa), Multi-Token Prediction (MTP), disaggregated serving, and sliding window attention are only supported on select models.

### PyTorch Backend Feature Matrix (Key Models)

Features like Overlap Scheduler, CUDA Graph, Attention Data Parallelism, Disaggregated Serving, Chunked Prefill, MTP, EAGLE-3, KV Cache Reuse, Sliding Window Attention, and Guided Decoding vary in support across models. For example:
- DeepSeek-V3 supports most features except EAGLE-3 two-model engine
- Qwen3MoE shows broad support including both EAGLE-3 variants
- Some model-feature combinations are marked "Untested" or "N/A"

---

## 9. Recent Additions (2024-2025 Timeline)

### v0.9.0 (April 2024)
- StarCoder2, VILA, Smaug-72B, distil-whisper
- FP8 FMHA (experimental), T5 and Mixtral 8x7B out-of-the-box

### v0.10.0 (June 2024)
- **Major additions**: LLaMA 3, DBRX, Qwen2, CogVLM, Arctic, Fuyu, Persimmon, Deplot, Kosmos-2, ByT5, NeVA, RecurrentGemma
- Weight-stripping and weight-streaming features
- Paged KV cache for encoder-decoder models

### v0.11.0 (August 2024)
- **Major additions**: Grok-1, Phi-3 variants (with block sparse attention), InternLM2 (7B/20B), Video NeVA, VILA 1.5, Jais, DiT
- Qwen1.5-110B FP8, Qwen1.5 MoE A2.7B
- FP8 LLaMA with FP16 LoRA, inflight batching for encoder-decoder models

### v0.12.0 (September 2024)
- **Major additions**: LLaMA 3.1, Mamba-2, EXAONE, Qwen 2 (via LLM class), GLM4
- LLaVA-NeXT multimodal support, ReDrafter speculative decoding
- FP8 FMHA for Ada Lovelace, FP8 MoE out-of-the-box

### v0.13.0 (September 2024)
- **Major addition**: Gemma 2
- Whisper in C++ runtime, Lookahead decoding (experimental)
- Tensor parallelism for Mamba2, `trust_remote_code` for custom HuggingFace models

### v1.1.0 (December 2024)
- **Major additions**: GPT-OSS, Hunyuan-Dense, Hunyuan-MoE, Seed-OSS

### v1.2.0rc / v1.3.0rc Series (January-February 2025)
- **Major additions and enhancements**:
  - DeepSeek-V3, DeepSeek-V3.2, DeepSeek-R1 (with tool parsing)
  - Qwen3-VL MoE, GLM-4.5-Air, GLM-4.7 Flash
  - Mistral Large 3 VLM, Eagle3 on Mistral Large 3
  - Nemotron Super with MTP, Nemotron-H with Eagle3
  - K-EXAONE with MTP, EXAONE 4.0
  - Llama 4 (40,000+ tokens/second on B200)
  - Llama 3.3-70B LoRA BF16 support
  - EPD disaggregation for multiple models
  - TRTLLM MoE as default backend on Blackwell

---

## 10. Supported Hardware

| GPU Architecture | Products | Key Capabilities |
|---|---|---|
| Blackwell (SM100+) | B100, B200, GB200 NVL72 | FP4, FP8, full precision range |
| Hopper (SM90) | H100, H200 | FP8, INT8, INT4 |
| Grace Hopper | GH200 Superchip | CPU+GPU unified memory |
| Ada Lovelace (SM89) | L40, L40S, RTX 4090 | FP8, INT8, INT4 |
| Ampere (SM80/86) | A100, A30, A10, RTX 3090 | INT8, INT4 (no FP8) |

**Operating Systems**: Linux x86_64 or Linux aarch64

**Software Stack** (as of v1.3.x):
- TensorRT 10.11
- CUDA 12.x
- PyTorch 2.4+
- Container: NVIDIA 25.06

---

## 11. Summary Statistics

| Category | Count |
|---|---|
| Total distinct model architectures | ~93+ |
| PyTorch backend language models | ~23 architecture classes |
| PyTorch backend multimodal models | ~13 architecture classes |
| TensorRT backend LLM models | ~48 architectures |
| TensorRT backend multimodal models | ~16 architectures |
| MoE model architectures | 13+ |
| Supported GPU generations | 5 (Ampere through Blackwell) |
| Quantization methods | 10+ |
| Source framework conversion paths | 5 (HuggingFace, NeMo, DeepSpeed, JAX, ModelOpt) |

---

## 12. Key Takeaways

1. **Breadth**: TensorRT-LLM covers virtually all major open-weight LLM families (Llama, Mistral, Qwen, Gemma, Falcon, BLOOM, GPT variants, Phi, DeepSeek, etc.) plus many specialized and multimodal models.

2. **Two backends**: The PyTorch backend is the modern path with active development; the TensorRT backend provides the widest legacy model coverage. NVIDIA is converging new features on the PyTorch backend.

3. **MoE first-class support**: Extensive MoE support including expert parallelism, FP8/FP4 quantization for experts, and optimized kernels for models like DeepSeek-V3 and Mixtral.

4. **HuggingFace-native**: Most models can be loaded directly from HuggingFace Hub with automatic download and conversion. Pre-quantized checkpoints are also available on the Hub.

5. **Rapid iteration**: NVIDIA maintains "Day-0" support for major new model releases (e.g., Llama 4, DeepSeek-R1, Qwen3). The 2024-2025 period saw particularly rapid expansion of the supported model list.

6. **Extensible**: Adding new model architectures follows a well-documented four-step process, making it feasible for the community to contribute support for additional models.
