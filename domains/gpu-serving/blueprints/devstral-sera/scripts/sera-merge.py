#!/usr/bin/env python3
"""
Merge LoRA adapters into base model and optionally quantize to FP8.

Usage:
  # Merge only
  python3 sera-merge.py --base-model /mnt/nvme/models/devstral-small-2-bf16 \
      --lora-path /mnt/nvme/sera-checkpoints/run-001 \
      --output /mnt/nvme/models/devstral-small-2-sera-bf16

  # Merge + quantize to FP8
  python3 sera-merge.py --base-model /mnt/nvme/models/devstral-small-2-bf16 \
      --lora-path /mnt/nvme/sera-checkpoints/run-001 \
      --output /mnt/nvme/models/devstral-small-2-sera-fp8 \
      --quantize fp8
"""

import argparse
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def merge_lora(base_model: str, lora_path: str, output: str):
    """Merge LoRA adapters into base model."""
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    log.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # merge on CPU to avoid OOM
        trust_remote_code=True,
    )

    log.info(f"Loading LoRA adapters: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    log.info("Merging LoRA into base model...")
    model = model.merge_and_unload()

    log.info(f"Saving merged model to: {output}")
    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    log.info("Merge complete")
    return output


def quantize_fp8(model_path: str, output: str):
    """Quantize merged model to FP8 using AutoFP8."""
    try:
        from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig
    except ImportError:
        log.error("auto_fp8 not installed. Install with: pip install auto-fp8")
        log.info("Alternatively, vLLM can dynamically quantize BF16 to FP8 at load time.")
        return

    log.info(f"Quantizing {model_path} to FP8...")
    quantize_config = BaseQuantizeConfig(quant_method="fp8", activation_scheme="dynamic")

    model = AutoFP8ForCausalLM.from_pretrained(model_path, quantize_config)
    model.quantize()

    log.info(f"Saving FP8 model to: {output}")
    os.makedirs(output, exist_ok=True)
    model.save_quantized(output)
    log.info("FP8 quantization complete")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters and optionally quantize")
    parser.add_argument("--base-model", required=True, help="Base model path (BF16)")
    parser.add_argument("--lora-path", required=True, help="LoRA checkpoint path")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--quantize", choices=["fp8", "none"], default="none",
                        help="Quantization method after merge")
    args = parser.parse_args()

    start = time.time()

    # Merge
    merged_path = merge_lora(args.base_model, args.lora_path, args.output)

    # Optionally quantize
    if args.quantize == "fp8":
        fp8_output = args.output.replace("-bf16", "-fp8") if "-bf16" in args.output else args.output + "-fp8"
        quantize_fp8(merged_path, fp8_output)
        log.info(f"FP8 model saved to: {fp8_output}")

    elapsed = time.time() - start
    log.info(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
