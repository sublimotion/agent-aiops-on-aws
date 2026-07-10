import os
import fire
import torch
from transformers import AutoModelForCausalLM


def main(
    model_name: str = "Qwen/Qwen3-0.6B", # choose between "Qwen/Qwen3-4B", "Qwen/Qwen2.5-3B-Instruct", and "Qwen/Qwen3-0.6B"
    output_dir: str = "./decomposed_models",
):
    output_dir = os.path.join(output_dir, model_name.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "svd_weights.pt")
    if os.path.exists(output_file):
        print(f"Model already decomposed: {output_file}")
        return

    # float32 load: original (older transformers) defaulted to fp32; transformers
    # 5.x loads the checkpoint's native bf16, and torch.svd has no bf16 CPU/CUDA
    # kernel ("linalg_svd_cpu not implemented for 'BFloat16'"). Decompose in fp32
    # to reproduce the checkpoint's SVD basis faithfully.
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    state_dict = model.state_dict()
    decomposed_weights = {}
    for k, v in state_dict.items():
        if v.ndim > 1 and all([d > 1 for d in v.shape]):
            print(f"Decomposing weight {k} with shape {v.shape}")
            U, S, V = torch.svd(v.float())
            decomposed_weights[f"{k}.U"] = U
            decomposed_weights[f"{k}.S"] = S
            decomposed_weights[f"{k}.V"] = V
        else:
            print(f"Skipping weight {k} with shape {v.shape}")
    torch.save(decomposed_weights, output_file)


if __name__ == "__main__":
    fire.Fire(main)