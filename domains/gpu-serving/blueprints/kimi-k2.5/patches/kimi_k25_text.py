# SPDX-License-Identifier: Apache-2.0
# Text-only wrapper for Kimi K2.5 (KimiK25ForConditionalGeneration)
# Loads only the language model (DeepseekV3-based MoE) backbone,
# skipping vision_tower and mm_projector weights.
#
# This enables SGLang HiCache benchmarking on Kimi K2.5 text generation
# without requiring full multimodal support.

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.utils import add_prefix


class KimiK25ForConditionalGeneration(nn.Module):
    """Text-only wrapper for Kimi K2.5 multimodal model.

    Instantiates only the language_model (DeepseekV2/V3 MoE backbone)
    and ignores vision_tower / mm_projector weights.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        # Use text_config if present (multimodal wrapper), else use config directly
        text_config = getattr(config, "text_config", config)
        self.language_model = DeepseekV2ForCausalLM(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        return self.language_model.forward(
            input_ids, positions, forward_batch, get_embedding
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Filter out vision and projector weights, pass language model weights through
        lm_weights = []
        for args in weights:
            name = args[0]
            if name.startswith("vision_tower.") or name.startswith("mm_projector."):
                continue
            lm_weights.append(args)
        self.language_model.load_weights(lm_weights)

    def get_hidden_dim(self, module_name):
        if hasattr(self.language_model, "get_hidden_dim"):
            return self.language_model.get_hidden_dim(module_name)
        return None


EntryClass = KimiK25ForConditionalGeneration
