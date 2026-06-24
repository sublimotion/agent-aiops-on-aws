import torch
from typing import Optional, Union, List, Tuple, Unpack

from transformers.models.qwen2.modeling_qwen2 import (
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache

# Import the unified hidden state extraction utility
from fugu.hidden_state_utils import extract_hidden_state_at_position


# @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
# @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
# @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        action_layer: torch.nn.Module = None,
        hidden_state_position: int = -2,  # NEW: Configurable position, defaults to -2
        debug_hidden_extraction: bool = False,  # NEW: Debug flag
        **kwargs
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
            If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
            This is useful when using packed tensor format (single dimension for batch and sequence length).

        action_layer (`torch.nn.Module`, *optional*):
            Optional action layer to apply to hidden states for router functionality.

        hidden_state_position (`int`, *optional*):
            Position to extract hidden state from (default -2 for second-to-last token).
            This replaces the previous trinity-dependent logic.

        debug_hidden_extraction (`bool`, *optional*):
            Whether to print debugging information for hidden state extraction.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

    >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    if action_layer is not None:
        # UPDATED: Use unified hidden state extraction, removing trinity dependency
        try:
            extracted_hidden = extract_hidden_state_at_position(
                hidden_states=hidden_states,
                position=hidden_state_position,
                debug=debug_hidden_extraction,
                context_name="modeling_qwen2"
            )
            return action_layer(extracted_hidden)
        except ValueError as e:
            if debug_hidden_extraction:
                print(f"[modeling_qwen2] Hidden state extraction failed: {e}")
                print(f"[modeling_qwen2] Falling back to last available position (-1)")
            # Fallback to last position if requested position is out of bounds
            fallback_hidden = extract_hidden_state_at_position(
                hidden_states=hidden_states,
                position=-1,
                debug=debug_hidden_extraction,
                context_name="modeling_qwen2_fallback"
            )
            return action_layer(fallback_hidden)

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )