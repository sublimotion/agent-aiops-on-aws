"""
Unified hidden state extraction utilities.

This module provides consistent hidden state extraction across the codebase,
removing dependencies on trinity mode for position selection and ensuring
all extraction uses the same approach.
"""

import torch
from typing import Union, Tuple, Optional


def extract_hidden_state_at_position(
        hidden_states: torch.Tensor,
        position: int = -2,
        debug: bool = False,
        context_name: str = "unknown"
) -> torch.Tensor:
    """
    Extract hidden state at a specific position from transformer outputs.

    This function provides unified hidden state extraction that works consistently
    across different parts of the codebase, removing dependency on trinity mode
    and using approach 2 (full forward pass) methodology.

    Args:
        hidden_states: Hidden states tensor of shape [batch_size, seq_len, hidden_dim]
        position: Position to extract from (default -2 for second-to-last token)
        debug: Whether to print debugging information
        context_name: Name of the calling context for debug messages

    Returns:
        torch.Tensor: Hidden state at the specified position [batch_size, hidden_dim]

    Raises:
        ValueError: If the position is out of bounds for the sequence
    """
    if debug:
        print(f"[{context_name}] extract_hidden_state_at_position called:")
        print(f"  Input shape: {hidden_states.shape}")
        print(f"  Requested position: {position}")

    batch_size, seq_len, hidden_dim = hidden_states.shape

    # Convert negative position to positive index
    if position < 0:
        actual_index = seq_len + position
    else:
        actual_index = position

    if debug:
        print(f"  Sequence length: {seq_len}")
        print(f"  Actual index: {actual_index}")
        print(f"  Available indices: 0 to {seq_len - 1}")

    # Check bounds
    if actual_index < 0 or actual_index >= seq_len:
        if debug:
            print(f"  ERROR: Position {position} (index {actual_index}) is out of bounds")
        raise ValueError(
            f"Position {position} (index {actual_index}) is out of bounds for sequence length {seq_len}. "
            f"Available range: {-seq_len} to {seq_len - 1}"
        )

    # Extract hidden state
    extracted_hidden = hidden_states[:, actual_index, :]

    if debug:
        print(f"  Successfully extracted hidden state:")
        print(f"    Output shape: {extracted_hidden.shape}")
        print(f"    Sample values: {extracted_hidden[0, :5].tolist() if extracted_hidden.numel() > 0 else 'empty'}")
        print(f"    Position represents: token at index {actual_index} in sequence")

    return extracted_hidden


def extract_hidden_state_from_generation_outputs(
        generation_outputs,
        tokenizer,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position: int = -2,
        debug: bool = False,
        context_name: str = "generation"
) -> torch.Tensor:
    """
    Extract hidden state from generation outputs using approach 2 (full forward pass).

    This method performs a separate forward pass on the complete generated sequence
    to extract hidden states, which provides access to all positions including the
    actual last token.

    Args:
        generation_outputs: Output from model.generate() call
        tokenizer: Tokenizer used for generation
        model: The transformer model
        input_ids: Original input token IDs
        attention_mask: Original attention mask
        position: Position to extract from (default -2)
        debug: Whether to print debugging information
        context_name: Name of the calling context for debug messages

    Returns:
        torch.Tensor: Hidden state at the specified position [batch_size, hidden_dim]
    """
    if debug:
        print(f"[{context_name}] extract_hidden_state_from_generation_outputs called:")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Requested position: {position}")

    # Get the complete generated sequence
    if hasattr(generation_outputs, 'sequences'):
        generated_sequence = generation_outputs.sequences[0]  # [full_seq_len]
    else:
        generated_sequence = generation_outputs[0]

    if debug:
        input_length = input_ids.shape[1]
        generated_length = generated_sequence.shape[0]
        new_tokens = generated_length - input_length
        print(f"  Input length: {input_length}")
        print(f"  Generated length: {generated_length}")
        print(f"  New tokens: {new_tokens}")

    # Create attention mask for full sequence
    full_attention_mask = torch.ones_like(generated_sequence).unsqueeze(0)

    # Forward pass on complete generated sequence (approach 2)
    with torch.no_grad():
        complete_outputs = model(
            input_ids=generated_sequence.unsqueeze(0),
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

    # Extract hidden states from the last layer
    complete_hidden_states = complete_outputs.hidden_states[-1]  # [1, full_seq_len, hidden_dim]

    if debug:
        print(f"  Complete hidden states shape: {complete_hidden_states.shape}")

    # Use the unified extraction function
    return extract_hidden_state_at_position(
        complete_hidden_states,
        position=position,
        debug=debug,
        context_name=f"{context_name}_generation"
    )


def get_last_token_hidden_state(
        hidden_states: torch.Tensor,
        last_token_predict: bool = False,
        model=None,
        tokenizer=None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        generation_outputs=None,
        position: int = -2,
        debug: bool = False,
        context_name: str = "unknown"
) -> torch.Tensor:
    """
    Get the appropriate hidden state based on last_token_predict setting.

    Args:
        hidden_states: Hidden states from input sequence (for last_token_predict=False)
        last_token_predict: Whether to use last token prediction mode
        model: Model for generation mode (required if last_token_predict=True)
        tokenizer: Tokenizer (required if last_token_predict=True)
        input_ids: Input token IDs (required if last_token_predict=True)
        attention_mask: Attention mask (required if last_token_predict=True)
        generation_outputs: Generation outputs (required if last_token_predict=True)
        position: Position to extract from (default -2)
        debug: Whether to print debugging information
        context_name: Name of the calling context for debug messages

    Returns:
        torch.Tensor: Appropriate hidden state based on mode
    """
    if debug:
        print(f"[{context_name}] get_last_token_hidden_state called:")
        print(f"  last_token_predict: {last_token_predict}")
        print(f"  position: {position}")

    if last_token_predict:
        # Use generation outputs to extract from complete sequence
        if generation_outputs is None or model is None or tokenizer is None or input_ids is None or attention_mask is None:
            raise ValueError(
                "For last_token_predict=True, must provide model, tokenizer, input_ids, attention_mask, and generation_outputs")

        return extract_hidden_state_from_generation_outputs(
            generation_outputs=generation_outputs,
            tokenizer=tokenizer,
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position=position,
            debug=debug,
            context_name=context_name
        )
    else:
        # Use input sequence hidden states
        return extract_hidden_state_at_position(
            hidden_states=hidden_states,
            position=position,
            debug=debug,
            context_name=context_name
        )