"""
Infrastructure module for multi-agent reasoning framework.

This module provides the core components for loading models, tasks, and coordinating
agent interactions, independent of the specific training strategy.

Key features:
- Centralized worker context management through WorkerContext class
- Standardized model evaluation through EvaluationManager
- Unified parameter handling with SVDParameterManager
- Clear infrastructure interfaces for algorithm implementations
- Token usage tracking for cost analysis
- Multi-GPU worker distribution support
- Trinity mode support for role-based agent selection
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any
from contextlib import nullcontext
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from fugu.model_mods.modeling_qwen2 import forward as qwen2_forward
from fugu.run_tasks import create_task
from fugu.utils import track_episode_tokens, configure_split_dir, InfrastructureFailure
from fugu.head_modules import create_router_head
from fugu.hidden_state_utils import get_last_token_hidden_state

@dataclass
class WorkerContext:
    """Context object for worker processes to replace global variables."""

    # Model components
    router_model: Optional[nn.Module] = None
    tokenizer: Optional[AutoTokenizer] = None
    linear_layer: Optional[nn.Module] = None
    task_instance: Optional[Any] = None

    # Configuration
    debug: bool = False
    debug_log_dir: Optional[str] = None
    log_dir: Optional[str] = None
    task_name: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    max_turns: Optional[int] = None
    router_model_name: Optional[str] = None
    llm_names: Optional[List[str]] = None
    servers: Dict[str, str] = None
    ports: Dict[str, int] = None
    test_split_enabled: bool = False
    valid_ratio: float = 0.5
    test_ratio: float = 0.2
    seed: int = 42
    use_consultant: bool = True
    use_verifier: bool = False
    trinity: bool = False  # NEW: Trinity mode flag
    last_token_predict: bool = False
    assigned_gpu: int = 1
    head_type: str = "linear"

    def __post_init__(self):
        """Initialize empty dictionaries if not provided."""
        if self.servers is None:
            self.servers = {}
        if self.ports is None:
            self.ports = {}

    def initialize_from_config(self, config: Dict[str, Any]) -> None:
        """Initialize context from configuration dictionary."""
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            # Special case for model_name -> router_model_name
            elif key == "model_name":
                self.router_model_name = value
            # Ensure special parameters are always set
            elif key in ["valid_ratio", "test_ratio", "use_consultant", "log_dir", "trinity"]:
                setattr(self, key, value)

        # Additional safety check to ensure log_dir is never None
        if not hasattr(self, 'log_dir') or self.log_dir is None:
            import tempfile
            import os
            self.log_dir = tempfile.mkdtemp(prefix="worker_log_")
            print(f"Warning: log_dir was None, using temporary directory: {self.log_dir}")


class SVDParameterManager:
    """Manager for SVD-decomposed model parameters."""

    @staticmethod
    def compose_model_weights(learnable_params: Dict, svd_weights: Dict) -> Dict:
        """Compose model weights from SVD components."""
        composed = {}
        for k, v in learnable_params.items():
            if k == "action_layer.weight":
                continue
            U = svd_weights[k + ".U"]
            V = svd_weights[k + ".V"]
            S = svd_weights[k + ".S"]
            scale = v + 1
            composed[k] = (U @ torch.diag_embed(S * scale) @ V.T) * (
                    S.sum() / (S * scale).sum()
            )
        return composed

    @staticmethod
    @torch.no_grad()
    def load_model_weights(model: nn.Module, learnable_params: Dict, svd_weights: Dict):
        """Load model weights into model."""
        new_w = SVDParameterManager.compose_model_weights(learnable_params, svd_weights)
        for k, v in new_w.items():
            model.get_parameter(k).copy_(v)
        return new_w

    @staticmethod
    def backpropagate_gradients(model: nn.Module, learnable_params: Dict, svd_weights: Dict):
        """Backpropagate gradients to learnable parameters."""
        new_w = SVDParameterManager.compose_model_weights(learnable_params, svd_weights)
        for k in learnable_params:
            if k != "action_layer.weight":
                new_w[k].backward(model.get_parameter(k).grad)

    @staticmethod
    def load_svd_weights(model_name: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Load SVD weights for a model."""
        svd_file = os.path.join(
            "decomposed_models",
            model_name.replace("/", "_"),
            "svd_weights.pt"
        )
        if not os.path.exists(svd_file):
            raise FileNotFoundError(f"SVD file not found: {svd_file}")
        return torch.load(svd_file, map_location=device)

    @staticmethod
    def filter_svd_weights_by_layers(
            svd_weights: Dict[str, torch.Tensor],
            layer_indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Filter SVD weights to only include specified layers."""
        if layer_indices is None:
            return svd_weights

        # Special case - if 999 is in layer_indices, exclude all SVD parameters
        if 999 in layer_indices:
            return {}

        filtered_weights = {}
        layer_count = {}

        for key, tensor in svd_weights.items():
            # Extract base parameter name (remove the .U, .S, .V suffix)
            base_key = key.rsplit(".", 1)[0] if "." in key else key

            # Include if it's a non-transformer layer or in the specified indices
            if "model.layers." not in base_key:
                filtered_weights[key] = tensor
                continue

            # Check if this layer should be included
            for idx in layer_indices:
                if f"model.layers.{idx}." in base_key:
                    filtered_weights[key] = tensor
                    layer_count[idx] = layer_count.get(idx, 0) + 1
                    break

        return filtered_weights


class EvaluationManager:
    """Manager for model evaluation across worker processes."""

    @staticmethod
    def get_action(
            model: nn.Module,
            linear_layer: nn.Module,
            tokenizer: AutoTokenizer,
            messages: List[Dict],
            inference: bool = True,
            last_token_predict: bool = True,
            max_tokens: int = 2048,
            temperature: float = 0.1,
            turn_num: int = 0,
            hidden_state_position: int = -2,  # NEW: Configurable position, defaults to -2
            debug_hidden_extraction: bool = False,  # this flag is currently hard coded to this value
    ) -> torch.Tensor:
        """Feed messages through router backbone + linear head."""

        if debug_hidden_extraction:
            print(
                f"[get_action] Called with last_token_predict={last_token_predict}, inference={inference}, turn_num={turn_num}")
            print(f"[get_action] Model path: {getattr(model, 'name_or_path', 'unknown')}")
            print(f"[get_action] Hidden state position: {hidden_state_position}")

        try:
            # NOTE: Disable thinking mode of qwen3 model for now, generation is too slow now.
            if "qwen3" in model.name_or_path.lower():
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
                if debug_hidden_extraction:
                    print(f"[get_action] Using qwen3 with thinking disabled")
            else:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                if debug_hidden_extraction:
                    print(f"[get_action] Using standard chat template")

            # Fix: Get both input_ids and attention_mask
            if debug_hidden_extraction:
                print(f"[get_action] About to tokenize text...")
            tokenized = tokenizer(text, return_tensors="pt")
            input_ids = tokenized.input_ids.to(model.device)
            attention_mask = tokenized.attention_mask.to(model.device)
            if debug_hidden_extraction:
                print(f"[get_action] Tokenized input shape: {input_ids.shape}, device: {model.device}")

            def _execute_without_generation(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                                            action_layer: nn.Module, turn_num: int) -> torch.Tensor:
                """Use the first predicted token's hidden state for action layer."""
                if debug_hidden_extraction:
                    print(f"[get_action] Using _execute_without_generation path (last_token_predict=False)")
                    print(f"[get_action] Input shape: {input_ids.shape}")

                try:
                    # Get hidden states from input sequence
                    with torch.no_grad() if inference else nullcontext():
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True
                        )

                        # UPDATED: Use unified hidden state extraction
                        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
                        extracted_hidden = get_last_token_hidden_state(
                            hidden_states=hidden_states,
                            last_token_predict=False,  # Using input sequence
                            position=hidden_state_position,
                            debug=debug_hidden_extraction,
                            context_name="get_action_no_generation"
                        )

                        # Pass turn_num to action layer if it supports it
                        if hasattr(action_layer, 'forward') and hasattr(action_layer.forward,
                                                                        '__code__') and 'turn_num' in action_layer.forward.__code__.co_varnames:
                            if debug_hidden_extraction:
                                print(f"[get_action] Action layer supports turn_num, passing turn_num={turn_num}")
                            action = action_layer(extracted_hidden, turn_num=turn_num)
                        else:
                            if debug_hidden_extraction:
                                print(f"[get_action] Action layer does not support turn_num")
                            action = action_layer(extracted_hidden)

                    if debug_hidden_extraction:
                        print(
                            f"[get_action] Action output shape: {action.shape if hasattr(action, 'shape') else type(action)}")

                    if inference:
                        result = action.float().cpu().numpy().squeeze()
                        if debug_hidden_extraction:
                            print(f"[get_action] Converted to numpy with shape: {result.shape}")
                        return result
                    else:
                        if debug_hidden_extraction:
                            print(f"[get_action] Returning tensor with gradients")
                        return action.squeeze()  # Keep as tensor with gradients

                except Exception as e:
                    if debug_hidden_extraction:
                        print(f"[get_action] ERROR in _execute_without_generation: {e}")
                        import traceback
                        traceback.print_exc()
                    raise

            def _execute_with_generation(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                                         action_layer: nn.Module, max_tokens: int, temperature: float,
                                         turn_num: int) -> torch.Tensor:
                """Use the last generated token's hidden state for action layer."""
                if debug_hidden_extraction:
                    print(f"[get_action] Using _execute_with_generation path (last_token_predict=True)")

                ROUTER_GENERATION_TOKENS = 2048  # Hardcoded

                if debug_hidden_extraction:
                    print(
                        f"[get_action] Input shape: {input_ids.shape}, using fixed router tokens: {ROUTER_GENERATION_TOKENS}, temperature: {temperature}")

                try:
                    generation_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "return_dict_in_generate": True,
                        "output_hidden_states": True,
                        "max_new_tokens": ROUTER_GENERATION_TOKENS,  # FIXED: Use hardcoded value
                        "temperature": temperature,
                        "do_sample": True if temperature > 0 else False,
                    }

                    # Add pad_token_id if not set to avoid warnings
                    if tokenizer.pad_token_id is None:
                        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

                    if debug_hidden_extraction:
                        print(f"[get_action] Starting generation with fixed {ROUTER_GENERATION_TOKENS} tokens...")

                    outputs = model.generate(**generation_kwargs)
                    if debug_hidden_extraction:
                        print(f"[get_action] Generation completed successfully")

                    # DEBUG: Check generation results
                    generated_sequence = outputs.sequences[0]
                    input_length = input_ids.shape[1]
                    new_tokens = generated_sequence.shape[0] - input_length
                    if debug_hidden_extraction:
                        print(
                            f"[generation_debug] Input length: {input_length}, Generated length: {generated_sequence.shape[0]}, New tokens: {new_tokens}")

                    # UPDATED: Use unified hidden state extraction with generation outputs
                    extracted_hidden = get_last_token_hidden_state(
                        hidden_states=None,  # Not needed for generation mode
                        last_token_predict=True,  # Using generation mode
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_outputs=outputs,
                        position=hidden_state_position,
                        debug=debug_hidden_extraction,
                        context_name="get_action_generation"
                    )

                    # Pass turn_num to action layer if it supports it
                    if hasattr(action_layer, 'forward') and hasattr(action_layer.forward,
                                                                    '__code__') and 'turn_num' in action_layer.forward.__code__.co_varnames:
                        if debug_hidden_extraction:
                            print(f"[get_action] Action layer supports turn_num, passing turn_num={turn_num}")
                        result = action_layer(extracted_hidden, turn_num=turn_num).float().cpu().numpy().squeeze()
                    else:
                        if debug_hidden_extraction:
                            print(f"[get_action] Action layer does not support turn_num")
                        if inference:
                            result = action_layer(extracted_hidden).float().cpu().numpy().squeeze()
                        else:
                            result = action_layer(extracted_hidden).squeeze()  # Keep as tensor

                    if debug_hidden_extraction:
                        print(
                            f"[get_action] Final action result shape: {result.shape if hasattr(result, 'shape') else type(result)}")
                    return result

                except Exception as e:
                    if debug_hidden_extraction:
                        print(f"[get_action] ERROR in _execute_with_generation: {e}")
                        import traceback
                        traceback.print_exc()
                    raise

            context = torch.no_grad() if inference else nullcontext()
            if debug_hidden_extraction:
                print(f"[get_action] About to enter inference context...")

            with context:
                if last_token_predict:
                    if debug_hidden_extraction:
                        print(f"[get_action] Taking generation path (last_token_predict=True)")
                    result = _execute_with_generation(input_ids, attention_mask, linear_layer, max_tokens, temperature,
                                                      turn_num)
                else:
                    if debug_hidden_extraction:
                        print(f"[get_action] Taking non-generation path (last_token_predict=False)")
                    result = _execute_without_generation(input_ids, attention_mask, linear_layer, turn_num)

            if debug_hidden_extraction:
                print(
                    f"[get_action] get_action returning result with shape: {result.shape if hasattr(result, 'shape') else type(result)}")
                print(f"[get_action] Result is None: {result is None}")
            return result

        except Exception as e:
            if debug_hidden_extraction:
                print(f"[get_action] CRITICAL ERROR in get_action: {e}")
                print(f"[get_action] Exception type: {type(e)}")
                import traceback
                traceback.print_exc()
                print(f"[get_action] Returning None due to exception")
            return None  # This is probably what's happening

    @staticmethod
    def _mk_iter_dir(root: str, it: int) -> str:
        """Create and return iteration directory."""
        path = os.path.join(root, f"iter_{it}")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def evaluate_episode_trinity(
            context: WorkerContext,
            task_id: int,
            split: str,
            iter_idx: int,
            eps_explore: Optional[float] = None,
            debug_log_file: Optional[str] = None
    ) -> Tuple:
        """Run a single evaluation episode with trinity mode (solver, thinker, verifier)."""
        obs = context.task_instance.reset(task_id=task_id, split=split)
        done = False
        sampled_ids = []
        sampled_role_ids = []

        while not done:
            # Log current turn and max allowed turns
            if context.debug and debug_log_file is not None:
                with open(debug_log_file, "a") as log_f:
                    turn_num = context.task_instance.num_turns
                    log_f.write(f"Turn {turn_num + 1}/{context.task_instance.max_turns}\n")
                    # Note if this is the final allowed turn
                    if turn_num == context.task_instance.max_turns:
                        log_f.write("Hitting max turns\n")

            # Get router messages
            router_messages = (
                context.task_instance._format_router_messages()
                if hasattr(context.task_instance, "_format_router_messages")
                else obs
            )

            # Log router prompt if debug enabled
            if context.debug and debug_log_file is not None:
                with open(debug_log_file, "a") as log_f:
                    log_f.write(f"Router prompt: ")
                    for msg in router_messages:
                        log_f.write(f"{msg['role']}: {msg['content']}\n")
                    log_f.write("\n")

            # Get the logits
            logits = EvaluationManager.get_action(
                context.router_model,
                context.linear_layer,
                context.tokenizer,
                router_messages,
                True,  # inference
                context.last_token_predict,
                context.max_tokens,
                context.temperature,
            )

            # Debugging purpose, fetch the agent prompt before the step
            import copy
            solver_agent_prompt = copy.deepcopy(context.task_instance._format_agent_messages())
            thinker_agent_prompt = copy.deepcopy(context.task_instance._format_thinker_messages())
            verifier_agent_prompt = copy.deepcopy(context.task_instance._format_verifier_messages())

            # Execute the logits
            obs, reward, done, obs_action, agent_id, role_id = context.task_instance.step_trinity(logits)
            sampled_ids.append(agent_id)
            sampled_role_ids.append(role_id)

            # Log the router logits, sampled agent and their reaction.
            if context.debug and debug_log_file is not None:
                with open(debug_log_file, "a") as log_f:
                    log_f.write(
                        f"Router logits : {logits}\n"
                        f"Sampled agent : {context.llm_names[agent_id] if agent_id >= 0 else 'None'}\n"
                        f"Sampled role id: {role_id}\n"
                        f"sampled_role_ids: {sampled_role_ids}\n"
                    )
                    if role_id == 0:  # Solver
                        log_f.write(f"Sampled role  : Solver \n")
                        log_f.write(f"Agent prompt:\n")
                        for msg in solver_agent_prompt:
                            log_f.write(f"{msg['role']}: {msg['content']}\n")
                        log_f.write(f"\n\n")
                        log_f.write(f"Agent response:\n")
                        log_f.write(f"{context.task_instance.response}\n")
                    elif role_id == 1:  # Thinker
                        log_f.write(f"Sampled role  : Thinker \n")
                        log_f.write(f"Agent prompt:\n")
                        for msg in thinker_agent_prompt:
                            log_f.write(f"{msg['role']}: {msg['content']}\n")
                        log_f.write(f"\n\n")
                        log_f.write(f"Agent response:\n")
                        log_f.write(f"{context.task_instance.thinker_response}\n")
                    elif role_id == 2:  # Verifier
                        log_f.write(f"Sampled role  : Verifier \n")
                        log_f.write(f"Agent prompt:\n")
                        # Only log verifier messages if there is a response to verify
                        if context.task_instance.response:
                            for msg in verifier_agent_prompt:
                                log_f.write(f"{msg['role']}: {msg['content']}\n")
                            log_f.write(f"\n")
                            log_f.write(f"Agent response:\n")
                            log_f.write(f"{context.task_instance.verifier_response}\n")
                            log_f.write(f"Is accept: {context.task_instance.verifier_is_accepted}\n")
                        else:
                            log_f.write("No response to verify\n")
                    else:  # -1 for no agent selected as hitting max_turn
                        log_f.write(f"No role or agent selected (max turns reached)\n")

                    # Mark end of the turn
                    log_f.write(f"\n\n\n")

        # Log the termination status.
        if context.debug and debug_log_file is not None:
            with open(debug_log_file, "a") as log_f:
                log_f.write("\n=== EPISODE COMPLETED ===\n")
                if (
                        hasattr(context.task_instance, 'verifier_is_accepted')
                        and context.task_instance.verifier_is_accepted is True
                ):
                    log_f.write("Reason: Verifier accepted the response.\n")
                elif not context.task_instance.response:
                    log_f.write("Reason: Select verifier but it has no response to verify.\n")
                else:
                    log_f.write("Reason: Maximum turns reached\n")
                log_f.write(f"Sampled Role IDs: {sampled_role_ids}\n")
                log_f.write(f"Final response: {context.task_instance.response}\n")
                log_f.write(f"Final reward: {reward}\n")

        token_stats = {
            "num_turns": context.task_instance.num_turns,
        }

        return (
            reward,
            context.task_instance.num_turns,
            obs_action,
            sampled_ids,
            context.task_instance.response,
            token_stats,
            sampled_role_ids,
        )

    @staticmethod
    def evaluate_episode(
            context: WorkerContext,
            task_id: int,
            split: str,
            iter_idx: int,
            eps_explore: float = 0.0,
            debug_log_file: Optional[str] = None
    ) -> Tuple:
        """Run a single evaluation episode with infrastructure failure handling."""
        try:
            # Reset task
            obs = context.task_instance.reset(task_id=task_id, split=split)
            done = False
            turn_num = 0
            sampled_ids = []
            previous_agent_id = -1
            infrastructure_failures = 0
            episode_had_infrastructure_failure = False

            # Token tracking variables
            episode_router_tokens = 0
            episode_agent_input_tokens = 0
            episode_agent_output_tokens = 0

            while not done:
                # Log current turn and max allowed turns
                if context.debug and debug_log_file is not None:
                    with open(debug_log_file, "a") as log_f:
                        log_f.write(f"Turn {turn_num + 1}/{context.task_instance.max_turns}\n")
                        if turn_num + 1 == context.task_instance.max_turns:
                            log_f.write("This is the final allowed turn\n")

                # Get router messages
                router_messages = (
                    context.task_instance._format_router_messages()
                    if hasattr(context.task_instance, "_format_router_messages")
                    else obs
                )

                # Log router prompt if debug enabled
                if context.debug and debug_log_file is not None:
                    with open(debug_log_file, "a") as log_f:
                        log_f.write(f"Router prompt: ")
                        for msg in router_messages:
                            log_f.write(f"{msg['role']}: {msg['content']}\n")

                # Get router action with turn information
                logits = EvaluationManager.get_action(
                    context.router_model,
                    context.linear_layer,
                    context.tokenizer,
                    router_messages,
                    True,  # inference
                    context.last_token_predict,
                    context.max_tokens,
                    context.temperature,
                    turn_num=turn_num
                )

                # Compute softmax probabilities
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                if eps_explore > 0:
                    probs = (1.0 - eps_explore) * probs + eps_explore / len(probs)
                    probs /= probs.sum()

                # Get predicted agent ID
                predicted_agent_id = np.random.choice(range(len(probs)), p=probs)
                consecutive_selection = (predicted_agent_id == previous_agent_id and previous_agent_id != -1)

                # Get agent messages before stepping
                agent_messages = context.task_instance.messages
                if hasattr(context.task_instance, "_format_agent_messages"):
                    agent_messages = context.task_instance._format_agent_messages(predicted_agent_id)

                # Log agent prompt if debug enabled
                if context.debug and debug_log_file is not None:
                    with open(debug_log_file, "a") as log_f:
                        log_f.write(f"Agent prompt: ")
                        for msg in agent_messages:
                            log_f.write(f"{msg['role']}: {msg['content']}\n")

                # Track agent model name for failure reporting
                agent_model_name = context.llm_names[predicted_agent_id]
                agent_response = context.task_instance.response if not context.use_verifier and consecutive_selection else None

                # Step the environment with the pre-selected agent ID
                try:
                    obs, reward, done, obs_action = context.task_instance.step(probs, sampling=False,
                                                                               preselected_agent_id=predicted_agent_id, )

                    # Check if the step resulted in an infrastructure failure
                    if hasattr(context.task_instance, 'last_agent_response_status'):
                        if context.task_instance.last_agent_response_status == "infrastructure_failure":
                            infrastructure_failures += 1
                            episode_had_infrastructure_failure = True
                            if context.debug and debug_log_file is not None:
                                with open(debug_log_file, "a") as log_f:
                                    log_f.write(f"INFRASTRUCTURE FAILURE: Agent {agent_model_name} failed\n")

                            # Still use the 3-failure threshold for immediate abort (severe cases)
                            if infrastructure_failures >= 3:
                                raise InfrastructureFailure(
                                    agent_name=agent_model_name,
                                    episode_info=f"Episode aborted after {infrastructure_failures} infrastructure failures"
                                )

                except InfrastructureFailure:
                    # Re-raise infrastructure failures
                    raise
                except Exception as e:
                    # Other step failures - treat as infrastructure issues
                    raise InfrastructureFailure(
                        agent_name=agent_model_name,
                        episode_info=f"Step failed: {str(e)}"
                    )

                # Update tracking variables
                actual_sampled_id = predicted_agent_id
                sampled_ids.append(actual_sampled_id)

                if not consecutive_selection:
                    agent_response = context.task_instance.response

                # Track token usage for this turn
                turn_tokens = track_episode_tokens(
                    context,
                    router_messages,
                    agent_messages,
                    agent_response,
                    agent_model_name,
                    debug_log_file
                )

                episode_router_tokens += turn_tokens["router_tokens"]
                episode_agent_input_tokens += turn_tokens["agent_input_tokens"]
                episode_agent_output_tokens += turn_tokens["agent_output_tokens"]

                previous_agent_id = actual_sampled_id

                # Log completion if debug enabled
                if context.debug and debug_log_file is not None:
                    with open(debug_log_file, "a") as log_f:
                        log_f.write(f"Router logits : {logits}\n")
                        log_f.write(f"Router probs  : {probs}\n")
                        log_f.write(f"Sampled agent : {context.llm_names[actual_sampled_id]}\n")

                        if not context.use_verifier and consecutive_selection:
                            log_f.write(f"[OPTIMIZATION] Reused previous response (consecutive selection)\n")
                        else:
                            log_f.write(f"Agent response:\n{context.task_instance.response}\n")

                        if hasattr(context.task_instance, 'verifier_response'):
                            verifier_response = context.task_instance.verifier_response
                            verifier_is_accepted = context.task_instance.verifier_is_accepted
                            log_f.write(f"Verifier response: {verifier_response}\n")
                            log_f.write(f"Verifier is accepted: {verifier_is_accepted}\n")

                        if done:
                            log_f.write("\n=== EPISODE COMPLETED ===\n")
                            if not context.use_verifier and consecutive_selection:
                                log_f.write("Reason: Same agent selected consecutively\n")
                            elif context.use_verifier and hasattr(context.task_instance,
                                                                  'verifier_is_accepted') and context.task_instance.verifier_is_accepted:
                                log_f.write("Reason: Verifier accepted the response\n")
                            elif turn_num + 1 >= context.task_instance.max_turns:
                                log_f.write("Reason: Maximum turns reached\n")
                            log_f.write(f"Final reward: {reward}\n")
                            if episode_had_infrastructure_failure:
                                log_f.write(
                                    f"EPISODE MARKED FOR EXCLUSION: Had {infrastructure_failures} infrastructure failures\n")
                        log_f.write("\n")

                turn_num += 1

            # Log episode token statistics summary
            if debug_log_file is not None:
                with open(debug_log_file, "a") as log_f:
                    log_f.write("\n===== TOKEN STATISTICS SUMMARY =====\n")
                    log_f.write(f"Episode Router Tokens: {episode_router_tokens}\n")
                    log_f.write(f"Episode Agent Input Tokens: {episode_agent_input_tokens}\n")
                    log_f.write(f"Episode Agent Output Tokens: {episode_agent_output_tokens}\n")
                    log_f.write(
                        f"Total Episode Tokens: {episode_router_tokens + episode_agent_input_tokens + episode_agent_output_tokens}\n")
                    if context.task_instance.num_turns > 0:
                        avg_tokens_per_turn = (
                                                      episode_router_tokens + episode_agent_input_tokens + episode_agent_output_tokens) / context.task_instance.num_turns
                        log_f.write(f"Average Tokens Per Turn: {avg_tokens_per_turn:.2f}\n")
                    if infrastructure_failures > 0:
                        log_f.write(f"Infrastructure Failures: {infrastructure_failures}\n")
                        log_f.write(f"Episode Excluded from Scoring: {episode_had_infrastructure_failure}\n")
                    log_f.write("=======================================\n")

            # Create token statistics dictionary
            token_stats = {
                "router_tokens": episode_router_tokens,
                "agent_input_tokens": episode_agent_input_tokens,
                "agent_output_tokens": episode_agent_output_tokens,
                "total_tokens": episode_router_tokens + episode_agent_input_tokens + episode_agent_output_tokens,
                "num_turns": context.task_instance.num_turns,
                "infrastructure_failures": infrastructure_failures,
                "episode_excluded": episode_had_infrastructure_failure
            }

            # Return special marker for episodes with infrastructure failures
            final_reward = -999.0 if episode_had_infrastructure_failure else reward

            return (
                final_reward,
                context.task_instance.num_turns,
                obs_action,
                sampled_ids,
                context.task_instance.response,
                token_stats
            )

        except InfrastructureFailure as inf_failure:
            # Log infrastructure failure
            if debug_log_file is not None:
                with open(debug_log_file, "a") as log_f:
                    log_f.write(f"\n=== INFRASTRUCTURE FAILURE ===\n")
                    log_f.write(f"Agent: {inf_failure.agent_name}\n")
                    log_f.write(f"Details: {inf_failure.episode_info}\n")
                    log_f.write("===============================\n")

            # Return special infrastructure failure result
            raise inf_failure

        except Exception as e:
            print(f"ERROR in episode evaluation: {e}")
            import traceback
            traceback.print_exc()
            # Return failed result for other errors
            return (-1.0, 0, [], [], "Error occurred", {
                "router_tokens": 0,
                "agent_input_tokens": 0,
                "agent_output_tokens": 0,
                "total_tokens": 0,
                "num_turns": 0,
                "infrastructure_failures": 0,
                "episode_excluded": True
            })


# Process-local storage for worker context - replaces module globals
_worker_process_context = WorkerContext()

def _init_worker(config: Dict) -> None:
    """Worker process initializer that sets up process-local context."""
    global _worker_process_context
    _worker_process_context.initialize_from_config(config)


def _do_eval(args: Tuple) -> Tuple:
    """Worker function that evaluates a model on a single task."""
    global _worker_process_context

    # Extract arguments
    if len(args) == 6:
        task_id, split_arg, model_state_dict, linear_layer_state_dict, servers_dict, iter_idx = args
        eps_explore = 0.0
    elif len(args) == 7:
        (task_id, split_arg, model_state_dict, linear_layer_state_dict,
            servers_dict, iter_idx, eps_explore) = args
    else:
        raise ValueError("do_eval expects either 6 or 7 arguments.")

    # Set up debug logging if enabled
    debug_log_file = None
    if _worker_process_context.debug and _worker_process_context.debug_log_dir is not None:
        iter_dir = EvaluationManager._mk_iter_dir(_worker_process_context.debug_log_dir, iter_idx)
        log_filename = f"debug_{task_id}_{split_arg}.txt"
        debug_log_file = os.path.join(iter_dir, log_filename)
        with open(debug_log_file, "w") as f:
            f.write(f"Debug log for task_id: {task_id}, split: {split_arg}\n")

    # Lazy initialization of models
    if _worker_process_context.router_model is None:
        worker_pid = os.getpid()
        assigned_gpu = getattr(_worker_process_context, 'assigned_gpu', 1)
        device_str = f"cuda:{assigned_gpu}"

        # Initialize router model on assigned GPU
        _worker_process_context.router_model = AutoModelForCausalLM.from_pretrained(
            _worker_process_context.router_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_str,
        )

        # Apply Qwen2 forward patch if needed
        if "qwen" in _worker_process_context.router_model_name.lower():
            _worker_process_context.router_model.forward = qwen2_forward.__get__(
                _worker_process_context.router_model,
                type(_worker_process_context.router_model)
            )

        # Initialize tokenizer
        _worker_process_context.tokenizer = AutoTokenizer.from_pretrained(
            _worker_process_context.router_model_name
        )

        # Initialize action layer on assigned GPU
        # NEW: Trinity mode determines output dimensions
        if getattr(_worker_process_context, 'trinity', False):
            # Trinity mode: agents + 3 roles (solver, thinker, verifier)
            output_features = len(_worker_process_context.llm_names) + 3
            head_type = "linear"  # Force linear head for trinity mode
        else:
            # Normal mode: agents + consultant (if enabled)
            consultant_outputs = 1 if _worker_process_context.use_consultant else 0
            output_features = len(_worker_process_context.llm_names) + consultant_outputs
            head_type = getattr(_worker_process_context, 'head_type', 'linear')

        _worker_process_context.linear_layer = create_router_head(
            hidden_size=_worker_process_context.router_model.config.hidden_size,
            num_agents=output_features,
            head_type=head_type,
            max_turns=_worker_process_context.max_turns,
            device=device_str,
            dtype=torch.bfloat16
        )

        # Initialize task instance
        _worker_process_context.task_instance = create_task(
            _worker_process_context.task_name,
            llm_names=_worker_process_context.llm_names,
            seed=_worker_process_context.seed,
            max_tokens=_worker_process_context.max_tokens,
            temperature=_worker_process_context.temperature,
            max_turns=_worker_process_context.max_turns,
            servers=servers_dict,
            ports=_worker_process_context.ports,
            valid_ratio=_worker_process_context.valid_ratio,
            test_ratio=_worker_process_context.test_ratio,
            log_dir=getattr(_worker_process_context, 'log_dir', None),
            trinity=getattr(_worker_process_context, 'trinity', False),  # NEW: Pass trinity flag
        )

    # Load current router weights into cached model and linear layer
    _worker_process_context.router_model.load_state_dict(model_state_dict)
    _worker_process_context.linear_layer.load_state_dict(linear_layer_state_dict)

    # Add this check to ensure test split is available when needed
    if split_arg == "test" and (not hasattr(_worker_process_context.task_instance, 'data_splits') or
                               _worker_process_context.task_instance.data_splits is None or
                               "test" not in _worker_process_context.task_instance.data_splits):
        worker_pid = os.getpid()
        _worker_process_context.task_instance.data_splits = _worker_process_context.task_instance._load_data(
            seed=np.random.randint(0, 10000),
            split="train",
            validation=True,
            valid_ratio=getattr(_worker_process_context, 'valid_ratio', 0.5),
            test_split=True,
            test_ratio=getattr(_worker_process_context, 'test_ratio', 0.2)
        )
        if debug_log_file is not None:
            with open(debug_log_file, "a") as f:
                f.write(f"Loaded test split with {len(_worker_process_context.task_instance.data_splits.get('test', []))} samples\n")

    # NEW: Choose evaluation method based on trinity flag
    if getattr(_worker_process_context, 'trinity', False):
        # Use trinity evaluation
        return EvaluationManager.evaluate_episode_trinity(
            context=_worker_process_context,
            task_id=task_id,
            split=split_arg,
            iter_idx=iter_idx,
            eps_explore=eps_explore,
            debug_log_file=debug_log_file
        )
    else:
        # Use standard evaluation
        return EvaluationManager.evaluate_episode(
            context=_worker_process_context,
            task_id=task_id,
            split=split_arg,
            iter_idx=iter_idx,
            eps_explore=eps_explore,
            debug_log_file=debug_log_file
        )


class RouterInfrastructure:
    """Base infrastructure for multi-agent router training."""

    def __init__(
            self,
            task: str,
            model_name: str,
            llm_names: List[str],
            log_dir: str,
            temperature: float = 0.8,
            max_tokens: int = 512,
            max_turns: int = 5,
            servers: Union[str, Dict[str, str]] = None,
            ports: Dict[str, int] = None,
            num_workers: int = 8,
            debug: bool = False,
            debug_log_dir: Optional[str] = None,
            eval_workers: int = 2,
            test_ratio: float = 0.2,
            valid_ratio: float = 0.5,
            seed: int = 42,
            configure_splits: bool = True,
            max_samples: int = -1,
            use_consultant: bool = True,
            worker_gpu_assignments: Optional[List[int]] = None,
            head_type: str = "linear",
            skip_router_init: bool = False,
            trinity: bool = False,  # NEW: Trinity mode flag
    ):
        """Initialize the infrastructure for multi-agent router training."""
        self.task = task
        self.model_name = model_name
        self.llm_names = llm_names
        self.log_dir = log_dir
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.num_workers = num_workers
        self.debug = debug
        self.eval_workers = eval_workers
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.seed = seed
        self.max_samples = max_samples
        self.use_consultant = use_consultant
        self.worker_gpu_assignments = worker_gpu_assignments or [1] * num_workers
        self.head_type = head_type
        self.skip_router_init = skip_router_init
        self.trinity = trinity  # NEW: Store trinity flag

        if debug and worker_gpu_assignments:
            print(f"Worker GPU assignments: {worker_gpu_assignments}")
            gpu_counts = {}
            for gpu_id in worker_gpu_assignments:
                gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1
            for gpu_id, count in gpu_counts.items():
                print(f"  GPU {gpu_id}: {count} workers")

        # Set up debug logging
        self.debug_log_dir = debug_log_dir
        os.makedirs(log_dir, exist_ok=True)
        configure_split_dir(self.log_dir)

        if debug and debug_log_dir is None:
            self.debug_log_dir = os.path.join(log_dir, "debug_logs")
            os.makedirs(self.debug_log_dir, exist_ok=True)
            if configure_splits:
                configure_split_dir(self.log_dir)

        # Set up server mapping
        self.servers = self._setup_server_mapping(servers)
        self.ports = ports if ports is not None else {}

        # Prepare data sizes (will use split_seed via create_task)
        self._initialize_dataset_sizes(
            test_ratio=self.test_ratio,
            valid_ratio=self.valid_ratio
        )

        # Create parameter manager
        self.param_manager = SVDParameterManager()

    def _setup_server_mapping(self, servers: Union[str, Dict[str, str]]) -> Dict[str, str]:
        """Set up server mapping from string or dictionary."""
        server_map = {}
        if isinstance(servers, str):
            server_list = servers.split(",")
            if len(server_list) == 1:
                for m in self.llm_names:
                    server_map[m] = server_list[0].strip()
            elif len(server_list) == len(self.llm_names):
                for i, m in enumerate(self.llm_names):
                    server_map[m] = server_list[i].strip()
            else:
                raise ValueError("Server count mismatch.")
        elif isinstance(servers, dict):
            server_map = servers.copy()
        return server_map

    def _initialize_dataset_sizes(
            self,
            test_ratio: float = None,
            valid_ratio: float = None
    ):
        """Initialize dataset sizes for planning, using a deterministic split."""
        # Use class-stored ratios if not explicitly passed
        test_ratio = test_ratio if test_ratio is not None else self.test_ratio
        valid_ratio = valid_ratio if valid_ratio is not None else self.valid_ratio

        # Create a temporary task to load splits, forwarding split_seed
        temp_task = create_task(
            self.task,
            seed=self.seed,
            llm_names=self.llm_names,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            max_turns=self.max_turns,
            servers=self.servers,
            ports=self.ports,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            max_samples=self.max_samples,
            trinity=self.trinity,  # NEW: Pass trinity flag
        )

        # Force a test split so we know how many examples end up in each
        temp_task.data_splits = temp_task._load_data(
            seed=self.seed,
            split="train",
            validation=True,
            valid_ratio=valid_ratio,
            test_split=True,
            test_ratio=test_ratio
        )

        self.train_dataset_size = len(temp_task.data_splits["train"])
        self.valid_dataset_size = len(temp_task.data_splits["valid"])
        self.test_dataset_size = len(temp_task.data_splits["test"])

    def dprint(self, *args, **kwargs):
        """Debug print helper."""
        if self.debug:
            print(*args, **kwargs)

    def initialize_models(self, layer_indices: Optional[List[int]] = None):
        """Initialize router and action layer models with optional layer filtering."""
        # Skip initialization if requested (for CMA-ES)
        if self.skip_router_init:
            return None, None, None, {}

        # Load SVD weights
        svd_weights = SVDParameterManager.load_svd_weights(
            self.model_name, device="cuda:0"
        )

        # Filter SVD weights if specific layers are requested
        if layer_indices is not None:
            svd_weights = SVDParameterManager.filter_svd_weights_by_layers(
                svd_weights, layer_indices
            )
            if self.debug:
                print(f"Filtered SVD weights to include only layers: {layer_indices}")
                print(f"Remaining SVD weight components: {len(svd_weights)}")

        # Initialize router backbone
        model_config = AutoConfig.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )

        # Apply Qwen2 forward patch if needed
        if "qwen" in self.model_name.lower():
            model.forward = qwen2_forward.__get__(model, type(model))

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # NEW: Initialize action layer based on trinity mode
        if self.trinity:
            # Trinity mode: agents + 3 roles (solver, thinker, verifier)
            num_outputs = len(self.llm_names) + 3
            actual_head_type = self.head_type  # Use specified head type, not forced linear
        else:
            # Normal mode: agents + consultant (if enabled)
            consultant_outputs = 1 if self.use_consultant else 0
            num_outputs = len(self.llm_names) + consultant_outputs
            actual_head_type = self.head_type

        linear_layer = create_router_head(
            hidden_size=model_config.hidden_size,
            num_agents=num_outputs,
            head_type=actual_head_type,  # Use specified head type for all modes
            device="cuda:0",
            dtype=torch.bfloat16
        )

        return model, tokenizer, linear_layer, svd_weights

    # Convenient access to parameter manager methods
    def compose_model_weights(self, learnable_params, svd_weights):
        """Compose model weights from SVD components."""
        return SVDParameterManager.compose_model_weights(learnable_params, svd_weights)

    def load_model_weights(self, model, learnable_params, svd_weights):
        """Load model weights."""
        return SVDParameterManager.load_model_weights(model, learnable_params, svd_weights)

    def backpropagate_gradients(self, model, learnable_params, svd_weights):
        """Backpropagate gradients to learnable parameters."""
        return SVDParameterManager.backpropagate_gradients(model, learnable_params, svd_weights)

    # Access to evaluation methods
    def get_action(self, model, linear_layer, tokenizer, messages, inference=True,last_token_predict=False):
        """Get action from router model."""
        return EvaluationManager.get_action(model, linear_layer, tokenizer, messages, inference, last_token_predict)

    # Direct access to do_eval for backward compatibility
    do_eval = staticmethod(_do_eval)

class ParameterApplier:
    """Utility for applying flattened parameters to models efficiently."""

    @staticmethod
    def apply_params_to_model(
            model: torch.nn.Module,
            linear_layer: torch.nn.Module,  # Can be either RouterHead or nn.Linear
            svd_weights: Dict[str, torch.Tensor],
            flat_params: np.ndarray,
            use_structured_router: bool = False
    ) -> None:
        """
        Given flattened parameters, reconstitute SVD-based parameters
        and action-layer weights, then copy them into the model in-place.

        Both standard and structured router modes use the action layer.
        """
        torch.cuda.empty_cache()

        model_dict = model.state_dict()
        offset = 0
        svd_layers_processed = 0

        # --- step 1: apply every decomposed (SVD) param as usual ---
        for full_key in model_dict.keys():
            sv_key = f"{full_key}.S"
            if sv_key in svd_weights:
                device = model.get_parameter(full_key).device
                dtype = model.get_parameter(full_key).dtype

                S = svd_weights[sv_key].to(device, dtype)
                s_size = S.numel()

                # grab exactly s_size scaling factors
                scale_chunk = flat_params[offset: offset + s_size]
                offset += s_size
                scale_factors = torch.from_numpy(scale_chunk).to(device, dtype) + 1.0

                U = svd_weights[f"{full_key}.U"].to(device, dtype)
                V = svd_weights[f"{full_key}.V"].to(device, dtype)

                scaled_S = S * scale_factors
                new_param = (U @ torch.diag_embed(scaled_S) @ V.transpose(-1, -2)) * (
                        S.sum() / scaled_S.sum()
                )

                model.get_parameter(full_key).data.copy_(new_param)

                del U, V, S, scaled_S, new_param
                torch.cuda.empty_cache()
                svd_layers_processed += 1

        # --- step 2: apply head weights (handle both RouterHead and nn.Linear) ---
        device = next(linear_layer.parameters()).device
        dtype = next(linear_layer.parameters()).dtype

        # Check if this is the new RouterHead or old nn.Linear
        if hasattr(linear_layer, 'get_parameter_count'):
            # New RouterHead
            head_param_count = linear_layer.get_parameter_count()
            if flat_params.shape[0] - offset >= head_param_count:
                head_chunk = flat_params[offset: offset + head_param_count]
                head_tensor = torch.from_numpy(head_chunk).to(device, dtype)
                linear_layer.set_weight_tensor(head_tensor)
                del head_tensor
                torch.cuda.empty_cache()
        else:
            # Old nn.Linear - backward compatibility
            w_size = linear_layer.weight.numel()
            if flat_params.shape[0] - offset >= w_size:
                w_chunk = flat_params[offset: offset + w_size]
                w_tensor = torch.from_numpy(w_chunk).to(device, dtype)
                linear_layer.weight.data.copy_(w_tensor.view_as(linear_layer.weight))
                del w_tensor
                torch.cuda.empty_cache()