import asyncio
import re
import os, json, random
import tempfile
import time
from typing import Optional, Any
from typing import Dict, List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_result, retry_if_exception_type
from fugu.llm_clients import (
    query_oai,
    query_anthropic,
    query_gemini,
    query_deepseek,
    query_locally_hosted_model,
)
from fugu.cost import track_cost, get_cost_summary, Calculator

from collections import Counter
import numpy as np
from scipy.stats import entropy
import threading
from dataclasses import dataclass

# Import the debug logger
from fugu.debug_worker_lifecycle import get_worker_logger

# Platform-specific imports for file locking
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    # fcntl is not available on Windows
    HAS_FCNTL = False
    import portalocker

_SPLIT_DIR: str = None

# Global registry for agent configurations (used in worker processes)
_WORKER_AGENT_CONFIGS = {}

@dataclass
class AgentResponse:
    """Wrapper for agent responses that tracks success/failure status."""
    text: str
    status: str  # "success" or "infrastructure_failure"

    def __bool__(self):
        """Allow truthiness checks - True if successful response."""
        return self.status == "success" and bool(self.text)


class InfrastructureFailure(Exception):
    """Exception raised when infrastructure fails during episode evaluation."""

    def __init__(self, agent_name: str, episode_info: str = ""):
        self.agent_name = agent_name
        self.episode_info = episode_info
        super().__init__(f"Infrastructure failure for agent {agent_name}: {episode_info}")

def configure_split_dir(base_dir: str):
    """
    Call this once (e.g. in RouterInfrastructure.__init__) with your chosen log_dir.
    All splits will be stored under <base_dir>/data_splits/.
    """
    global _SPLIT_DIR
    _SPLIT_DIR = os.path.join(base_dir, "data_splits")


def set_worker_agent_configs(configs):
    """
    Set agent configurations for the current worker process.

    Args:
        configs: Dictionary mapping agent names to their configurations
                Each config should have 'model_name' and 'payload' keys
    """
    global _WORKER_AGENT_CONFIGS
    _WORKER_AGENT_CONFIGS = configs.copy() if configs else {}


def get_worker_agent_configs():
    """
    Get current agent configurations for debugging.

    Returns:
        Dictionary of current agent configurations
    """
    global _WORKER_AGENT_CONFIGS
    return _WORKER_AGENT_CONFIGS.copy()


def _resolve_agent_model_and_payload(agent_name: str) -> Tuple[str, dict]:
    """
    Resolve agent name to actual model name and payload.

    Args:
        agent_name: Agent identifier (may be a display name or actual model name)

    Returns:
        Tuple of (actual_model_name, payload_dict)
    """
    global _WORKER_AGENT_CONFIGS

    if agent_name in _WORKER_AGENT_CONFIGS:
        config = _WORKER_AGENT_CONFIGS[agent_name]
        actual_model = config.get("model_name", agent_name)
        payload = config.get("payload", {})
        return actual_model, payload
    else:
        # If not in registry, assume it's already a model name
        return agent_name, {}


def _resolve_agent_complete_info(agent_name: str) -> Tuple[str, dict, str, int]:
    """
    Resolve agent name to complete information including server and port.

    Args:
        agent_name: Agent identifier (may be a display name or actual model name)
    Returns:
        Tuple of (actual_model_name, payload_dict, server, port)
    """
    global _WORKER_AGENT_CONFIGS

    if agent_name in _WORKER_AGENT_CONFIGS:
        config = _WORKER_AGENT_CONFIGS[agent_name]
        actual_model = config.get("model_name", agent_name)
        payload = config.get("payload", {})
        port = config.get("port")

        # For open-source models, we need a server address
        # Use a default server address for local models
        server = "10.128.31.193" if port else None

        return actual_model, payload, server, port
    else:
        return agent_name, {}, None, None


def debug_agent_resolution(agent_name: str) -> dict:
    """
    Debug function to show how an agent name gets resolved.

    Args:
        agent_name: Agent name to resolve

    Returns:
        Dictionary with resolution details
    """
    global _WORKER_AGENT_CONFIGS

    actual_model, payload = _resolve_agent_model_and_payload(agent_name)

    return {
        "input_agent_name": agent_name,
        "resolved_model_name": actual_model,
        "payload": payload,
        "was_resolved": agent_name in _WORKER_AGENT_CONFIGS,
        "available_configs": list(_WORKER_AGENT_CONFIGS.keys()) if _WORKER_AGENT_CONFIGS else []
    }


def validate_agent_configs(agent_configs: dict) -> List[str]:
    """
    Validate agent configuration dictionary structure.

    Args:
        agent_configs: Dictionary mapping agent names to configurations

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not isinstance(agent_configs, dict):
        errors.append("agent_configs must be a dictionary")
        return errors

    for agent_name, config in agent_configs.items():
        if not isinstance(config, dict):
            errors.append(f"Config for agent '{agent_name}' must be a dictionary")
            continue

        if "model_name" not in config:
            errors.append(f"Config for agent '{agent_name}' missing 'model_name' key")

        if "payload" not in config:
            errors.append(f"Config for agent '{agent_name}' missing 'payload' key")
        elif not isinstance(config["payload"], dict):
            errors.append(f"Payload for agent '{agent_name}' must be a dictionary")

    return errors


def log_agent_resolution_summary(agent_names: List[str], debug: bool = True) -> dict:
    """
    Log a summary of how agent names will be resolved.

    Args:
        agent_names: List of agent names used in training
        debug: Whether to print debug info

    Returns:
        Dictionary with resolution summary
    """
    global _WORKER_AGENT_CONFIGS

    summary = {
        "total_agents": len(agent_names),
        "configured_agents": len(_WORKER_AGENT_CONFIGS),
        "resolutions": {},
        "unique_models": set(),
    }

    for agent_name in agent_names:
        actual_model, payload = _resolve_agent_model_and_payload(agent_name)
        summary["resolutions"][agent_name] = {
            "actual_model": actual_model,
            "has_payload": bool(payload),
            "payload_keys": list(payload.keys()) if payload else []
        }
        summary["unique_models"].add(actual_model)

    summary["unique_models"] = list(summary["unique_models"])
    summary["num_unique_models"] = len(summary["unique_models"])

    if debug:
        print("=== AGENT RESOLUTION SUMMARY ===")
        print(f"Total agents: {summary['total_agents']}")
        print(f"Configured agents: {summary['configured_agents']}")
        print(f"Unique models: {summary['num_unique_models']}")
        print("\nResolution details:")
        for agent, details in summary["resolutions"].items():
            print(f"  {agent} -> {details['actual_model']}")
            if details["has_payload"]:
                print(f"    Payload: {details['payload_keys']}")
        print("================================")

    return summary


def _is_oai_model(model: str) -> bool:
    return "gpt" in model.lower()


def _is_anthropic_model(model: str) -> bool:
    return "claude" in model.lower()


def _is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower()


def _is_deepseek_model(model: str) -> bool:
    return "deepseek" in model.lower()


def _is_llama_model(model: str) -> bool:
    return "llama" in model.lower()


def _is_gemma3_model(model: str) -> bool:
    return "gemma3" in model.lower()


def _is_qwen_model(model: str) -> bool:
    return "qwen" in model.lower()


def _should_retry_response(result):
    """Check if we should retry based on the response content."""
    if not result:
        return True

    # Check for whitespace-only responses
    if result.strip() == "":
        return True

    # Check for common error indicators from Gemini
    if result.startswith("CONTENT_REMOVED_DUE_TO_") or result == "DEBUG: No text found in response from gemini":
        return True

    return False


def _print_retry_message(retry_state):
    """Enhanced retry message handler for both exceptions and empty responses."""
    if retry_state.outcome and retry_state.outcome.failed:
        # Exception-based retry
        # print(
        #     f"Retrying attempt {retry_state.attempt_number}/15 "
        #     + f"after error: {retry_state.outcome.exception()}"
        # )

        if retry_state.attempt_number==15:
            print(f"Warning: all 15 attempts to query the LLM failed. Last error: {retry_state.outcome.exception()}")
    else:
        # Result-based retry (empty response)
        # print(
        #     f"Retrying attempt {retry_state.attempt_number}/15 "
        #     + f"after empty/invalid response from LLM"
        # )

        if retry_state.attempt_number==15:
            print(f"Warning: all 15 attempts to query the LLM result. Last error: empty response from LLM")


def log_api_call_wrapper(original_func):
    """Decorator to log API calls for debugging."""

    def wrapper(*args, **kwargs):
        logger = get_worker_logger()
        start_time = time.time()

        # Extract model name if available
        model_name = args[0] if args else "unknown"

        logger.log_event("api_call_start", {
            "model": model_name,
            "function": original_func.__name__,
            "worker_pid": os.getpid(),
            "thread_id": threading.get_ident() if 'threading' in globals() else "unknown"
        })

        try:
            result = original_func(*args, **kwargs)
            duration = time.time() - start_time
            logger.log_event("api_call_success", {
                "model": model_name,
                "function": original_func.__name__,
                "duration": duration,
                "response_length": len(result) if isinstance(result, str) else 0,
                "worker_pid": os.getpid()
            })
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.log_event("api_call_error", {
                "model": model_name,
                "function": original_func.__name__,
                "duration": duration,
                "error": str(e),
                "error_type": type(e).__name__,
                "worker_pid": os.getpid()
            })
            raise

    return wrapper


@retry(
    stop=stop_after_attempt(15),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=(retry_if_exception_type(Exception) | retry_if_result(_should_retry_response)),
    reraise=True,
    before_sleep=_print_retry_message,
)
def _query_llm(
        model: str,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        server: Optional[str],
        port: Optional[int],
        debug: bool = False,
        together: bool = True,
        **kwargs
) -> str:
    if server is None or port is None:
        if _is_oai_model(model):
            response = query_oai(model, messages, max_tokens, temperature, **kwargs)
        elif _is_anthropic_model(model):
            response = query_anthropic(model, messages, max_tokens, temperature, **kwargs)
        elif _is_gemini_model(model):
            response = query_gemini(model, messages, max_tokens, temperature, **kwargs)
        elif _is_deepseek_model(model):
            response = query_deepseek(model, messages, max_tokens, temperature, use_together=together, **kwargs)
        else:
            print(f"Unsupported model: {model}")
            response = ""
    else:
        response = query_locally_hosted_model(model, messages, max_tokens, temperature, server, port, **kwargs)

    # Only track cost if both server and port are not provided (i.e. using cloud API)
    if response and server is None and port is None:
        cost_info = track_cost(model, messages, response)
        if debug:
            print(f"Query cost: ${cost_info['cost']:.6f} | Running total: ${cost_info['total_cost']:.6f}")

    return response


# Apply the debugging wrapper to the main query function
_query_llm = log_api_call_wrapper(_query_llm)

def query_llm(
        model: str,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        server: str = None,
        port: int = None,
        debug: bool = False,
        together: bool = True,
        **kwargs
) -> AgentResponse:
    """
    Query LLM with agent configuration resolution support and infrastructure failure tracking.

    Returns:
        AgentResponse object with text and status ("success" or "infrastructure_failure")
    """
    # Save current alarm state
    import signal
    old_alarm = signal.alarm(0)  # Disable and get current alarm

    try:
        # Use complete resolution if server/port not provided
        if server is None and port is None:
            actual_model, agent_payload, resolved_server, resolved_port = _resolve_agent_complete_info(model)
            server = resolved_server
            port = resolved_port
        else:
            # Use original resolution for backward compatibility
            actual_model, agent_payload = _resolve_agent_model_and_payload(model)

        # Merge agent payload with kwargs (kwargs take precedence)
        merged_kwargs = {**agent_payload, **kwargs}

        # Debug output for agent resolution
        if debug and actual_model != model:
            print(f"=== DEBUG: Agent Resolution ===")
            print(f"Agent name: {model}")
            print(f"Actual model: {actual_model}")
            if agent_payload:
                print(f"Agent payload: {agent_payload}")
            if kwargs:
                print(f"Additional kwargs: {kwargs}")
            if merged_kwargs != kwargs:
                print(f"Merged parameters: {merged_kwargs}")
            print("==============================")

        # Enhanced debug output for LLM API calls
        if debug:
            print("\n=== DEBUG: MESSAGES SENT TO LLM API ===")
            print(f"Original model parameter: {model}")
            print(f"Resolved model: {actual_model}")
            print(f"Server: {server}, Port: {port}")
            for i, msg in enumerate(messages):
                print(f"Message {i + 1} - Role: {msg['role']}")
                # Truncate very long content for readability
                content = msg['content']
                if len(content) > 500:
                    print(f"Content: {content}")
                else:
                    print(f"Content: {content}")
            if merged_kwargs:
                print(f"Model parameters: {merged_kwargs}")
            print("=======================================\n")

        try:
            # No timeouts during LLM API calls
            import time
            start_time = time.time()
            result = _query_llm(
                actual_model, messages, max_tokens, temperature, server, port, debug, together, **merged_kwargs
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            if debug:
                print(f"LLM API {actual_model} took {elapsed_time:.3f} seconds")
            return AgentResponse(text=result, status="success")

        except Exception as e:
            # Check if this is a RetryError from tenacity
            if "RetryError" in str(type(e)):
                # Extract attempt number from RetryError
                try:
                    attempt_number = e.last_attempt.attempt_number if hasattr(e, 'last_attempt') else 5
                    print(
                        f"INFRASTRUCTURE FAILURE: Agent '{actual_model}' failed after {attempt_number} retry attempts")
                except:
                    print(f"INFRASTRUCTURE FAILURE: Agent '{actual_model}' failed after multiple retry attempts")

                print(f"[INFRA_FAILURE_DEBUG] Agent {actual_model} failed after retries - returning empty response")
                # Return infrastructure failure instead of empty string
                return AgentResponse(text="", status="infrastructure_failure")
            else:
                # Regular exception handling - still treat as infrastructure failure
                error_msg = f"Failed to query LLM (model: {actual_model}"
                if actual_model != model:
                    error_msg += f", original: {model}"
                error_msg += f"): {e}"
                print(f"INFRASTRUCTURE FAILURE: {error_msg}")
                print(f"[INFRA_FAILURE_DEBUG] Agent {actual_model} other error: {e}")
                return AgentResponse(text="", status="infrastructure_failure")

    finally:
        # Restore previous alarm state
        if old_alarm > 0:
            signal.alarm(old_alarm)


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the answer the grader should run. Prefer <answer>…</answer> tags;
    if absent, FALL BACK to the last ```python fenced block, then to raw text.

    GRADING-FAIRNESS FIX (2026-06-27): models comply with the <answer> tag
    unevenly. DeepSeek-V3 (and others) return prose + a ```python fence instead
    of tags. The original tag-only version returned None for them → instant 0.0
    score, EVEN WHEN the model produced a correct full solution. That silently
    biased the whole experiment (training + eval) against any non-<answer> model
    — e.g. the cost-min router collapsed onto a tag-emitting model partly as a
    grading artifact. Mirrors the fix in scripts/differentiation_probe.py so the
    core grading path matches it. (lessons: probe-vs-core extraction parity.)
    """
    if not text:
        return None

    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback 1: the last ```python (or bare ```) fenced block — the substantive
    # code when a model ignores the tag instruction.
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()

    # Fallback 2: no tags, no fences — hand back the raw text; extract_code()
    # downstream strips stray markers and the executor decides. Better than a
    # guaranteed 0.0 for a model that may have emitted bare code.
    return text.strip() or None


def batch_completion(
        model_name: str,
        batch_messages: List[List[Dict]],
        max_tokens: int,
        temperature: float,
        max_concurrency: int,
        server: str = None,
        port: int = None,
        debug: bool = False,
        together: bool = True,
        **kwargs
) -> List[str]:
    """
    Process multiple LLM queries in parallel with concurrency control and agent resolution.

    Args:
        model_name: Model name or agent display name
        batch_messages: List of message lists to process
        max_tokens: Maximum tokens per completion
        temperature: Temperature for generation
        max_concurrency: Maximum concurrent requests
        server: Server address (for local models)
        port: Port number (for local models)
        debug: Enable debug output
        together: Use Together API for DeepSeek models
        **kwargs: Additional model parameters

    Returns:
        List of response strings
    """
    # Resolve agent configuration for the model
    actual_model, agent_payload = _resolve_agent_model_and_payload(model_name)
    merged_kwargs = {**agent_payload, **kwargs}

    if debug:
        print(f"=== BATCH COMPLETION DEBUG ===")
        print(f"Input model: {model_name}")
        print(f"Resolved model: {actual_model}")
        print(f"Batch size: {len(batch_messages)}")
        print(f"Max concurrency: {max_concurrency}")
        if agent_payload:
            print(f"Agent payload: {agent_payload}")
        if merged_kwargs != kwargs:
            print(f"Merged parameters: {merged_kwargs}")
        print("==============================")

    async def _async_batch_completion():
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_with_semaphore(prompt):
            async with semaphore:
                # query_llm returns AgentResponse
                return await asyncio.to_thread(
                    query_llm,
                    actual_model,  # Use resolved model name
                    prompt,
                    max_tokens,
                    temperature,
                    server,
                    port,
                    debug,
                    together,
                    **merged_kwargs  # Use merged parameters
                )

        tasks = [process_with_semaphore(prompt) for prompt in batch_messages]
        return await asyncio.gather(*tasks)

    # Gather AgentResponse objects
    responses = asyncio.run(_async_batch_completion())

    # Convert AgentResponse -> plain strings for downstream code & JSON logging
    texts: List[str] = []
    infra_failures = 0
    for r in responses:
        if isinstance(r, AgentResponse):
            if r.status != "success":
                infra_failures += 1
            texts.append(r.text or "")
        elif isinstance(r, str):
            texts.append(r)
        else:
            # Defensive fallback: stringify anything unexpected
            texts.append(str(r) if r is not None else "")

    if debug and infra_failures:
        print(f"[batch_completion] {infra_failures}/{len(responses)} requests reported infrastructure failures; empty strings returned for those.")

    # Print cost summary after batch completion (use resolved model name)
    cost_summary = get_cost_summary()
    if actual_model in [model_summary["model"] for model_summary in cost_summary["models"]]:
        for model_summary in cost_summary["models"]:
            if model_summary["model"] == actual_model:
                print(f"\nCumulative usage for {actual_model}" +
                      (f" (via {model_name})" if actual_model != model_name else "") + ":")
                print(
                    f"Total tokens: {model_summary['total_tokens']} "
                    f"(Input: {model_summary['total_input_tokens']}, "
                    f"Output: {model_summary['total_output_tokens']})"
                )
                print(f"Total cost: ${model_summary['total_cost']:.6f}")
                break

    return texts

def calculate_agent_stats(
        episode_agents: list,
        llm_names: list,
        total_agent_usage: dict = None,
        entropy_history: list = None
) -> dict:
    """
    Calculate detailed agent usage statistics for logging.

    Args:
        episode_agents: List of agent IDs used in the current iteration
        llm_names: List of agent model names
        total_agent_usage: Dict tracking cumulative agent usage (will be updated)
        entropy_history: List of entropy values from logits (optional)

    Returns:
        Dictionary with agent distribution statistics
    """

    # Current iteration agent distribution
    dist = Counter(episode_agents)
    agent_dist = {name: int(dist.get(j, 0)) for j, name in enumerate(llm_names)}

    # Update total agent usage if provided
    if total_agent_usage is None:
        total_agent_usage = {name: 0 for name in llm_names}

    for j, name in enumerate(llm_names):
        total_agent_usage[name] += int(dist.get(j, 0))

    # Calculate agent selection entropy (distribution of agents chosen)
    counts = np.array([dist.get(j, 0) for j in range(len(llm_names))])
    if counts.sum() > 0:
        probs = counts / counts.sum()
        agent_selection_entropy = float(entropy(probs, base=2))
    else:
        agent_selection_entropy = 0.0

    # Calculate mean entropy from model outputs if available
    mean_entropy = float(np.mean(entropy_history)) if entropy_history else None

    stats = {
        "agent_distribution": agent_dist,
        "total_agent_usage": dict(total_agent_usage),  # Make a copy
        "agent_selection_entropy": agent_selection_entropy,
    }

    if mean_entropy is not None:
        stats["mean_entropy"] = mean_entropy

    return stats, total_agent_usage


# ----- Token Statistics Tracking Functions -----

class TokenStatisticsTracker:
    """Global tracker for token usage statistics across episodes."""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = TokenStatisticsTracker()
        return cls._instance

    def __init__(self):
        """Initialize a new token statistics tracker."""
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.router_tokens = []
        self.agent_input_tokens = []
        self.agent_output_tokens = []
        self.total_tokens = []
        self.num_turns = []
        self.num_episodes = 0

    def add_episode(self, token_stats: Dict[str, int]):
        """Add statistics from a single episode."""
        if not token_stats:
            return

        self.router_tokens.append(token_stats.get("router_tokens", 0))
        self.agent_input_tokens.append(token_stats.get("agent_input_tokens", 0))
        self.agent_output_tokens.append(token_stats.get("agent_output_tokens", 0))
        self.total_tokens.append(token_stats.get("total_tokens", 0))
        self.num_turns.append(token_stats.get("num_turns", 0))
        self.num_episodes += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        if self.num_episodes == 0:
            return {
                "avg_router_tokens": 0,
                "avg_agent_input_tokens": 0,
                "avg_agent_output_tokens": 0,
                "avg_total_tokens": 0,
                "avg_turns": 0,
                "total_router_tokens": 0,
                "total_agent_input_tokens": 0,
                "total_agent_output_tokens": 0,
                "total_tokens": 0,
                "episodes_tracked": 0
            }

        return {
            "avg_router_tokens": float(np.mean(self.router_tokens)),
            "avg_agent_input_tokens": float(np.mean(self.agent_input_tokens)),
            "avg_agent_output_tokens": float(np.mean(self.agent_output_tokens)),
            "avg_total_tokens": float(np.mean(self.total_tokens)),
            "avg_turns": float(np.mean(self.num_turns)),
            "total_router_tokens": int(np.sum(self.router_tokens)),
            "total_agent_input_tokens": int(np.sum(self.agent_input_tokens)),
            "total_agent_output_tokens": int(np.sum(self.agent_output_tokens)),
            "total_tokens": int(np.sum(self.total_tokens)),
            "episodes_tracked": self.num_episodes
        }


def count_router_tokens(model_name: str, messages: List[Dict]) -> int:
    """Count tokens in router messages using Calculator."""
    calculator = Calculator(model_name, messages)
    return calculator.calculate_input_token_length()


def count_agent_tokens(model_name: str, messages: List[Dict], response: str = None) -> Tuple[int, int]:
    """Count input and output tokens for an agent interaction."""
    # Count input tokens
    input_calculator = Calculator(model_name, messages)
    input_tokens = input_calculator.calculate_input_token_length()

    # Count output tokens if response is provided
    output_tokens = 0
    if response:
        output_calculator = Calculator(model_name, output_sequence_string=response)
        output_tokens = output_calculator.calculate_output_token_length_GPT()

    return input_tokens, output_tokens


def track_episode_tokens(
        context: Any,
        router_messages: List[Dict],
        agent_messages: List[Dict],
        agent_response: str,
        agent_model_name: str,
        debug_log_file: Optional[str] = None
) -> Dict[str, int]:
    """
    Track token usage for a single agent interaction.

    Args:
        context: The worker context with model information
        router_messages: Messages sent to the router
        agent_messages: Messages sent to the agent
        agent_response: Response from the agent
        agent_model_name: Name of the agent model
        debug_log_file: Optional path to debug log file

    Returns:
        Dictionary with token counts
    """
    # Track router input tokens
    router_tokens = count_router_tokens(context.router_model_name, router_messages)

    # Track agent input and output tokens
    agent_input_tokens, agent_output_tokens = count_agent_tokens(
        agent_model_name, agent_messages, agent_response
    )

    # Log token usage if debug enabled
    if debug_log_file is not None:
        with open(debug_log_file, "a") as log_f:
            log_f.write(f"Router input tokens: {router_tokens}\n")
            log_f.write(f"Agent input tokens: {agent_input_tokens}\n")
            log_f.write(f"Agent output tokens: {agent_output_tokens}\n")
            log_f.write(f"Total tokens: {router_tokens + agent_input_tokens + agent_output_tokens}\n")

    return {
        "router_tokens": router_tokens,
        "agent_input_tokens": agent_input_tokens,
        "agent_output_tokens": agent_output_tokens,
        "total_tokens": router_tokens + agent_input_tokens + agent_output_tokens
    }


def extract_token_stats_from_results(results: List[Tuple]) -> List[Dict]:
    """
    Extract token statistics from evaluation results.

    Args:
        results: List of evaluation result tuples

    Returns:
        List of token statistics dictionaries
    """
    token_stats_list = []

    for result in results:
        # Check if result has token statistics (6th element)
        if len(result) >= 6 and isinstance(result[5], dict):
            # Skip failed evaluations
            if result[0] == -1.0:
                continue

            token_stats_list.append(result[5])

    return token_stats_list


def aggregate_token_statistics(results: List[Tuple]) -> Dict[str, Any]:
    """
    Aggregate token statistics from multiple evaluation results.

    Args:
        results: List of evaluation result tuples

    Returns:
        Dictionary with aggregated token statistics
    """
    # Extract token stats from results
    token_stats_list = extract_token_stats_from_results(results)

    # Create a local tracker for these specific results
    tracker = TokenStatisticsTracker()

    # Reset the tracker to ensure we're only counting these results
    tracker.reset()

    # Add all episodes
    for stats in token_stats_list:
        tracker.add_episode(stats)

    # Return aggregated statistics
    return tracker.get_stats()


def _split_path(task_name: str, seed: int, v_ratio: float, t_ratio: float) -> str:
    # pick our configured split folder, or fall back to ./.data_splits
    split_root = _SPLIT_DIR or os.path.join(os.getcwd(), ".data_splits")
    os.makedirs(split_root, exist_ok=True)
    filename = f"{task_name}_{seed}_v{v_ratio}_t{t_ratio}.json"
    return os.path.join(split_root, filename)


def get_or_create_indices(
        task_name: str,
        dataset_len: int,
        seed: int,
        valid_ratio: float,
        test_ratio: float
) -> Dict[str, List[int]]:
    """
    Return {"train":…, "valid":…, "test":…} index lists.
    Reads <split_path> if it exists, otherwise makes a new random.Random(seed)
    shuffle and writes it. Thread-safe with file locking and atomic writes.
    """
    path = _split_path(task_name, seed, valid_ratio, test_ratio)
    lock_path = path + ".lock"

    # Try to read existing file first (fast path)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # File exists but is corrupted/incomplete, delete it
            print(f"Warning: Corrupted split file {path}, regenerating... Error: {e}")
            try:
                os.remove(path)
            except:
                pass

    # Need to create the file - use file locking for thread safety
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # If fcntl is not available (Windows), use simpler approach
    if not HAS_FCNTL:
        # Simple retry mechanism without file locking
        max_retries = 10
        for retry in range(max_retries):
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        return json.load(f)
                except:
                    time.sleep(0.1 * (retry + 1))
                    continue

            # Try to create the file
            try:
                # Create the splits
                rnd = random.Random(seed)
                ids = list(range(dataset_len))
                rnd.shuffle(ids)

                n_test = int(dataset_len * test_ratio)
                n_valid = int(dataset_len * valid_ratio)

                split = {
                    "test": ids[:n_test],
                    "valid": ids[n_test: n_test + n_valid],
                    "train": ids[n_test + n_valid:]
                }

                # Write to temporary file first
                temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(path), text=True)
                try:
                    with os.fdopen(temp_fd, 'w') as temp_file:
                        json.dump(split, temp_file)
                    # Atomic rename
                    os.replace(temp_path, path)
                    return split
                except:
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    raise
            except Exception as e:
                if retry == max_retries - 1:
                    raise RuntimeError(f"Failed to create split file after {max_retries} retries: {e}")
                time.sleep(0.1 * (retry + 1))
                continue

    # Unix/Linux with fcntl support
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    # Use file locking to ensure only one process creates the file
    max_retries = 10
    retry_delay = 0.1

    for retry in range(max_retries):
        try:
            # Try to acquire exclusive lock
            with open(lock_path, 'w') as lock_file:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                    # Check again if file was created while waiting
                    if os.path.exists(path):
                        try:
                            with open(path, "r") as f:
                                result = json.load(f)
                                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                                return result
                        except:
                            # Still corrupted, continue to recreate
                            pass

                    # Create the splits
                    rnd = random.Random(seed)
                    ids = list(range(dataset_len))
                    rnd.shuffle(ids)

                    n_test = int(dataset_len * test_ratio)
                    n_valid = int(dataset_len * valid_ratio)

                    split = {
                        "test": ids[:n_test],
                        "valid": ids[n_test: n_test + n_valid],
                        "train": ids[n_test + n_valid:]
                    }

                    # Write to temporary file first (atomic write)
                    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(path), text=True)
                    try:
                        with os.fdopen(temp_fd, 'w') as temp_file:
                            json.dump(split, temp_file)
                        # Atomic rename
                        os.replace(temp_path, path)
                    except:
                        # Clean up temp file on error
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                        raise

                    # Release lock
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

                    # Clean up lock file
                    try:
                        os.remove(lock_path)
                    except:
                        pass

                    return split

                except IOError:
                    # Lock is held by another process, wait and retry
                    time.sleep(retry_delay * (retry + 1))

                    # Try to read the file again
                    if os.path.exists(path):
                        try:
                            with open(path, "r") as f:
                                return json.load(f)
                        except:
                            # Still not ready, continue retrying
                            pass

        except Exception as e:
            if retry == max_retries - 1:
                # Last retry, give up and raise
                raise RuntimeError(f"Failed to create/read split file after {max_retries} retries: {e}")

    # Should not reach here, but fallback to creating splits without lock
    print(f"Warning: Failed to acquire lock, creating splits without synchronization")
    rnd = random.Random(seed)
    ids = list(range(dataset_len))
    rnd.shuffle(ids)

    n_test = int(dataset_len * test_ratio)
    n_valid = int(dataset_len * valid_ratio)

    return {
        "test": ids[:n_test],
        "valid": ids[n_test: n_test + n_valid],
        "train": ids[n_test + n_valid:]
    }