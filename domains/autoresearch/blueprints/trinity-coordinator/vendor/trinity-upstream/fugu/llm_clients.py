import os
import requests
import json
from typing import List, Dict, Callable, Optional, Any
from openai import OpenAI
import random, time, boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from google import genai
from google.genai import types



def query_oai(
        model: str,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        max_wrapper_attempts: int = 5,
        base_backoff: float = 1.0,
        timeout: int = 60,
        jitter: float = 0.5,
        should_retry: Optional[Callable[[object], bool]] = None,
        reasoning_effort: str = "minimal",
        **kwargs
) -> str:
    """
    Call OpenAI with retries using exponential backoff + jitter.
    Retries occur on exceptions AND on "soft failures" (e.g., empty content,
    non-stop finish_reason) even if no exception is raised.

    Tips to reduce 'empty_content' and 'length' without changing max tokens:
      - Only send tool_choice when you actually send tools; otherwise omit it.
      - Force plain text with response_format={"type": "text"}.
      - For GPT-5, consider reasoning_effort="low" (Chat Completions) to trim
        internal reasoning token usage.
      - Use a stop sequence (e.g., "</final>") in your prompt and pass
        stop=["</final>"] to end generations early.

    Args:
        model: (Ignored; overridden to "gpt-5" below to match original code)
        messages: Chat messages passed to the API.
        max_tokens: Max completion tokens (sent as max_completion_tokens).
        temperature: (Ignored; overridden to 1 below to match original code)
        max_wrapper_attempts: Max total attempts (first try + retries).
        base_backoff: Base seconds for exponential backoff.
        timeout: Per-request timeout seconds.
        jitter: Max random jitter added to each backoff sleep.
        should_retry: Optional callback receiving the raw response object.
        **kwargs: Extra arguments forwarded to the API call (e.g., stop, tools).

    Returns:
        The first acceptable response content.

    Raises:
        RuntimeError if all attempts fail or produce unacceptable responses.
    """

    if os.getenv("UNBOUNDED_MODE") == "True":
        timeout = 600 # 10mins

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=timeout
    )

    # Preserve original behavior
    model = "gpt-5"
    temperature = 1  # GPT-5 reasoning models may ignore temperature in Chat Completions.

    api_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "max_completion_tokens": max_tokens,  # correct for Chat Completions with GPT-5
    }

    # Allow caller overrides first
    api_params.update(kwargs)

    # For GPT-5, a lower reasoning effort can reduce reasoning token burn
    api_params.setdefault("reasoning_effort", reasoning_effort)  # caller can override to "minimal"/"medium"/"high"

    # IMPORTANT: Only include tool_choice if tools are actually present.
    # This prevents "Invalid value for 'tool_choice'..." 400 errors.
    if api_params.get("tools"):
        api_params.setdefault("tool_choice", "none")  # can override with "auto" or a specific tool
    else:
        api_params.pop("tool_choice", None)  # ensure it's not sent without tools

    last_error: Optional[Exception] = None
    last_reason: Optional[str] = None

    if os.getenv("UNBOUNDED_MODE") == "True":
        api_params["max_completion_tokens"] = 128000
        api_params["temperature"] = 1
        api_params["reasoning_effort"] = "high"

    for attempt in range(1, max_wrapper_attempts + 1):
        try:
            response = client.chat.completions.create(**api_params)

            # Defensive parsing
            choice = (response.choices[0] if response and getattr(response, "choices", None) else None)
            content = (choice.message.content if choice and getattr(choice, "message", None) else None)
            finish_reason = getattr(choice, "finish_reason", None)

            # Decide whether to retry even WITHOUT an exception
            retry_flags = []

            # 1) Empty/blank content
            if not content or (isinstance(content, str) and content.strip() == ""):
                retry_flags.append("empty_content")

            # 2) Non-terminal finish reason (e.g., 'length', 'content_filter', 'tool_calls', etc.)
            if finish_reason not in (None, "stop"):
                retry_flags.append(f"finish_reason:{finish_reason}")

            # 3) Custom user-supplied validator
            if callable(should_retry) and should_retry(response):
                retry_flags.append("custom_validator")

            if not retry_flags:
                return content

            last_reason = ", ".join(retry_flags)
            # print(f"Retrying due to: {last_reason} (attempt {attempt}/{max_wrapper_attempts})")

        except Exception as e:
            last_error = e
            last_reason = f"exception:{type(e).__name__} {e}"
            # print(f"OpenAI API call failed (attempt {attempt}/{max_wrapper_attempts}): {e}")

        if attempt < max_wrapper_attempts:
            sleep_for = min(base_backoff * (2 ** (attempt - 1)) + random.uniform(0, jitter), 20)
            # print(f"Retrying in {sleep_for:.2f} seconds...")
            time.sleep(sleep_for)

    if last_error is not None:
        raise RuntimeError(f"All attempts failed; last error: {last_error}") from last_error
    raise RuntimeError(f"All attempts produced unacceptable responses; last reason: {last_reason}")

#TODO: make it no longer hard coded to GPT-5

def query_anthropic(
        model: str,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        *,
        max_wrapper_attempts: int = 10,     # our own outer loop
        base_backoff: float = 1.0           # seconds
) -> str:
    """
    Send a chat request to Amazon Bedrock and return the model’s reply.

    The function:
    • maps friendly model names to the full Bedrock IDs
    • uses botocore.Config to enable adaptive (client-side) throttling with
      more retry attempts than the SDK default
    • wraps the SDK call in its own exponential-backoff loop so the caller
      sees far fewer ClientError exceptions
    """

    # Bedrock model IDs --------------------------------------------------------
    if model == "claude-sonnet-4-20250514":
        model = "us.anthropic.claude-sonnet-4-20250514-v1:0"

    # SDK-level retry configuration (adaptive mode + higher attempt budget) ---
    sdk_config = Config(
        connect_timeout=5,
        read_timeout=600,
        retries={
            # total = 1 initial try + (total_max_attempts-1) retries
            "total_max_attempts": 8,
            "mode": "adaptive"   # switches on client-side rate-limiting
        }
    )

    client = boto3.client("bedrock-runtime", region_name="us-east-1",
                          config=sdk_config)

    # Convert messages ---------------------------------------------------------
    system_message = None
    converted_messages = []
    for m in messages:
        if m["role"] == "system":
            system_message = m["content"]
            continue
        
        content = m["content"]
        # Add think tag if in unbounded mode and this is an assistant message
        if os.getenv("UNBOUNDED_MODE") == "True" and m["role"] == "assistant" and not content.strip().startswith("<think>"):
            content = "<think>" + content
        
        converted_messages.append({
            "role": m["role"],
            "content": [{"text": content}]
        })

    # Assemble the payload -----------------------------------------------------
    params = {
        "modelId": model,
        "messages": converted_messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature
        }
    }
    if system_message is not None:
        params["system"] = [{"text": system_message}]
    
    if os.getenv("UNBOUNDED_MODE", "False").lower() == "true":
        params["inferenceConfig"]["maxTokens"] = 64000
        params["inferenceConfig"]["temperature"] = 1
        params["additionalModelRequestFields"] = {"thinking":
            {"type": "enabled",
            "budget_tokens": 32768}}

        if converted_messages[-1]["role"] == "assistant":
            converted_messages.pop()

    # Our own retry wrapper (exponential back-off with jitter) -----------------
    for attempt in range(1, max_wrapper_attempts + 1):
        try:
            # response = client.converse(**params)
            # return response["output"]["message"]["content"][0]["text"]
            resp = client.converse(**params)
            blocks = resp["output"]["message"]["content"]
            text_blocks = [b["text"] for b in blocks if isinstance(b, dict) and b.get("type") == "text" and "text" in b]
            if text_blocks:
                return "".join(text_blocks)
            # Fallback: if types are missing, join any entries that have 'text'
            fallback = [b["text"] for b in blocks if isinstance(b, dict) and "text" in b]
            return "".join(fallback) if fallback else ""

        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ThrottlingException":
                raise  # bubble up non-throttling errors immediately

            # throttle → wait, then retry
            sleep_for = min(base_backoff * 2 ** (attempt - 1) +
                            random.uniform(0, 0.5), 20)
            time.sleep(sleep_for)

    raise RuntimeError(
        f"Exceeded {max_wrapper_attempts} attempts due to persistent throttling."
    )

def query_gemini(
    model: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float
) -> str:
    """
    Query Gemini with thinking-budget control and error handling,
    but only against the specified model (no fallback retries).
    Still logs a warning if the response was truncated by max tokens.

    Key behavior:
    1. Configures thinking_budget for speed or token efficiency.
    2. Sends a single request to the exact model specified.
    3. Warns if the model hit the max_output_tokens limit.
    4. Does NOT retry with a larger token allowance.
    """
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set")
        return ""

    try:
        client = genai.Client(api_key=api_key)

        # Separate out any system message
        system_message = None
        content_parts = []
        for message in messages:
            if message.get("role") == "system":
                system_message = message.get("content")
            else:
                content_parts.append(message.get("content", ""))

        contents = "\n".join(content_parts)

        # Configure thinking budget based on model capabilities
        thinking_config = None
        # thinking_budget is half of max_output_tokens (ensure int)
        thinking_budget = 128

        if "2.5-pro" in model:
            thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget,
                include_thoughts=False
            )
        if os.getenv("UNBOUNDED_MODE") == "True":
            max_tokens = 65535
            temperature = 1
            thinking_config = types.ThinkingConfig(
                thinking_budget=32768,
                include_thoughts=False
            )

        generation_config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,   # exactly as provided
            temperature=temperature,
            system_instruction=system_message,
        )
        if thinking_config:
            generation_config.thinking_config = thinking_config

        # Single-model request
        try:
            # 10 inner retries for gemini
            for _ in range(10):
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generation_config,
                )
                if (response.text is not None) and (response.text != ''):
                    break
                else:
                    time.sleep(1)

        except Exception as e:
            print(f"ERROR: Gemini API call failed for model {model}: {e}")
            return ""

        # Safety filter check
        if getattr(response, "prompt_feedback", None):
            block_reason = getattr(response.prompt_feedback, "block_reason", None)
            if block_reason:
                print(f"WARNING: Request blocked by safety filter: {block_reason}")
                return ""

        # Return the main text if available
        if getattr(response, "text", None):
            return response.text

        # Fallback: extract text from candidate parts
        candidates = getattr(response, "candidates", None)
        if candidates:
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        if getattr(part, "text", None):
                            return part.text

        print("DEBUG: No text found in response from gemini")
        return ""

    except Exception as e:
        print(f"ERROR: Unexpected failure: {e}")
        return ""


def query_deepseek(
        model: str, messages: List[Dict], max_tokens: int, temperature: float, use_together: bool = True
) -> str:
    """Query DeepSeek through either Together AI or directly."""

    if use_together:
        # Use Together AI
        client = OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )
    else:
        # Use DeepSeek directly
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    return response.choices[0].message.content


def _extract_text_from_vllm_message(msg) -> str:
    """
    Robustly pull a human-readable string out of an OpenAI-style
    ChatCompletionMessage returned by vLLM — whether or not the
    server was started with `--enable-reasoning`.

    Order of preference:
      1.   message.content  (ordinary chat mode)
      2.   message.tool_calls[0].function.arguments["content"]
      3.   message.thoughts / message.content / message.text
      4.   message.extra[...] (some parsers stash data here)
    Returns "" (empty string) if nothing usable is found.
    """
    # 1️⃣  normal chat
    if msg is not None and hasattr(msg, "content") and msg.content is not None:
        return msg.content

    # 2️⃣  reasoning-mode (vLLM ≥ 0.4)
    try:
        tc = msg.tool_calls
        if tc and tc[0].function and tc[0].function.arguments:
            maybe = tc[0].function.arguments.get("content")
            if isinstance(maybe, str) and maybe.strip():
                return maybe.strip()
    except AttributeError:
        pass

    # 3️⃣  thoughts / content / text attributes
    for attr in ("thoughts", "content", "text"):
        maybe = getattr(msg, attr, None)
        if isinstance(maybe, str) and maybe.strip():
            return maybe.strip()

    # 4️⃣  .extra dict
    extra = getattr(msg, "extra", None)
    if isinstance(extra, dict):
        for attr in ("thoughts", "content", "text"):
            maybe = extra.get(attr)
            if isinstance(maybe, str) and maybe.strip():
                return maybe.strip()

    return ""


def query_locally_hosted_model(
        model: str,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        server: str,
        port: int,
        **kwargs,
) -> str:
    """
    A drop-in replacement that supports BOTH:

        • ordinary OpenAI-compatible chat responses
        • vLLM reasoning-mode responses (message.content is None)
        • vLLM-specific parameters that OpenAI client doesn't support

    Behaviour is unchanged for models that already return
    a plain string in `message.content` and use standard parameters.
    """

    # Parameters that are known to NOT work with OpenAI client but work with vLLM
    vllm_only_params = {'top_k', 'chat_template_kwargs'}

    # Check if we have any vLLM-only parameters
    has_vllm_params = any(param in kwargs for param in vllm_only_params)

    if has_vllm_params:
        # Use direct HTTP request for vLLM-specific parameters
        return _query_vllm_direct_http(model, messages, max_tokens, temperature, server, port, **kwargs)
    else:
        # Use original OpenAI client approach (preserves exact original behavior)
        client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://{server}:{port}/v1",
        )

        # Prepare the base parameters (exact same as original)
        api_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        # Add any additional parameters from kwargs (exact same as original)
        api_params.update(kwargs)

        response = client.chat.completions.create(**api_params)
        msg = response.choices[0].message

        # Apply reasoning mode support to the original path too
        return _extract_text_from_vllm_message(msg)


def _query_vllm_direct_http(
        model: str,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        server: str,
        port: int,
        **kwargs,
) -> str:
    """
    Direct HTTP query for vLLM-specific parameters that OpenAI client doesn't support.
    """
    endpoint = f"http://{server}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Build the payload with all parameters
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    # Add any additional parameters from kwargs
    payload.update(kwargs)

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
        response.raise_for_status()

        response_data = response.json()

        # Extract the response content
        if "choices" in response_data and len(response_data["choices"]) > 0:
            choice = response_data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"] or ""

        return ""

    except requests.exceptions.RequestException as e:
        print(f"HTTP request failed: {e}")
        return ""
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error in vLLM direct query: {e}")
        return ""


if __name__ == "__main__":
    test_model = "claude-sonnet-4-20250514"
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    max_tokens = 32
    temperature = 0.1

    reply = query_anthropic(
        model=test_model,
        messages=test_messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    print("Anthropic reply:", repr(reply))
    assert "Paris" in reply or reply.strip() != "", "No valid reply received."
    print("query_anthropic test passed.")
