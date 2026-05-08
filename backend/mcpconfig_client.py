"""Client for chat streaming.

Despite the filename, this no longer talks to mcpconfig — that server's API
shape (`/run` SSE with custom event kinds) is incompatible with how the chat
UI wants to consume tokens. We bypass it and call vLLM directly:

  • port 8001 → openai/gpt-oss-120b   (deep-reasoning route)
  • port 8002 → openai/gpt-oss-20b    (fast/exec route)

vLLM is OpenAI-compatible, exposes `/v1/chat/completions`, and supports SSE
streaming natively. Restoring tool-calling via mcpconfig `/run` is a follow-up.
"""

import json
import os
import time
from typing import AsyncGenerator

import httpx

from .coordinator import classify_intent
from .mock import is_mock_mode, mock_stream_chat, mock_stream_coordinated


# ── URL resolution ────────────────────────────────────────────────────

_DEFAULT_120B = "http://127.0.0.1:8001"
_DEFAULT_20B = "http://127.0.0.1:8002"


def _vllm_url(route: str) -> str:
    """Return base vLLM URL (without /v1) for the given route ('20b' or '120b').

    Env override order (per route):
      route='120b': VLLM_120B_URL → VLLM_BASE_URL → http://127.0.0.1:8001
      route='20b':  VLLM_20B_URL  → VLLM_BASE_URL_8002 → http://127.0.0.1:8002
    """
    if route == "120b":
        return (
            os.environ.get("VLLM_120B_URL", "").strip()
            or os.environ.get("VLLM_BASE_URL", "").strip()
            or _DEFAULT_120B
        ).rstrip("/")
    return (
        os.environ.get("VLLM_20B_URL", "").strip()
        or os.environ.get("VLLM_BASE_URL_8002", "").strip()
        or _DEFAULT_20B
    ).rstrip("/")


def _model_id(route: str) -> str:
    return "openai/gpt-oss-120b" if route == "120b" else "openai/gpt-oss-20b"


# ── Single-model streaming (Coordination = Off) ───────────────────────


async def stream_chat(
    prompt: str, history: list[dict] | None = None
) -> AsyncGenerator[dict, None]:
    """Stream chat completions from the 120B model directly.

    Yields dicts shaped like OpenAI SSE deltas: {"delta": {...}, "finish_reason"?}
    so the existing tabs/free_play.py rendering code keeps working unchanged.
    In mock mode, replays scripted fixtures.
    """
    if is_mock_mode():
        async for event in mock_stream_chat(prompt):
            yield event
        return

    base_url = _vllm_url("120b")
    model = _model_id("120b")
    messages = []
    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 8192,
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code != 200:
                    error_text = ""
                    async for chunk in response.aiter_text():
                        error_text += chunk
                    yield {
                        "delta": {"content": f"**Error** (HTTP {response.status_code}): {error_text[:300]}"},
                        "finish_reason": "error",
                    }
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data_str)
                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {}) or {}
                        # gpt-oss reasoning channel: surface it as content too
                        # so the chatbot shows something while the model thinks.
                        if not delta.get("content") and delta.get("reasoning_content"):
                            delta = dict(delta)
                            delta["content"] = delta["reasoning_content"]
                        finish = choice.get("finish_reason")
                        event = {"delta": delta}
                        if finish:
                            event["finish_reason"] = finish
                        yield event
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

    except httpx.ConnectError:
        yield {"delta": {"content": "**Backend unreachable** — cannot connect to vLLM 120B."}, "finish_reason": "error"}
    except httpx.ReadTimeout:
        yield {"delta": {"content": "**Timeout** — request exceeded 5 minutes."}, "finish_reason": "error"}
    except Exception as e:
        yield {"delta": {"content": f"**Error:** {type(e).__name__}: {e}"}, "finish_reason": "error"}


# ── Coordinated streaming (Coordination = Coordinator (β)) ────────────


async def stream_chat_coordinated(
    prompt: str, history: list[dict] | None = None
) -> AsyncGenerator[dict, None]:
    """Coordinator mode: classify intent with 20B, then stream from chosen model.

    Yields dicts with "type" key:
      {"type":"meta",  "route":"20b|120b", "reason":"PLAN|EXEC", "classifier_ms": int}
      {"type":"delta", "content":"...", "model":"..."}
      {"type":"done",  "total_ms": int, "total_tokens": int, "model":"...", "route":"..."}
    """
    if is_mock_mode():
        async for event in mock_stream_coordinated(prompt):
            yield event
        return

    # Step 1: Classify (hits vLLM 20B directly via classify_intent)
    route, reason, classifier_ms = await classify_intent(prompt, history)
    yield {"type": "meta", "route": route, "reason": reason, "classifier_ms": classifier_ms}

    # Step 2: Stream from the chosen model — direct to its vLLM port.
    base_url = _vllm_url(route)
    model = _model_id(route)
    messages = []
    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 8192,
    }

    start = time.monotonic()
    total_tokens = 0

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code != 200:
                    error_text = ""
                    async for chunk in response.aiter_text():
                        error_text += chunk
                    yield {
                        "type": "delta",
                        "content": f"**Error** (HTTP {response.status_code}): {error_text[:300]}",
                        "model": model,
                    }
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {}) or {}
                        # Prefer content; fall back to reasoning_content so the
                        # user sees the model thinking even if it doesn't emit
                        # a final answer in time.
                        content = delta.get("content") or delta.get("reasoning_content") or ""
                        if content:
                            total_tokens += len(content.split())
                            yield {"type": "delta", "content": content, "model": model}
                        usage = chunk.get("usage")
                        if usage:
                            total_tokens = usage.get("completion_tokens", total_tokens)
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

    except httpx.ConnectError:
        yield {"type": "delta", "content": f"**Backend unreachable** — cannot connect to vLLM ({route}).", "model": model}
    except httpx.ReadTimeout:
        yield {"type": "delta", "content": "**Timeout** — request exceeded 5 minutes.", "model": model}
    except Exception as e:
        yield {"type": "delta", "content": f"**Error:** {type(e).__name__}: {e}", "model": model}

    total_ms = int((time.monotonic() - start) * 1000)
    yield {"type": "done", "total_ms": total_ms, "total_tokens": total_tokens, "model": model, "route": route}
