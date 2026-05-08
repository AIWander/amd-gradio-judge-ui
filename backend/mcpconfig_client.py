"""Client for mcpconfig /chat/completions endpoint (OpenAI-compatible SSE)."""

import json
import os
import time
from typing import AsyncGenerator

import httpx

from .coordinator import classify_intent
from .mock import is_mock_mode, mock_stream_chat, mock_stream_coordinated


def _get_url() -> str:
    return os.environ.get("MCPCONFIG_URL", "").strip()


async def stream_chat(prompt: str, history: list[dict] | None = None) -> AsyncGenerator[dict, None]:
    """
    Stream chat completions. Each yielded dict has a 'delta' key matching
    OpenAI SSE format: {content?, tool_calls?, role?}.
    In mock mode, replays scripted fixtures.
    """
    if is_mock_mode():
        async for event in mock_stream_chat(prompt):
            yield event
        return

    base_url = _get_url()
    messages = []
    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "gpt-oss-120b",
        "messages": messages,
        "stream": True,
        "max_tokens": 8192,
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
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
                        delta = choice.get("delta", {})
                        finish = choice.get("finish_reason")
                        event = {"delta": delta}
                        if finish:
                            event["finish_reason"] = finish
                        yield event
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

    except httpx.ConnectError:
        yield {"delta": {"content": "**Backend unreachable** — cannot connect to mcpconfig."}, "finish_reason": "error"}
    except httpx.ReadTimeout:
        yield {"delta": {"content": "**Timeout** — request exceeded 5 minutes."}, "finish_reason": "error"}
    except Exception as e:
        yield {"delta": {"content": f"**Error:** {type(e).__name__}: {e}"}, "finish_reason": "error"}


async def stream_chat_coordinated(
    prompt: str, history: list[dict] | None = None
) -> AsyncGenerator[dict, None]:
    """Coordinator mode: classify intent with 20B, then stream from chosen model.

    Yields dicts with "type" key:
      {"type":"meta", "route":"20b|120b", "reason":"PLAN|EXEC", "classifier_ms": int}
      {"type":"delta", "content":"...", "model":"..."}
      {"type":"done", "total_ms": int, "total_tokens": int, "model":"...", "route":"..."}
    """
    if is_mock_mode():
        async for event in mock_stream_coordinated(prompt):
            yield event
        return

    # Step 1: Classify
    route, reason, classifier_ms = await classify_intent(prompt, history)
    yield {"type": "meta", "route": route, "reason": reason, "classifier_ms": classifier_ms}

    # Step 2: Stream from chosen model
    model = f"gpt-oss-{route}"
    base_url = _get_url()
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
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status_code != 200:
                    error_text = ""
                    async for chunk in response.aiter_text():
                        error_text += chunk
                    yield {"type": "delta", "content": f"**Error** (HTTP {response.status_code}): {error_text[:300]}", "model": model}
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
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            total_tokens += len(content.split())
                            yield {"type": "delta", "content": content, "model": model}
                        usage = chunk.get("usage")
                        if usage:
                            total_tokens = usage.get("completion_tokens", total_tokens)
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

    except httpx.ConnectError:
        yield {"type": "delta", "content": "**Backend unreachable** — cannot connect.", "model": model}
    except httpx.ReadTimeout:
        yield {"type": "delta", "content": "**Timeout** — request exceeded 5 minutes.", "model": model}
    except Exception as e:
        yield {"type": "delta", "content": f"**Error:** {type(e).__name__}: {e}", "model": model}

    total_ms = int((time.monotonic() - start) * 1000)
    yield {"type": "done", "total_ms": total_ms, "total_tokens": total_tokens, "model": model, "route": route}
