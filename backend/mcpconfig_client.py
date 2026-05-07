"""Client for mcpconfig /chat/completions endpoint (OpenAI-compatible SSE)."""

import json
import os
from typing import AsyncGenerator

import httpx

from .mock import is_mock_mode, mock_stream_chat


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
