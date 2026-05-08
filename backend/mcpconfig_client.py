"""Client for chat streaming.

In LIVE mode, tool-using turns go through backend.runner which drives MCP
servers directly via stdio and calls vLLM's /v1/responses (Harmony-native)
endpoint. Non-tool turns (coordinator classifier) still use /v1/chat/completions.

  • port 8001 → openai/gpt-oss-120b   (deep-reasoning route)
  • port 8002 → openai/gpt-oss-20b    (fast/exec route)
"""

import json
import os
import time
from typing import AsyncGenerator

import httpx

from .coordinator import classify_intent
from .mock import is_mock_mode, mock_stream_chat, mock_stream_coordinated
from .runner import run as runner_run


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
    """Stream chat from the 120B model with MCP tool support.

    In LIVE mode, routes through runner.run() which calls /v1/responses and
    drives MCP servers directly. Translates canonical events back to the
    delta-shaped dicts that tabs/free_play.py expects.
    In mock mode, replays scripted fixtures.
    """
    if is_mock_mode():
        async for event in mock_stream_chat(prompt):
            yield event
        return

    try:
        async for event in runner_run(
            model="gpt-oss-120b",
            user_prompt=prompt,
            history=history,
            mcp_servers=["workflow", "hands"],
            tool_filter=["api_list", "api_call", "credential_list"],
        ):
            kind = event.get("kind", "")

            if kind == "llm_response":
                content = event.get("content") or event.get("reasoning") or ""
                if content:
                    yield {"delta": {"content": content}}

            elif kind == "tool_call":
                yield {
                    "delta": {
                        "tool_calls": [{
                            "function": {
                                "name": event.get("name", ""),
                                "arguments": event.get("arguments", ""),
                            }
                        }]
                    }
                }

            elif kind == "tool_result":
                yield {
                    "delta": {
                        "tool_result": {
                            "content": event.get("content", ""),
                        }
                    }
                }

            elif kind == "final_answer":
                content = event.get("content", "")
                if content:
                    yield {"delta": {"content": content}}
                yield {"delta": {}, "finish_reason": "stop"}

            elif kind == "run_end" and not event.get("ok"):
                yield {
                    "delta": {"content": f"**Error:** {event.get('error', 'unknown')}"},
                    "finish_reason": "error",
                }

    except Exception as e:
        yield {"delta": {"content": f"**Error:** {type(e).__name__}: {e}"}, "finish_reason": "error"}


# ── Coordinated streaming (Coordination = Coordinator (β)) ────────────


async def stream_chat_coordinated(
    prompt: str, history: list[dict] | None = None
) -> AsyncGenerator[dict, None]:
    """Coordinator mode: classify intent with 20B, then run agent loop on chosen model.

    Yields dicts with "type" key:
      {"type":"meta",  "route":"20b|120b", "reason":"PLAN|EXEC", "classifier_ms": int}
      {"type":"delta", "content":"...", "model":"..."}
      {"type":"tool_call", "name":"...", "arguments":"..."}
      {"type":"tool_result", "content":"..."}
      {"type":"done",  "total_ms": int, "total_tokens": int, "model":"...", "route":"..."}
    """
    if is_mock_mode():
        async for event in mock_stream_coordinated(prompt):
            yield event
        return

    # Step 1: Classify (hits vLLM 20B directly via classify_intent)
    route, reason, classifier_ms = await classify_intent(prompt, history)
    yield {"type": "meta", "route": route, "reason": reason, "classifier_ms": classifier_ms}

    # Step 2: Run agent loop on the chosen model with MCP tools.
    model_short = f"gpt-oss-{route}"
    model_id = _model_id(route)
    start = time.monotonic()
    total_tokens = 0

    try:
        async for event in runner_run(
            model=model_short,
            user_prompt=prompt,
            history=history,
            mcp_servers=["workflow", "hands"],
            tool_filter=["api_list", "api_call", "credential_list"],
        ):
            kind = event.get("kind", "")

            if kind == "llm_response":
                content = event.get("content") or event.get("reasoning") or ""
                if content:
                    yield {"type": "delta", "content": content, "model": model_id}

            elif kind == "tool_call":
                yield {
                    "type": "tool_call",
                    "name": event.get("name", ""),
                    "arguments": event.get("arguments", ""),
                }

            elif kind == "tool_result":
                yield {
                    "type": "tool_result",
                    "content": event.get("content", ""),
                }

            elif kind == "final_answer":
                content = event.get("content", "")
                if content:
                    yield {"type": "delta", "content": content, "model": model_id}

            elif kind == "run_end":
                total_tokens = event.get("total_tokens", 0)

    except Exception as e:
        yield {"type": "delta", "content": f"**Error:** {type(e).__name__}: {e}", "model": model_id}

    total_ms = int((time.monotonic() - start) * 1000)
    yield {"type": "done", "total_ms": total_ms, "total_tokens": total_tokens, "model": model_id, "route": route}
