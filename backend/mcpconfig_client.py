"""Client for chat streaming.

Routes Free Play chat through one of two backends:

  1. **mcpconfig** (default) — Rust agent loop on port 8003 (/run SSE endpoint).
     End-to-end verified with --tool-call-parser openai on vLLM 0.11.2.
  2. **runner** — Python agent loop (backend/runner.py) driving MCP servers via
     stdio + vLLM /v1/responses.  Kept as fallback.

Select via env var JUDGE_BACKEND: "mcpconfig" | "runner" | "auto" (try mcpconfig, fall back to runner).

Non-tool turns (coordinator classifier) still use /v1/chat/completions directly.

  • port 8001 → openai/gpt-oss-120b   (deep-reasoning route)
  • port 8002 → openai/gpt-oss-20b    (fast/exec route)
  • port 8003 → mcpconfig /run         (Rust agent loop)
"""

import json
import logging
import os
import time
from typing import AsyncGenerator

import httpx

from .coordinator import classify_intent
from .mock import is_mock_mode, mock_stream_chat, mock_stream_coordinated
from .runner import run as runner_run

logger = logging.getLogger(__name__)

# ── Backend selector ──────────────────────────────────────────────────

_BACKEND = os.environ.get("JUDGE_BACKEND", "mcpconfig").strip().lower()

# ── URL resolution ────────────────────────────────────────────────────

_DEFAULT_120B = "http://127.0.0.1:8001"
_DEFAULT_20B = "http://127.0.0.1:8002"
_MCPCONFIG_URL = os.environ.get("MCPCONFIG_URL", "http://127.0.0.1:8003").rstrip("/")

# Full demo surface — curated for context-budget fit (gpt-oss models @ 16k context).
# Picks the "interesting" tools across hands (browser) + workflow (APIs/creds/2FA/flows) +
# graduated MCP servers (one tool per binary). ~37 tool definitions ~5-7k tokens of schema.
DEFAULT_TOOL_FILTER = [
    # Browser (hands) -- DOM + a11y + screenshots -- 15
    "browser_attach", "browser_navigate", "browser_back", "browser_forward", "browser_reload",
    "browser_get_text", "browser_get_html", "browser_extract_content",
    "browser_click", "browser_type", "browser_press", "browser_select",
    "browser_screenshot", "browser_a11y_snapshot", "browser_scroll",
    # Workflow APIs -- discovery + replay + graduation lifecycle -- 8
    "api_store", "api_call", "api_list", "api_test", "api_call_paginated",
    "api_pending_graduation", "api_graduate", "api_invocation_stats",
    # Workflow credentials -- 4
    "credential_store", "credential_list", "credential_get", "credential_delete",
    # Workflow TOTP / 2FA -- 3
    "totp_register", "totp_register_from_uri", "totp_generate",
    # Workflow flows (record + replay) -- 4
    "flow_record_start", "flow_record_step", "flow_record_stop", "flow_replay",
    # Workflow data shaping -- 1
    "transform_pipe",
    # Graduated MCP servers -- exposed by their binary names -- 1+
    # workflow:api_graduate produces standalone Rust MCP servers, each exposing
    # a single tool whose name is the original API name (snake_case).
    "httpbin_typed_demo",
    "httpbin_post_demo",
]

# MCP servers to spawn for each /run by default. Includes graduated binaries.
DEFAULT_MCP_SERVERS = ["workflow", "hands", "httpbin_typed_demo", "httpbin_post_demo"]


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


# ── Rust mcpconfig SSE streaming ──────────────────────────────────────


async def _stream_via_mcpconfig(
    model: str,
    user_prompt: str,
    history: list[dict] | None = None,
    mcp_servers: list[str] | None = None,
    tool_filter: list[str] | None = None,
    max_iterations: int = 6,
) -> AsyncGenerator[dict, None]:
    """Stream events from Rust mcpconfig /run endpoint (SSE).

    Yields dicts with a ``kind`` key matching the canonical event shapes
    (run_start, tools_registered, llm_request, llm_response, tool_call,
    tool_result, final_answer, run_end, error).

    Raises httpx.ConnectError / httpx.RequestError on connect failure so
    the caller can fall back to runner.
    """
    # TODO: mcpconfig /run accepts a single user_prompt — multi-turn history
    # is not yet supported. For now we send only the latest user message.
    body = {
        "model": model,
        "task": {
            "name": "chat",
            "model": model,
            "user_prompt": user_prompt,
            "max_iterations": max_iterations,
            "mcp_servers": mcp_servers or DEFAULT_MCP_SERVERS,
            "tool_filter": tool_filter or DEFAULT_TOOL_FILTER,
        },
    }

    timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{_MCPCONFIG_URL}/run",
            json=body,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()

            event_type: str | None = None
            data_buf: list[str] = []

            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event_type = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_buf.append(line[len("data:"):].strip())
                elif line == "":
                    # Blank line = end of SSE frame
                    if event_type and data_buf:
                        raw = "\n".join(data_buf)
                        try:
                            parsed = json.loads(raw)
                        except json.JSONDecodeError:
                            parsed = {"kind": event_type, "raw": raw}
                        # Ensure kind is set (mcpconfig already sends it,
                        # but belt-and-suspenders)
                        if "kind" not in parsed:
                            parsed["kind"] = event_type
                        yield parsed
                    event_type = None
                    data_buf = []


# ── Single-model streaming (Coordination = Off) ───────────────────────


def _canonical_event_source(
    backend: str,
    model: str,
    prompt: str,
    history: list[dict] | None,
    mcp_servers: list[str],
    tool_filter: list[str],
) -> AsyncGenerator[dict, None]:
    """Return the right canonical-event async generator based on backend."""
    if backend == "runner":
        return runner_run(
            model=model,
            user_prompt=prompt,
            history=history,
            mcp_servers=mcp_servers,
            tool_filter=tool_filter,
        )
    # "mcpconfig" or first leg of "auto"
    return _stream_via_mcpconfig(
        model=model,
        user_prompt=prompt,
        history=history,
        mcp_servers=mcp_servers,
        tool_filter=tool_filter,
    )


def _translate_canonical_to_delta(event: dict):
    """Translate a single canonical event dict to the delta-shaped dict
    that tabs/free_play.py expects.  Returns a list of 0-2 dicts."""
    kind = event.get("kind", "")
    out: list[dict] = []

    if kind == "llm_response":
        content = event.get("content") or event.get("reasoning") or ""
        if content:
            out.append({"delta": {"content": content}})

    elif kind == "tool_call":
        out.append({
            "delta": {
                "tool_calls": [{
                    "function": {
                        "name": event.get("name", ""),
                        "arguments": event.get("arguments", ""),
                    }
                }]
            }
        })

    elif kind == "tool_result":
        out.append({
            "delta": {
                "tool_result": {
                    "content": event.get("content", ""),
                }
            }
        })

    elif kind == "final_answer":
        content = event.get("content", "")
        if content:
            out.append({"delta": {"content": content}})
        out.append({"delta": {}, "finish_reason": "stop"})

    elif kind == "run_end" and not event.get("ok"):
        out.append({
            "delta": {"content": f"**Error:** {event.get('error', 'unknown')}"},
            "finish_reason": "error",
        })

    return out


async def stream_chat(
    prompt: str, history: list[dict] | None = None
) -> AsyncGenerator[dict, None]:
    """Stream chat from the 120B model with MCP tool support.

    Backend is selected by JUDGE_BACKEND env var:
      - "mcpconfig" (default): Rust agent loop on port 8003
      - "runner": Python agent loop (backend/runner.py)
      - "auto": try mcpconfig first, fall back to runner on connect error
    In mock mode, replays scripted fixtures.
    """
    if is_mock_mode():
        async for event in mock_stream_chat(prompt):
            yield event
        return

    backend = _BACKEND
    mcp_servers = ["workflow", "hands"]
    tool_filter = DEFAULT_TOOL_FILTER

    try:
        source = _canonical_event_source(
            backend if backend != "auto" else "mcpconfig",
            "gpt-oss-120b", prompt, history, mcp_servers, tool_filter,
        )
        async for event in source:
            for out in _translate_canonical_to_delta(event):
                yield out

    except (httpx.ConnectError, httpx.RequestError) as e:
        if backend == "auto":
            logger.warning("mcpconfig unreachable (%s), falling back to runner", e)
            try:
                async for event in runner_run(
                    model="gpt-oss-120b",
                    user_prompt=prompt,
                    history=history,
                    mcp_servers=mcp_servers,
                    tool_filter=tool_filter,
                ):
                    for out in _translate_canonical_to_delta(event):
                        yield out
            except Exception as e2:
                yield {"delta": {"content": f"**Error:** {type(e2).__name__}: {e2}"}, "finish_reason": "error"}
        else:
            yield {"delta": {"content": f"**Error:** {type(e).__name__}: {e}"}, "finish_reason": "error"}

    except Exception as e:
        yield {"delta": {"content": f"**Error:** {type(e).__name__}: {e}"}, "finish_reason": "error"}


# ── Coordinated streaming (Coordination = Coordinator (β)) ────────────


def _translate_canonical_to_coordinated(event: dict, model_id: str):
    """Translate a canonical event to the coordinated-mode shape.
    Returns a list of 0-1 dicts plus an optional total_tokens int."""
    kind = event.get("kind", "")
    out: list[dict] = []
    tokens: int | None = None

    if kind == "llm_response":
        content = event.get("content") or event.get("reasoning") or ""
        if content:
            out.append({"type": "delta", "content": content, "model": model_id})

    elif kind == "tool_call":
        out.append({
            "type": "tool_call",
            "name": event.get("name", ""),
            "arguments": event.get("arguments", ""),
        })

    elif kind == "tool_result":
        out.append({
            "type": "tool_result",
            "content": event.get("content", ""),
        })

    elif kind == "final_answer":
        content = event.get("content", "")
        if content:
            out.append({"type": "delta", "content": content, "model": model_id})

    elif kind == "run_end":
        tokens = event.get("total_tokens", 0)

    return out, tokens


async def stream_chat_coordinated(
    prompt: str, history: list[dict] | None = None
) -> AsyncGenerator[dict, None]:
    """Coordinator mode: classify intent with 20B, then run agent loop on chosen model.

    Backend is selected by JUDGE_BACKEND env var (same as stream_chat).

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
    backend = _BACKEND
    mcp_servers = ["workflow", "hands"]
    tool_filter = DEFAULT_TOOL_FILTER

    try:
        source = _canonical_event_source(
            backend if backend != "auto" else "mcpconfig",
            model_short, prompt, history, mcp_servers, tool_filter,
        )
        async for event in source:
            items, tokens = _translate_canonical_to_coordinated(event, model_id)
            for item in items:
                yield item
            if tokens is not None:
                total_tokens = tokens

    except (httpx.ConnectError, httpx.RequestError) as e:
        if backend == "auto":
            logger.warning("mcpconfig unreachable (%s), falling back to runner", e)
            try:
                async for event in runner_run(
                    model=model_short,
                    user_prompt=prompt,
                    history=history,
                    mcp_servers=mcp_servers,
                    tool_filter=tool_filter,
                ):
                    items, tokens = _translate_canonical_to_coordinated(event, model_id)
                    for item in items:
                        yield item
                    if tokens is not None:
                        total_tokens = tokens
            except Exception as e2:
                yield {"type": "delta", "content": f"**Error:** {type(e2).__name__}: {e2}", "model": model_id}
        else:
            yield {"type": "delta", "content": f"**Error:** {type(e).__name__}: {e}", "model": model_id}

    except Exception as e:
        yield {"type": "delta", "content": f"**Error:** {type(e).__name__}: {e}", "model": model_id}

    total_ms = int((time.monotonic() - start) * 1000)
    yield {"type": "done", "total_ms": total_ms, "total_tokens": total_tokens, "model": model_id, "route": route}
