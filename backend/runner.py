"""Agent loop driving MCP tools via vLLM /v1/responses.

Mirrors the loop in mcpconfig/src/agent.rs but uses the Responses API
(Harmony-native) instead of /v1/chat/completions. Yields canonical events
whose shapes match mcpconfig/src/events.rs exactly.

Public surface:
    async def run(...) -> AsyncGenerator[dict, None]
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import AsyncGenerator

from .mcp_stdio import MCPClient
from .responses_client import create as responses_create
from .tool_registry import ToolRegistry


_MCP_SERVERS: dict[str, dict] = {
    "workflow": {
        "path": os.environ.get(
            "MCP_WORKFLOW_PATH", "/root/workflow/target/release/workflow"
        ),
        "env": {"WORKFLOW_PLAIN_CREDS_OK": "1"},
    },
    "hands": {
        "path": os.environ.get(
            "MCP_HANDS_PATH", "/root/hands/target/release/hands"
        ),
        "env": {},
    },
}

_DEFAULT_SYSTEM = (
    "You are a tool-using agent. Use the provided tools to answer the "
    "user's request. After getting tool results, provide a clear final answer "
    "that incorporates what you learned from the tools."
)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _vllm_base(model: str) -> tuple[str, str]:
    if "120b" in model:
        base = (
            os.environ.get("VLLM_120B_URL", "").strip()
            or os.environ.get("VLLM_BASE_URL", "").strip()
            or "http://127.0.0.1:8001"
        ).rstrip("/")
        return base, "openai/gpt-oss-120b"
    base = (
        os.environ.get("VLLM_20B_URL", "").strip()
        or os.environ.get("VLLM_BASE_URL_8002", "").strip()
        or "http://127.0.0.1:8002"
    ).rstrip("/")
    return base, "openai/gpt-oss-20b"


async def _summarize_fallback(
    base_url: str,
    model_id: str,
    sys_prompt: str,
    user_prompt: str,
    tool_result: str,
) -> str:
    """Fallback when multi-turn function_call_output fails (vLLM #33089).

    Makes a fresh /v1/responses call with the tool result inlined as a
    user message so the model can summarize it without needing the
    function_call_output input format.
    """
    fallback_input = [
        {"role": "user", "content": user_prompt},
        {
            "role": "user",
            "content": (
                f"Here is the tool result:\n\n{tool_result}\n\n"
                "Please provide a clear answer based on this data."
            ),
        },
    ]
    try:
        resp = await responses_create(
            base_url=base_url,
            model=model_id,
            input_items=fallback_input,
            tools=None,
            instructions=sys_prompt,
            max_output_tokens=2048,
        )
        for item in resp.get("output", []):
            if item.get("type") == "message":
                parts = item.get("content", [])
                text = " ".join(
                    p.get("text", "")
                    for p in parts
                    if p.get("type") == "output_text"
                )
                if text:
                    return text
    except Exception:
        pass
    return tool_result


async def run(
    model: str,
    user_prompt: str,
    history: list[dict] | None = None,
    mcp_servers: list[str] = ("workflow",),
    tool_filter: list[str] | None = None,
    max_iterations: int = 4,
    system_prompt: str | None = None,
) -> AsyncGenerator[dict, None]:
    """Drive a tool-using agent loop. Yields canonical events."""
    base_url, model_id = _vllm_base(model)
    sys_prompt = system_prompt or _DEFAULT_SYSTEM

    yield {
        "kind": "run_start",
        "ts": _ts(),
        "task": "chat",
        "model": model_id,
        "base_url": base_url,
        "user_prompt": user_prompt,
        "mcp_servers": list(mcp_servers),
    }

    registry = ToolRegistry()
    start = time.monotonic()
    total_tokens = 0
    iteration = 0

    try:
        for server_name in mcp_servers:
            cfg = _MCP_SERVERS.get(server_name)
            if not cfg:
                continue
            client = await MCPClient.spawn(
                server_name, cfg["path"], env=cfg.get("env"),
            )
            await client.initialize()
            tools = await client.list_tools()
            await registry.add_server(
                server_name, client, tools,
                list(tool_filter) if tool_filter else None,
            )

        yield {
            "kind": "tools_registered",
            "ts": _ts(),
            "count": registry.tool_count,
            "names": registry.tool_names,
        }

        last_result_text = ""
        input_items: list[dict] = []
        if history:
            for msg in history:
                input_items.append({"role": msg["role"], "content": msg["content"]})
        input_items.append({"role": "user", "content": user_prompt})

        tools_for_llm = registry.to_responses_tools()

        for iteration in range(1, max_iterations + 1):
            yield {
                "kind": "llm_request",
                "ts": _ts(),
                "iteration": iteration,
                "model": model_id,
                "message_count": len(input_items),
            }

            try:
                resp = await responses_create(
                    base_url=base_url,
                    model=model_id,
                    input_items=input_items,
                    tools=tools_for_llm or None,
                    instructions=sys_prompt,
                    max_output_tokens=4096,
                )
            except Exception:
                if iteration == 1:
                    raise
                # Multi-turn broke (vLLM #33089). Fallback: ask the model
                # to summarize the tool results as a plain user message.
                fallback_answer = await _summarize_fallback(
                    base_url, model_id, sys_prompt,
                    user_prompt, last_result_text,
                )
                yield {
                    "kind": "final_answer",
                    "ts": _ts(),
                    "iteration": iteration,
                    "content": fallback_answer,
                }
                break

            usage = resp.get("usage", {})
            total_tokens += usage.get("total_tokens", 0)

            output = resp.get("output", [])
            content_text = ""
            reasoning_text = ""
            tool_calls: list[dict] = []

            for item in output:
                itype = item.get("type", "")
                if itype == "reasoning":
                    parts = item.get("content", [])
                    reasoning_text = " ".join(
                        p.get("text", "") for p in parts
                    )
                elif itype == "message":
                    parts = item.get("content", [])
                    content_text = " ".join(
                        p.get("text", "")
                        for p in parts
                        if p.get("type") == "output_text"
                    )
                elif itype == "function_call":
                    tool_calls.append({
                        "id": item.get("call_id", ""),
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    })

            yield {
                "kind": "llm_response",
                "ts": _ts(),
                "iteration": iteration,
                "content": content_text or None,
                "reasoning": reasoning_text or None,
                "tool_calls": tool_calls,
                "usage": usage,
            }

            if not tool_calls:
                yield {
                    "kind": "final_answer",
                    "ts": _ts(),
                    "iteration": iteration,
                    "content": content_text,
                }
                break

            # Dispatch each tool call
            for tc in tool_calls:
                yield {
                    "kind": "tool_call",
                    "ts": _ts(),
                    "iteration": iteration,
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                }

                try:
                    args = json.loads(tc["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {}

                ok, result_text = await registry.dispatch(tc["name"], args)
                last_result_text = result_text

                yield {
                    "kind": "tool_result",
                    "ts": _ts(),
                    "iteration": iteration,
                    "id": tc["id"],
                    "ok": ok,
                    "content": result_text,
                }

                input_items.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                })
                input_items.append({
                    "type": "function_call_output",
                    "call_id": tc["id"],
                    "output": result_text,
                })
        else:
            # Exhausted max_iterations with tool calls still pending.
            # Try one final LLM call without tools to get a summary.
            try:
                summary_resp = await responses_create(
                    base_url=base_url,
                    model=model_id,
                    input_items=input_items,
                    tools=None,
                    instructions=sys_prompt,
                    max_output_tokens=2048,
                )
                s_usage = summary_resp.get("usage", {})
                total_tokens += s_usage.get("total_tokens", 0)
                s_text = ""
                for item in summary_resp.get("output", []):
                    if item.get("type") == "message":
                        for p in item.get("content", []):
                            if p.get("type") == "output_text":
                                s_text += p.get("text", "")
                yield {
                    "kind": "final_answer",
                    "ts": _ts(),
                    "iteration": iteration,
                    "content": s_text or last_result_text,
                }
            except Exception:
                yield {
                    "kind": "final_answer",
                    "ts": _ts(),
                    "iteration": iteration,
                    "content": last_result_text,
                }

        duration_ms = int((time.monotonic() - start) * 1000)
        yield {
            "kind": "run_end",
            "ts": _ts(),
            "ok": True,
            "duration_ms": duration_ms,
            "iterations": iteration,
            "total_tokens": total_tokens,
        }

    except Exception as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        yield {
            "kind": "run_end",
            "ts": _ts(),
            "ok": False,
            "duration_ms": duration_ms,
            "iterations": iteration,
            "total_tokens": total_tokens,
            "error": str(e),
        }

    finally:
        await registry.shutdown_all()
