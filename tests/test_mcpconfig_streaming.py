"""Tests for mcpconfig SSE streaming and backend selector."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ── Helpers: fake SSE response ────────────────────────────────────────

SAMPLE_SSE = "\n".join([
    "event: run_start",
    'data: {"kind":"run_start","ts":"2025-01-01T00:00:00Z","model":"gpt-oss-120b"}',
    "",
    "event: tools_registered",
    'data: {"kind":"tools_registered","count":2,"names":["api_list","api_call"]}',
    "",
    "event: llm_request",
    'data: {"kind":"llm_request","iteration":1,"model":"gpt-oss-120b","message_count":1}',
    "",
    "event: llm_response",
    'data: {"kind":"llm_response","iteration":1,"content":null,"tool_calls":[{"id":"call_1","name":"api_list","arguments":"{}"}],"usage":{"total_tokens":50}}',
    "",
    "event: tool_call",
    'data: {"kind":"tool_call","iteration":1,"id":"call_1","name":"api_list","arguments":"{}"}',
    "",
    "event: tool_result",
    'data: {"kind":"tool_result","iteration":1,"id":"call_1","ok":true,"content":"[api1, api2]"}',
    "",
    "event: final_answer",
    'data: {"kind":"final_answer","iteration":2,"content":"Here are your APIs: api1, api2"}',
    "",
    "event: run_end",
    'data: {"kind":"run_end","ok":true,"duration_ms":645,"iterations":2,"total_tokens":120}',
    "",
])


class FakeLineStream:
    """Mimics httpx response.aiter_lines() for SSE content."""

    def __init__(self, text: str):
        self._lines = text.split("\n")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class FakeAsyncClient:
    """Mimics httpx.AsyncClient for stream()."""

    def __init__(self, sse_text: str):
        self._sse = sse_text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, method, url, **kwargs):
        return FakeLineStream(self._sse)


# ── Unit tests: SSE parsing ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_via_mcpconfig_parses_all_events():
    """Verify _stream_via_mcpconfig yields correct event sequence from SSE."""
    with patch("backend.mcpconfig_client.httpx.AsyncClient", return_value=FakeAsyncClient(SAMPLE_SSE)):
        from backend.mcpconfig_client import _stream_via_mcpconfig

        events = []
        async for event in _stream_via_mcpconfig(
            model="gpt-oss-120b",
            user_prompt="List APIs",
        ):
            events.append(event)

    kinds = [e["kind"] for e in events]
    assert kinds == [
        "run_start",
        "tools_registered",
        "llm_request",
        "llm_response",
        "tool_call",
        "tool_result",
        "final_answer",
        "run_end",
    ]

    # Verify tool_call fields
    tc = next(e for e in events if e["kind"] == "tool_call")
    assert tc["name"] == "api_list"
    assert tc["arguments"] == "{}"

    # Verify tool_result fields
    tr = next(e for e in events if e["kind"] == "tool_result")
    assert tr["ok"] is True

    # Verify final_answer
    fa = next(e for e in events if e["kind"] == "final_answer")
    assert "api1" in fa["content"]

    # Verify run_end
    end = next(e for e in events if e["kind"] == "run_end")
    assert end["ok"] is True
    assert end["total_tokens"] == 120


@pytest.mark.asyncio
async def test_stream_via_mcpconfig_handles_missing_kind():
    """If data JSON lacks 'kind', it should be injected from the event: line."""
    sse = "\n".join([
        "event: custom_event",
        'data: {"foo":"bar"}',
        "",
    ])
    with patch("backend.mcpconfig_client.httpx.AsyncClient", return_value=FakeAsyncClient(sse)):
        from backend.mcpconfig_client import _stream_via_mcpconfig

        events = []
        async for event in _stream_via_mcpconfig(model="gpt-oss-120b", user_prompt="test"):
            events.append(event)

    assert len(events) == 1
    assert events[0]["kind"] == "custom_event"
    assert events[0]["foo"] == "bar"


@pytest.mark.asyncio
async def test_stream_via_mcpconfig_handles_bad_json():
    """Non-JSON data should be wrapped with raw field."""
    sse = "\n".join([
        "event: error",
        "data: not valid json",
        "",
    ])
    with patch("backend.mcpconfig_client.httpx.AsyncClient", return_value=FakeAsyncClient(sse)):
        from backend.mcpconfig_client import _stream_via_mcpconfig

        events = []
        async for event in _stream_via_mcpconfig(model="gpt-oss-120b", user_prompt="test"):
            events.append(event)

    assert len(events) == 1
    assert events[0]["kind"] == "error"
    assert events[0]["raw"] == "not valid json"


# ── Unit tests: translation helpers ──────────────────────────────────


def test_translate_canonical_to_delta_tool_call():
    from backend.mcpconfig_client import _translate_canonical_to_delta

    event = {"kind": "tool_call", "name": "api_list", "arguments": "{}"}
    out = _translate_canonical_to_delta(event)
    assert len(out) == 1
    assert out[0]["delta"]["tool_calls"][0]["function"]["name"] == "api_list"


def test_translate_canonical_to_delta_final_answer():
    from backend.mcpconfig_client import _translate_canonical_to_delta

    event = {"kind": "final_answer", "content": "Done"}
    out = _translate_canonical_to_delta(event)
    assert len(out) == 2
    assert out[0]["delta"]["content"] == "Done"
    assert out[1]["finish_reason"] == "stop"


def test_translate_canonical_to_delta_error():
    from backend.mcpconfig_client import _translate_canonical_to_delta

    event = {"kind": "run_end", "ok": False, "error": "timeout"}
    out = _translate_canonical_to_delta(event)
    assert len(out) == 1
    assert "timeout" in out[0]["delta"]["content"]
    assert out[0]["finish_reason"] == "error"


def test_translate_canonical_to_delta_ignores_run_start():
    from backend.mcpconfig_client import _translate_canonical_to_delta

    event = {"kind": "run_start", "model": "gpt-oss-120b"}
    out = _translate_canonical_to_delta(event)
    assert out == []


def test_translate_canonical_to_coordinated():
    from backend.mcpconfig_client import _translate_canonical_to_coordinated

    event = {"kind": "tool_call", "name": "api_list", "arguments": "{}"}
    out, tokens = _translate_canonical_to_coordinated(event, "openai/gpt-oss-120b")
    assert len(out) == 1
    assert out[0]["type"] == "tool_call"
    assert tokens is None

    event2 = {"kind": "run_end", "total_tokens": 200}
    out2, tokens2 = _translate_canonical_to_coordinated(event2, "openai/gpt-oss-120b")
    assert out2 == []
    assert tokens2 == 200


# ── Backend selector: auto fallback ──────────────────────────────────


@pytest.mark.asyncio
async def test_auto_fallback_on_connect_error():
    """JUDGE_BACKEND=auto should fall back to runner when mcpconfig is unreachable."""
    from backend.mcpconfig_client import _translate_canonical_to_delta

    # Make _stream_via_mcpconfig raise ConnectError
    async def fake_mcpconfig_fail(**kw):
        raise httpx.ConnectError("refused")
        yield  # noqa: unreachable — makes this an async generator

    # Make runner yield a simple final_answer
    async def fake_runner(**kw):
        yield {"kind": "final_answer", "iteration": 1, "content": "from runner"}
        yield {"kind": "run_end", "ok": True, "duration_ms": 100, "iterations": 1, "total_tokens": 10}

    with patch("backend.mcpconfig_client._BACKEND", "auto"), \
         patch("backend.mcpconfig_client._stream_via_mcpconfig", side_effect=fake_mcpconfig_fail), \
         patch("backend.mcpconfig_client.runner_run", side_effect=fake_runner), \
         patch("backend.mcpconfig_client.is_mock_mode", return_value=False):

        from backend.mcpconfig_client import stream_chat

        events = []
        async for event in stream_chat("hello"):
            events.append(event)

    # Should have gotten the runner's final_answer translated to delta
    contents = [e["delta"].get("content", "") for e in events if "delta" in e and e["delta"].get("content")]
    assert any("from runner" in c for c in contents)


# ── Integration test (skips if mcpconfig not reachable) ───────────────


MCPCONFIG_URL = os.environ.get("MCPCONFIG_URL", "http://127.0.0.1:8003")


def _mcpconfig_reachable() -> bool:
    try:
        import httpx as _httpx
        r = _httpx.get(f"{MCPCONFIG_URL}/health", timeout=3)
        return r.status_code < 500
    except Exception:
        return False


@pytest.mark.asyncio
@pytest.mark.skipif(not _mcpconfig_reachable(), reason="mcpconfig not reachable")
async def test_live_mcpconfig_stream():
    """Integration: stream a real request through mcpconfig /run."""
    from backend.mcpconfig_client import _stream_via_mcpconfig

    events = []
    async for event in _stream_via_mcpconfig(
        model="gpt-oss-120b",
        user_prompt="List the stored APIs by calling api_list",
        mcp_servers=["workflow"],
        tool_filter=["api_list"],
        max_iterations=2,
    ):
        events.append(event)

    kinds = [e["kind"] for e in events]
    assert "run_start" in kinds
    assert "run_end" in kinds
