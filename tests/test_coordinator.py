"""Tests for the Option B coordinator (classify_intent + stream_chat_coordinated)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# Ensure MOCK_MODE is off for unit tests that mock vLLM directly
import os
os.environ["MOCK_MODE"] = "false"
os.environ["MCPCONFIG_URL"] = "http://fake:8000"

from backend.coordinator import classify_intent
from backend.mcpconfig_client import stream_chat_coordinated


def _make_classify_response(word: str) -> MagicMock:
    """Build a fake httpx.Response for the classifier."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": word}}]
    }
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_classify_plan_intent():
    """'Explain step by step why X' should route to 120B (PLAN)."""
    mock_resp = _make_classify_response("PLAN")
    with patch("backend.coordinator.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        route, reason, ms = await classify_intent("Explain step by step why transformers work")
        assert route == "120b"
        assert reason == "PLAN"
        assert isinstance(ms, int)


@pytest.mark.asyncio
async def test_classify_exec_intent():
    """'Click submit' should route to 20B (EXEC)."""
    mock_resp = _make_classify_response("EXEC")
    with patch("backend.coordinator.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        route, reason, ms = await classify_intent("Click submit")
        assert route == "20b"
        assert reason == "EXEC"


@pytest.mark.asyncio
async def test_classify_default_to_exec_on_garbage():
    """Malformed classifier reply defaults to 20B (EXEC)."""
    mock_resp = _make_classify_response("I don't understand the question")
    with patch("backend.coordinator.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        route, reason, ms = await classify_intent("asdf jkl;")
        assert route == "20b"
        assert reason == "EXEC"


@pytest.mark.asyncio
async def test_stream_coordinated_emits_meta_first():
    """In coordinator mode, the first event must be type=meta."""
    # Force mock mode for this test
    with patch("backend.mcpconfig_client.is_mock_mode", return_value=True):
        events = []
        async for event in stream_chat_coordinated("What time is it?"):
            events.append(event)
            if len(events) >= 2:
                break

        assert len(events) >= 1
        assert events[0]["type"] == "meta"
        assert events[0]["route"] in ("20b", "120b")
        assert events[0]["reason"] in ("PLAN", "EXEC")


@pytest.mark.asyncio
async def test_mock_mode_uses_fixture():
    """Mock mode reads from mocks/coordinator_routes.json."""
    fixture_path = Path(__file__).parent.parent / "mocks" / "coordinator_routes.json"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with patch("backend.mcpconfig_client.is_mock_mode", return_value=True):
        events = []
        async for event in stream_chat_coordinated("What time is it?"):
            events.append(event)

        # Should match exec_example events (simple lookup → EXEC)
        expected_events = data["exec_example"]["events"]
        assert len(events) == len(expected_events)
        assert events[0]["type"] == "meta"
        assert events[-1]["type"] == "done"
