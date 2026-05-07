"""Mock backend for MOCK_MODE=true. Replays scripted event sequences."""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import AsyncGenerator

MOCKS_DIR = Path(__file__).parent.parent / "mocks"

PROMPT_ROUTES = [
    (re.compile(r"api.?list|learned|session", re.I), "free_play_api_list.json"),
    (re.compile(r"browse.*httpbin|discover.*graduate|HAR|capture.*contract", re.I), "scenario_a.json"),
    (re.compile(r"what did you learn|narrate.*connector|capability", re.I), "scenario_b.json"),
    (re.compile(r"hierarchical|20b.*120b|two.?model|clarif", re.I), "scenario_c.json"),
]


def is_mock_mode() -> bool:
    val = os.environ.get("MOCK_MODE", "true").strip().lower()
    mcpconfig = os.environ.get("MCPCONFIG_URL", "").strip()
    return val == "true" or not mcpconfig


def _pick_fixture(prompt: str) -> str:
    for pattern, filename in PROMPT_ROUTES:
        if pattern.search(prompt):
            return filename
    return "free_play_default.json"


def _load_fixture(filename: str) -> list[dict]:
    path = MOCKS_DIR / filename
    if not path.exists():
        return [
            {"delta": {"role": "assistant", "content": ""}, "delay": 0.2},
            {"delta": {"content": f"[Mock] No fixture found: {filename}"}, "delay": 0.1},
            {"delta": {"content": ""}, "finish_reason": "stop", "delay": 0.1},
        ]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def mock_stream_chat(prompt: str) -> AsyncGenerator[dict, None]:
    """Yield mock SSE-shaped events with realistic timing."""
    filename = _pick_fixture(prompt)
    events = _load_fixture(filename)
    for event in events:
        delay = event.get("delay", 0.3)
        await asyncio.sleep(delay)
        yield event


def mock_vllm_metrics() -> dict:
    path = MOCKS_DIR / "system_metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mock_graduated_repos() -> list[dict]:
    path = MOCKS_DIR / "system_repos.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mock_lifecycle() -> dict:
    path = MOCKS_DIR / "system_lifecycle.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
