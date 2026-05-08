"""Smoke test for runner.run() against live vLLM + workflow on droplet.

Skips cleanly if the droplet is unreachable or env vars are missing.
"""

import os
import shutil

import pytest

WORKFLOW_PATH = os.environ.get(
    "MCP_WORKFLOW_PATH", "/root/workflow/target/release/workflow"
)
VLLM_URL = (
    os.environ.get("VLLM_120B_URL", "").strip()
    or os.environ.get("VLLM_BASE_URL", "").strip()
)


def _can_run() -> bool:
    has_binary = shutil.which(WORKFLOW_PATH) is not None or os.path.isfile(WORKFLOW_PATH)
    has_vllm = bool(VLLM_URL)
    return has_binary and has_vllm


pytestmark = pytest.mark.skipif(
    not _can_run(),
    reason="Requires vLLM URL + workflow binary on droplet",
)


@pytest.mark.asyncio
async def test_runner_single_tool_call():
    """Run a simple tool-call task and verify canonical event shapes."""
    from backend.runner import run

    events = []
    async for event in run(
        model="gpt-oss-120b",
        user_prompt="List the stored APIs by calling api_list",
        mcp_servers=["workflow"],
        tool_filter=["api_list"],
        max_iterations=2,
    ):
        events.append(event)

    kinds = [e["kind"] for e in events]

    assert "run_start" in kinds
    assert "tools_registered" in kinds
    assert "llm_request" in kinds
    assert "llm_response" in kinds
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "final_answer" in kinds
    assert "run_end" in kinds

    reg = next(e for e in events if e["kind"] == "tools_registered")
    assert reg["count"] >= 1
    assert "api_list" in reg["names"]

    tc = next(e for e in events if e["kind"] == "tool_call")
    assert tc["name"] == "api_list"

    tr = next(e for e in events if e["kind"] == "tool_result")
    assert tr["ok"] is True

    end = next(e for e in events if e["kind"] == "run_end")
    assert end["ok"] is True
    assert end["total_tokens"] > 0
