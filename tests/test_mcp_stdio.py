"""Integration tests for MCPClient over stdio.

These require the workflow MCP binary on the droplet. They skip cleanly
if the binary is missing or the droplet is unreachable.
"""

import asyncio
import os
import shutil

import pytest

WORKFLOW_PATH = os.environ.get(
    "MCP_WORKFLOW_PATH", "/root/workflow/target/release/workflow"
)


def _can_run() -> bool:
    return shutil.which(WORKFLOW_PATH) is not None or os.path.isfile(WORKFLOW_PATH)


pytestmark = pytest.mark.skipif(
    not _can_run(),
    reason=f"Workflow binary not found at {WORKFLOW_PATH}",
)


MCP_ENV = {"WORKFLOW_PLAIN_CREDS_OK": "1"}


@pytest.mark.asyncio
async def test_list_tools():
    from backend.mcp_stdio import MCPClient

    client = await MCPClient.spawn("workflow", WORKFLOW_PATH, env=MCP_ENV)
    try:
        await client.initialize()
        tools = await client.list_tools()
        assert len(tools) > 0
        names = [t["name"] for t in tools]
        assert "api_list" in names
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_call_api_list():
    from backend.mcp_stdio import MCPClient

    client = await MCPClient.spawn("workflow", WORKFLOW_PATH, env=MCP_ENV)
    try:
        await client.initialize()
        result = await client.call_tool("api_list", {})
        assert "content" in result
        assert isinstance(result["content"], list)
    finally:
        await client.shutdown()
