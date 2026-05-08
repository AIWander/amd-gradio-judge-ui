"""Multi-server tool registry. Mirrors mcpconfig/src/registry.rs.

Holds N MCPClients, aggregates their tools, applies glob filters,
dispatches calls by tool name, and handles namespace collisions with
``server__tool`` prefixing.
"""

import fnmatch

from .mcp_stdio import MCPClient


class ToolRegistry:

    def __init__(self):
        self._clients: dict[str, MCPClient] = {}
        self._tool_owners: dict[str, str] = {}
        self._raw_tools: dict[str, dict] = {}

    async def add_server(
        self,
        name: str,
        client: MCPClient,
        tools: list[dict],
        filters: list[str] | None = None,
    ) -> None:
        if filters:
            tools = [
                t for t in tools
                if any(fnmatch.fnmatch(t["name"], f) for f in filters)
            ]

        for tool in tools:
            bare = tool["name"]
            if bare in self._tool_owners:
                existing = self._tool_owners.pop(bare)
                old_def = self._raw_tools.pop(bare)
                self._raw_tools[f"{existing}__{bare}"] = old_def
                self._tool_owners[f"{existing}__{bare}"] = existing
                self._raw_tools[f"{name}__{bare}"] = tool
                self._tool_owners[f"{name}__{bare}"] = name
            else:
                self._raw_tools[bare] = tool
                self._tool_owners[bare] = name

        self._clients[name] = client

    def to_responses_tools(self) -> list[dict]:
        """Convert registered tools to the /v1/responses tools format."""
        out = []
        for exposed_name, mcp_tool in self._raw_tools.items():
            schema = mcp_tool.get("inputSchema", {"type": "object", "properties": {}})
            out.append({
                "type": "function",
                "name": exposed_name,
                "description": mcp_tool.get("description", ""),
                "parameters": schema,
            })
        return out

    async def dispatch(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        """Call a tool and return (ok, text_content).

        Handles model-emitted prefixes: the LLM may call ``server:tool``
        even though the tool was registered as ``tool``.
        """
        server = self._tool_owners.get(tool_name)

        # Model may add a server: prefix (e.g. "workflow:api_list")
        if not server and ":" in tool_name:
            tool_name = tool_name.split(":", 1)[1]
            server = self._tool_owners.get(tool_name)

        if not server:
            return False, f"Unknown tool: {tool_name}"

        mcp_name = tool_name
        if "__" in tool_name:
            prefix = f"{server}__"
            if tool_name.startswith(prefix):
                mcp_name = tool_name[len(prefix):]

        client = self._clients.get(server)
        if not client:
            return False, f"No client for server: {server}"

        result = await client.call_tool(mcp_name, arguments)

        is_error = result.get("isError", False)
        parts = result.get("content", [])
        text = "\n".join(
            p.get("text", "") for p in parts if p.get("type") == "text"
        )
        return not is_error, text

    @property
    def tool_count(self) -> int:
        return len(self._raw_tools)

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._raw_tools.keys())

    async def shutdown_all(self) -> None:
        for client in self._clients.values():
            try:
                await client.shutdown()
            except Exception:
                pass
