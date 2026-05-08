"""Async MCP client over stdio JSON-RPC.

Mirrors mcpconfig/src/mcp.rs: spawns an MCP server subprocess, sends
JSON-RPC requests over stdin, reads responses from stdout. Supports both
bare JSON lines and LSP Content-Length framing.
"""

import asyncio
import json
import os
from typing import Any


class MCPClient:

    def __init__(self, name: str, process: asyncio.subprocess.Process):
        self.name = name
        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._next_id = 1

    @classmethod
    async def spawn(
        cls,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> "MCPClient":
        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        proc = await asyncio.create_subprocess_exec(
            command,
            *(args or []),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=full_env,
        )
        return cls(name, proc)

    async def _send_request(self, method: str, params: Any = None) -> Any:
        req_id = self._next_id
        self._next_id += 1
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg) + "\n"
        self._stdin.write(line.encode())
        await self._stdin.drain()
        return await self._read_response()

    async def _send_notification(self, method: str, params: Any = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg) + "\n"
        self._stdin.write(line.encode())
        await self._stdin.drain()

    async def _read_response(self) -> Any:
        """Read one JSON-RPC response, handling bare JSON and LSP framing."""
        while True:
            raw = await self._stdout.readline()
            if not raw:
                raise ConnectionError(f"MCP server '{self.name}' closed stdout")
            text = raw.decode().strip()
            if not text:
                continue

            if text.startswith("{"):
                body = json.loads(text)
            elif text.startswith("Content-Length:"):
                length = int(text.split(":")[1].strip())
                while True:
                    hdr = await self._stdout.readline()
                    if not hdr or hdr.strip() == b"":
                        break
                data = await self._stdout.readexactly(length)
                body = json.loads(data)
            else:
                continue

            if body.get("id") is None:
                continue
            if body.get("error"):
                err = body["error"]
                raise RuntimeError(
                    f"MCP error {err.get('code')}: {err.get('message')}"
                )
            return body.get("result")

    async def initialize(self) -> dict:
        result = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "judge-ui-runner", "version": "0.1.0"},
            },
        )
        await self._send_notification("notifications/initialized")
        return result

    async def list_tools(self) -> list[dict]:
        result = await self._send_request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> dict:
        return await self._send_request(
            "tools/call", {"name": name, "arguments": arguments}
        )

    async def shutdown(self) -> None:
        if self._stdin and not self._stdin.is_closing():
            self._stdin.close()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._process.kill()
