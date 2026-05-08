"""Thin wrapper around vLLM's /v1/responses endpoint.

Pass-through over httpx — no parsing logic beyond JSON decode.
"""

from typing import Any

import httpx


async def create(
    base_url: str,
    model: str,
    input_items: str | list[dict],
    tools: list[dict] | None = None,
    instructions: str | None = None,
    max_output_tokens: int = 4096,
) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "max_output_tokens": max_output_tokens,
    }
    if tools:
        payload["tools"] = tools
    if instructions:
        payload["instructions"] = instructions

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/v1/responses",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()
