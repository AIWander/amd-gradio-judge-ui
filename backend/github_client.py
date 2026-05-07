"""Client for GitHub API — fetches graduated MCP repos from AIWander org."""

import os
import re
import time

import httpx

from .mock import is_mock_mode, mock_graduated_repos

_cache: dict = {"data": None, "ts": 0}
CACHE_TTL = 300  # 5 minutes

MCP_PATTERN = re.compile(r"^mcp-(?!template-from-)")


async def get_graduated_repos() -> list[dict]:
    """Fetch AIWander repos matching ^mcp- (excluding template forks). Cached 5min."""
    if is_mock_mode():
        return mock_graduated_repos()

    now = time.time()
    if _cache["data"] is not None and (now - _cache["ts"]) < CACHE_TTL:
        return _cache["data"]

    org = os.environ.get("GITHUB_ORG", "AIWander")
    token = os.environ.get("GITHUB_TOKEN", "").strip()

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"https://api.github.com/orgs/{org}/repos",
                params={"per_page": 100, "sort": "created", "direction": "desc"},
                headers=headers,
            )
            resp.raise_for_status()
            all_repos = resp.json()

        filtered = [
            r for r in all_repos
            if MCP_PATTERN.match(r.get("name", ""))
        ]

        result = []
        for r in filtered:
            result.append({
                "name": r["name"],
                "stargazers_count": r.get("stargazers_count", 0),
                "pushed_at": r.get("pushed_at", ""),
                "created_at": r.get("created_at", ""),
                "html_url": r.get("html_url", ""),
            })

        _cache["data"] = result
        _cache["ts"] = now
        return result

    except Exception:
        if _cache["data"] is not None:
            return _cache["data"]
        return mock_graduated_repos()
