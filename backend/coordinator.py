"""Option B coordinator: 20B classifies intent, routes turns to 20B or 120B."""

import json
import os
import time

import httpx

ROUTER_PROMPT = (
    "You're routing a user message between two models. "
    "PLAN = deep reasoning, multi-step planning, complex analysis, novel problems. "
    "EXEC = quick tool execution, simple lookups, conversational responses. "
    "Reply with exactly one word: PLAN or EXEC.\n\n"
    "User message: {prompt}"
)


def _get_url() -> str:
    return os.environ.get("MCPCONFIG_URL", "").strip()


async def classify_intent(
    prompt: str, history: list | None = None
) -> tuple[str, str, int]:
    """Call 20B with a router prompt to classify intent.

    Returns (route, reason, classifier_ms):
        route: "120b" or "20b"
        reason: "PLAN" or "EXEC"
        classifier_ms: wall-clock time in milliseconds
    """
    base_url = _get_url()
    start = time.monotonic()

    payload = {
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": ROUTER_PROMPT.format(prompt=prompt)}],
        "stream": False,
        "max_tokens": 10,
        "temperature": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .upper()
            )
    except Exception:
        text = ""

    elapsed_ms = int((time.monotonic() - start) * 1000)

    if "PLAN" in text:
        return "120b", "PLAN", elapsed_ms
    # Default to EXEC (fast model) on anything else
    return "20b", "EXEC", elapsed_ms
