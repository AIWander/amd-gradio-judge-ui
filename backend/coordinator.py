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


def _classifier_url() -> str:
    """Return base URL for the 20B classifier (without trailing /v1).

    Resolution order:
      1. VLLM_20B_URL (e.g. http://127.0.0.1:8002)
      2. VLLM_BASE_URL_8002
      3. http://127.0.0.1:8002 (sane default)
    """
    return (
        os.environ.get("VLLM_20B_URL", "").strip()
        or os.environ.get("VLLM_BASE_URL_8002", "").strip()
        or "http://127.0.0.1:8002"
    )


async def classify_intent(
    prompt: str, history: list | None = None
) -> tuple[str, str, int]:
    """Call 20B with a router prompt to classify intent.

    Returns (route, reason, classifier_ms):
        route: "120b" or "20b"
        reason: "PLAN" or "EXEC"
        classifier_ms: wall-clock time in milliseconds
    """
    base_url = _classifier_url().rstrip("/")
    start = time.monotonic()

    # gpt-oss is a reasoning model: tokens spent in reasoning_content come out
    # of the same budget as content. Need ~200 tokens to clear the thinking
    # phase and emit the final PLAN/EXEC verdict.
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": ROUTER_PROMPT.format(prompt=prompt)}],
        "stream": False,
        "max_tokens": 256,
        "temperature": 0,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("choices", [{}])[0].get("message", {}) or {}
            # Prefer answer channel; fall back to reasoning channel if the
            # model burned all tokens thinking.
            text = (msg.get("content") or msg.get("reasoning_content") or "").strip().upper()
    except Exception:
        text = ""

    elapsed_ms = int((time.monotonic() - start) * 1000)

    if "PLAN" in text:
        return "120b", "PLAN", elapsed_ms
    # Default to EXEC (fast model) on anything else
    return "20b", "EXEC", elapsed_ms
