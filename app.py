"""
AMD MI300X Multi-Agent Judge UI
Submit a task, watch 4 agents race in parallel via SSE streaming from mcpconfig.
"""

import asyncio
import base64
import json
import os
import time
from typing import AsyncGenerator

import gradio as gr
import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MCPCONFIG_URL = os.environ.get("MCPCONFIG_URL", "http://129.212.182.145:8003")
VLLM_DIRECT_URL = os.environ.get("VLLM_DIRECT_URL", "http://129.212.182.145:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")

MODELS = [
    "gpt-oss-120b",
    "qwen3.6-35b-a3b",
    "qwen2.5-72b-instruct-hf",
    "qwen3-32b-hf",
]

MODEL_LABELS = [
    "gpt-oss-120b (AMD MI300X)",
    "Qwen3.6-35B-A3B (AMD MI300X)",
    "Qwen2.5-72B (HF Inference)",
    "Qwen3-32B (HF Inference)",
]

COOLDOWN_SECONDS = 30
SSE_TIMEOUT = 60  # seconds with no events before declaring stall
REQUEST_TIMEOUT = 300  # total request timeout

# ---------------------------------------------------------------------------
# SSE Event Formatting
# ---------------------------------------------------------------------------


def format_event(event: dict) -> str:
    """Convert a parsed SSE event JSON into a renderable markdown/HTML string."""
    kind = event.get("kind", "unknown")

    if kind == "run_start":
        model = event.get("model", event.get("task", {}).get("model", "unknown"))
        return f"🚀 **Started** — model: `{model}`\n\n"

    elif kind == "tools_registered":
        count = event.get("count", 0)
        names = event.get("names", [])
        names_str = ", ".join(names[:10])
        if len(names) > 10:
            names_str += f" ... (+{len(names) - 10} more)"
        return (
            f"<details><summary>🔧 Registered {count} tools</summary>\n\n"
            f"`{names_str}`\n\n</details>\n\n"
        )

    elif kind == "llm_request":
        iteration = event.get("iteration", "?")
        return f"🧠 **Iteration {iteration}** — thinking...\n\n"

    elif kind == "llm_response":
        iteration = event.get("iteration", "?")
        content = event.get("content", "")
        tool_calls_count = event.get("tool_calls_count", 0)
        parts = []
        if content:
            parts.append(content + "\n\n")
        if tool_calls_count > 0:
            parts.append(f"→ {tool_calls_count} tool call(s)\n\n")
        return "".join(parts) if parts else f"🧠 Iteration {iteration} — no content\n\n"

    elif kind == "tool_call":
        name = event.get("name", "unknown")
        args = event.get("args", {})
        args_str = json.dumps(args, indent=2)
        return (
            f"<details><summary>🛠️ <code>{name}(...)</code></summary>\n\n"
            f"```json\n{args_str}\n```\n\n</details>\n\n"
        )

    elif kind == "tool_result":
        name = event.get("name", "unknown")
        result = event.get("result", "")
        # Check for base64 image (browser_screenshot)
        if name == "browser_screenshot" and _is_base64_image(result):
            return (
                f"📤 `{name}` returned screenshot:\n\n"
                f"<img src=\"data:image/png;base64,{result}\" "
                f"style=\"max-width:100%;border:1px solid #444;border-radius:4px;\" />\n\n"
            )
        # Truncate long text results
        result_str = str(result)
        if len(result_str) > 500:
            result_str = result_str[:500] + "\n\n...(truncated)"
        return (
            f"<details><summary>📤 <code>{name}</code> returned</summary>\n\n"
            f"```\n{result_str}\n```\n\n</details>\n\n"
        )

    elif kind == "final_answer":
        content = event.get("content", "")
        return f"---\n\n✅ **Final Answer:**\n\n{content}\n\n"

    elif kind == "run_end":
        ok = event.get("ok", False)
        iters = event.get("iteration_count", "?")
        duration = event.get("duration_ms", 0)
        tokens = event.get("total_tokens", "?")
        duration_s = duration / 1000 if isinstance(duration, (int, float)) else "?"
        status = "✅" if ok else "⚠️"
        return (
            f"---\n\n{status} **Done** — {iters} iterations, "
            f"{duration_s}s, {tokens} tokens\n\n"
        )

    elif kind == "error":
        error = event.get("error", "Unknown error")
        return f"❌ **Error:** {error}\n\n"

    else:
        return f"📎 `{kind}`: {json.dumps(event)}\n\n"


def _is_base64_image(s) -> bool:
    """Quick check if string looks like base64 PNG data."""
    if not isinstance(s, str):
        return False
    if len(s) < 100:
        return False
    try:
        base64.b64decode(s[:64])
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# SSE Stream Consumer
# ---------------------------------------------------------------------------


async def stream_model(model_name: str, prompt: str) -> AsyncGenerator[str, None]:
    """Connect to mcpconfig /run endpoint and yield formatted event strings."""
    payload = {
        "model": model_name,
        "task": {
            "name": f"judge_request_{int(time.time())}",
            "description": "Live judge request",
            "model": model_name,
            "mcp_servers": ["hands"],
            "system_prompt": None,
            "user_prompt": prompt,
            "max_iterations": 8,
            "tool_filter": ["browser_*"],
        },
    }

    accumulated = ""
    last_event_time = time.time()

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream(
                "POST", f"{MCPCONFIG_URL}/run", json=payload
            ) as response:
                if response.status_code != 200:
                    error_text = ""
                    async for chunk in response.aiter_text():
                        error_text += chunk
                    accumulated += f"❌ **Backend error** (HTTP {response.status_code}): {error_text[:200]}\n\n"
                    yield accumulated
                    return

                async for line in response.aiter_lines():
                    # SSE timeout check
                    if time.time() - last_event_time > SSE_TIMEOUT:
                        accumulated += "⏳ **Stalled** — no events for 60s, closing.\n\n"
                        yield accumulated
                        return

                    if not line.startswith("data: "):
                        continue

                    last_event_time = time.time()
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    formatted = format_event(event)
                    accumulated += formatted
                    yield accumulated

                    # Stop after run_end or error
                    if event.get("kind") in ("run_end", "error"):
                        return

    except httpx.ConnectError:
        accumulated += "❌ **Backend unreachable** — cannot connect to mcpconfig server.\n\n"
        yield accumulated
    except httpx.ReadTimeout:
        accumulated += "⏳ **Timeout** — request exceeded 5 minutes.\n\n"
        yield accumulated
    except Exception as e:
        accumulated += f"❌ **Unexpected error:** {type(e).__name__}: {e}\n\n"
        yield accumulated


# ---------------------------------------------------------------------------
# Parallel Runner
# ---------------------------------------------------------------------------


async def run_all_models(prompt: str, last_submit_time: float):
    """
    Run all 4 models in parallel. Yields a tuple of 4 markdown strings
    (one per panel) as events arrive from any stream.
    """
    # Cooldown check
    now = time.time()
    if last_submit_time and (now - last_submit_time) < COOLDOWN_SECONDS:
        remaining = int(COOLDOWN_SECONDS - (now - last_submit_time))
        msg = f"⏳ Cooldown — wait {remaining}s before next submission."
        yield msg, msg, msg, msg, last_submit_time
        return

    if not prompt or not prompt.strip():
        msg = "⚠️ Please enter a task description."
        yield msg, msg, msg, msg, last_submit_time
        return

    new_submit_time = now
    panels = [""] * 4
    panels = [f"⏳ Waiting for `{m}`...\n\n" for m in MODELS]
    yield panels[0], panels[1], panels[2], panels[3], new_submit_time

    # Create async generators for each model
    generators = [stream_model(MODELS[i], prompt) for i in range(4)]

    # Use asyncio.Task to consume each generator independently
    queues: list[asyncio.Queue] = [asyncio.Queue() for _ in range(4)]
    done_flags = [False] * 4

    async def consume(idx: int):
        try:
            async for content in generators[idx]:
                await queues[idx].put(content)
        except Exception as e:
            await queues[idx].put(
                f"❌ **Stream error for {MODELS[idx]}:** {e}\n\n"
            )
        finally:
            done_flags[idx] = True

    # Start all consumers
    tasks = [asyncio.create_task(consume(i)) for i in range(4)]

    # Poll queues and yield updates
    try:
        while not all(done_flags):
            updated = False
            for i in range(4):
                while not queues[i].empty():
                    panels[i] = await queues[i].get()
                    updated = True
            if updated:
                yield panels[0], panels[1], panels[2], panels[3], new_submit_time
            else:
                await asyncio.sleep(0.1)

        # Final drain
        for i in range(4):
            while not queues[i].empty():
                panels[i] = await queues[i].get()
        yield panels[0], panels[1], panels[2], panels[3], new_submit_time

    finally:
        for t in tasks:
            t.cancel()


# ---------------------------------------------------------------------------
# Simple Chat Tab (direct OpenAI-compatible API)
# ---------------------------------------------------------------------------


async def simple_chat(message: str, history: list[dict]) -> str:
    """Direct chat with gpt-oss-120b via OpenAI-compatible endpoint."""
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

    payload = {
        "model": "gpt-oss-120b",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{VLLM_DIRECT_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except httpx.ConnectError:
        return "❌ Backend unreachable — cannot connect to vLLM server."
    except httpx.HTTPStatusError as e:
        return f"❌ API error (HTTP {e.response.status_code}): {e.response.text[:300]}"
    except Exception as e:
        return f"❌ Error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="AMD MI300X Multi-Agent Judge UI",
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="blue"),
    ) as demo:
        gr.Markdown(
            "# AMD MI300X Multi-Agent Bakeoff — Judge UI\n"
            "Submit a custom task. Watch 4 agents race."
        )

        with gr.Tabs():
            # ---------------------------------------------------------------
            # Tab 1: Multi-Agent Bakeoff
            # ---------------------------------------------------------------
            with gr.Tab("Multi-Agent Bakeoff"):
                last_submit = gr.State(value=0.0)

                # 2x2 grid of panels
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"### {MODEL_LABELS[0]}")
                        panel_0 = gr.Markdown(
                            value="*Waiting for prompt...*",
                            label=MODEL_LABELS[0],
                            elem_classes=["agent-panel"],
                        )
                    with gr.Column():
                        gr.Markdown(f"### {MODEL_LABELS[1]}")
                        panel_1 = gr.Markdown(
                            value="*Waiting for prompt...*",
                            label=MODEL_LABELS[1],
                            elem_classes=["agent-panel"],
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"### {MODEL_LABELS[2]}")
                        panel_2 = gr.Markdown(
                            value="*Waiting for prompt...*",
                            label=MODEL_LABELS[2],
                            elem_classes=["agent-panel"],
                        )
                    with gr.Column():
                        gr.Markdown(f"### {MODEL_LABELS[3]}")
                        panel_3 = gr.Markdown(
                            value="*Waiting for prompt...*",
                            label=MODEL_LABELS[3],
                            elem_classes=["agent-panel"],
                        )

                # Shared input
                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="Task for all agents",
                        placeholder="e.g. Navigate to python.org and find the latest Python release version",
                        scale=5,
                    )
                    submit_btn = gr.Button("Submit", variant="primary", scale=1)
                    reset_btn = gr.Button("Reset", variant="secondary", scale=1)

                gr.Markdown(
                    f"*Cooldown: {COOLDOWN_SECONDS}s between submissions. "
                    f"Max 8 iterations per agent.*"
                )

                # Wire up submit
                submit_btn.click(
                    fn=run_all_models,
                    inputs=[prompt_input, last_submit],
                    outputs=[panel_0, panel_1, panel_2, panel_3, last_submit],
                )
                prompt_input.submit(
                    fn=run_all_models,
                    inputs=[prompt_input, last_submit],
                    outputs=[panel_0, panel_1, panel_2, panel_3, last_submit],
                )

                # Reset handler
                def reset_panels():
                    return (
                        "*Waiting for prompt...*",
                        "*Waiting for prompt...*",
                        "*Waiting for prompt...*",
                        "*Waiting for prompt...*",
                        "",
                    )

                reset_btn.click(
                    fn=reset_panels,
                    inputs=[],
                    outputs=[panel_0, panel_1, panel_2, panel_3, prompt_input],
                )

            # ---------------------------------------------------------------
            # Tab 2: Simple Chat
            # ---------------------------------------------------------------
            with gr.Tab("Simple Chat"):
                gr.Markdown(
                    "### Direct Chat with gpt-oss-120b\n"
                    "OpenAI-compatible API on AMD MI300X. No tools, no agent loop — just raw LLM chat."
                )
                gr.ChatInterface(
                    fn=simple_chat,
                    type="messages",
                    examples=[
                        "Explain quantum computing in simple terms",
                        "Write a Python function to find prime numbers",
                        "What are the key differences between TCP and UDP?",
                    ],
                )

    return demo


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
