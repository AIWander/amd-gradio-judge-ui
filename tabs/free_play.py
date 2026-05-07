"""Tab 1: Free Play — chat + live MCP tool-call trace."""

import asyncio
import random

import gradio as gr

from backend.mcpconfig_client import stream_chat

PLACEHOLDERS = [
    "What APIs have you learned this session?",
    "Find a public chart API and graduate it.",
    "Show me what's in your toolbox.",
    "Browse to wikipedia.org and tell me the featured article today.",
    "How many graduated MCP servers exist right now?",
]


def _format_trace_event(delta: dict) -> str | None:
    """Extract tool call info from a delta and format as a trace line."""
    tool_calls = delta.get("tool_calls")
    if not tool_calls:
        return None

    lines = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args = func.get("arguments", "")
        if name:
            lines.append(f"→ **{name}**({args})")

    # Check for tool_result (our mock format includes this)
    return "\n".join(lines) if lines else None


def _format_tool_result(event: dict) -> str | None:
    """Extract tool result from mock event format."""
    delta = event.get("delta", {})
    tool_result = delta.get("tool_result")
    if not tool_result:
        return None
    content = tool_result.get("content", "")
    preview = content[:200] + "..." if len(content) > 200 else content
    return f"← {preview}"


async def _handle_chat(message: str, history: list[dict]):
    """Process user message: stream response to chatbot, trace to panel."""
    if not message or not message.strip():
        yield history, ""
        return

    history = history + [{"role": "user", "content": message}]
    assistant_text = ""
    trace_lines = []

    async for event in stream_chat(message, history[:-1]):
        delta = event.get("delta", {})

        # Accumulate content for chatbot
        content = delta.get("content", "")
        if content:
            assistant_text += content

        # Check for tool calls → trace panel
        trace_line = _format_trace_event(delta)
        if trace_line:
            trace_lines.append(trace_line)

        # Check for tool results → trace panel
        result_line = _format_tool_result(event)
        if result_line:
            trace_lines.append(result_line)

        # Build current state
        display_history = history.copy()
        if assistant_text:
            display_history.append({"role": "assistant", "content": assistant_text})

        trace_md = "\n\n".join(trace_lines) if trace_lines else "*Waiting for tool calls...*"
        yield display_history, trace_md

    # Final yield with complete message
    if assistant_text:
        final_history = history + [{"role": "assistant", "content": assistant_text}]
    else:
        final_history = history
    trace_md = "\n\n".join(trace_lines) if trace_lines else "*No tool calls in this response.*"
    yield final_history, trace_md


def build_free_play_tab():
    with gr.Tab("Free Play"):
        gr.Markdown("### Chat with the agent — watch tool calls stream in real time")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type="messages",
                    label="Agent Response",
                    height=500,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder=random.choice(PLACEHOLDERS),
                        label="Your prompt",
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Column(scale=2):
                trace_panel = gr.Markdown(
                    value="*Tool call trace will appear here...*",
                    label="MCP Tool Trace",
                    elem_classes=["trace-panel"],
                )

        # Wire events
        send_btn.click(
            fn=_handle_chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, trace_panel],
        ).then(fn=lambda: "", outputs=[msg_input])

        msg_input.submit(
            fn=_handle_chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, trace_panel],
        ).then(fn=lambda: "", outputs=[msg_input])
