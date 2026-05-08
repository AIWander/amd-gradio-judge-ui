"""Tab 1: Free Play — chat + live MCP tool-call trace."""

import asyncio
import random

import gradio as gr

from backend.mcpconfig_client import stream_chat, stream_chat_coordinated

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


def _badge_md(route: str, total_ms: int, total_tokens: int) -> str:
    """Build a routing badge as inline-styled Markdown."""
    secs = total_ms / 1000
    if route == "120b":
        return (
            f'<span style="background:#7c3aed;color:#fff;padding:2px 8px;'
            f'border-radius:12px;font-size:0.8em;font-weight:600">'
            f'🧠 120B • {secs:.1f}s • {total_tokens}tok</span>'
        )
    return (
        f'<span style="background:#0891b2;color:#fff;padding:2px 8px;'
        f'border-radius:12px;font-size:0.8em;font-weight:600">'
        f'⚡ 20B • {secs:.1f}s • {total_tokens}tok</span>'
    )


async def _handle_chat(message: str, history: list[dict], coord_mode: str):
    """Process user message: stream response to chatbot, trace to panel."""
    if not message or not message.strip():
        yield history, ""
        return

    use_coordinator = coord_mode == "Coordinator (β)"
    history = history + [{"role": "user", "content": message}]
    assistant_text = ""
    trace_lines = []
    route_info: dict | None = None

    if use_coordinator:
        async for event in stream_chat_coordinated(message, history[:-1]):
            etype = event.get("type", "")

            if etype == "meta":
                route_info = event
                badge_label = "🧠 120B" if event["route"] == "120b" else "⚡ 20B"
                trace_lines.append(
                    f"**Router:** {event['reason']} → {badge_label} "
                    f"(classifier {event['classifier_ms']}ms)"
                )
                display_history = history.copy()
                trace_md = "\n\n".join(trace_lines)
                yield display_history, trace_md

            elif etype == "delta":
                content = event.get("content", "")
                if content:
                    assistant_text += content
                display_history = history.copy()
                if assistant_text:
                    display_history.append({"role": "assistant", "content": assistant_text})
                trace_md = "\n\n".join(trace_lines)
                yield display_history, trace_md

            elif etype == "done":
                badge = _badge_md(event["route"], event["total_ms"], event["total_tokens"])
                if assistant_text:
                    final_content = f"{badge}\n\n{assistant_text}"
                    final_history = history + [{"role": "assistant", "content": final_content}]
                else:
                    final_history = history
                trace_lines.append(
                    f"**Done:** {event['total_ms']}ms total, "
                    f"{event['total_tokens']} tokens, model={event['model']}"
                )
                trace_md = "\n\n".join(trace_lines)
                yield final_history, trace_md
    else:
        async for event in stream_chat(message, history[:-1]):
            delta = event.get("delta", {})

            content = delta.get("content", "")
            if content:
                assistant_text += content

            trace_line = _format_trace_event(delta)
            if trace_line:
                trace_lines.append(trace_line)

            result_line = _format_tool_result(event)
            if result_line:
                trace_lines.append(result_line)

            display_history = history.copy()
            if assistant_text:
                display_history.append({"role": "assistant", "content": assistant_text})

            trace_md = "\n\n".join(trace_lines) if trace_lines else "*Waiting for tool calls...*"
            yield display_history, trace_md

        if assistant_text:
            final_history = history + [{"role": "assistant", "content": assistant_text}]
        else:
            final_history = history
        trace_md = "\n\n".join(trace_lines) if trace_lines else "*No tool calls in this response.*"
        yield final_history, trace_md


def build_free_play_tab():
    with gr.Tab("Free Play"):
        gr.Markdown("### Chat with the agent — watch tool calls stream in real time")
        coord_mode = gr.Radio(
            ["Off", "Coordinator (β)"],
            value="Off",
            label="Coordination mode",
        )
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
            inputs=[msg_input, chatbot, coord_mode],
            outputs=[chatbot, trace_panel],
        ).then(fn=lambda: "", outputs=[msg_input])

        msg_input.submit(
            fn=_handle_chat,
            inputs=[msg_input, chatbot, coord_mode],
            outputs=[chatbot, trace_panel],
        ).then(fn=lambda: "", outputs=[msg_input])
