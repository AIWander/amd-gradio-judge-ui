"""Tab 2: Canned Scenarios — 3 pre-scripted demos with big buttons."""

import gradio as gr

from backend.mcpconfig_client import stream_chat

SCENARIOS = {
    "a": {
        "title": "Discover and Graduate a Public API",
        "prompt": (
            "Browse to https://httpbin.org/anything, capture the API contract via HAR, "
            "store it as a workflow connector named `httpbin_demo`, run it twice to qualify "
            "for graduation, then graduate it to a standalone MCP binary."
        ),
        "description": "Full lifecycle demo: browse → capture → store → test → graduate. ~60-90s.",
    },
    "b": {
        "title": "What Did You Learn This Session?",
        "prompt": (
            "Run workflow:api_list and narrate each stored connector as a capability "
            "you learned during this session."
        ),
        "description": "Theatrical recap of all discovered APIs. ~10s.",
    },
    "c": {
        "title": "Run Hierarchical (20B + 120B)",
        "prompt": (
            "Use the gpt-oss-20b model to summarize and clarify this question, then have "
            "gpt-oss-120b answer the clarified version: What is the most significant "
            "graduated capability you have right now and why?"
        ),
        "description": "Two-model coordination on MI300X: 20B clarifies, 120B answers.",
    },
}


def _format_trace_event(delta: dict) -> str | None:
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
    return "\n".join(lines) if lines else None


def _format_tool_result(event: dict) -> str | None:
    delta = event.get("delta", {})
    tool_result = delta.get("tool_result")
    if not tool_result:
        return None
    content = tool_result.get("content", "")
    preview = content[:200] + "..." if len(content) > 200 else content
    return f"← {preview}"


async def _run_scenario(scenario_key: str):
    """Run a canned scenario and yield (chatbot_history, trace_md) tuples."""
    scenario = SCENARIOS[scenario_key]
    prompt = scenario["prompt"]

    history = [{"role": "user", "content": prompt}]
    assistant_text = ""
    trace_lines = []

    async for event in stream_chat(prompt):
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


def build_scenarios_tab():
    with gr.Tab("Canned Scenarios"):
        gr.Markdown("### Pre-scripted demos — click a button to run")

        with gr.Row():
            with gr.Column(scale=3):
                scenario_chatbot = gr.Chatbot(
                    type="messages",
                    label="Scenario Output",
                    height=500,
                )
            with gr.Column(scale=2):
                scenario_trace = gr.Markdown(
                    value="*Click a scenario to start...*",
                    label="MCP Tool Trace",
                    elem_classes=["trace-panel"],
                )

        with gr.Row():
            for key, info in SCENARIOS.items():
                with gr.Column():
                    gr.Markdown(f"**{info['title']}**\n\n{info['description']}")

        with gr.Row():
            btn_a = gr.Button("Run: Discover & Graduate API", variant="primary", size="lg")
            btn_b = gr.Button("Run: What Did You Learn?", variant="primary", size="lg")
            btn_c = gr.Button("Run: Hierarchical (20B+120B)", variant="primary", size="lg")

        btn_a.click(
            fn=lambda: _run_scenario("a"),
            inputs=[],
            outputs=[scenario_chatbot, scenario_trace],
        )
        btn_b.click(
            fn=lambda: _run_scenario("b"),
            inputs=[],
            outputs=[scenario_chatbot, scenario_trace],
        )
        btn_c.click(
            fn=lambda: _run_scenario("c"),
            inputs=[],
            outputs=[scenario_chatbot, scenario_trace],
        )
