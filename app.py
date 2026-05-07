"""
AMD MI300X Judge Terminal — 3-tab Gradio app for hackathon judges.
Tab 1: Free Play chat with live MCP tool-call trace
Tab 2: Canned Scenarios (3 pre-scripted demos)
Tab 3: System Status (VRAM, models, graduated repos, lifecycle)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (local dev); on HF Space, env vars come from settings
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

import gradio as gr

from tabs.free_play import build_free_play_tab
from tabs.scenarios import build_scenarios_tab
from tabs.status import build_status_tab
from backend.mock import is_mock_mode


CSS = """
.trace-panel {
    background: #1a1a2e;
    color: #e0e0e0;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 12px;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.85em;
    max-height: 500px;
    overflow-y: auto;
}
"""


def build_app() -> gr.Blocks:
    mode = "MOCK" if is_mock_mode() else "LIVE"
    with gr.Blocks(
        title="AMD MI300X Judge Terminal",
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="blue"),
        css=CSS,
    ) as demo:
        gr.Markdown(
            f"# AMD MI300X — Self-Improving AI Agent\n"
            f"*Judge terminal — [{mode} MODE]*"
        )

        build_free_play_tab()
        build_scenarios_tab()
        status_refresh_fn, status_outputs = build_status_tab()

        demo.load(fn=status_refresh_fn, inputs=[], outputs=status_outputs)

    return demo


demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
