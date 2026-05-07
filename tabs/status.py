"""Tab 3: System Status — VRAM, models, graduated repos, lifecycle counts."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from datetime import datetime, timezone

import gradio as gr

from backend.vllm_client import get_vllm_metrics
from backend.github_client import get_graduated_repos
from backend.mock import is_mock_mode, mock_lifecycle

import httpx
import os


def _make_vram_chart(metrics: dict):
    """Create a matplotlib bar chart of VRAM usage per model."""
    models = metrics.get("models", [])
    if not models:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No model data", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return fig

    names = [m["name"] for m in models]
    used_gib = [m["gpu_memory_used_bytes"] / (1024**3) for m in models]
    total_gib = metrics.get("total_vram_bytes", 0) / (1024**3)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(names, used_gib, color=colors[:len(names)], width=0.5)

    for bar, val in zip(bars, used_gib):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f} GiB", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("VRAM Used (GiB)")
    ax.set_title(f"GPU Memory Usage — {total_gib:.0f} GiB Total (AMD MI300X)")
    ax.set_ylim(0, max(used_gib) * 1.25 if used_gib else 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _make_models_table(metrics: dict) -> list[list]:
    """Build rows for the models dataframe."""
    rows = []
    for m in metrics.get("models", []):
        rows.append([
            m["name"],
            m["port"],
            f"{m['gpu_utilization_pct']:.1f}%",
            f"{m['params_b']}B",
            m.get("last_activity_ago", "—"),
        ])
    return rows


def _make_repos_table(repos: list[dict]) -> list[list]:
    """Build rows for graduated repos dataframe."""
    rows = []
    for r in repos:
        rows.append([
            r["name"],
            r.get("stargazers_count", 0),
            r.get("pushed_at", "")[:10],
            f"pip install {r['name']}",
        ])
    return rows


def _make_recent_graduations(repos: list[dict]) -> list[list]:
    """Last 5 repos sorted by creation date."""
    sorted_repos = sorted(repos, key=lambda r: r.get("created_at", ""), reverse=True)
    rows = []
    for r in sorted_repos[:5]:
        rows.append([
            r["name"],
            r.get("created_at", "")[:10],
            r.get("html_url", ""),
        ])
    return rows


async def _get_lifecycle() -> dict:
    """Fetch lifecycle counts from mcpconfig or fall back to mock."""
    if is_mock_mode():
        return mock_lifecycle()

    mcpconfig_url = os.environ.get("MCPCONFIG_URL", "").strip()
    if not mcpconfig_url:
        return mock_lifecycle()

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{mcpconfig_url}/_workflow/lifecycle")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return mock_lifecycle()


def _format_lifecycle(data: dict) -> str:
    l1 = data.get("layer_1", "—")
    l2 = data.get("layer_2", "—")
    l3 = data.get("layer_3", "—")
    return (
        f"<div style='display:flex;gap:40px;justify-content:center;padding:10px;'>"
        f"<div style='text-align:center;'><div style='font-size:2.5em;font-weight:bold;'>{l1}</div>"
        f"<div>Layer 1<br/>Discovered</div></div>"
        f"<div style='text-align:center;'><div style='font-size:2.5em;font-weight:bold;'>{l2}</div>"
        f"<div>Layer 2<br/>Validated</div></div>"
        f"<div style='text-align:center;'><div style='font-size:2.5em;font-weight:bold;'>{l3}</div>"
        f"<div>Layer 3<br/>Graduated</div></div>"
        f"</div>"
    )


def _format_uptime(metrics: dict) -> str:
    boot = metrics.get("boot_timestamp")
    if not boot:
        # Fall back to oldest uptime_seconds
        models = metrics.get("models", [])
        if models:
            max_uptime = max(m.get("uptime_seconds", 0) for m in models)
            hours = int(max_uptime // 3600)
            mins = int((max_uptime % 3600) // 60)
            return f"**Uptime:** {hours}h {mins}m (oldest model load)"
        return "**Uptime:** —"

    try:
        boot_dt = datetime.fromisoformat(boot.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        elapsed = now - boot_dt
        hours = int(elapsed.total_seconds() // 3600)
        mins = int((elapsed.total_seconds() % 3600) // 60)
        return f"**Uptime:** {hours}h {mins}m (since {boot_dt.strftime('%Y-%m-%d %H:%M UTC')})"
    except Exception:
        return "**Uptime:** —"


async def _refresh_status():
    """Fetch all status data and return component updates."""
    metrics = await get_vllm_metrics()
    repos = await get_graduated_repos()
    lifecycle = await _get_lifecycle()

    vram_chart = _make_vram_chart(metrics)
    models_table = _make_models_table(metrics)
    repos_table = _make_repos_table(repos)
    recent_table = _make_recent_graduations(repos)
    lifecycle_md = _format_lifecycle(lifecycle)
    uptime_md = _format_uptime(metrics)

    return vram_chart, models_table, repos_table, lifecycle_md, uptime_md, recent_table


def build_status_tab():
    with gr.Tab("System Status"):
        gr.Markdown("### Live system metrics — refreshes every 5 seconds")

        with gr.Row():
            with gr.Column(scale=3):
                vram_plot = gr.Plot(label="VRAM Usage")
            with gr.Column(scale=2):
                lifecycle_md = gr.HTML(value="<em>Loading...</em>", label="Lifecycle Counts")
                uptime_md = gr.Markdown(value="*Loading...*")

        with gr.Row():
            models_df = gr.Dataframe(
                headers=["Model", "Port", "GPU Mem %", "Params", "Last Activity"],
                label="Loaded Models",
                interactive=False,
            )

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Graduated MCP Repos")
                repos_df = gr.Dataframe(
                    headers=["Repo", "Stars", "Last Commit", "Install"],
                    label="Graduated Repos",
                    interactive=False,
                )
            with gr.Column():
                gr.Markdown("#### Recent Graduations")
                recent_df = gr.Dataframe(
                    headers=["Repo", "Created", "URL"],
                    label="Recent Graduations",
                    interactive=False,
                )

        all_outputs = [vram_plot, models_df, repos_df, lifecycle_md, uptime_md, recent_df]

        timer = gr.Timer(value=5)
        timer.tick(fn=_refresh_status, inputs=[], outputs=all_outputs)

    return _refresh_status, all_outputs
