---
title: AMD MI300X Judge Terminal
emoji: "\U0001F3CE\uFE0F"
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "5.35.0"
app_file: app.py
pinned: true
license: mit
---

# AMD MI300X — Self-Improving AI Agent (Judge Terminal)

Three-tab Gradio interface for AMD/Lablab hackathon judges to experience the CPC self-improving agent system without needing to drive Claude Desktop.

## Local Dev Setup

```bash
cd C:\github\amd-gradio-judge-ui
pip install -r requirements.txt
python app.py
# Opens at http://localhost:7860 in MOCK mode (no backend needed)
```

## Env Vars

| Variable | Description | Default |
|----------|-------------|---------|
| `MCPCONFIG_URL` | mcpconfig server base URL (e.g. `http://<ip>:8003/v1`) | *(empty)* |
| `VLLM_BASE_URL` | vLLM 120b instance (e.g. `http://<ip>:8001`) | *(empty)* |
| `VLLM_BASE_URL_8002` | vLLM gpt-oss-20b instance | *(empty)* |
| `MOCK_MODE` | `true` = use scripted fixtures, `false` = real backend | `true` |
| `GITHUB_ORG` | GitHub org for graduated repos lookup | `AIWander` |
| `GITHUB_TOKEN` | Optional GitHub PAT (raises API rate limit) | *(empty)* |

When `MOCK_MODE=true` (default) or `MCPCONFIG_URL` is empty, all tabs use scripted fixtures from `mocks/` with realistic streaming timing.

## Tabs

1. **Free Play** — Chat with the agent. Left panel shows conversation, right panel shows live MCP tool-call trace.
2. **Canned Scenarios** — Three big buttons for pre-scripted demos:
   - *Discover & Graduate API* — full lifecycle from browse to binary
   - *What Did You Learn?* — theatrical recap of session capabilities
   - *Hierarchical (20B + 120B)* — two-model coordination on MI300X
3. **System Status** — VRAM bar chart, loaded models table, graduated repos, lifecycle counts. Auto-refreshes every 5s.

## Deploy to HuggingFace Space

```bash
# Add HF Space remote
git remote add hf https://huggingface.co/spaces/aiwanderai-amd-gradio-judge-ui

# Push
git push hf main
```

Then set env vars in the Space's Settings > Repository secrets:
- `MCPCONFIG_URL`
- `VLLM_BASE_URL`
- `VLLM_BASE_URL_8002`
- `MOCK_MODE=false`
- `GITHUB_TOKEN` (optional)

## Architecture

```
User (browser)
    |
    v
┌──────────────────────────────────┐
│  Gradio App (app.py)             │
│  ├── Tab 1: Free Play            │
│  ├── Tab 2: Canned Scenarios     │
│  └── Tab 3: System Status        │
├──────────────────────────────────┤
│  Backend Layer                   │
│  ├── mcpconfig_client.py ──────────► mcpconfig SSE (OpenAI-compatible)
│  ├── vllm_client.py ──────────────► vLLM /metrics (Prometheus)
│  ├── github_client.py ────────────► GitHub API (org repos)
│  └── mock.py (MOCK_MODE=true) ───► mocks/*.json fixtures
└──────────────────────────────────┘
```

## File Layout

```
amd-gradio-judge-ui/
├── app.py                  # Entry point
├── tabs/
│   ├── free_play.py        # Tab 1: Chat + trace
│   ├── scenarios.py        # Tab 2: 3 canned demos
│   └── status.py           # Tab 3: VRAM, models, repos
├── backend/
│   ├── mcpconfig_client.py # SSE streaming client
│   ├── vllm_client.py      # Prometheus metrics parser
│   ├── github_client.py    # Graduated repos fetcher
│   └── mock.py             # Mock dispatcher + fixtures loader
├── mocks/                  # JSON fixtures for offline mode
├── requirements.txt
├── .env.example
└── .gitignore
```
