---
title: AMD MI300X Multi-Agent Judge UI
emoji: "\U0001F3CE\uFE0F"
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: true
license: mit
---

# AMD MI300X Multi-Agent Judge UI

Competition judges submit a custom task and watch 4 AI agents race to complete it in parallel. Each agent streams its reasoning, tool calls, and browser screenshots in real time.

**Models:**
- gpt-oss-120b (AMD MI300X vLLM)
- Qwen3.6-35B-A3B (AMD MI300X vLLM)
- Qwen2.5-72B (HF Inference)
- Qwen3-32B (HF Inference)

**Tabs:**
- **Multi-Agent Bakeoff** — 4 parallel agent panels with SSE streaming from mcpconfig
- **Simple Chat** — Direct chat with gpt-oss-120b via OpenAI-compatible API
