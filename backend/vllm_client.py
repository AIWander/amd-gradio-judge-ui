"""Client for vLLM /metrics (Prometheus format) endpoints."""

import os
import re

import httpx

from .mock import is_mock_mode, mock_vllm_metrics

PORTS = {
    "gpt-oss-120b": "VLLM_BASE_URL",
    "qwen3.6-35b-A3B": "VLLM_BASE_URL_8001",
    "gpt-oss-20b": "VLLM_BASE_URL_8002",
}

_METRIC_RE = re.compile(r'^(\w+)\{?[^}]*\}?\s+([\d.eE+\-]+)', re.MULTILINE)


def _parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition format into {metric_name: value}."""
    result = {}
    for match in _METRIC_RE.finditer(text):
        name, val = match.group(1), match.group(2)
        try:
            result[name] = float(val)
        except ValueError:
            pass
    return result


async def get_vllm_metrics() -> dict:
    """
    Returns system_metrics.json-shaped dict with model info from each vLLM instance.
    Falls back to mock data if MOCK_MODE or on connection failure.
    """
    if is_mock_mode():
        return mock_vllm_metrics()

    models = []
    for model_name, env_var in PORTS.items():
        base_url = os.environ.get(env_var, "").strip()
        if not base_url:
            continue
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{base_url}/metrics")
                resp.raise_for_status()
                metrics = _parse_prometheus(resp.text)

                port = int(base_url.rstrip("/").split(":")[-1]) if ":" in base_url else 0
                models.append({
                    "name": model_name,
                    "port": port,
                    "gpu_memory_used_bytes": metrics.get("vllm:gpu_memory_used_bytes", 0),
                    "gpu_memory_total_bytes": metrics.get("vllm:gpu_memory_total_bytes", 0),
                    "gpu_utilization_pct": round(
                        metrics.get("vllm:gpu_memory_used_bytes", 0)
                        / max(metrics.get("vllm:gpu_memory_total_bytes", 1), 1) * 100, 1
                    ),
                    "params_b": {"gpt-oss-120b": 120, "qwen3.6-35b-A3B": 35, "gpt-oss-20b": 20}.get(model_name, 0),
                    "last_activity_ago": "live",
                    "uptime_seconds": metrics.get("vllm:uptime_seconds", 0),
                })
        except Exception:
            continue

    if not models:
        return mock_vllm_metrics()

    total_vram = max(m["gpu_memory_total_bytes"] for m in models) if models else 0
    boot_ts = None
    return {
        "models": models,
        "total_vram_bytes": total_vram,
        "boot_timestamp": boot_ts,
    }
