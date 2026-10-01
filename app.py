"""FastAPI dashboard: home page + effective-config JSON.

Run locally with::

    uvicorn app:app --reload
"""
from __future__ import annotations

import html
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

import config_loader
from cache import SnapshotCache
from pipeline import build_dashboard

app = FastAPI(title="AI Stock Dashboard")

# Single shared cache; keys are fingerprint-namespaced (see cache.py).
result_cache = SnapshotCache()


# ---------------------------------------------------------------------------
# Effective config flattening
# ---------------------------------------------------------------------------
def _flatten_config(snapshot) -> list:
    """Produce (key, value, source) rows for the effective-config table."""
    values = snapshot.to_plain()["values"]
    sources = snapshot.to_plain()["sources"]
    rows: list = []

    def walk(value, source_node, prefix):
        if isinstance(value, dict):
            for key, sub in value.items():
                walk(sub, source_node.get(key, {}), f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, (list, tuple)):
            index_sources = source_node.get("__list__", {}) if isinstance(source_node, dict) else {}
            for index, item in enumerate(value):
                rows.append((f"{prefix}[{index}]", item, index_sources.get(index, "default")))
        else:
            rows.append((prefix, value, source_node if isinstance(source_node, str) else "default"))

    walk(values, sources, "")
    return rows


@app.get("/config/effective")
def effective_config() -> Dict[str, Any]:
    """Full effective configuration: values, provenance, diagnostics."""
    snapshot = config_loader.get_snapshot()
    payload = snapshot.to_plain()
    payload["effective_rows"] = [
        {"key": key, "value": value, "source": source}
        for key, value, source in _flatten_config(snapshot)
    ]
    return JSONResponse(payload)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "fingerprint": config_loader.get_snapshot().fingerprint}


# ---------------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------------
def _render_page(dashboard: Dict[str, Any]) -> str:
    snapshot = dashboard["snapshot"]
    results = dashboard["results"]

    config_rows = "\n".join(
        "<tr><td>{key}</td><td><code>{value}</code></td>"
        "<td class='src-{source}'>{source}</td></tr>".format(
            key=html.escape(str(key)),
            value=html.escape(str(value)),
            source=html.escape(str(source)),
        )
        for key, value, source in _flatten_config(snapshot)
    )

    metric_rows = []
    for result in results:
        if result.error:
            metric_rows.append(
                "<tr><td>{symbol}</td><td colspan='7'>error: {error}</td></tr>".format(
                    symbol=html.escape(result.symbol),
                    error=html.escape(result.error),
                )
            )
            continue
        metrics = result.metrics
        windows = metrics.get("windows_used", {})
        sma_short_key = f"SMA_{windows.get('sma_short', 20)}"
        sma_long_key = f"SMA_{windows.get('sma_long', 50)}"
        metric_rows.append(
            "<tr><td>{symbol}</td><td>{source}</td><td>{date}</td>"
            "<td>{close}</td><td>{change}</td><td>{rsi}</td>"
            "<td>{sma_short}</td><td>{sma_long}</td><td>{signal}</td></tr>".format(
                symbol=html.escape(result.symbol),
                source=html.escape(result.source),
                date=html.escape(str(metrics.get("date", ""))),
                close=html.escape(str(metrics.get("close", ""))),
                change=html.escape(str(metrics.get("change_pct", ""))),
                rsi=html.escape(str(metrics.get("RSI", ""))),
                sma_short=html.escape(str(metrics.get(sma_short_key, ""))),
                sma_long=html.escape(str(metrics.get(sma_long_key, ""))),
                signal=html.escape(str(metrics.get("RSI_signal", ""))),
            )
        )

    diagnostics = "".join(
        f"<li>{html.escape(message)}</li>" for message in snapshot.diagnostics
    ) or "<li>none</li>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AI Stock Dashboard</title>
  <style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif;
           margin: 2rem; background: #0f1420; color: #e6e9ef; }}
    h1 {{ font-size: 1.4rem; }}
    h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
    table {{ border-collapse: collapse; margin-top: .5rem; }}
    th, td {{ border: 1px solid #2b3344; padding: .35rem .7rem;
              text-align: left; font-size: .9rem; }}
    th {{ background: #1a2130; }}
    code {{ color: #9ecbff; }}
    .src-env {{ color: #7ee787; font-weight: 600; }}
    .src-yaml {{ color: #d2a8ff; }}
    .src-default {{ color: #8b949e; }}
    .meta {{ color: #8b949e; font-size: .85rem; }}
  </style>
</head>
<body>
  <h1>AI Stock Dashboard</h1>
  <p class="meta">snapshot fingerprint <code>{html.escape(snapshot.fingerprint)}</code>
     &middot; config {html.escape(str(snapshot.config_path))}</p>

  <h2>Effective configuration</h2>
  <table>
    <thead><tr><th>Key</th><th>Value</th><th>Source</th></tr></thead>
    <tbody>
      {config_rows}
    </tbody>
  </table>

  <h2>Latest indicators</h2>
  <table>
    <thead><tr><th>Ticker</th><th>Source</th><th>Date</th><th>Close</th>
      <th>Change %</th><th>RSI</th><th>SMA short</th><th>SMA long</th>
      <th>RSI signal</th></tr></thead>
    <tbody>
      {"".join(metric_rows)}
    </tbody>
  </table>

  <h2>Config diagnostics</h2>
  <ul>{diagnostics}</ul>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # One snapshot for the whole render: this table and these indicators are
    # guaranteed to come from the same generation.
    dashboard = build_dashboard(cache=result_cache)
    return HTMLResponse(_render_page(dashboard))


@app.post("/cache/clear")
def clear_cache() -> Dict[str, int]:
    return {"cleared": result_cache.clear()}
