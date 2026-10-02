"""FastAPI dashboard service.

Run locally with::

    uvicorn app:app

Every request grabs one frozen snapshot via ``store.snapshot()`` and uses that
single object for config reads, calculation and rendering - so the page can
never mix a new "effective config" table with old indicator values.
"""

from __future__ import annotations

import html
from typing import Any

from fastapi import FastAPI

from config_loader import ConfigSnapshot, ConfigStore, SnapshotCache, to_plain
from datasource import DataUnavailable, fetch_prices
from indicators import compute_indicators, latest_metrics


def build_app(store: ConfigStore | None = None) -> FastAPI:
    app = FastAPI(title="AI Stock Dashboard")
    state_store = store or ConfigStore()
    metrics_cache = SnapshotCache()

    def current_snapshot() -> ConfigSnapshot:
        # Checks yaml mtime; swaps the whole snapshot atomically when changed.
        return state_store.snapshot()

    @app.get("/config/effective")
    def effective_config() -> dict[str, Any]:
        """Full effective configuration: values, precedence sources, fp, diags."""
        return current_snapshot().to_plain()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        snap = current_snapshot()
        return {"status": "ok", "fingerprint": snap.fingerprint}

    @app.get("/metrics/latest")
    def metrics_latest() -> dict[str, Any]:
        snap = current_snapshot()
        rows, errors = _collect_metrics(snap)
        return {
            "fingerprint": snap.fingerprint,
            "metrics": rows,
            "errors": errors,
        }

    @app.get("/")
    def home() -> str:
        snap = current_snapshot()
        rows, errors = _collect_metrics(snap)
        return render_home(snap, rows, errors)

    def _collect_metrics(snap: ConfigSnapshot) -> tuple[list[dict], list[dict]]:
        ttl = int(snap["cache"]["ttl_seconds"])
        max_entries = int(snap["cache"]["max_entries"])
        metrics_cache.ttl_seconds = ttl
        metrics_cache.max_entries = max_entries

        rows: list[dict] = []
        errors: list[dict] = []
        for symbol in snap["tickers"]:
            try:
                prices, source = fetch_prices(symbol, snap, cache=metrics_cache)
                frame = compute_indicators(
                    prices, snap, cache=metrics_cache, cache_key=f"indicators:{symbol}"
                )
                rows.append(latest_metrics(symbol, frame, source))
            except DataUnavailable as exc:
                errors.append({"symbol": symbol, "error": str(exc)})
        return rows, errors

    app.state.config_store = state_store
    app.state.metrics_cache = metrics_cache
    return app


app = build_app()


# ---------------------------------------------------------------------------
# Rendering (stdlib html only - no extra template dependency)
# ---------------------------------------------------------------------------


def _flatten(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(node, dict):
        out: list[tuple[str, Any]] = []
        for key in sorted(node):
            path = f"{prefix}.{key}" if prefix else key
            out.extend(_flatten(node[key], path))
        return out
    return [(prefix, node)]


def _render_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return html.escape(", ".join(str(v) for v in value))
    if isinstance(value, bool):
        return "true" if value else "false"
    return html.escape(str(value))


def render_home(snap: ConfigSnapshot, metric_rows: list[dict], errors: list[dict]) -> str:
    effective = to_plain(snap.config)
    sources = to_plain(snap.sources)

    config_rows_html = []
    for path, value in _flatten(effective):
        src = ""
        node: Any = sources
        for part in path.split("."):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if isinstance(node, str):
            src = node
        config_rows_html.append(
            "<tr><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                html.escape(path), _render_value(value), html.escape(src)
            )
        )

    columns = ["symbol", "source", "date", "close", "RSI", "MACD", "ATR", "volume_ratio"]
    sma_keys = sorted(
        {k for row in metric_rows for k in row if k.startswith("SMA_")},
        key=lambda k: int(k.split("_")[1]),
    )
    columns.extend(sma_keys)

    header_html = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body_rows = []
    for row in metric_rows:
        cells = "".join(
            "<td>{}</td>".format(html.escape("" if row.get(c) is None else str(row.get(c))))
            for c in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    metrics_body = "\n".join(body_rows) or "<tr><td colspan='{}'>No data</td></tr>".format(len(columns))

    error_html = ""
    if errors:
        items = "".join(
            "<li><code>{}</code>: {}</li>".format(html.escape(e["symbol"]), html.escape(e["error"]))
            for e in errors
        )
        error_html = f"<section><h2>Fetch errors</h2><ul>{items}</ul></section>"

    diag_html = ""
    if snap.diagnostics:
        items = "".join(
            "<li>[{}] <code>{}</code>: {}</li>".format(
                html.escape(d["code"]), html.escape(d["path"]), html.escape(d["message"])
            )
            for d in snap.diagnostics
        )
        diag_html = f"<details><summary>Config diagnostics ({len(snap.diagnostics)})</summary><ul>{items}</ul></details>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>AI Stock Dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1c1c1e; }}
    h1 {{ margin-bottom: 0; }}
    .meta {{ color: #666; margin: .25rem 0 1.5rem; }}
    table {{ border-collapse: collapse; margin: .5rem 0 2rem; min-width: 640px; }}
    th, td {{ border: 1px solid #d2d2d7; padding: 6px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #f5f5f7; }}
    code {{ font-size: 13px; }}
  </style>
</head>
<body>
  <h1>AI Stock Dashboard</h1>
  <p class="meta">
    snapshot fingerprint <code>{html.escape(snap.fingerprint)}</code>
    &middot; config <code>{html.escape(snap.yaml_path)}</code>
  </p>

  <section>
    <h2>Effective configuration</h2>
    <table>
      <thead><tr><th>Key</th><th>Effective value</th><th>Source</th></tr></thead>
      <tbody>
        {"".join(config_rows_html)}
      </tbody>
    </table>
  </section>

  <section>
    <h2>Latest indicators</h2>
    <table>
      <thead><tr>{header_html}</tr></thead>
      <tbody>
        {metrics_body}
      </tbody>
    </table>
  </section>

  {error_html}
  {diag_html}
</body>
</html>"""
