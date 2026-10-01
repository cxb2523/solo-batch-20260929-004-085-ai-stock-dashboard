"""Offline smoke tests: fixture fallback and fetch -> compute -> render."""
import pytest
from fastapi.testclient import TestClient

import app
import config_loader
import datasource
from config_loader import load_config


@pytest.fixture
def client():
    app.result_cache.clear()
    with TestClient(app.app) as test_client:
        yield test_client
    app.result_cache.clear()


def _patched_snapshot(tmp_path, monkeypatch, sma_short=20):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"indicator_windows:\n  sma_short: {sma_short}\n", encoding="utf-8"
    )
    snapshot = load_config(cfg, environ={})

    def current():
        # Reload on demand so the mtime hot-reload test sees config edits.
        return load_config(cfg, environ={})

    monkeypatch.setattr(config_loader, "get_snapshot", current)
    return snapshot


def test_offline_fallback_to_fixture_when_network_fails(monkeypatch):
    """Broken network (or missing yfinance) degrades to the local fixture."""

    def boom(symbol, period="1y"):
        raise ConnectionError("simulated network outage")

    monkeypatch.setattr(datasource, "_fetch_yfinance", boom)
    snapshot = load_config(None, environ={})

    result = datasource.fetch_prices("AAPL", snapshot=snapshot)
    assert result.data is not None
    assert result.source == "fixture"
    assert not result.data.empty


def test_offline_fallback_when_yfinance_not_installed(monkeypatch):
    # yfinance is genuinely absent from the offline test environment; even an
    # explicit import failure must degrade, never raise.
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("No module named 'yfinance'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    snapshot = load_config(None, environ={})
    result = datasource.fetch_prices("AAPL", snapshot=snapshot)
    assert result.source == "fixture"
    assert result.data is not None


def test_no_fixture_no_live_returns_error_without_raising(monkeypatch):
    snapshot = load_config(
        None,
        environ={
            "DATASOURCES__YFINANCE_ENABLED": "false",
            "DATASOURCES__LOCAL_FIXTURE_ENABLED": "false",
        },
    )
    result = datasource.fetch_prices("AAPL", snapshot=snapshot)
    assert result.data is None
    assert result.source == "none"
    assert result.error


def test_home_renders_config_and_indicators_from_stubbed_df(
    tmp_path, monkeypatch, fixed_ohlcv, client
):
    snapshot = _patched_snapshot(tmp_path, monkeypatch, sma_short=20)

    def fake_fetch(symbol, snapshot=None, period="1y"):
        return datasource.PriceFetchResult(
            symbol=symbol, data=fixed_ohlcv.copy(), source="stub"
        )

    monkeypatch.setattr(datasource, "fetch_prices", fake_fetch)

    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "Effective configuration" in body
    assert "Latest indicators" in body
    assert snapshot.fingerprint in body
    for ticker in snapshot.tickers:
        assert ticker in body
    assert "SMA short" in body


def test_config_effective_matches_home_snapshot(
    tmp_path, monkeypatch, fixed_ohlcv, client
):
    snapshot = _patched_snapshot(tmp_path, monkeypatch, sma_short=35)

    def fake_fetch(symbol, snapshot=None, period="1y"):
        return datasource.PriceFetchResult(
            symbol=symbol, data=fixed_ohlcv.copy(), source="stub"
        )

    monkeypatch.setattr(datasource, "fetch_prices", fake_fetch)

    page = client.get("/")
    effective = client.get("/config/effective").json()

    assert effective["fingerprint"] == snapshot.fingerprint
    assert effective["values"]["indicator_windows"]["sma_short"] == 35
    assert effective["fingerprint"] in page.text
    assert effective["config_path"] == str(tmp_path / "config.yaml")
    keys = {row["key"]: row for row in effective["effective_rows"]}
    assert keys["indicator_windows.sma_short"]["source"] == "yaml"
    assert keys["tickers[0]"]["source"] == "default"


def test_threshold_change_appears_on_home(tmp_path, monkeypatch, fixed_ohlcv, client):
    _patched_snapshot(tmp_path, monkeypatch, sma_short=20)

    def fake_fetch(symbol, snapshot=None, period="1y"):
        return datasource.PriceFetchResult(
            symbol=symbol, data=fixed_ohlcv.copy(), source="stub"
        )

    monkeypatch.setattr(datasource, "fetch_prices", fake_fetch)
    page20 = client.get("/").text

    cfg = tmp_path / "config.yaml"
    import os
    import time

    time.sleep(0.02)
    cfg.write_text(
        "indicator_windows:\n  sma_short: 50\n", encoding="utf-8"
    )
    os.utime(cfg, None)
    page50 = client.get("/").text

    assert page20 != page50


def test_offline_page_still_renders_via_fixture(
    tmp_path, monkeypatch, client
):
    _patched_snapshot(tmp_path, monkeypatch)

    def boom(symbol, snapshot=None, period="1y"):
        raise ConnectionError("offline")

    monkeypatch.setattr(datasource, "_fetch_yfinance", boom)
    response = client.get("/")
    assert response.status_code == 200
    assert "fixture" in response.text
