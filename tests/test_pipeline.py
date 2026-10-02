"""Offline smoke tests: fetch -> calculate -> render with monkeypatched data."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module
import config_loader as cl
import datasource
from config_loader import ConfigSnapshot, ConfigStore, SnapshotCache
from indicators import compute_indicators, latest_metrics


def test_indicators_windows_come_from_snapshot(fixed_prices):
    snap20 = cl.load_config("nope.yaml", env={})
    snap50 = cl.load_config("nope.yaml", env={"SD_INDICATORS_SMA_WINDOWS": "50,100,200"})

    f20 = compute_indicators(fixed_prices, snap20)
    f50 = compute_indicators(fixed_prices, snap50)

    assert "SMA_20" in f20.columns
    assert "SMA_50" in f50.columns
    assert "SMA_20" not in f50.columns

    # Threshold moved 20 -> 50: SMA_50 under the new config must equal the
    # rolling-50 mean, not the old rolling-20 mean.
    expected = fixed_prices["Close"].rolling(50).mean()
    pd.testing.assert_series_equal(
        f50["SMA_50"], expected, check_names=False, atol=1e-10
    )
    aligned = pd.concat([f20["SMA_20"], f50["SMA_50"]], axis=1).dropna()
    assert not np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1], atol=1e-6)


def test_cache_consistency_before_and_after_clear(fixed_prices):
    snap = cl.load_config("nope.yaml", env={})
    cache = SnapshotCache(ttl_seconds=600)

    first = compute_indicators(fixed_prices, snap, cache=cache)
    assert len(cache) >= 1
    cached = compute_indicators(fixed_prices, snap, cache=cache)
    pd.testing.assert_frame_equal(first, cached)

    cache.clear()
    recomputed = compute_indicators(fixed_prices, snap, cache=cache)
    pd.testing.assert_frame_equal(first, recomputed)


def test_threshold_change_invalidates_fingerprinted_cache(fixed_prices):
    snap20 = cl.load_config("nope.yaml", env={})
    snap50 = cl.load_config("nope.yaml", env={"SD_INDICATORS_SMA_WINDOWS": "50,50,200"})
    cache = SnapshotCache(ttl_seconds=600)

    f20 = compute_indicators(fixed_prices, snap20, cache=cache)
    f50 = compute_indicators(fixed_prices, snap50, cache=cache)

    # Same physical cache, different snapshot -> different cached results.
    assert not np.isclose(f20["SMA_20"].iloc[-1], f50["SMA_50"].iloc[-1], atol=1e-6)


def test_cache_keys_are_per_symbol(fixed_prices):
    snap = cl.load_config("nope.yaml", env={})
    cache = SnapshotCache(ttl_seconds=600)

    other = fixed_prices.copy()
    other["Close"] = fixed_prices["Close"] * 3 + 7

    a = compute_indicators(fixed_prices, snap, cache=cache, cache_key="indicators:AAA")
    b = compute_indicators(other, snap, cache=cache, cache_key="indicators:BBB")

    assert a["SMA_20"].iloc[-1] != b["SMA_20"].iloc[-1]
    a_again = compute_indicators(fixed_prices, snap, cache=cache, cache_key="indicators:AAA")
    pd.testing.assert_frame_equal(a, a_again)


def test_latest_metrics_json_safe(fixed_prices):
    snap = cl.load_config("nope.yaml", env={})
    frame = compute_indicators(fixed_prices, snap)
    metrics = latest_metrics("TEST", frame, "fixture")
    assert metrics["symbol"] == "TEST"
    assert isinstance(metrics["close"], float)
    assert "SMA_20" in metrics
    import json
    json.dumps(metrics)  # no NaN leaks


def test_offline_degrades_to_fixture_without_error(fixed_prices, monkeypatch):
    # Simulate total network failure at the data-source boundary.
    def boom(symbol, period):
        raise ConnectionError("network is unreachable")

    monkeypatch.setattr(datasource, "fetch_yfinance", boom)

    snap = cl.load_config("nope.yaml", env={})
    frame, source = datasource.fetch_prices("AAPL", snap)
    assert source == "fixture"
    assert not frame.empty


def test_no_network_when_yfinance_disabled(fixed_prices, monkeypatch):
    called = {"n": 0}

    def should_not_run(symbol, period):
        called["n"] += 1
        raise AssertionError("yfinance must not be called when disabled")

    monkeypatch.setattr(datasource, "fetch_yfinance", should_not_run)
    snap = cl.load_config(
        "nope.yaml", env={"SD_SOURCES_YFINANCE_ENABLED": "false"}
    )
    frame, source = datasource.fetch_prices("AAPL", snap)
    assert source == "fixture"
    assert called["n"] == 0


def _client(tmp_path, monkeypatch, env, fixture_csv):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "tickers: [TEST, TSLA, NVDA]\ncache:\n  ttl_seconds: 60\n",
        encoding="utf-8",
    )
    store = ConfigStore(str(yaml_path), env=env)

    def fake_yf(symbol, period):
        raise ConnectionError("offline test")

    monkeypatch.setattr(datasource, "fetch_yfinance", fake_yf)
    # Point the fixture path at the committed CSV (override default path).
    return TestClient(app_module.build_app(store)), store


def test_full_pipeline_fetch_calculate_render(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch, {}, None)

    # Offline: pipeline must still complete, degraded to fixture, no 500s.
    page = client.get("/")
    assert page.status_code == 200
    body = page.text
    assert "Effective configuration" in body
    assert "Latest indicators" in body
    assert "TEST" in body

    metrics = client.get("/metrics/latest").json()
    assert metrics["fingerprint"] == store.snapshot().fingerprint
    assert metrics["metrics"][0]["source"] == "fixture"
    assert metrics["errors"] == []


def test_effective_config_endpoint_shape(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch,
                            {"SD_INDICATORS_RSI_WINDOW": "33"}, None)
    payload = client.get("/config/effective").json()

    assert payload["fingerprint"] == store.snapshot().fingerprint
    assert payload["config"]["indicators"]["rsi_window"] == 33
    assert payload["config"]["tickers"] == ["TEST", "TSLA", "NVDA"]
    assert payload["sources"]["indicators"]["rsi_window"] == "env"
    assert payload["sources"]["tickers"] == "yaml"
    assert {"config", "sources", "diagnostics", "fingerprint"} <= set(payload)


def test_home_and_metrics_read_same_snapshot(tmp_path, monkeypatch):
    client, store = _client(tmp_path, monkeypatch, {}, None)
    snap = store.snapshot()
    page = client.get("/").text
    assert snap.fingerprint in page
    metrics = client.get("/metrics/latest").json()
    assert metrics["fingerprint"] == snap.fingerprint


def test_hot_reload_changes_indicators_and_fingerprint(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "tickers: [TEST]\nindicators:\n  sma_windows: [20, 50, 200]\n",
        encoding="utf-8",
    )
    store = ConfigStore(str(yaml_path), env={})
    def offline(symbol, period):
        raise ConnectionError("offline")
    monkeypatch.setattr(datasource, "fetch_yfinance", offline)
    client = TestClient(app_module.build_app(store))

    before = client.get("/metrics/latest").json()
    assert any(k == "SMA_20" for k in before["metrics"][0])
    fp_before = before["fingerprint"]

    import time
    time.sleep(0.02)
    yaml_path.write_text(
        "tickers: [TEST]\nindicators:\n  sma_windows: [50, 100, 200]\n",
        encoding="utf-8",
    )

    after = client.get("/metrics/latest").json()
    assert after["fingerprint"] != fp_before
    assert "SMA_20" not in after["metrics"][0]
    assert "SMA_50" in after["metrics"][0]


def test_data_unavailable_when_all_sources_off(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "sources:\n"
        "  yfinance: {enabled: false}\n"
        "  local_fixture:\n"
        "    enabled: true\n"
        "    path: does_not_exist.csv\n",
        encoding="utf-8",
    )
    store = ConfigStore(str(yaml_path), env={})
    client = TestClient(app_module.build_app(store))

    page = client.get("/")
    assert page.status_code == 200  # page renders; error surfaced, not raised
    metrics = client.get("/metrics/latest").json()
    assert metrics["metrics"] == []
    assert metrics["errors"] and "no price data" in metrics["errors"][0]["error"]
