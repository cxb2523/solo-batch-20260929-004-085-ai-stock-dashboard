"""Offline tests: config-driven indicators + fingerprint-namespaced cache."""
import pandas as pd
import pytest

from cache import SnapshotCache
from config_loader import load_config
from indicators import calculate_indicators, latest_metrics
from pipeline import compute_ticker


def _snapshot(tmp_path, sma_short):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"indicator_windows:\n  sma_short: {sma_short}\n", encoding="utf-8"
    )
    return load_config(cfg, environ={})


def test_indicator_threshold_follows_config_20_to_50(tmp_path, fixed_ohlcv):
    snap20 = _snapshot(tmp_path, 20)
    snap50 = _snapshot(tmp_path, 50)

    frame20 = calculate_indicators(fixed_ohlcv, snapshot=snap20)
    frame50 = calculate_indicators(fixed_ohlcv, snapshot=snap50)

    # The short moving average column really moves from window 20 to window 50.
    assert "SMA_20" in frame20.columns
    assert "SMA_50" in frame50.columns
    metrics50 = latest_metrics(frame50, snapshot=snap50)
    assert metrics50["windows_used"]["sma_short"] == 50

    expected = fixed_ohlcv["Close"].rolling(50).mean().iloc[-1]
    assert metrics50["SMA_50"] == round(float(expected), 4)
    # SMA_50 at this exact bar must differ from a 20-window mean.
    expected20 = fixed_ohlcv["Close"].rolling(20).mean().iloc[-1]
    assert metrics50["SMA_50"] != round(float(expected20), 4)
    assert snap20.fingerprint != snap50.fingerprint


def test_rsi_threshold_change_changes_signal(tmp_path, fixed_ohlcv):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "indicator_windows:\n  rsi_oversold: 99\n", encoding="utf-8"
    )
    snap_oversold = load_config(cfg, environ={})
    frame = calculate_indicators(fixed_ohlcv, snapshot=snap_oversold)
    # With oversold threshold at 99, a normal RSI (~0-100) reads oversold.
    assert (frame["RSI_signal"] == "oversold").any()


def test_cache_keys_carry_fingerprint_and_clear_is_safe(
    tmp_path, monkeypatch, fixed_ohlcv
):
    import datasource

    snapshot = _snapshot(tmp_path, 20)

    def fake_fetch(symbol, snapshot=None, period="1y"):
        return datasource.PriceFetchResult(
            symbol=symbol, data=fixed_ohlcv.copy(), source="stub"
        )

    monkeypatch.setattr(datasource, "fetch_prices", fake_fetch)

    cache = SnapshotCache()
    first = compute_ticker("AAPL", snapshot, cache)
    cached_keys = cache.entries()
    assert len(cached_keys) == 1
    assert cached_keys[0].startswith(f"{snapshot.fingerprint}:ticker:AAPL")

    # Second call is served from cache but is byte-identical.
    second = compute_ticker("AAPL", snapshot, cache)
    assert first == second

    # After clearing, recomputation yields the same metrics (determinism).
    assert cache.clear() == 1
    third = compute_ticker("AAPL", snapshot, cache)
    assert third.metrics == first.metrics

    # A new generation (threshold 20 -> 50) gets its own key namespace.
    snapshot50 = _snapshot(tmp_path, 50)
    compute_ticker("AAPL", snapshot50, cache)
    key20 = f"{snapshot.fingerprint}:ticker:AAPL"
    key50 = f"{snapshot50.fingerprint}:ticker:AAPL"
    assert set(cache.entries()) == {key20, key50}


def test_indicators_frame_deterministic_across_clears(tmp_path, fixed_ohlcv):
    snapshot = _snapshot(tmp_path, 20)
    frame_a = calculate_indicators(fixed_ohlcv, snapshot=snapshot)
    frame_b = calculate_indicators(fixed_ohlcv, snapshot=snapshot)
    pd.testing.assert_frame_equal(frame_a, frame_b)
