"""Offline tests for the config layer: merge rules, provenance, hot reload."""
import os
import time

import pytest

import config_loader
from config_loader import ConfigStore, load_config


def write_yaml(path, text):
    path.write_text(text, encoding="utf-8")


def test_defaults_when_no_file_and_no_env(tmp_path):
    snapshot = load_config(tmp_path / "missing.yaml", environ={})
    assert snapshot.tickers == ("AAPL", "MSFT", "GOOG")
    assert snapshot.indicator_windows["sma_short"] == 20
    assert snapshot.cache_ttl_seconds == 300
    assert snapshot.datasources["yfinance_enabled"] is True


def test_yaml_overrides_defaults_scalars_and_lists(tmp_path):
    cfg = tmp_path / "config.yaml"
    write_yaml(
        cfg,
        """
tickers: [NVDA, AMD]
indicator_windows:
  sma_short: 25
cache:
  ttl_seconds: 60
datasources:
  yfinance_enabled: false
""",
    )
    snapshot = load_config(cfg, environ={})
    # List merge by index: positions 0,1 replaced, position 2 keeps default.
    assert list(snapshot.tickers) == ["NVDA", "AMD", "GOOG"]
    assert snapshot.indicator_windows["sma_short"] == 25
    assert snapshot.cache_ttl_seconds == 60
    assert snapshot.datasources["yfinance_enabled"] is False

    rows = {
        key: source
        for key, _value, source in _rows(snapshot)
    }
    assert rows["tickers[0]"] == "yaml"
    assert rows["tickers[2]"] == "default"
    assert rows["indicator_windows.sma_short"] == "yaml"
    assert rows["datasources.yfinance_enabled"] == "yaml"


def test_list_out_of_range_indices_dropped_with_diagnostic(tmp_path):
    cfg = tmp_path / "config.yaml"
    write_yaml(cfg, "tickers: [AAA, BBB, CCC, DDD, EEE]\n")
    snapshot = load_config(cfg, environ={})
    assert list(snapshot.tickers) == ["AAA", "BBB", "CCC"]
    assert any("out of range" in d and "2 override" in d for d in snapshot.diagnostics)


def test_env_beats_yaml_beats_default(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    write_yaml(cfg, "indicator_windows:\n  sma_short: 35\n")
    monkeypatch.setenv("INDICATOR_WINDOWS__SMA_SHORT", "45")
    snapshot = load_config(cfg)
    assert snapshot.indicator_windows["sma_short"] == 45


def test_env_list_merges_by_index(monkeypatch, tmp_path):
    monkeypatch.setenv("TICKERS", "tsla,amzn")
    snapshot = load_config(tmp_path / "nope.yaml")
    assert list(snapshot.tickers) == ["TSLA", "AMZN", "GOOG"]
    rows = {key: source for key, _v, source in _rows(snapshot)}
    assert rows["tickers[0]"] == "env"
    assert rows["tickers[2]"] == "default"


def test_unset_env_differs_from_empty_env(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    write_yaml(cfg, "indicator_windows:\n  sma_short: 33\n")

    # Unset: YAML value shines through.
    monkeypatch.delenv("INDICATOR_WINDOWS__SMA_SHORT", raising=False)
    unset = load_config(cfg)
    assert unset.indicator_windows["sma_short"] == 33

    # Empty string: explicit reset to built-in default (20), YAML ignored.
    monkeypatch.setenv("INDICATOR_WINDOWS__SMA_SHORT", "")
    empty = load_config(cfg)
    assert empty.indicator_windows["sma_short"] == 20
    assert any("empty string" in d for d in empty.diagnostics)


def test_invalid_scalar_falls_back_not_raises(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    write_yaml(cfg, "cache:\n  ttl_seconds: bananas\n")
    snapshot = load_config(cfg, environ={})
    assert snapshot.cache_ttl_seconds == 300
    assert any("invalid yaml value" in d for d in snapshot.diagnostics)

    monkeypatch.setenv("DATASOURCES__YFINANCE_ENABLED", "maybe")
    snapshot_env = load_config(cfg)
    assert snapshot_env.datasources["yfinance_enabled"] is True
    assert any("invalid env value" in d for d in snapshot_env.diagnostics)

    # Corrupt YAML itself must degrade, not raise.
    write_yaml(cfg, "cache:\n  ttl_seconds: [1, 2\n")
    broken = load_config(cfg, environ={})
    assert broken.cache_ttl_seconds == 300


def test_snapshot_is_frozen(tmp_path):
    snapshot = load_config(tmp_path / "x.yaml", environ={})
    with pytest.raises(TypeError):
        snapshot.values["tickers"] = ("X",)  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.indicator_windows["sma_short"] = 99  # type: ignore[index]


def test_hot_reload_on_mtime_change(tmp_path):
    cfg = tmp_path / "config.yaml"
    write_yaml(cfg, "indicator_windows:\n  sma_short: 22\n")
    # poll_interval=0 so every snapshot() call stats the file.
    store = ConfigStore(cfg, poll_interval=0, environ={})
    first = store.snapshot()
    assert first.indicator_windows["sma_short"] == 22

    time.sleep(0.02)
    write_yaml(cfg, "indicator_windows:\n  sma_short: 50\n")
    os.utime(cfg, None)

    second = store.snapshot()
    assert second is not first
    assert second.indicator_windows["sma_short"] == 50
    assert second.fingerprint != first.fingerprint


def _rows(snapshot):
    plain = snapshot.to_plain()
    values = plain["values"]
    sources = plain["sources"]
    rows = []

    def walk(value, source_node, prefix):
        if isinstance(value, dict):
            for key, sub in value.items():
                walk(sub, source_node.get(key, {}), f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, list):
            indexed = source_node.get("__list__", {})
            for index, item in enumerate(value):
                rows.append((f"{prefix}[{index}]", item, indexed.get(index, "default")))
        else:
            rows.append((prefix, value, source_node))

    walk(values, sources, "")
    return rows
