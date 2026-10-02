"""Offline tests for the env > yaml > default merge rules."""

import time

import pytest

import config_loader as cl


def write_yaml(tmp_path, text, name="config.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_defaults_when_no_yaml_and_no_env(tmp_path):
    snap = cl.load_config(str(tmp_path / "missing.yaml"), env={})
    assert snap["tickers"] == ("AAPL", "MSFT", "GOOGL")
    assert snap["indicators"]["rsi_window"] == 14
    assert snap["indicators"]["sma_windows"] == (20, 50, 200)
    assert snap.sources["indicators"]["rsi_window"] == "default"
    assert snap.diagnostics[0]["code"] == "yaml_missing"


def test_env_overrides_yaml_overrides_default(tmp_path):
    path = write_yaml(tmp_path, "indicators:\n  rsi_window: 7\n")
    snap = cl.load_config(path, env={"SD_INDICATORS_RSI_WINDOW": "21"})
    assert snap["indicators"]["rsi_window"] == 21
    # yaml-only sibling still shows yaml source
    assert snap.sources["indicators"]["rsi_window"] == "env"

    snap_yaml = cl.load_config(path, env={})
    assert snap_yaml["indicators"]["rsi_window"] == 7
    assert snap_yaml.sources["indicators"]["rsi_window"] == "yaml"


def test_scalar_replaced_wholesale(tmp_path):
    path = write_yaml(tmp_path, "cache:\n  ttl_seconds: 60\n")
    snap = cl.load_config(path, env={"SD_CACHE_TTL_SECONDS": "90"})
    assert snap["cache"]["ttl_seconds"] == 90
    assert snap["cache"]["max_entries"] == 128  # untouched sibling keeps default


def test_list_merged_index_by_index(tmp_path):
    # yaml overrides index 0 and 1 only; index 2 keeps the default (200)
    path = write_yaml(tmp_path, "indicators:\n  sma_windows: [30, 40]\n")
    snap = cl.load_config(path, env={})
    assert snap["indicators"]["sma_windows"] == (30, 40, 200)


def test_list_out_of_range_index_dropped_with_diagnostic(tmp_path):
    path = write_yaml(tmp_path, "indicators:\n  sma_windows: [30, 40, 200, 500]\n")
    snap = cl.load_config(path, env={})
    assert snap["indicators"]["sma_windows"] == (30, 40, 200)
    codes = [d["code"] for d in snap.diagnostics]
    assert "yaml_index_out_of_range" in codes
    diag = next(d for d in snap.diagnostics if d["code"] == "yaml_index_out_of_range")
    assert "3" in diag["message"]

    # env out-of-range is dropped too
    snap_env = cl.load_config(path, env={"SD_INDICATORS_SMA_WINDOWS": "10,20,30,40,50"})
    assert snap_env["indicators"]["sma_windows"] == (10, 20, 30)
    assert any(d["code"] == "env_index_out_of_range" for d in snap_env.diagnostics)


def test_invalid_values_fall_back_without_raising(tmp_path):
    path = write_yaml(
        tmp_path,
        "indicators:\n  rsi_window: -5\n  sma_windows: [abc, 9999]\n"
        "cache:\n  ttl_seconds: 'forever'\n"
        "data:\n  period: 42\n",
    )
    snap = cl.load_config(path, env={"SD_SERVER_PORT": "banana"})
    assert snap["indicators"]["rsi_window"] == 14
    assert snap["indicators"]["sma_windows"] == (20, 50, 200)
    assert snap["cache"]["ttl_seconds"] == 300
    assert snap["data"]["period"] == "1y"
    assert snap["server"]["port"] == 8000
    codes = {d["code"] for d in snap.diagnostics}
    assert "invalid_yaml_value" in codes
    assert "invalid_env_value" in codes


def test_broken_yaml_never_raises(tmp_path):
    path = write_yaml(tmp_path, "tickers: [AAPL\n  - broken\n: :\n")
    snap = cl.load_config(path, env={})
    assert snap["tickers"] == ("AAPL", "MSFT", "GOOGL")
    assert any(d["code"] == "yaml_unreadable" for d in snap.diagnostics)


def test_unset_env_differs_from_empty_string(tmp_path):
    path = write_yaml(tmp_path, "tickers: [AAA, BBB, CCC]\n")

    # unset -> yaml wins
    unset = cl.load_config(path, env={})
    assert unset["tickers"] == ("AAA", "BBB", "CCC")

    # explicitly empty list env -> list cleared, source env
    emptied = cl.load_config(path, env={"SD_TICKERS": ""})
    assert cl.to_plain(emptied["tickers"]) == []
    assert emptied["tickers"] == ()
    assert emptied.sources["tickers"] == "env"

    # explicitly empty scalar env -> invalid, lower layer kept + diagnostic
    scalar_empty = cl.load_config(path, env={"SD_INDICATORS_RSI_WINDOW": ""})
    assert scalar_empty["indicators"]["rsi_window"] == 14
    assert any(d["code"] == "env_explicit_empty" for d in scalar_empty.diagnostics)


def test_bool_and_float_env_coercion(tmp_path):
    path = str(tmp_path / "config.yaml")
    snap = cl.load_config(
        path,
        env={"SD_SOURCES_YFINANCE_ENABLED": "no",
             "SD_INDICATORS_BOLLINGER_STD": "2.5"},
    )
    assert snap["sources"]["yfinance"]["enabled"] is False
    assert snap["indicators"]["bollinger"]["std"] == 2.5


def test_snapshot_is_frozen(tmp_path):
    snap = cl.load_config(str(tmp_path / "x.yaml"), env={})
    with pytest.raises(TypeError):
        snap.config["tickers"] = ("X",)
    with pytest.raises(TypeError):
        snap.config["indicators"]["rsi_window"] = 1


def test_fingerprint_changes_with_effective_config(tmp_path):
    path = str(tmp_path / "config.yaml")
    s1 = cl.load_config(path, env={})
    s2 = cl.load_config(path, env={"SD_INDICATORS_RSI_WINDOW": "33"})
    assert s1.fingerprint != s2.fingerprint


def test_store_hot_reloads_on_mtime_change(tmp_path):
    path = write_yaml(tmp_path, "indicators:\n  rsi_window: 7\n")
    store = cl.ConfigStore(path, env={})
    first = store.snapshot()
    assert first["indicators"]["rsi_window"] == 7

    time.sleep(0.02)
    write_yaml(tmp_path, "indicators:\n  rsi_window: 99\n")
    second = store.snapshot()
    assert second["indicators"]["rsi_window"] == 99
    assert second.fingerprint != first.fingerprint

    # no mtime change -> same object (atomic, stable)
    assert store.snapshot() is second


def test_cache_fingerprint_partitioning():
    s1 = cl.load_config("nope.yaml", env={})
    s2 = cl.load_config("nope.yaml", env={"SD_INDICATORS_RSI_WINDOW": "33"})
    cache = cl.SnapshotCache(ttl_seconds=60)
    cache.set(s1, "k", "v1")
    cache.set(s2, "k", "v2")
    assert cache.get(s1, "k") == "v1"
    assert cache.get(s2, "k") == "v2"
    cache.clear()
    assert cache.get(s1, "k") is None
