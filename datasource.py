"""Price data sources with offline fixture fallback.

Resolution order per fetch:

1. yfinance (when ``sources.yfinance.enabled`` is true in the snapshot)
2. local CSV fixture (when ``sources.local_fixture.enabled`` is true)

A live-source failure is caught and degraded to the local fixture without
raising, so the dashboard keeps working offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataUnavailable(RuntimeError):
    """Raised only when every enabled data source fails."""


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return df[cols].sort_index()


def fetch_yfinance(symbol: str, period: str) -> pd.DataFrame:
    """Fetch from yfinance. Imported lazily so offline test runs don't need it.

    Tests monkeypatch this function directly to return fixed DataFrames.
    """
    import yfinance as yf  # local import: keeps the module importable offline

    frame = yf.Ticker(symbol).history(period=period)
    if frame is None or frame.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")
    return _normalize(frame)


def load_fixture(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"fixture {path} is empty")
    return _normalize(frame)


def fetch_prices(symbol: str, snapshot, cache=None) -> tuple[pd.DataFrame, str]:
    """Return ``(prices, source_name)`` using the given frozen config snapshot.

    ``cache`` is an optional SnapshotCache; successful fetches are cached under
    a fingerprinted key so stale entries can't survive a config change.
    """
    if cache is not None:
        hit = cache.get(snapshot, f"prices:{symbol}")
        if hit is not None:
            return hit

    sources = snapshot["sources"]
    period = snapshot["data"]["period"]
    errors: list[str] = []

    if sources["yfinance"]["enabled"]:
        try:
            frame = fetch_yfinance(symbol, period)
            result = (frame, "yfinance")
            if cache is not None:
                cache.set(snapshot, f"prices:{symbol}", result)
            return result
        except Exception as exc:  # network down / symbol unknown / ...
            errors.append(f"yfinance: {type(exc).__name__}: {exc}")

    if sources["local_fixture"]["enabled"]:
        try:
            frame = load_fixture(sources["local_fixture"]["path"])
            result = (frame, "fixture")
            if cache is not None:
                cache.set(snapshot, f"prices:{symbol}", result)
            return result
        except Exception as exc:
            errors.append(f"fixture: {type(exc).__name__}: {exc}")

    raise DataUnavailable(
        f"no price data for {symbol}; all sources failed -> {' | '.join(errors)}"
    )
