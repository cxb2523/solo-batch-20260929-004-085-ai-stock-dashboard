"""Technical-indicator calculation driven by the frozen config snapshot.

Every rolling window / threshold is read from ``snapshot.indicator_windows``
at call time, so editing ``config.yaml`` (or an env override) from 20 to 50
changes the indicator columns on the next generation -- no code change, no
restart.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def calculate_indicators(
    data: pd.DataFrame, snapshot=None
) -> pd.DataFrame:
    """Return a copy of *data* with config-driven indicator columns.

    Window keys: sma_short, sma_long, rsi_window, rsi_oversold, rsi_overbought.
    MACD (12/26/9), Bollinger (sma_short) and ATR/Stochastic (14/3) derive
    from the same configured windows where applicable.
    """
    windows: Mapping[str, Any] = (
        snapshot.indicator_windows if snapshot is not None else None
    )

    def win(key: str, fallback: int) -> int:
        if windows is None:
            return fallback
        try:
            return int(windows.get(key, fallback))
        except (TypeError, ValueError):
            return fallback

    sma_short = win("sma_short", 20)
    sma_long = win("sma_long", 50)
    rsi_window = win("rsi_window", 14)
    rsi_oversold = win("rsi_oversold", 30)
    rsi_overbought = win("rsi_overbought", 70)

    df = data.copy()
    close = df["Close"]

    # Moving averages -- thresholds come straight from the snapshot.
    df[f"SMA_{sma_short}"] = close.rolling(window=sma_short).mean()
    df[f"SMA_{sma_long}"] = close.rolling(window=sma_long).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD.
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

    # RSI with the configured window.
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=rsi_window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df["RSI"] = (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], 100.0)

    # Bollinger Bands follow the short SMA window.
    df["BB_middle"] = close.rolling(window=sma_short).mean()
    bb_std = close.rolling(window=sma_short).std()
    df["BB_upper"] = df["BB_middle"] + 2 * bb_std
    df["BB_lower"] = df["BB_middle"] - 2 * bb_std

    # Volume / volatility / stochastic (fixed secondary windows).
    df["Volume_SMA"] = df["Volume"].rolling(window=sma_short).mean()
    df["Volume_ratio"] = df["Volume"] / df["Volume_SMA"]
    df["High_Low"] = df["High"] - df["Low"]
    df["High_Close"] = (df["High"] - close.shift()).abs()
    df["Low_Close"] = (df["Low"] - close.shift()).abs()
    true_range = df[["High_Low", "High_Close", "Low_Close"]].max(axis=1)
    df["ATR"] = true_range.rolling(window=14).mean()
    low_n = df["Low"].rolling(window=14).min()
    high_n = df["High"].rolling(window=14).max()
    df["Stoch_K"] = 100 * ((close - low_n) / (high_n - low_n))
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()

    # Threshold-derived signal columns (config drives the levels too).
    df["RSI_signal"] = np.select(
        [df["RSI"] <= rsi_oversold, df["RSI"] >= rsi_overbought],
        ["oversold", "overbought"],
        default="neutral",
    )
    df.attrs["windows_used"] = {
        "sma_short": sma_short,
        "sma_long": sma_long,
        "rsi_window": rsi_window,
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
    }
    return df


def latest_metrics(
    frame: pd.DataFrame, snapshot=None
) -> dict:
    """Summarise the most recent fully-formed row for the dashboard/API."""
    if frame is None or frame.empty:
        return {"error": "no data"}

    windows = frame.attrs.get("windows_used") or (
        dict(snapshot.indicator_windows) if snapshot is not None else {}
    )
    sma_short_col = f"SMA_{windows.get('sma_short', 20)}"
    sma_long_col = f"SMA_{windows.get('sma_long', 50)}"

    last = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else last

    def number(value) -> float | None:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 4)

    close = number(last["Close"])
    prev_close = number(previous["Close"])
    change_pct = None
    if close is not None and prev_close not in (None, 0):
        change_pct = round((close - prev_close) / prev_close * 100, 4)

    return {
        "date": str(frame.index[-1].date()),
        "close": close,
        "change_pct": change_pct,
        "volume": int(last["Volume"]) if not pd.isna(last["Volume"]) else None,
        sma_short_col: number(last.get(sma_short_col)),
        sma_long_col: number(last.get(sma_long_col)),
        "RSI": number(last.get("RSI")),
        "RSI_signal": str(last["RSI_signal"]) if "RSI_signal" in frame.columns else None,
        "MACD": number(last.get("MACD")),
        "MACD_signal": number(last.get("MACD_signal")),
        "ATR": number(last.get("ATR")),
        "windows_used": {k: int(v) for k, v in windows.items()},
    }
