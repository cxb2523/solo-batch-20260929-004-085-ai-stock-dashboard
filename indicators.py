"""Technical-indicator calculation driven entirely by the config snapshot.

Every window/span comes from ``snapshot['indicators']`` - nothing is hard
coded - so flipping e.g. ``sma_windows[0]`` from 20 to 50 changes the
computed indicators. Optional SnapshotCache keys results by snapshot
fingerprint, so a config change invalidates old results automatically.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_indicators(
    prices: pd.DataFrame, snapshot, cache=None, cache_key: str = "indicators"
) -> pd.DataFrame:
    """Return a copy of ``prices`` with indicator columns added."""
    if cache is not None:
        hit = cache.get(snapshot, cache_key)
        if hit is not None:
            return hit

    cfg = snapshot["indicators"]
    df = prices.copy()

    # Moving averages (windows configurable; columns named by the window)
    for window in cfg["sma_windows"]:
        df[f"SMA_{window}"] = df["Close"].rolling(int(window)).mean()

    ema_cols = []
    for span in cfg["ema_spans"]:
        col = f"EMA_{int(span)}"
        df[col] = df["Close"].ewm(span=int(span), adjust=False).mean()
        ema_cols.append(col)

    # MACD from the (configurable) fast/slow EMAs + configurable signal span
    if len(ema_cols) >= 2:
        df["MACD"] = df[ema_cols[0]] - df[ema_cols[1]]
        df["MACD_signal"] = df["MACD"].ewm(span=int(cfg["macd_signal"]), adjust=False).mean()
        df["MACD_histogram"] = df["MACD"] - df["MACD_signal"]

    # RSI (Wilder-style)
    rsi_w = int(cfg["rsi_window"])
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_w, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_w, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    # Bollinger bands
    bb_w = int(cfg["bollinger"]["window"])
    bb_std_mult = float(cfg["bollinger"]["std"])
    df["BB_middle"] = df["Close"].rolling(bb_w).mean()
    bb_std = df["Close"].rolling(bb_w).std(ddof=0)
    df["BB_upper"] = df["BB_middle"] + bb_std * bb_std_mult
    df["BB_lower"] = df["BB_middle"] - bb_std * bb_std_mult

    # Volume average
    vol_w = int(cfg["volume_window"])
    df["Volume_SMA"] = df["Volume"].rolling(vol_w).mean()
    df["Volume_ratio"] = df["Volume"] / df["Volume_SMA"]

    # Average True Range
    atr_w = int(cfg["atr_window"])
    prev_close = df["Close"].shift()
    true_range = pd.concat(
        [df["High"] - df["Low"],
         (df["High"] - prev_close).abs(),
         (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR"] = true_range.rolling(atr_w).mean()

    # Stochastic oscillator
    k_w = int(cfg["stoch"]["k_window"])
    d_w = int(cfg["stoch"]["d_window"])
    low_k = df["Low"].rolling(k_w).min()
    high_k = df["High"].rolling(k_w).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_k) / (high_k - low_k)
    df["Stoch_D"] = df["Stoch_K"].rolling(d_w).mean()

    if cache is not None:
        cache.set(snapshot, cache_key, df)
    return df


def _clean(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, 4)


def latest_metrics(symbol: str, frame: pd.DataFrame, source: str) -> dict:
    """JSON-safe summary of the latest row for the home page / API."""
    last = frame.iloc[-1]
    sma_windows = [int(c[4:]) for c in frame.columns
                   if c.startswith("SMA_") and c[4:].isdigit()]
    metrics: dict = {
        "symbol": symbol,
        "source": source,
        "date": str(frame.index[-1].date()),
        "close": _clean(last.get("Close")),
        "RSI": _clean(last.get("RSI")),
        "MACD": _clean(last.get("MACD")),
        "ATR": _clean(last.get("ATR")),
        "volume_ratio": _clean(last.get("Volume_ratio")),
    }
    for window in sorted(sma_windows):
        metrics[f"SMA_{window}"] = _clean(last.get(f"SMA_{window}"))
    return metrics
