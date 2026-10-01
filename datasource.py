"""Data-source layer: live Yahoo Finance fetch with offline fixture fallback."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("stock_dashboard.datasource")

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_FILE = FIXTURE_DIR / "sample_prices.csv"

_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass(frozen=True)
class PriceFetchResult:
    """Outcome of one fetch. Never raises: check ``error``."""

    symbol: str
    data: Optional[pd.DataFrame]
    source: str  # "yfinance" | "fixture"
    error: Optional[str] = None


def load_fixture(symbol: str = "AAPL") -> pd.DataFrame:
    """Load the bundled offline OHLCV fixture (symbol-independent series)."""
    df = pd.read_csv(FIXTURE_FILE, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    df.attrs["symbol"] = symbol
    return df[_OHLCV_COLUMNS]


def _fetch_yfinance(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch from yfinance. Imported lazily so offline installs still work."""
    import yfinance as yf  # local import: optional at runtime

    ticker = yf.Ticker(symbol)
    frame = ticker.history(period=period)
    if frame is None or frame.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol}")
    missing = [col for col in _OHLCV_COLUMNS if col not in frame.columns]
    if missing:
        raise RuntimeError(f"yfinance frame for {symbol} missing columns {missing}")
    return frame[_OHLCV_COLUMNS].copy()


def fetch_prices(
    symbol: str,
    snapshot=None,
    period: str = "1y",
) -> PriceFetchResult:
    """Fetch OHLCV for *symbol* according to *snapshot* datasource switches.

    Resolution order:
      1. yfinance when ``datasources.yfinance_enabled`` is true;
      2. local fixture when yfinance is disabled, unavailable, empty, or
         raises -- provided ``datasources.local_fixture_enabled`` is true.

    No exception escapes: network failures during an offline smoke test simply
    produce a ``source="fixture"`` result (or ``data=None`` if fixtures are
    disabled too, with ``error`` populated).
    """
    yfinance_enabled = True
    fixture_enabled = True
    if snapshot is not None:
        yfinance_enabled = bool(snapshot.datasources.get("yfinance_enabled", True))
        fixture_enabled = bool(
            snapshot.datasources.get("local_fixture_enabled", True)
        )

    if yfinance_enabled:
        try:
            frame = _fetch_yfinance(symbol, period=period)
            frame.attrs["symbol"] = symbol
            return PriceFetchResult(symbol=symbol, data=frame, source="yfinance")
        except Exception as exc:  # ImportError, network error, empty frame, ...
            logger.warning("yfinance fetch failed for %s: %s", symbol, exc)
            live_error = f"{exc.__class__.__name__}: {exc}"
    else:
        live_error = "yfinance disabled by configuration"

    if fixture_enabled:
        try:
            frame = load_fixture(symbol)
            return PriceFetchResult(symbol=symbol, data=frame, source="fixture")
        except Exception as exc:  # pragma: no cover - fixture ships in repo
            logger.error("local fixture load failed for %s: %s", symbol, exc)
            return PriceFetchResult(
                symbol=symbol,
                data=None,
                source="none",
                error=f"{live_error}; fixture also failed: {exc}",
            )

    return PriceFetchResult(
        symbol=symbol, data=None, source="none", error=live_error
    )
