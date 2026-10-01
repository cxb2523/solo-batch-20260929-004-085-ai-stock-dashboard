"""Orchestration: fetch -> calculate under ONE frozen config snapshot.

A dashboard pass calls :func:`build_dashboard`; the snapshot is acquired once
and passed to every stage, so the computed indicators and the "effective
config" table shown on the page always belong to the same generation -- never
half old, half new.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import config_loader
import datasource
from cache import SnapshotCache
from indicators import calculate_indicators, latest_metrics


@dataclass(frozen=True)
class TickerResult:
    symbol: str
    source: str
    metrics: Dict[str, Any]
    error: Optional[str]


def compute_ticker(symbol: str, snapshot, cache: Optional[SnapshotCache]) -> TickerResult:
    """Fetch + calculate for one symbol, memoised by snapshot fingerprint."""
    cached = None
    if cache is not None:
        cached = cache.get(snapshot.fingerprint, "ticker", symbol)
        if cached is not None:
            return cached

    fetched = datasource.fetch_prices(symbol, snapshot=snapshot)
    if fetched.data is None:
        result = TickerResult(
            symbol=symbol, source="none", metrics={}, error=fetched.error
        )
    else:
        frame = calculate_indicators(fetched.data, snapshot=snapshot)
        result = TickerResult(
            symbol=symbol,
            source=fetched.source,
            metrics=latest_metrics(frame, snapshot=snapshot),
            error=None,
        )

    if cache is not None:
        cache.set(
            snapshot.fingerprint, "ticker", symbol,
            result, snapshot.cache_ttl_seconds,
        )
    return result


def build_dashboard(
    snapshot=None, cache: Optional[SnapshotCache] = None
) -> Dict[str, Any]:
    """Compute results for every configured ticker under one snapshot."""
    snapshot = snapshot or config_loader.get_snapshot()
    results: List[TickerResult] = [
        compute_ticker(symbol, snapshot, cache) for symbol in snapshot.tickers
    ]
    return {
        "snapshot": snapshot,
        "results": results,
    }
