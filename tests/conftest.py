"""Shared offline fixtures."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fixed_ohlcv():
    """Deterministic 120-row OHLCV frame returned by the stubbed datasource."""
    rng = np.random.default_rng(7)
    n = 120
    dates = pd.bdate_range(end="2026-09-28", periods=n)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=dates)
    open_ = close + rng.normal(0, 0.4, n)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.5
    volume = rng.integers(500_000, 1_500_000, n)
    return pd.DataFrame(
        {
            "Open": open_.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Volume": volume,
        },
        index=dates,
    )
