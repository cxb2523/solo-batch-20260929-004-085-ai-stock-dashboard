"""Shared pytest fixtures (offline by design)."""

from pathlib import Path

import pandas as pd
import pytest

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_prices.csv"


@pytest.fixture
def fixed_prices() -> pd.DataFrame:
    frame = pd.read_csv(FIXTURE_CSV)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]


@pytest.fixture
def fixture_path() -> Path:
    return FIXTURE_CSV
