from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import DataLoader


def test_weekly_request_resamples_daily_source():
    loader = DataLoader()
    idx = pd.date_range("2024-01-01", periods=140, freq="D")
    daily_df = pd.DataFrame(
        {
            "open": [100.0] * len(idx),
            "high": [101.0] * len(idx),
            "low": [99.0] * len(idx),
            "close": [100.0] * len(idx),
            "volume": [1000.0] * len(idx),
        },
        index=idx,
    )

    loader._load_yfinance = lambda symbol, freq, start, end: daily_df.copy()
    out = loader.load("us", "FAKE_WEEKLY_CASE", freq="weekly", start="2024-01-01", end="2024-06-30")

    assert len(out) < len(daily_df)
    step = out.index.to_series().diff().dropna().mode().iloc[0]
    assert step >= pd.Timedelta(days=5)
