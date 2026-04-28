from pathlib import Path
import sys
import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import DataLoader


def test_data_loader_disk_cache_second_call_fast():
    loader = DataLoader()

    idx = pd.date_range("2024-01-01", periods=150, freq="D")
    fake_df = pd.DataFrame(
        {
            "open": [100.0] * 150,
            "high": [101.0] * 150,
            "low": [99.0] * 150,
            "close": [100.0] * 150,
            "volume": [1000.0] * 150,
        },
        index=idx,
    )

    def slow_load_yfinance(symbol, freq, start, end):
        time.sleep(0.7)
        return fake_df.copy()

    loader._load_yfinance = slow_load_yfinance

    unique_symbol = f"SPY_TEST_{int(time.time() * 1000)}"

    t0 = time.perf_counter()
    df1 = loader.load("us", unique_symbol, "daily", "2024-01-01", "2024-12-31")
    t1 = time.perf_counter()
    df2 = loader.load("us", unique_symbol, "daily", "2024-01-01", "2024-12-31")
    t2 = time.perf_counter()

    first = t1 - t0
    second = t2 - t1
    assert len(df1) == len(df2) == 150
    assert first > 0.6
    assert second < 0.5
