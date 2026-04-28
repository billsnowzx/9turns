from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester import Backtester
from combo_strategy import ComboStrategy
from reliability import ReliabilityAnalyzer


def _toy_price_df():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100, 100, 100],
            "high": [101, 101, 210, 101, 101, 101, 101, 101],
            "low": [99, 99, 99, 99, 99, 99, 99, 99],
            "close": [100, 100, 200, 100, 100, 100, 100, 100],
            "volume": [1_000] * 8,
        },
        index=idx,
    )


def test_combo_entries_are_shifted_one_bar():
    price_df = _toy_price_df()
    signals_df = pd.DataFrame(
        [{"date": price_df.index[2], "signal": "buy9", "bar_index": 2, "close_at_signal": 200}]
    )
    combo = ComboStrategy(price_df, signals_df)
    entries = combo._signals_to_series("buy9")

    assert entries.iloc[2] == False
    assert entries.iloc[3] == True


def test_backtester_lag_changes_execution_timing():
    price_df = _toy_price_df()
    bt = Backtester(price_df, init_cash=100_000)

    entries = pd.Series(False, index=price_df.index)
    exits = pd.Series(False, index=price_df.index)
    entries.iloc[2] = True
    exits.iloc[3] = True

    no_lag = bt.run(entries, exits, lag=0, sl_stop=1.0, tp_stop=10.0, fees=0.0, strategy_name="no_lag")
    with_lag = bt.run(entries, exits, lag=1, sl_stop=1.0, tp_stop=10.0, fees=0.0, strategy_name="lag1")

    assert no_lag["trades_df"].iloc[0]["return"] < 0
    assert with_lag["trades_df"].iloc[0]["return"] == 0


def test_reliability_uses_next_open_for_entry_price():
    price_df = _toy_price_df()
    signals_df = pd.DataFrame(
        [{"date": price_df.index[2], "signal": "buy9", "bar_index": 2, "close_at_signal": 200}]
    )
    analyzer = ReliabilityAnalyzer(price_df, signals_df)
    analyzer.FORWARD_DAYS = [1]

    fwd = analyzer._compute_forward_returns("buy9")
    assert len(fwd) == 1
    assert fwd.iloc[0]["close"] == 100
    assert fwd.iloc[0]["ret_1d"] == 0
