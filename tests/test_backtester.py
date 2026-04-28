from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester import Backtester


def _price_df(prices):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000] * len(prices),
        },
        index=idx,
    )


def test_single_trade_pnl_with_fees():
    df = _price_df([100, 100, 110, 110])
    bt = Backtester(df, init_cash=1000)
    entries = pd.Series([False, True, False, False], index=df.index)
    exits = pd.Series([False, False, True, False], index=df.index)
    r = bt.run(entries, exits, lag=0, sl_stop=1.0, tp_stop=10.0, fees=0.001, strategy_name="fee_test")
    ret = r["trades_df"].iloc[0]["return"]
    assert ret == pytest.approx(0.10, abs=1e-6)
    assert r["total_return"] > 0.09
    assert pd.notna(r["annual_return"])


def test_stop_loss_take_profit_trigger():
    df = _price_df([100, 100, 95, 120, 120])
    bt = Backtester(df, init_cash=1000)

    entries = pd.Series([False, True, False, False, False], index=df.index)
    exits = pd.Series([False, False, False, False, False], index=df.index)
    r = bt.run(entries, exits, lag=0, sl_stop=0.03, tp_stop=0.10, fees=0.0, strategy_name="sl_tp")
    assert len(r["trades_df"]) >= 1
    assert r["trades_df"].iloc[0]["return"] <= -0.03 or r["trades_df"].iloc[0]["return"] >= 0.10


def test_annual_return_is_derived_from_equity_curve():
    df = _price_df([100, 100, 105, 110, 115, 120, 125, 130])
    bt = Backtester(df, init_cash=1000)
    entries = pd.Series([False, True, False, False, False, False, False, False], index=df.index)
    exits = pd.Series([False, False, False, False, False, False, True, False], index=df.index)

    r = bt.run(entries, exits, lag=0, sl_stop=1.0, tp_stop=10.0, fees=0.0, strategy_name="annual")

    assert pd.notna(r["annual_return"])
    assert r["annual_return"] > 0


def test_vectorbt_and_simple_close_when_available():
    try:
        import vectorbt  # noqa: F401
    except Exception:
        pytest.skip("vectorbt not available")

    df = _price_df([100, 101, 102, 101, 103, 104, 103, 105, 106, 107, 108, 109])
    entries = pd.Series([False, True, False, False, False, True, False, False, False, False, False, False], index=df.index)
    exits = pd.Series([False, False, False, True, False, False, False, True, False, False, False, False], index=df.index)

    bt = Backtester(df, init_cash=1000)
    run_r = bt.run(entries, exits, lag=0, sl_stop=1.0, tp_stop=10.0, fees=0.0, strategy_name="compare")
    simple_r = bt._run_simple(entries, exits, sl_stop=1.0, tp_stop=10.0, fees=0.0, name="simple")
    assert abs(run_r["total_return"] - simple_r["total_return"]) <= 0.01
