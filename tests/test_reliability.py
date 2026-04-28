from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reliability import ReliabilityAnalyzer


def _base_df(n=40):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = [100 + i for i in range(n)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [1_000] * n,
        },
        index=idx,
    )


def test_win_rate_and_profit_factor_fields_exist():
    price_df = _base_df(40)
    signals_df = pd.DataFrame(
        [{"date": price_df.index[i], "signal": "buy9", "bar_index": i, "close_at_signal": price_df.iloc[i]["close"]} for i in range(5, 20)]
    )
    analyzer = ReliabilityAnalyzer(price_df, signals_df)
    analyzer.FORWARD_DAYS = [1]
    report = analyzer.full_report()
    df = report["buy9_overall"]
    assert not df.empty
    assert "win_rate" in df.columns
    assert "profit_factor" in df.columns


def test_regime_split_outputs_columns():
    price_df = _base_df(260)
    signals_df = pd.DataFrame(
        [{"date": price_df.index[i], "signal": "buy9", "bar_index": i, "close_at_signal": price_df.iloc[i]["close"]} for i in range(80, 160, 5)]
    )
    analyzer = ReliabilityAnalyzer(price_df, signals_df)
    report = analyzer.full_report()
    by_regime = report["buy9_by_regime"]
    assert "regime" in by_regime.columns


def test_bonferroni_fields_and_logic():
    price_df = _base_df(80)
    signals_df = pd.DataFrame(
        [{"date": price_df.index[i], "signal": "buy9", "bar_index": i, "close_at_signal": price_df.iloc[i]["close"]} for i in range(10, 40)]
    )
    analyzer = ReliabilityAnalyzer(price_df, signals_df)
    analyzer.FORWARD_DAYS = [1, 2, 3]
    report = analyzer.full_report()
    df = report["buy9_overall"]
    assert "p_raw" in df.columns
    assert "p_bonferroni" in df.columns
    assert (df["p_bonferroni"] >= df["p_raw"]).all()
