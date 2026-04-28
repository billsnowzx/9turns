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


def test_known_returns_win_rate_and_profit_factor():
    price_df = _base_df(30)
    signals_df = pd.DataFrame(
        [{"date": price_df.index[10], "signal": "buy9", "bar_index": 10, "close_at_signal": 110}]
    )
    analyzer = ReliabilityAnalyzer(price_df, signals_df)

    # 3 wins / 5 samples, gains avg=0.02, losses avg=0.02 -> pf=1.0
    returns = pd.Series([0.01, 0.03, -0.02, -0.02, 0.02])
    row = analyzer._win_rate_stats(returns, "buy9", window=5, n_tests=1)

    assert row["win_rate"] == 0.6
    assert row["profit_factor"] == 1.0


def test_market_regime_classification_contains_bull_and_bear():
    idx = pd.date_range("2020-01-01", periods=320, freq="D")
    close = [100] * 220 + [70] * 50 + [130] * 50
    price_df = pd.DataFrame(
        {
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [1_000] * len(close),
        },
        index=idx,
    )
    signals_df = pd.DataFrame(
        [{"date": idx[i], "signal": "buy9", "bar_index": i, "close_at_signal": close[i]} for i in range(230, 310, 10)]
    )

    analyzer = ReliabilityAnalyzer(price_df, signals_df)
    regimes = set(analyzer.price_df["regime"].dropna().unique().tolist())
    assert "bull" in regimes
    assert "bear" in regimes


def test_bonferroni_correction_and_significance_use_corrected_p():
    price_df = _base_df(50)
    signals_df = pd.DataFrame(
        [{"date": price_df.index[10], "signal": "buy9", "bar_index": 10, "close_at_signal": 110}]
    )
    analyzer = ReliabilityAnalyzer(price_df, signals_df)

    # wins=9/11 => raw p often <0.05; with n_tests=3 should become non-significant after correction
    returns = pd.Series([0.01] * 9 + [-0.01, -0.02])
    row = analyzer._win_rate_stats(returns, "buy9", window=5, n_tests=3)

    assert row["p_raw"] < 0.05
    assert row["p_bonferroni"] >= row["p_raw"]
    assert row["significant"] is False
