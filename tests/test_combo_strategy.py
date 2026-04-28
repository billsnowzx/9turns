from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from combo_strategy import ComboStrategy


def test_get_best_combo_enriches_train_backtest_metrics():
    idx = pd.date_range("2023-01-01", periods=260, freq="D")
    close = [100 + i * 0.2 for i in range(260)]
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

    signal_dates = idx[20::20]
    signals_df = pd.DataFrame(
        [
            {"date": d, "signal": "buy9", "bar_index": i, "close_at_signal": price_df.iloc[i]["close"]}
            for i, d in enumerate(signal_dates, start=20)
            if i < len(price_df)
        ]
    )

    combo = ComboStrategy(price_df, signals_df)
    results = combo.run_all_combos(hold_days=5)
    best = combo.get_best_combo(results, sl_stop=1.0, tp_stop=10.0, fees=0.0)

    assert "train_sharpe" in best
    assert "train_calmar" in best
    assert "train_max_dd" in best
    assert any("train_calmar" in r for r in results)
