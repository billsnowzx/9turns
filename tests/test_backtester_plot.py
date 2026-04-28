from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester import Backtester


def test_plot_contains_split_markers_and_heatmap():
    idx = pd.date_range("2024-01-01", periods=90, freq="D")
    prices = [100 + i * 0.2 for i in range(90)]
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000] * len(prices),
        },
        index=idx,
    )
    bt = Backtester(df, init_cash=1000)
    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    entries.iloc[10] = True
    exits.iloc[20] = True
    res = bt.run(entries, exits, lag=0, fees=0.0, sl_stop=1.0, tp_stop=10.0)

    out = Path("output/test_backtester_plot.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    bt.plot(res, save_path=str(out), split_date=df.index[60])
    text = out.read_text(encoding="utf-8")
    assert "triangle-up" in text
    assert "triangle-down" in text
    assert "heatmap" in text.lower()
