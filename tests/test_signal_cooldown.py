from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_detector import TDSequential


def test_detect_all_applies_directional_cooldown():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    close = list(range(200, 160, -1))
    df = pd.DataFrame(
        {
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [1_000] * len(close),
        },
        index=idx,
    )

    td = TDSequential(df)
    no_cd = td.detect_all(cooldown_bars=0)
    with_cd = td.detect_all(cooldown_bars=10)

    no_cd_buy9 = no_cd[no_cd["signal"] == "buy9"]
    with_cd_buy9 = with_cd[with_cd["signal"] == "buy9"]

    assert len(no_cd_buy9) > len(with_cd_buy9)
    bars = with_cd_buy9["bar_index"].tolist()
    assert all((b2 - b1) > 10 for b1, b2 in zip(bars, bars[1:]))
