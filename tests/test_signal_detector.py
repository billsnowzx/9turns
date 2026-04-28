from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_detector import TDSequential


def _make_ohlcv(close_values):
    idx = pd.date_range("2024-01-01", periods=len(close_values), freq="D")
    return pd.DataFrame(
        {
            "open": close_values,
            "high": [c + 1 for c in close_values],
            "low": [c - 1 for c in close_values],
            "close": close_values,
            "volume": [1_000] * len(close_values),
        },
        index=idx,
    )


def test_buy9_detected_on_known_decreasing_sequence():
    close = list(range(120, 100, -1))
    df = _make_ohlcv(close)
    td = TDSequential(df)
    signals = td.detect_all(cooldown_bars=0)
    buy9 = signals[signals["signal"] == "buy9"]
    assert len(buy9) >= 1


def test_sideways_sequence_has_no_setup_signals():
    close = [100] * 30
    df = _make_ohlcv(close)
    td = TDSequential(df)
    signals = td.detect_all(cooldown_bars=0)
    assert signals.empty


def test_countdown_interrupted_by_reversal_no_buy13():
    down_leg = list(range(120, 101, -1))
    up_leg = list(range(102, 132))
    close = down_leg + up_leg
    df = _make_ohlcv(close)
    td = TDSequential(df)
    signals = td.detect_all(cooldown_bars=0)
    buy13 = signals[signals["signal"] == "buy13"]
    assert buy13.empty


def test_short_series_less_than_9_bars_no_error():
    close = [100, 99, 98, 97, 96, 95, 94, 93]
    df = _make_ohlcv(close)
    td = TDSequential(df)
    signals = td.detect_all(cooldown_bars=0)
    assert isinstance(signals, pd.DataFrame)


def test_countdown_modes_strict_not_more_than_simplified():
    down = []
    v = 200
    for i in range(90):
        v -= 1
        if i % 6 == 0:
            v += 2
        down.append(v)
    up = []
    for i in range(90):
        v += 1
        if i % 6 == 0:
            v -= 2
        up.append(v)
    close = down + up
    df = _make_ohlcv(close)

    td_simple = TDSequential(df, countdown_mode="simplified")
    td_strict = TDSequential(df, countdown_mode="strict")
    simple = td_simple.detect_all(cooldown_bars=0)
    strict = td_strict.detect_all(cooldown_bars=0)

    simple_cd = simple[simple["signal"].isin(["buy13", "sell13"])]
    strict_cd = strict[strict["signal"].isin(["buy13", "sell13"])]
    assert len(strict_cd) <= len(simple_cd)
    assert len(strict_cd) != len(simple_cd)
