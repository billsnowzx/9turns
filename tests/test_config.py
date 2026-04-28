from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from combo_strategy import ComboStrategy
from main import load_config


def test_load_config_from_yaml():
    cfg = load_config("config.yaml")
    assert isinstance(cfg, dict)
    assert "FORWARD_DAYS" in cfg


def test_combo_uses_config_hold_days_without_explicit_arg():
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    close = [100 + i for i in range(40)]
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
    sigs = pd.DataFrame([{"date": idx[20], "signal": "buy9", "bar_index": 20, "close_at_signal": close[20]}])
    combo = ComboStrategy(df, sigs, config={"hold_days": 3})
    results = combo.run_all_combos()
    pure = [r for r in results if r["name"] == "pure_buy9"][0]
    entry_i = pure["entries"][pure["entries"]].index[0]
    exit_i = pure["exits"][pure["exits"]].index[0]
    assert (exit_i - entry_i).days == 3
