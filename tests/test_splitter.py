from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splitter import WalkForwardSplitter


def test_walkforward_single_split_respects_ratio_and_order():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({"close": range(100)}, index=idx)

    splitter = WalkForwardSplitter(train_ratio=0.7, n_splits=1)
    splits = splitter.split(df)
    train_df, test_df = splits[0]

    assert len(splits) == 1
    assert len(train_df) == 70
    assert len(test_df) == 30
    assert train_df.index.max() < test_df.index.min()
