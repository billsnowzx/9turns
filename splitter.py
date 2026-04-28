from typing import List, Tuple

import pandas as pd


class WalkForwardSplitter:
    def __init__(self, train_ratio: float = 0.7, n_splits: int = 1):
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        self.train_ratio = train_ratio
        self.n_splits = n_splits

    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        if len(df) < 20:
            raise ValueError("DataFrame is too short for walk-forward split")

        n = len(df)
        train_size = int(n * self.train_ratio)
        test_size = n - train_size
        if train_size < 10 or test_size < 5:
            raise ValueError("Invalid split size from current train_ratio/data length")

        if self.n_splits == 1:
            return [(df.iloc[:train_size].copy(), df.iloc[train_size:].copy())]

        splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        max_start = n - (train_size + test_size)
        step = max(1, max_start // (self.n_splits - 1)) if self.n_splits > 1 else 1

        start = 0
        for _ in range(self.n_splits):
            train_start = start
            train_end = train_start + train_size
            test_end = train_end + test_size
            if test_end > n:
                break
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()
            splits.append((train_df, test_df))
            start += step

        if not splits:
            raise ValueError("Unable to generate walk-forward splits")
        return splits
