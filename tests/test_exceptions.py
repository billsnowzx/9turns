from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import DataLoader
from exceptions import TDDataError


def test_empty_dataframe_raises_tddataerror():
    loader = DataLoader()
    with pytest.raises(TDDataError):
        loader._validate(pd.DataFrame())
