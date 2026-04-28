from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report_generator import generate_research_report


def test_generate_research_report_creates_markdown(tmp_path=None):
    rel = {
        "buy9_overall": pd.DataFrame(
            [
                {
                    "window": "5d",
                    "signal_count": 10,
                    "win_rate": 0.6,
                    "avg_return": 0.01,
                    "p_raw": 0.04,
                    "p_bonferroni": 0.08,
                    "significant": False,
                }
            ]
        )
    }
    combos = [{"name": "pure_buy9", "n_signals": 10, "win_rate": 0.6, "avg_ret": 0.01, "profit_factor": 1.2}]
    bt = {
        "train": {"annual_return": 0.2, "sharpe": 1.1, "max_drawdown": -0.1},
        "test": {"annual_return": 0.1, "sharpe": 0.9, "max_drawdown": -0.12},
        "summary": {"annual_return_gap": 0.5},
    }
    out = "output/test_research_report.md"
    path = generate_research_report(rel, combos, bt, output_path=out)
    text = Path(path).read_text(encoding="utf-8")
    assert "Research Report" in text
    assert "Backtest Summary" in text
    assert "Reliability Highlights" in text
