from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hstech_aggregation import aggregate_results, summarize_one


def test_summarize_one_uses_absolute_gap_when_train_return_near_zero(tmp_path):
    symbol_dir = tmp_path / "0020.HK"
    symbol_dir.mkdir()
    pd.DataFrame(
        [
            {
                "dataset": "train",
                "annual_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "bh_annual_return": 0.01,
                "excess_annual_return": -0.01,
                "beats_benchmark": False,
            },
            {
                "dataset": "test",
                "annual_return": -0.08,
                "sharpe": 0.5,
                "max_drawdown": -0.2,
                "bh_annual_return": -0.10,
                "excess_annual_return": 0.02,
                "beats_benchmark": True,
            },
        ]
    ).to_csv(symbol_dir / "walkforward_summary.csv", index=False)

    row = summarize_one(symbol_dir)

    assert row["annual_return_gap_denominator_small"] is True
    assert pd.isna(row["annual_return_gap_relative"])
    assert row["annual_return_gap_abs"] == 0.08
    assert row["overfit_warning_gap_gt_50pct"] is True
    assert row["oos_beats_benchmark"] is True


def test_aggregate_outputs_benchmark_and_regime_summary(tmp_path):
    root = tmp_path / "hstech30"
    symbol_dir = root / "09988.HK"
    symbol_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset": "train", "annual_return": 0.10, "sharpe": 1.0, "max_drawdown": -0.1},
            {
                "dataset": "test",
                "annual_return": 0.08,
                "sharpe": 0.9,
                "max_drawdown": -0.1,
                "bh_annual_return": 0.03,
                "bh_sharpe": 0.2,
                "bh_max_drawdown": -0.2,
                "excess_annual_return": 0.05,
                "beats_benchmark": True,
            },
        ]
    ).to_csv(symbol_dir / "walkforward_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "window": "5d",
                "signal_count": 10,
                "win_rate": 0.7,
                "avg_return": 0.02,
                "p_bonferroni": 0.01,
                "significant": True,
                "regime": "bear",
                "report_key": "buy9_by_regime",
            }
        ]
    ).to_csv(symbol_dir / "reliability_report.csv", index=False)

    detail_out = tmp_path / "detail.csv"
    summary_out = tmp_path / "summary.csv"
    regime_out = tmp_path / "regime.csv"
    verdict = aggregate_results(root, detail_out, summary_out, regime_out)

    detail = pd.read_csv(detail_out)
    summary = pd.read_csv(summary_out)
    regime = pd.read_csv(regime_out)

    assert verdict in ["未验证", "初步验证"]
    assert detail["test_excess_annual_return"].iloc[0] == 0.05
    assert summary["pct_oos_beats_benchmark"].iloc[0] == 1.0
    assert regime["regime"].iloc[0] == "bear"
    assert regime["pct_significant_after_bonferroni"].iloc[0] == 1.0
