from pathlib import Path

import numpy as np
import pandas as pd


GAP_DENOM_FLOOR = 0.02
GAP_ABS_THRESHOLD = 0.05
GAP_REL_THRESHOLD = 0.50


def safe_read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _first_value(df: pd.DataFrame, column: str, default=np.nan):
    if column not in df.columns or df.empty:
        return default
    return df[column].iloc[0]


def _parse_significant(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _parse_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).lower() in ["true", "1", "yes"]


def _walk_values(row: dict, prefix: str, df: pd.DataFrame):
    row[f"{prefix}_annual_return"] = _first_value(df, "annual_return")
    row[f"{prefix}_sharpe"] = _first_value(df, "sharpe")
    row[f"{prefix}_max_drawdown"] = _first_value(df, "max_drawdown")
    row[f"{prefix}_bh_annual_return"] = _first_value(df, "bh_annual_return")
    row[f"{prefix}_bh_sharpe"] = _first_value(df, "bh_sharpe")
    row[f"{prefix}_bh_max_drawdown"] = _first_value(df, "bh_max_drawdown")
    row[f"{prefix}_excess_annual_return"] = _first_value(df, "excess_annual_return")
    row[f"{prefix}_beats_benchmark"] = (
        _parse_bool(_first_value(df, "beats_benchmark")) if "beats_benchmark" in df.columns else np.nan
    )


def summarize_one(symbol_dir: Path) -> dict:
    symbol = symbol_dir.name
    walk = safe_read_csv(symbol_dir / "walkforward_summary.csv")
    rel = safe_read_csv(symbol_dir / "reliability_report.csv")

    row = {
        "symbol": symbol,
        "has_walkforward": walk is not None,
        "has_reliability": rel is not None,
        "train_annual_return": np.nan,
        "test_annual_return": np.nan,
        "train_sharpe": np.nan,
        "test_sharpe": np.nan,
        "train_max_drawdown": np.nan,
        "test_max_drawdown": np.nan,
        "train_bh_annual_return": np.nan,
        "test_bh_annual_return": np.nan,
        "train_excess_annual_return": np.nan,
        "test_excess_annual_return": np.nan,
        "train_beats_benchmark": np.nan,
        "test_beats_benchmark": np.nan,
        "annual_return_gap": np.nan,
        "annual_return_gap_abs": np.nan,
        "annual_return_gap_relative": np.nan,
        "annual_return_gap_denominator_small": False,
        "oos_profitable": False,
        "oos_sharpe_positive": False,
        "oos_beats_benchmark": np.nan,
        "overfit_warning_gap_gt_50pct": False,
        "any_significant_after_bonferroni": False,
        "n_significant_rows": 0,
        "n_significant_regime_rows": 0,
        "significant_regimes": "",
        "best_regime_by_avg_return": "",
        "n_reliability_rows": 0,
    }

    if walk is not None and not walk.empty and "dataset" in walk.columns:
        train = walk[walk["dataset"] == "train"]
        test = walk[walk["dataset"] == "test"]

        if not train.empty:
            _walk_values(row, "train", train)
        if not test.empty:
            _walk_values(row, "test", test)

        train_ann = row["train_annual_return"]
        test_ann = row["test_annual_return"]
        if pd.notna(train_ann) and pd.notna(test_ann):
            gap_abs = abs(train_ann - test_ann)
            denom_small = abs(train_ann) < GAP_DENOM_FLOOR
            gap_rel = np.nan if denom_small else gap_abs / abs(train_ann)
            row["annual_return_gap_abs"] = gap_abs
            row["annual_return_gap_relative"] = gap_rel
            row["annual_return_gap"] = gap_rel
            row["annual_return_gap_denominator_small"] = bool(denom_small)
            row["overfit_warning_gap_gt_50pct"] = bool(
                (pd.notna(gap_rel) and gap_rel > GAP_REL_THRESHOLD) or gap_abs > GAP_ABS_THRESHOLD
            )

        row["oos_profitable"] = bool(pd.notna(row["test_annual_return"]) and row["test_annual_return"] > 0)
        row["oos_sharpe_positive"] = bool(pd.notna(row["test_sharpe"]) and row["test_sharpe"] > 0)
        row["oos_beats_benchmark"] = (
            bool(row["test_excess_annual_return"] > 0) if pd.notna(row["test_excess_annual_return"]) else np.nan
        )

    if rel is not None and not rel.empty:
        row["n_reliability_rows"] = len(rel)
        if "significant" in rel.columns:
            sig = _parse_significant(rel["significant"])
            row["n_significant_rows"] = int(sig.sum())
            row["any_significant_after_bonferroni"] = bool(sig.any())

            by_regime = rel[rel.get("regime").notna()].copy() if "regime" in rel.columns else pd.DataFrame()
            if not by_regime.empty:
                regime_sig = _parse_significant(by_regime["significant"])
                row["n_significant_regime_rows"] = int(regime_sig.sum())
                row["significant_regimes"] = "|".join(sorted(by_regime.loc[regime_sig, "regime"].dropna().astype(str).unique()))

        if "regime" in rel.columns and "avg_return" in rel.columns:
            by_regime = rel[rel["regime"].notna()].copy()
            if not by_regime.empty:
                grouped = by_regime.groupby("regime")["avg_return"].mean().sort_values(ascending=False)
                if not grouped.empty:
                    row["best_regime_by_avg_return"] = str(grouped.index[0])

    return row


def collect_regime_rows(symbol_dir: Path) -> list[dict]:
    rel = safe_read_csv(symbol_dir / "reliability_report.csv")
    if rel is None or rel.empty or "regime" not in rel.columns:
        return []
    regime_rows = rel[rel["regime"].notna()].copy()
    if regime_rows.empty:
        return []
    regime_rows["symbol"] = symbol_dir.name
    return regime_rows.to_dict("records")


def build_regime_summary(regime_rows: list[dict]) -> pd.DataFrame:
    if not regime_rows:
        return pd.DataFrame(
            columns=[
                "report_key",
                "regime",
                "n_rows",
                "n_symbols",
                "mean_signal_count",
                "mean_win_rate",
                "mean_avg_return",
                "pct_significant_after_bonferroni",
            ]
        )

    df = pd.DataFrame(regime_rows)
    if "significant" in df.columns:
        df["significant_bool"] = _parse_significant(df["significant"])
    else:
        df["significant_bool"] = False

    grouped = (
        df.groupby(["report_key", "regime"], dropna=True)
        .agg(
            n_rows=("symbol", "size"),
            n_symbols=("symbol", "nunique"),
            mean_signal_count=("signal_count", "mean"),
            mean_win_rate=("win_rate", "mean"),
            mean_avg_return=("avg_return", "mean"),
            pct_significant_after_bonferroni=("significant_bool", "mean"),
        )
        .reset_index()
        .sort_values(["report_key", "mean_avg_return"], ascending=[True, False])
    )
    return grouped


def aggregate_results(
    root: Path = Path("output/hstech30"),
    detail_out: Path = Path("output/hstech30_detail.csv"),
    summary_out: Path = Path("output/hstech30_summary.csv"),
    regime_out: Path = Path("output/hstech30_regime_summary.csv"),
) -> str:
    if not root.exists():
        raise SystemExit(f"Directory not found: {root}. Run each symbol into output/hstech30/<symbol>/ first.")

    symbol_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
    if not symbol_dirs:
        raise SystemExit(f"No symbol directories found: {root}")

    detail = pd.DataFrame([summarize_one(path) for path in symbol_dirs])
    detail_out.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_out, index=False, encoding="utf-8-sig")

    regime_summary = build_regime_summary([row for path in symbol_dirs for row in collect_regime_rows(path)])
    regime_summary.to_csv(regime_out, index=False, encoding="utf-8-sig")

    n = len(detail)
    summary = {
        "n_symbols": n,
        "n_with_walkforward": int(detail["has_walkforward"].sum()) if n else 0,
        "n_with_reliability": int(detail["has_reliability"].sum()) if n else 0,
        "pct_oos_profitable": float(detail["oos_profitable"].mean()) if n else np.nan,
        "pct_oos_sharpe_positive": float(detail["oos_sharpe_positive"].mean()) if n else np.nan,
        "n_with_benchmark": int(detail["test_excess_annual_return"].notna().sum()) if n else 0,
        "pct_oos_beats_benchmark": float(detail["oos_beats_benchmark"].mean(skipna=True)) if n else np.nan,
        "pct_overfit_warning_gap_gt_50pct": float(detail["overfit_warning_gap_gt_50pct"].mean()) if n else np.nan,
        "pct_gap_denominator_small": float(detail["annual_return_gap_denominator_small"].mean()) if n else np.nan,
        "pct_any_significant_after_bonferroni": float(detail["any_significant_after_bonferroni"].mean()) if n else np.nan,
        "median_test_annual_return": float(detail["test_annual_return"].median(skipna=True)) if n else np.nan,
        "median_test_bh_annual_return": float(detail["test_bh_annual_return"].median(skipna=True)) if n else np.nan,
        "median_test_excess_annual_return": float(detail["test_excess_annual_return"].median(skipna=True)) if n else np.nan,
        "median_test_sharpe": float(detail["test_sharpe"].median(skipna=True)) if n else np.nan,
        "median_annual_return_gap_abs": float(detail["annual_return_gap_abs"].median(skipna=True)) if n else np.nan,
    }

    verdict = "未验证"
    if (
        summary["pct_any_significant_after_bonferroni"] >= 0.5
        and summary["pct_oos_profitable"] >= 0.6
        and summary["pct_oos_sharpe_positive"] >= 0.6
        and summary["pct_oos_beats_benchmark"] >= 0.6
        and summary["pct_overfit_warning_gap_gt_50pct"] <= 0.4
        and summary["median_test_excess_annual_return"] > 0
    ):
        verdict = "初步验证"

    summary["verdict"] = verdict
    pd.DataFrame([summary]).to_csv(summary_out, index=False, encoding="utf-8-sig")
    return verdict
