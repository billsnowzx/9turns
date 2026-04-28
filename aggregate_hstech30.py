# aggregate_hstech30.py
# 汇总每只股票输出目录中的研究结果，给出“是否验证九转神奇”的整体结论

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("output/hstech30")
SUMMARY_OUT = Path("output/hstech30_summary.csv")
DETAIL_OUT = Path("output/hstech30_detail.csv")


def safe_read_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def summarize_one(symbol_dir: Path):
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
        "annual_return_gap": np.nan,
        "oos_profitable": False,
        "oos_sharpe_positive": False,
        "overfit_warning_gap_gt_50pct": False,
        "any_significant_after_bonferroni": False,
        "n_significant_rows": 0,
        "n_reliability_rows": 0,
    }

    if walk is not None and not walk.empty:
        w = walk.copy()
        if "dataset" in w.columns:
            tr = w[w["dataset"] == "train"]
            te = w[w["dataset"] == "test"]
            if not tr.empty:
                row["train_annual_return"] = tr["annual_return"].iloc[0] if "annual_return" in tr.columns else np.nan
                row["train_sharpe"] = tr["sharpe"].iloc[0] if "sharpe" in tr.columns else np.nan
                row["train_max_drawdown"] = tr["max_drawdown"].iloc[0] if "max_drawdown" in tr.columns else np.nan
            if not te.empty:
                row["test_annual_return"] = te["annual_return"].iloc[0] if "annual_return" in te.columns else np.nan
                row["test_sharpe"] = te["sharpe"].iloc[0] if "sharpe" in te.columns else np.nan
                row["test_max_drawdown"] = te["max_drawdown"].iloc[0] if "max_drawdown" in te.columns else np.nan

        ta = row["train_annual_return"]
        oa = row["test_annual_return"]
        if pd.notna(ta) and pd.notna(oa):
            denom = max(abs(ta), 1e-12)
            gap = abs(ta - oa) / denom
            row["annual_return_gap"] = gap
            row["overfit_warning_gap_gt_50pct"] = bool(gap > 0.5)

        row["oos_profitable"] = bool(pd.notna(row["test_annual_return"]) and row["test_annual_return"] > 0)
        row["oos_sharpe_positive"] = bool(pd.notna(row["test_sharpe"]) and row["test_sharpe"] > 0)

    if rel is not None and not rel.empty:
        row["n_reliability_rows"] = len(rel)
        # 兼容 bool 列 or 字符串列
        if "significant" in rel.columns:
            sig = rel["significant"]
            if sig.dtype == bool:
                n_sig = int(sig.sum())
            else:
                n_sig = int(sig.astype(str).str.lower().isin(["true", "1", "yes"]).sum())
            row["n_significant_rows"] = n_sig
            row["any_significant_after_bonferroni"] = bool(n_sig > 0)

    return row


def main():
    if not ROOT.exists():
        raise SystemExit(f"目录不存在: {ROOT}. 请先按每只股票分别输出到 output/hstech30/<symbol>/")

    symbol_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    if not symbol_dirs:
        raise SystemExit(f"未发现子目录: {ROOT}")

    rows = [summarize_one(d) for d in symbol_dirs]
    detail = pd.DataFrame(rows)
    detail.to_csv(DETAIL_OUT, index=False, encoding="utf-8-sig")

    n = len(detail)
    summary = {
        "n_symbols": n,
        "n_with_walkforward": int(detail["has_walkforward"].sum()),
        "n_with_reliability": int(detail["has_reliability"].sum()),
        "pct_oos_profitable": float(detail["oos_profitable"].mean()) if n else np.nan,
        "pct_oos_sharpe_positive": float(detail["oos_sharpe_positive"].mean()) if n else np.nan,
        "pct_overfit_warning_gap_gt_50pct": float(detail["overfit_warning_gap_gt_50pct"].mean()) if n else np.nan,
        "pct_any_significant_after_bonferroni": float(detail["any_significant_after_bonferroni"].mean()) if n else np.nan,
        "median_test_annual_return": float(detail["test_annual_return"].median(skipna=True)),
        "median_test_sharpe": float(detail["test_sharpe"].median(skipna=True)),
    }

    verdict = "未验证"
    # 你可以按需要调整阈值
    if (
        summary["pct_any_significant_after_bonferroni"] >= 0.5
        and summary["pct_oos_profitable"] >= 0.6
        and summary["pct_oos_sharpe_positive"] >= 0.6
        and summary["pct_overfit_warning_gap_gt_50pct"] <= 0.4
    ):
        verdict = "初步验证"
    summary["verdict"] = verdict

    pd.DataFrame([summary]).to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    print("汇总完成:")
    print(f"- 明细: {DETAIL_OUT}")
    print(f"- 总结: {SUMMARY_OUT}")
    print(f"- 结论: {verdict}")


if __name__ == "__main__":
    main()
