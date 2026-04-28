# run_hstech30_full_pipeline.py
# 一键完成：
# 1) 抓取/兜底 HSTECH 30 只
# 2) 逐只运行 main.run_full_research
# 3) 归档到 output/hstech30/<symbol>/
# 4) 聚合输出 output/hstech30_detail.csv + output/hstech30_summary.csv

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URLS = [
    "https://m.aastocks.com/sc/stocks/market/index/hk-index-con.aspx?index=HSTECH",
    "https://www.aastocks.com/sc/stocks/market/index/hk-index-con.aspx?index=HSTECH",
]

FALLBACK_30 = [
    "00020.HK","00241.HK","00268.HK","00285.HK","00300.HK","00700.HK","00780.HK","00981.HK","00992.HK","01024.HK",
    "01211.HK","01347.HK","01698.HK","01810.HK","02015.HK","02382.HK","03690.HK","03888.HK","06618.HK","06690.HK",
    "09618.HK","09626.HK","09660.HK","09863.HK","09866.HK","09868.HK","09888.HK","09961.HK","09988.HK","09999.HK",
]

LIVE_OUT = Path("output")
ARCHIVE_ROOT = Path("output/hstech30")
DETAIL_OUT = Path("output/hstech30_detail.csv")
SUMMARY_OUT = Path("output/hstech30_summary.csv")


def fetch_hstech_codes():
    headers = {"User-Agent": "Mozilla/5.0"}
    codes = set()
    for u in URLS:
        try:
            html = requests.get(u, headers=headers, timeout=20).text
            codes |= set(re.findall(r"(\d{5}\.HK)", html, flags=re.IGNORECASE))
            for x in re.findall(r'code[=/:"\']+(\d{5})', html, flags=re.IGNORECASE):
                codes.add(f"{x}.HK")
        except Exception:
            pass
    codes = sorted(c.upper() for c in codes if re.fullmatch(r"\d{5}\.HK", c.upper()))
    return codes[:30] if len(codes) >= 25 else FALLBACK_30


def run_one(symbol):
    cmd = [
        sys.executable, "-c",
        (
            "from main import run_full_research; "
            f"run_full_research(market='hk', symbol='{symbol}', "
            "freq='daily', start='2018-01-01', end='2026-04-28')"
        ),
    ]
    return subprocess.run(cmd, check=False).returncode


def normalize_hk_symbol(sym: str) -> str:
    code = sym.upper().replace(".HK", "")
    if not code.isdigit():
        return sym.upper()
    code4 = str(int(code)).zfill(4)
    return f"{code4}.HK"


def archive_outputs(symbol):
    tgt = ARCHIVE_ROOT / symbol
    tgt.mkdir(parents=True, exist_ok=True)
    files = [
        "research_report.md",
        "walkforward_summary.csv",
        "reliability_report.csv",
        "combo_report_train.csv",
        "backtest_chart.html",
    ]
    for fn in files:
        src = LIVE_OUT / fn
        if src.exists():
            shutil.copy2(src, tgt / fn)


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

    if walk is not None and not walk.empty and "dataset" in walk.columns:
        tr = walk[walk["dataset"] == "train"]
        te = walk[walk["dataset"] == "test"]

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
            gap = abs(ta - oa) / max(abs(ta), 1e-12)
            row["annual_return_gap"] = gap
            row["overfit_warning_gap_gt_50pct"] = bool(gap > 0.5)

        row["oos_profitable"] = bool(pd.notna(row["test_annual_return"]) and row["test_annual_return"] > 0)
        row["oos_sharpe_positive"] = bool(pd.notna(row["test_sharpe"]) and row["test_sharpe"] > 0)

    if rel is not None and not rel.empty:
        row["n_reliability_rows"] = len(rel)
        if "significant" in rel.columns:
            sig = rel["significant"]
            if sig.dtype == bool:
                n_sig = int(sig.sum())
            else:
                n_sig = int(sig.astype(str).str.lower().isin(["true", "1", "yes"]).sum())
            row["n_significant_rows"] = n_sig
            row["any_significant_after_bonferroni"] = bool(n_sig > 0)

    return row


def aggregate_results():
    symbol_dirs = sorted([p for p in ARCHIVE_ROOT.iterdir() if p.is_dir()])
    rows = [summarize_one(d) for d in symbol_dirs]
    detail = pd.DataFrame(rows)
    detail.to_csv(DETAIL_OUT, index=False, encoding="utf-8-sig")

    n = len(detail)
    summary = {
        "n_symbols": n,
        "n_with_walkforward": int(detail["has_walkforward"].sum()) if n else 0,
        "n_with_reliability": int(detail["has_reliability"].sum()) if n else 0,
        "pct_oos_profitable": float(detail["oos_profitable"].mean()) if n else np.nan,
        "pct_oos_sharpe_positive": float(detail["oos_sharpe_positive"].mean()) if n else np.nan,
        "pct_overfit_warning_gap_gt_50pct": float(detail["overfit_warning_gap_gt_50pct"].mean()) if n else np.nan,
        "pct_any_significant_after_bonferroni": float(detail["any_significant_after_bonferroni"].mean()) if n else np.nan,
        "median_test_annual_return": float(detail["test_annual_return"].median(skipna=True)) if n else np.nan,
        "median_test_sharpe": float(detail["test_sharpe"].median(skipna=True)) if n else np.nan,
    }

    verdict = "未验证"
    if (
        summary["pct_any_significant_after_bonferroni"] >= 0.5
        and summary["pct_oos_profitable"] >= 0.6
        and summary["pct_oos_sharpe_positive"] >= 0.6
        and summary["pct_overfit_warning_gap_gt_50pct"] <= 0.4
    ):
        verdict = "初步验证"

    summary["verdict"] = verdict
    pd.DataFrame([summary]).to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
    return verdict


def main():
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    symbols = fetch_hstech_codes()
    print(f"Using {len(symbols)} symbols")

    status_rows = ["symbol,ok"]
    for s in symbols:
        s_norm = normalize_hk_symbol(s)
        print(f"Running {s} -> {s_norm} ...")
        rc = run_one(s_norm)
        ok = 1 if rc == 0 else 0
        if ok:
            archive_outputs(s_norm)
        status_rows.append(f"{s_norm},{ok}")

    (ARCHIVE_ROOT / "run_status.csv").write_text("\n".join(status_rows), encoding="utf-8")

    verdict = aggregate_results()
    print("Done.")
    print(f"- run status: {ARCHIVE_ROOT / 'run_status.csv'}")
    print(f"- detail: {DETAIL_OUT}")
    print(f"- summary: {SUMMARY_OUT}")
    print(f"- verdict: {verdict}")


if __name__ == "__main__":
    main()
