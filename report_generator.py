from pathlib import Path
from typing import Any

import pandas as pd


def _fmt_pct(v: Any) -> str:
    try:
        return f"{float(v):.2%}"
    except Exception:
        return "N/A"


def generate_research_report(
    reliability_report: dict,
    combo_results: list,
    backtest_bundle: dict,
    output_path: str = "output/research_report.md",
) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    train = backtest_bundle.get("train", {})
    test = backtest_bundle.get("test", {})
    summary = backtest_bundle.get("summary", {})

    lines = []
    lines.append("# Research Report")
    lines.append("")
    lines.append("## Backtest Summary")
    lines.append("")
    lines.append(f"- Train annual return: {_fmt_pct(train.get('annual_return'))}")
    lines.append(f"- Test annual return: {_fmt_pct(test.get('annual_return'))}")
    lines.append(f"- Train sharpe: {train.get('sharpe', 'N/A')}")
    lines.append(f"- Test sharpe: {test.get('sharpe', 'N/A')}")
    lines.append(f"- Train max drawdown: {_fmt_pct(train.get('max_drawdown'))}")
    lines.append(f"- Test max drawdown: {_fmt_pct(test.get('max_drawdown'))}")
    lines.append(f"- Annual return gap: {_fmt_pct(summary.get('annual_return_gap'))}")
    lines.append("")
    lines.append("## Best Combo Candidates")
    lines.append("")
    if combo_results:
        combo_df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, pd.Series)} for r in combo_results])
        view_cols = [c for c in ["name", "n_signals", "win_rate", "avg_ret", "profit_factor", "train_calmar", "train_sharpe", "train_max_dd"] if c in combo_df.columns]
        if view_cols:
            lines.append(combo_df[view_cols].to_markdown(index=False))
            lines.append("")

    lines.append("## Reliability Highlights")
    lines.append("")
    for key, df in reliability_report.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            lines.append(f"### {key}")
            cols = [c for c in ["window", "signal_count", "win_rate", "avg_return", "p_raw", "p_bonferroni", "significant"] if c in df.columns]
            if cols:
                lines.append(df[cols].to_markdown(index=False))
                lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append("- reliability csv: `output/reliability_report.csv`")
    lines.append("- combo csv: `output/combo_report_train.csv`")
    lines.append("- walk-forward summary: `output/walkforward_summary.csv`")
    lines.append("- backtest chart: `output/backtest_chart.html`")

    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)
