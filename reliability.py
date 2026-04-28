import os
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


class ReliabilityAnalyzer:
    """
    Reliability analysis for TD signals.
    """

    FORWARD_DAYS = [3, 5, 10, 20]
    ALPHA = 0.05

    def __init__(self, price_df: pd.DataFrame, signals_df: pd.DataFrame):
        self.price_df = price_df.copy()
        self.signals_df = signals_df.copy()
        self.close = self.price_df["close"]
        self._add_market_regime()

    def _add_market_regime(self):
        ma200 = self.close.rolling(200, min_periods=60).mean()
        conditions = [
            self.close > ma200,
            self.close < ma200 * 0.90,
        ]
        self.price_df["regime"] = np.select(conditions, ["bull", "bear"], default="range")
        self.signals_df = self.signals_df.merge(
            self.price_df[["regime"]], left_on="date", right_index=True, how="left"
        )

    def _compute_forward_returns(self, signal_type: str) -> pd.DataFrame:
        sigs = self.signals_df[self.signals_df["signal"] == signal_type].copy()
        close_arr = self.close.values
        open_arr = self.price_df["open"].values

        records = []
        for _, row in sigs.iterrows():
            bar_idx = int(row["bar_index"])
            exec_idx = bar_idx + 1
            if exec_idx >= len(open_arr):
                continue
            entry_price = open_arr[exec_idx]

            record = {
                "date": row["date"],
                "regime": row.get("regime", "unknown"),
                "close": entry_price,
            }
            for n in self.FORWARD_DAYS:
                future_idx = exec_idx + n
                if future_idx < len(close_arr):
                    record[f"ret_{n}d"] = (close_arr[future_idx] - entry_price) / entry_price
                else:
                    record[f"ret_{n}d"] = np.nan
            records.append(record)

        return pd.DataFrame(records)

    def _win_rate_stats(self, returns: pd.Series, signal_type: str, window: int, n_tests: int) -> dict:
        returns = returns.dropna()
        if len(returns) < 5:
            return {}

        wins = (returns > 0).sum() if "buy" in signal_type else (returns < 0).sum()
        n = len(returns)
        win_rate = wins / n

        p_raw = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue
        p_bonferroni = min(1.0, p_raw * n_tests)

        gain_mean = returns[returns > 0].mean() if (returns > 0).any() else 0
        loss_mean = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
        profit_factor = gain_mean / loss_mean if loss_mean > 0 else np.inf

        return {
            "signal_count": n,
            "win_rate": round(win_rate, 4),
            "avg_return": round(float(returns.mean()), 6),
            "std_return": round(float(returns.std()), 6),
            "profit_factor": round(float(profit_factor), 4),
            "max_win": round(float(returns.max()), 6),
            "max_loss": round(float(returns.min()), 6),
            "p_raw": round(float(p_raw), 6),
            "p_bonferroni": round(float(p_bonferroni), 6),
            "significant": bool(p_bonferroni < self.ALPHA),
            "window": f"{window}d",
        }

    def full_report(self) -> Dict[str, pd.DataFrame]:
        report = {}
        n_tests = len(self.FORWARD_DAYS)

        for sig_type in ["buy9", "sell9"]:
            fwd = self._compute_forward_returns(sig_type)
            if fwd.empty:
                continue

            overall_rows = []
            for n in self.FORWARD_DAYS:
                col = f"ret_{n}d"
                if col not in fwd.columns:
                    continue
                row = self._win_rate_stats(fwd[col], sig_type, n, n_tests=n_tests)
                if row:
                    overall_rows.append(row)
            report[f"{sig_type}_overall"] = pd.DataFrame(overall_rows)

            regime_rows = []
            for regime in ["bull", "bear", "range"]:
                sub = fwd[fwd["regime"] == regime]
                for n in self.FORWARD_DAYS:
                    col = f"ret_{n}d"
                    if col not in sub.columns or sub.empty:
                        continue
                    row = self._win_rate_stats(sub[col], sig_type, n, n_tests=n_tests)
                    if row:
                        row["regime"] = regime
                        regime_rows.append(row)
            report[f"{sig_type}_by_regime"] = pd.DataFrame(regime_rows)
            report[f"{sig_type}_regime_dist"] = fwd["regime"].value_counts()

        return report

    def print_summary(self, report: dict):
        for sig_type in ["buy9", "sell9"]:
            key_overall = f"{sig_type}_overall"
            if key_overall in report and not report[key_overall].empty:
                print(f"\n== {sig_type} overall ==")
                cols = ["window", "signal_count", "win_rate", "avg_return", "profit_factor", "p_raw", "p_bonferroni", "significant"]
                print(report[key_overall][cols].to_string(index=False))

    def save_report(self, report: dict, path: str = "output/reliability_report.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        all_rows = []
        for key, val in report.items():
            if isinstance(val, pd.DataFrame) and not val.empty:
                tmp = val.copy()
                tmp["report_key"] = key
                all_rows.append(tmp)
        if all_rows:
            pd.concat(all_rows, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  reliability report saved: {path}")
