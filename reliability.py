"""
reliability.py — 可靠性验证模块

分析维度：
  1. 胜率 & 盈亏比（N日后收益）
  2. 市场环境分层（牛/熊/震荡）
  3. 统计显著性检验（二项检验）
  4. 周期对比（可扩展到周线/月线）
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List
import os


class ReliabilityAnalyzer:
    """
    验证九转神奇信号的历史可靠性。

    Parameters
    ----------
    price_df   : OHLCV DataFrame
    signals_df : detect_all() 的输出
    """

    FORWARD_DAYS = [3, 5, 10, 20]   # 信号后观察窗口

    def __init__(self, price_df: pd.DataFrame, signals_df: pd.DataFrame):
        self.price_df   = price_df.copy()
        self.signals_df = signals_df.copy()
        self.close      = price_df["close"]
        self._add_market_regime()

    # ─────────────────────────────────────────────────────────────
    # 市场环境分类
    # ─────────────────────────────────────────────────────────────
    def _add_market_regime(self):
        """
        用 200日均线判断市场环境：
          close > MA200              → 牛市
          close < MA200 * 0.90       → 熊市
          其余                       → 震荡市
        """
        ma200 = self.close.rolling(200, min_periods=60).mean()
        conditions = [
            self.close > ma200,
            self.close < ma200 * 0.90,
        ]
        self.price_df["regime"] = np.select(
            conditions, ["bull", "bear"], default="range"
        )
        # 将 regime 合并到 signals
        self.signals_df = self.signals_df.merge(
            self.price_df[["regime"]],
            left_on="date", right_index=True,
            how="left"
        )

    # ─────────────────────────────────────────────────────────────
    # 核心统计：计算 N 日后收益
    # ─────────────────────────────────────────────────────────────
    def _compute_forward_returns(self, signal_type: str) -> pd.DataFrame:
        """
        对指定信号类型，计算每个信号发出后 N 日的实际收益率。
        """
        sigs = self.signals_df[self.signals_df["signal"] == signal_type].copy()
        price_arr  = self.close.values
        price_dates = self.close.index

        records = []
        for _, row in sigs.iterrows():
            bar_idx = row["bar_index"]
            entry_price = price_arr[bar_idx]

            record = {
                "date":    row["date"],
                "regime":  row.get("regime", "unknown"),
                "close":   entry_price,
            }
            for n in self.FORWARD_DAYS:
                future_idx = bar_idx + n
                if future_idx < len(price_arr):
                    ret = (price_arr[future_idx] - entry_price) / entry_price
                    record[f"ret_{n}d"] = ret
                else:
                    record[f"ret_{n}d"] = np.nan
            records.append(record)

        return pd.DataFrame(records)

    # ─────────────────────────────────────────────────────────────
    # 统计检验
    # ─────────────────────────────────────────────────────────────
    def _win_rate_stats(self, returns: pd.Series, signal_type: str, window: int) -> dict:
        """
        胜率统计 + 二项检验。
        买9信号：上涨为胜；卖9信号：下跌为胜。
        """
        returns = returns.dropna()
        if len(returns) < 5:
            return {}

        if "buy" in signal_type:
            wins = (returns > 0).sum()
        else:
            wins = (returns < 0).sum()

        n = len(returns)
        win_rate = wins / n

        # 二项检验（H0: 胜率=0.5）
        pval = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue

        # 盈亏比
        gain_mean = returns[returns > 0].mean() if (returns > 0).any() else 0
        loss_mean = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
        profit_factor = gain_mean / loss_mean if loss_mean > 0 else np.inf

        return {
            "信号数量":    n,
            "胜率":        round(win_rate, 4),
            "平均收益":    round(returns.mean(), 4),
            "收益标准差":  round(returns.std(), 4),
            "盈亏比":      round(profit_factor, 2),
            "最大盈利":    round(returns.max(), 4),
            "最大亏损":    round(returns.min(), 4),
            "p值(二项检验)": round(pval, 4),
            "统计显著":    "✓" if pval < 0.05 else "✗",
        }

    # ─────────────────────────────────────────────────────────────
    # 完整报告
    # ─────────────────────────────────────────────────────────────
    def full_report(self) -> Dict[str, pd.DataFrame]:
        """
        生成完整的可靠性报告：
          - overall      : 总体胜率（不分市场环境）
          - by_regime    : 按牛/熊/震荡分层
          - regime_dist  : 信号在各市场环境下的分布
        """
        report = {}

        for sig_type in ["buy9", "sell9"]:
            fwd = self._compute_forward_returns(sig_type)
            if fwd.empty:
                continue

            # 1) 总体统计
            overall_rows = []
            for n in self.FORWARD_DAYS:
                col = f"ret_{n}d"
                if col not in fwd.columns:
                    continue
                stats_dict = self._win_rate_stats(fwd[col], sig_type, n)
                if stats_dict:
                    stats_dict["观察窗口"] = f"{n}日"
                    overall_rows.append(stats_dict)
            report[f"{sig_type}_overall"] = pd.DataFrame(overall_rows)

            # 2) 按市场环境分层
            regime_rows = []
            for regime in ["bull", "bear", "range"]:
                sub = fwd[fwd["regime"] == regime]
                for n in self.FORWARD_DAYS:
                    col = f"ret_{n}d"
                    if col not in sub.columns or sub.empty:
                        continue
                    s = self._win_rate_stats(sub[col], sig_type, n)
                    if s:
                        s["市场环境"] = {"bull": "牛市", "bear": "熊市", "range": "震荡市"}[regime]
                        s["观察窗口"] = f"{n}日"
                        regime_rows.append(s)
            report[f"{sig_type}_by_regime"] = pd.DataFrame(regime_rows)

            # 3) 市场环境分布
            dist = fwd["regime"].value_counts()
            report[f"{sig_type}_regime_dist"] = dist

        return report

    # ─────────────────────────────────────────────────────────────
    # 打印摘要
    # ─────────────────────────────────────────────────────────────
    def print_summary(self, report: dict):
        REGIME_NAMES = {"bull": "牛市", "bear": "熊市", "range": "震荡市"}

        for sig_type in ["buy9", "sell9"]:
            label = "买9（逢低买入）" if sig_type == "buy9" else "卖9（逢高做空/减仓）"
            print(f"\n  ── {label} ──")

            key_overall = f"{sig_type}_overall"
            if key_overall in report and not report[key_overall].empty:
                df = report[key_overall]
                df_print = df[["观察窗口", "信号数量", "胜率", "平均收益", "盈亏比", "统计显著"]]
                print(df_print.to_string(index=False))

            key_regime = f"{sig_type}_by_regime"
            if key_regime in report and not report[key_regime].empty:
                print(f"\n  市场环境分层：")
                df_r = report[key_regime]
                # 只展示10日窗口的分层结果
                df_r_10 = df_r[df_r["观察窗口"] == "10日"] if "观察窗口" in df_r.columns else df_r
                if not df_r_10.empty:
                    df_print = df_r_10[["市场环境", "信号数量", "胜率", "平均收益", "盈亏比", "统计显著"]]
                    print(df_print.to_string(index=False))

    # ─────────────────────────────────────────────────────────────
    # 保存报告
    # ─────────────────────────────────────────────────────────────
    def save_report(self, report: dict, path: str = "output/reliability_report.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        all_rows = []
        for key, val in report.items():
            if isinstance(val, pd.DataFrame) and not val.empty:
                val = val.copy()
                val["report_key"] = key
                all_rows.append(val)
        if all_rows:
            pd.concat(all_rows, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  可靠性报告已保存：{path}")
