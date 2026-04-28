"""
signal_detector.py — 九转神奇信号识别模块

实现：
  - Setup Phase：连续9根K线计数（买9 / 卖9）
  - Countdown Phase：在Setup完成后继续计数到13（买13 / 卖13）
  - 信号可视化（mplfinance K线图标注）
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


class TDSequential:
    """
    TD Sequential 信号检测器。

    Parameters
    ----------
    df : DataFrame，必须含 open/high/low/close/volume 列
    """

    def __init__(self, df: pd.DataFrame, countdown_mode: str = "simplified", config: dict | None = None):
        """
        Parameters
        ----------
        df : OHLCV DataFrame
        countdown_mode : "simplified" | "strict"
            simplified: count when price hits basic Countdown condition.
            strict: adds qualifier and cancellation checks, so signals are fewer.
        """
        self.config = config or {}
        countdown_mode = self.config.get("countdown_mode", countdown_mode)
        if countdown_mode not in ("simplified", "strict"):
            raise ValueError("countdown_mode must be 'simplified' or 'strict'")
        self.df = df.copy()
        self.close = df["close"].values
        self.dates = df.index
        self.n = len(df)
        self.countdown_mode = countdown_mode

    # ─────────────────────────────────────────────────────────────
    # 公共入口
    # ─────────────────────────────────────────────────────────────
    def detect_all(self, cooldown_bars: int = 4) -> pd.DataFrame:
        """
        识别所有 Setup 信号（买9 / 卖9）和 Countdown 信号（买13 / 卖13）。

        Returns
        -------
        DataFrame，列：date, signal, bar_index, close_at_signal
            signal 取值："buy9" | "sell9" | "buy13" | "sell13"
        """
        setup_records   = self._detect_setup()
        countdown_records = self._detect_countdown(setup_records)
        all_records = setup_records + countdown_records
        all_records = self._apply_cooldown(all_records, cooldown_bars=cooldown_bars)
        all_records.sort(key=lambda x: x[0])   # 按 bar_index 排序

        if not all_records:
            return pd.DataFrame(columns=["date", "signal", "bar_index", "close_at_signal"])

        result = pd.DataFrame(all_records, columns=["bar_index", "signal"])
        result["bar_index"] = result["bar_index"].astype(int)
        result["date"] = self.dates[result["bar_index"]]
        result["close_at_signal"] = self.close[result["bar_index"]]
        result = result[["date", "signal", "bar_index", "close_at_signal"]]
        return result.reset_index(drop=True)

    def _apply_cooldown(
        self, records: List[Tuple[int, str]], cooldown_bars: int
    ) -> List[Tuple[int, str]]:
        if cooldown_bars <= 0:
            return records

        filtered: List[Tuple[int, str]] = []
        last_bar_by_direction = {"buy": -10**9, "sell": -10**9}
        for bar_idx, signal in sorted(records, key=lambda x: x[0]):
            direction = "buy" if signal.startswith("buy") else "sell"
            if bar_idx - last_bar_by_direction[direction] <= cooldown_bars:
                continue
            filtered.append((bar_idx, signal))
            last_bar_by_direction[direction] = bar_idx
        return filtered

    # ─────────────────────────────────────────────────────────────
    # Setup Phase（核心逻辑）
    # ─────────────────────────────────────────────────────────────
    def _detect_setup(self) -> List[Tuple[int, str]]:
        """
        规则：
          买9 - 连续9根K线满足 close[i] < close[i-4]
          卖9 - 连续9根K线满足 close[i] > close[i-4]
        当计数达到9时记录信号，并且允许继续计数（不重置）。
        """
        close = self.close
        records = []
        buy_count = 0
        sell_count = 0

        for i in range(4, self.n):
            # 买入计数
            if close[i] < close[i - 4]:
                buy_count += 1
                sell_count = 0
            # 卖出计数
            elif close[i] > close[i - 4]:
                sell_count += 1
                buy_count = 0
            else:
                buy_count = 0
                sell_count = 0

            if buy_count == 9:
                records.append((i, "buy9"))
                # 不重置计数，允许形成 10, 11... 但标准只记录 9
                buy_count = 0

            if sell_count == 9:
                records.append((i, "sell9"))
                sell_count = 0

        return records

    # ─────────────────────────────────────────────────────────────
    # Countdown Phase（在 Setup 完成后继续计数到 13）
    # ─────────────────────────────────────────────────────────────
    def _detect_countdown(self, setup_records: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        """
        Countdown 规则（简化版 DeMark）：
          买13 - 在 buy9 完成后，从下一根开始，close <= low[i-2] 计数到 13
          卖13 - 在 sell9 完成后，close >= high[i-2] 计数到 13
        """
        close = self.close
        high  = self.df["high"].values
        low   = self.df["low"].values
        records = []

        for setup_bar, setup_type in setup_records:
            if setup_bar + 2 >= self.n:
                continue

            direction = "buy" if setup_type == "buy9" else "sell"
            count = 0
            setup_close = close[setup_bar]
            setup_high = high[setup_bar]
            setup_low = low[setup_bar]
            for i in range(setup_bar + 1, self.n):
                if i < 2:
                    continue

                if direction == "buy":
                    hit = close[i] <= low[i - 2]
                    # strict cancel: upside reversal invalidates current countdown
                    if self.countdown_mode == "strict":
                        if close[i] > setup_high:
                            break
                        if i >= 4 and close[i] > close[i - 4]:
                            break
                    if hit:
                        if self.countdown_mode == "strict":
                            # strict qualifier: downside should persist vs setup bar
                            if close[i] <= setup_close and low[i] <= setup_low:
                                count += 1
                            else:
                                count = 0
                        else:
                            count += 1
                    elif self.countdown_mode == "strict":
                        count = 0
                else:
                    hit = close[i] >= high[i - 2]
                    # strict cancel: downside reversal invalidates current countdown
                    if self.countdown_mode == "strict":
                        if close[i] < setup_low:
                            break
                        if i >= 4 and close[i] < close[i - 4]:
                            break
                    if hit:
                        if self.countdown_mode == "strict":
                            # strict qualifier: upside should persist vs setup bar
                            if close[i] >= setup_close and high[i] >= setup_high:
                                count += 1
                            else:
                                count = 0
                        else:
                            count += 1
                    elif self.countdown_mode == "strict":
                        count = 0

                if count == 13:
                    records.append((i, f"{direction}13"))
                    break

        return records

    # ─────────────────────────────────────────────────────────────
    # 计数序列（用于可视化，返回每根K线对应的计数值）
    # ─────────────────────────────────────────────────────────────
    def get_count_series(self) -> pd.DataFrame:
        """
        返回每根K线的买入计数和卖出计数，用于可视化标注。
        """
        close = self.close
        buy_counts  = np.zeros(self.n, dtype=int)
        sell_counts = np.zeros(self.n, dtype=int)
        bc = sc = 0

        for i in range(4, self.n):
            if close[i] < close[i - 4]:
                bc += 1
                sc = 0
            elif close[i] > close[i - 4]:
                sc += 1
                bc = 0
            else:
                bc = sc = 0

            buy_counts[i]  = bc if bc <= 9 else 0
            sell_counts[i] = sc if sc <= 9 else 0

        return pd.DataFrame(
            {"buy_count": buy_counts, "sell_count": sell_counts},
            index=self.dates
        )

    # ─────────────────────────────────────────────────────────────
    # 可视化：K线图 + 信号标注
    # ─────────────────────────────────────────────────────────────
    def plot(
        self,
        signals_df: pd.DataFrame,
        last_n: int = 200,
        save_path: str = None,
        title: str = "九转神奇信号",
    ):
        """
        绘制最近 last_n 根K线的信号图。
        save_path 不为 None 时保存为图片。
        """
        try:
            import mplfinance as mpf
            import matplotlib.pyplot as plt
        except ImportError:
            print("  请先安装 mplfinance: pip install mplfinance")
            return

        plot_df = self.df.iloc[-last_n:].copy()
        sig_sub = signals_df[signals_df["date"] >= plot_df.index[0]]

        # 构建信号标记序列
        buy9_marker  = pd.Series(np.nan, index=plot_df.index)
        sell9_marker = pd.Series(np.nan, index=plot_df.index)

        for _, row in sig_sub.iterrows():
            if row["signal"] == "buy9" and row["date"] in plot_df.index:
                buy9_marker[row["date"]] = plot_df.loc[row["date"], "low"] * 0.985
            elif row["signal"] == "sell9" and row["date"] in plot_df.index:
                sell9_marker[row["date"]] = plot_df.loc[row["date"], "high"] * 1.015

        addplots = [
            mpf.make_addplot(buy9_marker,  type="scatter", marker="^",
                             markersize=100, color="red",  panel=0),
            mpf.make_addplot(sell9_marker, type="scatter", marker="v",
                             markersize=100, color="green", panel=0),
        ]

        fig, axes = mpf.plot(
            plot_df, type="candle", volume=True,
            addplot=addplots, style="charles",
            title=title, returnfig=True,
            figratio=(16, 9), figscale=1.2,
        )

        # 添加图例
        axes[0].plot([], [], "r^", markersize=8, label="买9信号")
        axes[0].plot([], [], "gv", markersize=8, label="卖9信号")
        axes[0].legend(loc="upper left", fontsize=9)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  信号图已保存：{save_path}")
        else:
            plt.show()
        plt.close()
