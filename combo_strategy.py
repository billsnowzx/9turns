"""
combo_strategy.py — 组合策略增强模块

测试以下过滤条件与九转神奇信号的组合效果：
  1. 趋势过滤    : MA20/MA50/MA200 方向
  2. RSI 超买超卖 : RSI < 35 或 RSI > 65
  3. 成交量放大   : 信号日成交量 > N日均量的 1.5 倍
  4. 布林带位置   : 价格在下轨附近（买9）/ 上轨附近（卖9）
  5. 综合过滤     : 条件①②③ 全满足

每种组合生成 entry/exit 信号序列，供 Backtester 回测。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class ComboStrategy:
    """
    在原始九转信号基础上叠加不同过滤条件，比较增强效果。
    """

    def __init__(self, price_df: pd.DataFrame, signals_df: pd.DataFrame):
        self.df         = price_df.copy()
        self.signals_df = signals_df.copy()
        self.close      = price_df["close"]
        self.volume     = price_df["volume"]
        self._compute_indicators()

    # ─────────────────────────────────────────────────────────────
    # 技术指标计算
    # ─────────────────────────────────────────────────────────────
    def _compute_indicators(self):
        df = self.df
        c  = self.close

        # 均线
        df["ma20"]  = c.rolling(20).mean()
        df["ma50"]  = c.rolling(50).mean()
        df["ma200"] = c.rolling(200, min_periods=60).mean()
        df["ema20"] = c.ewm(span=20, adjust=False).mean()

        # RSI（14日）
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - 100 / (1 + rs)

        # 布林带（20日，2σ）
        df["bb_mid"]   = c.rolling(20).mean()
        bb_std         = c.rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std
        df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])  # 0~1

        # 成交量比值（相对20日均量）
        df["vol_ratio"] = self.volume / self.volume.rolling(20).mean()

        # MACD（12-26-9）
        ema12   = c.ewm(span=12, adjust=False).mean()
        ema26   = c.ewm(span=26, adjust=False).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # ATR（用于止损计算参考）
        high, low = df["high"], df["low"]
        tr = pd.concat([
            high - low,
            (high - c.shift()).abs(),
            (low  - c.shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14).mean()

        self.df = df

    # ─────────────────────────────────────────────────────────────
    # 辅助：将信号列表转为布尔 Series
    # ─────────────────────────────────────────────────────────────
    def _signals_to_series(
        self, signal_type: str, extra_mask: pd.Series = None
    ) -> pd.Series:
        """
        返回 DatetimeIndex 对齐的布尔 Series。
        extra_mask 为额外过滤条件（在价格 DataFrame 上的布尔 Series）。
        """
        entries = pd.Series(False, index=self.df.index)
        sigs = self.signals_df[self.signals_df["signal"] == signal_type]
        for date in sigs["date"]:
            if date in entries.index:
                entries[date] = True

        if extra_mask is not None:
            entries = entries & extra_mask

        # Prevent lookahead bias: signal on T can only be traded from T+1.
        entries = entries.shift(1).fillna(False).astype(bool)
        return entries

    # ─────────────────────────────────────────────────────────────
    # 各过滤条件定义
    # ─────────────────────────────────────────────────────────────
    def _mask_trend(self, direction: str, ma: str = "ma200") -> pd.Series:
        """顺势过滤：买9时价格在均线之上，卖9时在均线之下"""
        if direction == "buy":
            return self.close > self.df[ma]
        else:
            return self.close < self.df[ma]

    def _mask_rsi(self, direction: str,
                  buy_thresh: float = 35, sell_thresh: float = 65) -> pd.Series:
        """RSI 超买超卖过滤"""
        if direction == "buy":
            return self.df["rsi"] < buy_thresh
        else:
            return self.df["rsi"] > sell_thresh

    def _mask_volume(self, ratio: float = 1.5) -> pd.Series:
        """成交量放大过滤"""
        return self.df["vol_ratio"] > ratio

    def _mask_bollinger(self, direction: str,
                        buy_pct: float = 0.25, sell_pct: float = 0.75) -> pd.Series:
        """布林带位置过滤（买9时接近下轨，卖9时接近上轨）"""
        if direction == "buy":
            return self.df["bb_pct"] < buy_pct
        else:
            return self.df["bb_pct"] > sell_pct

    def _mask_macd_hist(self, direction: str) -> pd.Series:
        """MACD 柱方向过滤（辅助动能确认）"""
        if direction == "buy":
            return self.df["macd_hist"] > self.df["macd_hist"].shift(1)  # 柱在走强
        else:
            return self.df["macd_hist"] < self.df["macd_hist"].shift(1)

    # ─────────────────────────────────────────────────────────────
    # 简单持仓退出：N日后平仓（用于模拟回测的 exit 信号）
    # ─────────────────────────────────────────────────────────────
    def _make_exits(self, entries: pd.Series, hold_days: int = 10) -> pd.Series:
        """在 entry N 日后强制退出"""
        exits = pd.Series(False, index=entries.index)
        idx_list = entries[entries].index.tolist()
        for entry_date in idx_list:
            pos = entries.index.get_loc(entry_date)
            exit_pos = min(pos + hold_days, len(exits) - 1)
            exits.iloc[exit_pos] = True
        return exits

    # ─────────────────────────────────────────────────────────────
    # 运行所有组合
    # ─────────────────────────────────────────────────────────────
    def run_all_combos(self, hold_days: int = 10) -> List[Dict]:
        """
        对买9信号（做多）测试以下 6 种组合：
          0. 纯信号（基准）
          1. + 趋势过滤（MA200）
          2. + RSI 过滤
          3. + 成交量过滤
          4. + 布林带过滤
          5. + 综合（趋势 + RSI + 成交量）

        Returns list of dicts：
          name, entries（Series), exits（Series), n_signals
        """
        direction = "buy"   # 这里只分析做多方向；卖9同理可复用
        sig_type  = f"{direction}9"
        df        = self.df
        results   = []

        combo_defs = [
            ("纯买9信号（基准）",          None),
            ("买9 + MA200趋势过滤",        self._mask_trend(direction, "ma200")),
            ("买9 + MA50趋势过滤",         self._mask_trend(direction, "ma50")),
            ("买9 + RSI<35过滤",           self._mask_rsi(direction, buy_thresh=35)),
            ("买9 + 成交量放大过滤",        self._mask_volume(1.5)),
            ("买9 + 布林带下轨过滤",        self._mask_bollinger(direction)),
            ("买9 + MACD柱增强确认",        self._mask_macd_hist(direction)),
            ("买9 + 综合过滤（趋势+RSI+量)", self._mask_trend(direction) & self._mask_rsi(direction) & self._mask_volume(1.5)),
        ]

        for name, mask in combo_defs:
            entries = self._signals_to_series(sig_type, extra_mask=mask)
            exits   = self._make_exits(entries, hold_days=hold_days)
            n_sigs  = entries.sum()

            # 计算简单胜率（不依赖回测，快速预筛）
            win_rate, avg_ret, profit_factor = self._quick_stats(entries, n_days=hold_days)

            results.append({
                "name":          name,
                "entries":       entries,
                "exits":         exits,
                "n_signals":     int(n_sigs),
                "win_rate":      win_rate,
                "avg_ret":       avg_ret,
                "profit_factor": profit_factor,
            })

        return results

    # ─────────────────────────────────────────────────────────────
    # 快速统计（不依赖回测引擎，直接计算前向收益）
    # ─────────────────────────────────────────────────────────────
    def _quick_stats(self, entries: pd.Series, n_days: int = 10) -> Tuple[float, float, float]:
        close_arr  = self.close.values
        entry_idxs = np.where(entries.values)[0]
        rets = []
        for idx in entry_idxs:
            fwd = idx + n_days
            if fwd < len(close_arr):
                ret = (close_arr[fwd] - close_arr[idx]) / close_arr[idx]
                rets.append(ret)
        if not rets:
            return 0.0, 0.0, 0.0
        rets = np.array(rets)
        win_rate = (rets > 0).mean()
        avg_ret  = rets.mean()
        gains    = rets[rets > 0].mean() if (rets > 0).any() else 0
        losses   = abs(rets[rets < 0].mean()) if (rets < 0).any() else 1e-9
        pf       = gains / losses
        return round(win_rate, 4), round(avg_ret, 4), round(pf, 2)

    # ─────────────────────────────────────────────────────────────
    # 打印摘要
    # ─────────────────────────────────────────────────────────────
    def print_summary(self, results: List[Dict]):
        print(f"\n  {'策略名称':<28} {'信号数':>6} {'胜率':>7} {'平均收益':>9} {'盈亏比':>7}")
        print("  " + "-" * 65)
        for r in results:
            mark = " ◀ 最优" if r.get("_best") else ""
            print(f"  {r['name']:<28} {r['n_signals']:>6} "
                  f"{r['win_rate']:>7.1%} {r['avg_ret']:>9.2%} {r['profit_factor']:>7.2f}{mark}")

    # ─────────────────────────────────────────────────────────────
    # 获取最优组合
    # ─────────────────────────────────────────────────────────────
    def get_best_combo(
        self,
        results: List[Dict],
        lag: int = 1,
        sl_stop: float = 0.05,
        tp_stop: float = 0.15,
        fees: float = 0.001,
    ) -> Dict:
        """
        以 胜率 × 盈亏比 × log(信号数+1) 为综合得分，筛选最优。
        至少需要 10 个信号，避免小样本偏差。
        """
        from backtester import Backtester

        bt = Backtester(self.df)
        enriched = []
        for r in results:
            bt_result = bt.run(
                entry_signals=r["entries"],
                exit_signals=r["exits"],
                lag=lag,
                sl_stop=sl_stop,
                tp_stop=tp_stop,
                fees=fees,
                strategy_name=r["name"],
            )
            r["train_sharpe"] = bt_result.get("sharpe", np.nan)
            r["train_calmar"] = bt_result.get("calmar", np.nan)
            r["train_max_dd"] = bt_result.get("max_drawdown", np.nan)
            enriched.append(r)

        valid = [r for r in enriched if r["n_signals"] >= 10]
        if not valid:
            valid = enriched

        scored = sorted(
            valid,
            key=lambda r: (-np.inf if pd.isna(r.get("train_calmar")) else r.get("train_calmar")),
            reverse=True,
        )
        best = scored[0]
        best["_best"] = True
        return best

    # ─────────────────────────────────────────────────────────────
    # 保存报告
    # ─────────────────────────────────────────────────────────────
    def save_report(self, results: List[Dict], path: str = "output/combo_report.csv"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = [{k: v for k, v in r.items()
                 if not isinstance(v, pd.Series)} for r in results]
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  组合策略报告已保存：{path}")
