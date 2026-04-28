"""
backtester.py — 回测引擎模块

功能：
  - 基于 vectorbt 做向量化回测（快速、准确）
  - 计算完整绩效指标：年化收益、夏普比率、最大回撤、卡玛比率等
  - 生成 Plotly 交互式回测图表（净值曲线 + 买卖点标注）
  - 与 Buy & Hold 基准策略对比

如果 vectorbt 未安装，自动退回到手写简版回测引擎。
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Optional


class Backtester:
    """
    回测器，支持 vectorbt（高精度）和内置简版（兜底）。

    Parameters
    ----------
    price_df : OHLCV DataFrame
    init_cash: 初始资金（默认 100 万）
    """

    def __init__(self, price_df: pd.DataFrame, init_cash: float = 1_000_000):
        self.df        = price_df.copy()
        self.close     = price_df["close"]
        self.init_cash = init_cash

    # ─────────────────────────────────────────────────────────────
    # 主运行入口
    # ─────────────────────────────────────────────────────────────
    def run(
        self,
        entry_signals: pd.Series,
        exit_signals:  pd.Series,
        lag: int = 1,
        sl_stop: float = 0.05,     # 止损比例
        tp_stop: float = 0.15,     # 止盈比例
        fees:    float = 0.001,    # 手续费（单边）
        strategy_name: str = "策略",
    ) -> Dict:
        """
        执行回测，返回绩效指标字典。
        信号在 T 日生成，T+lag 日执行。
        """
        if lag < 0:
            raise ValueError("lag must be >= 0")

        entry_signals = entry_signals.reindex(self.close.index).fillna(False).astype(bool)
        exit_signals = exit_signals.reindex(self.close.index).fillna(False).astype(bool)
        if lag > 0:
            entry_signals = entry_signals.shift(lag).fillna(False).astype(bool)
            exit_signals = exit_signals.shift(lag).fillna(False).astype(bool)

        try:
            return self._run_vectorbt(
                entry_signals, exit_signals, sl_stop, tp_stop, fees, strategy_name
            )
        except Exception as e:
            print(f"  vectorbt 不可用（{e}），使用内置回测引擎...")
            return self._run_simple(
                entry_signals, exit_signals, sl_stop, tp_stop, fees, strategy_name
            )

    # ─────────────────────────────────────────────────────────────
    # vectorbt 回测
    # ─────────────────────────────────────────────────────────────
    def _run_vectorbt(
        self, entries, exits, sl_stop, tp_stop, fees, name
    ) -> Dict:
        import vectorbt as vbt

        pf = vbt.Portfolio.from_signals(
            self.close,
            entries=entries,
            exits=exits,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            fees=fees,
            init_cash=self.init_cash,
            freq="D",
        )

        stats = pf.stats()
        equity = pf.value()

        # Buy & Hold 基准
        bh = vbt.Portfolio.from_holding(
            self.close, init_cash=self.init_cash, fees=fees, freq="D"
        )
        bh_stats = bh.stats()

        result = {
            "name":             name,
            "engine":           "vectorbt",
            "portfolio":        pf,
            "equity":           equity,
            "bh_equity":        bh.value(),
            "stats":            stats,
            "bh_stats":         bh_stats,
            # 核心指标
            "total_return":     stats.get("Total Return [%]", np.nan) / 100,
            "annual_return":    stats.get("Annualized Return [%]", np.nan) / 100,
            "sharpe":           stats.get("Sharpe Ratio", np.nan),
            "max_drawdown":     stats.get("Max Drawdown [%]", np.nan) / 100,
            "win_rate":         stats.get("Win Rate [%]", np.nan) / 100,
            "total_trades":     stats.get("Total Trades", np.nan),
            "bh_total_return":  bh_stats.get("Total Return [%]", np.nan) / 100,
            "bh_annual_return": bh_stats.get("Annualized Return [%]", np.nan) / 100,
            "bh_sharpe":        bh_stats.get("Sharpe Ratio", np.nan),
            "bh_max_drawdown":  bh_stats.get("Max Drawdown [%]", np.nan) / 100,
        }

        # 卡玛比率
        mdd = abs(result["max_drawdown"])
        result["calmar"] = result["annual_return"] / mdd if mdd > 0 else np.inf

        return result

    # ─────────────────────────────────────────────────────────────
    # 内置简版回测引擎（vectorbt 不可用时的兜底）
    # ─────────────────────────────────────────────────────────────
    def _run_simple(
        self, entries, exits, sl_stop, tp_stop, fees, name
    ) -> Dict:
        """
        每次信号：全仓买入，N日后止盈/止损/强制退出。
        """
        close   = self.close.values
        dates   = self.close.index
        n       = len(close)
        cash    = self.init_cash
        shares  = 0
        in_pos  = False
        entry_price = 0
        equity_curve = np.zeros(n)
        trades   = []

        entry_arr = entries.reindex(self.close.index).fillna(False).values
        exit_arr  = exits.reindex(self.close.index).fillna(False).values

        for i in range(n):
            price = close[i]

            if in_pos:
                # 止损 / 止盈
                ret = (price - entry_price) / entry_price
                force_exit = ret <= -sl_stop or ret >= tp_stop
                if exit_arr[i] or force_exit:
                    proceeds = shares * price * (1 - fees)
                    trades.append({
                        "exit_date":  dates[i],
                        "exit_price": price,
                        "return":     ret,
                        "win":        ret > 0,
                    })
                    cash    = proceeds
                    shares  = 0
                    in_pos  = False

            if not in_pos and entry_arr[i]:
                cost    = cash * (1 - fees)
                shares  = cost / price
                entry_price = price
                in_pos  = True

            equity_curve[i] = cash + shares * price

        equity = pd.Series(equity_curve, index=dates)

        # 计算绩效指标
        daily_ret = equity.pct_change().dropna()
        total_ret = (equity.iloc[-1] - self.init_cash) / self.init_cash

        years = (dates[-1] - dates[0]).days / 365.25
        annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if daily_ret.std() > 0 else 0)

        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / rolling_max
        max_dd      = drawdown.min()

        calmar = annual_ret / abs(max_dd) if max_dd < 0 else np.inf

        # B&H 指标
        bh_total = (close[-1] - close[0]) / close[0]
        bh_annual = (1 + bh_total) ** (1 / years) - 1 if years > 0 else 0
        bh_eq = pd.Series(close / close[0] * self.init_cash, index=dates)
        bh_dd = ((bh_eq - bh_eq.cummax()) / bh_eq.cummax()).min()

        trades_df = pd.DataFrame(trades)
        win_rate  = trades_df["win"].mean() if not trades_df.empty else 0

        return {
            "name":             name,
            "engine":           "simple",
            "equity":           equity,
            "bh_equity":        bh_eq,
            "total_return":     total_ret,
            "annual_return":    annual_ret,
            "sharpe":           sharpe,
            "max_drawdown":     max_dd,
            "calmar":           calmar,
            "win_rate":         win_rate,
            "total_trades":     len(trades),
            "bh_total_return":  bh_total,
            "bh_annual_return": bh_annual,
            "bh_sharpe":        0,
            "bh_max_drawdown":  bh_dd,
            "trades_df":        trades_df,
        }

    # ─────────────────────────────────────────────────────────────
    # 打印绩效摘要
    # ─────────────────────────────────────────────────────────────
    def print_stats(self, result: Dict):
        def fmt(v, pct=True):
            if isinstance(v, float) and np.isnan(v):
                return "N/A"
            return f"{v:.2%}" if pct else f"{v:.2f}"

        print(f"\n  ── 回测结果：{result['name']} ──")
        print(f"  {'指标':<18} {'策略':>12} {'买入持有':>12}")
        print("  " + "-" * 45)
        rows = [
            ("总收益率",     result["total_return"],     result["bh_total_return"],     True),
            ("年化收益率",   result["annual_return"],     result["bh_annual_return"],    True),
            ("夏普比率",     result["sharpe"],            result["bh_sharpe"],           False),
            ("最大回撤",     result["max_drawdown"],      result["bh_max_drawdown"],     True),
            ("卡玛比率",     result.get("calmar", 0),     None,                          False),
            ("胜率",         result["win_rate"],          None,                          True),
            ("总交易次数",   result["total_trades"],      None,                          False),
        ]
        for label, strat, bh, is_pct in rows:
            s_str = fmt(strat, is_pct)
            b_str = fmt(bh, is_pct) if bh is not None else "—"
            print(f"  {label:<18} {s_str:>12} {b_str:>12}")

    # ─────────────────────────────────────────────────────────────
    # 生成 Plotly 交互图表
    # ─────────────────────────────────────────────────────────────
    def plot(self, result: Dict, save_path: str = "output/backtest_chart.html"):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("  请先安装 plotly: pip install plotly")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        equity    = result["equity"]
        bh_equity = result["bh_equity"]
        close     = self.close

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.3, 0.2],
            subplot_titles=["净值曲线对比", "价格走势", "回撤曲线"],
            vertical_spacing=0.07,
        )

        # ── Row 1: 净值曲线 ──
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity / self.init_cash,
            name=result["name"], line=dict(color="#e74c3c", width=2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bh_equity.index, y=bh_equity / self.init_cash,
            name="买入持有", line=dict(color="#95a5a6", width=1.5, dash="dash"),
        ), row=1, col=1)
        fig.add_hline(y=1, line_dash="dot", line_color="gray", row=1, col=1)

        # ── Row 2: 价格走势（K线简化为收盘价折线）──
        fig.add_trace(go.Scatter(
            x=close.index, y=close,
            name="收盘价", line=dict(color="#2980b9", width=1),
            showlegend=False,
        ), row=2, col=1)

        # ── Row 3: 回撤曲线 ──
        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / rolling_max * 100
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown,
            name="回撤", fill="tozeroy",
            line=dict(color="#e74c3c", width=1),
            fillcolor="rgba(231,76,60,0.2)",
            showlegend=False,
        ), row=3, col=1)

        # ── 布局 ──
        annual  = result["annual_return"]
        sharpe  = result["sharpe"]
        max_dd  = result["max_drawdown"]
        n_trade = result["total_trades"]
        title = (
            f"{result['name']} | "
            f"年化:{annual:.1%}  夏普:{sharpe:.2f}  "
            f"最大回撤:{max_dd:.1%}  交易次数:{n_trade}"
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            height=750,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )
        fig.update_yaxes(title_text="净值", row=1, col=1)
        fig.update_yaxes(title_text="价格", row=2, col=1)
        fig.update_yaxes(title_text="回撤%", row=3, col=1)

        fig.write_html(save_path)
        print(f"  回测图表已保存：{save_path}")
