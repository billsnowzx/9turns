import os
from typing import Dict

import numpy as np
import pandas as pd


class Backtester:
    def __init__(self, price_df: pd.DataFrame, init_cash: float = 1_000_000, config: dict | None = None):
        self.config = config or {}
        self.df = price_df.copy()
        self.close = price_df["close"]
        self.init_cash = self.config.get("init_cash", init_cash)

    def run(
        self,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        lag: int | None = None,
        sl_stop: float | None = None,
        tp_stop: float | None = None,
        fees: float | None = None,
        strategy_name: str = "strategy",
    ) -> Dict:
        backtest_cfg = self.config.get("backtest", {})
        lag = backtest_cfg.get("lag", 1) if lag is None else lag
        sl_stop = backtest_cfg.get("sl_stop", 0.05) if sl_stop is None else sl_stop
        tp_stop = backtest_cfg.get("tp_stop", 0.15) if tp_stop is None else tp_stop
        fees = backtest_cfg.get("fees", 0.001) if fees is None else fees

        if lag < 0:
            raise ValueError("lag must be >= 0")

        entry_signals = entry_signals.reindex(self.close.index).fillna(False).astype(bool)
        exit_signals = exit_signals.reindex(self.close.index).fillna(False).astype(bool)
        if lag > 0:
            entry_signals = entry_signals.shift(lag).fillna(False).astype(bool)
            exit_signals = exit_signals.shift(lag).fillna(False).astype(bool)

        engine = backtest_cfg.get("engine", "simple")
        if engine == "vectorbt":
            try:
                result = self._run_vectorbt(entry_signals, exit_signals, sl_stop, tp_stop, fees, strategy_name)
            except Exception:
                result = self._run_simple(entry_signals, exit_signals, sl_stop, tp_stop, fees, strategy_name)
        else:
            result = self._run_simple(entry_signals, exit_signals, sl_stop, tp_stop, fees, strategy_name)
        result.update(self._metrics_from_equity(result["equity"]))
        result["entries"] = entry_signals
        result["exits"] = exit_signals
        return result

    def _metrics_from_equity(self, equity: pd.Series) -> Dict:
        equity = equity.dropna()
        if equity.empty:
            return {
                "total_return": np.nan,
                "annual_return": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "calmar": np.nan,
            }

        daily_ret = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        total_ret = (equity.iloc[-1] - self.init_cash) / self.init_cash

        years = (equity.index[-1] - equity.index[0]).days / 365.25
        annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else np.nan
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0

        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min()
        calmar = annual_ret / abs(max_dd) if pd.notna(annual_ret) and max_dd < 0 else np.inf

        return {
            "total_return": float(total_ret),
            "annual_return": float(annual_ret) if pd.notna(annual_ret) else np.nan,
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "calmar": float(calmar) if np.isfinite(calmar) else np.inf,
        }

    def _run_vectorbt(self, entries, exits, sl_stop, tp_stop, fees, name) -> Dict:
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
        bh = vbt.Portfolio.from_holding(self.close, init_cash=self.init_cash, fees=fees, freq="D")
        bh_stats = bh.stats()

        result = {
            "name": name,
            "engine": "vectorbt",
            "portfolio": pf,
            "equity": equity,
            "bh_equity": bh.value(),
            "stats": stats,
            "bh_stats": bh_stats,
            "total_return": stats.get("Total Return [%]", np.nan) / 100,
            "annual_return": stats.get("Annualized Return [%]", np.nan) / 100,
            "sharpe": stats.get("Sharpe Ratio", np.nan),
            "max_drawdown": stats.get("Max Drawdown [%]", np.nan) / 100,
            "win_rate": stats.get("Win Rate [%]", np.nan) / 100,
            "total_trades": stats.get("Total Trades", np.nan),
            "bh_total_return": bh_stats.get("Total Return [%]", np.nan) / 100,
            "bh_annual_return": bh_stats.get("Annualized Return [%]", np.nan) / 100,
            "bh_sharpe": bh_stats.get("Sharpe Ratio", np.nan),
            "bh_max_drawdown": bh_stats.get("Max Drawdown [%]", np.nan) / 100,
        }
        mdd = abs(result["max_drawdown"])
        result["calmar"] = result["annual_return"] / mdd if mdd > 0 else np.inf
        return result

    def _run_simple(self, entries, exits, sl_stop, tp_stop, fees, name) -> Dict:
        close = self.close.values
        dates = self.close.index
        n = len(close)
        cash = self.init_cash
        shares = 0.0
        in_pos = False
        entry_price = 0.0
        equity_curve = np.zeros(n)
        trades = []

        entry_arr = entries.reindex(self.close.index).fillna(False).values
        exit_arr = exits.reindex(self.close.index).fillna(False).values

        for i in range(n):
            price = close[i]

            if in_pos:
                ret = (price - entry_price) / entry_price
                force_exit = ret <= -sl_stop or ret >= tp_stop
                if exit_arr[i] or force_exit:
                    proceeds = shares * price * (1 - fees)
                    trades.append({"exit_date": dates[i], "exit_price": price, "return": ret, "win": ret > 0})
                    cash = proceeds
                    shares = 0.0
                    in_pos = False

            if (not in_pos) and entry_arr[i]:
                cost = cash * (1 - fees)
                shares = cost / price
                entry_price = price
                in_pos = True

            equity_curve[i] = cash + shares * price

        equity = pd.Series(equity_curve, index=dates)
        daily_ret = equity.pct_change().dropna()
        total_ret = (equity.iloc[-1] - self.init_cash) / self.init_cash
        years = (dates[-1] - dates[0]).days / 365.25
        annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0)

        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_dd = drawdown.min()
        calmar = annual_ret / abs(max_dd) if max_dd < 0 else np.inf

        bh_total = (close[-1] - close[0]) / close[0]
        bh_annual = (1 + bh_total) ** (1 / years) - 1 if years > 0 else 0
        bh_eq = pd.Series(close / close[0] * self.init_cash, index=dates)
        bh_dd = ((bh_eq - bh_eq.cummax()) / bh_eq.cummax()).min()

        trades_df = pd.DataFrame(trades)
        win_rate = trades_df["win"].mean() if not trades_df.empty else 0

        return {
            "name": name,
            "engine": "simple",
            "equity": equity,
            "bh_equity": bh_eq,
            "total_return": total_ret,
            "annual_return": annual_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "win_rate": win_rate,
            "total_trades": len(trades),
            "bh_total_return": bh_total,
            "bh_annual_return": bh_annual,
            "bh_sharpe": 0,
            "bh_max_drawdown": bh_dd,
            "trades_df": trades_df,
        }

    def print_stats(self, result: Dict):
        print(f"{result['name']} return={result['total_return']:.2%} sharpe={result['sharpe']:.2f} mdd={result['max_drawdown']:.2%}")

    def plot(self, result: Dict, save_path: str = "output/backtest_chart.html", split_date=None):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        equity = result["equity"]
        bh_equity = result["bh_equity"]
        close = self.close

        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.4, 0.25, 0.15, 0.2],
            vertical_spacing=0.06,
            subplot_titles=("Equity", "Price with Signals", "Drawdown", "Monthly Return Heatmap"),
        )
        fig.add_trace(go.Scatter(x=equity.index, y=equity / self.init_cash, name=result["name"]), row=1, col=1)
        fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity / self.init_cash, name="buy&hold"), row=1, col=1)
        fig.add_trace(go.Scatter(x=close.index, y=close, name="close", showlegend=False), row=2, col=1)

        entries = result.get("entries")
        exits = result.get("exits")
        if entries is not None:
            entry_idx = entries[entries].index
            if len(entry_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=entry_idx,
                        y=close.reindex(entry_idx),
                        mode="markers",
                        marker=dict(symbol="triangle-up", color="green", size=9),
                        name="buy",
                    ),
                    row=2,
                    col=1,
                )
        if exits is not None:
            exit_idx = exits[exits].index
            if len(exit_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=exit_idx,
                        y=close.reindex(exit_idx),
                        mode="markers",
                        marker=dict(symbol="triangle-down", color="red", size=9),
                        name="sell",
                    ),
                    row=2,
                    col=1,
                )

        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="drawdown", fill="tozeroy", showlegend=False), row=3, col=1)

        monthly = equity.pct_change().resample("M").apply(lambda s: (1 + s).prod() - 1)
        if not monthly.empty:
            heat = monthly.to_frame(name="ret")
            heat["year"] = heat.index.year
            heat["month"] = heat.index.month
            pivot = heat.pivot(index="year", columns="month", values="ret")
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=[int(c) for c in pivot.columns],
                    y=[int(i) for i in pivot.index],
                    colorscale="RdYlGn",
                    zmid=0,
                    name="monthly",
                    showscale=True,
                ),
                row=4,
                col=1,
            )

        if split_date is not None:
            fig.add_vline(x=split_date, line_dash="dash", line_color="black")

        fig.update_layout(template="plotly_white", height=1000)
        fig.write_html(save_path)
