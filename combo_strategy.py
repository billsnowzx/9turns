from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class ComboStrategy:
    def __init__(self, price_df: pd.DataFrame, signals_df: pd.DataFrame, config: dict | None = None):
        self.config = config or {}
        self.df = price_df.copy()
        self.signals_df = signals_df.copy()
        self.close = price_df["close"]
        self.volume = price_df["volume"]
        self._compute_indicators()

    def _compute_indicators(self):
        df = self.df
        c = self.close

        df["ma20"] = c.rolling(20).mean()
        df["ma50"] = c.rolling(50).mean()
        df["ma200"] = c.rolling(200, min_periods=60).mean()

        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - 100 / (1 + rs)

        df["bb_mid"] = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std
        df["bb_pct"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        df["vol_ratio"] = self.volume / self.volume.rolling(20).mean()

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        self.df = df

    def _signals_to_series(self, signal_type: str, extra_mask: pd.Series = None) -> pd.Series:
        entries = pd.Series(False, index=self.df.index)
        sigs = self.signals_df[self.signals_df["signal"] == signal_type]
        for date in sigs["date"]:
            if date in entries.index:
                entries.loc[date] = True

        if extra_mask is not None:
            entries = entries & extra_mask

        entries = entries.shift(1).fillna(False).astype(bool)
        return entries

    def _mask_trend(self, direction: str, ma: str = "ma200") -> pd.Series:
        return self.close > self.df[ma] if direction == "buy" else self.close < self.df[ma]

    def _mask_rsi(self, direction: str, buy_thresh: float = 35, sell_thresh: float = 65) -> pd.Series:
        return self.df["rsi"] < buy_thresh if direction == "buy" else self.df["rsi"] > sell_thresh

    def _mask_volume(self, ratio: float = 1.5) -> pd.Series:
        return self.df["vol_ratio"] > ratio

    def _mask_bollinger(self, direction: str, buy_pct: float = 0.25, sell_pct: float = 0.75) -> pd.Series:
        return self.df["bb_pct"] < buy_pct if direction == "buy" else self.df["bb_pct"] > sell_pct

    def _mask_macd_hist(self, direction: str) -> pd.Series:
        return self.df["macd_hist"] > self.df["macd_hist"].shift(1) if direction == "buy" else self.df["macd_hist"] < self.df["macd_hist"].shift(1)

    def _make_exits(self, entries: pd.Series, hold_days: int = 10) -> pd.Series:
        exits = pd.Series(False, index=entries.index)
        idx_list = entries[entries].index.tolist()
        for entry_date in idx_list:
            pos = entries.index.get_loc(entry_date)
            exit_pos = min(pos + hold_days, len(exits) - 1)
            exits.iloc[exit_pos] = True
        return exits

    def run_all_combos(self, hold_days: int | None = None) -> List[Dict]:
        hold_days = self.config.get("hold_days", 10) if hold_days is None else hold_days
        direction = "buy"
        sig_type = f"{direction}9"

        combo_defs = [
            ("pure_buy9", None),
            ("buy_ma200", self._mask_trend(direction, "ma200")),
            ("buy_ma50", self._mask_trend(direction, "ma50")),
            ("buy_rsi", self._mask_rsi(direction, buy_thresh=35)),
            ("buy_volume", self._mask_volume(1.5)),
            ("buy_bollinger", self._mask_bollinger(direction)),
            ("buy_macd", self._mask_macd_hist(direction)),
            ("buy_combo", self._mask_trend(direction) & self._mask_rsi(direction) & self._mask_volume(1.5)),
        ]

        results = []
        for name, mask in combo_defs:
            entries = self._signals_to_series(sig_type, extra_mask=mask)
            exits = self._make_exits(entries, hold_days=hold_days)
            n_sigs = int(entries.sum())
            win_rate, avg_ret, profit_factor = self._quick_stats(entries, n_days=hold_days)
            results.append(
                {
                    "name": name,
                    "entries": entries,
                    "exits": exits,
                    "n_signals": n_sigs,
                    "win_rate": win_rate,
                    "avg_ret": avg_ret,
                    "profit_factor": profit_factor,
                }
            )

        return results

    def _quick_stats(self, entries: pd.Series, n_days: int = 10) -> Tuple[float, float, float]:
        close_arr = self.close.values
        entry_idxs = np.where(entries.values)[0]
        rets = []
        for idx in entry_idxs:
            fwd = idx + n_days
            if fwd < len(close_arr):
                rets.append((close_arr[fwd] - close_arr[idx]) / close_arr[idx])
        if not rets:
            return 0.0, 0.0, 0.0
        rets = np.array(rets)
        win_rate = float((rets > 0).mean())
        avg_ret = float(rets.mean())
        gains = rets[rets > 0].mean() if (rets > 0).any() else 0
        losses = abs(rets[rets < 0].mean()) if (rets < 0).any() else 1e-9
        pf = gains / losses
        return round(win_rate, 4), round(avg_ret, 4), round(float(pf), 2)

    def print_summary(self, results: List[Dict]):
        for r in results:
            mark = " *" if r.get("_best") else ""
            print(f"{r['name']}: n={r['n_signals']} win={r['win_rate']:.1%} avg={r['avg_ret']:.2%} pf={r['profit_factor']:.2f}{mark}")

    def get_best_combo(self, results: List[Dict], lag: int = 1, sl_stop: float = 0.05, tp_stop: float = 0.15, fees: float = 0.001) -> Dict:
        from backtester import Backtester

        bt = Backtester(self.df, config=self.config)
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

        scored = sorted(valid, key=lambda r: (-np.inf if pd.isna(r.get("train_calmar")) else r.get("train_calmar")), reverse=True)
        best = scored[0]
        best["_best"] = True
        return best

    def save_report(self, results: List[Dict], path: str = "output/combo_report.csv"):
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = [{k: v for k, v in r.items() if not isinstance(v, pd.Series)} for r in results]
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
