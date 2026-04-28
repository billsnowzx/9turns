"""
data_loader.py — 多市场行情数据获取模块

支持：
  - A股（akshare / baostock）
  - 美股（yfinance）
  - 港股（yfinance，ticker 后缀 .HK）
"""

import pandas as pd
import numpy as np
from datetime import datetime
from cache import disk_cache


class DataLoader:
    """
    统一的行情数据接口。

    返回 DataFrame，index 为 DatetimeIndex，列为:
        open, high, low, close, volume
    """

    REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

    @disk_cache(ttl_hours=24)
    def load(
        self,
        market: str,
        symbol: str,
        freq: str = "daily",
        start: str = "2015-01-01",
        end: str = "2024-12-31",
    ) -> pd.DataFrame:
        """
        market : "a_share" | "us" | "hk"
        symbol : A股用 6位代码（如"000300"），美股/港股用 yfinance ticker
        freq   : "daily" | "weekly"
        """
        if market == "a_share":
            df = self._load_a_share(symbol, freq, start, end)
        elif market in ("us", "hk"):
            df = self._load_yfinance(symbol, freq, start, end)
        else:
            raise ValueError(f"不支持的市场类型: {market}")

        df = self._standardize(df, freq)
        self._validate(df)
        return df

    # ─────────────────────────────────────────────────────────────
    # A股：优先 akshare，失败自动切换到 baostock
    # ─────────────────────────────────────────────────────────────
    def _load_a_share(self, symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
        try:
            return self._load_akshare(symbol, freq, start, end)
        except Exception as e:
            print(f"  akshare 获取失败（{e}），尝试 baostock...")
            return self._load_baostock(symbol, freq, start, end)

    def _load_akshare(self, symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
        import akshare as ak
        period_map = {"daily": "daily", "weekly": "weekly"}
        period = period_map.get(freq, "daily")

        # 判断是指数还是个股（指数以 0/3/6 开头但长度6位且无字母）
        if symbol.startswith("0") and len(symbol) == 6 and symbol.isdigit():
            # 可能是指数（000300）或深市个股
            try:
                df = ak.index_zh_a_hist(
                    symbol=symbol, period=period,
                    start_date=start.replace("-", ""),
                    end_date=end.replace("-", ""),
                )
                col_map = {"日期": "date", "开盘": "open", "最高": "high",
                           "最低": "low", "收盘": "close", "成交量": "volume"}
                df = df.rename(columns=col_map)
            except Exception:
                df = ak.stock_zh_a_hist(
                    symbol=symbol, period=period,
                    start_date=start.replace("-", ""),
                    end_date=end.replace("-", ""),
                    adjust="qfq",
                )
                col_map = {"日期": "date", "开盘": "open", "最高": "high",
                           "最低": "low", "收盘": "close", "成交量": "volume"}
                df = df.rename(columns=col_map)
        else:
            df = ak.stock_zh_a_hist(
                symbol=symbol, period=period,
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",
            )
            col_map = {"日期": "date", "开盘": "open", "最高": "high",
                       "最低": "low", "收盘": "close", "成交量": "volume"}
            df = df.rename(columns=col_map)

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df[["open", "high", "low", "close", "volume"]]

    def _load_baostock(self, symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
        import baostock as bs
        freq_map = {"daily": "d", "weekly": "w"}
        bs.login()
        # baostock 代码格式：sh.000300 / sz.000001
        prefix = "sh" if symbol.startswith(("6", "0")) else "sz"
        bs_code = f"{prefix}.{symbol}"
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=start, end_date=end,
            frequency=freq_map.get(freq, "d"),
            adjustflag="3",   # 前复权
        )
        data = []
        while (rs.error_code == "0") and rs.next():
            data.append(rs.get_row_data())
        bs.logout()

        df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ─────────────────────────────────────────────────────────────
    # 美股 / 港股：yfinance
    # ─────────────────────────────────────────────────────────────
    def _load_yfinance(self, symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf
        interval_map = {"daily": "1d", "weekly": "1wk"}
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start, end=end,
            interval=interval_map.get(freq, "1d"),
            auto_adjust=True,
        )
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.columns = df.columns.str.lower()
        return df[["open", "high", "low", "close", "volume"]]

    # ─────────────────────────────────────────────────────────────
    # 标准化与验证
    # ─────────────────────────────────────────────────────────────
    def _standardize(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.dropna(subset=["close"])

        # 如果是日线数据请求周线，在此重采样
        # （通常直接请求周线更准，这里作为兜底）
        # if freq == "weekly" and df.index.freq != "W":
        #     df = df.resample("W").agg({
        #         "open": "first", "high": "max",
        #         "low": "min", "close": "last", "volume": "sum"
        #     }).dropna()

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _validate(self, df: pd.DataFrame):
        assert len(df) >= 100, "数据量不足100根K线，请检查数据源或放宽时间范围"
        for col in self.REQUIRED_COLS:
            assert col in df.columns, f"缺少列：{col}"
        assert not df["close"].isnull().all(), "收盘价全为空，数据获取失败"
        print(f"  数据验证通过 ✓  缺失值：{df.isnull().sum().sum()} 个")
