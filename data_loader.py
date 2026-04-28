from datetime import datetime

import pandas as pd

from cache import disk_cache


class DataLoader:
    REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @disk_cache(ttl_hours=24)
    def load(
        self,
        market: str,
        symbol: str,
        freq: str = "daily",
        start: str = "2015-01-01",
        end: str = "2024-12-31",
    ) -> pd.DataFrame:
        data_cfg = self.config.get("data", {})
        freq = data_cfg.get("freq", freq)
        start = data_cfg.get("start", start)
        end = data_cfg.get("end", end)

        if market == "a_share":
            df = self._load_a_share(symbol, freq, start, end)
        elif market in ("us", "hk"):
            df = self._load_yfinance(symbol, freq, start, end)
        else:
            raise ValueError(f"unsupported market: {market}")

        df = self._standardize(df, freq)
        self._validate(df)
        return df

    def _load_a_share(self, symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
        try:
            return self._load_akshare(symbol, freq, start, end)
        except Exception:
            return self._load_baostock(symbol, freq, start, end)

    def _load_akshare(self, symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
        import akshare as ak

        period_map = {"daily": "daily", "weekly": "weekly"}
        period = period_map.get(freq, "daily")

        try:
            df = ak.index_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
            )
        except Exception:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust="qfq",
            )

        col_map = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
        df = df.rename(columns=col_map)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df[["open", "high", "low", "close", "volume"]]

    def _load_baostock(self, symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
        import baostock as bs

        freq_map = {"daily": "d", "weekly": "w"}
        bs.login()
        prefix = "sh" if symbol.startswith(("6", "0")) else "sz"
        bs_code = f"{prefix}.{symbol}"
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=start,
            end_date=end,
            frequency=freq_map.get(freq, "d"),
            adjustflag="3",
        )
        data = []
        while (rs.error_code == "0") and rs.next():
            data.append(rs.get_row_data())
        bs.logout()

        df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        for col in self.REQUIRED_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _load_yfinance(self, symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf

        interval_map = {"daily": "1d", "weekly": "1wk"}
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval_map.get(freq, "1d"), auto_adjust=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.columns = df.columns.str.lower()
        return df[["open", "high", "low", "close", "volume"]]

    def _standardize(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.dropna(subset=["close"])
        for col in self.REQUIRED_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _validate(self, df: pd.DataFrame):
        assert len(df) >= 100, "not enough bars (<100)"
        for col in self.REQUIRED_COLS:
            assert col in df.columns, f"missing column: {col}"
        assert not df["close"].isnull().all(), "close is all null"
