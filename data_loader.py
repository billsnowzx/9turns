import warnings

import pandas as pd

from cache import disk_cache
from exceptions import TDDataError


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
        df = self._validate_and_align_frequency(df, freq)
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

    def _resample_ohlcv(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        return (
            df.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
        )

    def _validate_and_align_frequency(self, df: pd.DataFrame, requested_freq: str) -> pd.DataFrame:
        if len(df.index) < 3:
            return df
        step_series = df.index.to_series().diff().dropna()
        if step_series.empty:
            return df
        step = step_series.mode().iloc[0]

        if requested_freq == "weekly" and step < pd.Timedelta(days=5):
            warnings.warn("requested weekly but source appears daily; resampling to weekly", RuntimeWarning)
            return self._resample_ohlcv(df, "W")
        if requested_freq == "daily" and step > pd.Timedelta(days=2):
            warnings.warn("requested daily but source appears lower frequency; resampling to daily", RuntimeWarning)
            return self._resample_ohlcv(df, "D")
        return df

    def _validate(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise TDDataError("input DataFrame is empty")
        min_required = 100
        if len(df.index) >= 3:
            step = df.index.to_series().diff().dropna().mode().iloc[0]
            if step >= pd.Timedelta(days=5):
                min_required = 20
        if len(df) < min_required:
            raise TDDataError(f"not enough bars (<{min_required})")
        for col in self.REQUIRED_COLS:
            if col not in df.columns:
                raise TDDataError(f"missing column: {col}")
        if df["close"].isnull().all():
            raise TDDataError("close is all null")
