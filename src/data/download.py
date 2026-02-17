from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def download_yfinance_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return df
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df


def cache_prices(df: pd.DataFrame, cache_path: str | Path) -> None:
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
