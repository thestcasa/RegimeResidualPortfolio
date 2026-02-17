from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.download import download_yfinance_ohlcv


def load_prices(tickers: list[str], start: str, end: str, cache_dir: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for t in tickers:
        cache_path = Path(cache_dir) / "prices" / f"{t}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
        else:
            df = download_yfinance_ohlcv(t, start=start, end=end)
            if df.empty:
                continue
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]
        keep_cols = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
        tmp = df[keep_cols].copy()
        tmp["ticker"] = t
        tmp = tmp.reset_index().rename(columns={tmp.index.name or "Date": "date", "Date": "date"})
        tmp["date"] = pd.to_datetime(tmp["date"])
        frames.append(tmp)
    if not frames:
        raise RuntimeError("No price data downloaded.")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "ticker"])
