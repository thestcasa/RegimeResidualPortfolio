from __future__ import annotations

import pandas as pd


def month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.to_period("M").to_timestamp("M")


def to_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    gross = (1.0 + daily_returns).groupby(daily_returns.index.to_period("M")).prod()
    monthly = gross - 1.0
    monthly.index = monthly.index.to_timestamp("M")
    return monthly.sort_index()
