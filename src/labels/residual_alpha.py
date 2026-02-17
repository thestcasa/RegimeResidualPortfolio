from __future__ import annotations

import pandas as pd


def build_monthly_realized_returns(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices.copy().sort_values(["ticker", "date"])
    p["ret_1d"] = p.groupby("ticker")["adj_close"].pct_change()
    p["month"] = p["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = p.groupby(["ticker", "month"], as_index=False)["ret_1d"].apply(lambda x: (1 + x).prod() - 1)
    return monthly.rename(columns={"ret_1d": "monthly_return"})


def build_residual_alpha_labels(
    monthly_returns: pd.DataFrame, expected_returns: pd.DataFrame, horizon_months: int = 1
) -> pd.DataFrame:
    df = monthly_returns.merge(expected_returns, on=["ticker", "month"], how="left")
    df = df.sort_values(["ticker", "month"])
    df["fwd_return"] = df.groupby("ticker")["monthly_return"].shift(-horizon_months)
    df["target"] = df["fwd_return"] - df["expected_return"]
    df["target_month"] = df.groupby("ticker")["month"].shift(-horizon_months)
    return df[["ticker", "month", "target_month", "target", "fwd_return", "expected_return"]]
