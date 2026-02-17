from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_capm_expected_monthly(
    monthly_returns: pd.DataFrame,
    benchmark_monthly: pd.Series,
    beta_monthly: pd.DataFrame,
    market_premium_window: int,
) -> pd.DataFrame:
    """E[R] = beta * rolling mean market return."""
    prem = benchmark_monthly.rolling(market_premium_window).mean().rename("mkt_prem")
    merged = monthly_returns.merge(beta_monthly, on=["ticker", "month"], how="left")
    merged = merged.merge(prem.rename_axis("month").reset_index(), on="month", how="left")
    merged["expected_return"] = merged["beta"].fillna(1.0) * merged["mkt_prem"].fillna(0.0)
    merged["expected_return"] = merged["expected_return"].replace([np.inf, -np.inf], np.nan)
    return merged[["ticker", "month", "expected_return"]]
