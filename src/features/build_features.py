from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_beta(asset_ret: pd.Series, mkt_ret: pd.Series, window: int) -> pd.Series:
    cov = asset_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    return cov / var.replace(0.0, np.nan)


def build_daily_features(prices: pd.DataFrame, benchmark: str, cfg: dict) -> pd.DataFrame:
    prices = prices.copy().sort_values(["ticker", "date"])
    prices["ret_1d"] = prices.groupby("ticker")["adj_close"].pct_change()
    bench = prices[prices["ticker"] == benchmark][["date", "ret_1d"]].rename(columns={"ret_1d": "bench_ret"})
    data = prices.merge(bench, on="date", how="left")

    wcfg = cfg["features"]

    def _calc(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        skip = int(wcfg["skip_recent_days"])
        for w in wcfg["momentum_windows"]:
            g[f"mom_{w}"] = g["adj_close"].shift(skip) / g["adj_close"].shift(skip + w) - 1.0
        for w in wcfg["reversal_windows"]:
            g[f"rev_{w}"] = -g["ret_1d"].rolling(w).sum()
        for w in wcfg["vol_windows"]:
            g[f"vol_{w}"] = g["ret_1d"].rolling(w).std()

        ddw = int(wcfg["drawdown_window"])
        roll_max = g["adj_close"].rolling(ddw).max()
        g[f"drawdown_{ddw}"] = g["adj_close"] / roll_max - 1.0
        g[f"dist_from_high_{ddw}"] = g["adj_close"] / roll_max - 1.0

        liq = int(wcfg["liquidity_window"])
        g[f"dollar_vol_{liq}"] = (g["adj_close"] * g["volume"]).rolling(liq).mean()
        g[f"vol_vol_{liq}"] = g["volume"].pct_change().rolling(liq).std()

        bw = int(wcfg["beta_window"])
        g[f"beta_{bw}"] = _rolling_beta(g["ret_1d"], g["bench_ret"], bw)

        cw = int(wcfg["corr_window"])
        g[f"corr_bench_{cw}"] = g["ret_1d"].rolling(cw).corr(g["bench_ret"])

        if wcfg.get("add_trend", True):
            g["ma_ratio_21_252"] = g["adj_close"].rolling(21).mean() / g["adj_close"].rolling(252).mean() - 1.0
        if wcfg.get("add_skew_kurtosis", True):
            g["skew_63"] = g["ret_1d"].rolling(63).skew()
            g["kurt_63"] = g["ret_1d"].rolling(63).kurt()
        return g

    feats = data.groupby("ticker", group_keys=False).apply(_calc)
    feats["month"] = feats["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = feats.sort_values("date").groupby(["ticker", "month"], as_index=False).last()
    return monthly


def build_regime_features(prices: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    b = prices[prices["ticker"] == benchmark].copy().sort_values("date")
    b["ret_1d"] = b["adj_close"].pct_change()
    b["month"] = b["date"].dt.to_period("M").dt.to_timestamp("M")
    monthly = b.groupby("month", as_index=False).agg(
        bench_ret=("ret_1d", lambda x: (1 + x).prod() - 1),
        bench_vol=("ret_1d", "std"),
        bench_price=("adj_close", "last"),
    )
    monthly["trend_regime"] = (monthly["bench_price"] > monthly["bench_price"].rolling(6).mean()).astype(float)
    monthly["vol_regime"] = (monthly["bench_vol"] > monthly["bench_vol"].rolling(12).median()).astype(float)
    monthly["drawdown_regime"] = (monthly["bench_price"] / monthly["bench_price"].cummax() - 1.0 < -0.1).astype(float)
    monthly["regime_code"] = (
        monthly["trend_regime"].astype(int) * 4
        + monthly["vol_regime"].astype(int) * 2
        + monthly["drawdown_regime"].astype(int)
    )
    return monthly[["month", "trend_regime", "vol_regime", "drawdown_regime", "regime_code"]]
