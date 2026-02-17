from __future__ import annotations

FEATURE_GROUPS = {
    "momentum": ["mom_21", "mom_63", "mom_126", "mom_252"],
    "reversal": ["rev_5", "rev_21"],
    "volatility": ["vol_21", "vol_63"],
    "drawdown": ["drawdown_252", "dist_from_high_252"],
    "liquidity": ["dollar_vol_21", "vol_vol_21"],
    "beta_corr": ["beta_126", "corr_bench_63"],
    "trend": ["ma_ratio_21_252"],
    "distribution": ["skew_63", "kurt_63"],
}
