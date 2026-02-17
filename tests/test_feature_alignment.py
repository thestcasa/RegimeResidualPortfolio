import pandas as pd

from src.features.build_features import build_daily_features


def test_feature_month_alignment() -> None:
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    frame = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["SPY"] * len(dates) + ["QQQ"] * len(dates),
            "adj_close": 100.0,
            "volume": 1_000_000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
        }
    )
    cfg = {
        "features": {
            "skip_recent_days": 5,
            "momentum_windows": [21, 63, 126, 252],
            "reversal_windows": [5, 21],
            "vol_windows": [21, 63],
            "drawdown_window": 252,
            "liquidity_window": 21,
            "beta_window": 126,
            "corr_window": 63,
            "add_trend": True,
            "add_skew_kurtosis": True,
        }
    }
    out = build_daily_features(frame, benchmark="SPY", cfg=cfg)
    assert "month" in out.columns
    assert out["month"].dt.is_month_end.all()
