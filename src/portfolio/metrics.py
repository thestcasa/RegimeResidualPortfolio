from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    ann_ret = (1 + returns).prod() ** (12 / max(len(returns), 1)) - 1
    ann_vol = returns.std(ddof=0) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(12)
    sortino = ann_ret / downside if downside > 0 else 0.0
    eq = (1 + returns).cumprod()
    dd = eq / eq.cummax() - 1
    max_dd = float(dd.min()) if len(dd) else 0.0
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
    }
