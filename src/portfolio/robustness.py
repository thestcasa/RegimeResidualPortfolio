from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_sharpe(returns: pd.Series, samples: int, block_size: int) -> tuple[float, float]:
    arr = returns.values
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    vals = []
    for _ in range(samples):
        out = []
        while len(out) < n:
            i = np.random.randint(0, max(1, n - block_size + 1))
            out.extend(arr[i : i + block_size])
        rs = np.array(out[:n])
        ann_ret = (1 + rs).prod() ** (12 / n) - 1
        ann_vol = rs.std(ddof=0) * np.sqrt(12)
        vals.append(ann_ret / ann_vol if ann_vol > 0 else 0.0)
    return float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))


def subperiod_metrics(returns: pd.Series, split_dates: list[str]) -> pd.DataFrame:
    rows = []
    for s in split_dates:
        dt = pd.Timestamp(s)
        pre = returns[returns.index < dt]
        post = returns[returns.index >= dt]
        rows.append({"period": f"pre_{s}", "mean": pre.mean(), "vol": pre.std(ddof=0)})
        rows.append({"period": f"post_{s}", "mean": post.mean(), "vol": post.std(ddof=0)})
    return pd.DataFrame(rows)
