from __future__ import annotations

import numpy as np
import pandas as pd


def _apply_weight_cap(weights: pd.Series, max_weight: float) -> pd.Series:
    w = weights.clip(upper=max_weight)
    total = w.sum()
    if total <= 0:
        return w
    return w / total


def build_weights(
    pred_df: pd.DataFrame,
    method: str,
    top_k: int,
    max_weight: float,
    alpha_threshold: float,
    uncertainty_threshold: float,
) -> pd.Series:
    df = pred_df.copy()
    df = df[(df["pred"] >= alpha_threshold) & (df["uncertainty"] <= uncertainty_threshold)]
    if df.empty:
        return pd.Series(dtype=float)
    df = df.sort_values("pred", ascending=False).head(top_k)
    if method == "score_proportional":
        raw = df.set_index("ticker")["pred"].clip(lower=0.0)
    else:
        raw = pd.Series(1.0, index=df["ticker"])  # equal-weight over selected
    return _apply_weight_cap(raw / raw.sum(), max_weight=max_weight)


def compute_turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    all_idx = prev_w.index.union(new_w.index)
    p = prev_w.reindex(all_idx, fill_value=0.0)
    n = new_w.reindex(all_idx, fill_value=0.0)
    return float(np.abs(n - p).sum())
