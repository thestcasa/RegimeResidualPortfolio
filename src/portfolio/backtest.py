from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from src.models.baselines import momentum_12_1_scores, ridge_predict
from src.models.inference import mc_dropout_predict
from src.models.train import train_mlp
from src.portfolio.construction import build_weights, compute_turnover
from src.portfolio.costs import turnover_cost


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    monthly: pd.DataFrame


def walkforward_backtest(dataset: pd.DataFrame, feature_cols: list[str], cfg: dict, model_dir: str | Path) -> BacktestResult:
    months = sorted(dataset["month"].dropna().unique())
    min_train = cfg["walkforward"]["min_train_months"]
    val_months = cfg["walkforward"]["val_months"]
    embargo = cfg["walkforward"]["embargo_months"]

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    prev_w = pd.Series(dtype=float)
    weights_rows: list[dict] = []
    trade_rows: list[dict] = []
    perf_rows: list[dict] = []

    for i in range(min_train + val_months + embargo, len(months) - 1):
        pred_month = months[i]
        train_end = i - val_months - embargo
        train_months = months[:train_end]
        val_months_slice = months[train_end : i - embargo]

        train_df = dataset[dataset["month"].isin(train_months)].dropna(subset=feature_cols + ["target"])
        val_df = dataset[dataset["month"].isin(val_months_slice)].dropna(subset=feature_cols + ["target"])
        pred_df = dataset[dataset["month"] == pred_month].dropna(subset=feature_cols)

        if len(train_df) < 200 or pred_df.empty:
            continue

        art = train_mlp(
            train_df[feature_cols].values,
            train_df["target"].values,
            val_df[feature_cols].values,
            val_df["target"].values,
            cfg,
        )
        month_str = pd.Timestamp(pred_month).strftime("%Y-%m")
        torch.save(art.model.state_dict(), model_dir / f"mlp_{month_str}.pt")
        with (model_dir / f"scaler_{month_str}.pkl").open("wb") as f:
            pickle.dump(art.scaler, f)

        x_pred = art.scaler.transform(pred_df[feature_cols].values)
        mean_pred, unc = mc_dropout_predict(art.model, x_pred, cfg["model"]["mc_dropout_samples"])

        pred_out = pred_df[["ticker", "month", "fwd_return", "beta"]].copy()
        pred_out["pred"] = mean_pred
        pred_out["uncertainty"] = unc

        ridge_pred = ridge_predict(train_df[feature_cols].values, train_df["target"].values, pred_df[feature_cols].values)
        pred_out["pred_ridge"] = ridge_pred
        pred_out["pred_mom"] = momentum_12_1_scores(pred_df)

        new_w = build_weights(
            pred_out,
            method=cfg["portfolio"]["method"],
            top_k=cfg["portfolio"]["top_k"],
            max_weight=cfg["portfolio"]["max_weight"],
            alpha_threshold=cfg["portfolio"]["alpha_threshold"],
            uncertainty_threshold=cfg["portfolio"]["uncertainty_threshold"],
        )
        turnover = compute_turnover(prev_w, new_w)
        if turnover > cfg["portfolio"]["turnover_cap"]:
            scale = cfg["portfolio"]["turnover_cap"] / turnover
            idx = prev_w.index.union(new_w.index)
            p = prev_w.reindex(idx, fill_value=0.0)
            n = new_w.reindex(idx, fill_value=0.0)
            new_w = p + scale * (n - p)
            turnover = compute_turnover(prev_w, new_w)

        gross = float((pred_out.set_index("ticker")["fwd_return"].reindex(new_w.index).fillna(0.0) * new_w).sum())
        cost = turnover_cost(
            turnover,
            cfg["costs"]["transaction_bps_per_turnover"],
            cfg["costs"]["slippage_bps_per_turnover"],
        )
        net = gross - cost

        perf_rows.append(
            {
                "month": pred_month,
                "gross_return": gross,
                "net_return": net,
                "turnover": turnover,
                "n_holdings": int((new_w > 0).sum()),
                "acceptance_rate": float(
                    ((pred_out["pred"] >= cfg["portfolio"]["alpha_threshold"]) & (pred_out["uncertainty"] <= cfg["portfolio"]["uncertainty_threshold"])).mean()
                ),
            }
        )
        for t, w in new_w.items():
            weights_rows.append({"month": pred_month, "ticker": t, "weight": float(w)})
        for t in prev_w.index.union(new_w.index):
            trade_rows.append({"month": pred_month, "ticker": t, "trade": float(new_w.get(t, 0.0) - prev_w.get(t, 0.0))})
        prev_w = new_w

    monthly = pd.DataFrame(perf_rows).sort_values("month")
    monthly["equity_net"] = (1 + monthly["net_return"]).cumprod()
    monthly["equity_gross"] = (1 + monthly["gross_return"]).cumprod()
    equity = monthly[["month", "equity_net", "equity_gross"]]
    return BacktestResult(equity_curve=equity, weights=pd.DataFrame(weights_rows), trades=pd.DataFrame(trade_rows), monthly=monthly)
