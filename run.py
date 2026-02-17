from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.config import load_config, save_config
from src.data.factors import build_market_factor, fetch_ken_french_daily
from src.data.prices import load_prices
from src.data.universe import get_universe
from src.features.build_features import build_daily_features, build_regime_features
from src.features.leakage_checks import assert_no_lookahead
from src.labels.expected_return import estimate_capm_expected_monthly
from src.labels.residual_alpha import build_monthly_realized_returns, build_residual_alpha_labels
from src.portfolio.backtest import walkforward_backtest
from src.portfolio.metrics import performance_metrics
from src.portfolio.robustness import block_bootstrap_sharpe, subperiod_metrics
from src.utils.io import ensure_dir, hash_list, write_parquet
from src.utils.logging import get_logger
from src.utils.plotting import save_line_plot
from src.utils.seed import set_global_seed


def _init_wandb(cfg: dict) -> object | None:
    if not cfg["wandb"]["enabled"]:
        return None
    try:
        import wandb

        mode = cfg["wandb"].get("mode", "auto")
        if mode == "auto":
            mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            config=cfg,
            tags=cfg["wandb"].get("tags", []),
            mode=mode,
        )
        return run
    except Exception:
        return None




def _compute_baselines(dataset: pd.DataFrame, months: list[pd.Timestamp], cfg: dict) -> pd.DataFrame:
    rows = []
    for m in months:
        d = dataset[dataset["month"] == m].copy()
        if d.empty:
            continue
        ew = float(d["fwd_return"].mean())
        top = d.sort_values("mom_252", ascending=False).head(cfg["portfolio"]["top_k"])
        mom = float(top["fwd_return"].mean()) if not top.empty else 0.0
        rows.append({"month": m, "equal_weight": ew, "momentum_12_1": mom})
    return pd.DataFrame(rows).sort_values("month")
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_obj = load_config(args.config)
    cfg = cfg_obj.as_dict()
    logger = get_logger()
    set_global_seed(cfg["project"]["seed"], cfg["project"]["deterministic_torch"])

    out_dir = ensure_dir(cfg["project"]["output_dir"])
    model_dir = ensure_dir(cfg["project"]["model_dir"])
    ensure_dir(out_dir / "plots")
    save_config(cfg_obj, out_dir / "config_used.yaml")

    run = _init_wandb(cfg)

    universe = get_universe(cfg)
    prices = load_prices(universe, cfg["data"]["start_date"], cfg["data"]["end_date"], cfg["project"]["cache_dir"])
    write_parquet(prices, Path(cfg["project"]["cache_dir"]) / "prices_all.parquet")

    benchmark = cfg["data"]["benchmark"]
    feats = build_daily_features(prices, benchmark=benchmark, cfg=cfg)
    regimes = build_regime_features(prices, benchmark=benchmark)

    monthly_ret = build_monthly_realized_returns(prices)
    beta_col = f"beta_{cfg['features']['beta_window']}"
    beta_monthly = feats[["ticker", "month", beta_col]].rename(columns={beta_col: "beta"})
    bench_m = monthly_ret[monthly_ret["ticker"] == benchmark].set_index("month")["monthly_return"].sort_index()
    expected = estimate_capm_expected_monthly(
        monthly_ret,
        benchmark_monthly=bench_m,
        beta_monthly=beta_monthly,
        market_premium_window=cfg["factors"]["market_premium_window"],
    )
    labels = build_residual_alpha_labels(monthly_ret, expected, horizon_months=1)

    try:
        if cfg["factors"]["use_ken_french_if_available"]:
            ff = fetch_ken_french_daily(cfg["project"]["cache_dir"])
            logger.info("Loaded Kenneth French factors with %s rows", len(ff))
    except Exception as e:
        logger.warning("French factors unavailable, CAPM-only fallback used: %s", e)
        _ = build_market_factor(bench_m)

    dataset = feats.merge(labels, on=["ticker", "month"], how="left").merge(regimes, on="month", how="left")
    dataset = dataset.dropna(subset=["target", "fwd_return"])

    feature_cols = [
        c
        for c in dataset.columns
        if c
        not in {
            "date",
            "ticker",
            "month",
            "target",
            "target_month",
            "fwd_return",
            "expected_return",
            "monthly_return",
        }
        and dataset[c].dtype != "O"
    ]
    assert_no_lookahead(dataset)

    bt = walkforward_backtest(dataset, feature_cols, cfg, model_dir=model_dir)
    metrics = performance_metrics(bt.monthly.set_index("month")["net_return"])
    baselines = _compute_baselines(dataset, bt.monthly["month"].tolist(), cfg)
    metrics_gross = performance_metrics(bt.monthly.set_index("month")["gross_return"])
    lo, hi = block_bootstrap_sharpe(
        bt.monthly.set_index("month")["net_return"],
        cfg["robustness"]["bootstrap_samples"],
        cfg["robustness"]["block_size"],
    )
    metrics["sharpe_ci_5"] = lo
    metrics["sharpe_ci_95"] = hi
    metrics["avg_turnover"] = float(bt.monthly["turnover"].mean())
    metrics["avg_holdings"] = float(bt.monthly["n_holdings"].mean())
    metrics["feature_hash"] = hash_list(feature_cols)

    bt.equity_curve.to_csv(out_dir / "equity_curve.csv", index=False)
    bt.weights.to_csv(out_dir / "weights.csv", index=False)
    bt.trades.to_csv(out_dir / "trades.csv", index=False)
    bt.monthly.to_csv(out_dir / "metrics.csv", index=False)
    baselines.to_csv(out_dir / "baselines.csv", index=False)
    pd.DataFrame([metrics | {f"gross_{k}": v for k, v in metrics_gross.items()}]).to_json(
        out_dir / "backtest_summary.json", orient="records", indent=2
    )

    sub = subperiod_metrics(bt.monthly.set_index("month")["net_return"], cfg["robustness"]["subperiod_splits"])
    sub.to_csv(out_dir / "subperiod_metrics.csv", index=False)

    eq = bt.monthly.set_index("month")["equity_net"]
    dd = eq / eq.cummax() - 1.0
    roll_sharpe = (
        bt.monthly.set_index("month")["net_return"].rolling(cfg["robustness"]["rolling_window_months"]).mean()
        / bt.monthly.set_index("month")["net_return"].rolling(cfg["robustness"]["rolling_window_months"]).std(ddof=0)
    ) * (12**0.5)
    save_line_plot(eq, "Equity Curve (Net)", out_dir / "plots" / "equity.png")
    save_line_plot(dd, "Drawdown", out_dir / "plots" / "drawdown.png")
    save_line_plot(roll_sharpe, "Rolling Sharpe", out_dir / "plots" / "rolling_sharpe.png")
    save_line_plot(bt.monthly.set_index("month")["turnover"], "Monthly Turnover", out_dir / "plots" / "turnover.png")

    report_path = Path("docs/PROJECT_REPORT.md")
    report = f"""# Regime Residual Portfolio Report

## Methodology
- Universe: {len(universe)} ETFs ({', '.join(universe[:8])}...)
- Label: one-month forward residual alpha = realized return - CAPM expected return.
- Regime conditioning: trend/vol/drawdown regime features merged at month level.
- Model: MLP + MC Dropout uncertainty, walk-forward expanding window, embargo={cfg['walkforward']['embargo_months']} month.
- Portfolio: top-K with uncertainty and alpha threshold filter, transaction + slippage costs.

## Key Run Metadata
- Date range: {cfg['data']['start_date']} to {cfg['data']['end_date']}
- Feature hash: {metrics['feature_hash']}
- W&B enabled: {cfg['wandb']['enabled']}

## Performance Summary (from this local run)
- Net Sharpe: {metrics['sharpe']:.3f}
- Net Ann Return: {metrics['ann_return']:.3%}
- Net Max Drawdown: {metrics['max_drawdown']:.3%}
- Avg Turnover: {metrics['avg_turnover']:.3f}
- Sharpe 90% bootstrap CI: [{metrics['sharpe_ci_5']:.3f}, {metrics['sharpe_ci_95']:.3f}]

## Reproducibility
1. Install dependencies from `requirements.txt`.
2. Run `python run.py --config configs/base.yaml`.
3. Check outputs under `results/` and model artifacts under `models/`.

## Limitations
- Non-stationarity and regime shifts may degrade future performance.
- ETF universe reduces survivorship risk but does not eliminate data-quality issues.
- Simplified costs and constraints are approximations.
"""
    report_path.write_text(report, encoding="utf-8")

    if run is not None:
        import wandb

        wandb.log({f"metric/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        wandb.log({"universe_size": len(universe), "feature_count": len(feature_cols)})
        wandb.log({"plot_equity": wandb.Image(str(out_dir / "plots" / "equity.png"))})
        art = wandb.Artifact("results", type="backtest")
        for f in ["equity_curve.csv", "weights.csv", "trades.csv", "metrics.csv", "baselines.csv", "backtest_summary.json"]:
            art.add_file(str(out_dir / f))
        wandb.log_artifact(art)
        wandb.finish()

    logger.info("Pipeline complete. Results at %s", out_dir)


if __name__ == "__main__":
    main()
