# Regime Residual Portfolio

Research-grade, reproducible pipeline for residual-alpha prediction and monthly portfolio management.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --config configs/base.yaml
```

## Outputs
- `results/backtest_summary.json`
- `results/equity_curve.csv`
- `results/weights.csv`
- `results/trades.csv`
- `results/metrics.csv`
- `results/plots/*.png`
- `models/*.pt`, `models/*.pkl`
- `docs/PROJECT_REPORT.md`

## Key ideas implemented
- CAPM-aligned residual alpha labels.
- Regime features (trend/volatility/drawdown).
- MC Dropout uncertainty filtering.
- Monthly walk-forward with train/validation embargo.
- Cost- and turnover-aware portfolio construction.
- Baselines: equal-weight, momentum, ridge.
- W&B tracking enabled by default with offline fallback.
