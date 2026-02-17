# Regime Residual Portfolio Report

Run `python run.py --config configs/base.yaml` to populate this report with realized metrics from the local run.

## Research framing
This project studies out-of-sample monthly portfolio construction based on residual alpha forecasts with leakage-safe walk-forward training, regime conditioning features, and uncertainty-aware trade filtering.

## Warnings and limitations
- No guaranteed alpha; this is a research workflow.
- Results are sensitive to non-stationarity, costs, and configuration search.
- ETF universe mitigates but does not eliminate survivorship/data-quality biases.
