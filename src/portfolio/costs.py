from __future__ import annotations


def turnover_cost(turnover: float, tx_bps: float, slippage_bps: float) -> float:
    return turnover * (tx_bps + slippage_bps) / 10000.0
