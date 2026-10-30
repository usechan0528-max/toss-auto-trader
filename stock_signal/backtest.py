from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    total_signals: int
    win_rate: float
    average_return_pct: float
    average_loss_pct: float
    max_drawdown_pct: float
    target_hit_rate: float
    stop_hit_rate: float


def evaluate_signal(entry: float, stop: float, target: float, future_prices: list[float]) -> str:
    hit_stop = any(price <= stop for price in future_prices)
    hit_target = any(price >= target for price in future_prices)
    if hit_target and not hit_stop:
        return "WIN"
    if hit_stop and not hit_target:
        return "LOSS"
    if hit_target and hit_stop:
        return "AMBIGUOUS"
    return "NO_HIT"


def run_backtest(trades: pd.DataFrame) -> BacktestResult:
    if trades.empty:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0)

    wins = trades[trades["result"] == "WIN"]
    losses = trades[trades["result"] == "LOSS"]
    returns = trades["return_pct"].astype(float)
    drawdowns = trades["max_drawdown_pct"].astype(float)
    total = len(trades)
    return BacktestResult(
        total_signals=total,
        win_rate=len(wins) / total,
        average_return_pct=float(returns.mean()),
        average_loss_pct=float(losses["return_pct"].mean()) if not losses.empty else 0,
        max_drawdown_pct=float(drawdowns.min()) if not drawdowns.empty else 0,
        target_hit_rate=float((trades["result"].isin(["WIN", "AMBIGUOUS"])).mean()),
        stop_hit_rate=float((trades["result"].isin(["LOSS", "AMBIGUOUS"])).mean()),
    )
