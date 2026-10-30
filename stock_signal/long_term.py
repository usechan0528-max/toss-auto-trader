from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LongTermPlan:
    symbol: str
    current_price: float
    monthly_budget_usd: float
    initial_cash_usd: float
    horizon_years: int
    expected_return_annual_pct: float
    projected_value_usd: float
    total_contribution_usd: float
    estimated_flat_shares: float
    monthly_shares: float
    price_vs_sma50_pct: float | None
    price_vs_sma200_pct: float | None
    drawdown_from_high_pct: float
    max_drawdown_pct: float
    annualized_volatility_pct: float
    dca_signal: str
    suggested_action: str
    buy_ladder: list[dict[str, Any]]
    scenarios: list[dict[str, Any]]
    risk_notes: list[str]
    data_points: int


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _daily_close(data: pd.DataFrame) -> pd.Series:
    if data.empty or "Close" not in data.columns:
        raise ValueError("Close column is required")
    close = pd.to_numeric(data["Close"], errors="coerce").dropna()
    if close.empty:
        raise ValueError("Close data is empty")
    if isinstance(close.index, pd.DatetimeIndex):
        close = close.resample("1D").last().dropna()
    return close.astype(float)


def _max_drawdown_pct(close: pd.Series) -> float:
    running_high = close.cummax()
    drawdown = close / running_high - 1
    return round(float(drawdown.min() * 100), 2)


def _future_value(initial_cash: float, monthly_budget: float, months: int, annual_return_pct: float) -> float:
    monthly_return = (1 + annual_return_pct / 100) ** (1 / 12) - 1
    if abs(monthly_return) < 1e-9:
        return initial_cash + monthly_budget * months
    initial_future = initial_cash * ((1 + monthly_return) ** months)
    contribution_future = monthly_budget * (((1 + monthly_return) ** months - 1) / monthly_return)
    return initial_future + contribution_future


def _scenario_value(initial_cash: float, monthly_budget: float, years: int, annual_return_pct: float) -> dict[str, Any]:
    months = max(1, int(years * 12))
    total_contribution = initial_cash + monthly_budget * months
    projected = _future_value(initial_cash, monthly_budget, months, annual_return_pct)
    return {
        "연수": years,
        "연수익률": f"{annual_return_pct:+.1f}%",
        "투입원금": round(total_contribution, 2),
        "예상평가액": round(projected, 2),
        "예상손익": round(projected - total_contribution, 2),
    }


def build_long_term_plan(
    data: pd.DataFrame,
    *,
    symbol: str,
    monthly_budget_usd: float,
    initial_cash_usd: float = 0.0,
    horizon_years: int = 10,
    expected_return_annual_pct: float = 6.0,
) -> LongTermPlan:
    close = _daily_close(data)
    current_price = float(close.iloc[-1])
    if current_price <= 0:
        raise ValueError("current price must be positive")

    monthly_budget = max(0.0, float(monthly_budget_usd))
    initial_cash = max(0.0, float(initial_cash_usd))
    years = max(1, int(horizon_years))
    months = years * 12
    total_contribution = initial_cash + monthly_budget * months
    projected_value = _future_value(initial_cash, monthly_budget, months, float(expected_return_annual_pct))
    estimated_flat_shares = total_contribution / current_price
    monthly_shares = monthly_budget / current_price if current_price > 0 else 0

    high = float(close.max())
    drawdown_from_high_pct = (current_price / high - 1) * 100 if high > 0 else 0.0
    returns = close.pct_change().dropna()
    annualized_volatility_pct = float(returns.std() * sqrt(252) * 100) if not returns.empty else 0.0
    max_drawdown_pct = _max_drawdown_pct(close)

    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    price_vs_sma50_pct = (current_price / sma50 - 1) * 100 if sma50 else None
    price_vs_sma200_pct = (current_price / sma200 - 1) * 100 if sma200 else None

    risk_notes: list[str] = []
    if len(close) < 120:
        risk_notes.append("장투 판단에는 데이터 기간이 짧습니다. 1년 이상 일봉을 권장합니다.")
    if annualized_volatility_pct >= 55:
        risk_notes.append("연환산 변동성이 높아 적립금을 더 잘게 나누는 편이 낫습니다.")
    if drawdown_from_high_pct <= -25:
        risk_notes.append("고점 대비 낙폭이 큽니다. 추가 매수는 분할 기준을 지켜야 합니다.")
    if price_vs_sma200_pct is not None and price_vs_sma200_pct >= 20:
        risk_notes.append("200일선 대비 과열 구간입니다. 일시 매수보다 정액 적립 중심이 적합합니다.")

    if drawdown_from_high_pct <= -20 and (price_vs_sma200_pct is None or price_vs_sma200_pct <= 5):
        dca_signal = "분할 강화"
        suggested_action = "기본 적립은 유지하고, 추가 매수는 하락 단계별로 나눕니다."
    elif price_vs_sma200_pct is not None and price_vs_sma200_pct >= 18 and drawdown_from_high_pct > -8:
        dca_signal = "과열 주의"
        suggested_action = "정기 적립만 유지하고 추가 일시 매수는 보류합니다."
    elif annualized_volatility_pct >= 60:
        dca_signal = "소액 분산"
        suggested_action = "월 적립금을 2~4회로 쪼개 변동성 노출을 줄입니다."
    else:
        dca_signal = "정기매수 유지"
        suggested_action = "월 적립을 유지하고 과도한 타이밍 매매를 피합니다."

    ladder_steps = [
        ("현재", 0.00, 0.40),
        ("-5%", -0.05, 0.25),
        ("-10%", -0.10, 0.20),
        ("-15%", -0.15, 0.15),
    ]
    buy_ladder = []
    for label, drop_pct, allocation in ladder_steps:
        target_price = current_price * (1 + drop_pct)
        budget = monthly_budget * allocation
        buy_ladder.append(
            {
                "구간": label,
                "기준가": round(target_price, 4),
                "예산USD": round(budget, 2),
                "예상주수": round(budget / target_price if target_price > 0 else 0, 4),
            }
        )

    scenarios = [
        _scenario_value(initial_cash, monthly_budget, years, -5.0),
        _scenario_value(initial_cash, monthly_budget, years, 0.0),
        _scenario_value(initial_cash, monthly_budget, years, expected_return_annual_pct),
        _scenario_value(initial_cash, monthly_budget, years, 10.0),
    ]

    return LongTermPlan(
        symbol=symbol,
        current_price=round(current_price, 4),
        monthly_budget_usd=round(monthly_budget, 2),
        initial_cash_usd=round(initial_cash, 2),
        horizon_years=years,
        expected_return_annual_pct=round(float(expected_return_annual_pct), 2),
        projected_value_usd=round(projected_value, 2),
        total_contribution_usd=round(total_contribution, 2),
        estimated_flat_shares=round(estimated_flat_shares, 4),
        monthly_shares=round(monthly_shares, 4),
        price_vs_sma50_pct=round(price_vs_sma50_pct, 2) if price_vs_sma50_pct is not None else None,
        price_vs_sma200_pct=round(price_vs_sma200_pct, 2) if price_vs_sma200_pct is not None else None,
        drawdown_from_high_pct=round(drawdown_from_high_pct, 2),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        annualized_volatility_pct=round(annualized_volatility_pct, 2),
        dca_signal=dca_signal,
        suggested_action=suggested_action,
        buy_ladder=buy_ladder,
        scenarios=scenarios,
        risk_notes=risk_notes,
        data_points=int(len(close)),
    )


def long_term_plan_to_dict(plan: LongTermPlan) -> dict[str, Any]:
    return asdict(plan)
