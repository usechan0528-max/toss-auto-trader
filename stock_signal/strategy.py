from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import add_intraday_indicators
from .risk import RiskPlan, build_risk_plan


@dataclass
class StrategyConfig:
    min_volume_ratio: float = 3.0
    min_rr_ratio: float = 1.5
    account_krw: float = 1_000_000
    usdkrw: float = 1_527
    risk_pct: float = 0.02


@dataclass
class Recommendation:
    symbol: str
    action: str
    price: float
    entry_zone: tuple[float, float]
    take_profit: tuple[float, float]
    stop_loss: float
    confidence: float
    rr_ratio: float
    recommended_shares: int
    expected_loss_usd: float
    reason: list[str] = field(default_factory=list)
    invalid_if: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    risk_level: str = "NORMAL"


def generate_recommendation(
    symbol: str,
    ohlcv: pd.DataFrame,
    config: StrategyConfig | None = None,
) -> Recommendation:
    config = config or StrategyConfig()
    data = add_intraday_indicators(ohlcv)
    latest = data.iloc[-1]
    price = float(latest["Close"])
    vwap = float(latest["VWAP"])
    volume_ratio = float(latest["VolumeRatio"])
    recent_high = float(latest["RecentHigh"])
    recent_low = float(latest["RecentLow"])
    atr = max(float(latest["ATR14"]), price * 0.01)

    breakout_base = max(price, recent_high)
    entry_high = breakout_base + atr * 0.20
    entry_low = min(price * 0.995, entry_high - atr * 0.20)
    stop = min(recent_low - atr * 0.25, entry_low - atr)
    target_1 = entry_high + (entry_high - stop) * 1.5
    target_2 = entry_high + (entry_high - stop) * 2.2
    risk_plan: RiskPlan = build_risk_plan(config.account_krw, config.usdkrw, entry_high, stop, target_1, config.risk_pct)

    above_vwap = price > vwap
    breakout = price >= recent_high
    rr_ok = risk_plan.rr_ratio >= config.min_rr_ratio
    reason: list[str] = []
    invalid_if = [
        f"{stop:.4f} 하향 이탈",
        "VWAP 아래 3분 이상 체류",
        "거래량 급감",
    ]
    risk_flags: list[str] = []

    if above_vwap:
        reason.append("VWAP 위")
    if volume_ratio >= config.min_volume_ratio:
        reason.append(f"거래량 {volume_ratio:.1f}배")
    if breakout:
        reason.append("직전 고점 돌파 시도")
    if rr_ok:
        reason.append(f"손익비 {risk_plan.rr_ratio:.2f}")

    if price < 1:
        risk_flags.append("$1 미만 초저가")
    if float(latest["RSI14"]) >= 75:
        risk_flags.append("RSI 과열")
    if float(latest["VolumeRatio"]) >= 8:
        risk_flags.append("거래량 과열")

    if not rr_ok:
        action = "NO_TRADE"
        reason.append("손익비 부족")
    elif above_vwap and volume_ratio >= config.min_volume_ratio and breakout:
        action = "BUY_WATCH"
    elif price < vwap:
        action = "NO_TRADE"
        reason.append("VWAP 아래")
    else:
        action = "WAIT"

    if risk_flags:
        risk_level = "HIGH" if len(risk_flags) >= 2 else "MEDIUM"
    else:
        risk_level = "NORMAL"

    confidence = 0.35
    confidence += 0.18 if above_vwap else -0.08
    confidence += min(volume_ratio / 10, 0.20)
    confidence += 0.15 if breakout else 0
    confidence += 0.12 if rr_ok else -0.15
    confidence = max(0.0, min(confidence, 0.95))

    return Recommendation(
        symbol=symbol,
        action=action,
        price=price,
        entry_zone=(entry_low, entry_high),
        take_profit=(target_1, target_2),
        stop_loss=stop,
        confidence=confidence,
        rr_ratio=risk_plan.rr_ratio,
        recommended_shares=risk_plan.shares,
        expected_loss_usd=risk_plan.expected_loss_usd,
        reason=reason,
        invalid_if=invalid_if,
        risk_flags=risk_flags,
        risk_level=risk_level,
    )
