from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskPlan:
    account_usd: float
    max_loss_usd: float
    risk_per_share: float
    shares: int
    expected_loss_usd: float
    expected_loss_krw: float
    rr_ratio: float


def position_size(account_krw: float, usdkrw: float, entry: float, stop: float, risk_pct: float = 0.02) -> int:
    if account_krw <= 0 or usdkrw <= 0:
        return 0
    account_usd = account_krw / usdkrw
    max_loss_usd = account_usd * risk_pct
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    return int(max_loss_usd / risk_per_share)


def build_risk_plan(
    account_krw: float,
    usdkrw: float,
    entry: float,
    stop: float,
    target: float,
    risk_pct: float = 0.02,
) -> RiskPlan:
    shares = position_size(account_krw, usdkrw, entry, stop, risk_pct)
    account_usd = account_krw / usdkrw if usdkrw > 0 else 0
    max_loss_usd = account_usd * risk_pct
    risk_per_share = max(entry - stop, 0)
    expected_loss_usd = shares * risk_per_share
    reward_per_share = max(target - entry, 0)
    rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
    return RiskPlan(
        account_usd=account_usd,
        max_loss_usd=max_loss_usd,
        risk_per_share=risk_per_share,
        shares=shares,
        expected_loss_usd=expected_loss_usd,
        expected_loss_krw=expected_loss_usd * usdkrw,
        rr_ratio=rr_ratio,
    )
