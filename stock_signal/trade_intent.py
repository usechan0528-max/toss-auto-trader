from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Literal


TradeSide = Literal["BUY", "SELL", "HOLD"]
IntentStatus = Literal["BLOCKED", "DRY_RUN_READY"]


@dataclass
class TradeIntent:
    symbol: str
    side: TradeSide
    quantity: int
    limit_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    status: IntentStatus
    reason: list[str]
    risk_checks: dict[str, object]
    duplicate_key: str
    safety_notice: str


def decide_side(action: str, current_price: float, stop_loss: float, take_profit_1: float) -> TradeSide:
    if action == "BUY_WATCH":
        return "BUY"
    if action == "NO_TRADE":
        return "HOLD"
    if current_price <= stop_loss or current_price >= take_profit_1:
        return "SELL"
    return "HOLD"


def build_trade_intent(
    *,
    symbol: str,
    action: str,
    current_price: float,
    reference_entry: float,
    stop_loss: float,
    take_profit: list[float],
    recommended_shares: int,
    mode: Literal["disabled", "dry_run", "live_ready"] = "disabled",
    account_usd: float = 0.0,
    daily_realized_pnl_usd: float = 0.0,
    daily_loss_limit_pct: float = 2.0,
    daily_loss_limit_usd: float = 0.0,
    duplicate_keys: set[str] | None = None,
    trade_date: str | None = None,
    data_quality: dict[str, object] | None = None,
    manual_approval_complete: bool = False,
    holding_qty: int | None = None,
    current_session_entries: int = 0,
    max_session_entries: int = 0,
) -> TradeIntent:
    side = decide_side(action, current_price, stop_loss, take_profit[0])
    quantity = max(0, int(recommended_shares))
    holding_quantity = None if holding_qty is None else max(0, int(holding_qty))
    reasons = []
    trade_date = trade_date or date.today().isoformat()
    duplicate_key = f"{trade_date}:{symbol}:{side}:{round(reference_entry, 2)}:{round(stop_loss, 2)}"
    duplicate_keys = duplicate_keys or set()

    pct_loss_limit = account_usd * daily_loss_limit_pct / 100 if account_usd > 0 else 0
    active_loss_limits = [limit for limit in [pct_loss_limit, daily_loss_limit_usd] if limit and limit > 0]
    effective_daily_limit = min(active_loss_limits) if active_loss_limits else 0
    daily_loss_used = abs(min(daily_realized_pnl_usd, 0))
    daily_loss_remaining = max(effective_daily_limit - daily_loss_used, 0) if effective_daily_limit > 0 else 0
    expected_trade_risk = max(0, (reference_entry - stop_loss) * max(0, int(recommended_shares)))

    if action == "NO_TRADE":
        reasons.append("NO_TRADE action")

    if recommended_shares <= 0:
        reasons.append("추천 수량이 0이라 매매 준비 불가")
    if reference_entry <= 0 or stop_loss <= 0:
        reasons.append("가격 기준이 비정상")
    if side == "BUY" and stop_loss >= reference_entry:
        reasons.append("매수 준비 조건에서 손절가가 진입가보다 낮지 않음")
    if side == "SELL" and current_price <= 0:
        reasons.append("매도 준비 조건에서 현재가가 비정상")
    if side == "SELL":
        if holding_quantity is None:
            reasons.append("보유 수량 확인 없이 매도 준비 불가")
        elif holding_quantity <= 0:
            reasons.append("보유 수량이 없어 매도 준비 불가")
        else:
            quantity = min(quantity or holding_quantity, holding_quantity)
    if side == "HOLD":
        reasons.append("현재 신호는 대기 상태")
    if duplicate_key in duplicate_keys and side != "HOLD":
        reasons.append("중복 매매 의도 차단")
    if side != "HOLD" and not manual_approval_complete:
        reasons.append("수동 승인 체크리스트 미완료")
    if side != "HOLD" and max_session_entries > 0 and current_session_entries >= max_session_entries:
        reasons.append("세션 최대 진입 횟수 도달")
    if effective_daily_limit > 0 and daily_loss_used >= effective_daily_limit:
        reasons.append("일일 손실 한도 도달")
    if effective_daily_limit > 0 and expected_trade_risk > daily_loss_remaining and side == "BUY":
        reasons.append("남은 일일 손실 한도보다 예상 위험이 큼")

    if mode == "disabled":
        status: IntentStatus = "BLOCKED"
        reasons.append("실매매 게이트 비활성화")
    elif mode == "dry_run":
        status = "DRY_RUN_READY" if not reasons or reasons == ["현재 신호는 대기 상태"] else "BLOCKED"
    else:
        status = "DRY_RUN_READY" if not reasons or reasons == ["현재 신호는 대기 상태"] else "BLOCKED"

    risk_checks = {
        "account_usd": round(account_usd, 2),
        "daily_realized_pnl_usd": round(daily_realized_pnl_usd, 2),
        "daily_loss_limit_usd": round(effective_daily_limit, 2),
        "daily_loss_used_usd": round(daily_loss_used, 2),
        "daily_loss_remaining_usd": round(daily_loss_remaining, 2),
        "expected_trade_risk_usd": round(expected_trade_risk, 2),
        "duplicate_blocked": duplicate_key in duplicate_keys and side != "HOLD",
        "manual_approval_complete": manual_approval_complete,
        "holding_qty": holding_quantity,
        "sell_quantity_capped": bool(side == "SELL" and holding_quantity is not None and quantity < max(0, int(recommended_shares))),
        "current_session_entries": current_session_entries,
        "max_session_entries": max_session_entries,
    }
    if data_quality:
        risk_checks.update(
            {
                "data_guard_enabled": bool(data_quality.get("guard_enabled", False)),
                "data_age_minutes": data_quality.get("age_minutes"),
                "stale_data_blocked": bool(data_quality.get("stale_blocked", False)),
                "spread_pct": data_quality.get("spread_pct"),
                "wide_spread_blocked": bool(data_quality.get("spread_blocked", False)),
            }
        )

    return TradeIntent(
        symbol=symbol,
        side=side,
        quantity=quantity,
        limit_price=round(reference_entry if side == "BUY" else current_price, 4),
        stop_loss=round(stop_loss, 4),
        take_profit_1=round(take_profit[0], 4),
        take_profit_2=round(take_profit[1], 4),
        status=status,
        reason=reasons,
        risk_checks=risk_checks,
        duplicate_key=duplicate_key,
        safety_notice=(
            "매매 의도만 생성합니다. 실주문은 앱의 실계좌/수동 주문 패널 또는 자동매매 게이트에서 "
            "별도 확인을 통과해야만 전송됩니다."
        ),
    )


def validate_trade_intent(intent: TradeIntent) -> None:
    if not intent.symbol:
        raise ValueError("symbol이 비어 있습니다.")
    if intent.side not in {"BUY", "SELL", "HOLD"}:
        raise ValueError(f"지원하지 않는 side: {intent.side}")
    if intent.quantity < 0:
        raise ValueError("quantity는 음수일 수 없습니다.")
    if intent.limit_price <= 0:
        raise ValueError("limit_price는 0보다 커야 합니다.")
    if intent.stop_loss <= 0:
        raise ValueError("stop_loss는 0보다 커야 합니다.")
    if intent.take_profit_1 <= 0 or intent.take_profit_2 <= 0:
        raise ValueError("익절가는 0보다 커야 합니다.")
    if intent.take_profit_2 < intent.take_profit_1:
        raise ValueError("2차 익절가는 1차 익절가 이상이어야 합니다.")
    if intent.side == "BUY" and intent.stop_loss >= intent.limit_price:
        raise ValueError("BUY 의도에서 손절가는 지정가보다 낮아야 합니다.")
    if not intent.duplicate_key:
        raise ValueError("duplicate_key가 비어 있습니다.")
    required_checks = {
        "daily_loss_limit_usd",
        "daily_loss_used_usd",
        "daily_loss_remaining_usd",
        "expected_trade_risk_usd",
        "duplicate_blocked",
        "manual_approval_complete",
        "current_session_entries",
        "max_session_entries",
    }
    missing = required_checks - set(intent.risk_checks)
    if missing:
        raise ValueError(f"risk_checks 필수 키 누락: {sorted(missing)}")


def trade_intent_to_dict(intent: TradeIntent) -> dict:
    validate_trade_intent(intent)
    return asdict(intent)
