from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .execution import AccountSnapshot, BrokerExecutionAdapter, ExecutionPlan, ExecutionResult, OrderRequest


AutoTradeMode = Literal["disabled", "paper", "live"]
AutoTradeStatus = Literal[
    "AUTO_BLOCKED",
    "PAPER_SUBMITTED",
    "LIVE_SUBMITTED",
    "SUBMIT_REJECTED",
    "SUBMIT_ERROR",
]
AutoTradeState = Literal[
    "AUTO_DISABLED",
    "KILL_SWITCH_ACTIVE",
    "PLAN_NOT_READY",
    "READY_TO_SUBMIT",
    "ORDER_SUBMITTED",
    "ORDER_REJECTED",
    "ERROR",
]


@dataclass
class AutoTradeConfig:
    enabled: bool = False
    mode: AutoTradeMode = "disabled"
    kill_switch: bool = True
    max_order_notional_usd: float = 0.0
    allowed_symbols: list[str] | None = None


@dataclass
class AutoTradeDecision:
    status: AutoTradeStatus
    state: AutoTradeState
    mode: AutoTradeMode
    order: OrderRequest | None
    execution_result: ExecutionResult | None
    checks: dict[str, object]
    reason: list[str]
    safety_notice: str


class PaperBrokerExecutionAdapter:
    def __init__(self, account: AccountSnapshot | None = None) -> None:
        self.account = account or AccountSnapshot(account_id="paper", cash_buying_power=0.0, equity=0.0)

    def get_account(self) -> AccountSnapshot:
        return self.account

    def submit_order(self, order: OrderRequest) -> ExecutionResult:
        broker_order_id = f"paper-{order.client_order_id}"
        return ExecutionResult(
            status="SUBMITTED",
            broker_order_id=broker_order_id,
            client_order_id=order.client_order_id,
            message="Paper order accepted. No live broker order was sent.",
            raw={
                "paper": True,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "order_type": order.order_type,
                "limit_price": order.limit_price,
                "estimated_notional": round(order.estimated_notional, 2),
            },
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_fingerprint(decision: dict) -> str:
    order = decision.get("order") or {}
    raw = json.dumps(
        {
            "status": decision.get("status"),
            "state": decision.get("state"),
            "mode": decision.get("mode"),
            "client_order_id": order.get("client_order_id"),
            "reason": decision.get("reason", []),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def submitted_order_ids_from_events(path: Path) -> set[str]:
    if not path.exists():
        return set()
    submitted: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = str(event.get("status", ""))
        order = event.get("order") or {}
        client_order_id = order.get("client_order_id")
        if client_order_id and status in {"PAPER_SUBMITTED", "LIVE_SUBMITTED"}:
            submitted.add(str(client_order_id))
    return submitted


def run_auto_trade(
    plan: ExecutionPlan,
    config: AutoTradeConfig,
    *,
    adapter: BrokerExecutionAdapter | None = None,
    submitted_client_order_ids: set[str] | None = None,
) -> AutoTradeDecision:
    order = plan.order
    submitted_client_order_ids = submitted_client_order_ids or set()
    allowed_symbols = set(config.allowed_symbols or [])
    checks = {
        "enabled": bool(config.enabled),
        "mode": config.mode,
        "kill_switch": bool(config.kill_switch),
        "plan_status": plan.status,
        "has_order": order is not None,
        "estimated_notional": round(order.estimated_notional, 2) if order else 0.0,
        "max_order_notional_usd": round(max(0.0, float(config.max_order_notional_usd)), 2),
        "allowed_symbols": sorted(allowed_symbols),
        "adapter_connected": adapter is not None,
        "already_submitted": bool(order and order.client_order_id in submitted_client_order_ids),
    }
    reason: list[str] = []

    if not config.enabled or config.mode == "disabled":
        reason.append("AUTO_TRADE_DISABLED")
    if config.kill_switch:
        reason.append("AUTO_TRADE_KILL_SWITCH_ACTIVE")
    if plan.status != "READY_FOR_BROKER":
        reason.append("EXECUTION_PLAN_NOT_READY")
    if order is None:
        reason.append("ORDER_REQUEST_NOT_AVAILABLE")
    if order and allowed_symbols and order.symbol not in allowed_symbols:
        reason.append("SYMBOL_NOT_ALLOWED")
    if order and config.max_order_notional_usd > 0 and order.estimated_notional > config.max_order_notional_usd:
        reason.append("MAX_ORDER_NOTIONAL_EXCEEDED")
    if order and order.client_order_id in submitted_client_order_ids:
        reason.append("DUPLICATE_AUTO_ORDER_BLOCKED")
    if config.mode == "live" and adapter is None:
        reason.append("LIVE_BROKER_ADAPTER_NOT_CONNECTED")

    if reason:
        if "AUTO_TRADE_KILL_SWITCH_ACTIVE" in reason:
            state: AutoTradeState = "KILL_SWITCH_ACTIVE"
        elif "AUTO_TRADE_DISABLED" in reason:
            state = "AUTO_DISABLED"
        else:
            state = "PLAN_NOT_READY"
        return AutoTradeDecision(
            status="AUTO_BLOCKED",
            state=state,
            mode=config.mode,
            order=order,
            execution_result=None,
            checks=checks,
            reason=reason,
            safety_notice="Auto-trade gate blocked the order. No broker order was sent.",
        )

    submit_adapter = adapter
    if config.mode == "paper" and submit_adapter is None:
        submit_adapter = PaperBrokerExecutionAdapter(plan.account)
        checks["adapter_connected"] = True

    if submit_adapter is None or order is None:
        return AutoTradeDecision(
            status="AUTO_BLOCKED",
            state="PLAN_NOT_READY",
            mode=config.mode,
            order=order,
            execution_result=None,
            checks=checks,
            reason=["AUTO_TRADE_ADAPTER_NOT_AVAILABLE"],
            safety_notice="Auto-trade adapter is not available. No broker order was sent.",
        )

    try:
        result = submit_adapter.submit_order(order)
    except Exception as exc:
        return AutoTradeDecision(
            status="SUBMIT_ERROR",
            state="ERROR",
            mode=config.mode,
            order=order,
            execution_result=None,
            checks=checks,
            reason=[f"SUBMIT_EXCEPTION: {exc}"],
            safety_notice="Auto-trade submit raised an exception. Verify broker adapter before enabling live mode.",
        )

    if result.status == "SUBMITTED":
        status: AutoTradeStatus = "PAPER_SUBMITTED" if config.mode == "paper" else "LIVE_SUBMITTED"
        state: AutoTradeState = "ORDER_SUBMITTED"
        reason = []
    else:
        status = "SUBMIT_REJECTED"
        state = "ORDER_REJECTED"
        reason = [result.message]

    return AutoTradeDecision(
        status=status,
        state=state,
        mode=config.mode,
        order=order,
        execution_result=result,
        checks=checks,
        reason=reason,
        safety_notice=(
            "Paper mode only records a simulated order. Live mode requires a separately reviewed broker adapter."
            if config.mode == "paper"
            else "Live adapter path was used. Confirm broker logs and account state."
        ),
    )


def auto_trade_decision_to_dict(decision: AutoTradeDecision) -> dict:
    payload = asdict(decision)
    payload["created_at"] = _now_iso()
    return payload


def append_auto_trade_event(decision: dict, path: Path) -> None:
    fingerprint = _event_fingerprint(decision)
    previous_fingerprint = None
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines:
            try:
                previous_fingerprint = json.loads(lines[-1]).get("fingerprint")
            except json.JSONDecodeError:
                previous_fingerprint = None
    if previous_fingerprint == fingerprint:
        return

    event = dict(decision)
    event["fingerprint"] = fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")
