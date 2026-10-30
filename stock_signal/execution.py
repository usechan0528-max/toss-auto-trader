from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from .trade_intent import TradeIntent


OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["LIMIT", "MARKET"]
ExecutionStatus = Literal["BLOCKED", "READY_FOR_BROKER", "SUBMITTED", "REJECTED"]


@dataclass
class AccountSnapshot:
    account_id: str
    currency: str = "USD"
    cash_buying_power: float = 0.0
    equity: float = 0.0
    fetched_at: str | None = None


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: float | None
    time_in_force: str
    client_order_id: str

    @property
    def estimated_notional(self) -> float:
        if self.limit_price is None:
            return 0.0
        return max(0.0, self.quantity * self.limit_price)


@dataclass
class ExecutionPlan:
    status: ExecutionStatus
    order: OrderRequest | None
    account: AccountSnapshot | None
    checks: dict[str, object]
    reason: list[str]
    safety_notice: str


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    broker_order_id: str | None
    client_order_id: str
    message: str
    raw: dict | None = None


class BrokerExecutionAdapter(Protocol):
    def get_account(self) -> AccountSnapshot:
        ...

    def submit_order(self, order: OrderRequest) -> ExecutionResult:
        ...


def client_order_id_from_intent(intent: TradeIntent, prefix: str = "intent") -> str:
    raw = f"{intent.duplicate_key}:{intent.symbol}:{intent.side}:{intent.quantity}:{intent.limit_price}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    clean_prefix = re.sub(r"[^A-Za-z0-9_-]", "", prefix)[:12] or "intent"
    return f"{clean_prefix}-{digest}"[:36]


def order_request_from_intent(
    intent: TradeIntent,
    *,
    order_type: OrderType = "LIMIT",
    time_in_force: str = "DAY",
) -> OrderRequest | None:
    if intent.side not in {"BUY", "SELL"}:
        return None
    if intent.quantity <= 0:
        return None
    limit_price = None if order_type == "MARKET" else float(intent.limit_price)
    return OrderRequest(
        symbol=intent.symbol,
        side=intent.side,
        quantity=int(intent.quantity),
        order_type=order_type,
        limit_price=limit_price,
        time_in_force=time_in_force,
        client_order_id=client_order_id_from_intent(intent),
    )


def build_execution_plan(
    intent: TradeIntent,
    *,
    account: AccountSnapshot | None = None,
    execution_enabled: bool = False,
    user_confirmation: bool = False,
    order_type: OrderType = "LIMIT",
    time_in_force: str = "DAY",
) -> ExecutionPlan:
    reason: list[str] = []
    order = order_request_from_intent(intent, order_type=order_type, time_in_force=time_in_force)

    checks = {
        "execution_enabled": execution_enabled,
        "user_confirmation": user_confirmation,
        "intent_status": intent.status,
        "intent_side": intent.side,
        "has_order_request": order is not None,
        "account_connected": account is not None,
        "estimated_notional": round(order.estimated_notional, 2) if order else 0.0,
        "buying_power": round(account.cash_buying_power, 2) if account else 0.0,
    }

    if not execution_enabled:
        reason.append("EXECUTION_LAYER_DISABLED")
    if not user_confirmation:
        reason.append("USER_FINAL_CONFIRMATION_REQUIRED")
    if intent.status != "DRY_RUN_READY":
        reason.append("INTENT_NOT_DRY_RUN_READY")
    if intent.side == "HOLD":
        reason.append("HOLD_INTENT_NOT_EXECUTABLE")
    if order is None:
        reason.append("ORDER_REQUEST_NOT_AVAILABLE")
    if account is None:
        reason.append("ACCOUNT_NOT_CONNECTED")
    if order and account and order.side == "BUY" and account.cash_buying_power < order.estimated_notional:
        reason.append("INSUFFICIENT_BUYING_POWER")

    return ExecutionPlan(
        status="READY_FOR_BROKER" if not reason else "BLOCKED",
        order=order,
        account=account,
        checks=checks,
        reason=reason,
        safety_notice=(
            "Execution plan only. A broker adapter must pass this preflight before any live order call is allowed."
        ),
    )


class NoopBrokerExecutionAdapter:
    def get_account(self) -> AccountSnapshot:
        raise NotImplementedError("Broker account API is not connected")

    def submit_order(self, order: OrderRequest) -> ExecutionResult:
        return ExecutionResult(
            status="REJECTED",
            broker_order_id=None,
            client_order_id=order.client_order_id,
            message="Broker execution adapter is disabled",
            raw=None,
        )


def execution_plan_to_dict(plan: ExecutionPlan) -> dict:
    return asdict(plan)
