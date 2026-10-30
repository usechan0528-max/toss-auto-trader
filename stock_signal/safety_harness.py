from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .trade_intent import TradeIntent


DAILY_LOSS_REASON = "DAILY_LOSS_LIMIT_REACHED"
DAILY_LOSS_ALERT = "daily loss limit reached; action forced to NO_TRADE"
DAILY_PROFIT_LOCK_REASON = "DAILY_PROFIT_LOCK_STOP_REACHED"
DAILY_PROFIT_LOCK_ALERT = "daily profit lock stop reached; action forced to NO_TRADE"
DUPLICATE_ORDER_REASON = "DUPLICATE_ORDER_BLOCKED"
STALE_DATA_REASON = "STALE_DATA_BLOCKED"
STALE_DATA_ALERT = "data is stale; action forced to NO_TRADE"
WIDE_SPREAD_REASON = "WIDE_SPREAD_BLOCKED"
WIDE_SPREAD_ALERT = "spread is too wide; action forced to NO_TRADE"


@dataclass
class DailyLossLimit:
    account_usd: float
    realized_pnl_usd: float = 0.0
    max_loss_pct: float = 0.02
    max_loss_usd: float = 0.0

    @property
    def pct_limit_usd(self) -> float:
        if self.account_usd <= 0 or self.max_loss_pct <= 0:
            return 0.0
        return self.account_usd * self.max_loss_pct

    @property
    def active_limits(self) -> list[float]:
        limits = []
        if self.pct_limit_usd > 0:
            limits.append(self.pct_limit_usd)
        if self.max_loss_usd > 0:
            limits.append(self.max_loss_usd)
        return limits

    @property
    def threshold_usd(self) -> float:
        limits = self.active_limits
        return min(limits) if limits else 0.0

    def is_reached(self) -> bool:
        threshold = self.threshold_usd
        return threshold > 0 and self.realized_pnl_usd <= -threshold


@dataclass
class DailyProfitLock:
    account_usd: float
    realized_pnl_usd: float = 0.0
    trigger_pct: float = 20.0
    giveback_pct: float = 10.0

    @property
    def realized_pct(self) -> float:
        if self.account_usd <= 0:
            return 0.0
        return self.realized_pnl_usd / self.account_usd * 100


class DailyProfitLockLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read_state(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def write_state(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f".{self.path.name}.tmp")
        temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.path)


@dataclass
class DataQualityGuard:
    enabled: bool = False
    max_age_minutes: float = 5.0
    max_spread_pct: float = 2.0
    now: datetime | None = None


def parse_payload_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def data_age_minutes(latest_date: Any, now: datetime | None = None) -> float | None:
    parsed = parse_payload_datetime(latest_date)
    if parsed is None:
        return None
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    return max(0.0, (current - parsed).total_seconds() / 60)


def freshness_status(age_minutes: float | None, max_age_minutes: float) -> str:
    if age_minutes is None:
        return "UNKNOWN"
    if age_minutes <= max_age_minutes:
        return "FRESH"
    if age_minutes <= max_age_minutes * 3:
        return "DELAYED"
    return "STALE"


def apply_data_quality_harness(payload: dict[str, Any], guard: DataQualityGuard) -> dict[str, Any]:
    guarded = dict(payload)
    spread_pct = float(guarded.get("spread_pct", 0) or 0)
    age = data_age_minutes(guarded.get("latest_date"), guard.now)
    data_quality = {
        "guard_enabled": guard.enabled,
        "age_minutes": round(age, 2) if age is not None else None,
        "max_age_minutes": guard.max_age_minutes,
        "freshness_status": freshness_status(age, guard.max_age_minutes),
        "spread_pct": round(spread_pct, 4),
        "max_spread_pct": guard.max_spread_pct,
        "stale_blocked": False,
        "spread_blocked": False,
    }
    guarded["data_quality"] = data_quality

    if not guard.enabled:
        return guarded

    reasons = list(guarded.get("reason", []))
    alerts = list(guarded.get("alerts", []))
    risk_flags = list(guarded.get("risk_flags", []))
    should_block = False

    if age is None or age > guard.max_age_minutes:
        should_block = True
        data_quality["stale_blocked"] = True
        reasons.append(STALE_DATA_REASON)
        alerts.append(STALE_DATA_ALERT)
        risk_flags.append(f"data age {age if age is not None else 'unknown'} min > {guard.max_age_minutes} min")

    if spread_pct > guard.max_spread_pct:
        should_block = True
        data_quality["spread_blocked"] = True
        reasons.append(WIDE_SPREAD_REASON)
        alerts.append(WIDE_SPREAD_ALERT)
        risk_flags.append(f"spread {spread_pct:.2f}% > {guard.max_spread_pct:.2f}%")

    if should_block:
        guarded["action"] = "NO_TRADE"
        guarded["recommended_shares"] = 0
        guarded["reason"] = reasons
        guarded["alerts"] = alerts
        guarded["risk_flags"] = risk_flags

    return guarded


def apply_daily_loss_harness(payload: dict[str, Any], limit: DailyLossLimit) -> dict[str, Any]:
    guarded = dict(payload)
    if not limit.is_reached():
        return guarded

    guarded["action"] = "NO_TRADE"
    guarded["recommended_shares"] = 0
    guarded["reason"] = list(guarded.get("reason", [])) + [DAILY_LOSS_REASON]
    guarded["alerts"] = list(guarded.get("alerts", [])) + [DAILY_LOSS_ALERT]
    guarded["risk_flags"] = list(guarded.get("risk_flags", [])) + [
        f"daily loss {limit.realized_pnl_usd:.2f} USD <= -{limit.threshold_usd:.2f} USD"
    ]
    return guarded


def apply_daily_profit_lock_harness(
    payload: dict[str, Any],
    lock: DailyProfitLock,
    ledger: DailyProfitLockLedger,
) -> dict[str, Any]:
    guarded = dict(payload)
    trade_date = trade_date_from_payload(guarded)
    state = ledger.read_state()
    day_state = dict(state.get(trade_date) or {})
    realized_pct = round(lock.realized_pct, 4)
    trigger_pct = round(float(lock.trigger_pct), 4)
    giveback_pct = round(max(0.0, float(lock.giveback_pct)), 4)

    active = bool(day_state.get("active", False)) or realized_pct >= trigger_pct
    previous_peak = float(day_state.get("peak_pct", realized_pct if active else 0.0) or 0.0)
    peak_pct = max(previous_peak, realized_pct) if active else previous_peak
    stop_pct = peak_pct - giveback_pct if active and giveback_pct > 0 else None
    stopped = bool(active and stop_pct is not None and realized_pct <= stop_pct)

    day_state.update(
        {
            "trade_date": trade_date,
            "active": active,
            "trigger_pct": trigger_pct,
            "giveback_pct": giveback_pct,
            "peak_pct": round(peak_pct, 4),
            "stop_pct": round(stop_pct, 4) if stop_pct is not None else None,
            "realized_pct": realized_pct,
            "realized_pnl_usd": round(float(lock.realized_pnl_usd), 2),
            "stopped": stopped,
        }
    )
    state[trade_date] = day_state
    ledger.write_state(state)
    guarded["daily_profit_lock"] = day_state

    if not stopped:
        return guarded

    guarded["action"] = "NO_TRADE"
    guarded["recommended_shares"] = 0
    guarded["reason"] = list(guarded.get("reason", [])) + [DAILY_PROFIT_LOCK_REASON]
    guarded["alerts"] = list(guarded.get("alerts", [])) + [DAILY_PROFIT_LOCK_ALERT]
    guarded["risk_flags"] = list(guarded.get("risk_flags", [])) + [
        f"daily profit lock {realized_pct:.2f}% <= {float(stop_pct):.2f}% after peak {peak_pct:.2f}%"
    ]
    return guarded


def trade_date_from_payload(payload: dict[str, Any]) -> str:
    latest_date = str(payload.get("latest_date", ""))
    if len(latest_date) >= 10 and latest_date[4] == "-" and latest_date[7] == "-":
        return latest_date[:10]
    return date.today().isoformat()


def duplicate_order_key(
    *,
    symbol: str,
    side: str,
    reference_entry: float,
    trade_date: str,
    price_precision: int = 2,
) -> str:
    price_zone = round(float(reference_entry), price_precision)
    return f"{symbol.upper()}|{side.upper()}|{price_zone:.{price_precision}f}|{trade_date}"


class DuplicateOrderLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def contains(self, key: str) -> bool:
        return any(event.get("duplicate_key") == key for event in self.read_events())

    def count_entries(
        self,
        *,
        trade_date: str,
        symbol: str | None = None,
        sides: set[str] | None = None,
    ) -> int:
        sides = {side.upper() for side in (sides or {"BUY", "SELL"})}
        count = 0
        for event in self.read_events():
            if event.get("trade_date") != trade_date:
                continue
            if symbol and str(event.get("symbol", "")).upper() != symbol.upper():
                continue
            if str(event.get("side", "")).upper() not in sides:
                continue
            if event.get("status") == "DRY_RUN_READY":
                count += 1
        return count

    def record(self, key: str, intent: TradeIntent, payload: dict[str, Any]) -> None:
        event = {
            "duplicate_key": key,
            "symbol": intent.symbol,
            "side": intent.side,
            "price_zone": round(float(payload.get("reference_entry", intent.limit_price)), 2),
            "trade_date": trade_date_from_payload(payload),
            "status": intent.status,
            "latest_date": payload.get("latest_date"),
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")


def apply_duplicate_order_harness(
    intent: TradeIntent,
    payload: dict[str, Any],
    ledger: DuplicateOrderLedger,
) -> TradeIntent:
    if intent.status != "DRY_RUN_READY" or intent.side == "HOLD":
        return intent

    key = duplicate_order_key(
        symbol=intent.symbol,
        side=intent.side,
        reference_entry=float(payload.get("reference_entry", intent.limit_price)),
        trade_date=trade_date_from_payload(payload),
    )
    intent.duplicate_key = key
    if ledger.contains(key):
        intent.status = "BLOCKED"
        intent.reason = list(intent.reason) + [DUPLICATE_ORDER_REASON]
        intent.risk_checks = {**intent.risk_checks, "duplicate_blocked": True}
        return intent

    intent.risk_checks = {**intent.risk_checks, "duplicate_blocked": False}
    ledger.record(key, intent, payload)
    return intent
