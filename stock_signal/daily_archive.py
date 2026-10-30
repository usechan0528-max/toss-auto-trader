from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python 3.8 fallback
    ZoneInfo = None  # type: ignore[assignment]


KST = ZoneInfo("Asia/Seoul") if ZoneInfo else timezone(timedelta(hours=9))


@dataclass(frozen=True)
class DailyArchivePaths:
    session_events_path: Path
    auto_trade_events_path: Path
    order_intent_events_path: Path
    signal_events_path: Path
    archive_root: Path
    latest_summary_path: Path
    learning_samples_path: Path
    strategy_adjustments_path: Path
    settings_path: Path


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    temp_path.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=KST)
    return parsed


def _event_datetime(event: dict[str, Any]) -> datetime | None:
    for key in ("logged_at", "created_at", "latest_date", "orderedAt", "filledAt"):
        parsed = _parse_datetime(event.get(key))
        if parsed is not None:
            return parsed
    return None


def session_date_for_timestamp(value: datetime | str, *, cutoff_hour_kst: int = 9) -> str:
    parsed = _parse_datetime(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    local = parsed.astimezone(KST)
    session_day = local.date()
    if local.hour < int(cutoff_hour_kst):
        session_day -= timedelta(days=1)
    return session_day.isoformat()


def current_session_date(now: datetime | None = None, *, cutoff_hour_kst: int = 9) -> str:
    return session_date_for_timestamp(now or _now_utc(), cutoff_hour_kst=cutoff_hour_kst)


def _records_for_session(
    records: list[dict[str, Any]],
    session_date: str,
    *,
    cutoff_hour_kst: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        timestamp = _event_datetime(record)
        if timestamp is None:
            continue
        if session_date_for_timestamp(timestamp, cutoff_hour_kst=cutoff_hour_kst) == session_date:
            selected.append(record)
    return selected


def _event_session_dates(records: list[dict[str, Any]], *, cutoff_hour_kst: int) -> set[str]:
    dates: set[str] = set()
    for record in records:
        timestamp = _event_datetime(record)
        if timestamp is not None:
            dates.add(session_date_for_timestamp(timestamp, cutoff_hour_kst=cutoff_hour_kst))
    return dates


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _setting_bool(settings: dict[str, Any], key: str, default: bool) -> bool:
    value = settings.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _setting_float(settings: dict[str, Any], key: str, default: float) -> float:
    return _as_float(settings.get(key), default)


def _setting_int(settings: dict[str, Any], key: str, default: int) -> int:
    return _as_int(settings.get(key), default)


def _hash_sample(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=_json_default)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _extract_order_status(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    execution = raw.get("execution") if isinstance(raw.get("execution"), dict) else {}
    executions = raw.get("executions") if isinstance(raw.get("executions"), list) else []
    first_execution = executions[0] if executions and isinstance(executions[0], dict) else {}

    def pick(*keys: str) -> Any:
        for source in (execution, first_execution, raw):
            for key in keys:
                value = source.get(key) if isinstance(source, dict) else None
                if value not in {None, ""}:
                    return value
        return None

    return {
        "fill_status": str(pick("status", "orderStatus") or "UNKNOWN").upper(),
        "filled_quantity": _as_float(pick("filledQuantity", "executedQuantity", "quantity"), 0.0),
        "average_filled_price": _as_float(pick("averageFilledPrice", "averagePrice", "filledPrice", "price"), 0.0),
        "filled_amount": _as_float(pick("filledAmount", "orderAmount", "amount"), 0.0),
        "commission_usd": _as_float(pick("commission", "commissionAmount"), 0.0),
        "tax_usd": _as_float(pick("tax", "taxAmount"), 0.0),
        "filled_at": pick("filledAt", "executedAt"),
        "ordered_at": raw.get("orderedAt"),
        "raw": raw,
    }


def _query_order_status(broker: Any, broker_order_id: str, errors: list[dict[str, str]]) -> dict[str, Any]:
    if not broker or not broker_order_id:
        return {}
    try:
        result = broker.get_order(broker_order_id)
    except Exception as exc:  # noqa: BLE001 - broker errors should not block archiving
        errors.append({"broker_order_id": broker_order_id, "message": str(exc)})
        return {}
    return result if isinstance(result, dict) else {}


def _session_events_by_client_id(session_events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_client_id: dict[str, dict[str, Any]] = {}
    for event in session_events:
        client_id = event.get("order_client_id")
        if client_id:
            by_client_id[str(client_id)] = event
    return by_client_id


def _find_position_snapshot(session_event: dict[str, Any] | None, symbol: str) -> dict[str, Any]:
    if not session_event:
        return {}
    for row in session_event.get("position_summary") or []:
        if isinstance(row, dict) and str(row.get("symbol", "")).upper() == symbol.upper():
            return row
    return {}


def _candidate_metrics(session_event: dict[str, Any] | None, symbol: str) -> dict[str, Any]:
    if not session_event:
        return {}
    for row in session_event.get("candidate_scores") or []:
        if isinstance(row, dict) and str(row.get("symbol", "")).upper() == symbol.upper():
            return {
                "auto_score": _as_float(row.get("auto_score")),
                "confidence_pct": _as_float(row.get("confidence_pct")),
                "surge_score": _as_float(row.get("surge_score")),
                "lookback_change_pct": _as_float(row.get("lookback_change_pct")),
                "surge_volume_ratio": _as_float(row.get("surge_volume_ratio")),
                "volume_ratio": _as_float(row.get("volume_ratio")),
                "volume_zscore": _as_float(row.get("volume_zscore")),
                "recent_volume": _as_float(row.get("recent_volume")),
                "recent_dollar_volume": _as_float(row.get("recent_dollar_volume")),
                "latest_volume": _as_float(row.get("latest_volume")),
                "latest_dollar_volume": _as_float(row.get("latest_dollar_volume")),
                "risk": row.get("risk"),
                "direction": row.get("direction"),
            }
    auto_selection = session_event.get("auto_selection") if isinstance(session_event.get("auto_selection"), dict) else {}
    return {
        "auto_score": _as_float(auto_selection.get("score")),
        "confidence_pct": _as_float(auto_selection.get("confidence_pct")),
        "surge_score": _as_float(auto_selection.get("surge_score")),
    }


def build_trade_samples(
    *,
    session_date: str,
    auto_trade_events: list[dict[str, Any]],
    session_events: list[dict[str, Any]],
    broker: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    by_client_id = _session_events_by_client_id(session_events)
    errors: list[dict[str, str]] = []
    samples: list[dict[str, Any]] = []
    submitted_statuses = {"LIVE_SUBMITTED", "PAPER_SUBMITTED"}

    for event in auto_trade_events:
        if str(event.get("status")) not in submitted_statuses:
            continue
        order = event.get("order") if isinstance(event.get("order"), dict) else {}
        execution_result = event.get("execution_result") if isinstance(event.get("execution_result"), dict) else {}
        symbol = str(order.get("symbol") or "").upper()
        side = str(order.get("side") or "").upper()
        client_order_id = str(order.get("client_order_id") or execution_result.get("client_order_id") or "")
        broker_order_id = str(execution_result.get("broker_order_id") or "")
        status_raw = _query_order_status(broker, broker_order_id, errors)
        order_status = _extract_order_status(status_raw)
        session_event = by_client_id.get(client_order_id)
        position = _find_position_snapshot(session_event, symbol)
        candidate = _candidate_metrics(session_event, symbol)
        quantity = _as_float(order.get("quantity"))
        filled_qty = order_status["filled_quantity"] or quantity
        avg_fill = order_status["average_filled_price"] or _as_float(order.get("limit_price"))
        fees = order_status["commission_usd"] + order_status["tax_usd"]
        entry_price = _as_float(position.get("entry_price"))

        gross_pnl: float | None = None
        net_pnl: float | None = None
        realized_return_pct: float | None = None
        outcome = "OPEN_OR_UNKNOWN"
        if side == "SELL" and entry_price > 0 and avg_fill > 0 and order_status["fill_status"] == "FILLED":
            gross_pnl = (avg_fill - entry_price) * filled_qty
            net_pnl = gross_pnl - fees
            basis = entry_price * filled_qty
            realized_return_pct = net_pnl / basis * 100 if basis > 0 else None
            if net_pnl > 0:
                outcome = "WIN"
            elif net_pnl < 0:
                outcome = "LOSS"
            else:
                outcome = "BREAKEVEN"

        identity = {
            "session_date": session_date,
            "client_order_id": client_order_id,
            "broker_order_id": broker_order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
        }
        sample = {
            "sample_id": _hash_sample(identity),
            "session_date": session_date,
            "created_at": event.get("created_at"),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order.get("order_type"),
            "limit_price": _as_float(order.get("limit_price")),
            "estimated_notional": _as_float((event.get("checks") or {}).get("estimated_notional")),
            "client_order_id": client_order_id,
            "broker_order_id": broker_order_id,
            "auto_status": event.get("status"),
            "fill_status": order_status["fill_status"],
            "filled_quantity": round(filled_qty, 6),
            "average_filled_price": round(avg_fill, 6),
            "filled_amount": round(order_status["filled_amount"], 6),
            "commission_usd": round(order_status["commission_usd"], 6),
            "tax_usd": round(order_status["tax_usd"], 6),
            "fees_usd": round(fees, 6),
            "filled_at": order_status["filled_at"],
            "ordered_at": order_status["ordered_at"],
            "entry_price": round(entry_price, 6) if entry_price > 0 else None,
            "gross_pnl_usd": round(gross_pnl, 6) if gross_pnl is not None else None,
            "net_pnl_usd": round(net_pnl, 6) if net_pnl is not None else None,
            "realized_return_pct": round(realized_return_pct, 6) if realized_return_pct is not None else None,
            "outcome": outcome,
            "reason": list((session_event or {}).get("reasons") or event.get("reason") or []),
            "auto_exit": (session_event or {}).get("auto_exit"),
            "candidate_metrics": candidate,
            "position_snapshot": position,
        }
        samples.append(sample)
    return samples, errors


def _append_unique_jsonl(path: Path, records: list[dict[str, Any]], *, key: str) -> int:
    existing = read_jsonl(path)
    seen = {str(record.get(key)) for record in existing if record.get(key)}
    added = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            identifier = str(record.get(key) or "")
            if not identifier or identifier in seen:
                continue
            file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            seen.add(identifier)
            added += 1
    return added


def summarize_daily_archive(
    *,
    session_date: str,
    session_events: list[dict[str, Any]],
    auto_trade_events: list[dict[str, Any]],
    order_intent_events: list[dict[str, Any]],
    signal_events: list[dict[str, Any]],
    trade_samples: list[dict[str, Any]],
    order_status_errors: list[dict[str, str]],
) -> dict[str, Any]:
    closed_samples = [sample for sample in trade_samples if sample.get("outcome") in {"WIN", "LOSS", "BREAKEVEN"}]
    wins = [sample for sample in closed_samples if sample.get("outcome") == "WIN"]
    losses = [sample for sample in closed_samples if sample.get("outcome") == "LOSS"]
    net_values = [_as_float(sample.get("net_pnl_usd")) for sample in closed_samples]
    return_values = [_as_float(sample.get("realized_return_pct")) for sample in closed_samples]
    status_counts = Counter(str(event.get("status") or "UNKNOWN") for event in auto_trade_events)
    session_status_counts = Counter(str(event.get("status") or "UNKNOWN") for event in session_events)
    symbols = Counter(str(sample.get("symbol") or "") for sample in trade_samples if sample.get("symbol"))
    candidate_counts = [_as_float(event.get("candidate_scores_count")) for event in session_events]
    liquidity_counts = [_as_float(event.get("liquidity_passed_count")) for event in session_events]

    summary = {
        "session_date": session_date,
        "archived_at": _now_utc().isoformat(),
        "timezone": "Asia/Seoul",
        "cutoff_rule": "timestamps before 09:00 KST belong to the previous trading session",
        "counts": {
            "session_events": len(session_events),
            "auto_trade_events": len(auto_trade_events),
            "order_intent_events": len(order_intent_events),
            "signal_events": len(signal_events),
            "trade_samples": len(trade_samples),
            "closed_trade_samples": len(closed_samples),
            "submitted_orders": int(status_counts.get("LIVE_SUBMITTED", 0) + status_counts.get("PAPER_SUBMITTED", 0)),
            "submit_errors": int(status_counts.get("SUBMIT_ERROR", 0)),
            "submit_rejected": int(status_counts.get("SUBMIT_REJECTED", 0)),
            "auto_blocked": int(status_counts.get("AUTO_BLOCKED", 0)),
        },
        "auto_trade_status_counts": dict(status_counts),
        "session_status_counts": dict(session_status_counts),
        "performance": {
            "wins": len(wins),
            "losses": len(losses),
            "breakevens": len(closed_samples) - len(wins) - len(losses),
            "win_rate_pct": round(len(wins) / len(closed_samples) * 100, 2) if closed_samples else 0.0,
            "total_net_pnl_usd": round(sum(net_values), 6),
            "average_net_pnl_usd": round(mean(net_values), 6) if net_values else 0.0,
            "average_return_pct": round(mean(return_values), 6) if return_values else 0.0,
        },
        "symbols": dict(symbols),
        "candidate_observation": {
            "max_candidate_scores_count": int(max(candidate_counts)) if candidate_counts else 0,
            "avg_candidate_scores_count": round(mean(candidate_counts), 2) if candidate_counts else 0.0,
            "avg_liquidity_passed_count": round(mean(liquidity_counts), 2) if liquidity_counts else 0.0,
        },
        "latest_position_summary": (session_events[-1].get("position_summary") if session_events else []) or [],
        "order_status_errors": order_status_errors,
    }
    return summary


def build_strategy_adjustment(
    samples: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    min_closed_trades: int = 30,
    min_sessions: int = 5,
    auto_apply_enabled: bool = True,
) -> dict[str, Any]:
    closed = [sample for sample in samples if sample.get("outcome") in {"WIN", "LOSS", "BREAKEVEN"}]
    sessions = {str(sample.get("session_date")) for sample in closed if sample.get("session_date")}
    wins = [sample for sample in closed if sample.get("outcome") == "WIN"]
    losses = [sample for sample in closed if sample.get("outcome") == "LOSS"]
    net_values = [_as_float(sample.get("net_pnl_usd")) for sample in closed]
    return_values = [_as_float(sample.get("realized_return_pct")) for sample in closed]
    win_rate = len(wins) / len(closed) * 100 if closed else 0.0
    high_conf_losses = [
        sample
        for sample in losses
        if _as_float((sample.get("candidate_metrics") or {}).get("confidence_pct")) >= 60.0
    ]

    current_probability = _setting_float(settings, "auto_selection_min_probability", 55.0)
    current_volume_ratio = _setting_float(settings, "auto_selection_min_volume_ratio", 1.2)
    current_reward_ratio = _setting_float(settings, "reward_ratio", 1.8)
    max_probability_step = _setting_float(settings, "strategy_learning_max_probability_step", 1.0)
    proposed: dict[str, float] = {}
    reasons: list[str] = []

    status = "INSUFFICIENT_DATA"
    if len(closed) >= int(min_closed_trades) and len(sessions) >= int(min_sessions):
        status = "READY"
        avg_net = mean(net_values) if net_values else 0.0
        high_conf_loss_rate = len(high_conf_losses) / len(closed) * 100 if closed else 0.0
        if win_rate < 45.0 or avg_net <= 0:
            proposed["auto_selection_min_probability"] = min(70.0, current_probability + max(0.25, max_probability_step))
            proposed["auto_selection_min_volume_ratio"] = min(2.5, current_volume_ratio + 0.05)
            reasons.append("closed trades are not profitable enough; tighten entry gates")
        elif win_rate >= 60.0 and avg_net > 0 and high_conf_loss_rate <= 20.0:
            proposed["auto_selection_min_probability"] = max(52.0, current_probability - min(max_probability_step, 0.5))
            if mean(return_values) > 0.25:
                proposed["auto_selection_min_volume_ratio"] = max(1.0, current_volume_ratio - 0.05)
            reasons.append("closed trades are stable; allow slightly more opportunities")

        avg_win = mean([_as_float(sample.get("net_pnl_usd")) for sample in wins]) if wins else 0.0
        avg_loss_abs = abs(mean([_as_float(sample.get("net_pnl_usd")) for sample in losses])) if losses else 0.0
        if avg_loss_abs > 0 and avg_win > 0 and avg_loss_abs > avg_win * 1.2:
            proposed["reward_ratio"] = min(2.5, current_reward_ratio + 0.1)
            reasons.append("average loss is too large versus average win; require better reward/risk")

    clean_proposed = {
        key: round(value, 4)
        for key, value in proposed.items()
        if abs(_setting_float(settings, key, value) - value) > 1e-9
    }
    return {
        "created_at": _now_utc().isoformat(),
        "status": status if clean_proposed else ("NO_CHANGE" if status == "READY" else status),
        "auto_apply_enabled": bool(auto_apply_enabled),
        "metrics": {
            "closed_trades": len(closed),
            "sessions": len(sessions),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "total_net_pnl_usd": round(sum(net_values), 6),
            "average_net_pnl_usd": round(mean(net_values), 6) if net_values else 0.0,
            "average_return_pct": round(mean(return_values), 6) if return_values else 0.0,
            "high_confidence_losses": len(high_conf_losses),
        },
        "thresholds": {
            "min_closed_trades": int(min_closed_trades),
            "min_sessions": int(min_sessions),
        },
        "proposed_settings": clean_proposed,
        "reasons": reasons,
    }


def apply_strategy_adjustment(
    adjustment: dict[str, Any],
    settings: dict[str, Any],
    settings_path: Path,
) -> dict[str, Any]:
    proposed = adjustment.get("proposed_settings") if isinstance(adjustment.get("proposed_settings"), dict) else {}
    if not adjustment.get("auto_apply_enabled") or not proposed:
        adjustment["applied"] = False
        return adjustment
    next_settings = dict(settings)
    for key, value in proposed.items():
        next_settings[key] = value
    _atomic_write_json(settings_path, next_settings)
    adjustment["applied"] = True
    adjustment["applied_settings"] = proposed
    return adjustment


def archive_session(
    session_date: str,
    paths: DailyArchivePaths,
    *,
    settings: dict[str, Any] | None = None,
    broker: Any = None,
    cutoff_hour_kst: int = 9,
    force: bool = False,
) -> dict[str, Any] | None:
    archive_dir = paths.archive_root / session_date
    summary_path = archive_dir / "summary.json"
    if summary_path.exists() and not force:
        return None

    session_events = _records_for_session(read_jsonl(paths.session_events_path), session_date, cutoff_hour_kst=cutoff_hour_kst)
    auto_trade_events = _records_for_session(read_jsonl(paths.auto_trade_events_path), session_date, cutoff_hour_kst=cutoff_hour_kst)
    order_intent_events = _records_for_session(read_jsonl(paths.order_intent_events_path), session_date, cutoff_hour_kst=cutoff_hour_kst)
    signal_events = _records_for_session(read_jsonl(paths.signal_events_path), session_date, cutoff_hour_kst=cutoff_hour_kst)
    if not any([session_events, auto_trade_events, order_intent_events, signal_events]):
        return None

    trade_samples, order_status_errors = build_trade_samples(
        session_date=session_date,
        auto_trade_events=auto_trade_events,
        session_events=session_events,
        broker=broker,
    )
    _write_jsonl(archive_dir / "session_events.jsonl", session_events)
    _write_jsonl(archive_dir / "auto_trade_events.jsonl", auto_trade_events)
    _write_jsonl(archive_dir / "order_intent_events.jsonl", order_intent_events)
    _write_jsonl(archive_dir / "signal_events.jsonl", signal_events)
    _write_jsonl(archive_dir / "trade_samples.jsonl", trade_samples)
    added_samples = _append_unique_jsonl(paths.learning_samples_path, trade_samples, key="sample_id")

    summary = summarize_daily_archive(
        session_date=session_date,
        session_events=session_events,
        auto_trade_events=auto_trade_events,
        order_intent_events=order_intent_events,
        signal_events=signal_events,
        trade_samples=trade_samples,
        order_status_errors=order_status_errors,
    )
    summary["files"] = {
        "archive_dir": str(archive_dir),
        "summary": str(summary_path),
        "trade_samples": str(archive_dir / "trade_samples.jsonl"),
        "learning_samples": str(paths.learning_samples_path),
    }
    summary["learning_samples_added"] = added_samples

    settings = settings or {}
    learning_enabled = _setting_bool(settings, "strategy_learning_enabled", True)
    if learning_enabled:
        adjustment = build_strategy_adjustment(
            read_jsonl(paths.learning_samples_path),
            settings,
            min_closed_trades=_setting_int(settings, "strategy_learning_min_closed_trades", 30),
            min_sessions=_setting_int(settings, "strategy_learning_min_sessions", 5),
            auto_apply_enabled=_setting_bool(settings, "strategy_learning_auto_apply_enabled", True),
        )
        adjustment = apply_strategy_adjustment(adjustment, settings, paths.settings_path)
        _atomic_write_json(paths.strategy_adjustments_path, adjustment)
        summary["strategy_learning"] = adjustment
    else:
        summary["strategy_learning"] = {"status": "DISABLED", "applied": False}

    _atomic_write_json(summary_path, summary)
    _atomic_write_json(paths.latest_summary_path, summary)
    return summary


def maybe_archive_completed_sessions(
    paths: DailyArchivePaths,
    *,
    settings: dict[str, Any] | None = None,
    broker: Any = None,
    now: datetime | None = None,
    cutoff_hour_kst: int = 9,
    force: bool = False,
) -> list[dict[str, Any]]:
    active_session = current_session_date(now or _now_utc(), cutoff_hour_kst=cutoff_hour_kst)
    records = read_jsonl(paths.session_events_path) + read_jsonl(paths.auto_trade_events_path)
    due_dates = sorted(date for date in _event_session_dates(records, cutoff_hour_kst=cutoff_hour_kst) if date < active_session)
    archived: list[dict[str, Any]] = []
    for session_date in due_dates:
        summary = archive_session(
            session_date,
            paths,
            settings=settings,
            broker=broker,
            cutoff_hour_kst=cutoff_hour_kst,
            force=force,
        )
        if summary is not None:
            archived.append(summary)
    return archived
