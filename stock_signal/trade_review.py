from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Literal


TradeSide = Literal["BUY", "SELL"]
TradeOutcome = Literal["WIN", "LOSS", "BREAKEVEN"]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def _int(value: Any, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number


def _round(value: Any, digits: int = 4) -> float:
    return round(_float(value), digits)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_record(parts: Iterable[Any]) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _direction_from_move(move_pct: float, threshold_pct: float = 0.03) -> str:
    if move_pct > threshold_pct:
        return "UP"
    if move_pct < -threshold_pct:
        return "DOWN"
    return "FLAT"


def _outcome(net_pnl_usd: float, break_even_usd: float = 0.01) -> TradeOutcome:
    if net_pnl_usd > break_even_usd:
        return "WIN"
    if net_pnl_usd < -break_even_usd:
        return "LOSS"
    return "BREAKEVEN"


def _decision_quality(action: str, side: TradeSide, outcome: TradeOutcome) -> str:
    predicted_entry = action == "BUY_WATCH"
    if predicted_entry and outcome == "WIN":
        return "PREDICTED_ENTRY_WON"
    if predicted_entry and outcome == "LOSS":
        return "PREDICTED_ENTRY_LOST"
    if predicted_entry and outcome == "BREAKEVEN":
        return "PREDICTED_ENTRY_FLAT"
    if not predicted_entry and outcome == "WIN":
        return "MISSED_OR_MANUAL_WIN"
    if not predicted_entry and outcome == "LOSS":
        return "BLOCK_WOULD_HAVE_HELPED"
    return f"{side}_FLAT_WITHOUT_ENTRY_SIGNAL"


def _lesson_tags(record: dict[str, Any]) -> list[str]:
    outcome = record["actual"]["outcome"]
    comparison = record["comparison"]
    prediction = record["prediction"]
    tags: list[str] = []

    if outcome == "LOSS":
        tags.append("loss_review")
    elif outcome == "WIN":
        tags.append("winner_review")
    else:
        tags.append("breakeven_review")

    if comparison["decision_quality"] == "PREDICTED_ENTRY_LOST":
        tags.append("formula_or_filter_needs_review")
    if comparison["direction_correct"]:
        tags.append("direction_correct")
    else:
        tags.append("direction_wrong")
    if _float(prediction.get("ml_probability_up")) >= 0.65 and outcome == "LOSS":
        tags.append("high_ml_confidence_loss")
    if _float(prediction.get("formula_probability_up")) >= 0.65 and outcome == "LOSS":
        tags.append("high_formula_confidence_loss")
    if abs(_float(comparison.get("expected_move_error_pct"))) >= 1.0:
        tags.append("expected_move_error_large")
    if comparison.get("hit_stop_loss"):
        tags.append("stop_loss_hit")
    if comparison.get("hit_take_profit_1"):
        tags.append("take_profit_hit")
    return tags


def build_trade_review_record(
    *,
    signal_payload: dict[str, Any],
    trade_intent: dict[str, Any] | None,
    execution_plan: dict[str, Any] | None,
    side: TradeSide,
    entry_price: float,
    exit_price: float,
    quantity: int,
    fees_usd: float = 0.0,
    opened_at: str | None = None,
    closed_at: str | None = None,
    close_reason: str = "",
    user_note: str = "",
) -> dict[str, Any]:
    if side not in {"BUY", "SELL"}:
        raise ValueError(f"unsupported side: {side}")
    if entry_price <= 0 or exit_price <= 0:
        raise ValueError("entry_price and exit_price must be positive")
    if quantity <= 0:
        raise ValueError("quantity must be positive")

    trade_intent = trade_intent or {}
    execution_plan = execution_plan or {}
    metrics = signal_payload.get("prediction_metrics") or {}
    components = metrics.get("components") or {}

    gross_pnl_usd = (exit_price - entry_price) * quantity if side == "BUY" else (entry_price - exit_price) * quantity
    net_pnl_usd = gross_pnl_usd - max(0.0, fees_usd)
    realized_move_pct = ((exit_price / entry_price) - 1) * 100
    if side == "SELL":
        realized_move_pct *= -1
    outcome = _outcome(net_pnl_usd)

    stop_loss = _float(signal_payload.get("stop_loss"))
    take_profit = signal_payload.get("take_profit") or []
    take_profit_1 = _float(take_profit[0]) if take_profit else 0.0
    reference_entry = _float(signal_payload.get("reference_entry"), entry_price)
    risk_per_share = abs(reference_entry - stop_loss) if stop_loss > 0 else 0.0
    risk_usd = risk_per_share * quantity
    realized_r = net_pnl_usd / risk_usd if risk_usd > 0 else 0.0

    formula_expected_move = _float(metrics.get("expected_move_pct"))
    actual_direction = _direction_from_move(realized_move_pct)
    expected_direction = _direction_from_move(formula_expected_move)
    direction_correct = (
        expected_direction == actual_direction
        if expected_direction != "FLAT" and actual_direction != "FLAT"
        else outcome != "LOSS"
    )

    ml_probability_up = _float(signal_payload.get("confidence"), 0.5)
    formula_probability_up = _float(metrics.get("probability_up"), 0.5)
    actual_up_binary = 1.0 if actual_direction == "UP" else 0.0
    action = str(signal_payload.get("action", "UNKNOWN"))

    hit_stop_loss = False
    hit_take_profit_1 = False
    if stop_loss > 0:
        hit_stop_loss = exit_price <= stop_loss if side == "BUY" else exit_price >= stop_loss
    if take_profit_1 > 0:
        hit_take_profit_1 = exit_price >= take_profit_1 if side == "BUY" else exit_price <= take_profit_1

    closed_at = closed_at or _now_iso()
    opened_at = opened_at or str(signal_payload.get("latest_date") or signal_payload.get("candle_latest_date") or "")
    review_id = _hash_record(
        [
            signal_payload.get("symbol"),
            side,
            entry_price,
            exit_price,
            quantity,
            opened_at,
            closed_at,
        ]
    )

    record: dict[str, Any] = {
        "schema_version": 1,
        "review_id": review_id,
        "created_at": _now_iso(),
        "symbol": signal_payload.get("symbol"),
        "prediction": {
            "action": action,
            "ml_probability_up": round(ml_probability_up, 4),
            "formula_probability_up": round(formula_probability_up, 4),
            "formula_score": _round(metrics.get("score")),
            "formula_expected_move_pct": _round(formula_expected_move),
            "formula_projected_price": _round(metrics.get("projected_price")),
            "formula_projected_low": _round(metrics.get("projected_low")),
            "formula_projected_high": _round(metrics.get("projected_high")),
            "formula_components": {str(key): _round(value) for key, value in components.items()},
            "formula_reasons": list(metrics.get("reasons") or []),
            "current_price": _round(signal_payload.get("current_price")),
            "reference_entry": _round(reference_entry),
            "entry_zone": signal_payload.get("entry_zone") or [],
            "stop_loss": _round(stop_loss),
            "take_profit": signal_payload.get("take_profit") or [],
            "rr_ratio": _round(signal_payload.get("rr_ratio"), 2),
            "spread_pct": _round(signal_payload.get("spread_pct")),
            "risk_flags": list(signal_payload.get("risk_flags") or []),
            "alerts": list(signal_payload.get("alerts") or []),
            "priority": signal_payload.get("priority") or {},
            "data_quality": signal_payload.get("data_quality") or {},
            "latest_date": signal_payload.get("latest_date"),
        },
        "intent": {
            "status": trade_intent.get("status"),
            "side": trade_intent.get("side"),
            "quantity": trade_intent.get("quantity"),
            "reason": list(trade_intent.get("reason") or []),
            "risk_checks": trade_intent.get("risk_checks") or {},
            "duplicate_key": trade_intent.get("duplicate_key"),
        },
        "execution": {
            "status": execution_plan.get("status"),
            "reason": list(execution_plan.get("reason") or []),
            "checks": execution_plan.get("checks") or {},
        },
        "actual": {
            "side": side,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "quantity": int(quantity),
            "gross_pnl_usd": round(gross_pnl_usd, 4),
            "fees_usd": round(max(0.0, fees_usd), 4),
            "net_pnl_usd": round(net_pnl_usd, 4),
            "return_pct": round(realized_move_pct, 4),
            "risk_usd": round(risk_usd, 4),
            "realized_r": round(realized_r, 4),
            "outcome": outcome,
            "opened_at": opened_at,
            "closed_at": closed_at,
            "close_reason": close_reason,
            "user_note": user_note,
        },
        "comparison": {
            "actual_direction": actual_direction,
            "formula_expected_direction": expected_direction,
            "direction_correct": bool(direction_correct),
            "expected_move_error_pct": round(realized_move_pct - formula_expected_move, 4),
            "ml_brier_error": round((ml_probability_up - actual_up_binary) ** 2, 4),
            "formula_brier_error": round((formula_probability_up - actual_up_binary) ** 2, 4),
            "hit_stop_loss": bool(hit_stop_loss),
            "hit_take_profit_1": bool(hit_take_profit_1),
            "decision_quality": _decision_quality(action, side, outcome),
        },
    }
    record["llm_context"] = {
        "summary": _record_summary(record),
        "lesson_tags": _lesson_tags(record),
        "review_question": (
            "Compare formula_components, risk filters, spread/data_quality, and actual return. "
            "Suggest only evidence-backed formula or filter changes."
        ),
    }
    return record


def _record_summary(record: dict[str, Any]) -> str:
    prediction = record["prediction"]
    actual = record["actual"]
    comparison = record["comparison"]
    return (
        f"{record.get('symbol')} {actual['side']} closed as {actual['outcome']} "
        f"({actual['net_pnl_usd']:+.2f} USD, {actual['return_pct']:+.2f}%, {actual['realized_r']:+.2f}R). "
        f"Signal action={prediction['action']}, ML up={prediction['ml_probability_up']:.1%}, "
        f"formula up={prediction['formula_probability_up']:.1%}, expected move={prediction['formula_expected_move_pct']:+.2f}%. "
        f"Direction correct={comparison['direction_correct']}, quality={comparison['decision_quality']}."
    )


def append_trade_review(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_trade_reviews(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def summarize_trade_reviews(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "net_pnl_usd": 0.0,
            "avg_realized_r": 0.0,
            "direction_accuracy_pct": 0.0,
            "component_deltas": {},
        }

    wins = [record for record in records if record.get("actual", {}).get("outcome") == "WIN"]
    losses = [record for record in records if record.get("actual", {}).get("outcome") == "LOSS"]
    net_pnl = sum(_float(record.get("actual", {}).get("net_pnl_usd")) for record in records)
    avg_r = mean(_float(record.get("actual", {}).get("realized_r")) for record in records)
    direction_accuracy = mean(1.0 if record.get("comparison", {}).get("direction_correct") else 0.0 for record in records)

    component_keys = sorted(
        {
            key
            for record in records
            for key in (record.get("prediction", {}).get("formula_components") or {}).keys()
        }
    )
    component_deltas: dict[str, dict[str, float]] = {}
    for key in component_keys:
        win_values = [_float(record["prediction"]["formula_components"].get(key)) for record in wins if key in record.get("prediction", {}).get("formula_components", {})]
        loss_values = [_float(record["prediction"]["formula_components"].get(key)) for record in losses if key in record.get("prediction", {}).get("formula_components", {})]
        component_deltas[key] = {
            "avg_win": round(mean(win_values), 4) if win_values else 0.0,
            "avg_loss": round(mean(loss_values), 4) if loss_values else 0.0,
            "win_minus_loss": round((mean(win_values) if win_values else 0.0) - (mean(loss_values) if loss_values else 0.0), 4),
        }

    return {
        "total_trades": len(records),
        "wins": len(wins),
        "losses": len(losses),
        "breakevens": len(records) - len(wins) - len(losses),
        "win_rate_pct": round(len(wins) / len(records) * 100, 2),
        "net_pnl_usd": round(net_pnl, 2),
        "avg_realized_r": round(avg_r, 4),
        "direction_accuracy_pct": round(direction_accuracy * 100, 2),
        "high_confidence_losses": sum(
            1
            for record in records
            if record.get("actual", {}).get("outcome") == "LOSS"
            and (
                _float(record.get("prediction", {}).get("ml_probability_up")) >= 0.65
                or _float(record.get("prediction", {}).get("formula_probability_up")) >= 0.65
            )
        ),
        "component_deltas": component_deltas,
    }


def build_llm_trade_review_document(records: list[dict[str, Any]], max_records: int = 30) -> str:
    summary = summarize_trade_reviews(records)
    lines = [
        "# LLM Trade Review Dataset",
        "",
        "Purpose: compare the app prediction, formula components, safety filters, and realized trade outcomes. Use this file to suggest evidence-backed improvements only.",
        "",
        f"Generated at: {_now_iso()}",
        "",
        "## Aggregate",
        "",
        f"- total_trades: {summary['total_trades']}",
        f"- wins/losses/breakevens: {summary.get('wins', 0)}/{summary.get('losses', 0)}/{summary.get('breakevens', 0)}",
        f"- win_rate_pct: {summary['win_rate_pct']}",
        f"- net_pnl_usd: {summary['net_pnl_usd']}",
        f"- avg_realized_r: {summary['avg_realized_r']}",
        f"- direction_accuracy_pct: {summary['direction_accuracy_pct']}",
        f"- high_confidence_losses: {summary.get('high_confidence_losses', 0)}",
        "",
        "## Formula Component Win/Loss Averages",
        "",
    ]
    if summary["component_deltas"]:
        for key, values in summary["component_deltas"].items():
            lines.append(
                f"- {key}: avg_win={values['avg_win']:+.4f}, avg_loss={values['avg_loss']:+.4f}, win_minus_loss={values['win_minus_loss']:+.4f}"
            )
    else:
        lines.append("- no component data yet")

    lines.extend(
        [
            "",
            "## Review Instructions For LLM",
            "",
            "- Do not infer live-trading readiness from this file.",
            "- Compare winners vs losers by formula component, data quality, spread, alerts, and risk filters.",
            "- Suggest formula weight changes only when repeated records support the same direction.",
            "- Prefer adding or tightening filters before increasing aggressiveness.",
            "",
            "## Trade Records",
            "",
        ]
    )

    for record in records[-max_records:]:
        prediction = record.get("prediction", {})
        actual = record.get("actual", {})
        comparison = record.get("comparison", {})
        lines.extend(
            [
                f"### {record.get('symbol')} | {actual.get('closed_at', '')}",
                "",
                f"- summary: {record.get('llm_context', {}).get('summary', '')}",
                f"- tags: {', '.join(record.get('llm_context', {}).get('lesson_tags', []))}",
                f"- prediction: action={prediction.get('action')}, ml_up={_float(prediction.get('ml_probability_up')):.1%}, formula_up={_float(prediction.get('formula_probability_up')):.1%}, formula_score={prediction.get('formula_score')}, expected_move_pct={prediction.get('formula_expected_move_pct')}",
                f"- actual: side={actual.get('side')}, entry={actual.get('entry_price')}, exit={actual.get('exit_price')}, qty={actual.get('quantity')}, net_pnl_usd={actual.get('net_pnl_usd')}, return_pct={actual.get('return_pct')}, realized_r={actual.get('realized_r')}, outcome={actual.get('outcome')}",
                f"- comparison: direction_correct={comparison.get('direction_correct')}, expected_move_error_pct={comparison.get('expected_move_error_pct')}, ml_brier_error={comparison.get('ml_brier_error')}, formula_brier_error={comparison.get('formula_brier_error')}, quality={comparison.get('decision_quality')}",
                f"- close_reason: {actual.get('close_reason', '')}",
                f"- user_note: {actual.get('user_note', '')}",
                "- formula_components:",
            ]
        )
        components = prediction.get("formula_components") or {}
        if components:
            for key, value in components.items():
                lines.append(f"  - {key}: {value}")
        else:
            lines.append("  - none")
        lines.extend(
            [
                "- compact_json:",
                "```json",
                json.dumps(
                    {
                        "review_id": record.get("review_id"),
                        "symbol": record.get("symbol"),
                        "prediction": prediction,
                        "actual": actual,
                        "comparison": comparison,
                        "lesson_tags": record.get("llm_context", {}).get("lesson_tags", []),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_llm_trade_review_document(records: list[dict[str, Any]], path: Path, max_records: int = 30) -> str:
    document = build_llm_trade_review_document(records, max_records=max_records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return document
