from __future__ import annotations

import hashlib
import json
from typing import Any


REQUIRED_SIGNAL_KEYS = {
    "symbol",
    "action",
    "current_price",
    "entry_zone",
    "reference_entry",
    "take_profit",
    "stop_loss",
    "rr_ratio",
    "confidence",
    "recommended_shares",
    "expected_loss_usd",
    "expected_loss_krw",
    "reason",
    "invalid_if",
    "risk_flags",
    "alerts",
    "market_events",
    "latest_date",
    "safety_notice",
}

VALID_ACTIONS = {"BUY_WATCH", "WAIT", "NO_TRADE"}


def validate_signal_payload(payload: dict[str, Any]) -> None:
    missing = REQUIRED_SIGNAL_KEYS - set(payload)
    if missing:
        raise ValueError(f"신호 payload 필수 키 누락: {sorted(missing)}")

    if payload["action"] not in VALID_ACTIONS:
        raise ValueError(f"지원하지 않는 action: {payload['action']}")

    entry_zone = payload["entry_zone"]
    take_profit = payload["take_profit"]
    if not isinstance(entry_zone, list) or len(entry_zone) != 2:
        raise ValueError("entry_zone은 [low, high] 형식이어야 합니다.")
    if not isinstance(take_profit, list) or len(take_profit) != 2:
        raise ValueError("take_profit은 [target_1, target_2] 형식이어야 합니다.")

    numeric_keys = [
        "current_price",
        "reference_entry",
        "stop_loss",
        "rr_ratio",
        "confidence",
        "recommended_shares",
        "expected_loss_usd",
        "expected_loss_krw",
    ]
    for key in numeric_keys:
        value = payload[key]
        if not isinstance(value, int | float):
            raise ValueError(f"{key}는 숫자여야 합니다.")
        if key != "expected_loss_krw" and value < 0:
            raise ValueError(f"{key}는 음수일 수 없습니다.")

    if entry_zone[0] > entry_zone[1]:
        raise ValueError("entry_zone 하단이 상단보다 클 수 없습니다.")
    if payload["stop_loss"] >= payload["reference_entry"]:
        raise ValueError("stop_loss는 reference_entry보다 낮아야 합니다.")
    if take_profit[0] <= payload["reference_entry"]:
        raise ValueError("1차 익절가는 reference_entry보다 높아야 합니다.")
    if take_profit[1] < take_profit[0]:
        raise ValueError("2차 익절가는 1차 익절가 이상이어야 합니다.")
    if not 0 <= payload["confidence"] <= 1:
        raise ValueError("confidence는 0~1 범위여야 합니다.")

    prediction_metrics = payload.get("prediction_metrics")
    if prediction_metrics:
        if not isinstance(prediction_metrics, dict):
            raise ValueError("prediction_metrics must be an object")
        for key in ["score", "probability_up", "expected_move_pct", "projected_price"]:
            if key not in prediction_metrics:
                raise ValueError(f"prediction_metrics missing key: {key}")
            if not isinstance(prediction_metrics[key], int | float):
                raise ValueError(f"prediction_metrics.{key} must be numeric")
        if not 0 <= prediction_metrics["probability_up"] <= 1:
            raise ValueError("prediction_metrics.probability_up must be between 0 and 1")


def signal_fingerprint(payload: dict[str, Any]) -> str:
    stable = {
        "symbol": payload.get("symbol"),
        "action": payload.get("action"),
        "alerts": payload.get("alerts", []),
        "reference_entry": payload.get("reference_entry"),
        "stop_loss": payload.get("stop_loss"),
        "take_profit": payload.get("take_profit", []),
    }
    raw = json.dumps(stable, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
