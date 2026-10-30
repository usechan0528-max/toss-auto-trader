from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


PriorityLevel = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

PRIORITY_SCORE: dict[PriorityLevel, int] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "INFO": 0,
}


@dataclass(frozen=True)
class PriorityResult:
    level: PriorityLevel
    score: int
    reasons: list[str]


def _contains(items: list[Any], keyword: str) -> bool:
    return any(keyword in str(item) for item in items)


def evaluate_priority(payload: dict[str, Any], intent: dict[str, Any] | None = None) -> PriorityResult:
    alerts = list(payload.get("alerts") or [])
    events = list(payload.get("market_events") or [])
    risk_flags = list(payload.get("risk_flags") or [])
    data_quality = payload.get("data_quality") or {}
    checks = (intent or {}).get("risk_checks") or {}
    intent_reasons = list((intent or {}).get("reason") or [])

    reasons: list[str] = []
    level: PriorityLevel = "INFO"

    def raise_to(next_level: PriorityLevel, reason: str) -> None:
        nonlocal level
        if PRIORITY_SCORE[next_level] > PRIORITY_SCORE[level]:
            level = next_level
        reasons.append(reason)

    if payload.get("action") == "BUY_WATCH":
        raise_to("HIGH", "BUY_WATCH signal")
    if _contains(alerts, "BUY_WATCH"):
        raise_to("HIGH", "BUY_WATCH alert")
    if _contains(alerts, "손절") or _contains(intent_reasons, "손절"):
        raise_to("CRITICAL", "stop-loss proximity")
    if _contains(events, "목표가") or _contains(events, "익절"):
        raise_to("MEDIUM", "take-profit area")
    if data_quality.get("stale_blocked") or checks.get("stale_data_blocked"):
        raise_to("HIGH", "stale data blocked")
    if data_quality.get("freshness_status") in {"STALE", "UNKNOWN"}:
        raise_to("HIGH", "stale data warning")
    if data_quality.get("spread_blocked") or checks.get("wide_spread_blocked"):
        raise_to("HIGH", "wide spread blocked")
    if checks.get("duplicate_blocked") or _contains(intent_reasons, "중복"):
        raise_to("MEDIUM", "duplicate intent blocked")
    if _contains(intent_reasons, "일일 손실") or _contains(risk_flags, "daily loss"):
        raise_to("CRITICAL", "daily loss limit")
    if _contains(intent_reasons, "세션 최대"):
        raise_to("HIGH", "session entry limit")
    if _contains(intent_reasons, "수동 승인"):
        raise_to("LOW", "manual approval pending")

    if not reasons:
        reasons.append("routine update")

    return PriorityResult(level=level, score=PRIORITY_SCORE[level], reasons=reasons)


def should_notify(priority: PriorityResult, min_level: PriorityLevel = "LOW") -> bool:
    return priority.score >= PRIORITY_SCORE[min_level]
