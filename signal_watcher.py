from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stock_signal.alert_priority import PriorityLevel, evaluate_priority, should_notify
from stock_signal.contracts import signal_fingerprint, validate_signal_payload


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[WARN] JSON 파일 무시: {path} | {exc}")
        return None
    if not isinstance(data, dict):
        print(f"[WARN] JSON 객체가 아닙니다: {path}")
        return None
    return data


def read_payload(path: Path) -> dict[str, Any] | None:
    payload = read_json(path)
    if payload is None:
        return None
    try:
        validate_signal_payload(payload)
    except ValueError as exc:
        print(f"[WARN] 신호 파일 무시: {exc}")
        return None
    return payload


def read_trade_intent(path: Path) -> dict[str, Any] | None:
    intent = read_json(path)
    if intent is None:
        return None
    required = {"symbol", "side", "status", "reason", "risk_checks"}
    missing = required - set(intent)
    if missing:
        print(f"[WARN] 매매 의도 파일 무시: 필수 키 누락 {sorted(missing)}")
        return None
    return intent


def format_price(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"${float(value):.{digits}f}"


def format_alert(payload: dict[str, Any], intent: dict[str, Any] | None = None) -> str:
    priority = evaluate_priority(payload, intent)
    alerts = ", ".join(payload.get("alerts") or ["신규 알림 없음"])
    events = ", ".join(payload.get("market_events") or ["가격 이벤트 없음"])
    entry_low, entry_high = payload["entry_zone"]
    target_1, target_2 = payload["take_profit"]
    parts = [
        f"[{priority.level}] [{payload['symbol']}] {payload['action']}",
        f"현재 {format_price(payload['current_price'])}",
        f"진입 {format_price(entry_low)}~{format_price(entry_high)}",
        f"손절 {format_price(payload['stop_loss'])}",
        f"익절 {format_price(target_1)}/{format_price(target_2)}",
        f"경고: {alerts}",
        f"이벤트: {events}",
    ]
    data_quality = payload.get("data_quality") or {}
    if data_quality:
        parts.append(
            "데이터: "
            f"{data_quality.get('freshness_status', 'UNKNOWN')} · "
            f"{data_quality.get('age_minutes', 'n/a')}분 지연, "
            f"스프레드 {data_quality.get('spread_pct', 'n/a')}%"
        )
    if intent:
        reasons = ", ".join(intent.get("reason") or ["사유 없음"])
        checks = intent.get("risk_checks") or {}
        parts.extend(
            [
                f"의도: {intent.get('status')} {intent.get('side')} {intent.get('quantity', 0)}주",
                f"승인: {'완료' if checks.get('manual_approval_complete') else '대기'}",
                f"세션: {checks.get('current_session_entries', 0)}/{checks.get('max_session_entries', 0)}",
                f"차단사유: {reasons}",
            ]
        )
    return " | ".join(parts)


def watch_fingerprint(payload: dict[str, Any], intent: dict[str, Any] | None = None) -> str:
    intent_part = {}
    if intent:
        intent_part = {
            "status": intent.get("status"),
            "side": intent.get("side"),
            "quantity": intent.get("quantity"),
            "reason": intent.get("reason", []),
            "risk_checks": {
                key: (intent.get("risk_checks") or {}).get(key)
                for key in [
                    "manual_approval_complete",
                    "current_session_entries",
                    "max_session_entries",
                    "duplicate_blocked",
                    "stale_data_blocked",
                    "wide_spread_blocked",
                ]
            },
        }
    raw = json.dumps(
        {
            "signal": signal_fingerprint(payload),
            "intent": intent_part,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    import hashlib

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def append_watch_event(log_path: Path, payload: dict[str, Any], intent: dict[str, Any] | None, message: str) -> None:
    priority = evaluate_priority(payload, intent)
    event = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "symbol": payload.get("symbol"),
        "action": payload.get("action"),
        "priority_level": priority.level,
        "priority_score": priority.score,
        "priority_reasons": priority.reasons,
        "intent_status": intent.get("status") if intent else None,
        "intent_side": intent.get("side") if intent else None,
        "alerts": payload.get("alerts", []),
        "market_events": payload.get("market_events", []),
        "message": message,
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_snapshot(signal_path: Path, intent_path: Path | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    payload = read_payload(signal_path)
    intent = read_trade_intent(intent_path) if intent_path else None
    return payload, intent


def watch_signal(
    signal_path: Path,
    interval: float,
    intent_path: Path | None = None,
    event_log: Path | None = None,
    min_priority: PriorityLevel = "LOW",
) -> None:
    last_fingerprint = ""
    print(f"[INFO] 신호 감시 시작: {signal_path}")
    if intent_path:
        print(f"[INFO] 매매 의도 감시: {intent_path}")
    if event_log:
        print(f"[INFO] 이벤트 로그: {event_log}")

    while True:
        payload, intent = load_snapshot(signal_path, intent_path)
        if payload:
            fingerprint = watch_fingerprint(payload, intent)
            if fingerprint != last_fingerprint:
                priority = evaluate_priority(payload, intent)
                message = format_alert(payload, intent)
                if should_notify(priority, min_priority):
                    print(message)
                if event_log:
                    append_watch_event(event_log, payload, intent, message)
                last_fingerprint = fingerprint
        time.sleep(interval)


def print_once(
    signal_path: Path,
    intent_path: Path | None = None,
    event_log: Path | None = None,
    min_priority: PriorityLevel = "INFO",
) -> bool:
    payload, intent = load_snapshot(signal_path, intent_path)
    if not payload:
        print(f"[WARN] 유효한 신호가 없습니다: {signal_path}")
        return False
    message = format_alert(payload, intent)
    priority = evaluate_priority(payload, intent)
    if should_notify(priority, min_priority):
        print(message)
    if event_log:
        append_watch_event(event_log, payload, intent, message)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="latest_signal.json과 latest_trade_intent.json을 감시하는 조회 전용 콘솔")
    parser.add_argument("--path", default="latest_signal.json", help="감시할 신호 JSON 경로")
    parser.add_argument("--intent-path", default="latest_trade_intent.json", help="감시할 매매 의도 JSON 경로")
    parser.add_argument("--event-log", default="", help="변경 이벤트를 저장할 JSONL 경로")
    parser.add_argument("--interval", type=float, default=2.0, help="감시 간격 초")
    parser.add_argument(
        "--min-priority",
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
        default="LOW",
        help="콘솔에 출력할 최소 우선순위",
    )
    parser.add_argument("--once", action="store_true", help="한 번만 읽고 종료")
    parser.add_argument("--no-intent", action="store_true", help="매매 의도 파일을 읽지 않음")
    args = parser.parse_args()

    signal_path = Path(args.path)
    intent_path = None if args.no_intent else Path(args.intent_path)
    event_log = Path(args.event_log) if args.event_log else None

    if args.once:
        raise SystemExit(0 if print_once(signal_path, intent_path, event_log, args.min_priority) else 1)
    watch_signal(signal_path, args.interval, intent_path, event_log, args.min_priority)


if __name__ == "__main__":
    main()
