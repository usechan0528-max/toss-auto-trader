import json
import tempfile
import unittest
from pathlib import Path

from signal_watcher import append_watch_event, format_alert, watch_fingerprint


def valid_payload():
    return {
        "symbol": "LASE",
        "action": "BUY_WATCH",
        "current_price": 3.62,
        "entry_zone": [3.55, 3.7],
        "reference_entry": 3.62,
        "take_profit": [4.05, 4.28],
        "stop_loss": 3.35,
        "rr_ratio": 1.8,
        "confidence": 0.67,
        "recommended_shares": 50,
        "expected_loss_usd": 13.5,
        "expected_loss_krw": 20614,
        "reason": ["VWAP 위"],
        "invalid_if": ["3.35 하향 이탈"],
        "risk_flags": ["고변동"],
        "alerts": ["BUY_WATCH 조건 발생"],
        "market_events": ["진입 후보가 도달"],
        "latest_date": "2026-06-04 01:30",
        "safety_notice": "자동 주문 신호가 아니라 판단 보조 정보입니다.",
        "data_quality": {
            "age_minutes": 1.2,
            "spread_pct": 0.4,
            "stale_blocked": False,
            "spread_blocked": False,
        },
    }


def valid_intent():
    return {
        "symbol": "LASE",
        "side": "BUY",
        "quantity": 50,
        "limit_price": 3.62,
        "stop_loss": 3.35,
        "take_profit_1": 4.05,
        "take_profit_2": 4.28,
        "status": "BLOCKED",
        "reason": ["수동 승인 체크리스트 미완료"],
        "risk_checks": {
            "manual_approval_complete": False,
            "current_session_entries": 1,
            "max_session_entries": 3,
            "duplicate_blocked": False,
        },
    }


class SignalWatcherTests(unittest.TestCase):
    def test_format_alert_separates_signal_and_intent(self):
        text = format_alert(valid_payload(), valid_intent())

        self.assertIn("경고: BUY_WATCH 조건 발생", text)
        self.assertIn("이벤트: 진입 후보가 도달", text)
        self.assertIn("[HIGH] [LASE] BUY_WATCH", text)
        self.assertIn("의도: BLOCKED BUY 50주", text)
        self.assertIn("승인: 대기", text)
        self.assertIn("차단사유: 수동 승인 체크리스트 미완료", text)

    def test_fingerprint_changes_when_intent_status_changes(self):
        payload = valid_payload()
        blocked = valid_intent()
        ready = {**blocked, "status": "DRY_RUN_READY", "reason": []}

        self.assertNotEqual(watch_fingerprint(payload, blocked), watch_fingerprint(payload, ready))

    def test_append_watch_event_writes_jsonl(self):
        payload = valid_payload()
        intent = valid_intent()
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "watch.jsonl"
            append_watch_event(log_path, payload, intent, "message")
            event = json.loads(log_path.read_text(encoding="utf-8"))

        self.assertEqual(event["symbol"], "LASE")
        self.assertEqual(event["priority_level"], "HIGH")
        self.assertEqual(event["intent_status"], "BLOCKED")
        self.assertEqual(event["message"], "message")


if __name__ == "__main__":
    unittest.main()
