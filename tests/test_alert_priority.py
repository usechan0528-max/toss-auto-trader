import unittest

from stock_signal.alert_priority import evaluate_priority, should_notify


def payload(action="WAIT", alerts=None, events=None, data_quality=None, risk_flags=None):
    return {
        "symbol": "LASE",
        "action": action,
        "alerts": alerts or [],
        "market_events": events or [],
        "risk_flags": risk_flags or [],
        "data_quality": data_quality or {},
    }


def intent(reason=None, checks=None):
    return {
        "reason": reason or [],
        "risk_checks": checks or {},
    }


class AlertPriorityTests(unittest.TestCase):
    def test_buy_watch_is_high_priority(self):
        result = evaluate_priority(payload(action="BUY_WATCH"), intent())

        self.assertEqual(result.level, "HIGH")
        self.assertIn("BUY_WATCH signal", result.reasons)

    def test_stop_loss_is_critical_priority(self):
        result = evaluate_priority(payload(alerts=["손절선 접근"]), intent())

        self.assertEqual(result.level, "CRITICAL")
        self.assertTrue(should_notify(result, "HIGH"))

    def test_manual_approval_pending_is_low_priority(self):
        result = evaluate_priority(payload(), intent(reason=["수동 승인 체크리스트 미완료"]))

        self.assertEqual(result.level, "LOW")
        self.assertFalse(should_notify(result, "MEDIUM"))

    def test_data_quality_blocks_are_high_priority(self):
        result = evaluate_priority(payload(data_quality={"stale_blocked": True}), intent())

        self.assertEqual(result.level, "HIGH")
        self.assertIn("stale data blocked", result.reasons)

    def test_stale_data_warning_is_high_priority_even_without_block(self):
        result = evaluate_priority(payload(data_quality={"freshness_status": "STALE"}), intent())

        self.assertEqual(result.level, "HIGH")
        self.assertIn("stale data warning", result.reasons)

    def test_daily_loss_limit_is_critical_priority(self):
        result = evaluate_priority(payload(risk_flags=["daily loss -25.00 USD <= -20.00 USD"]), intent())

        self.assertEqual(result.level, "CRITICAL")


if __name__ == "__main__":
    unittest.main()
