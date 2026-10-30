import unittest

from stock_signal.contracts import signal_fingerprint, validate_signal_payload


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
    }


class SignalContractTests(unittest.TestCase):
    def test_valid_payload_passes(self):
        validate_signal_payload(valid_payload())

    def test_rejects_inverted_entry_zone(self):
        payload = valid_payload()
        payload["entry_zone"] = [3.8, 3.7]
        with self.assertRaises(ValueError):
            validate_signal_payload(payload)

    def test_rejects_stop_above_entry(self):
        payload = valid_payload()
        payload["stop_loss"] = 3.7
        with self.assertRaises(ValueError):
            validate_signal_payload(payload)

    def test_fingerprint_ignores_current_price_noise(self):
        first = valid_payload()
        second = valid_payload()
        second["current_price"] = 3.63
        self.assertEqual(signal_fingerprint(first), signal_fingerprint(second))

    def test_accepts_prediction_metrics(self):
        payload = valid_payload()
        payload["prediction_metrics"] = {
            "score": 0.31,
            "probability_up": 0.67,
            "expected_move_pct": 0.42,
            "projected_price": 3.64,
        }
        validate_signal_payload(payload)

    def test_rejects_invalid_prediction_probability(self):
        payload = valid_payload()
        payload["prediction_metrics"] = {
            "score": 0.31,
            "probability_up": 1.2,
            "expected_move_pct": 0.42,
            "projected_price": 3.64,
        }
        with self.assertRaises(ValueError):
            validate_signal_payload(payload)


if __name__ == "__main__":
    unittest.main()
