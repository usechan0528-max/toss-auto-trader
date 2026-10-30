import tempfile
import unittest
from pathlib import Path

from stock_signal.trade_review import (
    append_trade_review,
    build_llm_trade_review_document,
    build_trade_review_record,
    read_trade_reviews,
    summarize_trade_reviews,
    write_llm_trade_review_document,
)


def sample_signal(action: str = "BUY_WATCH") -> dict:
    return {
        "symbol": "TSLA",
        "action": action,
        "current_price": 101.0,
        "reference_entry": 100.0,
        "entry_zone": [99.0, 101.0],
        "stop_loss": 97.0,
        "take_profit": [104.5, 108.0],
        "rr_ratio": 1.5,
        "confidence": 0.62,
        "recommended_shares": 3,
        "spread_pct": 0.1,
        "risk_flags": [],
        "alerts": [],
        "latest_date": "2026-06-17T10:00:00",
        "prediction_metrics": {
            "score": 0.32,
            "probability_up": 0.68,
            "expected_move_pct": 1.2,
            "projected_price": 102.2,
            "projected_low": 100.8,
            "projected_high": 103.6,
            "components": {
                "trend": 0.4,
                "momentum": 0.3,
                "vwap": 0.2,
                "volume": 0.1,
            },
            "reasons": ["price is above VWAP"],
        },
    }


def sample_intent() -> dict:
    return {
        "status": "DRY_RUN_READY",
        "side": "BUY",
        "quantity": 3,
        "reason": [],
        "risk_checks": {"expected_trade_risk_usd": 9.0},
        "duplicate_key": "2026-06-17:TSLA:BUY:100.0:97.0",
    }


class TradeReviewTests(unittest.TestCase):
    def test_build_review_record_compares_prediction_with_win(self):
        record = build_trade_review_record(
            signal_payload=sample_signal(),
            trade_intent=sample_intent(),
            execution_plan={"status": "BLOCKED", "reason": ["TEST"]},
            side="BUY",
            entry_price=100.0,
            exit_price=105.0,
            quantity=3,
            fees_usd=0.5,
            close_reason="익절",
        )

        self.assertEqual(record["actual"]["outcome"], "WIN")
        self.assertAlmostEqual(record["actual"]["net_pnl_usd"], 14.5)
        self.assertGreater(record["actual"]["realized_r"], 1.0)
        self.assertTrue(record["comparison"]["direction_correct"])
        self.assertEqual(record["comparison"]["decision_quality"], "PREDICTED_ENTRY_WON")
        self.assertIn("winner_review", record["llm_context"]["lesson_tags"])

    def test_loss_from_entry_signal_is_tagged_for_formula_review(self):
        record = build_trade_review_record(
            signal_payload=sample_signal(),
            trade_intent=sample_intent(),
            execution_plan={},
            side="BUY",
            entry_price=100.0,
            exit_price=96.8,
            quantity=2,
            close_reason="손절",
        )

        self.assertEqual(record["actual"]["outcome"], "LOSS")
        self.assertTrue(record["comparison"]["hit_stop_loss"])
        self.assertIn("formula_or_filter_needs_review", record["llm_context"]["lesson_tags"])

    def test_document_and_jsonl_round_trip(self):
        win = build_trade_review_record(
            signal_payload=sample_signal(),
            trade_intent=sample_intent(),
            execution_plan={},
            side="BUY",
            entry_price=100.0,
            exit_price=105.0,
            quantity=1,
        )
        loss = build_trade_review_record(
            signal_payload=sample_signal(),
            trade_intent=sample_intent(),
            execution_plan={},
            side="BUY",
            entry_price=100.0,
            exit_price=96.0,
            quantity=1,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "trade_reviews.jsonl"
            md_path = Path(temp_dir) / "llm_trade_review.md"
            append_trade_review(win, jsonl_path)
            append_trade_review(loss, jsonl_path)

            records = read_trade_reviews(jsonl_path)
            summary = summarize_trade_reviews(records)
            document = write_llm_trade_review_document(records, md_path)

        self.assertEqual(len(records), 2)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertIn("Formula Component Win/Loss Averages", document)
        self.assertIn("compact_json", build_llm_trade_review_document(records))


if __name__ == "__main__":
    unittest.main()
