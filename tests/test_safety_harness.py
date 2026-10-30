import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from stock_signal.safety_harness import (
    DAILY_LOSS_REASON,
    DAILY_PROFIT_LOCK_REASON,
    DUPLICATE_ORDER_REASON,
    STALE_DATA_REASON,
    DataQualityGuard,
    DailyLossLimit,
    DailyProfitLock,
    DailyProfitLockLedger,
    DuplicateOrderLedger,
    apply_data_quality_harness,
    apply_daily_loss_harness,
    apply_daily_profit_lock_harness,
    apply_duplicate_order_harness,
)
from stock_signal.trade_intent import build_trade_intent


def payload() -> dict:
    return {
        "symbol": "LASE",
        "action": "BUY_WATCH",
        "current_price": 3.62,
        "entry_zone": [3.55, 3.70],
        "reference_entry": 3.62,
        "take_profit": [4.05, 4.28],
        "stop_loss": 3.35,
        "rr_ratio": 1.8,
        "confidence": 0.67,
        "recommended_shares": 50,
        "expected_loss_usd": 13.5,
        "expected_loss_krw": 20614,
        "reason": [],
        "invalid_if": [],
        "risk_flags": [],
        "alerts": [],
        "market_events": [],
        "latest_date": "2026-06-04 09:30:00",
        "safety_notice": "dry run only",
    }


def ready_intent():
    data = payload()
    return build_trade_intent(
        symbol=data["symbol"],
        action=data["action"],
        current_price=data["current_price"],
        reference_entry=data["reference_entry"],
        stop_loss=data["stop_loss"],
        take_profit=data["take_profit"],
        recommended_shares=data["recommended_shares"],
        mode="dry_run",
        manual_approval_complete=True,
    )


class SafetyHarnessTests(unittest.TestCase):
    def test_daily_loss_limit_forces_no_trade(self):
        guarded = apply_daily_loss_harness(
            payload(),
            DailyLossLimit(
                account_usd=1_000,
                realized_pnl_usd=-25,
                max_loss_pct=0.02,
                max_loss_usd=30,
            ),
        )
        self.assertEqual(guarded["action"], "NO_TRADE")
        self.assertEqual(guarded["recommended_shares"], 0)
        self.assertIn(DAILY_LOSS_REASON, guarded["reason"])

    def test_daily_loss_limit_allows_when_not_reached(self):
        guarded = apply_daily_loss_harness(
            payload(),
            DailyLossLimit(account_usd=1_000, realized_pnl_usd=-19, max_loss_pct=0.02, max_loss_usd=30),
        )
        self.assertEqual(guarded["action"], "BUY_WATCH")
        self.assertEqual(guarded["recommended_shares"], 50)

    def test_daily_profit_lock_activates_after_target_and_stops_on_giveback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = DailyProfitLockLedger(Path(temp_dir) / "profit_lock.json")
            first = apply_daily_profit_lock_harness(
                payload(),
                DailyProfitLock(account_usd=1_000, realized_pnl_usd=220, trigger_pct=20, giveback_pct=10),
                ledger,
            )
            second = apply_daily_profit_lock_harness(
                payload(),
                DailyProfitLock(account_usd=1_000, realized_pnl_usd=110, trigger_pct=20, giveback_pct=10),
                ledger,
            )

        self.assertEqual(first["action"], "BUY_WATCH")
        self.assertTrue(first["daily_profit_lock"]["active"])
        self.assertEqual(first["daily_profit_lock"]["peak_pct"], 22.0)
        self.assertEqual(first["daily_profit_lock"]["stop_pct"], 12.0)
        self.assertEqual(second["action"], "NO_TRADE")
        self.assertEqual(second["recommended_shares"], 0)
        self.assertIn(DAILY_PROFIT_LOCK_REASON, second["reason"])

    def test_daily_profit_lock_trails_new_peak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = DailyProfitLockLedger(Path(temp_dir) / "profit_lock.json")
            apply_daily_profit_lock_harness(
                payload(),
                DailyProfitLock(account_usd=1_000, realized_pnl_usd=210, trigger_pct=20, giveback_pct=10),
                ledger,
            )
            higher = apply_daily_profit_lock_harness(
                payload(),
                DailyProfitLock(account_usd=1_000, realized_pnl_usd=350, trigger_pct=20, giveback_pct=10),
                ledger,
            )

        self.assertEqual(higher["daily_profit_lock"]["peak_pct"], 35.0)
        self.assertEqual(higher["daily_profit_lock"]["stop_pct"], 25.0)
        self.assertEqual(higher["action"], "BUY_WATCH")

    def test_duplicate_order_blocks_second_ready_intent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = DuplicateOrderLedger(Path(temp_dir) / "orders.jsonl")
            first = apply_duplicate_order_harness(ready_intent(), payload(), ledger)
            second = apply_duplicate_order_harness(ready_intent(), payload(), ledger)

        self.assertEqual(first.status, "DRY_RUN_READY")
        self.assertEqual(second.status, "BLOCKED")
        self.assertIn(DUPLICATE_ORDER_REASON, second.reason)
        self.assertEqual(first.duplicate_key, second.duplicate_key)
        self.assertFalse(first.risk_checks["duplicate_blocked"])
        self.assertTrue(second.risk_checks["duplicate_blocked"])

    def test_ledger_counts_ready_entries_by_date_symbol_and_side(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = DuplicateOrderLedger(Path(temp_dir) / "orders.jsonl")
            first = apply_duplicate_order_harness(ready_intent(), payload(), ledger)

            self.assertEqual(first.status, "DRY_RUN_READY")
            self.assertEqual(
                ledger.count_entries(trade_date="2026-06-04", symbol="LASE", sides={"BUY"}),
                1,
            )
            self.assertEqual(ledger.count_entries(trade_date="2026-06-04", symbol="XOS"), 0)

    def test_data_quality_guard_blocks_stale_payload(self):
        data = payload()
        data["latest_date"] = "2026-06-04 09:30:00"
        guarded = apply_data_quality_harness(
            data,
            DataQualityGuard(
                enabled=True,
                max_age_minutes=5,
                now=datetime(2026, 6, 4, 9, 45, tzinfo=timezone.utc),
            ),
        )

        self.assertEqual(guarded["action"], "NO_TRADE")
        self.assertEqual(guarded["recommended_shares"], 0)
        self.assertIn(STALE_DATA_REASON, guarded["reason"])
        self.assertTrue(guarded["data_quality"]["stale_blocked"])

    def test_data_quality_guard_is_observability_only_when_disabled(self):
        data = payload()
        data["latest_date"] = "2026-06-04 09:30:00"
        guarded = apply_data_quality_harness(
            data,
            DataQualityGuard(
                enabled=False,
                max_age_minutes=5,
                now=datetime(2026, 6, 4, 9, 45, tzinfo=timezone.utc),
            ),
        )

        self.assertEqual(guarded["action"], "BUY_WATCH")
        self.assertEqual(guarded["recommended_shares"], 50)
        self.assertFalse(guarded["data_quality"]["guard_enabled"])
        self.assertEqual(guarded["data_quality"]["age_minutes"], 15.0)
        self.assertEqual(guarded["data_quality"]["freshness_status"], "DELAYED")

    def test_data_quality_marks_stale_even_when_guard_disabled(self):
        data = payload()
        data["latest_date"] = "2026-06-04 09:30:00"
        guarded = apply_data_quality_harness(
            data,
            DataQualityGuard(
                enabled=False,
                max_age_minutes=5,
                now=datetime(2026, 6, 4, 10, 30, tzinfo=timezone.utc),
            ),
        )

        self.assertEqual(guarded["action"], "BUY_WATCH")
        self.assertEqual(guarded["data_quality"]["freshness_status"], "STALE")


if __name__ == "__main__":
    unittest.main()
