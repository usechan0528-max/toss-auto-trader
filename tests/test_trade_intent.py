import unittest

from stock_signal.trade_intent import build_trade_intent, trade_intent_to_dict, validate_trade_intent


class TradeIntentTests(unittest.TestCase):
    def test_disabled_mode_blocks_intent(self):
        intent = build_trade_intent(
            symbol="LASE",
            action="BUY_WATCH",
            current_price=3.62,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=50,
            mode="disabled",
        )
        self.assertEqual(intent.side, "BUY")
        self.assertEqual(intent.status, "BLOCKED")
        self.assertIn("실매매 게이트 비활성화", intent.reason)
        validate_trade_intent(intent)

    def test_dry_run_can_prepare_buy(self):
        intent = build_trade_intent(
            symbol="LASE",
            action="BUY_WATCH",
            current_price=3.62,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=50,
            mode="dry_run",
            manual_approval_complete=True,
        )
        self.assertEqual(intent.status, "DRY_RUN_READY")
        self.assertEqual(trade_intent_to_dict(intent)["quantity"], 50)

    def test_live_ready_can_prepare_after_manual_approval(self):
        intent = build_trade_intent(
            symbol="LASE",
            action="BUY_WATCH",
            current_price=3.62,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=10,
            mode="live_ready",
            manual_approval_complete=True,
        )
        self.assertEqual(intent.status, "DRY_RUN_READY")
        self.assertNotIn("실주문 어댑터 미구현", intent.reason)

    def test_daily_loss_limit_blocks_buy(self):
        intent = build_trade_intent(
            symbol="LASE",
            action="BUY_WATCH",
            current_price=3.62,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=50,
            mode="dry_run",
            account_usd=1000,
            daily_realized_pnl_usd=-25,
            daily_loss_limit_pct=2,
            daily_loss_limit_usd=30,
            manual_approval_complete=True,
        )
        self.assertEqual(intent.status, "BLOCKED")
        self.assertIn("일일 손실 한도 도달", intent.reason)

    def test_remaining_daily_loss_limit_blocks_oversized_buy(self):
        intent = build_trade_intent(
            symbol="LASE",
            action="BUY_WATCH",
            current_price=3.62,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=50,
            mode="dry_run",
            account_usd=1000,
            daily_realized_pnl_usd=-10,
            daily_loss_limit_pct=2,
            daily_loss_limit_usd=30,
            manual_approval_complete=True,
        )
        self.assertEqual(intent.status, "BLOCKED")
        self.assertIn("남은 일일 손실 한도보다 예상 위험이 큼", intent.reason)

    def test_duplicate_key_blocks_repeat_intent(self):
        first = build_trade_intent(
            symbol="LASE",
            action="BUY_WATCH",
            current_price=3.62,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=10,
            mode="dry_run",
            trade_date="2026-06-05",
            manual_approval_complete=True,
        )
        duplicate = build_trade_intent(
            symbol="LASE",
            action="BUY_WATCH",
            current_price=3.62,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=10,
            mode="dry_run",
            duplicate_keys={first.duplicate_key},
            trade_date="2026-06-05",
            manual_approval_complete=True,
        )
        self.assertEqual(duplicate.status, "BLOCKED")
        self.assertIn("중복 매매 의도 차단", duplicate.reason)

    def test_no_trade_never_becomes_sell_intent(self):
        intent = build_trade_intent(
            symbol="LASE",
            action="NO_TRADE",
            current_price=3.20,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=50,
            mode="dry_run",
        )
        self.assertEqual(intent.side, "HOLD")
        self.assertEqual(intent.status, "BLOCKED")

    def test_sell_intent_requires_known_holding_quantity(self):
        intent = build_trade_intent(
            symbol="LASE",
            action="WAIT",
            current_price=4.10,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=50,
            mode="dry_run",
            manual_approval_complete=True,
        )

        self.assertEqual(intent.side, "SELL")
        self.assertEqual(intent.status, "BLOCKED")
        self.assertIn("보유 수량 확인 없이 매도 준비 불가", intent.reason)

    def test_sell_intent_caps_quantity_to_holding_quantity(self):
        intent = build_trade_intent(
            symbol="LASE",
            action="WAIT",
            current_price=4.10,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=50,
            mode="dry_run",
            manual_approval_complete=True,
            holding_qty=12,
        )

        self.assertEqual(intent.side, "SELL")
        self.assertEqual(intent.status, "DRY_RUN_READY")
        self.assertEqual(intent.quantity, 12)
        self.assertTrue(intent.risk_checks["sell_quantity_capped"])

    def test_manual_approval_required_for_ready_buy(self):
        intent = build_trade_intent(
            symbol="LASE",
            action="BUY_WATCH",
            current_price=3.62,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=50,
            mode="dry_run",
        )
        self.assertEqual(intent.status, "BLOCKED")
        self.assertIn("수동 승인 체크리스트 미완료", intent.reason)

    def test_session_entry_limit_blocks_buy(self):
        intent = build_trade_intent(
            symbol="LASE",
            action="BUY_WATCH",
            current_price=3.62,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=50,
            mode="dry_run",
            manual_approval_complete=True,
            current_session_entries=3,
            max_session_entries=3,
        )
        self.assertEqual(intent.status, "BLOCKED")
        self.assertIn("세션 최대 진입 횟수 도달", intent.reason)


if __name__ == "__main__":
    unittest.main()
