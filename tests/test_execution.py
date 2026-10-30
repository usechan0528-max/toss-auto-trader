import unittest

from stock_signal.execution import (
    AccountSnapshot,
    build_execution_plan,
    client_order_id_from_intent,
    execution_plan_to_dict,
    order_request_from_intent,
)
from stock_signal.trade_intent import build_trade_intent


def ready_buy_intent():
    return build_trade_intent(
        symbol="LASE",
        action="BUY_WATCH",
        current_price=3.62,
        reference_entry=3.62,
        stop_loss=3.35,
        take_profit=[4.05, 4.28],
        recommended_shares=10,
        mode="dry_run",
        manual_approval_complete=True,
    )


class ExecutionTests(unittest.TestCase):
    def test_order_request_from_ready_intent(self):
        intent = ready_buy_intent()

        order = order_request_from_intent(intent)

        self.assertEqual(order.symbol, "LASE")
        self.assertEqual(order.side, "BUY")
        self.assertEqual(order.quantity, 10)
        self.assertEqual(order.estimated_notional, 36.2)
        self.assertLessEqual(len(order.client_order_id), 36)

    def test_hold_intent_has_no_order_request(self):
        intent = build_trade_intent(
            symbol="LASE",
            action="NO_TRADE",
            current_price=3.20,
            reference_entry=3.62,
            stop_loss=3.35,
            take_profit=[4.05, 4.28],
            recommended_shares=10,
            mode="dry_run",
        )

        self.assertIsNone(order_request_from_intent(intent))

    def test_execution_plan_is_blocked_by_default(self):
        plan = build_execution_plan(ready_buy_intent())

        self.assertEqual(plan.status, "BLOCKED")
        self.assertIn("EXECUTION_LAYER_DISABLED", plan.reason)
        self.assertIn("ACCOUNT_NOT_CONNECTED", plan.reason)
        self.assertIn("USER_FINAL_CONFIRMATION_REQUIRED", plan.reason)

    def test_execution_plan_can_be_ready_when_all_gates_pass(self):
        account = AccountSnapshot(account_id="acct", cash_buying_power=1_000, equity=1_000)

        plan = build_execution_plan(
            ready_buy_intent(),
            account=account,
            execution_enabled=True,
            user_confirmation=True,
        )

        self.assertEqual(plan.status, "READY_FOR_BROKER")
        self.assertEqual(plan.reason, [])
        self.assertEqual(execution_plan_to_dict(plan)["order"]["symbol"], "LASE")

    def test_execution_plan_blocks_insufficient_buying_power(self):
        account = AccountSnapshot(account_id="acct", cash_buying_power=1, equity=1)

        plan = build_execution_plan(
            ready_buy_intent(),
            account=account,
            execution_enabled=True,
            user_confirmation=True,
        )

        self.assertEqual(plan.status, "BLOCKED")
        self.assertIn("INSUFFICIENT_BUYING_POWER", plan.reason)

    def test_client_order_id_is_stable(self):
        intent = ready_buy_intent()

        self.assertEqual(client_order_id_from_intent(intent), client_order_id_from_intent(intent))


if __name__ == "__main__":
    unittest.main()
