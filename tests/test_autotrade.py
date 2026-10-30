import tempfile
import unittest
from pathlib import Path

from stock_signal.autotrade import (
    AutoTradeConfig,
    PaperBrokerExecutionAdapter,
    append_auto_trade_event,
    auto_trade_decision_to_dict,
    run_auto_trade,
    submitted_order_ids_from_events,
)
from stock_signal.execution import AccountSnapshot, build_execution_plan
from stock_signal.trade_intent import build_trade_intent


def ready_plan():
    intent = build_trade_intent(
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
    return build_execution_plan(
        intent,
        account=AccountSnapshot(account_id="paper", cash_buying_power=1_000, equity=1_000),
        execution_enabled=True,
        user_confirmation=True,
    )


class AutoTradeTests(unittest.TestCase):
    def test_disabled_blocks_submit(self):
        decision = run_auto_trade(ready_plan(), AutoTradeConfig(enabled=False, mode="disabled"))

        self.assertEqual(decision.status, "AUTO_BLOCKED")
        self.assertIn("AUTO_TRADE_DISABLED", decision.reason)

    def test_kill_switch_blocks_submit(self):
        decision = run_auto_trade(ready_plan(), AutoTradeConfig(enabled=True, mode="paper", kill_switch=True))

        self.assertEqual(decision.status, "AUTO_BLOCKED")
        self.assertEqual(decision.state, "KILL_SWITCH_ACTIVE")
        self.assertIn("AUTO_TRADE_KILL_SWITCH_ACTIVE", decision.reason)

    def test_paper_mode_submits_when_all_gates_pass(self):
        plan = ready_plan()

        decision = run_auto_trade(
            plan,
            AutoTradeConfig(enabled=True, mode="paper", kill_switch=False, max_order_notional_usd=100),
            adapter=PaperBrokerExecutionAdapter(plan.account),
        )

        self.assertEqual(decision.status, "PAPER_SUBMITTED")
        self.assertEqual(decision.state, "ORDER_SUBMITTED")
        self.assertTrue(decision.execution_result.broker_order_id.startswith("paper-"))

    def test_duplicate_submission_is_blocked_by_event_log(self):
        plan = ready_plan()
        decision = run_auto_trade(
            plan,
            AutoTradeConfig(enabled=True, mode="paper", kill_switch=False),
            adapter=PaperBrokerExecutionAdapter(plan.account),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "auto_trade_events.jsonl"
            append_auto_trade_event(auto_trade_decision_to_dict(decision), path)
            submitted = submitted_order_ids_from_events(path)

        duplicate = run_auto_trade(
            plan,
            AutoTradeConfig(enabled=True, mode="paper", kill_switch=False),
            adapter=PaperBrokerExecutionAdapter(plan.account),
            submitted_client_order_ids=submitted,
        )

        self.assertEqual(duplicate.status, "AUTO_BLOCKED")
        self.assertIn("DUPLICATE_AUTO_ORDER_BLOCKED", duplicate.reason)

    def test_live_mode_requires_adapter(self):
        decision = run_auto_trade(
            ready_plan(),
            AutoTradeConfig(enabled=True, mode="live", kill_switch=False),
            adapter=None,
        )

        self.assertEqual(decision.status, "AUTO_BLOCKED")
        self.assertIn("LIVE_BROKER_ADAPTER_NOT_CONNECTED", decision.reason)


if __name__ == "__main__":
    unittest.main()
