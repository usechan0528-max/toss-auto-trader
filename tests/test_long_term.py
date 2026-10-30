import unittest

import pandas as pd

from stock_signal.long_term import build_long_term_plan, long_term_plan_to_dict


def price_data(rows: int = 260, start: float = 100.0, step: float = 0.2) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    price = start
    records = []
    for _ in dates:
        price += step
        records.append(
            {
                "Open": price - 0.5,
                "High": price + 1.0,
                "Low": price - 1.0,
                "Close": price,
                "Volume": 1_000_000,
            }
        )
    return pd.DataFrame(records, index=dates)


class LongTermPlanTests(unittest.TestCase):
    def test_build_long_term_plan_projects_dca_value(self):
        plan = build_long_term_plan(
            price_data(),
            symbol="AAPL",
            monthly_budget_usd=500,
            initial_cash_usd=1000,
            horizon_years=10,
            expected_return_annual_pct=6,
        )

        self.assertEqual(plan.symbol, "AAPL")
        self.assertEqual(plan.total_contribution_usd, 61_000)
        self.assertGreater(plan.projected_value_usd, plan.total_contribution_usd)
        self.assertGreater(plan.monthly_shares, 0)
        self.assertEqual(len(plan.buy_ladder), 4)
        self.assertIn(plan.dca_signal, {"정기매수 유지", "분할 강화", "과열 주의", "소액 분산"})
        self.assertIn("scenarios", long_term_plan_to_dict(plan))

    def test_short_history_adds_risk_note(self):
        plan = build_long_term_plan(
            price_data(rows=30),
            symbol="NVDA",
            monthly_budget_usd=300,
            horizon_years=5,
        )

        self.assertTrue(any("데이터 기간" in note for note in plan.risk_notes))
        self.assertIsNone(plan.price_vs_sma200_pct)

    def test_falling_market_can_trigger_split_accumulation(self):
        plan = build_long_term_plan(
            price_data(rows=260, start=160, step=-0.3),
            symbol="TSLA",
            monthly_budget_usd=400,
            horizon_years=3,
        )

        self.assertLess(plan.drawdown_from_high_pct, -20)
        self.assertIn(plan.dca_signal, {"분할 강화", "소액 분산"})


if __name__ == "__main__":
    unittest.main()
