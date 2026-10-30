import unittest

from stock_signal.risk import build_risk_plan, position_size


class RiskTests(unittest.TestCase):
    def test_position_size_uses_account_risk(self):
        shares = position_size(account_krw=1_000_000, usdkrw=1527, entry=3.60, stop=3.35, risk_pct=0.02)
        self.assertEqual(shares, 52)

    def test_invalid_risk_returns_zero(self):
        self.assertEqual(position_size(1_000_000, 1527, entry=3.35, stop=3.60), 0)

    def test_build_risk_plan(self):
        plan = build_risk_plan(1_000_000, 1527, entry=3.60, stop=3.35, target=4.05)
        self.assertEqual(plan.shares, 52)
        self.assertAlmostEqual(plan.risk_per_share, 0.25)
        self.assertGreaterEqual(plan.rr_ratio, 1.5)


if __name__ == "__main__":
    unittest.main()
