import unittest

import pandas as pd

from stock_signal.strategy import StrategyConfig, generate_recommendation


def make_breakout_data(rows: int = 80) -> pd.DataFrame:
    records = []
    price = 3.0
    for i in range(rows):
        price += 0.01
        volume = 100_000
        if i == rows - 1:
            price += 0.25
            volume = 1_000_000
        records.append(
            {
                "Open": price - 0.03,
                "High": price + 0.05,
                "Low": price - 0.05,
                "Close": price,
                "Volume": volume,
            }
        )
    return pd.DataFrame(records)


class StrategyTests(unittest.TestCase):
    def test_generates_structured_recommendation(self):
        rec = generate_recommendation("TEST", make_breakout_data(), StrategyConfig(min_volume_ratio=2.0))
        self.assertIn(rec.action, {"BUY_WATCH", "WAIT", "NO_TRADE"})
        self.assertGreater(rec.entry_zone[1], rec.entry_zone[0])
        self.assertLess(rec.stop_loss, rec.entry_zone[0])
        self.assertGreater(rec.take_profit[0], rec.entry_zone[1])
        self.assertGreaterEqual(rec.recommended_shares, 0)


if __name__ == "__main__":
    unittest.main()
