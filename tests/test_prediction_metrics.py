import unittest

import pandas as pd

from stock_signal.market_data import QuoteSnapshot
from stock_signal.prediction_metrics import calculate_formula_prediction, formula_prediction_to_dict


def trend_data(rows: int = 90, step: float = 0.012, final_volume: int = 700_000) -> pd.DataFrame:
    price = 3.0
    records = []
    for i in range(rows):
        price += step
        volume = 100_000 if i < rows - 1 else final_volume
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


class PredictionMetricsTests(unittest.TestCase):
    def test_uptrend_projects_higher_price(self):
        data = trend_data()
        quote = QuoteSnapshot(symbol="LASE", price=float(data["Close"].iloc[-1]), bid=4.05, ask=4.08)

        prediction = calculate_formula_prediction(data, quote=quote)

        self.assertGreater(prediction.score, 0)
        self.assertGreater(prediction.probability_up, 0.5)
        self.assertGreater(prediction.projected_price, quote.price)
        payload = formula_prediction_to_dict(prediction)
        self.assertIn("components", payload)
        self.assertIn("diagnostics", payload)
        self.assertIn("obv", prediction.components)
        self.assertIn("stochastic", prediction.components)
        self.assertIn("close_location", prediction.diagnostics)

    def test_downtrend_projects_lower_probability(self):
        data = trend_data(step=-0.008)
        quote = QuoteSnapshot(symbol="LASE", price=float(data["Close"].iloc[-1]), bid=2.25, ask=2.27)

        prediction = calculate_formula_prediction(data, quote=quote)

        self.assertLess(prediction.score, 0)
        self.assertLess(prediction.probability_up, 0.5)

    def test_wide_spread_reduces_signal_quality(self):
        data = trend_data()
        tight = calculate_formula_prediction(data, quote=QuoteSnapshot("LASE", price=4.08, bid=4.07, ask=4.09))
        wide = calculate_formula_prediction(data, quote=QuoteSnapshot("LASE", price=4.08, bid=3.90, ask=4.20))

        self.assertLess(wide.probability_up, tight.probability_up)
        self.assertLess(wide.components["spread_penalty"], tight.components["spread_penalty"])

    def test_longer_horizon_adds_wider_intraday_range(self):
        data = trend_data()
        quote = QuoteSnapshot(symbol="LASE", price=float(data["Close"].iloc[-1]), bid=4.05, ask=4.08)

        short = calculate_formula_prediction(data, quote=quote, horizon_bars=3)
        long = calculate_formula_prediction(data, quote=quote, horizon_bars=60)

        self.assertEqual(long.horizon_bars, 60)
        self.assertGreater(long.projected_high - long.projected_low, short.projected_high - short.projected_low)
        self.assertIn("today_observed_high", long.diagnostics)
        self.assertIn("today_projected_high", long.diagnostics)
        self.assertIn("today_projected_low", long.diagnostics)


if __name__ == "__main__":
    unittest.main()
