import unittest

import pandas as pd

from stock_signal.validators import validate_candle, validate_price_data, validate_quote


class ValidatorTests(unittest.TestCase):
    def test_valid_candle_passes(self):
        validate_candle({"open": 3.6, "high": 3.8, "low": 3.5, "close": 3.7, "volume": 1000})

    def test_invalid_high_low_fails(self):
        with self.assertRaises(AssertionError):
            validate_candle({"open": 3.6, "high": 3.4, "low": 3.5, "close": 3.7, "volume": 1000})

    def test_negative_price_fails(self):
        with self.assertRaises(AssertionError):
            validate_candle({"open": -1, "high": 3.8, "low": 3.5, "close": 3.7, "volume": 1000})

    def test_price_data_validation(self):
        data = pd.DataFrame(
            {
                "Open": [3.6],
                "High": [3.8],
                "Low": [3.5],
                "Close": [3.7],
                "Volume": [1000],
            }
        )
        validate_price_data(data)

    def test_spread_validation(self):
        with self.assertRaises(ValueError):
            validate_quote(price=3.6, bid=3.5, ask=3.7, max_spread_pct=2.0)


if __name__ == "__main__":
    unittest.main()
