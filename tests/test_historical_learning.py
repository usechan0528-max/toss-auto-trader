import unittest

import numpy as np
import pandas as pd

from stock_signal.historical_learning import (
    HistoricalLearningConfig,
    build_labeled_chart_samples,
    train_target_hit_model,
)


def synthetic_prices(rows: int = 220) -> pd.DataFrame:
    index = pd.date_range("2026-01-01 09:30", periods=rows, freq="5min", tz="UTC")
    base = 100 + np.sin(np.arange(rows) / 6) * 0.8 + np.arange(rows) * 0.01
    close = base + np.where((np.arange(rows) % 30) < 10, np.arange(rows) % 10 * 0.08, 0)
    open_ = close - 0.03
    high = close + 0.35 + np.where((np.arange(rows) % 40) < 8, 1.0, 0)
    low = close - 0.35 - np.where((np.arange(rows) % 45) > 35, 0.8, 0)
    volume = 100_000 + (np.arange(rows) % 20) * 10_000
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


class HistoricalLearningTests(unittest.TestCase):
    def test_build_labeled_chart_samples(self):
        samples = build_labeled_chart_samples("AAA", synthetic_prices(), target_pct=0.9, stop_pct=0.7, horizon_bars=6)

        self.assertFalse(samples.empty)
        self.assertIn("target_first", samples.columns)
        self.assertIn("volume_ratio", samples.columns)
        self.assertTrue(set(samples["target_first"].dropna().unique()).issubset({0, 1}))

    def test_train_target_hit_model(self):
        samples = build_labeled_chart_samples("AAA", synthetic_prices(), target_pct=0.9, stop_pct=0.7, horizon_bars=6)
        config = HistoricalLearningConfig(symbols=["AAA"], target_pct=0.9, stop_pct=0.7, horizon_bars=6)

        model, metrics, predictions, importance = train_target_hit_model(samples, config)

        self.assertIn("test", metrics)
        self.assertFalse(predictions.empty)
        self.assertGreater(len(importance), 0)
        self.assertTrue(hasattr(model, "predict_proba"))


if __name__ == "__main__":
    unittest.main()
