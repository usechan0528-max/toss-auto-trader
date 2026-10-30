import json
import unittest

import pandas as pd

from stock_signal.strategy import StrategyConfig, generate_recommendation
from stock_signal.output import recommendation_to_json, recommendation_to_text


def sample_data(rows: int = 90) -> pd.DataFrame:
    price = 3.0
    records = []
    for i in range(rows):
        price += 0.01
        records.append(
            {
                "Open": price - 0.02,
                "High": price + 0.04,
                "Low": price - 0.04,
                "Close": price,
                "Volume": 100_000 if i < rows - 1 else 800_000,
            }
        )
    return pd.DataFrame(records)


class OutputContractTests(unittest.TestCase):
    def test_json_contains_required_fields(self):
        rec = generate_recommendation("LASE", sample_data(), StrategyConfig(min_volume_ratio=2.0))
        payload = json.loads(recommendation_to_json(rec))
        for field in [
            "symbol",
            "action",
            "entry_zone",
            "take_profit",
            "stop_loss",
            "confidence",
            "invalid_if",
            "risk_flags",
            "safety_notice",
        ]:
            self.assertIn(field, payload)

    def test_text_contains_no_order_instruction(self):
        rec = generate_recommendation("LASE", sample_data(), StrategyConfig(min_volume_ratio=2.0))
        text = recommendation_to_text(rec)
        self.assertIn("판단", text)
        self.assertIn("자동 주문 신호가 아니라 판단 보조 정보", text)


if __name__ == "__main__":
    unittest.main()
