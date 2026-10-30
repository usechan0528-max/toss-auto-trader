import unittest

import pandas as pd

from stock_signal.backtest import evaluate_signal, run_backtest


class BacktestTests(unittest.TestCase):
    def test_evaluate_signal_win(self):
        self.assertEqual(evaluate_signal(3.6, 3.35, 4.05, [3.65, 3.9, 4.06]), "WIN")

    def test_evaluate_signal_loss(self):
        self.assertEqual(evaluate_signal(3.6, 3.35, 4.05, [3.5, 3.34, 3.4]), "LOSS")

    def test_run_backtest_summary(self):
        trades = pd.DataFrame(
            [
                {"result": "WIN", "return_pct": 10.0, "max_drawdown_pct": -1.0},
                {"result": "LOSS", "return_pct": -3.0, "max_drawdown_pct": -4.0},
            ]
        )
        result = run_backtest(trades)
        self.assertEqual(result.total_signals, 2)
        self.assertEqual(result.win_rate, 0.5)
        self.assertEqual(result.max_drawdown_pct, -4.0)


if __name__ == "__main__":
    unittest.main()
