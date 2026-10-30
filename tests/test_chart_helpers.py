import unittest
from types import SimpleNamespace

import pandas as pd

from app import (
    add_indicators,
    choose_auto_symbol,
    clean_secret,
    dataframe_selected_rows,
    format_stock_option,
    live_trading_preflight,
    live_order_validation,
    normalize_toss_auth,
    scanner_table_frame,
    scanner_view_frame,
    add_auto_selection_scores,
    stock_search_symbols,
    train_predictor,
    trade_win_labels,
    visible_price_range,
)
from stock_signal.execution import AccountSnapshot, OrderRequest
from stock_signal.market_data import QuoteSnapshot


class ChartHelperTests(unittest.TestCase):
    def test_add_indicators_includes_trade_quality_features(self):
        rows = []
        price = 100.0
        for idx in range(90):
            price += 0.2
            rows.append(
                {
                    "Open": price - 0.1,
                    "High": price + 0.4,
                    "Low": price - 0.4,
                    "Close": price,
                    "Volume": 100_000 + idx * 1_000,
                }
            )
        features = add_indicators(pd.DataFrame(rows))

        for column in [
            "atr_pct",
            "vwap_distance_pct",
            "boll_position",
            "stoch_k",
            "obv_slope_5",
            "close_location",
            "volume_zscore_20",
        ]:
            self.assertIn(column, features.columns)
            self.assertTrue(pd.notna(features[column].iloc[-1]))

    def test_trade_win_labels_use_target_before_stop_outcome(self):
        features = pd.DataFrame(
            {
                "Close": [100.0, 100.0, 100.0, 100.0],
                "High": [100.2, 101.2, 100.2, 100.2],
                "Low": [99.8, 99.8, 98.8, 99.8],
                "atr_14": [1.0, 1.0, 1.0, 1.0],
                "support": [99.2, 99.2, 99.2, 99.2],
            }
        )

        labels = trade_win_labels(features, horizon_bars=1, reward_ratio=1.0)

        self.assertEqual(labels.iloc[0], 1.0)
        self.assertEqual(labels.iloc[1], 0.0)
        self.assertTrue(pd.isna(labels.iloc[-1]))

    def test_train_predictor_reports_trade_win_confidence_metadata(self):
        rows = []
        for idx in range(160):
            close = 100.0 + idx * 0.01
            rows.append(
                {
                    "Close": close,
                    "High": close + 0.2,
                    "Low": close - 0.2,
                    "support": close - 0.8,
                    "return_1": 0.001,
                    "return_5": 0.005,
                    "ma_5": close,
                    "ma_20": close,
                    "ma_60": close,
                    "ema_9": close,
                    "ema_21": close - 0.1,
                    "volume_ratio": 1.4,
                    "volume_zscore_20": 0.3,
                    "volatility_20": 0.01,
                    "atr_14": 1.0,
                    "atr_pct": 1.0,
                    "rsi_14": 55.0,
                    "macd": 0.1,
                    "macd_signal": 0.05,
                    "macd_hist": 0.05,
                    "vwap_distance_pct": 0.2,
                    "ema_spread_pct": 0.1,
                    "ema_9_slope_5": 0.1,
                    "ma_20_slope_5": 0.1,
                    "boll_width_pct": 2.0,
                    "boll_position": 0.55,
                    "stoch_k": 60.0,
                    "stoch_d": 58.0,
                    "obv_slope_5": 0.1,
                    "close_location": 0.6,
                    "body_pct": 0.2,
                    "upper_wick_pct": 0.1,
                    "lower_wick_pct": 0.1,
                }
            )
        features = pd.DataFrame(rows)

        prediction, model_data = train_predictor(features, reward_ratio=1.0, horizon_bars=4)

        self.assertEqual(prediction.confidence_target, "target_before_stop")
        self.assertEqual(prediction.horizon_bars, 4)
        self.assertGreaterEqual(prediction.probability_up, 0.0)
        self.assertLessEqual(prediction.probability_up, 1.0)
        self.assertIn("target_trade_win", model_data.columns)

    def test_visible_price_range_includes_trade_levels(self):
        data = pd.DataFrame(
            {
                "Low": [9.8, 10.0, 10.1],
                "High": [10.5, 10.7, 10.9],
                "ema_9": [10.1, 10.2, 10.3],
                "ema_21": [10.0, 10.1, 10.2],
                "vwap_60": [10.05, 10.15, 10.25],
            }
        )
        plan = SimpleNamespace(
            stop_loss=9.5,
            take_profit_1=11.4,
            take_profit_2=12.0,
            pullback_entry=10.2,
            breakout_entry=10.8,
        )
        low, high = visible_price_range(data, plan)
        self.assertLess(low, 9.5)
        self.assertGreater(high, 12.0)

    def test_toss_auth_uses_api_key_secret_pair(self):
        token, client_id, client_secret = normalize_toss_auth(
            "tsck_live_abc",
            "your_toss_client_id_here",
            "tssk_live_def",
        )
        self.assertEqual(token, "")
        self.assertEqual(client_id, "tsck_live_abc")
        self.assertEqual(client_secret, "tssk_live_def")

    def test_clean_secret_ignores_placeholders(self):
        self.assertEqual(clean_secret("your_toss_account_seq_here"), "")
        self.assertEqual(clean_secret(" real-value "), "real-value")

    def test_stock_search_maps_names_to_symbols(self):
        self.assertEqual(stock_search_symbols("애플"), ["AAPL"])
        self.assertIn("NVDA", stock_search_symbols("Nvidia"))
        self.assertIn("TSLA", stock_search_symbols("TSLA"))

    def test_stock_option_format_is_readable(self):
        label = format_stock_option(
            {
                "symbol": "AAPL",
                "name": "애플",
                "englishName": "Apple",
                "market": "NASDAQ",
                "currency": "USD",
            }
        )
        self.assertIn("AAPL", label)
        self.assertIn("애플", label)
        self.assertIn("NASDAQ", label)

    def test_scanner_table_frame_marks_selected_symbol(self):
        scanner = pd.DataFrame(
            [
                {"종목": "AAPL", "현재가": 100.123, "등락률": 1.234, "상승확률": 62.4, "방향": "관망", "위험": "정상"},
                {"종목": "NVDA", "현재가": 200.0, "등락률": -0.5, "상승확률": 44.0, "방향": "주의", "위험": "거래량 둔화"},
            ]
        )

        table = scanner_table_frame(scanner, "NVDA")

        self.assertEqual(table.iloc[1]["상태"], "●")
        self.assertEqual(table.iloc[0]["현재가"], 100.12)
        self.assertEqual(table.iloc[0]["AI"], 62)

    def test_scanner_view_frame_sorts_and_filters(self):
        scanner = pd.DataFrame(
            [
                {"종목": "AAPL", "현재가": 100.0, "등락률": 1.0, "상승확률": 55.0, "방향": "관망", "거래량배율": 1.1, "위험": "정상"},
                {"종목": "NVDA", "현재가": 200.0, "등락률": -0.5, "상승확률": 80.0, "방향": "롱 우위", "거래량배율": 3.2, "위험": "거래량 과열"},
                {"종목": "MSFT", "현재가": 300.0, "등락률": 0.2, "상승확률": 60.0, "방향": "관망", "거래량배율": 2.0, "위험": "정상"},
            ]
        )

        self.assertEqual(scanner_view_frame(scanner, "AAPL", "AI 점수").iloc[0]["종목"], "NVDA")
        self.assertEqual(scanner_view_frame(scanner, "AAPL", "거래량").iloc[0]["종목"], "NVDA")
        self.assertNotIn("NVDA", scanner_view_frame(scanner, "AAPL", "위험 숨김")["종목"].tolist())

    def test_choose_auto_symbol_picks_eligible_long_bias_candidate(self):
        scanner = pd.DataFrame(
            [
                {"종목": "AAPL", "등락률": 3.0, "상승확률": 65.0, "방향": "관망", "거래량배율": 3.5, "RSI": 55.0, "위험": "정상"},
                {"종목": "NVDA", "등락률": 1.0, "상승확률": 62.0, "방향": "롱 우위", "거래량배율": 2.0, "RSI": 60.0, "위험": "정상"},
                {"종목": "TSLA", "등락률": 5.0, "상승확률": 80.0, "방향": "롱 우위", "거래량배율": 5.0, "RSI": 78.0, "위험": "거래량 과열"},
            ]
        )

        selected = choose_auto_symbol(add_auto_selection_scores(scanner), min_probability=55.0, min_volume_ratio=1.2)

        self.assertTrue(selected["eligible"])
        self.assertEqual(selected["symbol"], "NVDA")

    def test_choose_auto_symbol_blocks_when_no_candidate_meets_thresholds(self):
        scanner = pd.DataFrame(
            [
                {"종목": "AAPL", "등락률": 1.0, "상승확률": 50.0, "방향": "롱 우위", "거래량배율": 2.0, "RSI": 60.0, "위험": "정상"},
                {"종목": "MSFT", "등락률": 2.0, "상승확률": 65.0, "방향": "관망", "거래량배율": 3.0, "RSI": 55.0, "위험": "정상"},
            ]
        )

        selected = choose_auto_symbol(scanner, min_probability=55.0, min_volume_ratio=1.2)

        self.assertFalse(selected["eligible"])
        self.assertIn("NO_ELIGIBLE_AUTO_TRADE_CANDIDATE", selected["reason"])

    def test_dataframe_selected_rows_accepts_dict_and_attrs(self):
        class Selection:
            rows = [2]

        class State:
            selection = Selection()

        self.assertEqual(dataframe_selected_rows({"selection": {"rows": [1]}}), [1])
        self.assertEqual(dataframe_selected_rows(State()), [2])

    def test_live_order_validation_requires_toggle_account_and_confirmation(self):
        order = OrderRequest("AAPL", "BUY", 2, "LIMIT", 100.0, "DAY", "manual-1")

        result = live_order_validation(
            order=order,
            quote_price=100.0,
            account=None,
            live_order_enabled=False,
            user_acknowledged=False,
            confirmation_text="",
            max_notional_usd=500.0,
        )

        self.assertFalse(result["ready"])
        self.assertIn("LIVE_ORDER_TOGGLE_OFF", result["reasons"])
        self.assertIn("ACCOUNT_NOT_CONNECTED", result["reasons"])
        self.assertEqual(result["confirmation_required"], "BUY AAPL 2")

    def test_live_order_validation_allows_confirmed_small_order(self):
        order = OrderRequest("AAPL", "BUY", 2, "LIMIT", 100.0, "DAY", "manual-1")

        result = live_order_validation(
            order=order,
            quote_price=100.0,
            account=AccountSnapshot(account_id="1", cash_buying_power=500.0, equity=500.0),
            live_order_enabled=True,
            user_acknowledged=True,
            confirmation_text="BUY AAPL 2",
            max_notional_usd=500.0,
        )

        self.assertTrue(result["ready"])
        self.assertEqual(result["reasons"], [])

    def test_live_order_validation_fast_mode_skips_confirmation_text(self):
        order = OrderRequest("AAPL", "BUY", 2, "LIMIT", 100.0, "DAY", "manual-1")

        result = live_order_validation(
            order=order,
            quote_price=100.0,
            account=AccountSnapshot(account_id="1", cash_buying_power=500.0, equity=500.0),
            live_order_enabled=True,
            user_acknowledged=False,
            confirmation_text="",
            max_notional_usd=500.0,
            require_user_ack=False,
            require_confirmation_text=False,
        )

        self.assertTrue(result["ready"])
        self.assertEqual(result["reasons"], [])

    def test_live_order_validation_blocks_notional_limit(self):
        order = OrderRequest("AAPL", "BUY", 10, "MARKET", None, "DAY", "manual-1")

        result = live_order_validation(
            order=order,
            quote_price=100.0,
            account=AccountSnapshot(account_id="1", cash_buying_power=2_000.0, equity=2_000.0),
            live_order_enabled=True,
            user_acknowledged=True,
            confirmation_text="BUY AAPL 10",
            max_notional_usd=500.0,
        )

        self.assertFalse(result["ready"])
        self.assertIn("MAX_MANUAL_ORDER_NOTIONAL_EXCEEDED", result["reasons"])

    def test_live_trading_preflight_blocks_missing_account_and_toggle(self):
        result = live_trading_preflight(
            has_auth=True,
            account_sync_enabled=False,
            live_order_enabled=False,
            account=None,
            quote=QuoteSnapshot(symbol="AAPL", price=100.0),
            trade_intent={"status": "DRY_RUN_READY", "side": "BUY", "quantity": 1},
            execution_plan={"status": "BLOCKED"},
            auto_trade_mode="disabled",
            auto_preapproval=False,
            auto_kill_switch=True,
            max_manual_order_notional_usd=500.0,
        )

        self.assertFalse(result["ready_for_manual_live"])
        self.assertIn("account_sync", result["blocking"])
        self.assertIn("manual_live_toggle", result["blocking"])

    def test_live_trading_preflight_allows_manual_ready(self):
        result = live_trading_preflight(
            has_auth=True,
            account_sync_enabled=True,
            live_order_enabled=True,
            account=AccountSnapshot(account_id="1", cash_buying_power=1000.0, equity=1000.0),
            quote=QuoteSnapshot(symbol="AAPL", price=100.0),
            trade_intent={"status": "DRY_RUN_READY", "side": "BUY", "quantity": 1},
            execution_plan={"status": "READY_FOR_BROKER"},
            auto_trade_mode="live",
            auto_preapproval=True,
            auto_kill_switch=False,
            max_manual_order_notional_usd=500.0,
        )

        self.assertTrue(result["ready_for_manual_live"])
        self.assertTrue(result["ready_for_auto_live"])
        self.assertEqual(result["blocking"], [])


if __name__ == "__main__":
    unittest.main()
