import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from auto_session_trader import (
    _build_exit_trade_intent,
    _candidate_score_rows,
    _choose_auto_symbol_from_monitor_rows,
    _exit_reason,
    _fast_scan_features,
    _filter_broker_blocked_symbols,
    _filter_reentry_cooldown,
    _holding_quantity_map,
    _liquidity_metrics,
    _modeled_universe,
    _passes_liquidity_filter,
    _passes_wide_fast_filter,
    _position_summary_rows,
    _rebound_potential_signal,
    _rotation_batch,
    _select_exit_candidate,
    _support_resistance_signal,
    _apply_reentry_cooldown,
    _surge_candidate_scanner,
    _surge_metrics,
    _target_decay_exit_signal,
    _target_extension_signal,
    _wide_symbol_pool,
    _watchlist,
    acquire_worker_lock,
    apply_daily_order_guard,
    apply_buying_power_sizing_harness,
    apply_entry_edge_harness,
    apply_entry_liquidity_harness,
    apply_entry_structure_harness,
    apply_intraday_projection_targeting,
    _daily_order_guard_stats,
    effective_daily_order_guard_max_loss_usd,
    release_worker_lock,
    update_worker_lock,
)
from stock_signal.execution import AccountSnapshot


class AutoSessionTraderTests(unittest.TestCase):
    def test_worker_lock_is_created_updated_and_released(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "worker.lock"

            acquire_worker_lock(lock_path)
            first = json.loads(lock_path.read_text(encoding="utf-8"))
            update_worker_lock(lock_path)
            updated = json.loads(lock_path.read_text(encoding="utf-8"))
            release_worker_lock(lock_path)

            self.assertEqual(first["pid"], os.getpid())
            self.assertEqual(updated["pid"], os.getpid())
            self.assertIn("last_heartbeat", updated)
            self.assertFalse(lock_path.exists())

    def test_worker_lock_blocks_duplicate_current_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "worker.lock"
            lock_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

            with self.assertRaises(RuntimeError):
                acquire_worker_lock(lock_path)

    def test_worker_lock_replaces_stale_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "worker.lock"
            lock_path.write_text(json.dumps({"pid": 0}), encoding="utf-8")

            acquire_worker_lock(lock_path)
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            release_worker_lock(lock_path)

            self.assertEqual(lock["pid"], os.getpid())

    def test_broker_blocked_symbols_are_removed_but_holdings_stay(self):
        filtered = _filter_broker_blocked_symbols(
            ["AAPL", "TQQQ", "MSTR", "SQQQ"],
            ["TQQQ", "SQQQ"],
            holding_symbols=["SQQQ"],
        )

        self.assertEqual(filtered, ["AAPL", "MSTR", "SQQQ"])

    def test_auto_selection_skips_unaffordable_candidates(self):
        scanner = pd.DataFrame(
            [
                {
                    "symbol": "NET",
                    "auto_score": 90.0,
                    "confidence_pct": 80.0,
                    "volume_ratio": 4.0,
                    "direction": "LONG",
                    "risk": "NORMAL",
                    "reference_entry": 250.0,
                },
                {
                    "symbol": "U",
                    "auto_score": 70.0,
                    "confidence_pct": 68.0,
                    "volume_ratio": 3.0,
                    "direction": "LONG",
                    "risk": "NORMAL",
                    "reference_entry": 31.0,
                },
            ]
        )

        selected = _choose_auto_symbol_from_monitor_rows(
            scanner,
            min_probability=50.0,
            min_volume_ratio=1.2,
            require_risk_normal=True,
            allow_caution_entries=False,
            max_entry_price=60.0,
        )

        self.assertEqual(selected["symbol"], "U")
        self.assertTrue(selected["affordable"])

    def test_auto_selection_fallback_stays_with_affordable_candidates(self):
        scanner = pd.DataFrame(
            [
                {
                    "symbol": "OKTA",
                    "auto_score": 95.0,
                    "confidence_pct": 82.0,
                    "volume_ratio": 5.0,
                    "direction": "LONG",
                    "risk": "NORMAL",
                    "reference_entry": 140.0,
                },
                {
                    "symbol": "F",
                    "auto_score": 20.0,
                    "confidence_pct": 48.0,
                    "volume_ratio": 0.8,
                    "direction": "LONG",
                    "risk": "NORMAL",
                    "reference_entry": 13.7,
                },
            ]
        )

        selected = _choose_auto_symbol_from_monitor_rows(
            scanner,
            min_probability=50.0,
            min_volume_ratio=1.2,
            require_risk_normal=True,
            allow_caution_entries=False,
            max_entry_price=60.0,
        )

        self.assertEqual(selected["symbol"], "F")
        self.assertFalse(selected["eligible"])
        self.assertTrue(selected["affordable"])

    def test_buying_power_sizing_caps_buy_quantity(self):
        payload = {
            "action": "BUY_WATCH",
            "reference_entry": 383.08,
            "stop_loss": 381.58,
            "recommended_shares": 8,
            "expected_loss_usd": 12.0,
            "expected_loss_krw": 18_000.0,
            "risk_flags": [],
        }
        account = AccountSnapshot(account_id="1", cash_buying_power=393.61, equity=393.61)

        guarded = apply_buying_power_sizing_harness(
            payload,
            account=account,
            max_order_notional_usd=500.0,
            usdkrw=1527.0,
        )

        self.assertEqual(guarded["recommended_shares"], 1)
        self.assertEqual(guarded["position_sizing"]["effective_notional_limit_usd"], 393.61)
        self.assertIn("BUY_QUANTITY_CAPPED_TO_BUYING_POWER", guarded["risk_flags"])

    def test_entry_edge_harness_blocks_tight_target_after_costs(self):
        payload = {
            "action": "BUY_WATCH",
            "reference_entry": 100.0,
            "take_profit": [100.7, 101.4],
            "recommended_shares": 2,
            "reason": [],
            "alerts": [],
            "risk_flags": [],
        }

        guarded = apply_entry_edge_harness(
            payload,
            commission_rate_pct=0.1,
            min_gross_target_pct=1.2,
            min_net_target_pct=0.5,
            fee_multiplier=3.0,
            slippage_buffer_pct=0.15,
            min_order_notional_usd=75.0,
        )

        self.assertEqual(guarded["action"], "NO_TRADE")
        self.assertEqual(guarded["recommended_shares"], 0)
        self.assertFalse(guarded["entry_edge"]["passed"])
        self.assertIn("ENTRY_EDGE_BELOW_FEE_ADJUSTED_MINIMUM", guarded["reason"])

    def test_entry_liquidity_harness_blocks_thin_trade_energy(self):
        payload = {
            "action": "BUY_WATCH",
            "reference_entry": 20.0,
            "take_profit": [20.6, 21.0],
            "recommended_shares": 5,
            "reason": [],
            "alerts": [],
            "risk_flags": [],
        }

        guarded = apply_entry_liquidity_harness(
            payload,
            auto_selection={
                "recent_dollar_volume": 950_000,
                "latest_dollar_volume": 22_000,
                "volume_ratio": 0.67,
                "surge_volume_ratio": 0.72,
                "liquidity_filter_bypassed": False,
            },
            enabled=True,
            min_recent_dollar_volume_usd=5_000_000,
            min_latest_dollar_volume_usd=200_000,
            min_volume_ratio=1.2,
        )

        self.assertEqual(guarded["action"], "NO_TRADE")
        self.assertEqual(guarded["recommended_shares"], 0)
        self.assertFalse(guarded["entry_liquidity"]["passed"])
        self.assertIn("ENTRY_LIQUIDITY_BELOW_MINIMUM", guarded["reason"])
        self.assertIn("ENTRY_VOLUME_RATIO_BELOW_MINIMUM", guarded["reason"])

    def test_entry_liquidity_harness_allows_liquid_candidate(self):
        payload = {
            "action": "BUY_WATCH",
            "reference_entry": 100.0,
            "take_profit": [102.0, 103.0],
            "recommended_shares": 2,
            "reason": [],
            "alerts": [],
            "risk_flags": [],
        }

        guarded = apply_entry_liquidity_harness(
            payload,
            auto_selection={
                "recent_dollar_volume": 12_000_000,
                "latest_dollar_volume": 450_000,
                "volume_ratio": 1.35,
                "surge_volume_ratio": 1.1,
                "liquidity_filter_bypassed": False,
            },
            enabled=True,
            min_recent_dollar_volume_usd=5_000_000,
            min_latest_dollar_volume_usd=200_000,
            min_volume_ratio=1.2,
        )

        self.assertEqual(guarded["action"], "BUY_WATCH")
        self.assertTrue(guarded["entry_liquidity"]["passed"])

    def test_support_resistance_signal_detects_confirmed_lower_breakdown(self):
        rows = []
        for _ in range(28):
            rows.append(
                {
                    "Open": 10.2,
                    "High": 10.5,
                    "Low": 10.0,
                    "Close": 10.22,
                    "Volume": 10000,
                    "volume_ratio": 0.9,
                    "volume_zscore_20": -0.1,
                }
            )
        rows.extend(
            [
                {"Open": 10.08, "High": 10.1, "Low": 9.78, "Close": 9.82, "Volume": 18000, "volume_ratio": 1.45, "volume_zscore_20": 0.4},
                {"Open": 9.83, "High": 9.88, "Low": 9.7, "Close": 9.76, "Volume": 21000, "volume_ratio": 1.65, "volume_zscore_20": 0.7},
            ]
        )
        features = pd.DataFrame(rows)

        signal = _support_resistance_signal(
            features,
            current_price=9.76,
            lookback_bars=20,
            confirm_bars=2,
            min_break_pct=0.15,
            atr_buffer_multiplier=0.25,
            min_volume_ratio=1.0,
            min_volume_zscore=-0.2,
        )

        self.assertEqual(signal["state"], "LOWER_SUPPORT_BROKEN")
        self.assertTrue(signal["lower_breakdown"])
        self.assertFalse(signal["upper_breakout"])
        self.assertLess(signal["support_gap_pct"], 0)

    def test_support_resistance_signal_detects_confirmed_upper_breakout(self):
        rows = []
        for _ in range(28):
            rows.append(
                {
                    "Open": 10.2,
                    "High": 10.5,
                    "Low": 10.0,
                    "Close": 10.22,
                    "Volume": 10000,
                    "volume_ratio": 0.9,
                    "volume_zscore_20": -0.1,
                }
            )
        rows.extend(
            [
                {"Open": 10.48, "High": 10.72, "Low": 10.45, "Close": 10.62, "Volume": 18000, "volume_ratio": 1.45, "volume_zscore_20": 0.4},
                {"Open": 10.63, "High": 10.78, "Low": 10.6, "Close": 10.68, "Volume": 21000, "volume_ratio": 1.65, "volume_zscore_20": 0.7},
            ]
        )
        features = pd.DataFrame(rows)

        signal = _support_resistance_signal(
            features,
            current_price=10.68,
            lookback_bars=20,
            confirm_bars=2,
            min_break_pct=0.15,
            atr_buffer_multiplier=0.25,
            min_volume_ratio=1.0,
            min_volume_zscore=-0.2,
        )

        self.assertEqual(signal["state"], "UPPER_RESISTANCE_BROKEN")
        self.assertTrue(signal["upper_breakout"])
        self.assertGreater(signal["resistance_gap_pct"], 0)

    def test_entry_structure_harness_blocks_lower_support_breakdown(self):
        payload = {
            "action": "BUY_WATCH",
            "reference_entry": 9.76,
            "take_profit": [10.5, 11.0],
            "recommended_shares": 3,
            "reason": [],
            "alerts": [],
            "risk_flags": [],
            "support_resistance": {
                "state": "LOWER_SUPPORT_BROKEN",
                "lower_breakdown": True,
                "upper_breakout": False,
                "support": 10.0,
                "resistance": 10.5,
                "support_gap_pct": -2.4,
                "resistance_gap_pct": -7.0,
                "break_buffer_pct": 0.2,
                "reason": ["LOWER_SUPPORT_BROKEN"],
            },
        }

        guarded = apply_entry_structure_harness(payload, enabled=True)

        self.assertEqual(guarded["action"], "NO_TRADE")
        self.assertEqual(guarded["recommended_shares"], 0)
        self.assertFalse(guarded["entry_structure"]["passed"])
        self.assertIn("ENTRY_BLOCKED_BY_LOWER_SUPPORT_BREAK", guarded["reason"])

    def test_support_breakdown_creates_exit_candidate(self):
        rows = []
        for _ in range(28):
            rows.append(
                {
                    "Open": 10.2,
                    "High": 10.5,
                    "Low": 10.0,
                    "Close": 10.22,
                    "Volume": 10000,
                    "volume_ratio": 0.9,
                    "volume_zscore_20": -0.1,
                }
            )
        rows.extend(
            [
                {"Open": 10.08, "High": 10.1, "Low": 9.78, "Close": 9.82, "Volume": 18000, "volume_ratio": 1.45, "volume_zscore_20": 0.4},
                {"Open": 9.83, "High": 9.88, "Low": 9.7, "Close": 9.76, "Volume": 21000, "volume_ratio": 1.65, "volume_zscore_20": 0.7},
            ]
        )
        features = pd.DataFrame(rows)
        plan = SimpleNamespace(price=9.76, stop_loss=9.0, take_profit_1=11.0)

        candidate = _select_exit_candidate(
            {"MOMO": (pd.DataFrame(), features, SimpleNamespace(probability_up=0.6), plan, pd.DataFrame())},
            {"MOMO": 4.0},
            tolerance_pct=0.5,
            position_summary=[
                {
                    "symbol": "MOMO",
                    "quantity": 4.0,
                    "entry_price": 10.1,
                    "current_price": 9.76,
                    "fee_adjusted_return_pct": -1.1,
                    "fee_adjusted_pnl_usd": -0.44,
                }
            ],
            manual_position_exit_enabled=False,
            target_decay_exit_enabled=False,
            support_breakdown_exit_enabled=True,
            structure_lookback_bars=20,
            structure_confirm_bars=2,
            structure_min_break_pct=0.15,
            structure_atr_buffer_multiplier=0.25,
            structure_min_volume_ratio=1.0,
            structure_min_volume_zscore=-0.2,
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["reason"], "SUPPORT_BREAKDOWN_EXIT")
        self.assertEqual(candidate["quantity"], 4)
        self.assertTrue(candidate["support_resistance"]["lower_breakdown"])

    def test_entry_edge_harness_allows_sufficient_net_target(self):
        payload = {
            "action": "BUY_WATCH",
            "reference_entry": 100.0,
            "take_profit": [102.0, 103.0],
            "recommended_shares": 2,
            "reason": [],
            "alerts": [],
            "risk_flags": [],
        }

        guarded = apply_entry_edge_harness(
            payload,
            commission_rate_pct=0.1,
            min_gross_target_pct=1.2,
            min_net_target_pct=0.5,
            fee_multiplier=3.0,
            slippage_buffer_pct=0.15,
            min_order_notional_usd=75.0,
        )

        self.assertEqual(guarded["action"], "BUY_WATCH")
        self.assertTrue(guarded["entry_edge"]["passed"])
        self.assertGreater(guarded["entry_edge"]["estimated_net_target_pct"], 1.0)

    def test_intraday_projection_targeting_expands_target_from_projected_high(self):
        payload = {
            "action": "BUY_WATCH",
            "reference_entry": 100.0,
            "current_price": 100.1,
            "take_profit": [101.2, 101.8],
            "prediction_metrics": {
                "horizon_bars": 60,
                "projected_high": 103.0,
                "projected_low": 98.8,
                "diagnostics": {
                    "today_observed_high": 102.0,
                    "today_observed_low": 99.1,
                    "today_projected_high": 103.0,
                    "today_projected_low": 98.8,
                },
            },
        }

        guarded = apply_intraday_projection_targeting(
            payload,
            capture_pct=85.0,
            min_gross_target_pct=1.6,
            max_gross_target_pct=3.0,
        )

        self.assertTrue(guarded["intraday_targeting"]["applied"])
        self.assertAlmostEqual(guarded["take_profit"][0], 102.55, places=2)
        self.assertGreater(guarded["take_profit"][0], payload["take_profit"][0])

    def test_intraday_projection_targeting_does_not_force_weak_projection(self):
        payload = {
            "action": "BUY_WATCH",
            "reference_entry": 100.0,
            "current_price": 100.0,
            "take_profit": [101.0, 101.5],
            "prediction_metrics": {
                "horizon_bars": 60,
                "projected_high": 101.2,
                "projected_low": 99.2,
                "diagnostics": {},
            },
        }

        guarded = apply_intraday_projection_targeting(
            payload,
            capture_pct=85.0,
            min_gross_target_pct=1.6,
            max_gross_target_pct=3.0,
        )

        self.assertFalse(guarded["intraday_targeting"]["applied"])
        self.assertEqual(guarded["take_profit"], payload["take_profit"])
        self.assertIn("PROJECTED_HIGH_BELOW_MIN_TARGET", guarded["intraday_targeting"]["reason"])

    def test_entry_edge_harness_blocks_when_projection_cannot_reach_target(self):
        payload = {
            "action": "BUY_WATCH",
            "reference_entry": 113.865,
            "take_profit": [115.0947, 115.5047],
            "recommended_shares": 3,
            "prediction_metrics": {
                "expected_move_pct": 0.0607,
                "projected_high": 114.2748,
            },
            "reason": [],
            "alerts": [],
            "risk_flags": [],
        }

        guarded = apply_entry_edge_harness(
            payload,
            commission_rate_pct=0.1,
            min_gross_target_pct=0.5,
            min_net_target_pct=0.0,
            fee_multiplier=0.0,
            slippage_buffer_pct=0.0,
            min_order_notional_usd=75.0,
            min_projected_target_coverage_pct=75.0,
            min_expected_move_pct=0.08,
        )

        self.assertEqual(guarded["action"], "NO_TRADE")
        self.assertFalse(guarded["entry_edge"]["passed"])
        self.assertIn("ENTRY_PROJECTED_TARGET_COVERAGE_WEAK", guarded["reason"])
        self.assertIn("ENTRY_EXPECTED_MOVE_BELOW_MINIMUM", guarded["reason"])
        self.assertLess(guarded["entry_edge"]["projected_target_coverage_pct"], 75.0)

    def test_daily_order_guard_blocks_after_three_estimated_losses(self):
        now = datetime(2026, 7, 1, 3, 40, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "auto_trade_events.jsonl"
            rows = [
                ("2026-07-01T00:01:00+09:00", "AAA", "BUY", 1, 100.0),
                ("2026-07-01T00:04:00+09:00", "AAA", "SELL", 1, 99.5),
                ("2026-07-01T00:06:00+09:00", "BBB", "BUY", 1, 50.0),
                ("2026-07-01T00:09:00+09:00", "BBB", "SELL", 1, 49.7),
                ("2026-07-01T00:11:00+09:00", "CCC", "BUY", 1, 20.0),
                ("2026-07-01T00:14:00+09:00", "CCC", "SELL", 1, 19.9),
            ]
            with path.open("w", encoding="utf-8") as file:
                for created_at, symbol, side, quantity, price in rows:
                    file.write(
                        json.dumps(
                            {
                                "created_at": created_at,
                                "status": "LIVE_SUBMITTED",
                                "order": {
                                    "symbol": symbol,
                                    "side": side,
                                    "quantity": quantity,
                                    "limit_price": price,
                                },
                            }
                        )
                        + "\n"
                    )

            stats = _daily_order_guard_stats(path, now=now, commission_rate_pct=0.1)

        guarded = apply_daily_order_guard(
            {"action": "BUY_WATCH", "recommended_shares": 1, "reason": [], "alerts": [], "risk_flags": []},
            stats=stats,
            enabled=True,
            max_loss_usd=3.0,
            max_consecutive_losses=3,
        )

        self.assertEqual(stats["closed_roundtrips"], 3)
        self.assertEqual(stats["consecutive_losses"], 3)
        self.assertEqual(guarded["action"], "NO_TRADE")
        self.assertIn("DAILY_CONSECUTIVE_LOSS_LIMIT_REACHED", guarded["reason"])

    def test_daily_order_guard_uses_account_pct_when_fixed_loss_is_zero(self):
        limit = effective_daily_order_guard_max_loss_usd(
            configured_max_loss_usd=0.0,
            account_usd=448.03,
            daily_loss_limit_pct=10.0,
            daily_loss_limit_usd=0.0,
        )

        self.assertAlmostEqual(limit, 44.803)

    def test_daily_order_guard_fixed_loss_overrides_pct_when_set(self):
        limit = effective_daily_order_guard_max_loss_usd(
            configured_max_loss_usd=3.0,
            account_usd=448.03,
            daily_loss_limit_pct=10.0,
            daily_loss_limit_usd=0.0,
        )

        self.assertEqual(limit, 3.0)

    def test_watchlist_no_longer_caps_at_twelve_symbols(self):
        symbols = [f"T{i}" for i in range(20)]

        result = _watchlist({"watchlist": symbols})

        self.assertEqual(len(result), 20)
        self.assertEqual(result[-1], "T19")

    def test_wide_symbol_pool_merges_configured_and_default_symbols(self):
        result = _wide_symbol_pool(
            {
                "watchlist": ["AAA"],
                "candidate_universe": ["BBB"],
                "wide_candidate_universe": ["CCC"],
            },
            max_symbols=6,
        )

        self.assertEqual(result[:3], ["AAA", "CCC", "BBB"])
        self.assertIn("AAPL", result)

    def test_rotation_batch_wraps_and_advances_cursor(self):
        batch, cursor = _rotation_batch(["A", "B", "C", "D"], cursor=2, batch_size=3)

        self.assertEqual(batch, ["C", "D", "A"])
        self.assertEqual(cursor, 1)

    def test_wide_fast_filter_promotes_liquid_volume_or_surge(self):
        row = {
            "liquidity_passed": True,
            "candle_count": 80,
            "surge_score": 2.0,
            "lookback_change_pct": 0.1,
            "surge_volume_ratio": 1.4,
        }

        self.assertTrue(
            _passes_wide_fast_filter(
                row,
                min_surge_score=4.0,
                min_lookback_change_pct=0.2,
                min_volume_ratio=1.1,
                min_model_candles=80,
            )
        )
        row["candle_count"] = 30
        self.assertFalse(
            _passes_wide_fast_filter(
                row,
                min_surge_score=4.0,
                min_lookback_change_pct=0.2,
                min_volume_ratio=1.1,
                min_model_candles=80,
            )
        )
        row["candle_count"] = 80
        row["liquidity_passed"] = False
        self.assertFalse(
            _passes_wide_fast_filter(
                row,
                min_surge_score=4.0,
                min_lookback_change_pct=0.2,
                min_volume_ratio=1.1,
                min_model_candles=80,
            )
        )

    def test_fast_scan_features_work_with_short_candle_windows(self):
        prices = pd.DataFrame(
            {
                "Open": [10.0 + i * 0.1 for i in range(30)],
                "High": [10.2 + i * 0.1 for i in range(30)],
                "Low": [9.9 + i * 0.1 for i in range(30)],
                "Close": [10.1 + i * 0.1 for i in range(30)],
                "Volume": [1000 + i * 100 for i in range(30)],
            }
        )

        features = _fast_scan_features(prices)

        self.assertEqual(len(features), 30)
        self.assertIn("volume_ratio", features.columns)
        self.assertIn("volume_zscore_20", features.columns)
        self.assertIn("rsi_14", features.columns)

    def test_modeled_universe_prioritizes_pinned_and_dynamic_symbols(self):
        result = _modeled_universe(
            pinned_symbols=["AAPL", "MSTR"],
            dynamic_symbols=["FAST", "AAPL"],
            base_symbols=["SLOW", "META", "NVDA"],
            max_symbols=4,
        )

        self.assertEqual(result, ["AAPL", "MSTR", "FAST", "SLOW"])

    def test_liquidity_filter_uses_recent_volume_and_dollar_volume(self):
        features = pd.DataFrame(
            {
                "Close": [10.0, 11.0, 12.0],
                "Volume": [1000.0, 2000.0, 3000.0],
            }
        )

        metrics = _liquidity_metrics(features, recent_bars=2)

        self.assertEqual(metrics["recent_volume"], 5000.0)
        self.assertEqual(metrics["recent_dollar_volume"], 58_000.0)
        self.assertTrue(
            _passes_liquidity_filter(
                metrics,
                min_recent_volume=5000.0,
                min_recent_dollar_volume=58_000.0,
            )
        )
        self.assertFalse(
            _passes_liquidity_filter(
                metrics,
                min_recent_volume=5001.0,
                min_recent_dollar_volume=58_000.0,
            )
        )

    def test_holding_quantity_map_parses_items(self):
        payload = {"items": [{"symbol": "MANU", "quantity": "3"}, {"symbol": "AAPL", "quantity": "0.25"}]}

        quantities = _holding_quantity_map(payload)

        self.assertEqual(quantities["MANU"], 3.0)
        self.assertEqual(quantities["AAPL"], 0.25)

    def test_position_summary_includes_fee_adjusted_return(self):
        payload = {
            "items": [
                {
                    "symbol": "AAPL",
                    "quantity": "2",
                    "averagePurchasePrice": "100",
                    "currentPrice": "110",
                }
            ]
        }
        commissions = [{"marketCountry": "US", "commissionRate": "0.015"}]

        rows = _position_summary_rows(payload, {}, commissions)

        self.assertEqual(rows[0]["symbol"], "AAPL")
        self.assertEqual(rows[0]["quantity"], 2.0)
        self.assertEqual(rows[0]["entry_price"], 100.0)
        self.assertEqual(rows[0]["current_price"], 110.0)
        self.assertEqual(rows[0]["gross_pnl_usd"], 20.0)
        self.assertAlmostEqual(rows[0]["estimated_roundtrip_fee_usd"], 0.063, places=3)
        self.assertAlmostEqual(rows[0]["fee_adjusted_pnl_usd"], 19.937, places=3)
        self.assertAlmostEqual(rows[0]["fee_adjusted_return_pct"], 9.9685, places=4)

    def test_position_summary_prefers_manual_target_when_plan_target_is_tighter(self):
        payload = {
            "items": [
                {
                    "symbol": "F",
                    "quantity": "12",
                    "averagePurchasePrice": "13.8",
                    "currentPrice": "13.9",
                }
            ]
        }
        plan = SimpleNamespace(price=13.9, stop_loss=13.7, take_profit_1=13.95)
        loaded = {"F": (pd.DataFrame(), pd.DataFrame(), SimpleNamespace(), plan, pd.DataFrame())}

        rows = _position_summary_rows(payload, loaded, [], manual_take_profit_pct=7.0)

        self.assertEqual(rows[0]["target_source"], "manual")
        self.assertAlmostEqual(rows[0]["take_profit_1"], 14.766, places=3)
        self.assertGreater(rows[0]["target_gap_pct"], 6.0)

    def test_manual_position_take_profit_can_create_exit_candidate(self):
        candidate = _select_exit_candidate(
            {},
            {"UPC": 3.0},
            tolerance_pct=0.5,
            position_summary=[
                {
                    "symbol": "UPC",
                    "quantity": 3.0,
                    "entry_price": 12.0,
                    "current_price": 12.5,
                    "fee_adjusted_return_pct": 4.1,
                    "fee_adjusted_pnl_usd": 1.48,
                }
            ],
            manual_position_exit_enabled=True,
            manual_take_profit_pct=2.0,
            manual_stop_loss_pct=2.0,
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["symbol"], "UPC")
        self.assertEqual(candidate["reason"], "MANUAL_POSITION_TAKE_PROFIT")
        self.assertEqual(candidate["quantity"], 3)
        self.assertTrue(candidate["manual_exit"])

    def test_loaded_position_does_not_exit_at_tight_plan_target_when_manual_target_is_higher(self):
        features = pd.DataFrame([{"Close": 10.55, "High": 10.56, "Low": 10.5, "Volume": 100_000}])
        plan = SimpleNamespace(price=10.55, stop_loss=10.0, take_profit_1=10.45)
        result = SimpleNamespace(probability_up=0.7)

        candidate = _select_exit_candidate(
            {"AAA": (pd.DataFrame(), features, result, plan, pd.DataFrame())},
            {"AAA": 2.0},
            tolerance_pct=0.5,
            position_summary=[
                {
                    "symbol": "AAA",
                    "quantity": 2.0,
                    "entry_price": 10.0,
                    "current_price": 10.55,
                    "fee_adjusted_return_pct": 5.0,
                    "fee_adjusted_pnl_usd": 1.0,
                }
            ],
            manual_position_exit_enabled=True,
            manual_take_profit_pct=7.0,
            manual_stop_loss_pct=2.0,
            target_decay_exit_enabled=False,
        )

        self.assertIsNone(candidate)

    def test_loaded_position_prefers_wider_manual_stop_over_tight_plan_stop(self):
        features = pd.DataFrame([{"Close": 9.9, "High": 10.0, "Low": 9.88, "Volume": 100_000}])
        plan = SimpleNamespace(price=9.9, stop_loss=9.95, take_profit_1=10.7)
        result = SimpleNamespace(probability_up=0.7)

        candidate = _select_exit_candidate(
            {"AAA": (pd.DataFrame(), features, result, plan, pd.DataFrame())},
            {"AAA": 2.0},
            tolerance_pct=0.5,
            position_summary=[
                {
                    "symbol": "AAA",
                    "quantity": 2.0,
                    "entry_price": 10.0,
                    "current_price": 9.9,
                    "fee_adjusted_return_pct": -1.2,
                    "fee_adjusted_pnl_usd": -0.24,
                }
            ],
            manual_position_exit_enabled=True,
            manual_take_profit_pct=7.0,
            manual_stop_loss_pct=2.0,
            target_decay_exit_enabled=False,
        )

        self.assertIsNone(candidate)

    def test_manual_position_stop_loss_has_priority_over_take_profit(self):
        candidate = _select_exit_candidate(
            {},
            {"UPC": 3.0, "DCOY": 2.0},
            tolerance_pct=0.5,
            position_summary=[
                {
                    "symbol": "UPC",
                    "quantity": 3.0,
                    "entry_price": 12.0,
                    "current_price": 12.5,
                    "fee_adjusted_return_pct": 4.1,
                },
                {
                    "symbol": "DCOY",
                    "quantity": 2.0,
                    "entry_price": 10.0,
                    "current_price": 9.6,
                    "fee_adjusted_return_pct": -4.2,
                },
            ],
            manual_position_exit_enabled=True,
            manual_take_profit_pct=2.0,
            manual_stop_loss_pct=2.0,
        )

        self.assertEqual(candidate["symbol"], "DCOY")
        self.assertEqual(candidate["reason"], "MANUAL_POSITION_STOP_LOSS")

    def test_target_extension_signal_detects_live_momentum(self):
        features = pd.DataFrame(
            {
                "Open": [10.0, 10.03, 10.06, 10.1, 10.14, 10.18],
                "High": [10.04, 10.08, 10.13, 10.17, 10.22, 10.26],
                "Low": [9.98, 10.02, 10.05, 10.09, 10.13, 10.17],
                "Close": [10.03, 10.07, 10.12, 10.16, 10.2, 10.23],
                "Volume": [1000, 1200, 2000, 3500, 5000, 7500],
                "volume_ratio": [1.0, 1.1, 1.4, 1.8, 2.2, 2.6],
                "volume_zscore_20": [0.0, 0.1, 0.5, 1.0, 1.4, 2.0],
                "vwap_distance_pct": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8],
                "rsi_14": [55, 57, 60, 64, 67, 71],
                "ema_9": [10.0, 10.03, 10.06, 10.1, 10.14, 10.18],
                "ema_21": [9.98, 10.0, 10.03, 10.06, 10.09, 10.12],
            }
        )

        signal = _target_extension_signal(
            features,
            current_price=10.9,
            target_price=10.88,
            enabled=True,
            max_extension_pct=0.6,
            trailing_stop_pct=0.35,
            min_surge_score=8.0,
            min_lookback_change_pct=0.4,
            min_latest_bar_change_pct=0.0,
            min_volume_ratio=1.1,
            max_rsi=84.0,
        )

        self.assertTrue(signal["active"])
        self.assertEqual(signal["reason"], ["TARGET_EXTENSION_ACTIVE"])
        self.assertGreater(signal["extended_target_price"], 10.88)

    def test_manual_take_profit_can_be_held_when_target_extension_is_active(self):
        features = pd.DataFrame(
            {
                "Open": [10.0, 10.1, 10.2, 10.35, 10.5, 10.7],
                "High": [10.12, 10.22, 10.42, 10.56, 10.75, 10.95],
                "Low": [9.98, 10.05, 10.16, 10.3, 10.45, 10.66],
                "Close": [10.08, 10.18, 10.38, 10.52, 10.72, 10.9],
                "Volume": [1000, 1200, 2000, 3500, 5000, 7500],
                "volume_ratio": [1.0, 1.1, 1.4, 1.8, 2.2, 2.6],
                "volume_zscore_20": [0.0, 0.1, 0.5, 1.0, 1.4, 2.0],
                "vwap_distance_pct": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8],
                "rsi_14": [55, 57, 60, 64, 67, 71],
                "ema_9": [10.0, 10.08, 10.2, 10.35, 10.52, 10.7],
                "ema_21": [9.98, 10.02, 10.08, 10.18, 10.3, 10.45],
            }
        )

        candidate = _select_exit_candidate(
            {"MOMO": (pd.DataFrame(), features, None, None, pd.DataFrame())},
            {"MOMO": 5.0},
            tolerance_pct=0.5,
            position_summary=[
                {
                    "symbol": "MOMO",
                    "quantity": 5.0,
                    "entry_price": 10.0,
                    "current_price": 10.23,
                    "fee_adjusted_return_pct": 2.2,
                    "fee_adjusted_pnl_usd": 1.1,
                }
            ],
            manual_position_exit_enabled=True,
            manual_take_profit_pct=2.0,
            manual_stop_loss_pct=2.0,
            target_extension_enabled=True,
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["reason"], "TARGET_EXTENSION_ACTIVE")
        self.assertEqual(candidate["quantity"], 0)
        self.assertTrue(candidate["target_extension"]["active"])

    def test_target_decay_exit_signal_detects_weak_probability_and_volume(self):
        features = pd.DataFrame(
            {
                "Open": [10.2, 10.12, 10.02, 9.92],
                "High": [10.25, 10.14, 10.04, 9.94],
                "Low": [10.08, 9.98, 9.88, 9.78],
                "Close": [10.1, 10.0, 9.9, 9.82],
                "Volume": [4000, 2600, 1800, 1200],
                "volume_ratio": [1.2, 0.8, 0.62, 0.55],
                "volume_zscore_20": [0.2, -0.1, -0.45, -0.65],
            }
        )

        signal = _target_decay_exit_signal(
            features,
            SimpleNamespace(probability_up=0.36),
            current_price=9.82,
            entry_price=10.0,
            stop_loss=9.55,
            target_price=10.45,
            enabled=True,
            min_probability_pct=40.0,
            min_target_gap_pct=0.8,
            max_volume_ratio=0.7,
            max_volume_zscore=-0.35,
            min_weak_bars=2,
        )

        self.assertTrue(signal["active"])
        self.assertEqual(signal["reason"], ["TARGET_DECAY_EXIT_ACTIVE"])
        self.assertLess(signal["probability_pct"], 40.0)
        self.assertLess(signal["volume_ratio"], 0.7)

    def test_target_decay_exit_becomes_sell_candidate(self):
        features = pd.DataFrame(
            {
                "Open": [10.2, 10.12, 10.02, 9.92],
                "High": [10.25, 10.14, 10.04, 9.94],
                "Low": [10.08, 9.98, 9.88, 9.78],
                "Close": [10.1, 10.0, 9.9, 9.82],
                "Volume": [4000, 2600, 1800, 1200],
                "volume_ratio": [1.2, 0.8, 0.62, 0.55],
                "volume_zscore_20": [0.2, -0.1, -0.45, -0.65],
            }
        )
        plan = SimpleNamespace(price=9.82, stop_loss=9.55, take_profit_1=10.45)

        candidate = _select_exit_candidate(
            {"MOMO": (pd.DataFrame(), features, SimpleNamespace(probability_up=0.36), plan, pd.DataFrame())},
            {"MOMO": 4.0},
            tolerance_pct=0.5,
            position_summary=[
                {
                    "symbol": "MOMO",
                    "quantity": 4.0,
                    "entry_price": 10.0,
                    "current_price": 9.82,
                    "fee_adjusted_return_pct": -0.6,
                    "fee_adjusted_pnl_usd": -0.24,
                }
            ],
            manual_position_exit_enabled=True,
            manual_take_profit_pct=2.0,
            manual_stop_loss_pct=2.0,
            target_decay_exit_enabled=True,
            target_decay_exit_min_probability_pct=40.0,
            target_decay_exit_min_target_gap_pct=0.8,
            target_decay_exit_max_volume_ratio=0.7,
            target_decay_exit_max_volume_zscore=-0.35,
            target_decay_exit_min_weak_bars=2,
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["reason"], "EARLY_EXIT_TARGET_PROBABILITY_DECAY")
        self.assertEqual(candidate["quantity"], 4)
        self.assertTrue(candidate["target_decay_exit"]["active"])

    def test_target_decay_exit_does_not_fire_when_volume_is_still_strong(self):
        features = pd.DataFrame(
            {
                "Open": [10.2, 10.12, 10.02, 9.92],
                "High": [10.25, 10.14, 10.04, 9.94],
                "Low": [10.08, 9.98, 9.88, 9.78],
                "Close": [10.1, 10.0, 9.9, 9.82],
                "Volume": [4000, 4200, 4300, 4500],
                "volume_ratio": [1.2, 1.25, 1.3, 1.35],
                "volume_zscore_20": [0.2, 0.35, 0.45, 0.55],
            }
        )

        signal = _target_decay_exit_signal(
            features,
            SimpleNamespace(probability_up=0.36),
            current_price=9.82,
            entry_price=10.0,
            stop_loss=9.55,
            target_price=10.45,
            enabled=True,
            min_probability_pct=40.0,
            min_target_gap_pct=0.8,
            max_volume_ratio=0.7,
            max_volume_zscore=-0.35,
            min_weak_bars=2,
        )

        self.assertFalse(signal["active"])
        self.assertIn("VOLUME_NOT_WEAK", signal["reason"])

    def test_rebound_potential_signal_detects_historical_support_bounce(self):
        rows = []
        for i in range(140):
            trend = 10.0 + i * 0.004
            phase = i % 6
            if phase in (0, 1, 2):
                close = trend + phase * 0.015
            elif phase == 3:
                close = trend - 0.03
            elif phase == 4:
                close = trend + 0.05
            else:
                close = trend + 0.08
            rows.append(
                {
                    "Open": close + 0.01,
                    "High": close + 0.05,
                    "Low": close - 0.04,
                    "Close": close,
                    "Volume": 10000,
                    "volume_ratio": 0.6,
                    "volume_zscore_20": -0.2,
                    "rsi_14": 46,
                }
            )
        rows[-3].update({"Close": 10.66, "Open": 10.70, "High": 10.71, "Low": 10.63, "Volume": 8000, "volume_ratio": 0.55, "volume_zscore_20": -0.7, "rsi_14": 45})
        rows[-2].update({"Close": 10.58, "Open": 10.63, "High": 10.64, "Low": 10.55, "Volume": 6500, "volume_ratio": 0.42, "volume_zscore_20": -0.9, "rsi_14": 43})
        rows[-1].update({"Close": 10.50, "Open": 10.56, "High": 10.57, "Low": 10.47, "Volume": 5000, "volume_ratio": 0.35, "volume_zscore_20": -1.2, "rsi_14": 41})
        features = pd.DataFrame(rows)
        features["support_20_prev"] = features["Low"].rolling(20, min_periods=5).min().shift(1)
        features["ema_21"] = features["Close"].ewm(span=21, adjust=False).mean()
        typical = (features["High"] + features["Low"] + features["Close"]) / 3
        features["vwap_60"] = (typical * features["Volume"]).rolling(60, min_periods=10).sum() / features["Volume"].rolling(
            60, min_periods=10
        ).sum()

        rebound = _rebound_potential_signal(
            features,
            current_price=10.50,
            entry_price=10.60,
            stop_loss=10.25,
            min_score=60,
            min_samples=5,
            lookback_bars=160,
            forward_bars=8,
        )

        self.assertTrue(rebound["active"])
        self.assertGreaterEqual(rebound["hit_rate_pct"], 45)
        self.assertGreaterEqual(rebound["score"], 60)

    def test_target_decay_exit_is_blocked_by_rebound_potential(self):
        rows = []
        for i in range(140):
            trend = 10.0 + i * 0.004
            phase = i % 6
            if phase in (0, 1, 2):
                close = trend + phase * 0.015
            elif phase == 3:
                close = trend - 0.03
            elif phase == 4:
                close = trend + 0.05
            else:
                close = trend + 0.08
            rows.append(
                {
                    "Open": close + 0.01,
                    "High": close + 0.05,
                    "Low": close - 0.04,
                    "Close": close,
                    "Volume": 10000,
                    "volume_ratio": 0.6,
                    "volume_zscore_20": -0.2,
                    "rsi_14": 46,
                }
            )
        rows[-3].update({"Close": 10.66, "Open": 10.70, "High": 10.71, "Low": 10.63, "Volume": 8000, "volume_ratio": 0.55, "volume_zscore_20": -0.7, "rsi_14": 45})
        rows[-2].update({"Close": 10.58, "Open": 10.63, "High": 10.64, "Low": 10.55, "Volume": 6500, "volume_ratio": 0.42, "volume_zscore_20": -0.9, "rsi_14": 43})
        rows[-1].update({"Close": 10.50, "Open": 10.56, "High": 10.57, "Low": 10.47, "Volume": 5000, "volume_ratio": 0.35, "volume_zscore_20": -1.2, "rsi_14": 41})
        features = pd.DataFrame(rows)
        features["support_20_prev"] = features["Low"].rolling(20, min_periods=5).min().shift(1)
        features["ema_21"] = features["Close"].ewm(span=21, adjust=False).mean()
        typical = (features["High"] + features["Low"] + features["Close"]) / 3
        features["vwap_60"] = (typical * features["Volume"]).rolling(60, min_periods=10).sum() / features["Volume"].rolling(
            60, min_periods=10
        ).sum()

        signal = _target_decay_exit_signal(
            features,
            SimpleNamespace(probability_up=0.28),
            current_price=10.50,
            entry_price=10.60,
            stop_loss=10.25,
            target_price=11.342,
            enabled=True,
            min_probability_pct=35.0,
            min_target_gap_pct=3.0,
            max_volume_ratio=0.4,
            max_volume_zscore=-1.0,
            min_weak_bars=2,
            max_fee_adjusted_return_pct=-0.8,
            fee_adjusted_return_pct=-0.95,
            rebound_protection_enabled=True,
            rebound_min_score=60,
            rebound_min_samples=5,
            rebound_lookback_bars=160,
            rebound_forward_bars=8,
        )

        self.assertFalse(signal["active"])
        self.assertIn("REBOUND_POTENTIAL_PRESENT", signal["reason"])
        self.assertTrue(signal["rebound_potential"]["active"])

    def test_big_target_decay_exit_ignores_small_fee_adjusted_noise(self):
        features = pd.DataFrame(
            {
                "Open": [10.2, 10.12, 10.02, 9.92, 9.86, 9.83],
                "High": [10.25, 10.14, 10.04, 9.94, 9.88, 9.85],
                "Low": [10.08, 9.98, 9.88, 9.78, 9.76, 9.74],
                "Close": [10.1, 10.0, 9.9, 9.82, 9.79, 9.77],
                "Volume": [4000, 2600, 1800, 1200, 900, 700],
                "volume_ratio": [1.2, 0.8, 0.62, 0.45, 0.35, 0.25],
                "volume_zscore_20": [0.2, -0.1, -0.45, -0.9, -1.1, -1.3],
            }
        )

        signal = _target_decay_exit_signal(
            features,
            SimpleNamespace(probability_up=0.28),
            current_price=9.77,
            entry_price=10.0,
            stop_loss=9.5,
            target_price=10.7,
            enabled=True,
            min_probability_pct=35.0,
            min_target_gap_pct=3.0,
            max_volume_ratio=0.4,
            max_volume_zscore=-1.0,
            min_weak_bars=5,
            max_fee_adjusted_return_pct=-0.8,
            fee_adjusted_return_pct=-0.31,
        )

        self.assertFalse(signal["active"])
        self.assertIn("POSITION_NOT_WEAK_AFTER_FEES", signal["reason"])

    def test_reentry_cooldown_blocks_buy_after_target_decay_exit(self):
        now = datetime(2026, 7, 1, 1, 50, tzinfo=timezone.utc)
        events = [
            {
                "logged_at": (now - timedelta(minutes=2)).isoformat(),
                "status": "LIVE_SUBMITTED",
                "symbol": "SMCI",
                "trade_intent_side": "SELL",
                "auto_selection": {
                    "symbol": "SMCI",
                    "reason": ["EARLY_EXIT_TARGET_PROBABILITY_DECAY", "AUTO_EXIT_PRIORITY"],
                    "current_price": 29.58,
                },
                "reasons": ["EARLY_EXIT_TARGET_PROBABILITY_DECAY"],
            }
        ]
        scanner = pd.DataFrame(
            [
                {
                    "symbol": "SMCI",
                    "current_price": 29.63,
                    "confidence_pct": 63.7,
                    "volume_ratio": 3.17,
                    "surge_volume_ratio": 3.17,
                    "direction": "LONG",
                    "risk": "NORMAL",
                }
            ]
        )

        annotated, summary = _apply_reentry_cooldown(
            scanner,
            events,
            now=now,
            enabled=True,
            cooldown_minutes=8.0,
            min_breakout_pct=0.5,
            min_volume_ratio=1.5,
            min_confidence_pct=60.0,
        )

        self.assertEqual(summary["blocked_count"], 1)
        self.assertTrue(bool(annotated.iloc[0]["reentry_cooldown_blocked"]))
        self.assertTrue(_filter_reentry_cooldown(annotated).empty)

    def test_reentry_cooldown_allows_strong_rebreakout(self):
        now = datetime(2026, 7, 1, 1, 50, tzinfo=timezone.utc)
        events = [
            {
                "logged_at": (now - timedelta(minutes=2)).isoformat(),
                "status": "LIVE_SUBMITTED",
                "symbol": "SMCI",
                "trade_intent_side": "SELL",
                "auto_selection": {
                    "symbol": "SMCI",
                    "reason": ["EARLY_EXIT_TARGET_PROBABILITY_DECAY", "AUTO_EXIT_PRIORITY"],
                    "current_price": 29.58,
                },
                "reasons": ["EARLY_EXIT_TARGET_PROBABILITY_DECAY"],
            }
        ]
        scanner = pd.DataFrame(
            [
                {
                    "symbol": "SMCI",
                    "current_price": 29.78,
                    "confidence_pct": 63.7,
                    "volume_ratio": 3.17,
                    "surge_volume_ratio": 3.17,
                    "direction": "LONG",
                    "risk": "NORMAL",
                }
            ]
        )

        annotated, summary = _apply_reentry_cooldown(
            scanner,
            events,
            now=now,
            enabled=True,
            cooldown_minutes=8.0,
            min_breakout_pct=0.5,
            min_volume_ratio=1.5,
            min_confidence_pct=60.0,
        )

        self.assertEqual(summary["blocked_count"], 0)
        self.assertEqual(summary["reentry_allowed_count"], 1)
        self.assertFalse(bool(annotated.iloc[0]["reentry_cooldown_blocked"]))

    def test_exit_reason_uses_stop_and_target_tolerance(self):
        self.assertEqual(_exit_reason(99.0, 100.0, 110.0, 0.5), "STOP_LOSS_REACHED")
        self.assertEqual(_exit_reason(109.45, 100.0, 110.0, 0.5), "TARGET_WITHIN_TOLERANCE")
        self.assertEqual(_exit_reason(109.44, 100.0, 110.0, 0.5), "")

    def test_build_exit_trade_intent_creates_sell_intent(self):
        intent = _build_exit_trade_intent(
            symbol="MANU",
            quantity=3,
            current_price=22.94,
            stop_loss=21.5,
            take_profit=[23.0, 24.0],
            account_usd=500.0,
            daily_realized_pnl_usd=0.0,
            daily_loss_limit_pct=10.0,
            daily_loss_limit_usd=0.0,
            data_quality={},
            reason="TARGET_WITHIN_TOLERANCE",
            holding_qty=3.0,
            current_session_entries=0,
            max_session_entries=3,
            manual_approval_complete=True,
        )

        self.assertEqual(intent.side, "SELL")
        self.assertEqual(intent.quantity, 3)
        self.assertEqual(intent.status, "DRY_RUN_READY")
        self.assertIn("TARGET_WITHIN_TOLERANCE", intent.reason)

    def test_surge_metrics_capture_short_term_gain_and_volume(self):
        features = pd.DataFrame(
            {
                "Close": [100.0, 100.2, 100.5, 101.0, 101.3, 101.8],
                "High": [100.1, 100.3, 100.6, 101.1, 101.4, 101.9],
                "Volume": [1000, 1200, 1300, 2500, 4000, 7000],
                "volume_ratio": [1.0, 1.1, 1.2, 1.8, 2.4, 3.2],
                "volume_zscore_20": [0.0, 0.1, 0.2, 0.8, 1.2, 2.0],
                "vwap_distance_pct": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8],
                "rsi_14": [55, 56, 58, 61, 64, 67],
            }
        )

        metrics = _surge_metrics(features, lookback_bars=5)

        self.assertGreater(metrics["lookback_change_pct"], 1.0)
        self.assertGreater(metrics["surge_score"], 8.0)

    def test_surge_candidate_scanner_filters_non_surge_rows(self):
        scanner = pd.DataFrame(
            [
                {
                    "symbol": "FAST",
                    "lookback_change_pct": 1.1,
                    "latest_bar_change_pct": 0.2,
                    "surge_volume_ratio": 2.0,
                    "surge_volume_zscore": 0.5,
                    "surge_score": 18.0,
                },
                {
                    "symbol": "SLOW",
                    "lookback_change_pct": 0.1,
                    "latest_bar_change_pct": 0.0,
                    "surge_volume_ratio": 3.0,
                    "surge_volume_zscore": 1.0,
                    "surge_score": 4.0,
                },
            ]
        )

        focused = _surge_candidate_scanner(
            scanner,
            enabled=True,
            min_lookback_change_pct=0.4,
            min_latest_bar_change_pct=0.0,
            min_volume_ratio=1.5,
            min_volume_zscore=0.0,
            min_surge_score=8.0,
        )

        self.assertEqual(list(focused["symbol"]), ["FAST"])

    def test_candidate_score_rows_expose_full_monitor_fields(self):
        scanner = pd.DataFrame(
            [
                {
                    "symbol": "SLOW",
                    "auto_score": 12.0,
                    "confidence_pct": 50.0,
                    "direction": "WAIT",
                    "change_pct": 0.1,
                    "volume_ratio": 1.0,
                    "volume_zscore": -0.2,
                    "rsi": 54.0,
                    "risk": "NORMAL",
                    "surge_score": 4.0,
                    "lookback_change_pct": 0.1,
                    "latest_bar_change_pct": 0.0,
                    "surge_volume_ratio": 1.0,
                    "surge_volume_zscore": -0.1,
                    "breakout_pct": -0.2,
                    "recent_volume": 12_000.0,
                    "recent_dollar_volume": 1_200_000.0,
                },
                {
                    "symbol": "FAST",
                    "auto_score": 88.0,
                    "confidence_pct": 62.0,
                    "direction": "LONG",
                    "change_pct": 1.4,
                    "volume_ratio": 2.1,
                    "volume_zscore": 1.2,
                    "rsi": 61.0,
                    "risk": "NORMAL",
                    "surge_score": 18.0,
                    "lookback_change_pct": 1.1,
                    "latest_bar_change_pct": 0.3,
                    "surge_volume_ratio": 2.0,
                    "surge_volume_zscore": 0.5,
                    "breakout_pct": 0.2,
                    "recent_volume": 44_000.0,
                    "recent_dollar_volume": 4_400_000.0,
                },
            ]
        )

        rows = _candidate_score_rows(
            scanner,
            min_probability=55.0,
            min_volume_ratio=1.2,
            require_risk_normal=True,
            surge_focus_enabled=True,
            surge_min_lookback_change_pct=0.4,
            surge_min_latest_bar_change_pct=0.0,
            surge_min_volume_ratio=1.5,
            surge_min_volume_zscore=0.0,
            surge_min_score=8.0,
        )

        self.assertEqual(rows[0]["symbol"], "FAST")
        self.assertTrue(rows[0]["surge_passed"])
        self.assertTrue(rows[0]["auto_gate_passed"])
        self.assertEqual(rows[1]["symbol"], "SLOW")
        self.assertFalse(rows[1]["surge_passed"])


if __name__ == "__main__":
    unittest.main()
