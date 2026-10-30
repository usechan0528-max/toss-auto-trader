import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from stock_signal.daily_archive import (
    DailyArchivePaths,
    archive_session,
    build_strategy_adjustment,
    maybe_archive_completed_sessions,
    read_jsonl,
    session_date_for_timestamp,
)


class FakeBroker:
    def get_order(self, order_id: str) -> dict:
        return {
            "orderId": order_id,
            "status": "FILLED",
            "price": "12.82",
            "quantity": "3",
            "orderAmount": "38.46",
            "execution": {
                "filledQuantity": "3",
                "averageFilledPrice": "12.82",
                "filledAmount": "38.46",
                "commission": "0.03",
                "tax": "0.01",
                "filledAt": "2026-06-30T00:43:14.771+09:00",
            },
        }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def paths(temp_dir: str) -> DailyArchivePaths:
    base = Path(temp_dir)
    return DailyArchivePaths(
        session_events_path=base / "auto_session_events.jsonl",
        auto_trade_events_path=base / "auto_trade_events.jsonl",
        order_intent_events_path=base / "order_intent_events.jsonl",
        signal_events_path=base / "signal_events.jsonl",
        archive_root=base / "daily_archives",
        latest_summary_path=base / "latest_daily_summary.json",
        learning_samples_path=base / "strategy_learning_samples.jsonl",
        strategy_adjustments_path=base / "strategy_adjustments.json",
        settings_path=base / "app_settings.json",
    )


class DailyArchiveTests(unittest.TestCase):
    def test_session_date_uses_kst_cutoff(self):
        self.assertEqual(
            session_date_for_timestamp("2026-06-30T00:42:53+09:00", cutoff_hour_kst=9),
            "2026-06-29",
        )
        self.assertEqual(
            session_date_for_timestamp("2026-06-30T09:00:00+09:00", cutoff_hour_kst=9),
            "2026-06-30",
        )

    def test_archive_session_writes_summary_and_learning_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_paths = paths(temp_dir)
            archive_paths.settings_path.write_text(
                json.dumps(
                    {
                        "strategy_learning_enabled": True,
                        "strategy_learning_auto_apply_enabled": True,
                        "strategy_learning_min_closed_trades": 30,
                        "strategy_learning_min_sessions": 5,
                    }
                ),
                encoding="utf-8",
            )
            write_jsonl(
                archive_paths.session_events_path,
                [
                    {
                        "logged_at": "2026-06-29T15:42:53.900000+00:00",
                        "status": "LIVE_SUBMITTED",
                        "order_client_id": "intent-1",
                        "reasons": ["MANUAL_POSITION_STOP_LOSS"],
                        "position_summary": [
                            {
                                "symbol": "UPC",
                                "quantity": 3,
                                "entry_price": 13.2,
                                "current_price": 12.82,
                            }
                        ],
                        "candidate_scores": [
                            {
                                "symbol": "UPC",
                                "confidence_pct": 61.0,
                                "auto_score": 55.0,
                                "surge_score": 9.0,
                            }
                        ],
                    }
                ],
            )
            write_jsonl(
                archive_paths.auto_trade_events_path,
                [
                    {
                        "created_at": "2026-06-29T15:42:53.821101+00:00",
                        "status": "LIVE_SUBMITTED",
                        "order": {
                            "symbol": "UPC",
                            "side": "SELL",
                            "quantity": 3,
                            "order_type": "LIMIT",
                            "limit_price": 12.82,
                            "client_order_id": "intent-1",
                        },
                        "execution_result": {"broker_order_id": "order-1", "client_order_id": "intent-1"},
                        "checks": {"estimated_notional": 38.46},
                    }
                ],
            )

            summary = archive_session(
                "2026-06-29",
                archive_paths,
                settings=json.loads(archive_paths.settings_path.read_text(encoding="utf-8")),
                broker=FakeBroker(),
            )

            self.assertIsNotNone(summary)
            self.assertEqual(summary["counts"]["closed_trade_samples"], 1)
            self.assertEqual(summary["performance"]["losses"], 1)
            samples = read_jsonl(archive_paths.learning_samples_path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["outcome"], "LOSS")
            self.assertLess(samples[0]["net_pnl_usd"], 0)

    def test_maybe_archive_only_completed_sessions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_paths = paths(temp_dir)
            archive_paths.settings_path.write_text("{}", encoding="utf-8")
            write_jsonl(
                archive_paths.auto_trade_events_path,
                [
                    {
                        "created_at": "2026-06-29T15:42:53+00:00",
                        "status": "AUTO_BLOCKED",
                    }
                ],
            )

            archived = maybe_archive_completed_sessions(
                archive_paths,
                settings={},
                now=datetime(2026, 6, 30, 0, 1, tzinfo=timezone.utc),
                cutoff_hour_kst=9,
            )

            self.assertEqual([item["session_date"] for item in archived], ["2026-06-29"])
            self.assertTrue((archive_paths.archive_root / "2026-06-29" / "summary.json").exists())

    def test_strategy_adjustment_waits_for_meaningful_data_then_tightens(self):
        small = [
            {
                "sample_id": "s1",
                "session_date": "2026-06-29",
                "outcome": "LOSS",
                "net_pnl_usd": -1,
                "realized_return_pct": -1,
            }
        ]
        pending = build_strategy_adjustment(
            small,
            {"auto_selection_min_probability": 55.0, "auto_selection_min_volume_ratio": 1.2},
            min_closed_trades=3,
            min_sessions=2,
        )
        self.assertEqual(pending["status"], "INSUFFICIENT_DATA")

        enough_losses = [
            {
                "sample_id": f"s{i}",
                "session_date": "2026-06-29" if i < 2 else "2026-06-30",
                "outcome": "LOSS",
                "net_pnl_usd": -1.0,
                "realized_return_pct": -1.0,
            }
            for i in range(3)
        ]
        adjustment = build_strategy_adjustment(
            enough_losses,
            {"auto_selection_min_probability": 55.0, "auto_selection_min_volume_ratio": 1.2},
            min_closed_trades=3,
            min_sessions=2,
        )

        self.assertEqual(adjustment["status"], "READY")
        self.assertGreater(adjustment["proposed_settings"]["auto_selection_min_probability"], 55.0)
        self.assertGreater(adjustment["proposed_settings"]["auto_selection_min_volume_ratio"], 1.2)


if __name__ == "__main__":
    unittest.main()
