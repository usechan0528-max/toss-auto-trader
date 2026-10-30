import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from stock_signal.toss_data_collector import (
    TossApiClient,
    extract_candles,
    extract_orders,
    load_dotenv_file,
    load_symbols_from_local_state,
    normalize_toss_auth,
    parse_symbol_text,
    secret_value,
    unique_symbols,
)


class TossDataCollectorTests(unittest.TestCase):
    def test_parse_symbol_text_normalizes_and_deduplicates(self):
        self.assertEqual(parse_symbol_text("aapl, nvda\nAAPL,BRK.B"), ["AAPL", "NVDA", "BRK.B"])

    def test_extract_paginated_orders(self):
        rows, cursor = extract_orders({"result": {"orders": [{"orderId": "1"}], "nextCursor": "next"}})

        self.assertEqual(rows, [{"orderId": "1"}])
        self.assertEqual(cursor, "next")

    def test_extract_paginated_candles(self):
        rows, before = extract_candles({"result": {"candles": [{"closePrice": "10"}], "nextBefore": "2026"}})

        self.assertEqual(rows, [{"closePrice": "10"}])
        self.assertEqual(before, "2026")

    def test_load_symbols_from_local_state_combines_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app_settings.json").write_text('{"watchlist":["aapl","nvda"]}', encoding="utf-8")
            (root / "latest_signal.json").write_text(
                '{"symbol":"tsla","position_summary":[{"symbol":"smci"}],"auto_selection":{"top_surge_candidates":[{"symbol":"mstr"}]}}',
                encoding="utf-8",
            )
            symbols = load_symbols_from_local_state(root, include_default_universe=False)

        self.assertEqual(symbols, ["AAPL", "NVDA", "TSLA", "SMCI", "MSTR"])

    def test_dotenv_loader_does_not_override_existing_value(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"TOSS_BASE_URL": "existing"}, clear=False):
            path = Path(tmp) / ".env"
            path.write_text("TOSS_BASE_URL=from-file\n", encoding="utf-8")
            load_dotenv_file(path)

            import os

            self.assertEqual(os.environ["TOSS_BASE_URL"], "existing")

    def test_normalize_toss_auth_moves_client_key_from_token_slot(self):
        token, client_id, client_secret = normalize_toss_auth("tsck_live_key", "여기에_client_id", "tssk_secret")

        self.assertEqual(token, "")
        self.assertEqual(client_id, "tsck_live_key")
        self.assertEqual(client_secret, "tssk_secret")

    def test_secret_value_ignores_korean_placeholder(self):
        with patch.dict("os.environ", {"TOSS_ACCOUNT_SEQ": "여기에_account_seq"}, clear=False):
            self.assertEqual(secret_value("TOSS_ACCOUNT_SEQ"), "")

    def test_client_retries_rate_limit(self):
        first = Mock()
        first.status_code = 429
        first.headers = {}
        second = Mock()
        second.status_code = 200
        second.json.return_value = {"result": {"ok": True}}
        second.raise_for_status.return_value = None
        with patch("requests.get", side_effect=[first, second]), patch("time.sleep"):
            result = TossApiClient(token="token", request_pause_seconds=0, max_retries=1).get("/api/v1/accounts")

        self.assertEqual(result["result"]["ok"], True)

    def test_unique_symbols_filters_invalid_empty_values(self):
        self.assertEqual(unique_symbols(["", "aapl", "bad symbol", "AAPL", "BRK-B"]), ["AAPL", "BRK-B"])

    def test_unique_symbols_maps_legacy_ticker(self):
        self.assertEqual(unique_symbols(["SQ"]), ["XYZ"])


if __name__ == "__main__":
    unittest.main()
