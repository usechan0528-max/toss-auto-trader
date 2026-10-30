import unittest
from unittest.mock import Mock, patch

import pandas as pd

from stock_signal.market_data import CSVMarketDataAdapter, MockMarketDataAdapter, QuoteSnapshot, TossMarketDataAdapter


class MarketDataAdapterTests(unittest.TestCase):
    def test_quote_snapshot_calculates_spread_pct(self):
        quote = QuoteSnapshot(symbol="LASE", price=3.60, bid=3.58, ask=3.62)

        self.assertAlmostEqual(quote.spread_pct, 1.1111, places=4)

    def test_mock_adapter_returns_default_quote_from_latest_close(self):
        data = pd.DataFrame(
            {
                "Open": [3.1, 3.2],
                "High": [3.3, 3.5],
                "Low": [3.0, 3.1],
                "Close": [3.2, 3.4],
                "Volume": [1000, 2000],
            }
        )
        adapter = MockMarketDataAdapter(data=data)

        quote = adapter.get_quote("LASE")

        self.assertEqual(quote.symbol, "LASE")
        self.assertEqual(quote.price, 3.4)
        self.assertEqual(quote.spread_pct, 0.0)

    def test_csv_adapter_returns_quote_from_latest_close(self):
        data = pd.DataFrame(
            {
                "Datetime": ["2026-06-04 09:30:00", "2026-06-04 09:31:00"],
                "Open": [3.1, 3.2],
                "High": [3.3, 3.5],
                "Low": [3.0, 3.1],
                "Close": [3.2, 3.4],
                "Volume": [1000, 2000],
            }
        )
        with patch("pandas.read_csv", return_value=data):
            quote = CSVMarketDataAdapter("sample.csv").get_quote("CSV")

        self.assertEqual(quote.symbol, "CSV")
        self.assertEqual(quote.price, 3.4)
        self.assertIn("2026-06-04", quote.timestamp)

    def test_toss_adapter_parses_quote_snapshot(self):
        price_response = Mock()
        price_response.json.return_value = {
            "result": [
                {
                    "symbol": "LASE",
                    "lastPrice": "3.60",
                    "timestamp": "2026-06-04T09:31:00Z",
                    "currency": "USD",
                }
            ]
        }
        price_response.raise_for_status.return_value = None
        orderbook_response = Mock()
        orderbook_response.json.return_value = {
            "result": {
                "timestamp": "2026-06-04T09:31:01Z",
                "currency": "USD",
                "bids": [{"price": "3.58", "volume": "100"}],
                "asks": [{"price": "3.62", "volume": "120"}],
            }
        }
        orderbook_response.raise_for_status.return_value = None
        with patch("requests.get", side_effect=[price_response, orderbook_response]) as get:
            quote = TossMarketDataAdapter(token="token", account_seq="acct").get_quote("LASE")

        self.assertEqual(quote.symbol, "LASE")
        self.assertEqual(quote.price, 3.60)
        self.assertEqual(quote.bid, 3.58)
        self.assertEqual(quote.ask, 3.62)
        self.assertEqual(quote.currency, "USD")
        self.assertAlmostEqual(quote.spread_pct, 1.1111, places=4)
        self.assertEqual(quote.timestamp, "2026-06-04T09:31:01Z")
        self.assertIn("Authorization", get.call_args.kwargs["headers"])

    def test_toss_adapter_parses_candles(self):
        response = Mock()
        response.json.return_value = {
            "result": {
                "candles": [
                    {
                        "timestamp": "2026-06-04T09:31:00+09:00",
                        "openPrice": "3.50",
                        "highPrice": "3.65",
                        "lowPrice": "3.45",
                        "closePrice": "3.60",
                        "volume": "1000",
                        "currency": "USD",
                    },
                    {
                        "timestamp": "2026-06-04T09:30:00+09:00",
                        "openPrice": "3.40",
                        "highPrice": "3.55",
                        "lowPrice": "3.35",
                        "closePrice": "3.50",
                        "volume": "900",
                        "currency": "USD",
                    },
                ]
            }
        }
        response.raise_for_status.return_value = None
        with patch("requests.get", return_value=response) as get:
            data = TossMarketDataAdapter(token="token", interval="5m").get_ohlcv("LASE")

        self.assertEqual(list(data.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(float(data.iloc[-1]["Close"]), 3.60)
        self.assertEqual(get.call_args.kwargs["params"]["interval"], "1m")

    def test_toss_adapter_can_issue_oauth_token(self):
        token_response = Mock()
        token_response.json.return_value = {"access_token": "issued-token", "token_type": "Bearer", "expires_in": 86400}
        token_response.raise_for_status.return_value = None
        price_response = Mock()
        price_response.json.return_value = {
            "result": [{"symbol": "LASE", "lastPrice": "3.60", "currency": "USD"}]
        }
        price_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=token_response) as post, patch("requests.get", return_value=price_response):
            quote = TossMarketDataAdapter(client_id="client", client_secret="secret").get_price("LASE")

        self.assertEqual(quote.price, 3.60)
        self.assertEqual(post.call_args.kwargs["data"]["grant_type"], "client_credentials")

    def test_toss_adapter_reference_info_endpoints(self):
        responses = [
            {"result": {"upperLimitPrice": None, "lowerLimitPrice": None, "currency": "USD"}},
            {"result": [{"symbol": "AAPL", "name": "애플", "market": "NASDAQ", "currency": "USD"}]},
            {"result": [{"warningType": "VI_STATIC", "exchange": "KRX"}]},
            {"result": {"baseCurrency": "USD", "quoteCurrency": "KRW", "rate": "1380.5"}},
            {"result": {"today": {"date": "2026-03-25", "regularMarket": {"startTime": "2026-03-25T22:30:00+09:00", "endTime": "2026-03-26T05:00:00+09:00"}}}},
        ]
        mocked = []
        for payload in responses:
            item = Mock()
            item.json.return_value = payload
            item.raise_for_status.return_value = None
            mocked.append(item)

        with patch("requests.get", side_effect=mocked) as get:
            adapter = TossMarketDataAdapter(token="token")
            limits = adapter.get_price_limits("AAPL")
            stocks = adapter.get_stocks(["AAPL"])
            warnings = adapter.get_stock_warnings("AAPL")
            exchange_rate = adapter.get_exchange_rate("USD", "KRW")
            calendar = adapter.get_us_market_calendar("2026-03-25")

        self.assertEqual(limits["currency"], "USD")
        self.assertEqual(stocks[0]["symbol"], "AAPL")
        self.assertEqual(warnings[0]["warningType"], "VI_STATIC")
        self.assertEqual(exchange_rate["rate"], "1380.5")
        self.assertIn("regularMarket", calendar["today"])
        self.assertTrue(get.call_args_list[0].args[0].endswith("/api/v1/price-limits"))
        self.assertEqual(get.call_args_list[1].kwargs["params"]["symbols"], "AAPL")
        self.assertTrue(get.call_args_list[2].args[0].endswith("/api/v1/stocks/AAPL/warnings"))
        self.assertEqual(get.call_args_list[4].kwargs["params"]["date"], "2026-03-25")


if __name__ == "__main__":
    unittest.main()
