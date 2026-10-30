from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urljoin

import pandas as pd
import requests
import yfinance as yf

from .toss_auth import get_cached_access_token, invalidate_cached_access_token
from .validators import validate_price_data


@dataclass
class QuoteSnapshot:
    symbol: str
    price: float
    bid: float | None = None
    ask: float | None = None
    timestamp: str | None = None
    currency: str | None = None

    @property
    def spread_pct(self) -> float:
        if self.bid is None or self.ask is None or self.price <= 0:
            return 0.0
        return max(0.0, (self.ask - self.bid) / self.price * 100)


class MarketDataAdapter(Protocol):
    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        ...

    def get_quote(self, symbol: str) -> QuoteSnapshot:
        ...


@dataclass
class CSVMarketDataAdapter:
    path: str

    def get_ohlcv(self, symbol: str = "CSV") -> pd.DataFrame:
        data = pd.read_csv(self.path)
        data.columns = [str(col).strip().title() for col in data.columns]
        date_col = next((col for col in ["Datetime", "Date", "Time", "Timestamp"] if col in data.columns), None)
        if date_col:
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.set_index(date_col)
        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
        validate_price_data(data)
        return data

    def get_quote(self, symbol: str = "CSV") -> QuoteSnapshot:
        data = self.get_ohlcv(symbol)
        latest = data.iloc[-1]
        return QuoteSnapshot(symbol=symbol, price=float(latest["Close"]), timestamp=str(data.index[-1]))


@dataclass
class MockMarketDataAdapter:
    data: pd.DataFrame
    quote: QuoteSnapshot | None = None

    def get_ohlcv(self, symbol: str = "MOCK") -> pd.DataFrame:
        data = self.data.copy()
        validate_price_data(data)
        return data

    def get_quote(self, symbol: str = "MOCK") -> QuoteSnapshot:
        if self.quote is not None:
            return self.quote
        latest = self.data.iloc[-1]
        return QuoteSnapshot(symbol=symbol, price=float(latest["Close"]))


@dataclass
class YFinanceMarketDataAdapter:
    period: str = "1mo"
    interval: str = "5m"
    prepost: bool = True

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        data = yf.download(
            symbol,
            period=self.period,
            interval=self.interval,
            auto_adjust=True,
            progress=False,
            threads=False,
            prepost=self.prepost,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna()
        validate_price_data(data)
        return data

    def get_quote(self, symbol: str) -> QuoteSnapshot:
        data = self.get_ohlcv(symbol)
        latest = data.iloc[-1]
        return QuoteSnapshot(symbol=symbol, price=float(latest["Close"]), timestamp=str(data.index[-1]))


@dataclass
class TossMarketDataAdapter:
    token: str = ""
    client_id: str = ""
    client_secret: str = ""
    account_seq: str | None = None
    base_url: str = "https://openapi.tossinvest.com"
    timeout_seconds: int = 10
    interval: str = "1m"
    count: int = 200
    adjusted: bool = True

    def _base_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        if base.endswith("/api"):
            base = base[:-4]
        return base + "/"

    def _url(self, path: str) -> str:
        return urljoin(self._base_url(), path.lstrip("/"))

    def _access_token(self) -> str:
        if self.token:
            return self.token
        if not self.client_id or not self.client_secret:
            raise ValueError("Toss API token or client_id/client_secret is required")
        token = get_cached_access_token(
            base_url=self.base_url,
            client_id=self.client_id,
            client_secret=self.client_secret,
            timeout_seconds=self.timeout_seconds,
        )
        self.token = str(token)
        return self.token

    def _headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._access_token()}"}
        if self.account_seq:
            headers["X-Tossinvest-Account"] = self.account_seq
        return headers

    def _get(self, path: str, params: dict) -> dict | list:
        response = requests.get(self._url(path), headers=self._headers(), params=params, timeout=self.timeout_seconds)
        if response.status_code == 401 and self.client_id and self.client_secret:
            invalidate_cached_access_token(base_url=self.base_url, client_id=self.client_id)
            self.token = ""
            response = requests.get(self._url(path), headers=self._headers(), params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return payload.get("result", payload) if isinstance(payload, dict) else payload

    def _chart_interval(self) -> str:
        return "1d" if self.interval == "1d" else "1m"

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Return Toss OpenAPI candles as the app's standard OHLCV frame."""
        payload = self._get(
            "/api/v1/candles",
            {
                "symbol": symbol,
                "interval": self._chart_interval(),
                "count": max(1, min(int(self.count), 200)),
                "adjusted": str(bool(self.adjusted)).lower(),
            },
        )
        rows = payload.get("candles", []) if isinstance(payload, dict) else payload
        data = pd.DataFrame(rows)
        if data.empty:
            raise ValueError(f"Toss candles response is empty for {symbol}")

        rename_map = {
            "open": "Open",
            "openPrice": "Open",
            "high": "High",
            "highPrice": "High",
            "low": "Low",
            "lowPrice": "Low",
            "close": "Close",
            "closePrice": "Close",
            "volume": "Volume",
            "timestamp": "Datetime",
            "time": "Datetime",
        }
        data = data.rename(columns={key: value for key, value in rename_map.items() if key in data.columns})
        if "Datetime" in data.columns:
            data["Datetime"] = pd.to_datetime(data["Datetime"])
            data = data.set_index("Datetime")
        data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna().sort_index()
        validate_price_data(data)
        return data

    def get_price(self, symbol: str) -> QuoteSnapshot:
        payload = self._get("/api/v1/prices", {"symbols": symbol})
        rows = payload if isinstance(payload, list) else [payload]
        if not rows:
            raise ValueError(f"Toss prices response is empty for {symbol}")
        row = rows[0]
        return QuoteSnapshot(
            symbol=str(row.get("symbol", symbol)),
            price=float(row.get("lastPrice", row.get("price", 0))),
            timestamp=row.get("timestamp"),
            currency=row.get("currency"),
        )

    def get_orderbook(self, symbol: str) -> dict:
        payload = self._get("/api/v1/orderbook", {"symbol": symbol})
        return payload if isinstance(payload, dict) else {}

    def get_trades(self, symbol: str, count: int = 50) -> list[dict]:
        payload = self._get("/api/v1/trades", {"symbol": symbol, "count": max(1, min(int(count), 50))})
        return payload if isinstance(payload, list) else []

    def get_price_limits(self, symbol: str) -> dict:
        payload = self._get("/api/v1/price-limits", {"symbol": symbol})
        return payload if isinstance(payload, dict) else {}

    def get_stocks(self, symbols: str | list[str]) -> list[dict]:
        symbol_text = ",".join(symbols) if isinstance(symbols, list) else symbols
        payload = self._get("/api/v1/stocks", {"symbols": symbol_text})
        return payload if isinstance(payload, list) else []

    def get_stock_warnings(self, symbol: str) -> list[dict]:
        payload = self._get(f"/api/v1/stocks/{symbol}/warnings", {})
        return payload if isinstance(payload, list) else []

    def get_exchange_rate(
        self,
        base_currency: str = "USD",
        quote_currency: str = "KRW",
        date_time: str | None = None,
    ) -> dict:
        params = {"baseCurrency": base_currency, "quoteCurrency": quote_currency}
        if date_time:
            params["dateTime"] = date_time
        payload = self._get("/api/v1/exchange-rate", params)
        return payload if isinstance(payload, dict) else {}

    def get_market_calendar(self, country: str = "US", date: str | None = None) -> dict:
        normalized = country.upper()
        if normalized not in {"KR", "US"}:
            raise ValueError("country must be KR or US")
        params = {"date": date} if date else {}
        payload = self._get(f"/api/v1/market-calendar/{normalized}", params)
        return payload if isinstance(payload, dict) else {}

    def get_us_market_calendar(self, date: str | None = None) -> dict:
        return self.get_market_calendar("US", date=date)

    def get_kr_market_calendar(self, date: str | None = None) -> dict:
        return self.get_market_calendar("KR", date=date)

    def get_quote(self, symbol: str) -> QuoteSnapshot:
        price = self.get_price(symbol)
        orderbook = self.get_orderbook(symbol)
        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []
        best_bid = float(bids[0]["price"]) if bids else None
        best_ask = float(asks[0]["price"]) if asks else None
        return QuoteSnapshot(
            symbol=price.symbol or symbol,
            price=price.price,
            bid=best_bid,
            ask=best_ask,
            timestamp=orderbook.get("timestamp") or price.timestamp,
            currency=orderbook.get("currency") or price.currency,
        )
