from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import requests

from .toss_auth import get_cached_access_token, invalidate_cached_access_token


DEFAULT_MARKET_SYMBOLS = [
    "AAPL",
    "NVDA",
    "TSLA",
    "MSFT",
    "AMZN",
    "META",
    "GOOGL",
    "AMD",
    "AVGO",
    "SMCI",
    "COIN",
    "MSTR",
    "SOFI",
    "PLTR",
    "HOOD",
    "RIVN",
    "LCID",
    "NIO",
    "BABA",
    "PDD",
    "JD",
    "UBER",
    "SHOP",
    "SNOW",
    "DDOG",
    "NET",
    "CRWD",
    "PANW",
    "ROKU",
    "XYZ",
    "PYPL",
    "AFRM",
    "DKNG",
    "U",
    "RBLX",
    "SNAP",
    "PINS",
    "DIS",
    "NKE",
    "SBUX",
    "WMT",
    "COST",
    "JPM",
    "BAC",
    "V",
    "MA",
    "XOM",
    "CVX",
    "LLY",
    "MRNA",
    "BA",
    "CAT",
    "F",
    "GM",
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "SMH",
    "SOXL",
    "TQQQ",
    "SQQQ",
    "MARA",
    "RIOT",
]


LEGACY_SYMBOL_ALIASES = {
    "SQ": "XYZ",
}


TOSS_READ_ENDPOINTS = [
    "GET /api/v1/accounts",
    "GET /api/v1/buying-power",
    "GET /api/v1/candles",
    "GET /api/v1/commissions",
    "GET /api/v1/exchange-rate",
    "GET /api/v1/holdings",
    "GET /api/v1/market-calendar/KR",
    "GET /api/v1/market-calendar/US",
    "GET /api/v1/orderbook",
    "GET /api/v1/orders",
    "GET /api/v1/orders/{orderId}",
    "GET /api/v1/price-limits",
    "GET /api/v1/prices",
    "GET /api/v1/sellable-quantity",
    "GET /api/v1/stocks",
    "GET /api/v1/stocks/{symbol}/warnings",
    "GET /api/v1/trades",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def normalize_symbol(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text.replace(".", "").replace("-", "").isalnum():
        return ""
    return LEGACY_SYMBOL_ALIASES.get(text, text)


def unique_symbols(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    symbols: list[str] = []
    for value in values:
        symbol = normalize_symbol(value)
        if symbol and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    return symbols


def parse_symbol_text(value: str | None) -> list[str]:
    if not value:
        return []
    return unique_symbols(part for part in value.replace("\n", ",").split(","))


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def clean_secret(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    placeholder_markers = [
        "your_",
        "example",
        "placeholder",
        "여기에",
        "발급",
        "실제_",
        "api_key_here",
        "secret_key_here",
        "account_seq_here",
    ]
    if any(marker in lowered for marker in placeholder_markers):
        return ""
    return text


def looks_like_toss_client_key(value: str) -> bool:
    return clean_secret(value).lower().startswith(("tsck", "test_ck", "live_ck"))


def normalize_toss_auth(token: str, client_id: str, client_secret: str) -> tuple[str, str, str]:
    token = clean_secret(token)
    client_id = clean_secret(client_id)
    client_secret = clean_secret(client_secret)
    if not client_id and client_secret and looks_like_toss_client_key(token):
        client_id = token
        token = ""
    if client_id and client_secret:
        return "", client_id, client_secret
    return token, client_id, client_secret


def secret_value(name: str, *aliases: str) -> str:
    for key in (name, *aliases):
        value = clean_secret(os.getenv(key))
        if value:
            return value.strip()
    return ""


def load_symbols_from_local_state(app_dir: Path, *, include_default_universe: bool = True) -> list[str]:
    symbols: list[str] = []
    settings = read_json(app_dir / "app_settings.json")
    if isinstance(settings, dict):
        symbols.extend(settings.get("watchlist") or [])
        symbols.extend(parse_symbol_text(str(settings.get("candidate_universe") or "")))
        symbols.extend(parse_symbol_text(str(settings.get("wide_candidate_universe") or "")))

    latest_signal = read_json(app_dir / "latest_signal.json")
    if isinstance(latest_signal, dict):
        symbols.append(latest_signal.get("symbol"))
        for row in latest_signal.get("position_summary") or []:
            if isinstance(row, dict):
                symbols.append(row.get("symbol"))
        auto_selection = latest_signal.get("auto_selection")
        if isinstance(auto_selection, dict):
            symbols.append(auto_selection.get("symbol"))
            for row in auto_selection.get("top_surge_candidates") or []:
                if isinstance(row, dict):
                    symbols.append(row.get("symbol"))

    realtime_positions = read_json(app_dir / "latest_realtime_positions.json")
    if isinstance(realtime_positions, dict):
        for row in realtime_positions.get("positions") or []:
            if isinstance(row, dict):
                symbols.append(row.get("symbol"))

    wide_state = read_json(app_dir / "wide_candidate_state.json")
    if isinstance(wide_state, dict):
        active = wide_state.get("active")
        if isinstance(active, dict):
            symbols.extend(active.keys())
            for row in active.values():
                if isinstance(row, dict):
                    symbols.append(row.get("symbol"))

    if include_default_universe:
        symbols.extend(DEFAULT_MARKET_SYMBOLS)
    return unique_symbols(symbols)


def result_payload(payload: Any) -> Any:
    return payload.get("result", payload) if isinstance(payload, dict) else payload


def rows_from_payload(payload: Any, *candidate_keys: str) -> list[dict[str, Any]]:
    payload = result_payload(payload)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def extract_symbols_from_holdings(payload: Any) -> list[str]:
    rows = rows_from_payload(payload, "holdings", "items", "stocks", "positions")
    symbols = []
    for row in rows:
        symbols.append(row.get("symbol") or row.get("ticker") or row.get("stockCode"))
    return unique_symbols(symbols)


def extract_orders(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    result = result_payload(payload)
    if isinstance(result, dict):
        next_cursor = result.get("nextCursor") or result.get("next_cursor")
        return rows_from_payload(result, "orders"), str(next_cursor) if next_cursor else None
    if isinstance(result, list):
        return [row for row in result if isinstance(row, dict)], None
    return [], None


def extract_candles(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    result = result_payload(payload)
    if isinstance(result, dict):
        next_before = result.get("nextBefore") or result.get("next_before")
        return rows_from_payload(result, "candles"), str(next_before) if next_before else None
    if isinstance(result, list):
        return [row for row in result if isinstance(row, dict)], None
    return [], None


@dataclass
class TossApiClient:
    token: str = ""
    client_id: str = ""
    client_secret: str = ""
    account_seq: str | None = None
    base_url: str = "https://openapi.tossinvest.com"
    timeout_seconds: int = 15
    request_pause_seconds: float = 0.2
    max_retries: int = 3

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
            raise ValueError("TOSS_API_TOKEN or TOSS_CLIENT_ID/TOSS_CLIENT_SECRET is required")
        self.token = get_cached_access_token(
            base_url=self.base_url,
            client_id=self.client_id,
            client_secret=self.client_secret,
            timeout_seconds=self.timeout_seconds,
        )
        return self.token

    def headers(self, account_seq: str | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._access_token()}"}
        selected_account = account_seq or self.account_seq
        if selected_account:
            headers["X-Tossinvest-Account"] = str(selected_account)
        return headers

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        account_seq: str | None = None,
    ) -> Any:
        params = {key: value for key, value in (params or {}).items() if value not in {None, ""}}
        last_error: Exception | None = None
        for attempt in range(int(self.max_retries) + 1):
            if self.request_pause_seconds > 0:
                time.sleep(float(self.request_pause_seconds))
            response = requests.get(
                self._url(path),
                headers=self.headers(account_seq),
                params=params,
                timeout=self.timeout_seconds,
            )
            if response.status_code == 401 and self.client_id and self.client_secret:
                invalidate_cached_access_token(base_url=self.base_url, client_id=self.client_id)
                self.token = ""
                response = requests.get(
                    self._url(path),
                    headers=self.headers(account_seq),
                    params=params,
                    timeout=self.timeout_seconds,
                )
            if response.status_code == 429 and attempt < int(self.max_retries):
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(8.0, 1.5 * (attempt + 1))
                time.sleep(delay)
                continue
            try:
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001 - caller records endpoint errors
                last_error = exc
                if attempt >= int(self.max_retries):
                    break
                time.sleep(min(5.0, 1.0 + attempt))
        raise last_error or RuntimeError(f"Toss API request failed: {path}")


@dataclass
class CollectionConfig:
    app_dir: Path
    output_dir: Path
    symbols: list[str]
    include_default_universe: bool = True
    max_symbols: int = 80
    candle_intervals: list[str] = field(default_factory=lambda: ["1m", "1d"])
    candle_count: int = 200
    max_candle_pages: int = 2
    trades_count: int = 50
    order_statuses: list[str] = field(default_factory=lambda: ["OPEN", "CLOSED"])
    max_order_pages: int = 10
    max_order_details: int = 200
    order_from: str | None = None
    order_to: str | None = None
    calendar_days: int = 7
    request_pause_seconds: float = 0.2
    timeout_seconds: int = 15
    max_retries: int = 3
    save_official_docs: bool = True


class TossDataCollector:
    def __init__(self, client: TossApiClient, config: CollectionConfig) -> None:
        self.client = client
        self.config = config
        self.generated_at = utc_now().isoformat()
        self.errors_path = self.config.output_dir / "errors.jsonl"
        self.summary: dict[str, Any] = {
            "generated_at": self.generated_at,
            "base_url": self.client.base_url,
            "read_endpoints": TOSS_READ_ENDPOINTS,
            "files": {},
            "counts": {},
            "errors": 0,
            "notes": [
                "Read-only collection: order create, modify, and cancel endpoints are not called.",
                "Secrets are not written to the export.",
            ],
        }

    def collect(self) -> dict[str, Any]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_official_docs:
            self.collect_official_docs()
        accounts = self.collect_account_scope()
        holdings_symbols = []
        for account in accounts:
            holdings_symbols.extend(account.get("holding_symbols") or [])
        symbols = unique_symbols(self.config.symbols + holdings_symbols)
        if self.config.max_symbols > 0:
            symbols = symbols[: int(self.config.max_symbols)]
        self.summary["symbols"] = symbols
        self.summary["counts"]["symbols"] = len(symbols)
        self.collect_market_scope(symbols)
        write_json(self.config.output_dir / "manifest.json", self.summary)
        return self.summary

    def record_file(self, key: str, path: Path) -> None:
        self.summary.setdefault("files", {})[key] = str(path)

    def record_error(self, endpoint: str, error: Exception, context: dict[str, Any] | None = None) -> None:
        self.summary["errors"] = int(self.summary.get("errors", 0)) + 1
        append_jsonl(
            self.errors_path,
            {
                "logged_at": utc_now().isoformat(),
                "endpoint": endpoint,
                "context": context or {},
                "message": str(error),
            },
        )

    def safe_get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        account_seq: str | None = None,
        label: str | None = None,
    ) -> Any | None:
        endpoint = label or path
        try:
            return self.client.get(path, params or {}, account_seq=account_seq)
        except Exception as exc:  # noqa: BLE001 - collector should keep going
            self.record_error(endpoint, exc, {"params": params, "account_seq": account_seq})
            return None

    def collect_official_docs(self) -> None:
        official_dir = self.config.output_dir / "official"
        endpoints_path = official_dir / "read_endpoints.json"
        write_json(endpoints_path, {"generated_at": self.generated_at, "read_endpoints": TOSS_READ_ENDPOINTS})
        self.record_file("official_read_endpoints", endpoints_path)
        docs = {
            "openapi": ("https://openapi.tossinvest.com/openapi-docs/latest/openapi.json", official_dir / "openapi.json"),
            "overview": ("https://openapi.tossinvest.com/openapi-docs/overview.md", official_dir / "overview.md"),
        }
        for key, (url, path) in docs.items():
            try:
                response = requests.get(url, timeout=self.config.timeout_seconds)
                response.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
                self.record_file(f"official_{key}", path)
            except Exception as exc:  # noqa: BLE001
                self.record_error(url, exc)

    def collect_account_scope(self) -> list[dict[str, Any]]:
        account_dir = self.config.output_dir / "account"
        flat_orders: list[dict[str, Any]] = []
        flat_holdings: list[dict[str, Any]] = []
        account_summaries: list[dict[str, Any]] = []

        accounts_payload = self.safe_get("/api/v1/accounts", label="accounts")
        if accounts_payload is None:
            return []
        accounts_path = account_dir / "accounts.json"
        write_json(accounts_path, accounts_payload)
        self.record_file("accounts", accounts_path)
        accounts = rows_from_payload(accounts_payload, "accounts")
        if not accounts and isinstance(result_payload(accounts_payload), list):
            accounts = rows_from_payload(accounts_payload)
        if not accounts and isinstance(result_payload(accounts_payload), dict):
            account = result_payload(accounts_payload)
            accounts = [account] if account else []
        self.summary["counts"]["accounts"] = len(accounts)

        for account in accounts:
            account_seq = str(account.get("accountSeq") or account.get("accountId") or self.client.account_seq or "")
            if not account_seq:
                continue
            account_safe = account_seq.replace("/", "_")
            target_dir = account_dir / account_safe
            holding_symbols: list[str] = []

            for currency in ("USD", "KRW"):
                payload = self.safe_get(
                    "/api/v1/buying-power",
                    {"currency": currency},
                    account_seq=account_seq,
                    label=f"buying_power_{currency}",
                )
                if payload is not None:
                    path = target_dir / f"buying_power_{currency}.json"
                    write_json(path, payload)
                    self.record_file(f"buying_power_{account_safe}_{currency}", path)

            holdings_payload = self.safe_get("/api/v1/holdings", account_seq=account_seq, label="holdings")
            if holdings_payload is not None:
                path = target_dir / "holdings.json"
                write_json(path, holdings_payload)
                self.record_file(f"holdings_{account_safe}", path)
                holding_symbols = extract_symbols_from_holdings(holdings_payload)
                for row in rows_from_payload(holdings_payload, "holdings", "items", "stocks", "positions"):
                    flat_holdings.append({"accountSeq": account_seq, **row})

            commissions_payload = self.safe_get("/api/v1/commissions", account_seq=account_seq, label="commissions")
            if commissions_payload is not None:
                path = target_dir / "commissions.json"
                write_json(path, commissions_payload)
                self.record_file(f"commissions_{account_safe}", path)

            sellable_rows = []
            for symbol in holding_symbols:
                payload = self.safe_get(
                    "/api/v1/sellable-quantity",
                    {"symbol": symbol},
                    account_seq=account_seq,
                    label="sellable_quantity",
                )
                if payload is not None:
                    sellable_rows.append({"symbol": symbol, "payload": result_payload(payload)})
            if sellable_rows:
                path = target_dir / "sellable_quantities.json"
                write_json(path, sellable_rows)
                self.record_file(f"sellable_quantities_{account_safe}", path)

            for status in self.config.order_statuses:
                orders = self.collect_orders_for_account(account_seq, status)
                flat_orders.extend({"accountSeq": account_seq, **row} for row in orders)

            account_summaries.append(
                {
                    "accountSeq": account_seq,
                    "accountType": account.get("accountType"),
                    "holding_symbols": holding_symbols,
                }
            )

        flat_dir = self.config.output_dir / "flat"
        write_csv(flat_dir / "holdings.csv", flat_holdings)
        write_csv(flat_dir / "orders.csv", flat_orders)
        self.record_file("flat_holdings", flat_dir / "holdings.csv")
        self.record_file("flat_orders", flat_dir / "orders.csv")
        self.summary["counts"]["holding_rows"] = len(flat_holdings)
        self.summary["counts"]["order_rows"] = len(flat_orders)
        return account_summaries

    def collect_orders_for_account(self, account_seq: str, status: str) -> list[dict[str, Any]]:
        status = status.upper()
        account_safe = account_seq.replace("/", "_")
        target_dir = self.config.output_dir / "account" / account_safe / "orders"
        all_orders: list[dict[str, Any]] = []
        cursor: str | None = None
        for page in range(max(1, int(self.config.max_order_pages))):
            params: dict[str, Any] = {
                "status": status,
                "limit": 100,
                "from": self.config.order_from,
                "to": self.config.order_to,
                "cursor": cursor,
            }
            payload = self.safe_get(
                "/api/v1/orders",
                params,
                account_seq=account_seq,
                label=f"orders_{status}",
            )
            if payload is None:
                break
            path = target_dir / f"{status.lower()}_page_{page + 1}.json"
            write_json(path, payload)
            self.record_file(f"orders_{account_safe}_{status.lower()}_page_{page + 1}", path)
            orders, next_cursor = extract_orders(payload)
            all_orders.extend(orders)
            if status == "OPEN" or not next_cursor:
                break
            cursor = next_cursor

        detail_count = 0
        for order in all_orders:
            if detail_count >= int(self.config.max_order_details):
                break
            order_id = str(order.get("orderId") or "")
            if not order_id:
                continue
            payload = self.safe_get(
                f"/api/v1/orders/{order_id}",
                account_seq=account_seq,
                label="order_detail",
            )
            if payload is None:
                continue
            detail_path = target_dir / "details" / f"{order_id}.json"
            write_json(detail_path, payload)
            detail_count += 1
        return all_orders

    def collect_market_scope(self, symbols: list[str]) -> None:
        market_dir = self.config.output_dir / "market"
        flat_dir = self.config.output_dir / "flat"
        quote_rows: list[dict[str, Any]] = []
        stock_rows: list[dict[str, Any]] = []
        candle_rows: list[dict[str, Any]] = []
        trade_rows: list[dict[str, Any]] = []
        orderbook_rows: list[dict[str, Any]] = []
        price_limit_rows: list[dict[str, Any]] = []
        warning_rows: list[dict[str, Any]] = []

        self.collect_market_info(market_dir)

        for batch_index, batch in enumerate(chunked(symbols, 200), start=1):
            payload = self.safe_get("/api/v1/prices", {"symbols": ",".join(batch)}, label="prices")
            if payload is not None:
                path = market_dir / f"prices_batch_{batch_index}.json"
                write_json(path, payload)
                self.record_file(f"prices_batch_{batch_index}", path)
                quote_rows.extend(rows_from_payload(payload, "prices"))

            payload = self.safe_get("/api/v1/stocks", {"symbols": ",".join(batch)}, label="stocks")
            if payload is not None:
                path = market_dir / f"stocks_batch_{batch_index}.json"
                write_json(path, payload)
                self.record_file(f"stocks_batch_{batch_index}", path)
                stock_rows.extend(rows_from_payload(payload, "stocks"))

        for symbol in symbols:
            symbol_dir = self.config.output_dir / "symbols" / symbol
            orderbook = self.safe_get("/api/v1/orderbook", {"symbol": symbol}, label="orderbook")
            if orderbook is not None:
                path = symbol_dir / "orderbook.json"
                write_json(path, orderbook)
                rows = rows_from_payload(orderbook, "orderbook")
                orderbook_rows.extend(flatten_orderbook(symbol, result_payload(orderbook) if not rows else rows[0]))

            trades = self.safe_get(
                "/api/v1/trades",
                {"symbol": symbol, "count": max(1, min(int(self.config.trades_count), 50))},
                label="trades",
            )
            if trades is not None:
                path = symbol_dir / "trades.json"
                write_json(path, trades)
                for row in rows_from_payload(trades, "trades"):
                    trade_rows.append({"symbol": symbol, **row})

            limits = self.safe_get("/api/v1/price-limits", {"symbol": symbol}, label="price_limits")
            if limits is not None:
                path = symbol_dir / "price_limits.json"
                write_json(path, limits)
                result = result_payload(limits)
                if isinstance(result, dict):
                    price_limit_rows.append({"symbol": symbol, **result})

            warnings = self.safe_get(f"/api/v1/stocks/{symbol}/warnings", label="stock_warnings")
            if warnings is not None:
                path = symbol_dir / "warnings.json"
                write_json(path, warnings)
                for row in rows_from_payload(warnings, "warnings"):
                    warning_rows.append({"symbol": symbol, **row})

            for interval in self.config.candle_intervals:
                candles = self.collect_candles(symbol, interval)
                for row in candles:
                    candle_rows.append({"symbol": symbol, "interval": interval, **row})

        write_csv(flat_dir / "quotes.csv", quote_rows)
        write_csv(flat_dir / "stocks.csv", stock_rows)
        write_csv(flat_dir / "orderbook_top.csv", orderbook_rows)
        write_csv(flat_dir / "trades.csv", trade_rows)
        write_csv(flat_dir / "price_limits.csv", price_limit_rows)
        write_csv(flat_dir / "warnings.csv", warning_rows)
        write_csv(flat_dir / "candles.csv", candle_rows)
        self.record_file("flat_quotes", flat_dir / "quotes.csv")
        self.record_file("flat_stocks", flat_dir / "stocks.csv")
        self.record_file("flat_orderbook_top", flat_dir / "orderbook_top.csv")
        self.record_file("flat_trades", flat_dir / "trades.csv")
        self.record_file("flat_price_limits", flat_dir / "price_limits.csv")
        self.record_file("flat_warnings", flat_dir / "warnings.csv")
        self.record_file("flat_candles", flat_dir / "candles.csv")
        self.summary["counts"]["quote_rows"] = len(quote_rows)
        self.summary["counts"]["stock_rows"] = len(stock_rows)
        self.summary["counts"]["orderbook_rows"] = len(orderbook_rows)
        self.summary["counts"]["trade_rows"] = len(trade_rows)
        self.summary["counts"]["price_limit_rows"] = len(price_limit_rows)
        self.summary["counts"]["warning_rows"] = len(warning_rows)
        self.summary["counts"]["candle_rows"] = len(candle_rows)

    def collect_market_info(self, market_dir: Path) -> None:
        for base, quote in (("USD", "KRW"), ("KRW", "USD")):
            payload = self.safe_get(
                "/api/v1/exchange-rate",
                {"baseCurrency": base, "quoteCurrency": quote},
                label="exchange_rate",
            )
            if payload is not None:
                path = market_dir / f"exchange_rate_{base}_{quote}.json"
                write_json(path, payload)
                self.record_file(f"exchange_rate_{base}_{quote}", path)

        start_day = date.today()
        for offset in range(max(1, int(self.config.calendar_days))):
            day = (start_day + timedelta(days=offset)).isoformat()
            for country in ("KR", "US"):
                payload = self.safe_get(f"/api/v1/market-calendar/{country}", {"date": day}, label=f"calendar_{country}")
                if payload is not None:
                    path = market_dir / "calendar" / f"{country}_{day}.json"
                    write_json(path, payload)

    def collect_candles(self, symbol: str, interval: str) -> list[dict[str, Any]]:
        all_candles: list[dict[str, Any]] = []
        before: str | None = None
        symbol_dir = self.config.output_dir / "symbols" / symbol
        for page in range(max(1, int(self.config.max_candle_pages))):
            params: dict[str, Any] = {
                "symbol": symbol,
                "interval": interval,
                "count": max(1, min(int(self.config.candle_count), 200)),
                "adjusted": "true",
                "before": before,
            }
            payload = self.safe_get("/api/v1/candles", params, label=f"candles_{interval}")
            if payload is None:
                break
            path = symbol_dir / f"candles_{interval}_page_{page + 1}.json"
            write_json(path, payload)
            candles, next_before = extract_candles(payload)
            all_candles.extend(candles)
            if not next_before:
                break
            before = next_before
        if all_candles:
            write_csv(symbol_dir / f"candles_{interval}.csv", all_candles)
        return all_candles


def flatten_orderbook(symbol: str, payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    bids = payload.get("bids") if isinstance(payload.get("bids"), list) else []
    asks = payload.get("asks") if isinstance(payload.get("asks"), list) else []
    rows = []
    for side, levels in (("bid", bids), ("ask", asks)):
        for index, level in enumerate(levels[:10], start=1):
            if isinstance(level, dict):
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": payload.get("timestamp"),
                        "currency": payload.get("currency"),
                        "side": side,
                        "level": index,
                        **level,
                    }
                )
    return rows


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def create_client_from_env(
    *,
    base_url: str | None = None,
    request_pause_seconds: float = 0.2,
    timeout_seconds: int = 15,
    max_retries: int = 3,
) -> TossApiClient:
    token = secret_value("TOSS_API_TOKEN", "TOSS_ACCESS_TOKEN", "TOSS_BEARER_TOKEN")
    client_id = secret_value("TOSS_CLIENT_ID", "TOSS_API_KEY", "TOSS_APP_KEY")
    client_secret = secret_value("TOSS_CLIENT_SECRET", "TOSS_SECRET_KEY", "TOSS_APP_SECRET")
    token, client_id, client_secret = normalize_toss_auth(token, client_id, client_secret)
    return TossApiClient(
        token=token,
        client_id=client_id,
        client_secret=client_secret,
        account_seq=secret_value("TOSS_ACCOUNT_SEQ", "TOSS_ACCOUNT_ID") or None,
        base_url=base_url or secret_value("TOSS_BASE_URL") or "https://openapi.tossinvest.com",
        timeout_seconds=timeout_seconds,
        request_pause_seconds=request_pause_seconds,
        max_retries=max_retries,
    )
