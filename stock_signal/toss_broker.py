from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urljoin

import requests

from .execution import AccountSnapshot, BrokerExecutionAdapter, ExecutionResult, OrderRequest
from .toss_auth import get_cached_access_token, invalidate_cached_access_token


Currency = Literal["KRW", "USD"]
_ACCOUNT_SEQ_CACHE: dict[tuple[str, str], str] = {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_decimal(value: float | None) -> str | None:
    if value is None:
        return None
    if value >= 1:
        text = f"{value:.2f}"
    else:
        text = f"{value:.4f}"
    return text.rstrip("0").rstrip(".")


def _http_error_message(response: requests.Response) -> str:
    body = ""
    try:
        body = response.text[:500]
    except Exception:
        body = ""
    suffix = f" body={body}" if body else ""
    return f"{response.status_code} {response.reason} for url: {response.url}{suffix}"


@dataclass
class TossBrokerExecutionAdapter(BrokerExecutionAdapter):
    token: str = ""
    client_id: str = ""
    client_secret: str = ""
    account_seq: str | int | None = None
    base_url: str = "https://openapi.tossinvest.com"
    timeout_seconds: int = 10
    currency: Currency = "USD"
    allow_live_orders: bool = False

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

    def _headers(self, *, account_required: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._access_token()}",
            "Content-Type": "application/json",
        }
        if account_required:
            account_seq = self._resolve_account_seq()
            headers["X-Tossinvest-Account"] = str(account_seq)
        elif self.account_seq is not None:
            headers["X-Tossinvest-Account"] = str(self.account_seq)
        return headers

    def _result(self, payload: dict | list) -> Any:
        return payload.get("result", payload) if isinstance(payload, dict) else payload

    def _get(self, path: str, params: dict[str, Any] | None = None, *, account_required: bool = False) -> Any:
        response = requests.get(
            self._url(path),
            headers=self._headers(account_required=account_required),
            params=params or {},
            timeout=self.timeout_seconds,
        )
        if response.status_code == 401 and self.client_id and self.client_secret:
            invalidate_cached_access_token(base_url=self.base_url, client_id=self.client_id)
            self.token = ""
            response = requests.get(
                self._url(path),
                headers=self._headers(account_required=account_required),
                params=params or {},
                timeout=self.timeout_seconds,
            )
        if response.status_code >= 400:
            raise requests.HTTPError(_http_error_message(response), response=response)
        return self._result(response.json())

    def _post(self, path: str, body: dict[str, Any] | None = None, *, account_required: bool = False) -> Any:
        response = requests.post(
            self._url(path),
            headers=self._headers(account_required=account_required),
            json=body or {},
            timeout=self.timeout_seconds,
        )
        if response.status_code == 401 and self.client_id and self.client_secret:
            invalidate_cached_access_token(base_url=self.base_url, client_id=self.client_id)
            self.token = ""
            response = requests.post(
                self._url(path),
                headers=self._headers(account_required=account_required),
                json=body or {},
                timeout=self.timeout_seconds,
            )
        if response.status_code >= 400:
            raise requests.HTTPError(_http_error_message(response), response=response)
        return self._result(response.json())

    def _resolve_account_seq(self) -> str:
        if self.account_seq not in {None, ""}:
            return str(self.account_seq)
        cache_key = (self._base_url(), self.client_id or self.token)
        cached = _ACCOUNT_SEQ_CACHE.get(cache_key)
        if cached:
            self.account_seq = cached
            return cached
        accounts = self.get_accounts()
        brokerage = next((account for account in accounts if account.get("accountType") == "BROKERAGE"), None)
        selected = brokerage or (accounts[0] if accounts else None)
        if not selected:
            raise ValueError("Toss account list is empty")
        self.account_seq = str(selected["accountSeq"])
        _ACCOUNT_SEQ_CACHE[cache_key] = str(self.account_seq)
        return str(self.account_seq)

    def get_accounts(self) -> list[dict[str, Any]]:
        result = self._get("/api/v1/accounts")
        return result if isinstance(result, list) else []

    def get_buying_power(self, currency: Currency | None = None) -> dict[str, Any]:
        selected_currency = currency or self.currency
        result = self._get(
            "/api/v1/buying-power",
            {"currency": selected_currency},
            account_required=True,
        )
        return result if isinstance(result, dict) else {}

    def get_holdings(self, symbol: str | None = None) -> dict[str, Any]:
        params = {"symbol": symbol} if symbol else {}
        result = self._get("/api/v1/holdings", params, account_required=True)
        return result if isinstance(result, dict) else {}

    def get_sellable_quantity(self, symbol: str) -> float:
        result = self._get("/api/v1/sellable-quantity", {"symbol": symbol}, account_required=True)
        return _as_float(result.get("sellableQuantity") if isinstance(result, dict) else 0)

    def get_commissions(self) -> list[dict[str, Any]]:
        result = self._get("/api/v1/commissions", account_required=True)
        return result if isinstance(result, list) else []

    def get_orders(
        self,
        *,
        status: Literal["OPEN", "CLOSED"] = "OPEN",
        symbol: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"status": status, "limit": max(1, min(int(limit), 100))}
        if symbol:
            params["symbol"] = symbol
        result = self._get("/api/v1/orders", params, account_required=True)
        return result if isinstance(result, dict) else {}

    def get_order(self, order_id: str) -> dict[str, Any]:
        result = self._get(f"/api/v1/orders/{order_id}", account_required=True)
        return result if isinstance(result, dict) else {}

    def get_account(self) -> AccountSnapshot:
        account_seq = self._resolve_account_seq()
        buying_power = self.get_buying_power(self.currency)
        cash = _as_float(buying_power.get("cashBuyingPower"))
        return AccountSnapshot(
            account_id=str(account_seq),
            currency=str(buying_power.get("currency") or self.currency),
            cash_buying_power=cash,
            equity=cash,
        )

    def order_payload(self, order: OrderRequest, *, confirm_high_value_order: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "clientOrderId": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "orderType": order.order_type,
            "timeInForce": order.time_in_force,
            "quantity": str(int(order.quantity)),
            "confirmHighValueOrder": bool(confirm_high_value_order),
        }
        if order.order_type == "LIMIT":
            price = _format_decimal(order.limit_price)
            if price is None:
                raise ValueError("LIMIT order requires limit_price")
            payload["price"] = price
        return payload

    def submit_order(self, order: OrderRequest, *, confirm_high_value_order: bool = False) -> ExecutionResult:
        if not self.allow_live_orders:
            return ExecutionResult(
                status="REJECTED",
                broker_order_id=None,
                client_order_id=order.client_order_id,
                message="Toss live order adapter is connected but live order sending is disabled",
                raw={
                    "live_order_disabled": True,
                    "request": self.order_payload(order, confirm_high_value_order=confirm_high_value_order),
                },
            )

        result = self._post(
            "/api/v1/orders",
            self.order_payload(order, confirm_high_value_order=confirm_high_value_order),
            account_required=True,
        )
        broker_order_id = str(result.get("orderId", "")) if isinstance(result, dict) else ""
        return ExecutionResult(
            status="SUBMITTED",
            broker_order_id=broker_order_id or None,
            client_order_id=str(result.get("clientOrderId") or order.client_order_id) if isinstance(result, dict) else order.client_order_id,
            message="Toss order submitted",
            raw=result if isinstance(result, dict) else {"result": result},
        )

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        result = self._post(f"/api/v1/orders/{order_id}/cancel", {}, account_required=True)
        return result if isinstance(result, dict) else {}

    def modify_order(
        self,
        order_id: str,
        *,
        order_type: Literal["LIMIT", "MARKET"] = "LIMIT",
        quantity: int | None = None,
        price: float | None = None,
        confirm_high_value_order: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "orderType": order_type,
            "confirmHighValueOrder": bool(confirm_high_value_order),
        }
        if quantity is not None:
            body["quantity"] = str(int(quantity))
        if order_type == "LIMIT":
            formatted_price = _format_decimal(price)
            if formatted_price is None:
                raise ValueError("LIMIT modify requires price")
            body["price"] = formatted_price
        result = self._post(f"/api/v1/orders/{order_id}/modify", body, account_required=True)
        return result if isinstance(result, dict) else {}
