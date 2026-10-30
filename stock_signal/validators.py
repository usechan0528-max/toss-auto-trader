from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def validate_candle(candle: Mapping[str, Any]) -> None:
    assert candle["high"] > 0, "high must be > 0"
    assert candle["low"] > 0, "low must be > 0"
    assert candle["open"] > 0, "open must be > 0"
    assert candle["close"] > 0, "close must be > 0"
    assert candle["high"] >= candle["low"], "high must be >= low"
    assert candle["high"] >= candle["open"], "high must be >= open"
    assert candle["high"] >= candle["close"], "high must be >= close"
    assert candle["low"] <= candle["open"], "low must be <= open"
    assert candle["low"] <= candle["close"], "low must be <= close"
    assert candle["volume"] >= 0, "volume must be >= 0"


def validate_quote(price: float, bid: float | None = None, ask: float | None = None, max_spread_pct: float = 2.0) -> None:
    if price <= 0:
        raise ValueError("price must be > 0")
    if bid is None or ask is None:
        return
    if bid <= 0 or ask <= 0 or ask < bid:
        raise ValueError("invalid bid/ask")
    spread_pct = (ask - bid) / price * 100
    if spread_pct > max_spread_pct:
        raise ValueError(f"spread too wide: {spread_pct:.2f}%")


def validate_price_data(data: pd.DataFrame, allow_zero_volume: bool = True) -> None:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("price data is empty")
    if data[["Open", "High", "Low", "Close", "Volume"]].isna().any().any():
        raise ValueError("price data contains missing values")

    zero_volume_count = int((data["Volume"] == 0).sum())
    if zero_volume_count and not allow_zero_volume:
        raise ValueError(f"zero-volume candles found: {zero_volume_count}")

    for index, row in data.iterrows():
        candle = {
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        }
        try:
            validate_candle(candle)
        except AssertionError as exc:
            raise ValueError(f"Invalid candle at {index}: {exc}") from exc
