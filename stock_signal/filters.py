from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StockFilterConfig:
    min_price: float = 0.5
    max_price: float = 10.0
    min_day_volume: int = 5_000_000
    min_dollar_volume: float = 5_000_000
    max_spread_pct: float = 2.0
    min_change_pct: float = 5.0
    allowed_exchanges: set[str] = field(default_factory=lambda: {"NASDAQ", "NYSE", "AMEX"})


def filter_stock(snapshot: dict, config: StockFilterConfig | None = None) -> tuple[bool, list[str]]:
    config = config or StockFilterConfig()
    reasons: list[str] = []
    price = float(snapshot.get("price", 0))
    day_volume = float(snapshot.get("day_volume", 0))
    dollar_volume = float(snapshot.get("dollar_volume", price * day_volume))
    spread_pct = float(snapshot.get("spread_pct", 0))
    change_pct = float(snapshot.get("change_pct", 0))
    exchange = str(snapshot.get("exchange", "")).upper()

    if price < config.min_price or price > config.max_price:
        reasons.append("price outside penny-stock daytrade range")
    if day_volume < config.min_day_volume:
        reasons.append("day volume too low")
    if dollar_volume < config.min_dollar_volume:
        reasons.append("dollar volume too low")
    if spread_pct > config.max_spread_pct:
        reasons.append("spread too wide")
    if change_pct < config.min_change_pct:
        reasons.append("day change too low")
    if exchange and exchange not in config.allowed_exchanges:
        reasons.append("exchange not allowed")

    return len(reasons) == 0, reasons
