from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import streamlit as st

from stock_signal.autotrade import (
    append_auto_trade_event,
    auto_trade_decision_to_dict,
    run_auto_trade,
    submitted_order_ids_from_events,
)
from stock_signal.market_data import QuoteSnapshot, TossMarketDataAdapter
from stock_signal.prediction_metrics import (
    FormulaPrediction,
    add_prediction_features,
    calculate_formula_prediction,
)


APP_DIR = Path(__file__).resolve().parent
SETTINGS_SNAPSHOT_PATH = APP_DIR / "app_settings.json"
TRADE_INTENT_PATH = APP_DIR / "latest_trade_intent.json"
TRADE_INTENT_HISTORY_PATH = APP_DIR / "trade_intents.jsonl"
ORDER_LEDGER_PATH = APP_DIR / "order_intent_events.jsonl"
EXECUTION_PLAN_PATH = APP_DIR / "latest_execution_plan.json"
TRADE_REVIEW_PATH = APP_DIR / "trade_reviews.jsonl"
LLM_TRADE_REVIEW_PATH = APP_DIR / "llm_trade_review.md"
AUTO_TRADE_PATH = APP_DIR / "latest_auto_trade.json"
AUTO_TRADE_EVENTS_PATH = APP_DIR / "auto_trade_events.jsonl"
AUTO_SESSION_EVENTS_PATH = APP_DIR / "auto_session_events.jsonl"
DAILY_PROFIT_LOCK_PATH = APP_DIR / "daily_profit_lock.json"
LONG_TERM_PLAN_PATH = APP_DIR / "latest_long_term_plan.json"
LIVE_ORDER_EVENTS_PATH = APP_DIR / "live_order_events.jsonl"
LATEST_SIGNAL_PATH = APP_DIR / "latest_signal.json"
SIGNAL_EVENTS_PATH = APP_DIR / "signal_events.jsonl"
REALTIME_POSITIONS_PATH = APP_DIR / "latest_realtime_positions.json"

TOSS_DATA_SOURCE = "Toss API"
DEFAULT_WATCHLIST_TEXT = "AAPL, NVDA, TSLA, MSFT, AMZN"

POPULAR_STOCK_SEARCH_INDEX = [
    {"symbol": "AAPL", "name": "Apple", "englishName": "Apple", "market": "NASDAQ"},
    {"symbol": "NVDA", "name": "Nvidia", "englishName": "Nvidia", "market": "NASDAQ"},
    {"symbol": "TSLA", "name": "Tesla", "englishName": "Tesla", "market": "NASDAQ"},
    {"symbol": "MSFT", "name": "Microsoft", "englishName": "Microsoft", "market": "NASDAQ"},
    {"symbol": "AMZN", "name": "Amazon", "englishName": "Amazon", "market": "NASDAQ"},
    {"symbol": "META", "name": "Meta", "englishName": "Meta", "market": "NASDAQ"},
    {"symbol": "GOOGL", "name": "Alphabet", "englishName": "Alphabet", "market": "NASDAQ"},
    {"symbol": "AMD", "name": "AMD", "englishName": "Advanced Micro Devices", "market": "NASDAQ"},
    {"symbol": "AVGO", "name": "Broadcom", "englishName": "Broadcom", "market": "NASDAQ"},
    {"symbol": "COIN", "name": "Coinbase", "englishName": "Coinbase", "market": "NASDAQ"},
    {"symbol": "MSTR", "name": "MicroStrategy", "englishName": "MicroStrategy", "market": "NASDAQ"},
    {"symbol": "SOXL", "name": "SOXL", "englishName": "SOXL", "market": "NYSEARCA"},
    {"symbol": "TQQQ", "name": "TQQQ", "englishName": "TQQQ", "market": "NASDAQ"},
]

Signal = Literal["LONG", "WAIT", "CAUTION"]


@dataclass
class PredictionResult:
    probability_up: float
    signal: Signal
    action_note: str
    accuracy: float
    precision: float
    latest_close: float
    latest_date: str
    confidence_target: str = "target_before_stop"
    horizon_bars: int = 8
    label_positive_rate: float = 0.0


@dataclass
class TradePlan:
    direction: Signal
    price: float
    upper_line: float
    lower_line: float
    support: float
    resistance: float
    breakout_entry: float
    pullback_entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_per_share: float
    rr_ratio_1: float
    rr_ratio_2: float
    memo: str
    risk_flags: list[str]
    invalid_if: list[str]


def clean_secret(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if any(marker in lowered for marker in ["your_", "example", "placeholder", "여기에", "발급받은", "실제_"]):
        return ""
    return text


def looks_like_toss_client_key(value: str) -> bool:
    return clean_secret(value).lower().startswith(("tsck", "test_ck", "live_ck"))


def secret_value(name: str, default: str = "") -> str:
    aliases = {
        "TOSS_API_TOKEN": ["TOSS_ACCESS_TOKEN", "TOSS_BEARER_TOKEN"],
        "TOSS_CLIENT_ID": ["TOSS_API_KEY", "TOSS_APP_KEY"],
        "TOSS_CLIENT_SECRET": ["TOSS_SECRET_KEY", "TOSS_APP_SECRET"],
        "TOSS_ACCOUNT_SEQ": ["TOSS_ACCOUNT_ID"],
    }
    value = clean_secret(os.getenv(name))
    if value:
        return value
    for alias in aliases.get(name, []):
        value = clean_secret(os.getenv(alias))
        if value:
            return value
    try:
        value = clean_secret(st.secrets.get(name))
        if value:
            return value
        nested_key = {
            "TOSS_API_TOKEN": "api_token",
            "TOSS_CLIENT_ID": "client_id",
            "TOSS_CLIENT_SECRET": "client_secret",
            "TOSS_ACCOUNT_SEQ": "account_seq",
            "TOSS_BASE_URL": "base_url",
        }.get(name)
        toss_secrets = st.secrets.get("toss", {})
        if nested_key and hasattr(toss_secrets, "get"):
            value = clean_secret(toss_secrets.get(nested_key))
            if value:
                return value
    except Exception:
        pass
    return default


def secret_bool(name: str, default: bool = False) -> bool:
    value = secret_value(name, "")
    if value == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on", "live"}


def secret_choice(name: str, choices: list[str], default: str) -> str:
    value = secret_value(name, "")
    return value if value in choices else default


def secret_float(name: str, default: float) -> float:
    value = secret_value(name, "")
    if value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


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


LEGACY_TICKER_ALIASES = {
    "SQ": "XYZ",
}


def normalize_ticker(ticker: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9.\-]", "", str(ticker or "").strip().upper())
    return LEGACY_TICKER_ALIASES.get(normalized, normalized)


def parse_watchlist(text: str) -> list[str]:
    symbols: list[str] = []
    for part in re.split(r"[,\s]+", str(text or "")):
        symbol = normalize_ticker(part)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def load_toss_price_data(
    ticker: str,
    token: str,
    client_id: str,
    client_secret: str,
    account_seq: str,
    base_url: str,
    interval: str,
) -> pd.DataFrame:
    return TossMarketDataAdapter(
        token=token,
        client_id=client_id,
        client_secret=client_secret,
        account_seq=account_seq or None,
        base_url=base_url,
        interval=interval,
        count=200,
    ).get_ohlcv(ticker)


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    return add_prediction_features(data)


def train_predictor(features: pd.DataFrame, reward_ratio: float = 1.8) -> tuple[PredictionResult, pd.DataFrame]:
    model_data = add_prediction_features(features)
    prediction = calculate_formula_prediction(features, horizon_bars=8)
    probability = float(prediction.probability_up)
    latest = model_data.iloc[-1]
    signal: Signal = "LONG" if probability >= 0.45 else "WAIT" if probability >= 0.40 else "CAUTION"
    return (
        PredictionResult(
            probability_up=probability,
            signal=signal,
            action_note="formula_prediction",
            accuracy=probability,
            precision=probability,
            latest_close=float(latest["Close"]),
            latest_date=str(model_data.index[-1]),
            confidence_target="target_before_stop",
            horizon_bars=int(prediction.horizon_bars),
            label_positive_rate=probability,
        ),
        model_data,
    )


def _latest_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        value = default
    return value if np.isfinite(value) else default


def build_trade_plan(features: pd.DataFrame, probability_up: float, reward_ratio: float) -> TradePlan:
    latest = features.iloc[-1]
    price = _latest_float(latest, "Close")
    atr = max(_latest_float(latest, "atr_14", price * 0.006), price * 0.006)
    support = float(features["Low"].tail(20).min())
    resistance = float(features["High"].tail(20).max())
    upper_line = _latest_float(latest, "boll_upper", resistance)
    lower_line = _latest_float(latest, "boll_lower", support)
    stop_loss = max(0.01, price - atr)
    risk_per_share = max(price - stop_loss, price * 0.003)
    breakout_entry = max(price, resistance * 0.999)
    pullback_entry = price
    take_profit_1 = price + risk_per_share * max(1.0, float(reward_ratio))
    take_profit_2 = price + risk_per_share * max(1.5, float(reward_ratio) + 0.6)
    rr_ratio_1 = (take_profit_1 - price) / risk_per_share if risk_per_share > 0 else 0.0
    rr_ratio_2 = (take_profit_2 - price) / risk_per_share if risk_per_share > 0 else 0.0
    rsi = _latest_float(latest, "rsi_14", 50.0)
    volume_ratio = _latest_float(latest, "volume_ratio", 0.0)
    risk_flags: list[str] = []
    if price < 1:
        risk_flags.append("PRICE_UNDER_1")
    if rsi >= 84:
        risk_flags.append("RSI_OVERHEATED")
    if volume_ratio >= 8:
        risk_flags.append("VOLUME_SPIKE_EXTREME")
    direction: Signal = "LONG" if float(probability_up) >= 0.45 else "WAIT" if float(probability_up) >= 0.40 else "CAUTION"
    return TradePlan(
        direction=direction,
        price=price,
        upper_line=upper_line,
        lower_line=lower_line,
        support=support,
        resistance=resistance,
        breakout_entry=breakout_entry,
        pullback_entry=pullback_entry,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        risk_per_share=risk_per_share,
        rr_ratio_1=rr_ratio_1,
        rr_ratio_2=rr_ratio_2,
        memo="auto formula plan",
        risk_flags=risk_flags,
        invalid_if=["price breaks stop", "data guard blocks"],
    )


def calculate_position(account_krw: float, usdkrw: float, entry: float, stop: float, risk_pct: float) -> dict[str, float]:
    account_usd = max(0.0, float(account_krw)) / max(float(usdkrw), 1.0)
    risk_budget = account_usd * max(0.0, float(risk_pct)) / 100
    risk_per_share = max(0.0, float(entry) - float(stop))
    shares = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
    return {
        "account_usd": account_usd,
        "risk_budget_usd": risk_budget,
        "shares": max(0, shares),
        "expected_loss_usd": max(0, shares) * risk_per_share,
    }


def detect_alerts(plan: TradePlan, previous: dict | None = None) -> list[str]:
    alerts: list[str] = []
    if plan.risk_flags:
        alerts.extend(plan.risk_flags)
    if previous and previous.get("symbol"):
        alerts.append("previous signal available")
    return alerts


def detect_market_events(plan: TradePlan, previous: dict | None = None) -> list[str]:
    events = ["entry candidate ready"] if plan.direction == "LONG" else ["waiting"]
    return events


def fallback_quote(symbol: str, features: pd.DataFrame) -> QuoteSnapshot:
    latest = features.iloc[-1]
    return QuoteSnapshot(symbol=symbol, price=float(latest["Close"]), timestamp=str(features.index[-1]))


def load_quote_snapshot(
    *,
    source: str,
    symbol: str,
    features: pd.DataFrame,
    period: str,
    interval: str,
    toss_token: str,
    toss_client_id: str,
    toss_client_secret: str,
    toss_account_seq: str,
    toss_base_url: str,
) -> tuple[QuoteSnapshot, str | None]:
    try:
        if source == TOSS_DATA_SOURCE and (toss_token or (toss_client_id and toss_client_secret)):
            quote = TossMarketDataAdapter(
                token=toss_token,
                client_id=toss_client_id,
                client_secret=toss_client_secret,
                account_seq=toss_account_seq or None,
                base_url=toss_base_url,
                interval=interval,
            ).get_quote(symbol)
            return quote, None
    except Exception as exc:
        return fallback_quote(symbol, features), f"quote fallback: {exc}"
    return fallback_quote(symbol, features), None


def formula_prediction_to_dict(prediction: FormulaPrediction) -> dict:
    return asdict(prediction)


def signal_payload(
    symbol: str,
    result: PredictionResult,
    plan: TradePlan,
    position: dict[str, float],
    alerts: list[str],
    market_events: list[str],
    *,
    quote: QuoteSnapshot | None = None,
    data_source: str = TOSS_DATA_SOURCE,
    data_warnings: list[str] | None = None,
    prediction_metrics: dict | None = None,
) -> dict:
    current_price = float(getattr(quote, "price", plan.price) or plan.price)
    action = "BUY_WATCH" if float(result.probability_up) >= 0.45 else "WAIT"
    raw = f"{symbol}:{current_price:.4f}:{plan.stop_loss:.4f}:{plan.take_profit_1:.4f}:{result.probability_up:.4f}"
    return {
        "fingerprint": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
        "symbol": symbol,
        "action": action,
        "current_price": current_price,
        "reference_entry": plan.breakout_entry if plan.direction == "LONG" else plan.pullback_entry,
        "stop_loss": plan.stop_loss,
        "take_profit": [plan.take_profit_1, plan.take_profit_2],
        "recommended_shares": int(position.get("shares", 0)),
        "alerts": list(alerts),
        "market_events": list(market_events),
        "risk_flags": list(plan.risk_flags),
        "latest_date": result.latest_date,
        "probability_up": round(float(result.probability_up) * 100, 4),
        "confidence_pct": round(float(result.probability_up) * 100, 4),
        "data_source": data_source,
        "data_warnings": data_warnings or [],
        "prediction_metrics": prediction_metrics or {},
        "safety_notice": "Live order is controlled by the auto-trade gate and broker preflight.",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_latest_signal(payload: dict) -> None:
    _write_json(LATEST_SIGNAL_PATH, payload)


def read_latest_signal() -> dict | None:
    if not LATEST_SIGNAL_PATH.exists():
        return None
    try:
        return json.loads(LATEST_SIGNAL_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def append_signal_event(payload: dict) -> None:
    event = dict(payload)
    event["logged_at"] = datetime.now(timezone.utc).isoformat()
    _append_jsonl(SIGNAL_EVENTS_PATH, event)


def save_trade_intent(intent: dict) -> None:
    _write_json(TRADE_INTENT_PATH, intent)
    event = dict(intent)
    event["logged_at"] = datetime.now(timezone.utc).isoformat()
    _append_jsonl(TRADE_INTENT_HISTORY_PATH, event)


def save_execution_plan(plan: dict) -> None:
    _write_json(EXECUTION_PLAN_PATH, plan)


def save_auto_trade_decision(decision: dict) -> None:
    _write_json(AUTO_TRADE_PATH, decision)


def auto_selection_score(row: pd.Series | dict) -> float:
    get = row.get if hasattr(row, "get") else lambda key, default=None: default
    confidence = float(get("confidence_pct", get("probability_up", 0)) or 0)
    volume_ratio = float(get("volume_ratio", 0) or 0)
    change_pct = float(get("change_pct", 0) or 0)
    surge_score = float(get("surge_score", 0) or 0)
    direction = str(get("direction", ""))
    risk = str(get("risk", ""))
    support_breakdown = bool(get("support_breakdown", False) or get("lower_breakdown", False))
    resistance_breakout = bool(get("resistance_breakout", False) or get("upper_breakout", False))
    support_gap_pct = float(get("support_gap_pct", 0) or 0)
    resistance_gap_pct = float(get("resistance_gap_pct", 0) or 0)
    score = confidence - 50
    score += min(max(volume_ratio, 0), 5) * 4
    score += max(change_pct, -2) * 3
    score += surge_score * 0.6
    if resistance_breakout:
        score += 12
    elif -0.35 <= resistance_gap_pct <= 0:
        score += 3
    if support_breakdown:
        score -= 28
    elif support_gap_pct < 0:
        score -= 8
    if direction in {"LONG", "BUY"}:
        score += 15
    elif direction in {"CAUTION", "WAIT"}:
        score -= 2
    if risk == "FLAGGED":
        score -= 8
    return float(round(score, 4))


def add_auto_selection_scores(scanner: pd.DataFrame) -> pd.DataFrame:
    ranked = scanner.copy()
    ranked["auto_score"] = ranked.apply(auto_selection_score, axis=1)
    return ranked


def choose_auto_symbol(
    scanner: pd.DataFrame,
    *,
    min_probability: float = 55.0,
    min_volume_ratio: float = 1.2,
    require_risk_normal: bool = True,
    allow_caution_entries: bool = False,
) -> dict[str, object]:
    if scanner.empty:
        return {
            "enabled": True,
            "symbol": "",
            "eligible": False,
            "reason": ["NO_SCANNER_ROWS"],
            "score": 0.0,
            "min_probability": float(min_probability),
            "min_volume_ratio": float(min_volume_ratio),
            "require_risk_normal": bool(require_risk_normal),
            "allow_caution_entries": bool(allow_caution_entries),
        }
    ranked = add_auto_selection_scores(scanner)
    ranked = ranked.sort_values("auto_score", ascending=False)
    allowed = {"LONG", "BUY"}
    if allow_caution_entries:
        allowed.update({"CAUTION", "WAIT"})
    direction = ranked.get("direction", pd.Series([""] * len(ranked), index=ranked.index)).astype(str)
    confidence = ranked.get("confidence_pct", pd.Series([0] * len(ranked), index=ranked.index)).astype(float)
    volume = ranked.get("volume_ratio", pd.Series([0] * len(ranked), index=ranked.index)).astype(float)
    risk = ranked.get("risk", pd.Series([""] * len(ranked), index=ranked.index)).astype(str)
    mask = direction.isin(allowed) & (confidence >= float(min_probability)) & (volume >= float(min_volume_ratio))
    if require_risk_normal:
        mask &= risk.isin({"", "NORMAL"})
    eligible = ranked[mask]
    selected = eligible.iloc[0] if not eligible.empty else ranked.iloc[0]
    reasons: list[str] = []
    if eligible.empty:
        reasons.append("NO_ELIGIBLE_AUTO_TRADE_CANDIDATE")
        if not bool(direction.isin(allowed).any()):
            reasons.append("NO_ALLOWED_DIRECTION")
        if bool((confidence < float(min_probability)).all()):
            reasons.append("PROBABILITY_BELOW_THRESHOLD")
        if bool((volume < float(min_volume_ratio)).all()):
            reasons.append("VOLUME_BELOW_THRESHOLD")
        if require_risk_normal and not bool(risk.isin({"", "NORMAL"}).any()):
            reasons.append("RISK_FLAGS_PRESENT")
    return {
        "enabled": True,
        "symbol": str(selected.get("symbol", "")),
        "eligible": not eligible.empty,
        "reason": reasons or ["AUTO_CANDIDATE_READY"],
        "score": float(selected.get("auto_score", 0.0)),
        "min_probability": float(min_probability),
        "min_volume_ratio": float(min_volume_ratio),
        "require_risk_normal": bool(require_risk_normal),
        "allow_caution_entries": bool(allow_caution_entries),
    }


def score_snapshot(ticker: str, features: pd.DataFrame, result: PredictionResult, plan: TradePlan) -> dict[str, object]:
    latest = features.iloc[-1]
    previous = features.iloc[-2] if len(features) >= 2 else latest
    close = float(latest["Close"])
    previous_close = float(previous["Close"])
    change_pct = (close / previous_close - 1) * 100 if previous_close > 0 else 0.0
    return {
        "symbol": ticker,
        "current_price": close,
        "change_pct": round(change_pct, 4),
        "confidence_pct": round(float(result.probability_up) * 100, 4),
        "direction": plan.direction,
        "upper_line": plan.upper_line,
        "lower_line": plan.lower_line,
        "reference_entry": plan.breakout_entry if plan.direction == "LONG" else plan.pullback_entry,
        "stop_loss": plan.stop_loss,
        "take_profit_1": plan.take_profit_1,
        "volume_ratio": float(latest.get("volume_ratio", 0)),
        "volume_zscore": float(latest.get("volume_zscore_20", 0)),
        "rsi": float(latest.get("rsi_14", 50)),
        "risk": "NORMAL" if not plan.risk_flags else "FLAGGED",
        "confidence_target": result.confidence_target,
        "confidence_horizon_bars": int(result.horizon_bars),
        "confidence_label_positive_rate": float(result.label_positive_rate) * 100,
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail_jsonl(path: Path, limit: int = 20) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> None:
    st.set_page_config(page_title="Auto Trading Monitor", layout="wide")
    st.title("Auto Trading Monitor")
    settings = _read_json(SETTINGS_SNAPSHOT_PATH)
    latest_signal = _read_json(LATEST_SIGNAL_PATH)
    latest_auto = _read_json(AUTO_TRADE_PATH)
    latest_intent = _read_json(TRADE_INTENT_PATH)
    realtime_positions = _read_json(REALTIME_POSITIONS_PATH)
    events = _tail_jsonl(AUTO_SESSION_EVENTS_PATH, 30)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Worker status", events[-1].get("status", "-") if events else "-")
    c2.metric("Intent", latest_intent.get("side", "-"))
    c3.metric("Symbol", latest_signal.get("symbol", "-"))
    c4.metric("Auto status", latest_auto.get("status", "-"))

    st.subheader("Relaxed Criteria")
    st.json(
        {
            "min_probability": settings.get("auto_selection_min_probability"),
            "min_volume_ratio": settings.get("auto_selection_min_volume_ratio"),
            "require_risk_normal": settings.get("auto_selection_require_risk_normal"),
            "allow_caution_entries": settings.get("auto_selection_allow_caution_entries"),
            "surge_focus_enabled": settings.get("surge_focus_enabled"),
        }
    )

    st.subheader("Latest Signal")
    st.json(latest_signal or {})

    st.subheader("Real Account Positions")
    positions = realtime_positions.get("positions") if isinstance(realtime_positions.get("positions"), list) else []
    st.caption(
        f"source={realtime_positions.get('source', '-')} "
        f"updated={realtime_positions.get('generated_at', '-')}"
    )
    if realtime_positions.get("error"):
        st.warning(f"position refresh fallback: {realtime_positions.get('error')}")
    if positions:
        st.dataframe(pd.DataFrame(positions), use_container_width=True)
    else:
        st.info("No real account positions to display yet.")

    st.subheader("Recent Worker Events")
    if events:
        frame = pd.DataFrame(events)
        keep = [column for column in ["logged_at", "status", "state", "symbol", "trade_intent_side", "trade_intent_quantity", "execution_status", "reasons"] if column in frame.columns]
        st.dataframe(frame[keep].tail(20), use_container_width=True)
    else:
        st.info("No worker events yet.")


if __name__ == "__main__":
    main()
