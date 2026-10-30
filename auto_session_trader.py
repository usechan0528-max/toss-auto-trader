from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

from app import (  # noqa: E402
    APP_DIR,
    AUTO_TRADE_EVENTS_PATH,
    ORDER_LEDGER_PATH,
    SETTINGS_SNAPSHOT_PATH,
    TOSS_DATA_SOURCE,
    add_auto_selection_scores,
    add_indicators,
    append_auto_trade_event,
    append_signal_event,
    auto_selection_score,
    auto_trade_decision_to_dict,
    build_trade_plan,
    calculate_position,
    choose_auto_symbol,
    detect_alerts,
    detect_market_events,
    formula_prediction_to_dict,
    load_quote_snapshot,
    load_toss_price_data,
    normalize_toss_auth,
    normalize_ticker,
    parse_watchlist,
    POPULAR_STOCK_SEARCH_INDEX,
    read_latest_signal,
    run_auto_trade,
    save_auto_trade_decision,
    save_execution_plan,
    save_latest_signal,
    save_trade_intent,
    score_snapshot,
    secret_bool,
    secret_choice,
    secret_float,
    secret_value,
    signal_payload,
    submitted_order_ids_from_events,
    train_predictor,
)

logging.disable(logging.NOTSET)

from stock_signal.autotrade import AutoTradeConfig, PaperBrokerExecutionAdapter  # noqa: E402
from stock_signal.daily_archive import DailyArchivePaths, maybe_archive_completed_sessions  # noqa: E402
from stock_signal.execution import AccountSnapshot, build_execution_plan, execution_plan_to_dict  # noqa: E402
from stock_signal.market_data import TossMarketDataAdapter  # noqa: E402
from stock_signal.prediction_metrics import calculate_formula_prediction as _calculate_formula_prediction  # noqa: E402
from stock_signal.safety_harness import (  # noqa: E402
    DataQualityGuard,
    DailyLossLimit,
    DailyProfitLock,
    DailyProfitLockLedger,
    DuplicateOrderLedger,
    apply_daily_loss_harness,
    apply_daily_profit_lock_harness,
    apply_data_quality_harness,
    apply_duplicate_order_harness,
)
from stock_signal.toss_broker import TossBrokerExecutionAdapter  # noqa: E402
from stock_signal.trade_intent import TradeIntent, build_trade_intent, trade_intent_to_dict  # noqa: E402


load_dotenv(APP_DIR / ".env", override=False)

SESSION_EVENT_PATH = APP_DIR / "auto_session_events.jsonl"
STOP_FILE = APP_DIR / "auto_trader.stop"
LOCK_FILE = APP_DIR / "auto_session_trader.lock"
DAILY_PROFIT_LOCK_PATH = APP_DIR / "daily_profit_lock.json"
WIDE_CANDIDATE_STATE_PATH = APP_DIR / "wide_candidate_state.json"
DAILY_ARCHIVE_ROOT = APP_DIR / "daily_archives"
LATEST_DAILY_SUMMARY_PATH = APP_DIR / "latest_daily_summary.json"
STRATEGY_LEARNING_SAMPLES_PATH = APP_DIR / "strategy_learning_samples.jsonl"
STRATEGY_ADJUSTMENTS_PATH = APP_DIR / "strategy_adjustments.json"
SIGNAL_EVENTS_PATH = APP_DIR / "signal_events.jsonl"
KST = timezone(timedelta(hours=9))

DEFAULT_LIQUID_US_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "TSLA",
    "AVGO",
    "AMD",
    "NFLX",
    "CRM",
    "ORCL",
    "ADBE",
    "INTC",
    "MU",
    "QCOM",
    "AMAT",
    "LRCX",
    "ASML",
    "ARM",
    "SMCI",
    "PLTR",
    "COIN",
    "MSTR",
    "SOFI",
    "HOOD",
    "RIVN",
    "LCID",
    "NIO",
    "XPEV",
    "BABA",
    "PDD",
    "JD",
    "UBER",
    "ABNB",
    "SHOP",
    "SNOW",
    "DDOG",
    "NET",
    "MDB",
    "CRWD",
    "PANW",
    "ZS",
    "OKTA",
    "ROKU",
    "SQ",
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
    "TGT",
    "HD",
    "LOW",
    "JPM",
    "BAC",
    "C",
    "GS",
    "MS",
    "V",
    "MA",
    "AXP",
    "XOM",
    "CVX",
    "OXY",
    "SLB",
    "COP",
    "UNH",
    "LLY",
    "NVO",
    "MRNA",
    "PFE",
    "JNJ",
    "ABBV",
    "TMO",
    "BA",
    "CAT",
    "GE",
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
    "UVXY",
    "MARA",
    "RIOT",
    "UPC",
    "DCOY",
    "LGCL",
    "NNBR",
    "TVRD",
    "AZI",
]

WIDE_ROTATION_US_UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "VOO",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLY",
    "XLI",
    "XLC",
    "XLP",
    "XLB",
    "XLU",
    "SMH",
    "SOXX",
    "ARKK",
    "ARKG",
    "IBIT",
    "GBTC",
    "BITO",
    "GLD",
    "SLV",
    "TLT",
    "HYG",
    "LQD",
    "TQQQ",
    "SQQQ",
    "SOXL",
    "SOXS",
    "SPXL",
    "SPXS",
    "UPRO",
    "SDS",
    "UVXY",
    "VXX",
    "LABU",
    "LABD",
    "FNGU",
    "FNGD",
    "NVDL",
    "NVDQ",
    "TSLL",
    "TSLS",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "TSLA",
    "AVGO",
    "AMD",
    "NFLX",
    "CRM",
    "ORCL",
    "ADBE",
    "INTC",
    "MU",
    "QCOM",
    "AMAT",
    "LRCX",
    "ASML",
    "ARM",
    "SMCI",
    "PLTR",
    "COIN",
    "MSTR",
    "SOFI",
    "HOOD",
    "RIVN",
    "LCID",
    "NIO",
    "XPEV",
    "BABA",
    "PDD",
    "JD",
    "UBER",
    "ABNB",
    "SHOP",
    "SNOW",
    "DDOG",
    "NET",
    "MDB",
    "CRWD",
    "PANW",
    "ZS",
    "OKTA",
    "ROKU",
    "SQ",
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
    "TGT",
    "HD",
    "LOW",
    "JPM",
    "BAC",
    "C",
    "GS",
    "MS",
    "V",
    "MA",
    "AXP",
    "XOM",
    "CVX",
    "OXY",
    "SLB",
    "COP",
    "UNH",
    "LLY",
    "NVO",
    "MRNA",
    "PFE",
    "JNJ",
    "ABBV",
    "TMO",
    "BA",
    "CAT",
    "GE",
    "F",
    "GM",
    "MARA",
    "RIOT",
    "CLSK",
    "IREN",
    "HUT",
    "BITF",
    "WULF",
    "BTDR",
    "CAN",
    "UPST",
    "AI",
    "SOUN",
    "BBAI",
    "PATH",
    "IONQ",
    "QBTS",
    "RGTI",
    "SERV",
    "ACHR",
    "JOBY",
    "LUNR",
    "RKLB",
    "ASTS",
    "SIDU",
    "LAES",
    "LASE",
    "XOS",
    "WKHS",
    "NKLA",
    "FSR",
    "GOEV",
    "FFIE",
    "MULN",
    "GME",
    "AMC",
    "BB",
    "KOSS",
    "BYND",
    "OPEN",
    "CVNA",
    "CHWY",
    "W",
    "ETSY",
    "BILL",
    "MELI",
    "SE",
    "GRAB",
    "NU",
    "WBD",
    "PARA",
    "RBLX",
    "TTD",
    "APP",
    "RUM",
    "DJT",
    "WOLF",
    "MP",
    "QS",
    "ENVX",
    "CHPT",
    "BLNK",
    "BE",
    "PLUG",
    "FCEL",
    "RUN",
    "ENPH",
    "SEDG",
    "FSLR",
    "TLRY",
    "CGC",
    "SNDL",
    "TLT",
    "XBI",
    "MRNA",
    "NVAX",
    "BNTX",
    "SAVA",
    "VKTX",
    "ALT",
    "IOVA",
    "CRSP",
    "EDIT",
    "NTLA",
    "BEAM",
    "RXRX",
    "DNA",
    "PACB",
    "ILMN",
    "TDOC",
    "HIMS",
    "TEM",
    "CRWD",
    "ZS",
    "S",
    "NET",
    "DDOG",
    "MDB",
    "ESTC",
    "CFLT",
    "SNOW",
    "NOW",
    "TEAM",
    "WDAY",
    "INTU",
    "ADSK",
    "ANET",
    "DELL",
    "HPE",
    "HPQ",
    "TXN",
    "ADI",
    "ON",
    "WDC",
    "STX",
    "MRVL",
    "KLAC",
    "TER",
    "TSM",
    "BIDU",
    "TME",
    "LI",
    "FUTU",
    "TAL",
    "EDU",
    "BEKE",
    "NCLH",
    "CCL",
    "RCL",
    "DAL",
    "UAL",
    "AAL",
    "LUV",
    "SAVE",
    "CCJ",
    "UEC",
    "UUUU",
    "SMR",
    "OKLO",
    "DNN",
    "FCX",
    "AA",
    "CLF",
    "NUE",
    "X",
    "NEM",
    "GOLD",
    "MOS",
    "CF",
    "DE",
    "URI",
    "PWR",
    "VRT",
    "ETN",
    "EMR",
    "HON",
    "LMT",
    "RTX",
    "NOC",
    "GD",
    "T",
    "VZ",
    "CMCSA",
    "PEP",
    "KO",
    "MCD",
    "CMG",
    "CAVA",
    "CELH",
    "ELF",
    "LULU",
    "ONON",
    "CROX",
    "DOCU",
    "ZM",
    "PINS",
    "TTWO",
    "EA",
    "RIVN",
    "LCID",
    "UPC",
    "DCOY",
    "LGCL",
    "NNBR",
    "TVRD",
    "AZI",
]


def _json_default(value: Any) -> str:
    return str(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_lock(lock_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def acquire_worker_lock(lock_path: Path = LOCK_FILE) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    current_pid = os.getpid()
    payload = {
        "pid": current_pid,
        "started_at": _now_iso(),
        "last_heartbeat": _now_iso(),
        "argv": sys.argv,
    }
    for _attempt in range(2):
        try:
            with lock_path.open("x", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return
        except FileExistsError:
            existing = _read_lock(lock_path)
            existing_pid = int(existing.get("pid") or 0)
            if existing_pid and _process_exists(existing_pid):
                raise RuntimeError(f"auto session worker already running: pid={existing_pid}")
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
    raise RuntimeError(f"could not acquire worker lock: {lock_path}")


def update_worker_lock(lock_path: Path = LOCK_FILE) -> None:
    existing = _read_lock(lock_path)
    if int(existing.get("pid") or 0) != os.getpid():
        return
    existing["last_heartbeat"] = _now_iso()
    temp_path = lock_path.with_suffix(lock_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    temp_path.replace(lock_path)


def release_worker_lock(lock_path: Path = LOCK_FILE) -> None:
    existing = _read_lock(lock_path)
    if int(existing.get("pid") or 0) != os.getpid():
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _read_settings() -> dict[str, Any]:
    if not SETTINGS_SNAPSHOT_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _setting(settings: dict[str, Any], key: str, default: Any) -> Any:
    return settings.get(key, default)


def _setting_float(settings: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(_setting(settings, key, default))
    except (TypeError, ValueError):
        return default


def _setting_int(settings: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(_setting(settings, key, default))
    except (TypeError, ValueError):
        return default


def _setting_bool(settings: dict[str, Any], key: str, default: bool) -> bool:
    value = _setting(settings, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _watchlist(settings: dict[str, Any], override: str = "") -> list[str]:
    source = override or os.getenv("AUTO_SESSION_WATCHLIST", "")
    if source:
        symbols = parse_watchlist(source)
    else:
        raw = settings.get("watchlist") or []
        symbols = [normalize_ticker(str(symbol)) for symbol in raw] if isinstance(raw, list) else parse_watchlist(str(raw))
    return _unique_symbols(symbols)


def _unique_symbols(symbols: list[str]) -> list[str]:
    unique: list[str] = []
    for symbol in symbols:
        normalized = normalize_ticker(str(symbol))
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _symbols_from_setting(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique_symbols([str(symbol) for symbol in value])
    return parse_watchlist(str(value or ""))


def _filter_broker_blocked_symbols(
    symbols: list[str],
    blocked_symbols: list[str],
    *,
    holding_symbols: list[str] | None = None,
) -> list[str]:
    blocked = set(_unique_symbols(blocked_symbols))
    protected = set(_unique_symbols(holding_symbols or []))
    if not blocked:
        return _unique_symbols(symbols)
    return [symbol for symbol in _unique_symbols(symbols) if symbol not in blocked or symbol in protected]


def _liquidity_universe(settings: dict[str, Any], override: str = "") -> list[str]:
    explicit = override or os.getenv("AUTO_SESSION_WATCHLIST", "")
    if explicit:
        return _watchlist(settings, explicit)

    configured = _symbols_from_setting(settings.get("candidate_universe"))
    popular = [str(item.get("symbol", "")) for item in POPULAR_STOCK_SEARCH_INDEX]
    return _unique_symbols(configured + _watchlist(settings) + popular + DEFAULT_LIQUID_US_UNIVERSE)


def _wide_symbol_pool(settings: dict[str, Any], override: str = "", max_symbols: int = 0) -> list[str]:
    configured = _symbols_from_setting(settings.get("wide_candidate_universe"))
    configured += _symbols_from_setting(settings.get("candidate_universe"))
    popular = [str(item.get("symbol", "")) for item in POPULAR_STOCK_SEARCH_INDEX]
    pool = _unique_symbols(
        _watchlist(settings, override)
        + configured
        + popular
        + DEFAULT_LIQUID_US_UNIVERSE
        + WIDE_ROTATION_US_UNIVERSE
    )
    if max_symbols and max_symbols > 0:
        return pool[: int(max_symbols)]
    return pool


def _parse_utc_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_wide_candidate_state(path: Path = WIDE_CANDIDATE_STATE_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"cursor": 0, "active": {}}
    if not isinstance(payload, dict):
        return {"cursor": 0, "active": {}}
    if not isinstance(payload.get("active"), dict):
        payload["active"] = {}
    payload["cursor"] = int(_as_float(payload.get("cursor"), 0.0))
    return payload


def _write_wide_candidate_state(state: dict[str, Any], path: Path = WIDE_CANDIDATE_STATE_PATH) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, default=_json_default, indent=2), encoding="utf-8")


def _rotation_batch(symbols: list[str], cursor: int, batch_size: int) -> tuple[list[str], int]:
    pool = _unique_symbols(symbols)
    if not pool:
        return [], 0
    size = max(1, min(int(batch_size), len(pool)))
    start = int(cursor) % len(pool)
    batch = [pool[(start + offset) % len(pool)] for offset in range(size)]
    return batch, (start + size) % len(pool)


def _passes_wide_fast_filter(
    row: dict[str, object],
    *,
    min_surge_score: float,
    min_lookback_change_pct: float,
    min_volume_ratio: float,
    min_model_candles: int = 0,
) -> bool:
    if not bool(row.get("liquidity_passed")):
        return False
    if int(min_model_candles) > 0 and _as_float(row.get("candle_count")) < int(min_model_candles):
        return False
    return (
        _as_float(row.get("surge_score")) >= float(min_surge_score)
        or _as_float(row.get("lookback_change_pct")) >= float(min_lookback_change_pct)
        or _as_float(row.get("surge_volume_ratio")) >= float(min_volume_ratio)
    )


def _fast_scan_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        return df

    volume_window = df["Volume"].rolling(20, min_periods=3)
    volume_ma = volume_window.mean()
    volume_std = volume_window.std()
    df["volume_ratio"] = (df["Volume"] / volume_ma.where(volume_ma != 0)).fillna(0.0)
    df["volume_zscore_20"] = ((df["Volume"] - volume_ma) / volume_std).fillna(0.0)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    rolling_volume = df["Volume"].rolling(20, min_periods=1).sum()
    rolling_value = (typical_price * df["Volume"]).rolling(20, min_periods=1).sum()
    vwap = rolling_value / rolling_volume.where(rolling_volume != 0)
    df["vwap_distance_pct"] = ((df["Close"] / vwap.where(vwap != 0)) - 1).fillna(0.0) * 100

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=3).mean()
    rs = gain / loss.where(loss != 0)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)
    df.loc[(loss == 0) & (gain > 0), "rsi_14"] = 100.0
    df.loc[(loss == 0) & (gain == 0), "rsi_14"] = 50.0
    return df


def _fast_scan_wide_symbols(
    symbols: list[str],
    *,
    interval: str,
    toss_token: str,
    toss_client_id: str,
    toss_client_secret: str,
    toss_account_seq: str,
    toss_base_url: str,
    candle_count: int,
    liquidity_recent_bars: int,
    min_recent_volume: float,
    min_recent_dollar_volume: float,
    surge_lookback_bars: int,
    min_surge_score: float,
    min_lookback_change_pct: float,
    min_volume_ratio: float,
    min_model_candles: int,
    request_pause_seconds: float = 0.0,
) -> tuple[list[dict[str, object]], list[tuple[str, str]]]:
    rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    adapter = TossMarketDataAdapter(
        token=toss_token,
        client_id=toss_client_id,
        client_secret=toss_client_secret,
        account_seq=toss_account_seq or None,
        base_url=toss_base_url,
        interval=interval,
        count=max(20, min(int(candle_count), 200)),
        timeout_seconds=8,
    )
    for symbol in symbols:
        ticker = normalize_ticker(symbol)
        if not ticker:
            continue
        try:
            prices = adapter.get_ohlcv(ticker)
            features = _fast_scan_features(prices)
            if len(features) < 3:
                raise ValueError("not enough candles for fast scan")
            liquidity = _liquidity_metrics(features, liquidity_recent_bars)
            surge = _surge_metrics(features, surge_lookback_bars)
            latest = features.iloc[-1]
            previous = features.iloc[-2] if len(features) >= 2 else latest
            latest_close = _as_float(latest.get("Close"))
            previous_close = _as_float(previous.get("Close"), latest_close)
            liquidity_passed = _passes_liquidity_filter(
                liquidity,
                min_recent_volume=min_recent_volume,
                min_recent_dollar_volume=min_recent_dollar_volume,
            )
            row = {
                "symbol": ticker,
                "last_seen_at": _now_iso(),
                "candle_count": int(len(features)),
                "liquidity_passed": bool(liquidity_passed),
                "recent_volume": round(liquidity["recent_volume"], 2),
                "recent_dollar_volume": round(liquidity["recent_dollar_volume"], 2),
                "latest_volume": round(liquidity["latest_volume"], 2),
                "latest_dollar_volume": round(liquidity["latest_dollar_volume"], 2),
                "price": round(latest_close, 4),
                "change_pct": round((latest_close / previous_close - 1) * 100, 4) if previous_close > 0 else 0.0,
                "lookback_change_pct": surge["lookback_change_pct"],
                "latest_bar_change_pct": surge["latest_bar_change_pct"],
                "surge_volume_ratio": surge["volume_ratio"],
                "surge_volume_zscore": surge["volume_zscore"],
                "breakout_pct": surge["breakout_pct"],
                "surge_score": surge["surge_score"],
            }
            row["promoted"] = _passes_wide_fast_filter(
                row,
                min_surge_score=min_surge_score,
                min_lookback_change_pct=min_lookback_change_pct,
                min_volume_ratio=min_volume_ratio,
                min_model_candles=min_model_candles,
            )
            rows.append(row)
        except Exception as exc:
            failures.append((ticker, str(exc)))
        if request_pause_seconds > 0:
            time.sleep(float(request_pause_seconds))
    return rows, failures


def _rank_wide_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            _as_float(row.get("surge_score")),
            _as_float(row.get("lookback_change_pct")),
            _as_float(row.get("surge_volume_ratio")),
            _as_float(row.get("recent_dollar_volume")),
        ),
        reverse=True,
    )


def _refresh_wide_candidates(
    settings: dict[str, Any],
    *,
    enabled: bool,
    interval: str,
    toss_token: str,
    toss_client_id: str,
    toss_client_secret: str,
    toss_account_seq: str,
    toss_base_url: str,
    liquidity_recent_bars: int,
    min_recent_volume: float,
    min_recent_dollar_volume: float,
    surge_lookback_bars: int,
    watchlist_override: str,
    max_symbols: int,
    batch_size: int,
    fast_candle_count: int,
    active_ttl_minutes: float,
    active_max_symbols: int,
    min_surge_score: float,
    min_lookback_change_pct: float,
    min_volume_ratio: float,
    min_model_candles: int,
    request_pause_seconds: float,
) -> tuple[list[str], dict[str, Any], list[tuple[str, str]]]:
    if not enabled:
        return [], {"enabled": False}, []

    pool = _wide_symbol_pool(settings, watchlist_override, max_symbols=max_symbols)
    state = _read_wide_candidate_state()
    batch, next_cursor = _rotation_batch(pool, int(state.get("cursor", 0)), batch_size)
    scanned_rows, failures = _fast_scan_wide_symbols(
        batch,
        interval=interval,
        toss_token=toss_token,
        toss_client_id=toss_client_id,
        toss_client_secret=toss_client_secret,
        toss_account_seq=toss_account_seq,
        toss_base_url=toss_base_url,
        candle_count=fast_candle_count,
        liquidity_recent_bars=liquidity_recent_bars,
        min_recent_volume=min_recent_volume,
        min_recent_dollar_volume=min_recent_dollar_volume,
        surge_lookback_bars=surge_lookback_bars,
        min_surge_score=min_surge_score,
        min_lookback_change_pct=min_lookback_change_pct,
        min_volume_ratio=min_volume_ratio,
        min_model_candles=min_model_candles,
        request_pause_seconds=request_pause_seconds,
    )

    now = datetime.now(timezone.utc)
    active: dict[str, dict[str, object]] = {}
    for symbol, row in (state.get("active") or {}).items():
        expires_at = _parse_utc_datetime(row.get("expires_at") if isinstance(row, dict) else None)
        if isinstance(row, dict) and expires_at and expires_at > now:
            active[normalize_ticker(symbol)] = row

    promoted_rows = [row for row in scanned_rows if bool(row.get("promoted"))]
    expires_at = (now + timedelta(minutes=max(1.0, float(active_ttl_minutes)))).isoformat()
    for row in promoted_rows:
        symbol = normalize_ticker(str(row.get("symbol") or ""))
        if not symbol:
            continue
        active[symbol] = {
            **row,
            "source": "wide_scan",
            "added_at": active.get(symbol, {}).get("added_at") or _now_iso(),
            "expires_at": expires_at,
        }

    ranked_active = _rank_wide_rows(list(active.values()))[: max(1, int(active_max_symbols))]
    active = {str(row.get("symbol")): row for row in ranked_active if row.get("symbol")}
    _write_wide_candidate_state(
        {
            "cursor": next_cursor,
            "updated_at": _now_iso(),
            "active": active,
            "last_batch": batch,
        }
    )

    summary = {
        "enabled": True,
        "universe_size": len(pool),
        "batch_size": len(batch),
        "scanned_count": len(scanned_rows),
        "promoted_count": len(promoted_rows),
        "active_count": len(ranked_active),
        "cursor": next_cursor,
        "fast_candle_count": int(fast_candle_count),
        "active_ttl_minutes": float(active_ttl_minutes),
        "min_surge_score": float(min_surge_score),
        "min_lookback_change_pct": float(min_lookback_change_pct),
        "min_volume_ratio": float(min_volume_ratio),
        "min_model_candles": int(min_model_candles),
        "request_pause_seconds": float(request_pause_seconds),
        "top_active_candidates": _rank_wide_rows(ranked_active)[:10],
        "top_scanned_candidates": _rank_wide_rows(scanned_rows)[:10],
    }
    active_symbols = [str(row.get("symbol")) for row in ranked_active if row.get("symbol")]
    return active_symbols, summary, failures


def _modeled_universe(
    *,
    base_symbols: list[str],
    dynamic_symbols: list[str],
    pinned_symbols: list[str],
    max_symbols: int,
) -> list[str]:
    pinned = _unique_symbols(pinned_symbols)
    combined = _unique_symbols(pinned + dynamic_symbols + base_symbols)
    if int(max_symbols) <= 0:
        return combined
    limit = max(int(max_symbols), len(pinned))
    return combined[:limit]


def _liquidity_metrics(features: pd.DataFrame, recent_bars: int) -> dict[str, float]:
    window = features.tail(max(1, int(recent_bars)))
    recent_volume = float(window["Volume"].sum())
    recent_dollar_volume = float((window["Close"] * window["Volume"]).sum())
    latest = features.iloc[-1]
    latest_volume = float(latest["Volume"])
    latest_dollar_volume = float(latest["Close"] * latest["Volume"])
    return {
        "recent_volume": recent_volume,
        "recent_dollar_volume": recent_dollar_volume,
        "latest_volume": latest_volume,
        "latest_dollar_volume": latest_dollar_volume,
    }


def _passes_liquidity_filter(
    metrics: dict[str, float],
    *,
    min_recent_volume: float,
    min_recent_dollar_volume: float,
) -> bool:
    return (
        float(metrics.get("recent_volume", 0.0)) >= max(0.0, float(min_recent_volume))
        and float(metrics.get("recent_dollar_volume", 0.0)) >= max(0.0, float(min_recent_dollar_volume))
    )


def _surge_metrics(features: pd.DataFrame, lookback_bars: int) -> dict[str, float]:
    if features.empty:
        return {
            "lookback_change_pct": 0.0,
            "latest_bar_change_pct": 0.0,
            "volume_ratio": 0.0,
            "volume_zscore": 0.0,
            "vwap_distance_pct": 0.0,
            "breakout_pct": 0.0,
            "surge_score": 0.0,
        }

    latest = features.iloc[-1]
    latest_close = _as_float(latest.get("Close"))
    prev_close = _as_float(features.iloc[-2].get("Close") if len(features) >= 2 else latest_close, latest_close)
    lookback = max(1, int(lookback_bars))
    start_index = max(0, len(features) - 1 - lookback)
    start_close = _as_float(features.iloc[start_index].get("Close"), latest_close)
    prior_window = features.iloc[max(0, len(features) - 1 - lookback) : -1]
    prior_high = _as_float(prior_window["High"].max() if not prior_window.empty and "High" in prior_window else latest_close, latest_close)

    lookback_change_pct = (latest_close / start_close - 1) * 100 if start_close > 0 else 0.0
    latest_bar_change_pct = (latest_close / prev_close - 1) * 100 if prev_close > 0 else 0.0
    breakout_pct = (latest_close / prior_high - 1) * 100 if prior_high > 0 else 0.0
    volume_ratio = _as_float(latest.get("volume_ratio"))
    volume_zscore = _as_float(latest.get("volume_zscore_20"))
    vwap_distance_pct = _as_float(latest.get("vwap_distance_pct"))
    rsi = _as_float(latest.get("rsi_14"), 50.0)

    score = 0.0
    score += max(lookback_change_pct, 0.0) * 12.0
    score += max(latest_bar_change_pct, 0.0) * 8.0
    score += max(breakout_pct, 0.0) * 8.0
    score += min(max(volume_ratio, 0.0), 8.0) * 4.0
    score += max(volume_zscore, 0.0) * 3.0
    score += max(vwap_distance_pct, 0.0) * 1.5
    if rsi >= 84:
        score -= min((rsi - 84) * 1.2, 20.0)

    return {
        "lookback_change_pct": round(lookback_change_pct, 4),
        "latest_bar_change_pct": round(latest_bar_change_pct, 4),
        "volume_ratio": round(volume_ratio, 4),
        "volume_zscore": round(volume_zscore, 4),
        "vwap_distance_pct": round(vwap_distance_pct, 4),
        "breakout_pct": round(breakout_pct, 4),
        "surge_score": round(score, 4),
    }


def _surge_candidate_scanner(
    scanner: pd.DataFrame,
    *,
    enabled: bool,
    min_lookback_change_pct: float,
    min_latest_bar_change_pct: float,
    min_volume_ratio: float,
    min_volume_zscore: float,
    min_surge_score: float,
) -> pd.DataFrame:
    if not enabled or scanner.empty:
        return scanner

    required = {
        "lookback_change_pct",
        "latest_bar_change_pct",
        "surge_volume_ratio",
        "surge_volume_zscore",
        "surge_score",
    }
    if not required.issubset(set(scanner.columns)):
        return scanner.iloc[0:0].copy()

    focused = scanner[
        (scanner["lookback_change_pct"].astype(float) >= float(min_lookback_change_pct))
        & (scanner["latest_bar_change_pct"].astype(float) >= float(min_latest_bar_change_pct))
        & (scanner["surge_volume_ratio"].astype(float) >= float(min_volume_ratio))
        & (scanner["surge_volume_zscore"].astype(float) >= float(min_volume_zscore))
        & (scanner["surge_score"].astype(float) >= float(min_surge_score))
    ].copy()
    if focused.empty:
        return focused
    return focused.sort_values(
        ["surge_score", "lookback_change_pct", "surge_volume_ratio", "latest_bar_change_pct"],
        ascending=[False, False, False, False],
    )


def _top_surge_candidates(scanner: pd.DataFrame, limit: int = 5) -> list[dict[str, object]]:
    if scanner.empty or "surge_score" not in scanner.columns:
        return []
    ranked = scanner.sort_values(
        ["surge_score", "lookback_change_pct", "surge_volume_ratio", "latest_bar_change_pct"],
        ascending=[False, False, False, False],
    ).head(max(1, int(limit)))
    items: list[dict[str, object]] = []
    for _index, row in ranked.iterrows():
        items.append(
            {
                "symbol": str(row.get("symbol") or row.get("ticker") or ""),
                "surge_score": round(_as_float(row.get("surge_score")), 4),
                "lookback_change_pct": round(_as_float(row.get("lookback_change_pct")), 4),
                "latest_bar_change_pct": round(_as_float(row.get("latest_bar_change_pct")), 4),
                "volume_ratio": round(_as_float(row.get("surge_volume_ratio")), 4),
                "volume_zscore": round(_as_float(row.get("surge_volume_zscore")), 4),
                "breakout_pct": round(_as_float(row.get("breakout_pct")), 4),
            }
        )
    return items


def _candidate_score_rows(
    scanner: pd.DataFrame,
    *,
    min_probability: float,
    min_volume_ratio: float,
    require_risk_normal: bool,
    surge_focus_enabled: bool,
    surge_min_lookback_change_pct: float,
    surge_min_latest_bar_change_pct: float,
    surge_min_volume_ratio: float,
    surge_min_volume_zscore: float,
    surge_min_score: float,
    allow_caution_entries: bool = False,
    limit: int = 250,
) -> list[dict[str, object]]:
    if scanner.empty:
        return []

    ranked = scanner.copy()
    if "auto_score" not in ranked.columns:
        ranked = add_auto_selection_scores(ranked)
    sort_columns = [
        column
        for column in ["auto_score", "surge_score", "confidence_pct", "surge_volume_ratio"]
        if column in ranked.columns
    ]
    if sort_columns:
        ranked = ranked.sort_values(sort_columns, ascending=[False] * len(sort_columns))

    rows: list[dict[str, object]] = []
    allowed_directions = {"LONG", "BUY"}
    if allow_caution_entries:
        allowed_directions.update({"CAUTION", "WAIT"})

    for rank, (_index, row) in enumerate(ranked.head(max(1, int(limit))).iterrows(), start=1):
        direction = str(row.get("direction", ""))
        risk = str(row.get("risk", ""))
        confidence_pct = _as_float(row.get("confidence_pct"))
        volume_ratio = _as_float(row.get("volume_ratio"))
        reentry_cooldown_blocked = bool(row.get("reentry_cooldown_blocked", False))
        support_breakdown = bool(row.get("support_breakdown", False))
        surge_passed = (
            not surge_focus_enabled
            or (
                _as_float(row.get("lookback_change_pct")) >= float(surge_min_lookback_change_pct)
                and _as_float(row.get("latest_bar_change_pct")) >= float(surge_min_latest_bar_change_pct)
                and _as_float(row.get("surge_volume_ratio")) >= float(surge_min_volume_ratio)
                and _as_float(row.get("surge_volume_zscore")) >= float(surge_min_volume_zscore)
                and _as_float(row.get("surge_score")) >= float(surge_min_score)
            )
        )
        auto_gate_passed = (
            direction in allowed_directions
            and confidence_pct >= float(min_probability)
            and volume_ratio >= float(min_volume_ratio)
            and (not require_risk_normal or risk in {"", "NORMAL"})
            and surge_passed
            and not bool(row.get("liquidity_filter_bypassed", False))
            and not reentry_cooldown_blocked
            and not support_breakdown
        )
        rows.append(
            {
                "rank": rank,
                "symbol": str(row.get("symbol", "")),
                "auto_score": round(_as_float(row.get("auto_score")), 2),
                "confidence_pct": round(confidence_pct, 2),
                "confidence_target": str(row.get("confidence_target", "")),
                "confidence_horizon_bars": int(_as_float(row.get("confidence_horizon_bars"))),
                "confidence_label_positive_rate": round(_as_float(row.get("confidence_label_positive_rate")), 2),
                "direction": direction,
                "change_pct": round(_as_float(row.get("change_pct")), 3),
                "volume_ratio": round(volume_ratio, 2),
                "volume_zscore": round(_as_float(row.get("volume_zscore")), 2),
                "rsi": round(_as_float(row.get("rsi")), 1),
                "risk": risk,
                "surge_score": round(_as_float(row.get("surge_score")), 2),
                "lookback_change_pct": round(_as_float(row.get("lookback_change_pct")), 3),
                "latest_bar_change_pct": round(_as_float(row.get("latest_bar_change_pct")), 3),
                "surge_volume_ratio": round(_as_float(row.get("surge_volume_ratio")), 2),
                "surge_volume_zscore": round(_as_float(row.get("surge_volume_zscore")), 2),
                "breakout_pct": round(_as_float(row.get("breakout_pct")), 3),
                "structure_state": str(row.get("structure_state") or ""),
                "support": round(_as_float(row.get("support")), 4),
                "resistance": round(_as_float(row.get("resistance")), 4),
                "support_gap_pct": round(_as_float(row.get("support_gap_pct")), 3),
                "resistance_gap_pct": round(_as_float(row.get("resistance_gap_pct")), 3),
                "support_breakdown": support_breakdown,
                "resistance_breakout": bool(row.get("resistance_breakout", False)),
                "recent_volume": round(_as_float(row.get("recent_volume")), 2),
                "recent_dollar_volume": round(_as_float(row.get("recent_dollar_volume")), 2),
                "latest_volume": round(_as_float(row.get("latest_volume")), 2),
                "latest_dollar_volume": round(_as_float(row.get("latest_dollar_volume")), 2),
                "liquidity_filter_bypassed": bool(row.get("liquidity_filter_bypassed", False)),
                "reentry_cooldown_blocked": reentry_cooldown_blocked,
                "reentry_cooldown_reason": str(row.get("reentry_cooldown_reason") or ""),
                "reentry_cooldown_expires_at": str(row.get("reentry_cooldown_expires_at") or ""),
                "reentry_cooldown_breakout_pct": round(_as_float(row.get("reentry_cooldown_breakout_pct")), 4),
                "surge_passed": bool(surge_passed),
                "auto_gate_passed": bool(auto_gate_passed),
            }
        )
    return rows


def _tail_session_events(path: Path = SESSION_EVENT_PATH, limit: int = 400) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-max(1, int(limit)) :]
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _parse_event_time(value: Any) -> datetime | None:
    try:
        text = str(value or "").strip()
        if not text:
            return None
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _current_kst_day_start(now: datetime | None = None) -> datetime:
    local_now = (now or datetime.now(timezone.utc)).astimezone(KST)
    return local_now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def _auto_trade_event_time(event: dict[str, Any]) -> datetime | None:
    return _parse_event_time(event.get("created_at") or event.get("logged_at"))


def _submitted_live_orders_for_day(
    path: Path = AUTO_TRADE_EVENTS_PATH,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    start = _current_kst_day_start(now)
    orders: list[dict[str, Any]] = []
    for event in _read_jsonl_records(path):
        event_time = _auto_trade_event_time(event)
        if event_time is None or event_time < start:
            continue
        if str(event.get("status") or "").upper() != "LIVE_SUBMITTED":
            continue
        order = event.get("order") if isinstance(event.get("order"), dict) else {}
        if not order:
            continue
        symbol = normalize_ticker(str(order.get("symbol") or ""))
        side = str(order.get("side") or "").upper()
        if not symbol or side not in {"BUY", "SELL"}:
            continue
        orders.append(
            {
                "time": event_time,
                "symbol": symbol,
                "side": side,
                "quantity": _as_float(order.get("quantity")),
                "price": _as_float(order.get("limit_price")),
            }
        )
    orders.sort(key=lambda item: item["time"])
    return orders


def _estimate_daily_roundtrips(
    orders: list[dict[str, Any]],
    *,
    commission_rate_pct: float,
    sell_tax_usd: float = 0.01,
) -> list[dict[str, Any]]:
    lots: dict[str, list[dict[str, Any]]] = {}
    roundtrips: list[dict[str, Any]] = []
    commission_rate = max(0.0, float(commission_rate_pct)) / 100
    for order in orders:
        symbol = str(order.get("symbol") or "")
        side = str(order.get("side") or "").upper()
        quantity = _as_float(order.get("quantity"))
        price = _as_float(order.get("price"))
        if not symbol or quantity <= 0 or price <= 0:
            continue
        notional = quantity * price
        fee = notional * commission_rate
        if side == "SELL":
            fee += max(0.0, float(sell_tax_usd))
        if side == "BUY":
            lots.setdefault(symbol, []).append(
                {
                    "quantity": quantity,
                    "price": price,
                    "fee_remaining": fee,
                    "time": order.get("time"),
                }
            )
            continue
        remaining = quantity
        symbol_lots = lots.setdefault(symbol, [])
        while remaining > 1e-9 and symbol_lots:
            lot = symbol_lots[0]
            matched = min(remaining, _as_float(lot.get("quantity")))
            if matched <= 0:
                symbol_lots.pop(0)
                continue
            lot_qty = _as_float(lot.get("quantity"))
            buy_fee = _as_float(lot.get("fee_remaining")) * (matched / lot_qty) if lot_qty > 0 else 0.0
            sell_fee = fee * (matched / quantity) if quantity > 0 else 0.0
            gross = (price - _as_float(lot.get("price"))) * matched
            net = gross - buy_fee - sell_fee
            roundtrips.append(
                {
                    "symbol": symbol,
                    "quantity": matched,
                    "buy_price": _as_float(lot.get("price")),
                    "sell_price": price,
                    "gross_pnl_usd": gross,
                    "fees_usd": buy_fee + sell_fee,
                    "net_pnl_usd": net,
                    "closed_at": order.get("time"),
                }
            )
            lot["quantity"] = lot_qty - matched
            lot["fee_remaining"] = _as_float(lot.get("fee_remaining")) - buy_fee
            remaining -= matched
            if _as_float(lot.get("quantity")) <= 1e-9:
                symbol_lots.pop(0)
    return roundtrips


def _daily_order_guard_stats(
    path: Path = AUTO_TRADE_EVENTS_PATH,
    *,
    now: datetime | None = None,
    commission_rate_pct: float = 0.1,
) -> dict[str, Any]:
    orders = _submitted_live_orders_for_day(path, now=now)
    roundtrips = _estimate_daily_roundtrips(orders, commission_rate_pct=commission_rate_pct)
    consecutive_losses = 0
    for item in reversed(roundtrips):
        if _as_float(item.get("net_pnl_usd")) < 0:
            consecutive_losses += 1
        else:
            break
    gross = sum(_as_float(item.get("gross_pnl_usd")) for item in roundtrips)
    fees = sum(_as_float(item.get("fees_usd")) for item in roundtrips)
    net = sum(_as_float(item.get("net_pnl_usd")) for item in roundtrips)
    return {
        "orders": len(orders),
        "closed_roundtrips": len(roundtrips),
        "wins": sum(1 for item in roundtrips if _as_float(item.get("net_pnl_usd")) > 0),
        "losses": sum(1 for item in roundtrips if _as_float(item.get("net_pnl_usd")) < 0),
        "consecutive_losses": consecutive_losses,
        "gross_pnl_usd": round(gross, 4),
        "fees_usd": round(fees, 4),
        "net_pnl_usd": round(net, 4),
        "commission_rate_pct": round(max(0.0, float(commission_rate_pct)), 6),
    }


def apply_daily_order_guard(
    payload: dict[str, Any],
    *,
    stats: dict[str, Any],
    enabled: bool = True,
    max_loss_usd: float = 0.0,
    max_consecutive_losses: int = 3,
) -> dict[str, Any]:
    guarded = dict(payload)
    guarded["daily_order_guard"] = {
        "enabled": bool(enabled),
        "max_loss_usd": round(max(0.0, float(max_loss_usd)), 2),
        "max_consecutive_losses": max(0, int(max_consecutive_losses)),
        **stats,
    }
    entry_edge_already_blocked = guarded.get("action") == "NO_TRADE" and isinstance(guarded.get("entry_edge"), dict)
    if not enabled or (guarded.get("action") != "BUY_WATCH" and not entry_edge_already_blocked):
        return guarded

    reasons: list[str] = []
    if max_loss_usd > 0 and _as_float(stats.get("net_pnl_usd")) <= -abs(float(max_loss_usd)):
        reasons.append("DAILY_ESTIMATED_NET_LOSS_LIMIT_REACHED")
    if max_consecutive_losses > 0 and int(stats.get("consecutive_losses") or 0) >= int(max_consecutive_losses):
        reasons.append("DAILY_CONSECUTIVE_LOSS_LIMIT_REACHED")
    if not reasons:
        return guarded

    guarded["action"] = "NO_TRADE"
    guarded["recommended_shares"] = 0
    guarded["reason"] = list(guarded.get("reason", [])) + reasons
    guarded["alerts"] = list(guarded.get("alerts", [])) + reasons
    guarded["risk_flags"] = list(guarded.get("risk_flags", [])) + [
        (
            "daily order guard blocked new entry: "
            f"net={_as_float(stats.get('net_pnl_usd')):.2f} "
            f"consecutive_losses={int(stats.get('consecutive_losses') or 0)}"
        )
    ]
    return guarded


def effective_daily_order_guard_max_loss_usd(
    *,
    configured_max_loss_usd: float,
    account_usd: float,
    daily_loss_limit_pct: float,
    daily_loss_limit_usd: float,
) -> float:
    configured = max(0.0, float(configured_max_loss_usd))
    if configured > 0:
        return configured
    pct_limit = max(0.0, float(account_usd)) * max(0.0, float(daily_loss_limit_pct)) / 100
    fixed_limit = max(0.0, float(daily_loss_limit_usd))
    active_limits = [limit for limit in [pct_limit, fixed_limit] if limit > 0]
    return min(active_limits) if active_limits else 0.0


def _event_reasons(event: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for source in [
        event.get("reasons"),
        event.get("reason"),
        (event.get("auto_selection") or {}).get("reason") if isinstance(event.get("auto_selection"), dict) else None,
        (event.get("auto_exit") or {}).get("reason") if isinstance(event.get("auto_exit"), dict) else None,
    ]:
        if isinstance(source, list):
            reasons.extend(str(item) for item in source)
        elif source:
            reasons.append(str(source))
    return reasons


def _event_exit_price(event: dict[str, Any], symbol: str) -> float:
    auto_selection = event.get("auto_selection") if isinstance(event.get("auto_selection"), dict) else {}
    auto_exit = event.get("auto_exit") if isinstance(event.get("auto_exit"), dict) else {}
    price = _as_float(auto_selection.get("current_price"))
    if price <= 0:
        price = _as_float(auto_exit.get("current_price"))
    if price <= 0:
        for position in event.get("position_summary") or []:
            if isinstance(position, dict) and normalize_ticker(str(position.get("symbol") or "")) == symbol:
                price = _as_float(position.get("current_price"))
                break
    return price


def _target_decay_exit_cooldowns(
    events: list[dict[str, Any]],
    *,
    now: datetime,
    cooldown_minutes: float,
) -> dict[str, dict[str, Any]]:
    if cooldown_minutes <= 0:
        return {}
    cooldowns: dict[str, dict[str, Any]] = {}
    cooldown_delta = timedelta(minutes=float(cooldown_minutes))
    cutoff = now - cooldown_delta
    for event in events:
        logged_at = _parse_event_time(event.get("logged_at") or event.get("created_at"))
        if logged_at is None or logged_at < cutoff:
            continue
        reasons = _event_reasons(event)
        if "EARLY_EXIT_TARGET_PROBABILITY_DECAY" not in reasons:
            continue
        if str(event.get("status") or "").upper() != "LIVE_SUBMITTED":
            continue
        if str(event.get("trade_intent_side") or "").upper() != "SELL":
            continue
        symbol = normalize_ticker(
            str(
                event.get("symbol")
                or ((event.get("auto_selection") or {}).get("symbol") if isinstance(event.get("auto_selection"), dict) else "")
                or ""
            )
        )
        if not symbol:
            continue
        previous = cooldowns.get(symbol)
        if previous and logged_at <= previous["logged_at"]:
            continue
        cooldowns[symbol] = {
            "symbol": symbol,
            "logged_at": logged_at,
            "expires_at": logged_at + cooldown_delta,
            "exit_price": _event_exit_price(event, symbol),
            "reason": "EARLY_EXIT_TARGET_PROBABILITY_DECAY",
        }
    return cooldowns


def _apply_reentry_cooldown(
    scanner: pd.DataFrame,
    events: list[dict[str, Any]],
    *,
    now: datetime,
    enabled: bool,
    cooldown_minutes: float,
    min_breakout_pct: float,
    min_volume_ratio: float,
    min_confidence_pct: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary: dict[str, Any] = {
        "enabled": bool(enabled),
        "cooldown_minutes": float(cooldown_minutes),
        "min_breakout_pct": float(min_breakout_pct),
        "min_volume_ratio": float(min_volume_ratio),
        "min_confidence_pct": float(min_confidence_pct),
        "blocked": [],
        "reentry_allowed": [],
        "blocked_count": 0,
        "reentry_allowed_count": 0,
    }
    if scanner.empty or not enabled or cooldown_minutes <= 0:
        return scanner, summary

    cooldowns = _target_decay_exit_cooldowns(events, now=now, cooldown_minutes=cooldown_minutes)
    if not cooldowns:
        return scanner, summary

    annotated = scanner.copy()
    for column, default in [
        ("reentry_cooldown_blocked", False),
        ("reentry_cooldown_reason", ""),
        ("reentry_cooldown_expires_at", ""),
        ("reentry_cooldown_exit_price", 0.0),
        ("reentry_cooldown_breakout_pct", 0.0),
    ]:
        if column not in annotated.columns:
            annotated[column] = default

    for index, row in annotated.iterrows():
        symbol = normalize_ticker(str(row.get("symbol") or row.get("ticker") or ""))
        cooldown = cooldowns.get(symbol)
        if not cooldown:
            continue
        expires_at = cooldown["expires_at"]
        if now >= expires_at:
            continue
        exit_price = _as_float(cooldown.get("exit_price"))
        current_price = _as_float(row.get("current_price"), _as_float(row.get("price"), _as_float(row.get("reference_entry"))))
        breakout_pct = (current_price / exit_price - 1) * 100 if current_price > 0 and exit_price > 0 else 0.0
        volume_ratio = max(_as_float(row.get("volume_ratio")), _as_float(row.get("surge_volume_ratio")))
        confidence_pct = _as_float(row.get("confidence_pct"))
        reentry_allowed = (
            breakout_pct >= float(min_breakout_pct)
            and volume_ratio >= float(min_volume_ratio)
            and confidence_pct >= float(min_confidence_pct)
        )
        target = {
            "symbol": symbol,
            "expires_at": expires_at.isoformat(),
            "exit_price": round(exit_price, 4),
            "current_price": round(current_price, 4),
            "breakout_pct": round(breakout_pct, 4),
            "volume_ratio": round(volume_ratio, 4),
            "confidence_pct": round(confidence_pct, 4),
        }
        if reentry_allowed:
            summary["reentry_allowed"].append(target)
            continue
        annotated.at[index, "reentry_cooldown_blocked"] = True
        annotated.at[index, "reentry_cooldown_reason"] = "REENTRY_COOLDOWN_AFTER_TARGET_DECAY_EXIT"
        annotated.at[index, "reentry_cooldown_expires_at"] = expires_at.isoformat()
        annotated.at[index, "reentry_cooldown_exit_price"] = round(exit_price, 4)
        annotated.at[index, "reentry_cooldown_breakout_pct"] = round(breakout_pct, 4)
        summary["blocked"].append(target)
    summary["blocked_count"] = len(summary["blocked"])
    summary["reentry_allowed_count"] = len(summary["reentry_allowed"])
    return annotated, summary


def _filter_reentry_cooldown(scanner: pd.DataFrame) -> pd.DataFrame:
    if scanner.empty or "reentry_cooldown_blocked" not in scanner.columns:
        return scanner
    return scanner[~scanner["reentry_cooldown_blocked"].fillna(False)].copy()


def _choose_auto_symbol_from_monitor_rows(
    scanner: pd.DataFrame,
    *,
    min_probability: float,
    min_volume_ratio: float,
    require_risk_normal: bool,
    allow_caution_entries: bool,
    max_entry_price: float = 0.0,
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
            "max_entry_price": round(max(0.0, float(max_entry_price)), 4),
        }

    ranked = scanner.copy()
    if "auto_score" not in ranked.columns:
        ranked = add_auto_selection_scores(ranked)
    sort_columns = [
        column
        for column in ["auto_score", "confidence_pct", "volume_ratio", "surge_score", "change_pct"]
        if column in ranked.columns
    ]
    if sort_columns:
        ranked = ranked.sort_values(sort_columns, ascending=[False] * len(sort_columns))

    allowed_directions = {"LONG", "BUY"}
    if allow_caution_entries:
        allowed_directions.update({"CAUTION", "WAIT"})

    direction = ranked.get("direction", pd.Series([""] * len(ranked), index=ranked.index)).astype(str)
    confidence = ranked.get("confidence_pct", pd.Series([0.0] * len(ranked), index=ranked.index)).astype(float)
    volume_ratio = ranked.get("volume_ratio", pd.Series([0.0] * len(ranked), index=ranked.index)).astype(float)
    risk = ranked.get("risk", pd.Series([""] * len(ranked), index=ranked.index)).astype(str)
    entry_price = ranked.get(
        "reference_entry",
        ranked.get("current_price", pd.Series([0.0] * len(ranked), index=ranked.index)),
    ).astype(float)
    bypassed = ranked.get("liquidity_filter_bypassed", pd.Series([False] * len(ranked), index=ranked.index)).astype(bool)
    reentry_cooldown = (
        ranked.get("reentry_cooldown_blocked", pd.Series([False] * len(ranked), index=ranked.index))
        .fillna(False)
        .astype(bool)
    )
    support_breakdown = (
        ranked.get("support_breakdown", pd.Series([False] * len(ranked), index=ranked.index)).fillna(False).astype(bool)
    )
    max_affordable_entry = max(0.0, float(max_entry_price))
    affordable = (entry_price > 0) & ((max_affordable_entry <= 0) | (entry_price <= max_affordable_entry))

    eligible_mask = (
        direction.isin(allowed_directions)
        & (confidence >= float(min_probability))
        & (volume_ratio >= float(min_volume_ratio))
        & (~bypassed)
        & (~reentry_cooldown)
        & (~support_breakdown)
        & affordable
    )
    if require_risk_normal:
        eligible_mask &= risk.isin({"", "NORMAL"})

    eligible = ranked[eligible_mask]
    affordable_ranked = ranked[affordable]
    selected = eligible.iloc[0] if not eligible.empty else affordable_ranked.iloc[0] if not affordable_ranked.empty else ranked.iloc[0]
    reasons: list[str] = []
    if eligible.empty:
        reasons.append("NO_ELIGIBLE_AUTO_TRADE_CANDIDATE")
        if not bool(direction.isin(allowed_directions).any()):
            reasons.append("NO_ALLOWED_DIRECTION")
        if bool((confidence < float(min_probability)).all()):
            reasons.append("PROBABILITY_BELOW_THRESHOLD")
        if bool((volume_ratio < float(min_volume_ratio)).all()):
            reasons.append("VOLUME_BELOW_THRESHOLD")
        if require_risk_normal and not bool(risk.isin({"", "NORMAL"}).any()):
            reasons.append("RISK_FLAGS_PRESENT")
        if bool(reentry_cooldown.all()):
            reasons.append("REENTRY_COOLDOWN_ACTIVE")
        if bool(support_breakdown.all()):
            reasons.append("LOWER_SUPPORT_BREAKDOWN_ACTIVE")
        if max_affordable_entry > 0 and bool((~affordable).all()):
            reasons.append("PRICE_ABOVE_AVAILABLE_BUYING_POWER")

    return {
        "enabled": True,
        "symbol": str(selected.get("symbol") or selected.get("ticker") or ""),
        "eligible": not eligible.empty,
        "reason": reasons or ["AUTO_CANDIDATE_READY"],
        "score": float(_as_float(selected.get("auto_score"))),
        "confidence_pct": float(_as_float(selected.get("confidence_pct"))),
        "direction": str(selected.get("direction") or ""),
        "risk": str(selected.get("risk") or ""),
        "change_pct": float(_as_float(selected.get("change_pct"))),
        "volume_ratio": float(_as_float(selected.get("volume_ratio"))),
        "volume_zscore": float(_as_float(selected.get("volume_zscore"))),
        "surge_score": float(_as_float(selected.get("surge_score"))),
        "lookback_change_pct": float(_as_float(selected.get("lookback_change_pct"))),
        "latest_bar_change_pct": float(_as_float(selected.get("latest_bar_change_pct"))),
        "surge_volume_ratio": float(_as_float(selected.get("surge_volume_ratio"))),
        "surge_volume_zscore": float(_as_float(selected.get("surge_volume_zscore"))),
        "breakout_pct": float(_as_float(selected.get("breakout_pct"))),
        "structure_state": str(selected.get("structure_state") or ""),
        "support": float(_as_float(selected.get("support"))),
        "resistance": float(_as_float(selected.get("resistance"))),
        "support_gap_pct": float(_as_float(selected.get("support_gap_pct"))),
        "resistance_gap_pct": float(_as_float(selected.get("resistance_gap_pct"))),
        "support_breakdown": bool(selected.get("support_breakdown", False)),
        "resistance_breakout": bool(selected.get("resistance_breakout", False)),
        "structure_break_buffer_pct": float(_as_float(selected.get("structure_break_buffer_pct"))),
        "recent_volume": float(_as_float(selected.get("recent_volume"))),
        "recent_dollar_volume": float(_as_float(selected.get("recent_dollar_volume"))),
        "latest_volume": float(_as_float(selected.get("latest_volume"))),
        "latest_dollar_volume": float(_as_float(selected.get("latest_dollar_volume"))),
        "liquidity_filter_bypassed": bool(selected.get("liquidity_filter_bypassed", False)),
        "reentry_cooldown_blocked": bool(selected.get("reentry_cooldown_blocked", False)),
        "min_probability": float(min_probability),
        "min_volume_ratio": float(min_volume_ratio),
        "require_risk_normal": bool(require_risk_normal),
        "allow_caution_entries": bool(allow_caution_entries),
        "max_entry_price": round(max_affordable_entry, 4),
        "selected_entry_price": round(_as_float(selected.get("reference_entry"), _as_float(selected.get("current_price"))), 4),
        "affordable": bool(_as_float(selected.get("reference_entry"), _as_float(selected.get("current_price"))) <= max_affordable_entry)
        if max_affordable_entry > 0
        else True,
    }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_float(item: dict[str, Any], keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in item and item.get(key) not in {None, ""}:
            value = _as_float(item.get(key), default)
            if value != default or str(item.get(key)).strip() not in {"", "0", "0.0"}:
                return value
    return default


def _holding_symbol(item: dict[str, Any]) -> str:
    return normalize_ticker(
        str(
            item.get("symbol")
            or item.get("ticker")
            or item.get("stockCode")
            or item.get("symbolCode")
            or item.get("code")
            or ""
        )
    )


def _holding_quantity(item: dict[str, Any]) -> float:
    return _first_float(
        item,
        [
            "quantity",
            "holdingQuantity",
            "qty",
            "balanceQuantity",
            "availableQuantity",
            "ownedQuantity",
        ],
    )


def _holding_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("items")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    for key in ["holdings", "stocks", "data", "result"]:
        nested = payload.get(key)
        if isinstance(nested, dict):
            nested_items = nested.get("items")
            if isinstance(nested_items, list):
                return [item for item in nested_items if isinstance(item, dict)]
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    return []


def _holding_quantity_map(payload: Any) -> dict[str, float]:
    quantities: dict[str, float] = {}
    for item in _holding_items(payload):
        symbol = _holding_symbol(item)
        quantity = _holding_quantity(item)
        if symbol and quantity > 0:
            quantities[symbol] = quantity
    return quantities


def _commission_rate_pct(commissions: Any, market_country: str = "US") -> float:
    if not isinstance(commissions, list):
        return 0.0
    preferred = []
    fallback = []
    for item in commissions:
        if not isinstance(item, dict):
            continue
        rate = _first_float(item, ["commissionRate", "rate", "feeRate", "orderCommissionRate"], default=-1.0)
        if rate < 0:
            continue
        country = str(item.get("marketCountry") or item.get("country") or item.get("market") or "").upper()
        if country == market_country.upper():
            preferred.append(rate)
        else:
            fallback.append(rate)
    rate = (preferred or fallback or [0.0])[0]
    if rate > 1:
        return rate
    return max(0.0, rate)


def _effective_commission_rate_pct(commissions: Any, fallback_rate_pct: float = 0.1) -> float:
    rate = _commission_rate_pct(commissions)
    if rate > 0:
        return rate
    return max(0.0, float(fallback_rate_pct))


def _roundtrip_cost_pct(
    *,
    commission_rate_pct: float,
    spread_pct: float = 0.0,
    slippage_buffer_pct: float = 0.0,
) -> float:
    return max(0.0, float(commission_rate_pct)) * 2 + max(0.0, float(spread_pct)) + max(
        0.0, float(slippage_buffer_pct)
    )


def _target_gap_pct(reference_entry: float, target_price: float) -> float:
    if reference_entry <= 0 or target_price <= 0:
        return 0.0
    return (float(target_price) / float(reference_entry) - 1) * 100


def apply_intraday_projection_targeting(
    payload: dict[str, Any],
    *,
    enabled: bool = True,
    capture_pct: float = 85.0,
    min_gross_target_pct: float = 1.6,
    max_gross_target_pct: float = 3.0,
) -> dict[str, Any]:
    guarded = dict(payload)
    metrics = guarded.get("prediction_metrics") if isinstance(guarded.get("prediction_metrics"), dict) else {}
    diagnostics = metrics.get("diagnostics") if isinstance(metrics.get("diagnostics"), dict) else {}
    take_profit = list(guarded.get("take_profit") or [])
    reference_entry = _as_float(guarded.get("reference_entry"))
    current_price = _as_float(guarded.get("current_price"), reference_entry)
    original_tp1 = _as_float(take_profit[0] if take_profit else 0.0)
    original_tp2 = _as_float(take_profit[1] if len(take_profit) > 1 else original_tp1)
    projected_high = _as_float(metrics.get("projected_high")) if metrics else 0.0
    projected_low = _as_float(metrics.get("projected_low")) if metrics else 0.0
    today_observed_high = _as_float(diagnostics.get("today_observed_high"), projected_high)
    today_observed_low = _as_float(diagnostics.get("today_observed_low"), projected_low)
    today_projected_high = _as_float(diagnostics.get("today_projected_high"), max(today_observed_high, projected_high))
    today_projected_low = _as_float(diagnostics.get("today_projected_low"), min(today_observed_low, projected_low))
    capture = _as_float(capture_pct, 85.0)
    if capture > 1:
        capture /= 100
    capture = max(0.05, min(1.0, capture))
    min_gross = max(0.0, _as_float(min_gross_target_pct, 1.6))
    max_gross = max(min_gross, _as_float(max_gross_target_pct, 3.0))
    original_gap_pct = _target_gap_pct(reference_entry, original_tp1)
    projected_high_gap_pct = _target_gap_pct(reference_entry, projected_high)
    projected_low_gap_pct = _target_gap_pct(reference_entry, projected_low)
    target = {
        "enabled": bool(enabled),
        "applied": False,
        "horizon_bars": int(_as_float(metrics.get("horizon_bars"), 0.0)) if metrics else 0,
        "reference_entry": round(reference_entry, 4),
        "current_price": round(current_price, 4),
        "original_take_profit_1": round(original_tp1, 4),
        "original_take_profit_2": round(original_tp2, 4),
        "original_gross_target_pct": round(original_gap_pct, 4),
        "projected_high": round(projected_high, 4),
        "projected_low": round(projected_low, 4),
        "projected_high_gap_pct": round(projected_high_gap_pct, 4),
        "projected_low_gap_pct": round(projected_low_gap_pct, 4),
        "today_observed_high": round(today_observed_high, 4),
        "today_observed_low": round(today_observed_low, 4),
        "today_projected_high": round(today_projected_high, 4),
        "today_projected_low": round(today_projected_low, 4),
        "capture_pct": round(capture * 100, 2),
        "min_gross_target_pct": round(min_gross, 4),
        "max_gross_target_pct": round(max_gross, 4),
        "reason": [],
    }

    if not enabled:
        target["reason"].append("INTRADAY_TARGETING_DISABLED")
    elif guarded.get("action") != "BUY_WATCH":
        target["reason"].append("ACTION_NOT_BUY_WATCH")
    elif not metrics:
        target["reason"].append("PREDICTION_METRICS_NOT_AVAILABLE")
    elif reference_entry <= 0 or original_tp1 <= 0:
        target["reason"].append("TARGET_INPUT_NOT_READY")
    elif projected_high_gap_pct <= 0:
        target["reason"].append("PROJECTED_HIGH_NOT_ABOVE_ENTRY")
    elif projected_high_gap_pct < min_gross:
        target["reason"].append("PROJECTED_HIGH_BELOW_MIN_TARGET")
    else:
        desired_gap_pct = min(max(projected_high_gap_pct * capture, min_gross), max_gross, projected_high_gap_pct)
        target["target_gross_pct"] = round(desired_gap_pct, 4)
        if desired_gap_pct <= original_gap_pct:
            target["reason"].append("ORIGINAL_TARGET_ALREADY_WIDE_ENOUGH")
        else:
            new_tp1 = reference_entry * (1 + desired_gap_pct / 100)
            target2_gap_pct = min(max(projected_high_gap_pct, desired_gap_pct), max_gross * 1.25)
            new_tp2 = reference_entry * (1 + target2_gap_pct / 100)
            take_profit = [
                round(new_tp1, 4),
                round(max(original_tp2, new_tp1, new_tp2), 4),
            ]
            guarded["take_profit"] = take_profit
            target["applied"] = True
            target["take_profit_1"] = take_profit[0]
            target["take_profit_2"] = take_profit[1]
            target["target_gross_pct"] = round(_target_gap_pct(reference_entry, take_profit[0]), 4)
            target["reason"].append("INTRADAY_PROJECTED_HIGH_TARGET_APPLIED")

    guarded["intraday_targeting"] = target
    return guarded


def apply_entry_liquidity_harness(
    payload: dict[str, Any],
    *,
    auto_selection: dict[str, Any],
    enabled: bool = True,
    min_recent_dollar_volume_usd: float = 5_000_000.0,
    min_latest_dollar_volume_usd: float = 200_000.0,
    min_volume_ratio: float = 1.2,
) -> dict[str, Any]:
    guarded = dict(payload)
    if not enabled or guarded.get("action") != "BUY_WATCH":
        return guarded

    selection = auto_selection if isinstance(auto_selection, dict) else {}
    recent_dollar_volume = _as_float(selection.get("recent_dollar_volume"))
    latest_dollar_volume = _as_float(selection.get("latest_dollar_volume"))
    recent_volume = _as_float(selection.get("recent_volume"))
    latest_volume = _as_float(selection.get("latest_volume"))
    spot_volume_ratio = _as_float(selection.get("volume_ratio"))
    surge_volume_ratio = _as_float(selection.get("surge_volume_ratio"))
    effective_volume_ratio = max(spot_volume_ratio, surge_volume_ratio)
    bypassed = bool(selection.get("liquidity_filter_bypassed", False))

    liquidity = {
        "enabled": bool(enabled),
        "recent_volume": round(recent_volume, 2),
        "recent_dollar_volume": round(recent_dollar_volume, 2),
        "latest_volume": round(latest_volume, 2),
        "latest_dollar_volume": round(latest_dollar_volume, 2),
        "volume_ratio": round(spot_volume_ratio, 4),
        "surge_volume_ratio": round(surge_volume_ratio, 4),
        "effective_volume_ratio": round(effective_volume_ratio, 4),
        "liquidity_filter_bypassed": bypassed,
        "min_recent_dollar_volume_usd": round(max(0.0, float(min_recent_dollar_volume_usd)), 2),
        "min_latest_dollar_volume_usd": round(max(0.0, float(min_latest_dollar_volume_usd)), 2),
        "min_volume_ratio": round(max(0.0, float(min_volume_ratio)), 4),
        "passed": True,
    }

    reasons: list[str] = []
    if bypassed:
        reasons.append("ENTRY_LIQUIDITY_FILTER_BYPASSED")
    if recent_dollar_volume < max(0.0, float(min_recent_dollar_volume_usd)):
        reasons.append("ENTRY_RECENT_DOLLAR_VOLUME_BELOW_MINIMUM")
    if latest_dollar_volume < max(0.0, float(min_latest_dollar_volume_usd)):
        reasons.append("ENTRY_LATEST_DOLLAR_VOLUME_BELOW_MINIMUM")
    if effective_volume_ratio < max(0.0, float(min_volume_ratio)):
        reasons.append("ENTRY_VOLUME_RATIO_BELOW_MINIMUM")

    if reasons:
        all_reasons = ["ENTRY_LIQUIDITY_BELOW_MINIMUM"] + reasons
        liquidity["passed"] = False
        guarded["action"] = "NO_TRADE"
        guarded["recommended_shares"] = 0
        guarded["reason"] = list(guarded.get("reason", [])) + all_reasons
        guarded["alerts"] = list(guarded.get("alerts", [])) + ["ENTRY_LIQUIDITY_BELOW_MINIMUM"]
        guarded["risk_flags"] = list(guarded.get("risk_flags", [])) + [
            (
                "entry liquidity blocked: "
                f"recent_value=${recent_dollar_volume:,.0f} "
                f"latest_value=${latest_dollar_volume:,.0f} "
                f"volume_ratio={effective_volume_ratio:.2f}"
            )
        ]
    guarded["entry_liquidity"] = liquidity
    return guarded


def apply_entry_structure_harness(
    payload: dict[str, Any],
    *,
    enabled: bool = True,
    block_lower_breakdown: bool = True,
    require_upper_breakout: bool = False,
) -> dict[str, Any]:
    guarded = dict(payload)
    structure = guarded.get("support_resistance") if isinstance(guarded.get("support_resistance"), dict) else {}
    entry_structure = {
        "enabled": bool(enabled),
        "passed": True,
        "block_lower_breakdown": bool(block_lower_breakdown),
        "require_upper_breakout": bool(require_upper_breakout),
        "state": str(structure.get("state") or ""),
        "upper_breakout": bool(structure.get("upper_breakout", False)),
        "lower_breakdown": bool(structure.get("lower_breakdown", False)),
        "support": _as_float(structure.get("support")),
        "resistance": _as_float(structure.get("resistance")),
        "support_gap_pct": _as_float(structure.get("support_gap_pct")),
        "resistance_gap_pct": _as_float(structure.get("resistance_gap_pct")),
        "break_buffer_pct": _as_float(structure.get("break_buffer_pct")),
        "reason": list(structure.get("reason") or []),
    }
    guarded["entry_structure"] = entry_structure
    if not enabled or guarded.get("action") != "BUY_WATCH":
        return guarded

    reasons: list[str] = []
    if not structure:
        reasons.append("ENTRY_STRUCTURE_DATA_UNAVAILABLE")
    if block_lower_breakdown and bool(structure.get("lower_breakdown")):
        reasons.append("ENTRY_BLOCKED_BY_LOWER_SUPPORT_BREAK")
    if require_upper_breakout and not bool(structure.get("upper_breakout")):
        reasons.append("ENTRY_WAITING_FOR_UPPER_RESISTANCE_BREAK")

    if bool(structure.get("upper_breakout")):
        guarded["market_events"] = list(guarded.get("market_events", [])) + ["upper resistance breakout confirmed"]
    if reasons:
        entry_structure["passed"] = False
        entry_structure["reason"] = list(entry_structure.get("reason") or []) + reasons
        guarded["action"] = "NO_TRADE"
        guarded["recommended_shares"] = 0
        guarded["reason"] = list(guarded.get("reason", [])) + reasons
        guarded["alerts"] = list(guarded.get("alerts", [])) + ["ENTRY_STRUCTURE_BLOCKED"]
        guarded["risk_flags"] = list(guarded.get("risk_flags", [])) + [
            (
                "entry structure blocked: "
                f"state={entry_structure['state'] or '-'} "
                f"support_gap={entry_structure['support_gap_pct']:.3f}% "
                f"resistance_gap={entry_structure['resistance_gap_pct']:.3f}%"
            )
        ]
    guarded["entry_structure"] = entry_structure
    return guarded


def apply_entry_edge_harness(
    payload: dict[str, Any],
    *,
    commission_rate_pct: float,
    enabled: bool = True,
    min_gross_target_pct: float = 1.2,
    min_net_target_pct: float = 0.5,
    fee_multiplier: float = 3.0,
    slippage_buffer_pct: float = 0.15,
    min_order_notional_usd: float = 75.0,
    min_projected_target_coverage_pct: float = 75.0,
    min_expected_move_pct: float = 0.08,
) -> dict[str, Any]:
    guarded = dict(payload)
    if not enabled or guarded.get("action") != "BUY_WATCH":
        return guarded

    reference_entry = _as_float(guarded.get("reference_entry"))
    take_profit = guarded.get("take_profit") if isinstance(guarded.get("take_profit"), list) else []
    target_price = _as_float(take_profit[0] if take_profit else 0.0)
    shares = max(0, int(_as_float(guarded.get("recommended_shares"))))
    spread_pct = _as_float((guarded.get("data_quality") or {}).get("spread_pct") if isinstance(guarded.get("data_quality"), dict) else 0.0)
    roundtrip_pct = _roundtrip_cost_pct(
        commission_rate_pct=commission_rate_pct,
        spread_pct=spread_pct,
        slippage_buffer_pct=slippage_buffer_pct,
    )
    gross_target_pct = _target_gap_pct(reference_entry, target_price)
    prediction_metrics = guarded.get("prediction_metrics") if isinstance(guarded.get("prediction_metrics"), dict) else {}
    projected_high = _as_float(prediction_metrics.get("projected_high")) if prediction_metrics else 0.0
    projected_high_gap_pct = _target_gap_pct(reference_entry, projected_high)
    projected_target_coverage_pct = (
        (projected_high_gap_pct / gross_target_pct) * 100 if gross_target_pct > 0 and projected_high_gap_pct > 0 else 0.0
    )
    expected_move_pct = _as_float(prediction_metrics.get("expected_move_pct")) if prediction_metrics else 0.0
    required_gross_pct = max(
        max(0.0, float(min_gross_target_pct)),
        roundtrip_pct * max(0.0, float(fee_multiplier)) + max(0.0, float(min_net_target_pct)),
    )
    estimated_net_target_pct = gross_target_pct - roundtrip_pct
    estimated_notional = reference_entry * shares if reference_entry > 0 and shares > 0 else 0.0
    edge = {
        "enabled": bool(enabled),
        "commission_rate_pct": round(max(0.0, float(commission_rate_pct)), 6),
        "roundtrip_cost_pct": round(roundtrip_pct, 4),
        "gross_target_pct": round(gross_target_pct, 4),
        "estimated_net_target_pct": round(estimated_net_target_pct, 4),
        "required_gross_target_pct": round(required_gross_pct, 4),
        "min_gross_target_pct": round(max(0.0, float(min_gross_target_pct)), 4),
        "min_net_target_pct": round(max(0.0, float(min_net_target_pct)), 4),
        "fee_multiplier": round(max(0.0, float(fee_multiplier)), 4),
        "slippage_buffer_pct": round(max(0.0, float(slippage_buffer_pct)), 4),
        "min_order_notional_usd": round(max(0.0, float(min_order_notional_usd)), 2),
        "estimated_notional_usd": round(estimated_notional, 2),
        "projected_high": round(projected_high, 4),
        "projected_high_gap_pct": round(projected_high_gap_pct, 4),
        "projected_target_coverage_pct": round(projected_target_coverage_pct, 2),
        "min_projected_target_coverage_pct": round(max(0.0, float(min_projected_target_coverage_pct)), 2),
        "expected_move_pct": round(expected_move_pct, 4),
        "min_expected_move_pct": round(max(0.0, float(min_expected_move_pct)), 4),
        "passed": True,
    }

    reasons: list[str] = []
    if gross_target_pct < required_gross_pct:
        reasons.append("ENTRY_EDGE_BELOW_FEE_ADJUSTED_MINIMUM")
    if estimated_notional > 0 and estimated_notional < max(0.0, float(min_order_notional_usd)):
        reasons.append("ORDER_NOTIONAL_BELOW_MINIMUM")
    if prediction_metrics and max(0.0, float(min_projected_target_coverage_pct)) > 0:
        if projected_target_coverage_pct < max(0.0, float(min_projected_target_coverage_pct)):
            reasons.append("ENTRY_PROJECTED_TARGET_COVERAGE_WEAK")
    if prediction_metrics and max(0.0, float(min_expected_move_pct)) > 0:
        if expected_move_pct < max(0.0, float(min_expected_move_pct)):
            reasons.append("ENTRY_EXPECTED_MOVE_BELOW_MINIMUM")

    if reasons:
        edge["passed"] = False
        guarded["action"] = "NO_TRADE"
        guarded["recommended_shares"] = 0
        guarded["reason"] = list(guarded.get("reason", [])) + reasons
        guarded["alerts"] = list(guarded.get("alerts", [])) + reasons
        guarded["risk_flags"] = list(guarded.get("risk_flags", [])) + [
            (
                "entry edge blocked: "
                f"gross_target={gross_target_pct:.2f}% required={required_gross_pct:.2f}% "
                f"net_after_cost={estimated_net_target_pct:.2f}%"
            )
        ]
        if prediction_metrics:
            guarded["risk_flags"] = list(guarded.get("risk_flags", [])) + [
                (
                    "entry projection blocked: "
                    f"coverage={projected_target_coverage_pct:.1f}% "
                    f"required={max(0.0, float(min_projected_target_coverage_pct)):.1f}% "
                    f"expected_move={expected_move_pct:.3f}%"
                )
            ]
    guarded["entry_edge"] = edge
    return guarded


def _holding_entry_price(item: dict[str, Any], quantity: float) -> float:
    entry = _first_float(
        item,
        [
            "averagePrice",
            "avgPrice",
            "averagePurchasePrice",
            "averageBuyPrice",
            "avgBuyPrice",
            "purchasePrice",
            "buyPrice",
            "averageUnitPrice",
            "avgUnitPrice",
            "unitCost",
            "averageCost",
            "acquisitionPrice",
            "averageAcquisitionPrice",
        ],
    )
    if entry > 0:
        return entry
    cost_basis = _first_float(
        item,
        [
            "purchaseAmount",
            "buyAmount",
            "investedAmount",
            "costBasis",
            "totalCost",
            "totalPurchaseAmount",
            "acquisitionAmount",
        ],
    )
    if quantity > 0 and cost_basis > 0:
        return cost_basis / quantity
    return 0.0


def _holding_current_price(
    symbol: str,
    item: dict[str, Any],
    loaded: dict[str, tuple[pd.DataFrame, pd.DataFrame, Any, Any, pd.DataFrame]],
) -> float:
    if symbol in loaded:
        _prices, features, _result, plan, _model_data = loaded[symbol]
        plan_price = _as_float(getattr(plan, "price", 0.0))
        if plan_price > 0:
            return plan_price
        if not features.empty:
            latest_close = _as_float(features.iloc[-1].get("Close"))
            if latest_close > 0:
                return latest_close
    return _first_float(
        item,
        [
            "currentPrice",
            "lastPrice",
            "price",
            "marketPrice",
            "evaluationPrice",
            "presentPrice",
        ],
    )


def _position_summary_rows(
    holdings_payload: Any,
    loaded: dict[str, tuple[pd.DataFrame, pd.DataFrame, Any, Any, pd.DataFrame]],
    commissions: Any,
    *,
    manual_take_profit_pct: float = 2.0,
    manual_stop_loss_pct: float = 2.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    commission_pct = _commission_rate_pct(commissions)
    commission_rate = commission_pct / 100
    for item in _holding_items(holdings_payload):
        symbol = _holding_symbol(item)
        quantity = _holding_quantity(item)
        if not symbol or quantity <= 0:
            continue
        entry_price = _holding_entry_price(item, quantity)
        current_price = _holding_current_price(symbol, item, loaded)
        cost_basis = entry_price * quantity if entry_price > 0 else _first_float(
            item,
            ["purchaseAmount", "buyAmount", "investedAmount", "costBasis", "totalCost", "totalPurchaseAmount"],
        )
        market_value = current_price * quantity if current_price > 0 else _first_float(
            item,
            ["marketValue", "evaluationAmount", "currentAmount", "totalEvaluationAmount"],
        )
        gross_pnl = market_value - cost_basis if market_value > 0 and cost_basis > 0 else 0.0
        gross_return_pct = gross_pnl / cost_basis * 100 if cost_basis > 0 else 0.0
        estimated_fee = (cost_basis + market_value) * commission_rate if commission_rate > 0 and cost_basis > 0 and market_value > 0 else 0.0
        fee_adjusted_pnl = gross_pnl - estimated_fee if cost_basis > 0 and market_value > 0 else 0.0
        fee_adjusted_return_pct = fee_adjusted_pnl / cost_basis * 100 if cost_basis > 0 else 0.0

        stop_loss = 0.0
        take_profit_1 = 0.0
        target_source = ""
        if symbol in loaded:
            _prices, _features, _result, plan, _model_data = loaded[symbol]
            stop_loss = _as_float(getattr(plan, "stop_loss", 0.0))
            take_profit_1 = _as_float(getattr(plan, "take_profit_1", 0.0))
            target_source = "plan"
        manual_target = entry_price * (1 + abs(float(manual_take_profit_pct)) / 100) if entry_price > 0 else 0.0
        manual_stop_loss = entry_price * (1 - abs(float(manual_stop_loss_pct)) / 100) if entry_price > 0 else 0.0
        if manual_target > take_profit_1:
            take_profit_1 = manual_target
            target_source = "manual"
        if manual_stop_loss > 0 and (stop_loss <= 0 or manual_stop_loss < stop_loss):
            stop_loss = manual_stop_loss
        stop_gap_pct = (current_price / stop_loss - 1) * 100 if current_price > 0 and stop_loss > 0 else 0.0
        target_gap_pct = (take_profit_1 / current_price - 1) * 100 if current_price > 0 and take_profit_1 > 0 else 0.0
        rows.append(
            {
                "symbol": symbol,
                "quantity": round(quantity, 6),
                "entry_price": round(entry_price, 4),
                "current_price": round(current_price, 4),
                "cost_basis_usd": round(cost_basis, 4),
                "market_value_usd": round(market_value, 4),
                "gross_pnl_usd": round(gross_pnl, 4),
                "gross_return_pct": round(gross_return_pct, 4),
                "estimated_roundtrip_fee_usd": round(estimated_fee, 4),
                "fee_adjusted_pnl_usd": round(fee_adjusted_pnl, 4),
                "fee_adjusted_return_pct": round(fee_adjusted_return_pct, 4),
                "commission_rate_pct": round(commission_pct, 6),
                "stop_loss": round(stop_loss, 4),
                "take_profit_1": round(take_profit_1, 4),
                "stop_gap_pct": round(stop_gap_pct, 4),
                "target_gap_pct": round(target_gap_pct, 4),
                "target_source": target_source,
                "manual_stop_loss_pct": round(abs(float(manual_stop_loss_pct)), 4),
                "manual_take_profit_pct": round(abs(float(manual_take_profit_pct)), 4),
            }
        )
    rows.sort(key=lambda row: abs(float(row.get("market_value_usd") or 0.0)), reverse=True)
    return rows


def _whole_share_quantity(quantity: float) -> int:
    return max(0, int(float(quantity)))


def _target_exit_trigger(current_price: float, target_price: float, tolerance_pct: float) -> float:
    return max(0.0, float(target_price) * (1 - max(0.0, float(tolerance_pct)) / 100))


def _exit_reason(current_price: float, stop_loss: float, target_price: float, tolerance_pct: float) -> str:
    if current_price <= stop_loss:
        return "STOP_LOSS_REACHED"
    if current_price >= _target_exit_trigger(current_price, target_price, tolerance_pct):
        return "TARGET_WITHIN_TOLERANCE"
    return ""


def _prediction_probability_pct(result: Any) -> float:
    for name in ["probability_up", "accuracy", "precision", "label_positive_rate"]:
        value = getattr(result, name, None)
        if value is None:
            continue
        probability = _as_float(value, -1.0)
        if probability < 0:
            continue
        return probability * 100 if probability <= 1.0 else probability
    return 100.0


def _series_or_default(df: pd.DataFrame, key: str, default: pd.Series) -> pd.Series:
    if key in df:
        return pd.to_numeric(df[key], errors="coerce")
    return default


def _support_resistance_signal(
    features: pd.DataFrame,
    *,
    current_price: float = 0.0,
    enabled: bool = True,
    lookback_bars: int = 20,
    confirm_bars: int = 2,
    min_break_pct: float = 0.15,
    atr_buffer_multiplier: float = 0.25,
    min_volume_ratio: float = 1.0,
    min_volume_zscore: float = -0.2,
) -> dict[str, object]:
    if not enabled:
        return {"enabled": False, "state": "DISABLED", "reason": ["STRUCTURE_GUARD_DISABLED"]}
    if features.empty or "Close" not in features:
        return {"enabled": True, "state": "DATA_UNAVAILABLE", "reason": ["STRUCTURE_DATA_UNAVAILABLE"]}

    bars = max(5, int(lookback_bars))
    confirms = max(1, int(confirm_bars))
    df = features.copy().tail(max(bars + confirms + 5, 40))
    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df.get("High", close), errors="coerce")
    low = pd.to_numeric(df.get("Low", close), errors="coerce")
    valid = pd.DataFrame({"Close": close, "High": high, "Low": low}).dropna()
    if len(valid) < max(8, confirms + 5):
        return {"enabled": True, "state": "DATA_UNAVAILABLE", "reason": ["STRUCTURE_NOT_ENOUGH_HISTORY"]}

    close = valid["Close"]
    high = valid["High"]
    low = valid["Low"]
    indexed = df.reindex(valid.index)
    stable_support = low.rolling(bars, min_periods=5).min().shift(confirms)
    stable_resistance = high.rolling(bars, min_periods=5).max().shift(confirms)
    feature_support = _series_or_default(indexed, "support_20_prev", stable_support)
    feature_resistance = _series_or_default(
        indexed,
        "resistance_20_prev",
        stable_resistance,
    )
    support = stable_support.fillna(feature_support).fillna(low.rolling(bars, min_periods=5).min().shift(1))
    resistance = stable_resistance.fillna(feature_resistance).fillna(high.rolling(bars, min_periods=5).max().shift(1))
    lower_line = _series_or_default(indexed, "boll_lower", support).fillna(support)
    upper_line = _series_or_default(indexed, "boll_upper", resistance).fillna(resistance)
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = _series_or_default(indexed, "atr_14", tr.rolling(14, min_periods=5).mean()).fillna(
        tr.rolling(14, min_periods=5).mean()
    )
    volume = pd.to_numeric(indexed.get("Volume", pd.Series([0.0] * len(valid), index=valid.index)), errors="coerce")
    volume_ratio = _series_or_default(
        indexed,
        "volume_ratio",
        volume / volume.rolling(20, min_periods=5).mean().replace(0, pd.NA),
    ).fillna(0.0)
    volume_zscore = _series_or_default(
        indexed,
        "volume_zscore_20",
        pd.Series([0.0] * len(valid), index=valid.index),
    ).fillna(0.0)

    price = _as_float(current_price, float(close.iloc[-1]))
    if price <= 0:
        price = float(close.iloc[-1])
    latest_support = _as_float(support.iloc[-1], float(low.tail(bars).min()))
    latest_resistance = _as_float(resistance.iloc[-1], float(high.tail(bars).max()))
    latest_lower_line = _as_float(lower_line.iloc[-1], latest_support)
    latest_upper_line = _as_float(upper_line.iloc[-1], latest_resistance)
    latest_atr = max(_as_float(atr.iloc[-1], price * 0.003), price * 0.003)
    atr_pct = latest_atr / price * 100 if price > 0 else 0.0
    buffer_pct = max(0.0, float(min_break_pct), atr_pct * max(0.0, float(atr_buffer_multiplier)))
    latest_volume_ratio = max(0.0, _as_float(volume_ratio.iloc[-1]))
    latest_volume_zscore = _as_float(volume_zscore.iloc[-1])
    latest_close = float(close.iloc[-1])
    previous_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close
    latest_bar_change_pct = (price / previous_close - 1) * 100 if previous_close > 0 else 0.0
    candle_range = max(float(high.iloc[-1]) - float(low.iloc[-1]), 0.0)
    close_location = (price - float(low.iloc[-1])) / candle_range if candle_range > 0 else 0.5
    volume_confirmed = latest_volume_ratio >= float(min_volume_ratio) or latest_volume_zscore >= float(min_volume_zscore)

    support_floor = latest_support * (1 - buffer_pct / 100) if latest_support > 0 else 0.0
    resistance_ceiling = latest_resistance * (1 + buffer_pct / 100) if latest_resistance > 0 else 0.0
    recent = valid.tail(confirms)
    recent_support = support.reindex(recent.index).fillna(latest_support)
    recent_resistance = resistance.reindex(recent.index).fillna(latest_resistance)
    recent_close = close.reindex(recent.index)
    lower_confirm_count = int((recent_close < recent_support * (1 - buffer_pct / 100)).sum()) if latest_support > 0 else 0
    upper_confirm_count = int((recent_close > recent_resistance * (1 + buffer_pct / 100)).sum()) if latest_resistance > 0 else 0

    lower_break_now = bool(support_floor > 0 and price < support_floor)
    upper_break_now = bool(resistance_ceiling > 0 and price > resistance_ceiling)
    lower_impulse = latest_bar_change_pct <= -max(buffer_pct * 0.75, 0.08) and close_location <= 0.35
    upper_impulse = latest_bar_change_pct >= max(buffer_pct * 0.75, 0.08) and close_location >= 0.65
    lower_breakdown = bool(lower_break_now and volume_confirmed and (lower_confirm_count >= confirms or lower_impulse))
    upper_breakout = bool(upper_break_now and volume_confirmed and (upper_confirm_count >= confirms or upper_impulse))

    support_gap_pct = (price / latest_support - 1) * 100 if latest_support > 0 else 0.0
    resistance_gap_pct = (price / latest_resistance - 1) * 100 if latest_resistance > 0 else 0.0
    lower_line_gap_pct = (price / latest_lower_line - 1) * 100 if latest_lower_line > 0 else 0.0
    upper_line_gap_pct = (price / latest_upper_line - 1) * 100 if latest_upper_line > 0 else 0.0

    if lower_breakdown:
        state = "LOWER_SUPPORT_BROKEN"
        reasons = ["LOWER_SUPPORT_BROKEN"]
    elif upper_breakout:
        state = "UPPER_RESISTANCE_BROKEN"
        reasons = ["UPPER_RESISTANCE_BROKEN"]
    elif lower_break_now and not volume_confirmed:
        state = "LOWER_BREAK_UNCONFIRMED"
        reasons = ["LOWER_BREAK_WITHOUT_VOLUME_CONFIRMATION"]
    elif upper_break_now and not volume_confirmed:
        state = "UPPER_BREAK_UNCONFIRMED"
        reasons = ["UPPER_BREAK_WITHOUT_VOLUME_CONFIRMATION"]
    else:
        state = "RANGE"
        reasons = ["STRUCTURE_RANGE_INTACT"]

    return {
        "enabled": True,
        "state": state,
        "reason": reasons,
        "upper_breakout": upper_breakout,
        "lower_breakdown": lower_breakdown,
        "support": round(latest_support, 4),
        "resistance": round(latest_resistance, 4),
        "lower_line": round(latest_lower_line, 4),
        "upper_line": round(latest_upper_line, 4),
        "support_floor": round(support_floor, 4),
        "resistance_ceiling": round(resistance_ceiling, 4),
        "support_gap_pct": round(support_gap_pct, 4),
        "resistance_gap_pct": round(resistance_gap_pct, 4),
        "lower_line_gap_pct": round(lower_line_gap_pct, 4),
        "upper_line_gap_pct": round(upper_line_gap_pct, 4),
        "break_buffer_pct": round(buffer_pct, 4),
        "atr_pct": round(atr_pct, 4),
        "volume_ratio": round(latest_volume_ratio, 4),
        "volume_zscore": round(latest_volume_zscore, 4),
        "volume_confirmed": bool(volume_confirmed),
        "latest_bar_change_pct": round(latest_bar_change_pct, 4),
        "close_location": round(close_location, 4),
        "upper_confirm_count": upper_confirm_count,
        "lower_confirm_count": lower_confirm_count,
        "confirm_bars": confirms,
        "lookback_bars": bars,
    }


def _rebound_potential_signal(
    features: pd.DataFrame,
    *,
    current_price: float,
    entry_price: float = 0.0,
    stop_loss: float = 0.0,
    enabled: bool = True,
    min_score: float = 60.0,
    min_samples: int = 5,
    lookback_bars: int = 160,
    forward_bars: int = 8,
) -> dict[str, object]:
    if not enabled:
        return {"active": False, "reason": ["REBOUND_PROTECTION_DISABLED"]}
    if features.empty or current_price <= 0 or "Close" not in features:
        return {"active": False, "reason": ["REBOUND_DATA_UNAVAILABLE"]}

    df = features.copy().tail(max(40, int(lookback_bars) + int(forward_bars) + 25))
    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df.get("High", close), errors="coerce")
    low = pd.to_numeric(df.get("Low", close), errors="coerce")
    volume = pd.to_numeric(df.get("Volume", pd.Series([0.0] * len(df), index=df.index)), errors="coerce")
    valid = pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": volume}).dropna()
    if len(valid) < max(20, int(forward_bars) + int(min_samples) + 5):
        return {
            "active": False,
            "reason": ["REBOUND_NOT_ENOUGH_HISTORY"],
            "sample_count": 0,
            "required_samples": int(min_samples),
        }

    close = valid["Close"]
    high = valid["High"]
    low = valid["Low"]
    volume = valid["Volume"]
    typical = (high + low + close) / 3
    rolling_volume = volume.rolling(60, min_periods=10).sum().replace(0, pd.NA)
    fallback_vwap = (typical * volume).rolling(60, min_periods=10).sum() / rolling_volume
    vwap = _series_or_default(df.reindex(valid.index), "vwap_60", fallback_vwap).fillna(fallback_vwap).fillna(close)
    ema21 = _series_or_default(df.reindex(valid.index), "ema_21", close.ewm(span=21, adjust=False).mean()).fillna(
        close.ewm(span=21, adjust=False).mean()
    )
    support = _series_or_default(
        df.reindex(valid.index),
        "support_20_prev",
        low.rolling(20, min_periods=5).min().shift(1),
    ).fillna(low.rolling(20, min_periods=5).min().shift(1)).fillna(low)
    rolling_high = high.rolling(20, min_periods=5).max().shift(1).fillna(high.rolling(5, min_periods=1).max())
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = _series_or_default(df.reindex(valid.index), "atr_14", tr.rolling(14, min_periods=5).mean()).fillna(
        tr.rolling(14, min_periods=5).mean()
    ).fillna((high - low).rolling(5, min_periods=1).mean())
    atr_pct = (atr / close.replace(0, pd.NA) * 100).fillna(0.3).clip(lower=0.05)
    volume_ratio = _series_or_default(
        df.reindex(valid.index),
        "volume_ratio",
        volume / volume.rolling(20, min_periods=5).mean().replace(0, pd.NA),
    ).fillna(0.0)
    rsi = _series_or_default(df.reindex(valid.index), "rsi_14", pd.Series([50.0] * len(valid), index=valid.index)).fillna(50.0)

    drawdown_pct = (close / rolling_high.replace(0, pd.NA) - 1) * 100
    support_gap_pct = (close / support.replace(0, pd.NA) - 1) * 100
    vwap_gap_pct = (close / vwap.replace(0, pd.NA) - 1) * 100
    ema_gap_pct = (close / ema21.replace(0, pd.NA) - 1) * 100

    latest_close = float(close.iloc[-1])
    latest_drawdown = float(drawdown_pct.iloc[-1]) if pd.notna(drawdown_pct.iloc[-1]) else 0.0
    latest_support_gap = float(support_gap_pct.iloc[-1]) if pd.notna(support_gap_pct.iloc[-1]) else 0.0
    latest_vwap_gap = float(vwap_gap_pct.iloc[-1]) if pd.notna(vwap_gap_pct.iloc[-1]) else 0.0
    latest_ema_gap = float(ema_gap_pct.iloc[-1]) if pd.notna(ema_gap_pct.iloc[-1]) else 0.0
    latest_atr_pct = max(0.05, float(atr_pct.iloc[-1]) if pd.notna(atr_pct.iloc[-1]) else 0.3)
    latest_volume_ratio = max(0.0, float(volume_ratio.iloc[-1]) if pd.notna(volume_ratio.iloc[-1]) else 0.0)
    latest_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0

    forward = max(2, int(forward_bars))
    sample_end = len(valid) - forward - 1
    if sample_end <= 5:
        return {"active": False, "reason": ["REBOUND_NOT_ENOUGH_FORWARD_HISTORY"], "sample_count": 0}

    history = pd.DataFrame(
        {
            "close": close,
            "high": high,
            "drawdown_pct": drawdown_pct,
            "support_gap_pct": support_gap_pct,
            "vwap_gap_pct": vwap_gap_pct,
            "ema_gap_pct": ema_gap_pct,
            "rsi": rsi,
        }
    ).iloc[: sample_end + 1].dropna()
    if history.empty:
        return {"active": False, "reason": ["REBOUND_PATTERN_HISTORY_EMPTY"], "sample_count": 0}

    similarity = (
        (history["drawdown_pct"] - latest_drawdown).abs() <= max(0.7, latest_atr_pct * 1.5)
    ) & ((history["support_gap_pct"] - latest_support_gap).abs() <= max(0.8, latest_atr_pct * 2.0))
    similarity &= (history["vwap_gap_pct"] - latest_vwap_gap).abs() <= max(0.9, latest_atr_pct * 2.0)
    similarity &= (history["rsi"] - latest_rsi).abs() <= 18
    samples = history[similarity]

    if len(samples) < int(min_samples):
        relaxed = history[
            (history["drawdown_pct"] <= 0.5)
            & (history["drawdown_pct"] >= -4.0)
            & (history["support_gap_pct"] >= -1.0)
            & (history["support_gap_pct"] <= 2.2)
            & (history["rsi"] >= 28)
            & (history["rsi"] <= 65)
        ]
        samples = relaxed

    max_returns: list[float] = []
    close_returns: list[float] = []
    threshold_pct = min(1.5, max(0.35, latest_atr_pct * 0.7))
    for idx in samples.index:
        pos = valid.index.get_loc(idx)
        if not isinstance(pos, int) or pos + forward >= len(valid):
            continue
        base = float(close.iloc[pos])
        if base <= 0:
            continue
        future_high = float(high.iloc[pos + 1 : pos + forward + 1].max())
        future_close = float(close.iloc[pos + forward])
        max_returns.append((future_high / base - 1) * 100)
        close_returns.append((future_close / base - 1) * 100)

    sample_count = len(max_returns)
    if sample_count < int(min_samples):
        return {
            "active": False,
            "reason": ["REBOUND_SAMPLE_COUNT_BELOW_MINIMUM"],
            "sample_count": sample_count,
            "required_samples": int(min_samples),
            "threshold_pct": round(threshold_pct, 4),
        }

    hit_rate = sum(1 for value in max_returns if value >= threshold_pct) / sample_count * 100
    avg_max_return = sum(max_returns) / sample_count
    avg_close_return = sum(close_returns) / sample_count
    support_score = max(0.0, 100 - abs(latest_support_gap) * 45)
    vwap_score = max(0.0, 100 - abs(latest_vwap_gap) * 35)
    ema_score = max(0.0, 100 - abs(latest_ema_gap) * 35)
    pullback_score = max(0.0, 100 - abs(min(latest_drawdown, 0.0)) * 22)
    rsi_score = 100.0 if 34 <= latest_rsi <= 58 else max(0.0, 100 - abs(latest_rsi - 46) * 3)
    volume_score = min(100.0, latest_volume_ratio / 0.8 * 100) if latest_volume_ratio > 0 else 35.0
    structure_score = (
        support_score * 0.24
        + vwap_score * 0.2
        + ema_score * 0.16
        + pullback_score * 0.16
        + rsi_score * 0.14
        + volume_score * 0.1
    )
    history_score = min(100.0, hit_rate * 0.75 + max(0.0, avg_max_return) / max(threshold_pct * 2.0, 0.5) * 25)
    score = history_score * 0.58 + structure_score * 0.42
    stop_gap_pct = (current_price / stop_loss - 1) * 100 if current_price > 0 and stop_loss > 0 else 999.0
    entry_return_pct = (current_price / entry_price - 1) * 100 if current_price > 0 and entry_price > 0 else 0.0

    reasons: list[str] = []
    if hit_rate < 45:
        reasons.append("HISTORICAL_REBOUND_HIT_RATE_LOW")
    if avg_max_return < threshold_pct:
        reasons.append("HISTORICAL_AVG_REBOUND_WEAK")
    if structure_score < 50:
        reasons.append("CURRENT_STRUCTURE_NOT_SUPPORTIVE")
    if score < float(min_score):
        reasons.append("REBOUND_SCORE_BELOW_MINIMUM")
    if stop_gap_pct <= 0:
        reasons.append("PRICE_ALREADY_AT_OR_BELOW_STOP")

    return {
        "active": not reasons,
        "reason": ["REBOUND_POTENTIAL_PRESENT"] if not reasons else reasons,
        "score": round(score, 2),
        "min_score": round(float(min_score), 2),
        "sample_count": sample_count,
        "required_samples": int(min_samples),
        "hit_rate_pct": round(hit_rate, 2),
        "threshold_pct": round(threshold_pct, 4),
        "avg_max_return_pct": round(avg_max_return, 4),
        "avg_close_return_pct": round(avg_close_return, 4),
        "history_score": round(history_score, 2),
        "structure_score": round(structure_score, 2),
        "drawdown_from_20_high_pct": round(latest_drawdown, 4),
        "support_gap_pct": round(latest_support_gap, 4),
        "vwap_gap_pct": round(latest_vwap_gap, 4),
        "ema21_gap_pct": round(latest_ema_gap, 4),
        "volume_ratio": round(latest_volume_ratio, 4),
        "rsi": round(latest_rsi, 4),
        "stop_gap_pct": round(stop_gap_pct, 4),
        "entry_return_pct": round(entry_return_pct, 4),
        "lookback_bars": int(lookback_bars),
        "forward_bars": forward,
    }


def _target_decay_exit_signal(
    features: pd.DataFrame,
    result: Any,
    *,
    current_price: float,
    entry_price: float,
    fee_adjusted_return_pct: float | None = None,
    stop_loss: float,
    target_price: float,
    enabled: bool,
    min_probability_pct: float,
    min_target_gap_pct: float,
    max_volume_ratio: float,
    max_volume_zscore: float,
    min_weak_bars: int,
    max_fee_adjusted_return_pct: float = -0.15,
    rebound_protection_enabled: bool = True,
    rebound_min_score: float = 60.0,
    rebound_min_samples: int = 5,
    rebound_lookback_bars: int = 160,
    rebound_forward_bars: int = 8,
) -> dict[str, object]:
    if not enabled:
        return {"active": False, "reason": ["TARGET_DECAY_EXIT_DISABLED"]}
    if features.empty or current_price <= 0 or target_price <= 0:
        return {"active": False, "reason": ["TARGET_DECAY_EXIT_DATA_UNAVAILABLE"]}

    weak_bars = max(1, int(min_weak_bars))
    if len(features) < weak_bars + 1:
        return {
            "active": False,
            "reason": ["TARGET_DECAY_EXIT_NOT_ENOUGH_BARS"],
            "available_bars": int(len(features)),
            "required_bars": int(weak_bars + 1),
        }

    latest = features.iloc[-1]
    recent = features.tail(weak_bars + 1)
    closes = pd.to_numeric(recent["Close"], errors="coerce").dropna()
    if len(closes) < weak_bars + 1:
        return {"active": False, "reason": ["TARGET_DECAY_EXIT_CLOSE_DATA_UNAVAILABLE"]}

    target_gap_pct = (target_price / current_price - 1) * 100 if current_price > 0 else 0.0
    stop_gap_pct = (current_price / stop_loss - 1) * 100 if current_price > 0 and stop_loss > 0 else 999.0
    entry_return_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0.0
    fee_adjusted_pct = (
        _as_float(fee_adjusted_return_pct, entry_return_pct)
        if fee_adjusted_return_pct is not None
        else entry_return_pct
    )
    probability_pct = _prediction_probability_pct(result)
    surge = _surge_metrics(features, min(max(weak_bars + 1, 3), 8))
    volume_ratio = max(_as_float(latest.get("volume_ratio"), _as_float(surge.get("volume_ratio"), 0.0)), 0.0)
    volume_zscore = _as_float(latest.get("volume_zscore_20"), _as_float(surge.get("volume_zscore"), 0.0))
    latest_bar_change_pct = _as_float(surge.get("latest_bar_change_pct"))
    recent_change_pct = (float(closes.iloc[-1]) / float(closes.iloc[0]) - 1) * 100 if float(closes.iloc[0]) > 0 else 0.0

    returns = closes.pct_change().dropna() * 100
    negative_bar_count = int((returns <= 0).sum())
    highs = pd.to_numeric(recent.get("High", pd.Series(dtype=float)), errors="coerce").dropna()
    lows = pd.to_numeric(recent.get("Low", pd.Series(dtype=float)), errors="coerce").dropna()
    lower_highs = bool(len(highs) >= weak_bars + 1 and all(float(highs.iloc[i]) <= float(highs.iloc[i - 1]) for i in range(1, len(highs))))
    lower_lows = bool(len(lows) >= weak_bars + 1 and all(float(lows.iloc[i]) <= float(lows.iloc[i - 1]) for i in range(1, len(lows))))

    probability_weak = probability_pct <= float(min_probability_pct)
    target_far_enough = target_gap_pct >= float(min_target_gap_pct)
    volume_weak = volume_ratio <= float(max_volume_ratio) or volume_zscore <= float(max_volume_zscore)
    momentum_weak = (
        negative_bar_count >= weak_bars
        or recent_change_pct <= -0.15
        or latest_bar_change_pct <= -0.1
        or (lower_highs and lower_lows)
    )
    pnl_weak = fee_adjusted_pct <= float(max_fee_adjusted_return_pct)
    risk_reward_weak = (stop_gap_pct <= target_gap_pct or entry_return_pct <= 0) and pnl_weak
    rebound_potential = _rebound_potential_signal(
        features,
        current_price=current_price,
        entry_price=entry_price,
        stop_loss=stop_loss,
        enabled=rebound_protection_enabled,
        min_score=rebound_min_score,
        min_samples=rebound_min_samples,
        lookback_bars=rebound_lookback_bars,
        forward_bars=rebound_forward_bars,
    )

    failed: list[str] = []
    if not probability_weak:
        failed.append("TARGET_PROBABILITY_STILL_OK")
    if not target_far_enough:
        failed.append("TARGET_ALREADY_CLOSE")
    if not volume_weak:
        failed.append("VOLUME_NOT_WEAK")
    if not momentum_weak:
        failed.append("MOMENTUM_NOT_WEAK")
    if not risk_reward_weak:
        failed.append("DISTANCE_TO_STOP_NOT_DOMINANT")
    if not pnl_weak:
        failed.append("POSITION_NOT_WEAK_AFTER_FEES")
    if bool(rebound_potential.get("active")):
        failed.append("REBOUND_POTENTIAL_PRESENT")

    return {
        "active": not failed,
        "reason": ["TARGET_DECAY_EXIT_ACTIVE"] if not failed else failed,
        "probability_pct": round(probability_pct, 4),
        "min_probability_pct": round(float(min_probability_pct), 4),
        "target_gap_pct": round(target_gap_pct, 4),
        "min_target_gap_pct": round(float(min_target_gap_pct), 4),
        "stop_gap_pct": round(stop_gap_pct, 4),
        "entry_return_pct": round(entry_return_pct, 4),
        "fee_adjusted_return_pct": round(fee_adjusted_pct, 4),
        "max_fee_adjusted_return_pct": round(float(max_fee_adjusted_return_pct), 4),
        "volume_ratio": round(volume_ratio, 4),
        "max_volume_ratio": round(float(max_volume_ratio), 4),
        "volume_zscore": round(volume_zscore, 4),
        "max_volume_zscore": round(float(max_volume_zscore), 4),
        "recent_change_pct": round(recent_change_pct, 4),
        "latest_bar_change_pct": round(latest_bar_change_pct, 4),
        "negative_bar_count": negative_bar_count,
        "min_weak_bars": weak_bars,
        "lower_highs": lower_highs,
        "lower_lows": lower_lows,
        "risk_reward_weak": bool(risk_reward_weak),
        "rebound_potential": rebound_potential,
    }


def _target_extension_signal(
    features: pd.DataFrame,
    *,
    current_price: float,
    target_price: float,
    enabled: bool,
    max_extension_pct: float,
    trailing_stop_pct: float,
    min_surge_score: float,
    min_lookback_change_pct: float,
    min_latest_bar_change_pct: float,
    min_volume_ratio: float,
    max_rsi: float,
    lookback_bars: int = 5,
) -> dict[str, object]:
    if not enabled:
        return {"active": False, "reason": ["TARGET_EXTENSION_DISABLED"]}
    if features.empty or current_price <= 0 or target_price <= 0:
        return {"active": False, "reason": ["TARGET_EXTENSION_DATA_UNAVAILABLE"]}

    max_extension_pct = max(0.0, float(max_extension_pct))
    if max_extension_pct <= 0:
        return {"active": False, "reason": ["TARGET_EXTENSION_MAX_ZERO"]}

    target_overrun_pct = (current_price / target_price - 1) * 100
    if target_overrun_pct >= max_extension_pct:
        return {
            "active": False,
            "reason": ["TARGET_EXTENSION_CAP_REACHED"],
            "target_overrun_pct": round(target_overrun_pct, 4),
        }

    latest = features.iloc[-1]
    surge = _surge_metrics(features, lookback_bars)
    latest_close = _as_float(latest.get("Close"), current_price)
    ema_9 = _as_float(latest.get("ema_9"), latest_close)
    ema_21 = _as_float(latest.get("ema_21"), ema_9)
    volume_ratio = max(_as_float(latest.get("volume_ratio"), _as_float(surge.get("volume_ratio"), 0.0)), 0.0)
    volume_zscore = _as_float(latest.get("volume_zscore_20"), _as_float(surge.get("volume_zscore"), 0.0))
    rsi = _as_float(latest.get("rsi_14"), 50.0)

    momentum_ok = (
        _as_float(surge.get("surge_score")) >= float(min_surge_score)
        or _as_float(surge.get("lookback_change_pct")) >= float(min_lookback_change_pct)
    )
    latest_ok = _as_float(surge.get("latest_bar_change_pct")) >= float(min_latest_bar_change_pct)
    volume_ok = volume_ratio >= float(min_volume_ratio) or volume_zscore >= 0.5
    trend_ok = latest_close >= ema_9 and ema_9 >= ema_21
    rsi_ok = rsi <= float(max_rsi)

    failed = []
    if not momentum_ok:
        failed.append("MOMENTUM_NOT_STRONG")
    if not latest_ok:
        failed.append("LATEST_BAR_WEAK")
    if not volume_ok:
        failed.append("VOLUME_CONFIRMATION_WEAK")
    if not trend_ok:
        failed.append("TREND_NOT_CONFIRMED")
    if not rsi_ok:
        failed.append("RSI_EXTENSION_TOO_HOT")

    extended_target = target_price * (1 + max_extension_pct / 100)
    protective_stop = max(target_price * (1 - max(0.0, float(trailing_stop_pct)) / 100), 0.0)
    return {
        "active": not failed,
        "reason": ["TARGET_EXTENSION_ACTIVE"] if not failed else failed,
        "original_target_price": round(target_price, 4),
        "extended_target_price": round(extended_target, 4),
        "protective_stop_price": round(protective_stop, 4),
        "max_extension_pct": round(max_extension_pct, 4),
        "trailing_stop_pct": round(max(0.0, float(trailing_stop_pct)), 4),
        "target_overrun_pct": round(target_overrun_pct, 4),
        "surge_score": round(_as_float(surge.get("surge_score")), 4),
        "lookback_change_pct": round(_as_float(surge.get("lookback_change_pct")), 4),
        "latest_bar_change_pct": round(_as_float(surge.get("latest_bar_change_pct")), 4),
        "volume_ratio": round(volume_ratio, 4),
        "volume_zscore": round(volume_zscore, 4),
        "rsi": round(rsi, 4),
        "trend_ok": bool(trend_ok),
    }


def _build_exit_trade_intent(
    *,
    symbol: str,
    quantity: int,
    current_price: float,
    stop_loss: float,
    take_profit: list[float],
    account_usd: float,
    daily_realized_pnl_usd: float,
    daily_loss_limit_pct: float,
    daily_loss_limit_usd: float,
    data_quality: dict[str, object],
    reason: str,
    holding_qty: float,
    current_session_entries: int,
    max_session_entries: int,
    manual_approval_complete: bool,
) -> TradeIntent:
    pct_loss_limit = account_usd * daily_loss_limit_pct / 100 if account_usd > 0 else 0
    active_loss_limits = [limit for limit in [pct_loss_limit, daily_loss_limit_usd] if limit and limit > 0]
    effective_daily_limit = min(active_loss_limits) if active_loss_limits else 0
    daily_loss_used = abs(min(daily_realized_pnl_usd, 0))
    daily_loss_remaining = max(effective_daily_limit - daily_loss_used, 0) if effective_daily_limit > 0 else 0
    reasons = [reason]
    if quantity <= 0:
        reasons.append("NO_WHOLE_SHARE_SELL_QUANTITY")
    if not manual_approval_complete:
        reasons.append("MANUAL_APPROVAL_REQUIRED")
    if bool(data_quality.get("stale_blocked", False)):
        reasons.append("STALE_DATA_BLOCKED")
    if bool(data_quality.get("spread_blocked", False)):
        reasons.append("WIDE_SPREAD_BLOCKED")

    safe_current_price = max(_as_float(current_price), 0.0001)
    safe_stop_loss = max(_as_float(stop_loss), 0.0001)
    take_profit_1 = max(_as_float(take_profit[0] if take_profit else 0.0), safe_current_price)
    take_profit_2 = max(_as_float(take_profit[1] if len(take_profit) > 1 else take_profit_1), take_profit_1)
    duplicate_key = f"{datetime.now().date().isoformat()}:{symbol}:SELL:EXIT:{reason}"
    return TradeIntent(
        symbol=symbol,
        side="SELL",
        quantity=max(0, int(quantity)),
        limit_price=round(safe_current_price, 4),
        stop_loss=round(safe_stop_loss, 4),
        take_profit_1=round(take_profit_1, 4),
        take_profit_2=round(take_profit_2, 4),
        status="DRY_RUN_READY" if quantity > 0 and len(reasons) == 1 else "BLOCKED",
        reason=reasons,
        risk_checks={
            "account_usd": round(account_usd, 2),
            "daily_realized_pnl_usd": round(daily_realized_pnl_usd, 2),
            "daily_loss_limit_usd": round(effective_daily_limit, 2),
            "daily_loss_used_usd": round(daily_loss_used, 2),
            "daily_loss_remaining_usd": round(daily_loss_remaining, 2),
            "expected_trade_risk_usd": 0.0,
            "duplicate_blocked": False,
            "manual_approval_complete": manual_approval_complete,
            "holding_qty": holding_qty,
            "sell_quantity_capped": quantity < _whole_share_quantity(holding_qty),
            "current_session_entries": current_session_entries,
            "max_session_entries": max_session_entries,
            "exit_reason": reason,
            **{
                "data_guard_enabled": bool(data_quality.get("guard_enabled", False)),
                "data_age_minutes": data_quality.get("age_minutes"),
                "stale_data_blocked": bool(data_quality.get("stale_blocked", False)),
                "spread_pct": data_quality.get("spread_pct"),
                "wide_spread_blocked": bool(data_quality.get("spread_blocked", False)),
            },
        },
        duplicate_key=duplicate_key,
        safety_notice="Exit intent created from real holdings. Broker submit still requires the execution gate.",
    )


def _buy_candidate_scanner(scanner: pd.DataFrame) -> pd.DataFrame:
    if scanner.empty or "liquidity_filter_bypassed" not in scanner.columns:
        return scanner
    return scanner[~scanner["liquidity_filter_bypassed"].fillna(False)].copy()


def _liquidity_passed_count(scanner: pd.DataFrame) -> int:
    if scanner.empty:
        return 0
    if "liquidity_filter_bypassed" not in scanner.columns:
        return int(len(scanner))
    return int((~scanner["liquidity_filter_bypassed"].fillna(False)).sum())


def _select_exit_candidate(
    loaded: dict[str, tuple[pd.DataFrame, pd.DataFrame, Any, Any, pd.DataFrame]],
    holding_quantities: dict[str, float],
    *,
    tolerance_pct: float,
    position_summary: list[dict[str, object]] | None = None,
    manual_position_exit_enabled: bool = True,
    manual_take_profit_pct: float = 2.0,
    manual_stop_loss_pct: float = 2.0,
    target_extension_enabled: bool = False,
    target_extension_max_pct: float = 0.6,
    target_extension_trailing_stop_pct: float = 0.35,
    target_extension_min_surge_score: float = 8.0,
    target_extension_min_lookback_change_pct: float = 0.4,
    target_extension_min_latest_bar_change_pct: float = 0.0,
    target_extension_min_volume_ratio: float = 1.1,
    target_extension_max_rsi: float = 84.0,
    target_decay_exit_enabled: bool = True,
    target_decay_exit_min_probability_pct: float = 40.0,
    target_decay_exit_min_target_gap_pct: float = 0.8,
    target_decay_exit_max_volume_ratio: float = 0.7,
    target_decay_exit_max_volume_zscore: float = -0.35,
    target_decay_exit_min_weak_bars: int = 2,
    target_decay_exit_max_fee_adjusted_return_pct: float = -0.15,
    target_decay_rebound_protection_enabled: bool = True,
    target_decay_rebound_min_score: float = 60.0,
    target_decay_rebound_min_samples: int = 5,
    target_decay_rebound_lookback_bars: int = 160,
    target_decay_rebound_forward_bars: int = 8,
    support_breakdown_exit_enabled: bool = True,
    structure_lookback_bars: int = 20,
    structure_confirm_bars: int = 2,
    structure_min_break_pct: float = 0.15,
    structure_atr_buffer_multiplier: float = 0.25,
    structure_min_volume_ratio: float = 1.0,
    structure_min_volume_zscore: float = -0.2,
) -> dict[str, object] | None:
    candidates: list[dict[str, object]] = []
    positions = {
        normalize_ticker(str(item.get("symbol") or "")): item
        for item in (position_summary or [])
        if isinstance(item, dict) and item.get("symbol")
    }
    for symbol, holding_qty in holding_quantities.items():
        whole_qty = _whole_share_quantity(holding_qty)
        if whole_qty <= 0:
            continue
        position = positions.get(symbol, {})
        position_current_price = _as_float(position.get("current_price") if isinstance(position, dict) else None)
        entry_price = _as_float(position.get("entry_price") if isinstance(position, dict) else None)
        fee_adjusted_return_pct = _as_float(position.get("fee_adjusted_return_pct") if isinstance(position, dict) else None)
        manual_stop_loss = 0.0
        manual_target = 0.0

        if manual_position_exit_enabled and entry_price > 0 and position_current_price > 0:
            manual_stop_loss = entry_price * (1 - abs(float(manual_stop_loss_pct)) / 100)
            manual_target = entry_price * (1 + abs(float(manual_take_profit_pct)) / 100)
            reason = ""
            if fee_adjusted_return_pct <= -abs(float(manual_stop_loss_pct)):
                reason = "MANUAL_POSITION_STOP_LOSS"
            elif fee_adjusted_return_pct >= abs(float(manual_take_profit_pct)):
                reason = "MANUAL_POSITION_TAKE_PROFIT"
            if reason:
                if reason == "MANUAL_POSITION_TAKE_PROFIT" and symbol in loaded:
                    _prices, features, _result, _plan, _model_data = loaded[symbol]
                    extension = _target_extension_signal(
                        features,
                        current_price=position_current_price,
                        target_price=manual_target,
                        enabled=target_extension_enabled,
                        max_extension_pct=target_extension_max_pct,
                        trailing_stop_pct=target_extension_trailing_stop_pct,
                        min_surge_score=target_extension_min_surge_score,
                        min_lookback_change_pct=target_extension_min_lookback_change_pct,
                        min_latest_bar_change_pct=target_extension_min_latest_bar_change_pct,
                        min_volume_ratio=target_extension_min_volume_ratio,
                        max_rsi=target_extension_max_rsi,
                    )
                    if bool(extension.get("active")):
                        candidates.append(
                            {
                                "symbol": symbol,
                                "reason": "TARGET_EXTENSION_ACTIVE",
                                "quantity": 0,
                                "holding_qty": holding_qty,
                                "current_price": position_current_price,
                                "entry_price": entry_price,
                                "stop_loss": _as_float(extension.get("protective_stop_price"), manual_stop_loss),
                                "target_price": _as_float(extension.get("extended_target_price"), manual_target),
                                "original_target_price": manual_target,
                                "target_trigger": _as_float(extension.get("extended_target_price"), manual_target),
                                "target_progress": position_current_price / _as_float(extension.get("extended_target_price"), manual_target)
                                if _as_float(extension.get("extended_target_price"), manual_target) > 0
                                else 0.0,
                                "tolerance_pct": 0.0,
                                "fee_adjusted_return_pct": fee_adjusted_return_pct,
                                "fee_adjusted_pnl_usd": _as_float(position.get("fee_adjusted_pnl_usd") if isinstance(position, dict) else None),
                                "manual_exit": True,
                                "manual_take_profit_pct": abs(float(manual_take_profit_pct)),
                                "manual_stop_loss_pct": abs(float(manual_stop_loss_pct)),
                                "target_extension": extension,
                            }
                        )
                        continue
                candidates.append(
                    {
                        "symbol": symbol,
                        "reason": reason,
                        "quantity": whole_qty,
                        "holding_qty": holding_qty,
                        "current_price": position_current_price,
                        "entry_price": entry_price,
                        "stop_loss": manual_stop_loss,
                        "target_price": manual_target,
                        "target_trigger": manual_target,
                        "target_progress": position_current_price / manual_target if manual_target > 0 else 0.0,
                        "tolerance_pct": 0.0,
                        "fee_adjusted_return_pct": fee_adjusted_return_pct,
                        "fee_adjusted_pnl_usd": _as_float(position.get("fee_adjusted_pnl_usd") if isinstance(position, dict) else None),
                        "manual_exit": True,
                        "manual_take_profit_pct": abs(float(manual_take_profit_pct)),
                        "manual_stop_loss_pct": abs(float(manual_stop_loss_pct)),
                    }
                )

        if symbol not in loaded:
            continue
        _prices, features, result, plan, _model_data = loaded[symbol]
        latest_close = _as_float(features.iloc[-1].get("Close") if not features.empty else None)
        current_price = position_current_price or _as_float(getattr(plan, "price", 0.0), latest_close) or latest_close
        stop_loss = _as_float(getattr(plan, "stop_loss", 0.0))
        target_price = _as_float(getattr(plan, "take_profit_1", 0.0))
        if manual_stop_loss > 0 and (stop_loss <= 0 or manual_stop_loss < stop_loss):
            stop_loss = manual_stop_loss
        if manual_target > target_price:
            target_price = manual_target
        support_resistance = _support_resistance_signal(
            features,
            current_price=current_price,
            enabled=support_breakdown_exit_enabled,
            lookback_bars=structure_lookback_bars,
            confirm_bars=structure_confirm_bars,
            min_break_pct=structure_min_break_pct,
            atr_buffer_multiplier=structure_atr_buffer_multiplier,
            min_volume_ratio=structure_min_volume_ratio,
            min_volume_zscore=structure_min_volume_zscore,
        )
        reason = _exit_reason(current_price, stop_loss, target_price, tolerance_pct)
        if not reason and bool(support_resistance.get("lower_breakdown")):
            reason = "SUPPORT_BREAKDOWN_EXIT"
        if reason:
            extension: dict[str, object] = {}
            quantity = whole_qty
            if reason == "TARGET_WITHIN_TOLERANCE":
                extension = _target_extension_signal(
                    features,
                    current_price=current_price,
                    target_price=target_price,
                    enabled=target_extension_enabled,
                    max_extension_pct=target_extension_max_pct,
                    trailing_stop_pct=target_extension_trailing_stop_pct,
                    min_surge_score=target_extension_min_surge_score,
                    min_lookback_change_pct=target_extension_min_lookback_change_pct,
                    min_latest_bar_change_pct=target_extension_min_latest_bar_change_pct,
                    min_volume_ratio=target_extension_min_volume_ratio,
                    max_rsi=target_extension_max_rsi,
                )
                if bool(extension.get("active")):
                    reason = "TARGET_EXTENSION_ACTIVE"
                    quantity = 0
                    stop_loss = max(stop_loss, _as_float(extension.get("protective_stop_price"), stop_loss))
                    target_price = _as_float(extension.get("extended_target_price"), target_price)
            target_progress = current_price / target_price if target_price > 0 else 0.0
            candidates.append(
                {
                    "symbol": symbol,
                    "reason": reason,
                    "quantity": quantity,
                    "holding_qty": holding_qty,
                    "current_price": current_price,
                    "stop_loss": stop_loss,
                    "target_price": target_price,
                    "original_target_price": _as_float(extension.get("original_target_price"), target_price) if extension else target_price,
                    "target_trigger": target_price
                    if reason == "TARGET_EXTENSION_ACTIVE"
                    else _target_exit_trigger(current_price, target_price, tolerance_pct),
                    "target_progress": target_progress,
                    "tolerance_pct": float(tolerance_pct),
                    "manual_exit": False,
                    "target_extension": extension,
                    "support_resistance": support_resistance,
                }
            )
        else:
            target_decay_exit = _target_decay_exit_signal(
                features,
                result,
                current_price=current_price,
                entry_price=entry_price,
                fee_adjusted_return_pct=fee_adjusted_return_pct,
                stop_loss=stop_loss,
                target_price=target_price,
                enabled=target_decay_exit_enabled,
                min_probability_pct=target_decay_exit_min_probability_pct,
                min_target_gap_pct=target_decay_exit_min_target_gap_pct,
                max_volume_ratio=target_decay_exit_max_volume_ratio,
                max_volume_zscore=target_decay_exit_max_volume_zscore,
                min_weak_bars=target_decay_exit_min_weak_bars,
                max_fee_adjusted_return_pct=target_decay_exit_max_fee_adjusted_return_pct,
                rebound_protection_enabled=target_decay_rebound_protection_enabled,
                rebound_min_score=target_decay_rebound_min_score,
                rebound_min_samples=target_decay_rebound_min_samples,
                rebound_lookback_bars=target_decay_rebound_lookback_bars,
                rebound_forward_bars=target_decay_rebound_forward_bars,
            )
            if bool(target_decay_exit.get("active")):
                target_progress = current_price / target_price if target_price > 0 else 0.0
                candidates.append(
                    {
                        "symbol": symbol,
                        "reason": "EARLY_EXIT_TARGET_PROBABILITY_DECAY",
                        "quantity": whole_qty,
                        "holding_qty": holding_qty,
                        "current_price": current_price,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "target_price": target_price,
                        "original_target_price": target_price,
                        "target_trigger": current_price,
                        "target_progress": target_progress,
                        "tolerance_pct": 0.0,
                        "fee_adjusted_return_pct": fee_adjusted_return_pct,
                        "fee_adjusted_pnl_usd": _as_float(position.get("fee_adjusted_pnl_usd") if isinstance(position, dict) else None),
                        "manual_exit": False,
                        "target_decay_exit": target_decay_exit,
                    }
                )
    if not candidates:
        return None
    priority = {
        "MANUAL_POSITION_STOP_LOSS": 0,
        "STOP_LOSS_REACHED": 1,
        "SUPPORT_BREAKDOWN_EXIT": 2,
        "MANUAL_POSITION_TAKE_PROFIT": 3,
        "TARGET_WITHIN_TOLERANCE": 4,
        "EARLY_EXIT_TARGET_PROBABILITY_DECAY": 5,
        "TARGET_EXTENSION_ACTIVE": 6,
    }
    candidates.sort(
        key=lambda item: (
            priority.get(str(item.get("reason")), 9),
            -float(item.get("target_progress", 0.0)),
            -float(item.get("quantity", 0)),
        )
    )
    return candidates[0]


def apply_buying_power_sizing_harness(
    payload: dict[str, Any],
    *,
    account: AccountSnapshot | None,
    max_order_notional_usd: float,
    usdkrw: float,
) -> dict[str, Any]:
    guarded = dict(payload)
    if guarded.get("action") != "BUY_WATCH":
        return guarded

    reference_entry = float(guarded.get("reference_entry") or 0)
    stop_loss = float(guarded.get("stop_loss") or 0)
    if reference_entry <= 0:
        return guarded

    original_shares = max(0, int(guarded.get("recommended_shares") or 0))
    buying_power = max(0.0, float(account.cash_buying_power)) if account else 0.0
    order_cap = max(0.0, float(max_order_notional_usd))
    notional_limits = [limit for limit in (buying_power, order_cap) if limit > 0]
    if not notional_limits:
        return guarded

    effective_notional_limit = min(notional_limits)
    capped_shares = max(0, min(original_shares, int(effective_notional_limit // reference_entry)))
    risk_per_share = max(0.0, reference_entry - stop_loss)
    sizing = dict(guarded.get("position_sizing") or {})
    sizing.update(
        {
            "buying_power_usd": round(buying_power, 2),
            "max_order_notional_usd": round(order_cap, 2),
            "effective_notional_limit_usd": round(effective_notional_limit, 2),
            "original_recommended_shares": original_shares,
            "cash_capped_shares": capped_shares,
        }
    )
    guarded["position_sizing"] = sizing

    if capped_shares != original_shares:
        guarded["recommended_shares"] = capped_shares
        guarded["expected_loss_usd"] = round(capped_shares * risk_per_share, 2)
        guarded["expected_loss_krw"] = round(capped_shares * risk_per_share * max(0.0, float(usdkrw)), 0)
        guarded.setdefault("risk_flags", []).append("BUY_QUANTITY_CAPPED_TO_BUYING_POWER")
        if capped_shares <= 0 and original_shares > 0:
            guarded.setdefault("alerts", []).append("BUY_BLOCKED_BY_BUYING_POWER")

    return guarded


def _broker(*, allow_live_orders: bool) -> TossBrokerExecutionAdapter:
    token, client_id, client_secret = normalize_toss_auth(
        secret_value("TOSS_API_TOKEN"),
        secret_value("TOSS_CLIENT_ID"),
        secret_value("TOSS_CLIENT_SECRET"),
    )
    return TossBrokerExecutionAdapter(
        token=token,
        client_id=client_id,
        client_secret=client_secret,
        account_seq=secret_value("TOSS_ACCOUNT_SEQ") or None,
        base_url=secret_value("TOSS_BASE_URL", "https://openapi.tossinvest.com"),
        currency=os.getenv("TOSS_ORDER_CURRENCY", "USD"),
        allow_live_orders=allow_live_orders,
    )


def _daily_archive_paths() -> DailyArchivePaths:
    return DailyArchivePaths(
        session_events_path=SESSION_EVENT_PATH,
        auto_trade_events_path=AUTO_TRADE_EVENTS_PATH,
        order_intent_events_path=ORDER_LEDGER_PATH,
        signal_events_path=SIGNAL_EVENTS_PATH,
        archive_root=DAILY_ARCHIVE_ROOT,
        latest_summary_path=LATEST_DAILY_SUMMARY_PATH,
        learning_samples_path=STRATEGY_LEARNING_SAMPLES_PATH,
        strategy_adjustments_path=STRATEGY_ADJUSTMENTS_PATH,
        settings_path=SETTINGS_SNAPSHOT_PATH,
    )


def maybe_run_daily_archive(settings: dict[str, Any], *, force: bool = False) -> list[dict[str, Any]]:
    enabled = secret_bool("DAILY_ARCHIVE_ENABLED", _setting_bool(settings, "daily_archive_enabled", True))
    if not enabled:
        return []
    cutoff_hour = int(
        secret_float(
            "DAILY_ARCHIVE_CUTOFF_HOUR_KST",
            _setting_float(settings, "daily_archive_cutoff_hour_kst", 9.0),
        )
    )
    broker = None
    query_broker = secret_bool(
        "DAILY_ARCHIVE_QUERY_BROKER_ENABLED",
        _setting_bool(settings, "daily_archive_query_broker_enabled", True),
    )
    if query_broker:
        token, client_id, client_secret = normalize_toss_auth(
            secret_value("TOSS_API_TOKEN"),
            secret_value("TOSS_CLIENT_ID"),
            secret_value("TOSS_CLIENT_SECRET"),
        )
        if token or (client_id and client_secret):
            broker = _broker(allow_live_orders=False)
    return maybe_archive_completed_sessions(
        _daily_archive_paths(),
        settings=settings,
        broker=broker,
        now=datetime.now(timezone.utc),
        cutoff_hour_kst=cutoff_hour,
        force=force,
    )


def append_session_event(event: dict[str, Any]) -> None:
    payload = {
        "logged_at": _now_iso(),
        **event,
    }
    with SESSION_EVENT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def scan_candidates(
    tickers: list[str],
    *,
    interval: str,
    reward_ratio: float,
    toss_token: str,
    toss_client_id: str,
    toss_client_secret: str,
    toss_account_seq: str,
    toss_base_url: str,
    liquidity_recent_bars: int = 20,
    min_recent_volume: float = 0.0,
    min_recent_dollar_volume: float = 0.0,
    surge_lookback_bars: int = 5,
    always_include_symbols: set[str] | None = None,
    request_pause_seconds: float = 0.0,
) -> tuple[
    pd.DataFrame,
    dict[str, tuple[pd.DataFrame, pd.DataFrame, Any, Any, pd.DataFrame]],
    list[tuple[str, str]],
    list[dict[str, object]],
]:
    rows: list[dict[str, object]] = []
    loaded: dict[str, tuple[pd.DataFrame, pd.DataFrame, Any, Any, pd.DataFrame]] = {}
    failures: list[tuple[str, str]] = []
    liquidity_filtered: list[dict[str, object]] = []
    always_include_symbols = {normalize_ticker(symbol) for symbol in (always_include_symbols or set())}

    for ticker in tickers:
        try:
            prices = load_toss_price_data(
                ticker,
                toss_token,
                toss_client_id,
                toss_client_secret,
                toss_account_seq,
                toss_base_url,
                interval,
            )
            features = add_indicators(prices)
            liquidity = _liquidity_metrics(features, liquidity_recent_bars)
            liquidity_passed = _passes_liquidity_filter(
                liquidity,
                min_recent_volume=min_recent_volume,
                min_recent_dollar_volume=min_recent_dollar_volume,
            )
            liquidity_bypassed = False
            if not liquidity_passed:
                liquidity_filtered.append(
                    {
                        "symbol": ticker,
                        "recent_volume": round(liquidity["recent_volume"], 2),
                        "recent_dollar_volume": round(liquidity["recent_dollar_volume"], 2),
                    }
                )
                if ticker not in always_include_symbols:
                    continue
                liquidity_bypassed = True
            result, model_data = train_predictor(features, reward_ratio=reward_ratio)
            plan = build_trade_plan(features, result.probability_up, reward_ratio)
            row = score_snapshot(ticker, features, result, plan)
            surge = _surge_metrics(features, surge_lookback_bars)
            structure = _support_resistance_signal(features, current_price=_as_float(features.iloc[-1].get("Close")))
            row["recent_volume"] = round(liquidity["recent_volume"], 2)
            row["recent_dollar_volume"] = round(liquidity["recent_dollar_volume"], 2)
            row["latest_volume"] = round(liquidity["latest_volume"], 2)
            row["latest_dollar_volume"] = round(liquidity["latest_dollar_volume"], 2)
            row["recent_bars"] = int(liquidity_recent_bars)
            row["liquidity_filter_bypassed"] = bool(liquidity_bypassed)
            row["lookback_change_pct"] = surge["lookback_change_pct"]
            row["latest_bar_change_pct"] = surge["latest_bar_change_pct"]
            row["breakout_pct"] = surge["breakout_pct"]
            row["surge_volume_ratio"] = surge["volume_ratio"]
            row["surge_volume_zscore"] = surge["volume_zscore"]
            row["surge_vwap_distance_pct"] = surge["vwap_distance_pct"]
            row["surge_score"] = surge["surge_score"]
            row["structure_state"] = str(structure.get("state") or "")
            row["support"] = _as_float(structure.get("support"))
            row["resistance"] = _as_float(structure.get("resistance"))
            row["support_gap_pct"] = _as_float(structure.get("support_gap_pct"))
            row["resistance_gap_pct"] = _as_float(structure.get("resistance_gap_pct"))
            row["support_breakdown"] = bool(structure.get("lower_breakdown", False))
            row["resistance_breakout"] = bool(structure.get("upper_breakout", False))
            row["structure_break_buffer_pct"] = _as_float(structure.get("break_buffer_pct"))
            latest = features.iloc[-1]
            previous = features.iloc[-2]
            latest_close = _as_float(latest.get("Close"))
            previous_close = _as_float(previous.get("Close"), latest_close)
            trend_up = (
                _as_float(latest.get("ema_9")) > _as_float(latest.get("ema_21"))
                and latest_close > _as_float(latest.get("ma_20"))
            )
            if float(result.probability_up) >= 0.58 and trend_up:
                monitor_direction = "LONG"
            elif latest_close < _as_float(latest.get("ma_20")) or float(result.probability_up) <= 0.42:
                monitor_direction = "CAUTION"
            else:
                monitor_direction = "WAIT"
            row["symbol"] = ticker
            row["confidence_pct"] = round(float(result.probability_up) * 100, 4)
            row["direction"] = monitor_direction
            row["change_pct"] = round((latest_close / previous_close - 1) * 100, 4) if previous_close > 0 else 0.0
            row["volume_ratio"] = round(_as_float(latest.get("volume_ratio")), 4)
            row["volume_zscore"] = round(_as_float(latest.get("volume_zscore_20")), 4)
            row["rsi"] = round(_as_float(latest.get("rsi_14")), 4)
            row["risk"] = "NORMAL" if not plan.risk_flags else "FLAGGED"
            row["risk_flags_count"] = len(plan.risk_flags)
            row["auto_score"] = auto_selection_score(row)
            row["recent_volume_alias"] = round(liquidity["recent_volume"], 2)
            row["recent_dollar_volume_alias"] = round(liquidity["recent_dollar_volume"], 2)
            row["recent_bars_alias"] = int(liquidity_recent_bars)
            rows.append(row)
            loaded[ticker] = (prices, features, result, plan, model_data)
        except Exception as exc:
            failures.append((ticker, str(exc)))
        if request_pause_seconds > 0:
            time.sleep(float(request_pause_seconds))

    if not rows:
        return pd.DataFrame(), loaded, failures, liquidity_filtered
    return add_auto_selection_scores(pd.DataFrame(rows)), loaded, failures, liquidity_filtered


def run_cycle(settings: dict[str, Any], *, watchlist_override: str = "") -> dict[str, Any]:
    candidate_universe_mode = str(
        os.getenv("CANDIDATE_UNIVERSE_MODE") or _setting(settings, "candidate_universe_mode", "watchlist")
    ).strip().lower()
    if candidate_universe_mode in {"all", "liquidity", "volume_value", "all_liquid"}:
        tickers = _liquidity_universe(settings, watchlist_override)
    else:
        tickers = _watchlist(settings, watchlist_override)
    broker_blocked_symbols = _symbols_from_setting(os.getenv("BROKER_BLOCKED_SYMBOLS") or settings.get("broker_blocked_symbols"))
    if not tickers:
        raise ValueError("watchlist is empty")

    toss_token, toss_client_id, toss_client_secret = normalize_toss_auth(
        secret_value("TOSS_API_TOKEN"),
        secret_value("TOSS_CLIENT_ID"),
        secret_value("TOSS_CLIENT_SECRET"),
    )
    toss_account_seq = secret_value("TOSS_ACCOUNT_SEQ")
    toss_base_url = secret_value("TOSS_BASE_URL", "https://openapi.tossinvest.com")
    if not toss_token and not (toss_client_id and toss_client_secret):
        raise ValueError("Toss API token or client credentials are required")

    interval = str(_setting(settings, "interval", "1m") or "1m")
    reward_ratio = _setting_float(settings, "reward_ratio", 1.8)
    account_krw = _setting_float(settings, "account_krw", 1_000_000.0)
    usdkrw = _setting_float(settings, "usdkrw", 1527.0)
    risk_pct = _setting_float(settings, "risk_pct", 2.0)
    daily_realized_pnl_usd = _setting_float(settings, "daily_realized_pnl_usd", 0.0)
    daily_target_min_pct = secret_float("DAILY_TARGET_MIN_PCT", _setting_float(settings, "daily_target_min_pct", 10.0))
    daily_target_max_pct = secret_float("DAILY_TARGET_MAX_PCT", _setting_float(settings, "daily_target_max_pct", 20.0))
    daily_loss_limit_pct = secret_float("DAILY_LOSS_LIMIT_PCT", _setting_float(settings, "daily_loss_limit_pct", 10.0))
    daily_profit_lock_trigger_pct = secret_float(
        "DAILY_PROFIT_LOCK_TRIGGER_PCT",
        _setting_float(settings, "daily_profit_lock_trigger_pct", 20.0),
    )
    daily_profit_lock_giveback_pct = secret_float(
        "DAILY_PROFIT_LOCK_GIVEBACK_PCT",
        _setting_float(settings, "daily_profit_lock_giveback_pct", 10.0),
    )
    daily_loss_limit_usd = secret_float("DAILY_LOSS_LIMIT_USD", _setting_float(settings, "daily_loss_limit_usd", 0.0))
    data_guard_enabled = secret_bool("DATA_GUARD_ENABLED", _setting_bool(settings, "data_guard_enabled", False))
    max_data_age_minutes = _setting_float(settings, "max_data_age_minutes", 5.0)
    max_spread_pct = _setting_float(settings, "max_spread_pct", 2.0)
    holding_qty = _setting_int(settings, "holding_qty", 0)
    max_session_entries = _setting_int(settings, "max_session_entries", 3)
    paper_buying_power_usd = _setting_float(settings, "paper_buying_power_usd", 1000.0)
    max_auto_order_notional_usd = _setting_float(settings, "max_auto_order_notional_usd", 500.0)
    liquidity_recent_bars = int(
        secret_float("CANDIDATE_LIQUIDITY_RECENT_BARS", _setting_float(settings, "candidate_liquidity_recent_bars", 20.0))
    )
    min_recent_volume = secret_float(
        "CANDIDATE_MIN_RECENT_VOLUME",
        _setting_float(settings, "candidate_min_recent_volume", 50_000.0),
    )
    min_recent_dollar_volume = secret_float(
        "CANDIDATE_MIN_RECENT_DOLLAR_VOLUME_USD",
        _setting_float(settings, "candidate_min_recent_dollar_volume_usd", 5_000_000.0),
    )
    candidate_request_pause_seconds = secret_float(
        "CANDIDATE_REQUEST_PAUSE_SECONDS",
        _setting_float(settings, "candidate_request_pause_seconds", 0.08),
    )
    target_exit_tolerance_pct = secret_float(
        "TARGET_EXIT_TOLERANCE_PCT",
        _setting_float(settings, "target_exit_tolerance_pct", 0.5),
    )
    target_extension_enabled = secret_bool(
        "TARGET_EXTENSION_ENABLED",
        _setting_bool(settings, "target_extension_enabled", True),
    )
    target_extension_max_pct = secret_float(
        "TARGET_EXTENSION_MAX_PCT",
        _setting_float(settings, "target_extension_max_pct", 0.6),
    )
    target_extension_trailing_stop_pct = secret_float(
        "TARGET_EXTENSION_TRAILING_STOP_PCT",
        _setting_float(settings, "target_extension_trailing_stop_pct", 0.35),
    )
    target_extension_min_surge_score = secret_float(
        "TARGET_EXTENSION_MIN_SURGE_SCORE",
        _setting_float(settings, "target_extension_min_surge_score", 8.0),
    )
    target_extension_min_lookback_change_pct = secret_float(
        "TARGET_EXTENSION_MIN_LOOKBACK_CHANGE_PCT",
        _setting_float(settings, "target_extension_min_lookback_change_pct", 0.4),
    )
    target_extension_min_latest_bar_change_pct = secret_float(
        "TARGET_EXTENSION_MIN_LATEST_BAR_CHANGE_PCT",
        _setting_float(settings, "target_extension_min_latest_bar_change_pct", 0.0),
    )
    target_extension_min_volume_ratio = secret_float(
        "TARGET_EXTENSION_MIN_VOLUME_RATIO",
        _setting_float(settings, "target_extension_min_volume_ratio", 1.1),
    )
    target_extension_max_rsi = secret_float(
        "TARGET_EXTENSION_MAX_RSI",
        _setting_float(settings, "target_extension_max_rsi", 84.0),
    )
    manual_position_exit_enabled = secret_bool(
        "MANUAL_POSITION_EXIT_ENABLED",
        _setting_bool(settings, "manual_position_exit_enabled", True),
    )
    manual_position_take_profit_pct = secret_float(
        "MANUAL_POSITION_TAKE_PROFIT_PCT",
        _setting_float(settings, "manual_position_take_profit_pct", 2.0),
    )
    manual_position_stop_loss_pct = secret_float(
        "MANUAL_POSITION_STOP_LOSS_PCT",
        _setting_float(settings, "manual_position_stop_loss_pct", 2.0),
    )
    target_decay_exit_enabled = secret_bool(
        "TARGET_DECAY_EXIT_ENABLED",
        _setting_bool(settings, "target_decay_exit_enabled", True),
    )
    target_decay_exit_min_probability_pct = secret_float(
        "TARGET_DECAY_EXIT_MIN_PROBABILITY_PCT",
        _setting_float(settings, "target_decay_exit_min_probability_pct", 40.0),
    )
    target_decay_exit_min_target_gap_pct = secret_float(
        "TARGET_DECAY_EXIT_MIN_TARGET_GAP_PCT",
        _setting_float(settings, "target_decay_exit_min_target_gap_pct", 0.8),
    )
    target_decay_exit_max_volume_ratio = secret_float(
        "TARGET_DECAY_EXIT_MAX_VOLUME_RATIO",
        _setting_float(settings, "target_decay_exit_max_volume_ratio", 0.7),
    )
    target_decay_exit_max_volume_zscore = secret_float(
        "TARGET_DECAY_EXIT_MAX_VOLUME_ZSCORE",
        _setting_float(settings, "target_decay_exit_max_volume_zscore", -0.35),
    )
    target_decay_exit_min_weak_bars = int(
        secret_float(
            "TARGET_DECAY_EXIT_MIN_WEAK_BARS",
            _setting_float(settings, "target_decay_exit_min_weak_bars", 2.0),
        )
    )
    target_decay_exit_max_fee_adjusted_return_pct = secret_float(
        "TARGET_DECAY_EXIT_MAX_FEE_ADJUSTED_RETURN_PCT",
        _setting_float(settings, "target_decay_exit_max_fee_adjusted_return_pct", -0.15),
    )
    target_decay_rebound_protection_enabled = secret_bool(
        "TARGET_DECAY_REBOUND_PROTECTION_ENABLED",
        _setting_bool(settings, "target_decay_rebound_protection_enabled", True),
    )
    target_decay_rebound_min_score = secret_float(
        "TARGET_DECAY_REBOUND_MIN_SCORE",
        _setting_float(settings, "target_decay_rebound_min_score", 60.0),
    )
    target_decay_rebound_min_samples = int(
        secret_float(
            "TARGET_DECAY_REBOUND_MIN_SAMPLES",
            _setting_float(settings, "target_decay_rebound_min_samples", 5.0),
        )
    )
    target_decay_rebound_lookback_bars = int(
        secret_float(
            "TARGET_DECAY_REBOUND_LOOKBACK_BARS",
            _setting_float(settings, "target_decay_rebound_lookback_bars", 160.0),
        )
    )
    target_decay_rebound_forward_bars = int(
        secret_float(
            "TARGET_DECAY_REBOUND_FORWARD_BARS",
            _setting_float(settings, "target_decay_rebound_forward_bars", 8.0),
        )
    )
    structure_guard_enabled = secret_bool(
        "STRUCTURE_GUARD_ENABLED",
        _setting_bool(settings, "structure_guard_enabled", True),
    )
    structure_lookback_bars = int(
        secret_float("STRUCTURE_LOOKBACK_BARS", _setting_float(settings, "structure_lookback_bars", 20.0))
    )
    structure_confirm_bars = int(
        secret_float("STRUCTURE_CONFIRM_BARS", _setting_float(settings, "structure_confirm_bars", 2.0))
    )
    structure_min_break_pct = secret_float(
        "STRUCTURE_MIN_BREAK_PCT",
        _setting_float(settings, "structure_min_break_pct", 0.15),
    )
    structure_atr_buffer_multiplier = secret_float(
        "STRUCTURE_ATR_BUFFER_MULTIPLIER",
        _setting_float(settings, "structure_atr_buffer_multiplier", 0.25),
    )
    structure_min_volume_ratio = secret_float(
        "STRUCTURE_MIN_VOLUME_RATIO",
        _setting_float(settings, "structure_min_volume_ratio", 1.0),
    )
    structure_min_volume_zscore = secret_float(
        "STRUCTURE_MIN_VOLUME_ZSCORE",
        _setting_float(settings, "structure_min_volume_zscore", -0.2),
    )
    entry_structure_block_lower_breakdown = secret_bool(
        "ENTRY_STRUCTURE_BLOCK_LOWER_BREAKDOWN",
        _setting_bool(settings, "entry_structure_block_lower_breakdown", True),
    )
    entry_structure_require_upper_breakout = secret_bool(
        "ENTRY_STRUCTURE_REQUIRE_UPPER_BREAKOUT",
        _setting_bool(settings, "entry_structure_require_upper_breakout", False),
    )
    support_breakdown_exit_enabled = secret_bool(
        "SUPPORT_BREAKDOWN_EXIT_ENABLED",
        _setting_bool(settings, "support_breakdown_exit_enabled", True),
    )
    reentry_cooldown_enabled = secret_bool(
        "REENTRY_COOLDOWN_ENABLED",
        _setting_bool(settings, "reentry_cooldown_enabled", True),
    )
    reentry_cooldown_minutes = secret_float(
        "REENTRY_COOLDOWN_MINUTES",
        _setting_float(settings, "reentry_cooldown_minutes", 8.0),
    )
    reentry_cooldown_min_breakout_pct = secret_float(
        "REENTRY_COOLDOWN_MIN_BREAKOUT_PCT",
        _setting_float(settings, "reentry_cooldown_min_breakout_pct", 0.5),
    )
    reentry_cooldown_min_volume_ratio = secret_float(
        "REENTRY_COOLDOWN_MIN_VOLUME_RATIO",
        _setting_float(settings, "reentry_cooldown_min_volume_ratio", 1.5),
    )
    reentry_cooldown_min_confidence_pct = secret_float(
        "REENTRY_COOLDOWN_MIN_CONFIDENCE_PCT",
        _setting_float(settings, "reentry_cooldown_min_confidence_pct", 60.0),
    )
    surge_focus_enabled = secret_bool(
        "SURGE_FOCUS_ENABLED",
        _setting_bool(settings, "surge_focus_enabled", False),
    )
    surge_lookback_bars = int(secret_float("SURGE_LOOKBACK_BARS", _setting_float(settings, "surge_lookback_bars", 5.0)))
    surge_min_lookback_change_pct = secret_float(
        "SURGE_MIN_LOOKBACK_CHANGE_PCT",
        _setting_float(settings, "surge_min_lookback_change_pct", 0.4),
    )
    surge_min_latest_bar_change_pct = secret_float(
        "SURGE_MIN_LATEST_BAR_CHANGE_PCT",
        _setting_float(settings, "surge_min_latest_bar_change_pct", 0.0),
    )
    surge_min_volume_ratio = secret_float(
        "SURGE_MIN_VOLUME_RATIO",
        _setting_float(settings, "surge_min_volume_ratio", 1.5),
    )
    surge_min_volume_zscore = secret_float(
        "SURGE_MIN_VOLUME_ZSCORE",
        _setting_float(settings, "surge_min_volume_zscore", 0.0),
    )
    surge_min_score = secret_float(
        "SURGE_MIN_SCORE",
        _setting_float(settings, "surge_min_score", 8.0),
    )
    wide_universe_enabled = secret_bool(
        "WIDE_UNIVERSE_ENABLED",
        _setting_bool(settings, "wide_universe_enabled", False),
    )
    wide_universe_max_symbols = int(
        secret_float("WIDE_UNIVERSE_MAX_SYMBOLS", _setting_float(settings, "wide_universe_max_symbols", 300.0))
    )
    wide_batch_size = int(secret_float("WIDE_BATCH_SIZE", _setting_float(settings, "wide_batch_size", 80.0)))
    wide_fast_candle_count = int(
        secret_float("WIDE_FAST_CANDLE_COUNT", _setting_float(settings, "wide_fast_candle_count", 80.0))
    )
    wide_active_ttl_minutes = secret_float(
        "WIDE_ACTIVE_TTL_MINUTES",
        _setting_float(settings, "wide_active_ttl_minutes", 30.0),
    )
    wide_active_max_symbols = int(
        secret_float("WIDE_ACTIVE_MAX_SYMBOLS", _setting_float(settings, "wide_active_max_symbols", 80.0))
    )
    wide_model_max_symbols = int(
        secret_float("WIDE_MODEL_MAX_SYMBOLS", _setting_float(settings, "wide_model_max_symbols", 60.0))
    )
    wide_min_surge_score = secret_float(
        "WIDE_MIN_SURGE_SCORE",
        _setting_float(settings, "wide_min_surge_score", 4.0),
    )
    wide_min_lookback_change_pct = secret_float(
        "WIDE_MIN_LOOKBACK_CHANGE_PCT",
        _setting_float(settings, "wide_min_lookback_change_pct", 0.2),
    )
    wide_min_volume_ratio = secret_float(
        "WIDE_MIN_VOLUME_RATIO",
        _setting_float(settings, "wide_min_volume_ratio", 1.1),
    )
    wide_min_model_candles = int(
        secret_float("WIDE_MIN_MODEL_CANDLES", _setting_float(settings, "wide_min_model_candles", 80.0))
    )
    wide_request_pause_seconds = secret_float(
        "WIDE_REQUEST_PAUSE_SECONDS",
        _setting_float(settings, "wide_request_pause_seconds", 0.15),
    )

    auto_symbol_selection_enabled = secret_bool(
        "AUTO_SYMBOL_SELECTION_ENABLED",
        _setting_bool(settings, "auto_symbol_selection_enabled", True),
    )
    min_probability = secret_float(
        "AUTO_SELECTION_MIN_PROBABILITY",
        _setting_float(settings, "auto_selection_min_probability", 55.0),
    )
    min_volume_ratio = secret_float(
        "AUTO_SELECTION_MIN_VOLUME_RATIO",
        _setting_float(settings, "auto_selection_min_volume_ratio", 1.2),
    )
    require_risk_normal = secret_bool(
        "AUTO_SELECTION_REQUIRE_RISK_NORMAL",
        _setting_bool(settings, "auto_selection_require_risk_normal", True),
    )
    allow_caution_entries = secret_bool(
        "AUTO_SELECTION_ALLOW_CAUTION_ENTRIES",
        _setting_bool(settings, "auto_selection_allow_caution_entries", False),
    )
    entry_liquidity_guard_enabled = secret_bool(
        "ENTRY_LIQUIDITY_GUARD_ENABLED",
        _setting_bool(settings, "entry_liquidity_guard_enabled", True),
    )
    entry_liquidity_min_recent_dollar_volume_usd = secret_float(
        "ENTRY_LIQUIDITY_MIN_RECENT_DOLLAR_VOLUME_USD",
        _setting_float(settings, "entry_liquidity_min_recent_dollar_volume_usd", 5_000_000.0),
    )
    entry_liquidity_min_latest_dollar_volume_usd = secret_float(
        "ENTRY_LIQUIDITY_MIN_LATEST_DOLLAR_VOLUME_USD",
        _setting_float(settings, "entry_liquidity_min_latest_dollar_volume_usd", 200_000.0),
    )
    entry_liquidity_min_volume_ratio = secret_float(
        "ENTRY_LIQUIDITY_MIN_VOLUME_RATIO",
        _setting_float(settings, "entry_liquidity_min_volume_ratio", 1.2),
    )
    intraday_projection_horizon_bars = int(
        secret_float(
            "INTRADAY_PROJECTION_HORIZON_BARS",
            _setting_float(settings, "intraday_projection_horizon_bars", 60.0),
        )
    )
    intraday_targeting_enabled = secret_bool(
        "INTRADAY_TARGETING_ENABLED",
        _setting_bool(settings, "intraday_targeting_enabled", True),
    )
    intraday_target_capture_pct = secret_float(
        "INTRADAY_TARGET_CAPTURE_PCT",
        _setting_float(settings, "intraday_target_capture_pct", 85.0),
    )
    intraday_target_min_gross_pct = secret_float(
        "INTRADAY_TARGET_MIN_GROSS_PCT",
        _setting_float(settings, "intraday_target_min_gross_pct", 1.6),
    )
    intraday_target_max_gross_pct = secret_float(
        "INTRADAY_TARGET_MAX_GROSS_PCT",
        _setting_float(settings, "intraday_target_max_gross_pct", 3.0),
    )
    entry_edge_guard_enabled = secret_bool(
        "ENTRY_EDGE_GUARD_ENABLED",
        _setting_bool(settings, "entry_edge_guard_enabled", True),
    )
    entry_edge_min_gross_target_pct = secret_float(
        "ENTRY_EDGE_MIN_GROSS_TARGET_PCT",
        _setting_float(settings, "entry_edge_min_gross_target_pct", 1.2),
    )
    entry_edge_min_net_target_pct = secret_float(
        "ENTRY_EDGE_MIN_NET_TARGET_PCT",
        _setting_float(settings, "entry_edge_min_net_target_pct", 0.5),
    )
    entry_edge_fee_multiplier = secret_float(
        "ENTRY_EDGE_FEE_MULTIPLIER",
        _setting_float(settings, "entry_edge_fee_multiplier", 3.0),
    )
    entry_edge_slippage_buffer_pct = secret_float(
        "ENTRY_EDGE_SLIPPAGE_BUFFER_PCT",
        _setting_float(settings, "entry_edge_slippage_buffer_pct", 0.15),
    )
    entry_edge_min_order_notional_usd = secret_float(
        "ENTRY_EDGE_MIN_ORDER_NOTIONAL_USD",
        _setting_float(settings, "entry_edge_min_order_notional_usd", 75.0),
    )
    entry_edge_min_projected_target_coverage_pct = secret_float(
        "ENTRY_EDGE_MIN_PROJECTED_TARGET_COVERAGE_PCT",
        _setting_float(settings, "entry_edge_min_projected_target_coverage_pct", 75.0),
    )
    entry_edge_min_expected_move_pct = secret_float(
        "ENTRY_EDGE_MIN_EXPECTED_MOVE_PCT",
        _setting_float(settings, "entry_edge_min_expected_move_pct", 0.08),
    )
    daily_order_guard_enabled = secret_bool(
        "DAILY_ORDER_GUARD_ENABLED",
        _setting_bool(settings, "daily_order_guard_enabled", True),
    )
    daily_order_guard_max_loss_usd = secret_float(
        "DAILY_ORDER_GUARD_MAX_LOSS_USD",
        _setting_float(settings, "daily_order_guard_max_loss_usd", 0.0),
    )
    daily_order_guard_max_consecutive_losses = int(
        secret_float(
            "DAILY_ORDER_GUARD_MAX_CONSECUTIVE_LOSSES",
            _setting_float(settings, "daily_order_guard_max_consecutive_losses", 3.0),
        )
    )
    daily_order_guard_commission_floor_pct = secret_float(
        "DAILY_ORDER_GUARD_COMMISSION_FLOOR_PCT",
        _setting_float(settings, "daily_order_guard_commission_floor_pct", 0.1),
    )
    trade_prepare_mode = secret_choice(
        "TRADE_PREPARE_MODE",
        ["disabled", "dry_run", "live_ready"],
        str(_setting(settings, "trade_prepare_mode", "disabled")),
    )
    auto_trade_mode = secret_choice(
        "AUTO_TRADE_MODE",
        ["disabled", "paper", "live"],
        str(_setting(settings, "auto_trade_mode", "disabled")),
    )
    auto_kill_switch = secret_bool("AUTO_TRADE_KILL_SWITCH", _setting_bool(settings, "auto_kill_switch", True))
    auto_preapproval = secret_bool("AUTO_TRADE_PREAPPROVAL", _setting_bool(settings, "auto_preapproval", False))
    toss_account_sync_enabled = secret_bool(
        "TOSS_ACCOUNT_SYNC_ENABLED",
        _setting_bool(settings, "toss_account_sync_enabled", False),
    )
    toss_live_order_enabled = secret_bool(
        "TOSS_LIVE_ORDER_ENABLED",
        _setting_bool(settings, "toss_live_order_enabled", False),
    )
    manual_approval_complete = secret_bool(
        "TRADE_APPROVAL_CHECKLIST_COMPLETE",
        _setting_bool(settings, "manual_approval_complete", False),
    )

    toss_broker_adapter = None
    real_account_snapshot = None
    real_holdings_payload: dict[str, Any] | None = None
    real_commissions_payload: list[dict[str, Any]] = []
    real_holding_qty: dict[str, float] = {}
    account_sync_error = ""
    holdings_sync_error = ""
    commissions_sync_error = ""
    if toss_token or (toss_client_id and toss_client_secret):
        toss_broker_adapter = _broker(allow_live_orders=toss_live_order_enabled)
        if toss_account_sync_enabled or auto_trade_mode == "live":
            try:
                real_account_snapshot = toss_broker_adapter.get_account()
            except Exception as exc:
                account_sync_error = f"Toss account sync failed: {exc}"
            try:
                real_holdings_payload = toss_broker_adapter.get_holdings()
                real_holding_qty = _holding_quantity_map(real_holdings_payload)
            except Exception as exc:
                holdings_sync_error = f"Toss holdings sync failed: {exc}"
            try:
                real_commissions_payload = toss_broker_adapter.get_commissions()
            except Exception as exc:
                commissions_sync_error = f"Toss commissions sync failed: {exc}"

    if real_holding_qty:
        tickers = _unique_symbols(tickers + list(real_holding_qty))
    tickers = _filter_broker_blocked_symbols(
        tickers,
        broker_blocked_symbols,
        holding_symbols=list(real_holding_qty),
    )
    if not tickers:
        raise ValueError("candidate universe is empty after broker blocked symbol filter")

    base_tickers = list(tickers)
    wide_scan_summary: dict[str, Any] = {"enabled": bool(wide_universe_enabled)}
    wide_failures: list[tuple[str, str]] = []
    if wide_universe_enabled:
        dynamic_symbols, wide_scan_summary, wide_failures = _refresh_wide_candidates(
            settings,
            enabled=wide_universe_enabled,
            interval=interval,
            toss_token=toss_token,
            toss_client_id=toss_client_id,
            toss_client_secret=toss_client_secret,
            toss_account_seq=toss_account_seq,
            toss_base_url=toss_base_url,
            liquidity_recent_bars=liquidity_recent_bars,
            min_recent_volume=min_recent_volume,
            min_recent_dollar_volume=min_recent_dollar_volume,
            surge_lookback_bars=surge_lookback_bars,
            watchlist_override=watchlist_override,
            max_symbols=wide_universe_max_symbols,
            batch_size=wide_batch_size,
            fast_candle_count=wide_fast_candle_count,
            active_ttl_minutes=wide_active_ttl_minutes,
            active_max_symbols=wide_active_max_symbols,
            min_surge_score=wide_min_surge_score,
            min_lookback_change_pct=wide_min_lookback_change_pct,
            min_volume_ratio=wide_min_volume_ratio,
            min_model_candles=wide_min_model_candles,
            request_pause_seconds=wide_request_pause_seconds,
        )
        pinned_symbols = _unique_symbols(_watchlist(settings, watchlist_override) + list(real_holding_qty))
        tickers = _modeled_universe(
            base_symbols=base_tickers,
            dynamic_symbols=dynamic_symbols,
            pinned_symbols=pinned_symbols,
            max_symbols=wide_model_max_symbols,
        )
        tickers = _filter_broker_blocked_symbols(
            tickers,
            broker_blocked_symbols,
            holding_symbols=list(real_holding_qty),
        )
        wide_scan_summary["modeled_symbols_count"] = len(tickers)
        wide_scan_summary["base_universe_size"] = len(base_tickers)
        wide_scan_summary["pinned_symbols_count"] = len(pinned_symbols)
        wide_scan_summary["broker_blocked_symbols_count"] = len(broker_blocked_symbols)

    account_snapshot = (
        AccountSnapshot(
            account_id="paper",
            cash_buying_power=paper_buying_power_usd,
            equity=paper_buying_power_usd,
        )
        if auto_trade_mode == "paper"
        else real_account_snapshot
    )

    scanner, loaded, failures, liquidity_filtered = scan_candidates(
        tickers,
        interval=interval,
        reward_ratio=reward_ratio,
        toss_token=toss_token,
        toss_client_id=toss_client_id,
        toss_client_secret=toss_client_secret,
        toss_account_seq=toss_account_seq,
        toss_base_url=toss_base_url,
        liquidity_recent_bars=liquidity_recent_bars,
        min_recent_volume=min_recent_volume,
        min_recent_dollar_volume=min_recent_dollar_volume,
        surge_lookback_bars=surge_lookback_bars,
        always_include_symbols=set(real_holding_qty),
        request_pause_seconds=candidate_request_pause_seconds,
    )
    failures = wide_failures + failures
    position_summary = _position_summary_rows(
        real_holdings_payload,
        loaded,
        real_commissions_payload,
        manual_take_profit_pct=manual_position_take_profit_pct,
        manual_stop_loss_pct=manual_position_stop_loss_pct,
    )
    effective_commission_rate_pct = _effective_commission_rate_pct(
        real_commissions_payload,
        fallback_rate_pct=daily_order_guard_commission_floor_pct,
    )
    daily_order_guard_stats = _daily_order_guard_stats(
        AUTO_TRADE_EVENTS_PATH,
        commission_rate_pct=effective_commission_rate_pct,
    )
    scanner, reentry_cooldown = _apply_reentry_cooldown(
        scanner,
        _tail_session_events(SESSION_EVENT_PATH, limit=500),
        now=datetime.now(timezone.utc),
        enabled=reentry_cooldown_enabled,
        cooldown_minutes=reentry_cooldown_minutes,
        min_breakout_pct=reentry_cooldown_min_breakout_pct,
        min_volume_ratio=reentry_cooldown_min_volume_ratio,
        min_confidence_pct=reentry_cooldown_min_confidence_pct,
    )
    if scanner.empty:
        auto_trade = {
            "status": "AUTO_BLOCKED",
            "state": "PLAN_NOT_READY",
            "mode": auto_trade_mode,
            "order": None,
            "execution_result": None,
            "checks": {
                "enabled": auto_trade_mode != "disabled",
                "mode": auto_trade_mode,
                "kill_switch": auto_kill_switch,
                "plan_status": "BLOCKED",
                "has_order": False,
                "estimated_notional": 0.0,
                "max_order_notional_usd": round(max_auto_order_notional_usd, 2),
                "allowed_symbols": tickers,
                "adapter_connected": toss_broker_adapter is not None,
                "already_submitted": False,
                "candidate_universe_mode": candidate_universe_mode,
                "candidate_universe_size": len(tickers),
                "liquidity_filtered_count": len(liquidity_filtered),
                "min_recent_volume": float(min_recent_volume),
                "min_recent_dollar_volume": float(min_recent_dollar_volume),
                "liquidity_recent_bars": int(liquidity_recent_bars),
                "wide_scan": wide_scan_summary,
                "position_count": len(position_summary),
                "reentry_cooldown": reentry_cooldown,
            },
            "reason": ["NO_LIQUIDITY_CANDIDATE"],
            "safety_notice": "No symbols passed the liquidity filter. No broker order was sent.",
            "created_at": _now_iso(),
        }
        save_auto_trade_decision(auto_trade)
        append_auto_trade_event(auto_trade, AUTO_TRADE_EVENTS_PATH)
        event = {
            "status": auto_trade["status"],
            "state": auto_trade["state"],
            "mode": auto_trade["mode"],
            "symbol": "",
            "auto_selection": {
                "enabled": auto_symbol_selection_enabled,
                "symbol": "",
                "eligible": False,
                "reason": ["NO_LIQUIDITY_CANDIDATE"],
                "candidate_universe_mode": candidate_universe_mode,
                "candidate_universe_size": len(tickers),
                "liquidity_filtered_count": len(liquidity_filtered),
                "wide_scan": wide_scan_summary,
                "position_count": len(position_summary),
                "reentry_cooldown": reentry_cooldown,
            },
            "trade_intent_status": "BLOCKED",
            "trade_intent_side": "HOLD",
            "trade_intent_quantity": 0,
            "execution_status": "BLOCKED",
            "order_client_id": None,
            "broker_order_id": None,
            "reasons": ["NO_LIQUIDITY_CANDIDATE"],
            "wide_scan": wide_scan_summary,
            "position_summary": position_summary,
            "position_count": len(position_summary),
            "reentry_cooldown": reentry_cooldown,
            "failures": failures[:20],
        }
        append_session_event(event)
        return event

    buy_scanner = _buy_candidate_scanner(scanner)
    buy_scanner = _filter_reentry_cooldown(buy_scanner)
    liquid_buy_candidate_count = int(len(buy_scanner))
    top_surge_candidates = _top_surge_candidates(buy_scanner)
    if surge_focus_enabled:
        buy_scanner = _surge_candidate_scanner(
            buy_scanner,
            enabled=True,
            min_lookback_change_pct=surge_min_lookback_change_pct,
            min_latest_bar_change_pct=surge_min_latest_bar_change_pct,
            min_volume_ratio=surge_min_volume_ratio,
            min_volume_zscore=surge_min_volume_zscore,
            min_surge_score=surge_min_score,
        )
    surge_candidate_count = int(len(buy_scanner)) if surge_focus_enabled else 0
    liquidity_passed_count = _liquidity_passed_count(scanner)
    candidate_scores = _candidate_score_rows(
        scanner,
        min_probability=min_probability,
        min_volume_ratio=min_volume_ratio,
        require_risk_normal=require_risk_normal,
        allow_caution_entries=allow_caution_entries,
        surge_focus_enabled=surge_focus_enabled,
        surge_min_lookback_change_pct=surge_min_lookback_change_pct,
        surge_min_latest_bar_change_pct=surge_min_latest_bar_change_pct,
        surge_min_volume_ratio=surge_min_volume_ratio,
        surge_min_volume_zscore=surge_min_volume_zscore,
        surge_min_score=surge_min_score,
    )
    exit_candidate = _select_exit_candidate(
        loaded,
        real_holding_qty,
        tolerance_pct=target_exit_tolerance_pct,
        position_summary=position_summary,
        manual_position_exit_enabled=manual_position_exit_enabled,
        manual_take_profit_pct=manual_position_take_profit_pct,
        manual_stop_loss_pct=manual_position_stop_loss_pct,
        target_extension_enabled=target_extension_enabled,
        target_extension_max_pct=target_extension_max_pct,
        target_extension_trailing_stop_pct=target_extension_trailing_stop_pct,
        target_extension_min_surge_score=target_extension_min_surge_score,
        target_extension_min_lookback_change_pct=target_extension_min_lookback_change_pct,
        target_extension_min_latest_bar_change_pct=target_extension_min_latest_bar_change_pct,
        target_extension_min_volume_ratio=target_extension_min_volume_ratio,
        target_extension_max_rsi=target_extension_max_rsi,
        target_decay_exit_enabled=target_decay_exit_enabled,
        target_decay_exit_min_probability_pct=target_decay_exit_min_probability_pct,
        target_decay_exit_min_target_gap_pct=target_decay_exit_min_target_gap_pct,
        target_decay_exit_max_volume_ratio=target_decay_exit_max_volume_ratio,
        target_decay_exit_max_volume_zscore=target_decay_exit_max_volume_zscore,
        target_decay_exit_min_weak_bars=target_decay_exit_min_weak_bars,
        target_decay_exit_max_fee_adjusted_return_pct=target_decay_exit_max_fee_adjusted_return_pct,
        target_decay_rebound_protection_enabled=target_decay_rebound_protection_enabled,
        target_decay_rebound_min_score=target_decay_rebound_min_score,
        target_decay_rebound_min_samples=target_decay_rebound_min_samples,
        target_decay_rebound_lookback_bars=target_decay_rebound_lookback_bars,
        target_decay_rebound_forward_bars=target_decay_rebound_forward_bars,
        support_breakdown_exit_enabled=support_breakdown_exit_enabled and structure_guard_enabled,
        structure_lookback_bars=structure_lookback_bars,
        structure_confirm_bars=structure_confirm_bars,
        structure_min_break_pct=structure_min_break_pct,
        structure_atr_buffer_multiplier=structure_atr_buffer_multiplier,
        structure_min_volume_ratio=structure_min_volume_ratio,
        structure_min_volume_zscore=structure_min_volume_zscore,
    )
    if exit_candidate is None:
        scanner = buy_scanner
        if scanner.empty:
            no_buy_reason = (
                "REENTRY_COOLDOWN_ACTIVE"
                if int(reentry_cooldown.get("blocked_count") or 0) > 0
                else "NO_SURGE_BUY_CANDIDATE"
                if surge_focus_enabled
                else "NO_LIQUIDITY_BUY_CANDIDATE"
            )
            auto_trade = {
                "status": "AUTO_BLOCKED",
                "state": "PLAN_NOT_READY",
                "mode": auto_trade_mode,
                "order": None,
                "execution_result": None,
                "checks": {
                    "enabled": auto_trade_mode != "disabled",
                    "mode": auto_trade_mode,
                    "kill_switch": auto_kill_switch,
                    "plan_status": "BLOCKED",
                    "has_order": False,
                    "estimated_notional": 0.0,
                    "max_order_notional_usd": round(max_auto_order_notional_usd, 2),
                    "allowed_symbols": tickers,
                    "adapter_connected": toss_broker_adapter is not None,
                    "already_submitted": False,
                    "candidate_universe_mode": candidate_universe_mode,
                    "candidate_universe_size": len(tickers),
                    "liquidity_filtered_count": len(liquidity_filtered),
                    "liquidity_passed_count": liquidity_passed_count,
                    "liquid_buy_candidate_count": liquid_buy_candidate_count,
                    "surge_focus_enabled": bool(surge_focus_enabled),
                    "surge_candidate_count": surge_candidate_count,
                    "surge_lookback_bars": int(surge_lookback_bars),
                    "surge_min_lookback_change_pct": float(surge_min_lookback_change_pct),
                    "surge_min_latest_bar_change_pct": float(surge_min_latest_bar_change_pct),
                    "surge_min_volume_ratio": float(surge_min_volume_ratio),
                    "surge_min_volume_zscore": float(surge_min_volume_zscore),
                    "surge_min_score": float(surge_min_score),
                    "top_surge_candidates": top_surge_candidates,
                    "holding_symbols_count": len(real_holding_qty),
                    "position_count": len(position_summary),
                    "min_recent_volume": float(min_recent_volume),
                    "min_recent_dollar_volume": float(min_recent_dollar_volume),
                    "liquidity_recent_bars": int(liquidity_recent_bars),
                    "wide_scan": wide_scan_summary,
                    "reentry_cooldown": reentry_cooldown,
                },
                "reason": [no_buy_reason],
                "safety_notice": "No buy symbol passed the active liquidity/surge filters and no holding reached an exit trigger.",
                "created_at": _now_iso(),
            }
            save_auto_trade_decision(auto_trade)
            append_auto_trade_event(auto_trade, AUTO_TRADE_EVENTS_PATH)
            event = {
                "status": auto_trade["status"],
                "state": auto_trade["state"],
                "mode": auto_trade["mode"],
                "symbol": "",
                "auto_selection": {
                    "enabled": auto_symbol_selection_enabled,
                    "symbol": "",
                    "eligible": False,
                    "reason": [no_buy_reason],
                    "candidate_universe_mode": candidate_universe_mode,
                    "candidate_universe_size": len(tickers),
                    "liquidity_filtered_count": len(liquidity_filtered),
                    "liquidity_passed_count": liquidity_passed_count,
                    "liquid_buy_candidate_count": liquid_buy_candidate_count,
                    "surge_focus_enabled": bool(surge_focus_enabled),
                    "surge_candidate_count": surge_candidate_count,
                    "top_surge_candidates": top_surge_candidates,
                    "holding_symbols_count": len(real_holding_qty),
                    "candidate_scores_count": len(candidate_scores),
                    "position_count": len(position_summary),
                    "wide_scan": wide_scan_summary,
                    "reentry_cooldown": reentry_cooldown,
                },
                "trade_intent_status": "BLOCKED",
                "trade_intent_side": "HOLD",
                "trade_intent_quantity": 0,
                "execution_status": "BLOCKED",
                "order_client_id": None,
                "broker_order_id": None,
                "reasons": [no_buy_reason],
                "candidate_universe_mode": candidate_universe_mode,
                "candidate_universe_size": len(tickers),
                "liquidity_passed_count": liquidity_passed_count,
                "liquid_buy_candidate_count": liquid_buy_candidate_count,
                "surge_focus_enabled": bool(surge_focus_enabled),
                "surge_candidate_count": surge_candidate_count,
                "top_surge_candidates": top_surge_candidates,
                "candidate_scores": candidate_scores,
                "candidate_scores_count": len(candidate_scores),
                "wide_scan": wide_scan_summary,
                "reentry_cooldown": reentry_cooldown,
                "position_summary": position_summary,
                "position_count": len(position_summary),
                "failures": failures[:20],
            }
            append_session_event(event)
            return event

    if exit_candidate is not None:
        auto_selection = {
            "enabled": True,
            "symbol": str(exit_candidate["symbol"]),
            "eligible": True,
            "reason": [str(exit_candidate["reason"]), "AUTO_EXIT_PRIORITY"],
            "exit": True,
            "exit_reason": str(exit_candidate["reason"]),
            "exit_executable": str(exit_candidate["reason"]) != "TARGET_EXTENSION_ACTIVE",
            "holding_qty": float(exit_candidate["holding_qty"]),
            "exit_quantity": int(exit_candidate["quantity"]),
            "current_price": round(float(exit_candidate["current_price"]), 4),
            "original_target_price": round(_as_float(exit_candidate.get("original_target_price"), _as_float(exit_candidate["target_price"])), 4),
            "target_price": round(float(exit_candidate["target_price"]), 4),
            "target_trigger": round(float(exit_candidate["target_trigger"]), 4),
            "target_exit_tolerance_pct": round(float(target_exit_tolerance_pct), 4),
            "target_extension": exit_candidate.get("target_extension") if isinstance(exit_candidate.get("target_extension"), dict) else {},
            "target_decay_exit": exit_candidate.get("target_decay_exit") if isinstance(exit_candidate.get("target_decay_exit"), dict) else {},
            "support_resistance": exit_candidate.get("support_resistance") if isinstance(exit_candidate.get("support_resistance"), dict) else {},
            "manual_exit": bool(exit_candidate.get("manual_exit", False)),
            "entry_price": round(_as_float(exit_candidate.get("entry_price")), 4),
            "fee_adjusted_return_pct": round(_as_float(exit_candidate.get("fee_adjusted_return_pct")), 4),
            "fee_adjusted_pnl_usd": round(_as_float(exit_candidate.get("fee_adjusted_pnl_usd")), 4),
            "manual_take_profit_pct": round(_as_float(exit_candidate.get("manual_take_profit_pct")), 4),
            "manual_stop_loss_pct": round(_as_float(exit_candidate.get("manual_stop_loss_pct")), 4),
            "candidate_universe_mode": candidate_universe_mode,
            "candidate_universe_size": len(tickers),
            "liquidity_filtered_count": len(liquidity_filtered),
            "liquidity_passed_count": liquidity_passed_count,
            "liquid_buy_candidate_count": liquid_buy_candidate_count,
            "surge_focus_enabled": bool(surge_focus_enabled),
            "surge_candidate_count": surge_candidate_count,
            "surge_lookback_bars": int(surge_lookback_bars),
            "surge_min_lookback_change_pct": float(surge_min_lookback_change_pct),
            "surge_min_latest_bar_change_pct": float(surge_min_latest_bar_change_pct),
            "surge_min_volume_ratio": float(surge_min_volume_ratio),
            "surge_min_volume_zscore": float(surge_min_volume_zscore),
            "surge_min_score": float(surge_min_score),
            "top_surge_candidates": top_surge_candidates,
            "holding_symbols_count": len(real_holding_qty),
            "position_count": len(position_summary),
            "min_recent_volume": float(min_recent_volume),
            "min_recent_dollar_volume": float(min_recent_dollar_volume),
            "liquidity_recent_bars": int(liquidity_recent_bars),
            "wide_scan": wide_scan_summary,
            "reentry_cooldown": reentry_cooldown,
        }
    elif auto_symbol_selection_enabled:
        available_buying_power_usd = _as_float(getattr(account_snapshot, "cash_buying_power", 0.0))
        affordable_entry_price_limit = available_buying_power_usd
        if max_auto_order_notional_usd > 0:
            affordable_entry_price_limit = min(affordable_entry_price_limit, max_auto_order_notional_usd)
        auto_selection = _choose_auto_symbol_from_monitor_rows(
            scanner,
            min_probability=min_probability,
            min_volume_ratio=min_volume_ratio,
            require_risk_normal=require_risk_normal,
            allow_caution_entries=allow_caution_entries,
            max_entry_price=affordable_entry_price_limit,
        )
        auto_selection.update(
            {
                "candidate_universe_mode": candidate_universe_mode,
                "candidate_universe_size": len(tickers),
                "available_buying_power_usd": round(available_buying_power_usd, 2),
                "liquidity_filtered_count": len(liquidity_filtered),
                "liquidity_passed_count": liquidity_passed_count,
                "liquid_buy_candidate_count": liquid_buy_candidate_count,
                "surge_focus_enabled": bool(surge_focus_enabled),
                "surge_candidate_count": surge_candidate_count,
                "surge_lookback_bars": int(surge_lookback_bars),
                "surge_min_lookback_change_pct": float(surge_min_lookback_change_pct),
                "surge_min_latest_bar_change_pct": float(surge_min_latest_bar_change_pct),
                "surge_min_volume_ratio": float(surge_min_volume_ratio),
                "surge_min_volume_zscore": float(surge_min_volume_zscore),
                "surge_min_score": float(surge_min_score),
                "top_surge_candidates": top_surge_candidates,
                "min_recent_volume": float(min_recent_volume),
                "min_recent_dollar_volume": float(min_recent_dollar_volume),
                "liquidity_recent_bars": int(liquidity_recent_bars),
                "position_count": len(position_summary),
                "wide_scan": wide_scan_summary,
                "reentry_cooldown": reentry_cooldown,
            }
        )
        sort_columns = [column for column in ["auto_score", "confidence_pct", "volume_ratio", "change_pct"] if column in scanner.columns]
        if sort_columns:
            scanner = scanner.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    else:
        auto_selection = {"enabled": False, "symbol": str(scanner.iloc[0].get("symbol", "")), "eligible": False, "reason": ["AUTO_SELECTION_DISABLED"]}
        auto_selection.update(
            {
                "candidate_universe_mode": candidate_universe_mode,
                "candidate_universe_size": len(tickers),
                "liquidity_filtered_count": len(liquidity_filtered),
                "liquidity_passed_count": liquidity_passed_count,
                "liquid_buy_candidate_count": liquid_buy_candidate_count,
                "surge_focus_enabled": bool(surge_focus_enabled),
                "surge_candidate_count": surge_candidate_count,
                "top_surge_candidates": top_surge_candidates,
                "position_count": len(position_summary),
                "wide_scan": wide_scan_summary,
                "reentry_cooldown": reentry_cooldown,
            }
        )
        sort_columns = [column for column in ["change_pct", "confidence_pct", "auto_score"] if column in scanner.columns]
        if sort_columns:
            scanner = scanner.sort_values(sort_columns, ascending=[False] * len(sort_columns))

    selected = str(auto_selection.get("symbol") or scanner.iloc[0].get("symbol", ""))
    if selected not in loaded:
        selected = str(scanner.iloc[0].get("symbol", ""))
    _, features, result, plan, _model_data = loaded[selected]

    reference_entry = plan.breakout_entry if str(plan.direction) in {"濡??곗쐞", "LONG", "BUY"} else plan.pullback_entry
    sizing_account_krw = account_krw
    if auto_trade_mode == "live" and real_account_snapshot is not None:
        sizing_account_krw = float(real_account_snapshot.cash_buying_power) * usdkrw
    position = calculate_position(sizing_account_krw, usdkrw, reference_entry, plan.stop_loss, risk_pct)
    previous_signal = read_latest_signal()
    alerts = detect_alerts(plan, previous_signal)
    market_events = detect_market_events(plan, previous_signal)
    quote, quote_warning = load_quote_snapshot(
        source=TOSS_DATA_SOURCE,
        symbol=selected,
        features=features,
        period="Toss API",
        interval=interval,
        toss_token=toss_token,
        toss_client_id=toss_client_id,
        toss_client_secret=toss_client_secret,
        toss_account_seq=toss_account_seq,
        toss_base_url=toss_base_url,
    )
    data_warnings = [quote_warning] if quote_warning else []
    try:
        prediction_metrics = formula_prediction_to_dict(
            _calculate_formula_prediction(
                features,
                quote=quote,
                horizon_bars=max(1, int(intraday_projection_horizon_bars)),
            )
        )
    except Exception as exc:
        prediction_metrics = {}
        data_warnings.append(f"formula prediction failed: {exc}")

    payload = signal_payload(
        selected,
        result,
        plan,
        position,
        alerts,
        market_events,
        quote=quote,
        data_source=TOSS_DATA_SOURCE,
        data_warnings=data_warnings,
        prediction_metrics=prediction_metrics,
    )
    payload = apply_intraday_projection_targeting(
        payload,
        enabled=intraday_targeting_enabled,
        capture_pct=intraday_target_capture_pct,
        min_gross_target_pct=intraday_target_min_gross_pct,
        max_gross_target_pct=intraday_target_max_gross_pct,
    )
    payload["support_resistance"] = _support_resistance_signal(
        features,
        current_price=_as_float(payload.get("current_price")),
        enabled=structure_guard_enabled,
        lookback_bars=structure_lookback_bars,
        confirm_bars=structure_confirm_bars,
        min_break_pct=structure_min_break_pct,
        atr_buffer_multiplier=structure_atr_buffer_multiplier,
        min_volume_ratio=structure_min_volume_ratio,
        min_volume_zscore=structure_min_volume_zscore,
    )
    payload["rebound_potential"] = _rebound_potential_signal(
        features,
        current_price=_as_float(payload.get("current_price")),
        entry_price=_as_float(payload.get("reference_entry")),
        stop_loss=_as_float(payload.get("stop_loss")),
        enabled=target_decay_rebound_protection_enabled,
        min_score=target_decay_rebound_min_score,
        min_samples=target_decay_rebound_min_samples,
        lookback_bars=target_decay_rebound_lookback_bars,
        forward_bars=target_decay_rebound_forward_bars,
    )
    account_usd = float(position.get("account_usd", 0) or 0)
    effective_daily_order_guard_max_loss = effective_daily_order_guard_max_loss_usd(
        configured_max_loss_usd=daily_order_guard_max_loss_usd,
        account_usd=account_usd,
        daily_loss_limit_pct=daily_loss_limit_pct,
        daily_loss_limit_usd=daily_loss_limit_usd,
    )
    payload["daily_goal"] = {
        "target_min_pct": round(float(daily_target_min_pct), 2),
        "target_max_pct": round(float(daily_target_max_pct), 2),
        "realized_pnl_usd": round(float(daily_realized_pnl_usd), 2),
        "realized_pnl_pct": round(float(daily_realized_pnl_usd) / account_usd * 100, 2) if account_usd > 0 else 0.0,
        "profit_target_is_blocking": False,
        "profit_lock_trigger_pct": round(float(daily_profit_lock_trigger_pct), 2),
        "profit_lock_giveback_pct": round(float(daily_profit_lock_giveback_pct), 2),
    }
    payload = apply_data_quality_harness(
        payload,
        DataQualityGuard(
            enabled=data_guard_enabled,
            max_age_minutes=max_data_age_minutes,
            max_spread_pct=max_spread_pct,
        ),
    )
    payload = apply_daily_profit_lock_harness(
        payload,
        DailyProfitLock(
            account_usd=account_usd,
            realized_pnl_usd=daily_realized_pnl_usd,
            trigger_pct=daily_profit_lock_trigger_pct,
            giveback_pct=daily_profit_lock_giveback_pct,
        ),
        DailyProfitLockLedger(DAILY_PROFIT_LOCK_PATH),
    )
    payload = apply_daily_loss_harness(
        payload,
        DailyLossLimit(
            account_usd=float(position["account_usd"]),
            realized_pnl_usd=daily_realized_pnl_usd,
            max_loss_pct=daily_loss_limit_pct / 100,
            max_loss_usd=daily_loss_limit_usd,
        ),
    )
    payload = apply_buying_power_sizing_harness(
        payload,
        account=account_snapshot,
        max_order_notional_usd=max_auto_order_notional_usd,
        usdkrw=usdkrw,
    )
    if exit_candidate is None:
        payload = apply_entry_liquidity_harness(
            payload,
            auto_selection=auto_selection,
            enabled=entry_liquidity_guard_enabled,
            min_recent_dollar_volume_usd=entry_liquidity_min_recent_dollar_volume_usd,
            min_latest_dollar_volume_usd=entry_liquidity_min_latest_dollar_volume_usd,
            min_volume_ratio=entry_liquidity_min_volume_ratio,
        )
        payload = apply_entry_structure_harness(
            payload,
            enabled=structure_guard_enabled,
            block_lower_breakdown=entry_structure_block_lower_breakdown,
            require_upper_breakout=entry_structure_require_upper_breakout,
        )
        payload = apply_entry_edge_harness(
            payload,
            commission_rate_pct=effective_commission_rate_pct,
            enabled=entry_edge_guard_enabled,
            min_gross_target_pct=entry_edge_min_gross_target_pct,
            min_net_target_pct=entry_edge_min_net_target_pct,
            fee_multiplier=entry_edge_fee_multiplier,
            slippage_buffer_pct=entry_edge_slippage_buffer_pct,
            min_order_notional_usd=entry_edge_min_order_notional_usd,
            min_projected_target_coverage_pct=entry_edge_min_projected_target_coverage_pct,
            min_expected_move_pct=entry_edge_min_expected_move_pct,
        )
        payload = apply_daily_order_guard(
            payload,
            stats=daily_order_guard_stats,
            enabled=daily_order_guard_enabled,
            max_loss_usd=effective_daily_order_guard_max_loss,
            max_consecutive_losses=daily_order_guard_max_consecutive_losses,
        )
    if account_sync_error:
        payload.setdefault("risk_flags", []).append(account_sync_error)
    if holdings_sync_error:
        payload.setdefault("risk_flags", []).append(holdings_sync_error)
    if commissions_sync_error:
        payload.setdefault("risk_flags", []).append(commissions_sync_error)
    payload["position_summary"] = position_summary
    payload["position_count"] = len(position_summary)
    payload["auto_selection"] = dict(auto_selection)
    if exit_candidate is not None:
        exit_current_price = _as_float(exit_candidate.get("current_price"), _as_float(payload.get("current_price")))
        exit_stop_loss = _as_float(exit_candidate.get("stop_loss"), _as_float(payload.get("stop_loss")))
        exit_target_price = _as_float(
            exit_candidate.get("target_price"),
            _as_float((payload.get("take_profit") or [0])[0]),
        )
        if bool(exit_candidate.get("manual_exit", False)):
            if str(exit_candidate.get("reason")) == "MANUAL_POSITION_STOP_LOSS":
                rechecked_reason = (
                    "MANUAL_POSITION_STOP_LOSS" if exit_current_price <= exit_stop_loss else "EXIT_TRIGGER_RECHECK_FAILED"
                )
            elif str(exit_candidate.get("reason")) == "MANUAL_POSITION_TAKE_PROFIT":
                rechecked_reason = (
                    "MANUAL_POSITION_TAKE_PROFIT" if exit_current_price >= exit_target_price else "EXIT_TRIGGER_RECHECK_FAILED"
                )
            elif str(exit_candidate.get("reason")) == "TARGET_EXTENSION_ACTIVE":
                rechecked_reason = "TARGET_EXTENSION_ACTIVE"
            else:
                rechecked_reason = "EXIT_TRIGGER_RECHECK_FAILED"
        elif str(exit_candidate.get("reason")) == "TARGET_EXTENSION_ACTIVE":
            rechecked_reason = "TARGET_EXTENSION_ACTIVE"
        elif str(exit_candidate.get("reason")) == "SUPPORT_BREAKDOWN_EXIT":
            support_resistance = (
                exit_candidate.get("support_resistance") if isinstance(exit_candidate.get("support_resistance"), dict) else {}
            )
            if selected in loaded:
                _prices, exit_features, _exit_result, _exit_plan, _model_data = loaded[selected]
                support_resistance = _support_resistance_signal(
                    exit_features,
                    current_price=exit_current_price,
                    enabled=support_breakdown_exit_enabled and structure_guard_enabled,
                    lookback_bars=structure_lookback_bars,
                    confirm_bars=structure_confirm_bars,
                    min_break_pct=structure_min_break_pct,
                    atr_buffer_multiplier=structure_atr_buffer_multiplier,
                    min_volume_ratio=structure_min_volume_ratio,
                    min_volume_zscore=structure_min_volume_zscore,
                )
            rechecked_reason = "SUPPORT_BREAKDOWN_EXIT" if bool(support_resistance.get("lower_breakdown")) else "EXIT_TRIGGER_RECHECK_FAILED"
            exit_candidate["support_resistance"] = support_resistance
        elif str(exit_candidate.get("reason")) == "EARLY_EXIT_TARGET_PROBABILITY_DECAY":
            target_decay_exit = exit_candidate.get("target_decay_exit") if isinstance(exit_candidate.get("target_decay_exit"), dict) else {}
            if selected in loaded:
                _prices, exit_features, exit_result, _exit_plan, _model_data = loaded[selected]
                target_decay_exit = _target_decay_exit_signal(
                    exit_features,
                    exit_result,
                    current_price=exit_current_price,
                    entry_price=_as_float(exit_candidate.get("entry_price")),
                    fee_adjusted_return_pct=_as_float(exit_candidate.get("fee_adjusted_return_pct")),
                    stop_loss=exit_stop_loss,
                    target_price=exit_target_price,
                    enabled=target_decay_exit_enabled,
                    min_probability_pct=target_decay_exit_min_probability_pct,
                    min_target_gap_pct=target_decay_exit_min_target_gap_pct,
                    max_volume_ratio=target_decay_exit_max_volume_ratio,
                    max_volume_zscore=target_decay_exit_max_volume_zscore,
                    min_weak_bars=target_decay_exit_min_weak_bars,
                    max_fee_adjusted_return_pct=target_decay_exit_max_fee_adjusted_return_pct,
                    rebound_protection_enabled=target_decay_rebound_protection_enabled,
                    rebound_min_score=target_decay_rebound_min_score,
                    rebound_min_samples=target_decay_rebound_min_samples,
                    rebound_lookback_bars=target_decay_rebound_lookback_bars,
                    rebound_forward_bars=target_decay_rebound_forward_bars,
                )
            rechecked_reason = (
                "EARLY_EXIT_TARGET_PROBABILITY_DECAY"
                if bool(target_decay_exit.get("active"))
                else "EXIT_TRIGGER_RECHECK_FAILED"
            )
            exit_candidate["target_decay_exit"] = target_decay_exit
        else:
            rechecked_reason = _exit_reason(
                exit_current_price,
                exit_stop_loss,
                exit_target_price,
                target_exit_tolerance_pct,
            )
        exit_holding_qty = _as_float(exit_candidate.get("holding_qty"))
        holding_whole_qty = _whole_share_quantity(exit_holding_qty)
        sellable_quantity = holding_whole_qty
        sellable_error = ""
        exit_is_executable = bool(rechecked_reason and rechecked_reason != "TARGET_EXTENSION_ACTIVE")
        if exit_is_executable and toss_broker_adapter is not None and auto_trade_mode == "live":
            try:
                sellable_quantity = _whole_share_quantity(toss_broker_adapter.get_sellable_quantity(selected))
            except Exception as exc:
                sellable_error = f"Toss sellable quantity sync failed: {exc}"
        exit_quantity = min(holding_whole_qty, sellable_quantity) if exit_is_executable else 0
        exit_candidate = {
            **exit_candidate,
            "reason": rechecked_reason or "EXIT_TRIGGER_RECHECK_FAILED",
            "quantity": exit_quantity,
            "sellable_quantity": sellable_quantity,
        }
        target_extension = exit_candidate.get("target_extension") if isinstance(exit_candidate.get("target_extension"), dict) else {}
        target_decay_exit = exit_candidate.get("target_decay_exit") if isinstance(exit_candidate.get("target_decay_exit"), dict) else {}
        support_resistance_exit = (
            exit_candidate.get("support_resistance") if isinstance(exit_candidate.get("support_resistance"), dict) else {}
        )
        payload["auto_exit"] = {
            "enabled": True,
            "reason": exit_candidate["reason"],
            "executable": bool(exit_is_executable),
            "manual_exit": bool(exit_candidate.get("manual_exit", False)),
            "entry_price": round(_as_float(exit_candidate.get("entry_price")), 4),
            "fee_adjusted_return_pct": round(_as_float(exit_candidate.get("fee_adjusted_return_pct")), 4),
            "fee_adjusted_pnl_usd": round(_as_float(exit_candidate.get("fee_adjusted_pnl_usd")), 4),
            "holding_qty": exit_holding_qty,
            "sellable_quantity": sellable_quantity,
            "quantity": exit_quantity,
            "stop_loss": round(exit_stop_loss, 4),
            "target_price": round(exit_target_price, 4),
            "original_target_price": round(_as_float(exit_candidate.get("original_target_price"), exit_target_price), 4),
            "target_exit_tolerance_pct": round(float(target_exit_tolerance_pct), 4),
            "target_extension": target_extension,
            "target_decay_exit": target_decay_exit,
            "support_resistance": support_resistance_exit,
        }
        payload.setdefault("alerts", []).append(f"AUTO_EXIT_PRIORITY: {exit_candidate['reason']}")
        if sellable_error:
            payload.setdefault("risk_flags", []).append(sellable_error)
    if auto_symbol_selection_enabled and not auto_selection.get("eligible"):
        payload.setdefault("alerts", []).append("?먮룞 ?좎젙 湲곗? 異⑹” 醫낅ぉ ?놁쓬")
    save_latest_signal(payload)

    order_ledger = DuplicateOrderLedger(ORDER_LEDGER_PATH)
    trade_date = str(payload.get("latest_date", ""))[:10]
    current_session_entries = order_ledger.count_entries(trade_date=trade_date, symbol=selected)
    effective_trade_prepare_mode = (
        trade_prepare_mode
        if not auto_symbol_selection_enabled or bool(auto_selection.get("eligible"))
        else "disabled"
    )
    if exit_candidate is not None:
        trade_intent_source = _build_exit_trade_intent(
            symbol=selected,
            quantity=int(exit_candidate.get("quantity", 0)),
            current_price=_as_float(exit_candidate.get("current_price"), _as_float(payload.get("current_price"))),
            stop_loss=_as_float(exit_candidate.get("stop_loss"), _as_float(payload.get("stop_loss"))),
            take_profit=[
                _as_float(exit_candidate.get("target_price"), _as_float((payload.get("take_profit") or [0, 0])[0])),
                _as_float((payload.get("take_profit") or [0, 0])[1] if len(payload.get("take_profit") or []) > 1 else 0),
            ],
            account_usd=float(position["account_usd"]),
            daily_realized_pnl_usd=daily_realized_pnl_usd,
            daily_loss_limit_pct=daily_loss_limit_pct,
            daily_loss_limit_usd=daily_loss_limit_usd,
            data_quality=payload.get("data_quality", {}),
            reason=str(exit_candidate.get("reason") or "AUTO_EXIT"),
            holding_qty=_as_float(exit_candidate.get("holding_qty")),
            current_session_entries=current_session_entries,
            max_session_entries=max_session_entries,
            manual_approval_complete=manual_approval_complete,
        )
    else:
        trade_intent_source = build_trade_intent(
            symbol=selected,
            action=payload["action"],
            current_price=payload["current_price"],
            reference_entry=payload["reference_entry"],
            stop_loss=payload["stop_loss"],
            take_profit=payload["take_profit"],
            recommended_shares=payload["recommended_shares"],
            mode=effective_trade_prepare_mode,
            account_usd=float(position["account_usd"]),
            daily_realized_pnl_usd=daily_realized_pnl_usd,
            daily_loss_limit_pct=daily_loss_limit_pct,
            daily_loss_limit_usd=daily_loss_limit_usd,
            data_quality=payload.get("data_quality", {}),
            manual_approval_complete=manual_approval_complete,
            holding_qty=holding_qty,
            current_session_entries=current_session_entries,
            max_session_entries=max_session_entries,
        )
    trade_intent_obj = apply_duplicate_order_harness(
        trade_intent_source,
        payload,
        order_ledger,
    )
    trade_intent = trade_intent_to_dict(trade_intent_obj)
    save_trade_intent(trade_intent)

    execution_plan_obj = build_execution_plan(
        trade_intent_obj,
        account=account_snapshot,
        execution_enabled=bool(auto_trade_mode in {"paper", "live"} and auto_preapproval),
        user_confirmation=bool(auto_preapproval),
    )
    execution_plan = execution_plan_to_dict(execution_plan_obj)
    save_execution_plan(execution_plan)

    auto_trade_obj = run_auto_trade(
        execution_plan_obj,
        AutoTradeConfig(
            enabled=auto_trade_mode != "disabled",
            mode=auto_trade_mode,
            kill_switch=auto_kill_switch,
            max_order_notional_usd=0.0 if exit_candidate is not None else max_auto_order_notional_usd,
            allowed_symbols=tickers,
        ),
        adapter=(
            PaperBrokerExecutionAdapter(account_snapshot)
            if auto_trade_mode == "paper"
            else toss_broker_adapter
            if auto_trade_mode == "live" and toss_live_order_enabled
            else None
        ),
        submitted_client_order_ids=submitted_order_ids_from_events(AUTO_TRADE_EVENTS_PATH),
    )
    auto_trade = auto_trade_decision_to_dict(auto_trade_obj)
    save_auto_trade_decision(auto_trade)
    append_auto_trade_event(auto_trade, AUTO_TRADE_EVENTS_PATH)
    if payload.get("alerts"):
        append_signal_event(payload)

    event = {
        "status": auto_trade.get("status"),
        "state": auto_trade.get("state"),
        "mode": auto_trade.get("mode"),
        "symbol": selected,
        "auto_selection": auto_selection,
        "trade_intent_status": trade_intent.get("status"),
        "trade_intent_side": trade_intent.get("side"),
        "trade_intent_quantity": trade_intent.get("quantity"),
        "execution_status": execution_plan.get("status"),
        "order_client_id": (auto_trade.get("order") or {}).get("client_order_id"),
        "broker_order_id": ((auto_trade.get("execution_result") or {}).get("broker_order_id")),
        "reasons": list(trade_intent.get("reason") or [])
        + list(execution_plan.get("reason") or [])
        + list(auto_trade.get("reason") or []),
        "candidate_universe_mode": candidate_universe_mode,
        "candidate_universe_size": len(tickers),
        "broker_blocked_symbols": broker_blocked_symbols,
        "broker_blocked_symbols_count": len(broker_blocked_symbols),
        "liquidity_filtered_count": len(liquidity_filtered),
        "liquidity_passed_count": liquidity_passed_count,
        "liquid_buy_candidate_count": liquid_buy_candidate_count,
        "surge_focus_enabled": bool(surge_focus_enabled),
        "surge_candidate_count": surge_candidate_count,
        "surge_lookback_bars": int(surge_lookback_bars),
        "surge_min_lookback_change_pct": float(surge_min_lookback_change_pct),
        "surge_min_latest_bar_change_pct": float(surge_min_latest_bar_change_pct),
        "surge_min_volume_ratio": float(surge_min_volume_ratio),
        "surge_min_volume_zscore": float(surge_min_volume_zscore),
        "surge_min_score": float(surge_min_score),
        "top_surge_candidates": top_surge_candidates,
        "candidate_scores": candidate_scores,
        "candidate_scores_count": len(candidate_scores),
        "wide_scan": wide_scan_summary,
        "reentry_cooldown": reentry_cooldown,
        "daily_order_guard": daily_order_guard_stats,
        "entry_liquidity": payload.get("entry_liquidity"),
        "entry_structure": payload.get("entry_structure"),
        "entry_edge": payload.get("entry_edge"),
        "intraday_targeting": payload.get("intraday_targeting"),
        "prediction_metrics": payload.get("prediction_metrics"),
        "support_resistance": payload.get("support_resistance"),
        "rebound_potential": payload.get("rebound_potential"),
        "current_price": payload.get("current_price"),
        "reference_entry": payload.get("reference_entry"),
        "stop_loss": payload.get("stop_loss"),
        "take_profit": payload.get("take_profit"),
        "holding_symbols_count": len(real_holding_qty),
        "position_summary": position_summary,
        "position_count": len(position_summary),
        "auto_exit": payload.get("auto_exit"),
        "failures": failures[:20],
    }
    append_session_event(event)
    return event


def run_loop(args: argparse.Namespace) -> int:
    stop_file = Path(args.stop_file).resolve()
    lock_file = Path(args.lock_file).resolve()
    if args.clear_stop and stop_file.exists():
        stop_file.unlink()

    acquire_worker_lock(lock_file)
    try:
        print(f"[INFO] auto session worker started: interval={args.interval_seconds}s stop_file={stop_file}")
        print(f"[INFO] lock_file={lock_file}")
        print("[INFO] live trading is controlled by configured auto-selection and order gates.")
        cycle = 0
        consecutive_errors = 0
        last_archive_check = 0.0
        while True:
            update_worker_lock(lock_file)
            if stop_file.exists():
                settings = _read_settings()
                archived_sessions: list[str] = []
                try:
                    archived = maybe_run_daily_archive(settings)
                    archived_sessions = [str(summary.get("session_date")) for summary in archived]
                except Exception as exc:
                    append_session_event({"status": "DAILY_ARCHIVE_ERROR", "message": str(exc)})
                print(f"[INFO] stop file detected, exiting: {stop_file}")
                append_session_event(
                    {
                        "status": "STOPPED",
                        "message": f"stop file detected: {stop_file}",
                        "archived_sessions": archived_sessions,
                    }
                )
                return 0
            cycle += 1
            try:
                settings = _read_settings()
                event = run_cycle(settings, watchlist_override=args.watchlist)
                consecutive_errors = 0
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"{event['symbol']} {event['status']} {event['state']} "
                    f"intent={event['trade_intent_status']} qty={event['trade_intent_quantity']}"
                )
                archive_interval = max(60.0, _setting_float(settings, "daily_archive_check_interval_seconds", 300.0))
                now_monotonic = time.monotonic()
                if now_monotonic - last_archive_check >= archive_interval:
                    last_archive_check = now_monotonic
                    try:
                        archived = maybe_run_daily_archive(settings)
                        if archived:
                            append_session_event(
                                {
                                    "status": "DAILY_ARCHIVE",
                                    "archived_sessions": [str(summary.get("session_date")) for summary in archived],
                                    "latest_summary": str(LATEST_DAILY_SUMMARY_PATH),
                                }
                            )
                    except Exception as exc:
                        append_session_event({"status": "DAILY_ARCHIVE_ERROR", "message": str(exc)})
            except requests.HTTPError as exc:
                consecutive_errors += 1
                status = exc.response.status_code if exc.response is not None else "HTTP"
                message = f"Toss API ?붿껌 ?ㅽ뙣: {status} {exc}"
                print(f"[ERROR] {message}", file=sys.stderr)
                append_session_event({"status": "ERROR", "message": message, "consecutive_errors": consecutive_errors})
            except Exception as exc:
                consecutive_errors += 1
                print(f"[ERROR] {exc}", file=sys.stderr)
                append_session_event({"status": "ERROR", "message": str(exc), "consecutive_errors": consecutive_errors})

            if args.once or (args.max_cycles and cycle >= args.max_cycles):
                return 0
            if consecutive_errors >= int(args.max_consecutive_errors):
                message = f"max consecutive errors reached: {consecutive_errors}"
                print(f"[ERROR] {message}", file=sys.stderr)
                append_session_event({"status": "STOPPED", "message": message})
                return 1
            update_worker_lock(lock_file)
            time.sleep(max(1.0, float(args.interval_seconds)))
    finally:
        release_worker_lock(lock_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="?μ쨷 ?먮룞 ?좎젙/?먮룞留ㅻℓ 寃뚯씠???뚯빱")
    parser.add_argument("--interval-seconds", type=float, default=60.0, help="諛섎났 ?ㅽ뻾 媛꾧꺽")
    parser.add_argument("--max-cycles", type=int, default=0, help="0?대㈃ stop ?뚯씪???앷만 ?뚭퉴吏 ?ㅽ뻾")
    parser.add_argument("--once", action="store_true", help="??踰덈쭔 ?ㅽ뻾")
    parser.add_argument("--watchlist", default="", help="?쇳몴濡?援щ텇???꾩떆 媛먯떆紐⑸줉")
    parser.add_argument("--stop-file", default=str(STOP_FILE), help="???뚯씪???앷린硫??뚯빱 醫낅즺")
    parser.add_argument("--lock-file", default=str(LOCK_FILE), help="以묐났 ?ㅽ뻾 諛⑹? lock ?뚯씪")
    parser.add_argument("--max-consecutive-errors", type=int, default=5, help="?곗냽 ?ㅻ쪟媛 ???잛닔???우쑝硫?醫낅즺")
    parser.add_argument("--clear-stop", action="store_true", help="?쒖옉 ??stop ?뚯씪 ??젣")
    return parser


def main() -> int:
    return run_loop(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
