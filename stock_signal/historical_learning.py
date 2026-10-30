from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .market_data import YFinanceMarketDataAdapter
from .prediction_metrics import add_prediction_features
from .validators import validate_price_data


DEFAULT_MAJOR_SYMBOLS = [
    "AAPL",
    "NVDA",
    "TSLA",
    "MSFT",
    "AMZN",
    "META",
    "AMD",
    "GOOGL",
    "AVGO",
    "SMCI",
    "QQQ",
    "SOXL",
    "TQQQ",
]

DEFAULT_CONTEXT_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "XLK",
    "SOXX",
    "^VIX",
    "^TNX",
]

CONTEXT_PREFIXES = {
    "SPY": "spy",
    "QQQ": "qqq",
    "IWM": "iwm",
    "DIA": "dia",
    "XLK": "xlk",
    "SOXX": "soxx",
    "^VIX": "vix",
    "^TNX": "tnx",
}


FEATURE_COLUMNS = [
    "return_1",
    "return_2",
    "return_3",
    "return_5",
    "return_10",
    "return_20",
    "log_return_1",
    "abs_return_1",
    "gap_pct",
    "intraday_return_pct",
    "high_low_range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "ema_spread_pct",
    "ema_9_slope_5",
    "ma_20_slope_5",
    "ma_5_distance_pct",
    "ma_10_distance_pct",
    "ma_20_distance_pct",
    "ma_50_distance_pct",
    "ma_100_distance_pct",
    "ma_200_distance_pct",
    "ma_50_slope_10",
    "ma_100_slope_20",
    "macd_pct",
    "macd_signal_pct",
    "macd_hist_pct",
    "volume_ratio",
    "volume_ratio_5",
    "volume_ratio_60",
    "volume_zscore_20",
    "dollar_volume_zscore_20",
    "relative_volume_trend_5_20",
    "log_latest_dollar_volume",
    "log_recent_dollar_volume_20",
    "atr_pct",
    "atr_ratio_5_20",
    "rsi_14",
    "vwap_distance_pct",
    "boll_width_pct",
    "boll_position",
    "donchian_position_20",
    "donchian_position_55",
    "drawdown_from_high_20",
    "drawdown_from_high_60",
    "stoch_k",
    "stoch_d",
    "williams_r_14",
    "mfi_14",
    "cmf_20",
    "cci_20",
    "adx_14",
    "obv_slope_5",
    "close_location",
    "body_pct",
    "volatility_20",
    "rolling_volatility_5",
    "rolling_volatility_10",
    "rolling_volatility_60",
    "breakout_gap_pct",
    "support_gap_pct",
    "relative_spy_return_5",
    "relative_qqq_return_5",
    "relative_spy_return_20",
    "relative_qqq_return_20",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_month_start",
    "is_month_end",
    "is_etf",
    "is_leveraged_etf",
    "is_semiconductor",
    "is_mega_cap",
]

for _prefix in CONTEXT_PREFIXES.values():
    FEATURE_COLUMNS.extend(
        [
            f"context_{_prefix}_return_1",
            f"context_{_prefix}_return_5",
            f"context_{_prefix}_return_20",
            f"context_{_prefix}_ma20_distance_pct",
            f"context_{_prefix}_volatility_20",
        ]
    )


@dataclass
class HistoricalLearningConfig:
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_MAJOR_SYMBOLS))
    context_symbols: list[str] = field(default_factory=lambda: list(DEFAULT_CONTEXT_SYMBOLS))
    period: str = "60d"
    interval: str = "5m"
    prepost: bool = True
    target_pct: float = 1.2
    stop_pct: float = 0.6
    horizon_bars: int = 6
    min_rows_per_symbol: int = 120
    min_labeled_rows_per_symbol: int = 120
    strict_data_quality: bool = True
    max_abs_return_pct: float = 60.0
    max_stale_days: int = 14
    test_size: float = 0.25
    random_state: int = 42
    output_dir: str = "model_artifacts/historical_chart"


@dataclass
class HistoricalLearningResult:
    status: str
    config: dict[str, Any]
    rows: int
    symbols: list[str]
    skipped_symbols: dict[str, str]
    data_quality: dict[str, Any]
    positive_rate: float
    metrics: dict[str, Any]
    feature_importance: list[dict[str, Any]]
    files: dict[str, str]


def _future_outcome(
    data: pd.DataFrame,
    row_pos: int,
    *,
    target_pct: float,
    stop_pct: float,
    horizon_bars: int,
) -> dict[str, Any] | None:
    entry = float(data["Close"].iloc[row_pos])
    if entry <= 0:
        return None
    target = entry * (1 + max(0.0, float(target_pct)) / 100)
    stop = entry * (1 - max(0.0, float(stop_pct)) / 100)
    future = data.iloc[row_pos + 1 : row_pos + 1 + max(1, int(horizon_bars))]
    if future.empty:
        return None

    min_low = float(future["Low"].min())
    max_high = float(future["High"].max())
    last_close = float(future["Close"].iloc[-1])
    for offset, (_index, row) in enumerate(future.iterrows(), start=1):
        high_hit = float(row["High"]) >= target
        low_hit = float(row["Low"]) <= stop
        if high_hit and low_hit:
            return {
                "outcome": "AMBIGUOUS",
                "target_first": np.nan,
                "bars_to_event": offset,
                "entry_price": entry,
                "target_price": target,
                "stop_price": stop,
                "max_runup_pct": (max_high / entry - 1) * 100,
                "max_drawdown_pct": (min_low / entry - 1) * 100,
                "horizon_return_pct": (last_close / entry - 1) * 100,
            }
        if high_hit:
            return {
                "outcome": "TARGET_FIRST",
                "target_first": 1,
                "bars_to_event": offset,
                "entry_price": entry,
                "target_price": target,
                "stop_price": stop,
                "max_runup_pct": (max_high / entry - 1) * 100,
                "max_drawdown_pct": (min_low / entry - 1) * 100,
                "horizon_return_pct": (last_close / entry - 1) * 100,
            }
        if low_hit:
            return {
                "outcome": "STOP_FIRST",
                "target_first": 0,
                "bars_to_event": offset,
                "entry_price": entry,
                "target_price": target,
                "stop_price": stop,
                "max_runup_pct": (max_high / entry - 1) * 100,
                "max_drawdown_pct": (min_low / entry - 1) * 100,
                "horizon_return_pct": (last_close / entry - 1) * 100,
            }

    return {
        "outcome": "NO_HIT",
        "target_first": 0,
        "bars_to_event": max(1, int(horizon_bars)),
        "entry_price": entry,
        "target_price": target,
        "stop_price": stop,
        "max_runup_pct": (max_high / entry - 1) * 100,
        "max_drawdown_pct": (min_low / entry - 1) * 100,
        "horizon_return_pct": (last_close / entry - 1) * 100,
    }


def _pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods) * 100


def _rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_p = min_periods or max(3, window // 4)
    mean = series.rolling(window, min_periods=min_p).mean()
    std = series.rolling(window, min_periods=min_p).std().replace(0, np.nan)
    return (series - mean) / std


def _money_flow_index(data: pd.DataFrame, window: int = 14) -> pd.Series:
    typical = (data["High"] + data["Low"] + data["Close"]) / 3
    money_flow = typical * data["Volume"]
    direction = np.sign(typical.diff()).fillna(0)
    positive_flow = money_flow.where(direction > 0, 0.0).rolling(window, min_periods=5).sum()
    negative_flow = money_flow.where(direction < 0, 0.0).rolling(window, min_periods=5).sum()
    ratio = positive_flow / negative_flow.replace(0, np.nan)
    mfi = 100 - (100 / (1 + ratio))
    mfi[(negative_flow == 0) & (positive_flow > 0)] = 100
    mfi[(negative_flow == 0) & (positive_flow == 0)] = 50
    return mfi


def _chaikin_money_flow(data: pd.DataFrame, window: int = 20) -> pd.Series:
    spread = (data["High"] - data["Low"]).replace(0, np.nan)
    multiplier = (((data["Close"] - data["Low"]) - (data["High"] - data["Close"])) / spread).fillna(0.0)
    money_flow_volume = multiplier * data["Volume"]
    volume_sum = data["Volume"].rolling(window, min_periods=5).sum().replace(0, np.nan)
    return money_flow_volume.rolling(window, min_periods=5).sum() / volume_sum


def _commodity_channel_index(data: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (data["High"] + data["Low"] + data["Close"]) / 3
    mean = typical.rolling(window, min_periods=5).mean()
    mad = (typical - mean).abs().rolling(window, min_periods=5).mean().replace(0, np.nan)
    return (typical - mean) / (0.015 * mad)


def _adx(data: pd.DataFrame, window: int = 14) -> pd.Series:
    high = data["High"]
    low = data["Low"]
    close = data["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=data.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=data.index)
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window, min_periods=5).sum().replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(window, min_periods=5).sum() / atr
    minus_di = 100 * minus_dm.rolling(window, min_periods=5).sum() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(window, min_periods=5).mean()


def _symbol_static_features(symbol: str) -> dict[str, float]:
    ticker = symbol.upper()
    etfs = {"QQQ", "SPY", "IWM", "DIA", "XLK", "SOXX", "SMH"}
    leveraged = {"TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "SPXL", "UVXY"}
    semis = {"NVDA", "AMD", "AVGO", "SMCI", "SOXL", "SOXX", "SMH", "AMAT", "LRCX", "MU", "ASML"}
    mega_caps = {"AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA"}
    return {
        "is_etf": float(ticker in etfs or ticker in leveraged),
        "is_leveraged_etf": float(ticker in leveraged),
        "is_semiconductor": float(ticker in semis),
        "is_mega_cap": float(ticker in mega_caps),
    }


def add_rich_chart_features(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    features = add_prediction_features(data)
    if features.empty:
        return features

    close = features["Close"]
    open_ = features["Open"]
    high = features["High"]
    low = features["Low"]
    volume = features["Volume"]
    dollar_volume = close * volume

    features["return_2"] = _pct_change(close, 2)
    features["return_3"] = _pct_change(close, 3)
    features["return_10"] = _pct_change(close, 10)
    features["return_20"] = _pct_change(close, 20)
    features["log_return_1"] = np.log(close / close.shift(1).replace(0, np.nan))
    features["abs_return_1"] = features["return_1"].abs()
    features["gap_pct"] = (open_ / close.shift(1).replace(0, np.nan) - 1) * 100
    features["intraday_return_pct"] = (close / open_.replace(0, np.nan) - 1) * 100
    features["high_low_range_pct"] = (high / low.replace(0, np.nan) - 1) * 100
    candle_range = (high - low).replace(0, np.nan)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    features["upper_wick_pct"] = (high - body_high) / close.replace(0, np.nan) * 100
    features["lower_wick_pct"] = (body_low - low) / close.replace(0, np.nan) * 100

    for window in [5, 10, 20, 50, 100, 200]:
        min_periods = min(window, max(5, window // 2))
        ma = close.rolling(window, min_periods=min_periods).mean()
        features[f"ma_{window}_distance_pct"] = (close / ma.replace(0, np.nan) - 1) * 100
    features["ma_50_slope_10"] = close.rolling(50, min_periods=25).mean().pct_change(10) * 100
    features["ma_100_slope_20"] = close.rolling(100, min_periods=50).mean().pct_change(20) * 100

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    features["macd_pct"] = macd / close.replace(0, np.nan) * 100
    features["macd_signal_pct"] = macd_signal / close.replace(0, np.nan) * 100
    features["macd_hist_pct"] = (macd - macd_signal) / close.replace(0, np.nan) * 100

    volume_ma_5 = volume.rolling(5, min_periods=3).mean()
    volume_ma_20 = volume.rolling(20, min_periods=5).mean()
    volume_ma_60 = volume.rolling(60, min_periods=20).mean()
    features["volume_ratio_5"] = volume / volume_ma_5.replace(0, np.nan)
    features["volume_ratio_60"] = volume / volume_ma_60.replace(0, np.nan)
    features["dollar_volume_zscore_20"] = _rolling_zscore(dollar_volume, 20, min_periods=5)
    features["relative_volume_trend_5_20"] = volume_ma_5 / volume_ma_20.replace(0, np.nan)
    features["log_latest_dollar_volume"] = np.log1p(dollar_volume)
    features["log_recent_dollar_volume_20"] = np.log1p(dollar_volume.rolling(20, min_periods=5).sum())

    atr_short = features["tr"].rolling(5, min_periods=3).mean()
    atr_long = features["tr"].rolling(20, min_periods=5).mean()
    features["atr_ratio_5_20"] = atr_short / atr_long.replace(0, np.nan)
    for window in [5, 10, 60]:
        features[f"rolling_volatility_{window}"] = close.pct_change().rolling(window, min_periods=3).std() * 100

    for window in [20, 55]:
        rolling_high = high.rolling(window, min_periods=5).max()
        rolling_low = low.rolling(window, min_periods=5).min()
        spread = (rolling_high - rolling_low).replace(0, np.nan)
        features[f"donchian_position_{window}"] = (close - rolling_low) / spread
    for window in [20, 60]:
        rolling_high = high.rolling(window, min_periods=5).max()
        features[f"drawdown_from_high_{window}"] = (close / rolling_high.replace(0, np.nan) - 1) * 100

    low_14 = low.rolling(14, min_periods=5).min()
    high_14 = high.rolling(14, min_periods=5).max()
    features["williams_r_14"] = -100 * (high_14 - close) / (high_14 - low_14).replace(0, np.nan)
    features["mfi_14"] = _money_flow_index(features, 14)
    features["cmf_20"] = _chaikin_money_flow(features, 20)
    features["cci_20"] = _commodity_channel_index(features, 20)
    features["adx_14"] = _adx(features, 14)

    features["breakout_gap_pct"] = (close / features["resistance_20_prev"].replace(0, np.nan) - 1) * 100
    features["support_gap_pct"] = (close / features["support_20_prev"].replace(0, np.nan) - 1) * 100

    index = pd.to_datetime(features.index)
    day_of_week = pd.Series(index.dayofweek, index=features.index).astype(float)
    month = pd.Series(index.month, index=features.index).astype(float)
    features["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    features["is_month_start"] = pd.Series(index.is_month_start, index=features.index).astype(float)
    features["is_month_end"] = pd.Series(index.is_month_end, index=features.index).astype(float)
    for name, value in _symbol_static_features(symbol).items():
        features[name] = value

    return features.replace([np.inf, -np.inf], np.nan)


def build_market_context_features(context_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, prices in context_data.items():
        prefix = CONTEXT_PREFIXES.get(symbol.upper(), symbol.lower().replace("^", "").replace("-", "_"))
        data = prices.copy().sort_index()
        if data.empty:
            continue
        close = pd.to_numeric(data["Close"], errors="coerce")
        frame = pd.DataFrame(index=data.index)
        frame[f"context_{prefix}_return_1"] = _pct_change(close, 1)
        frame[f"context_{prefix}_return_5"] = _pct_change(close, 5)
        frame[f"context_{prefix}_return_20"] = _pct_change(close, 20)
        ma20 = close.rolling(20, min_periods=5).mean()
        frame[f"context_{prefix}_ma20_distance_pct"] = (close / ma20.replace(0, np.nan) - 1) * 100
        frame[f"context_{prefix}_volatility_20"] = close.pct_change().rolling(20, min_periods=5).std() * 100
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    context = pd.concat(frames, axis=1).sort_index()
    return context.replace([np.inf, -np.inf], np.nan).ffill()


def build_labeled_chart_samples(
    symbol: str,
    prices: pd.DataFrame,
    *,
    target_pct: float = 1.2,
    stop_pct: float = 0.6,
    horizon_bars: int = 6,
    market_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    data = prices.copy()
    validate_price_data(data)
    features = add_rich_chart_features(data, symbol)
    if features.empty:
        return pd.DataFrame()

    if market_context is not None and not market_context.empty:
        context = market_context.reindex(features.index).ffill()
        features = features.join(context, how="left")
    context_columns = [column for column in FEATURE_COLUMNS if column.startswith("context_")]
    for column in context_columns:
        if column not in features.columns:
            features[column] = 0.0
    if "context_spy_return_5" in features.columns:
        features["relative_spy_return_5"] = features["return_5"] - features["context_spy_return_5"]
    if "context_qqq_return_5" in features.columns:
        features["relative_qqq_return_5"] = features["return_5"] - features["context_qqq_return_5"]
    if "context_spy_return_20" in features.columns:
        features["relative_spy_return_20"] = features["return_20"] - features["context_spy_return_20"]
    if "context_qqq_return_20" in features.columns:
        features["relative_qqq_return_20"] = features["return_20"] - features["context_qqq_return_20"]
    for column in ["relative_spy_return_5", "relative_qqq_return_5", "relative_spy_return_20", "relative_qqq_return_20"]:
        if column not in features.columns:
            features[column] = 0.0
    features = features.replace([np.inf, -np.inf], np.nan).ffill().dropna(subset=FEATURE_COLUMNS)
    if features.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for row_pos in range(0, len(features) - max(1, int(horizon_bars))):
        outcome = _future_outcome(
            features,
            row_pos,
            target_pct=target_pct,
            stop_pct=stop_pct,
            horizon_bars=horizon_bars,
        )
        if outcome is None or outcome["outcome"] == "AMBIGUOUS":
            continue
        row = features.iloc[row_pos]
        record = {
            "symbol": symbol.upper(),
            "timestamp": str(features.index[row_pos]),
            **{column: float(row[column]) for column in FEATURE_COLUMNS},
            **outcome,
        }
        records.append(record)
    return pd.DataFrame(records)


def _history_quality_report(symbol: str, candles: pd.DataFrame, config: HistoricalLearningConfig) -> dict[str, Any]:
    data = candles.copy()
    validate_price_data(data)
    if len(data) < int(config.min_rows_per_symbol):
        raise ValueError(f"not enough candles: {len(data)}")
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()
    if data.index.has_duplicates:
        raise ValueError("duplicate timestamps detected")

    numeric = data[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("non-numeric OHLCV values detected")
    if (numeric[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("non-positive OHLC prices detected")
    if (numeric["Volume"] <= 0).any():
        raise ValueError("non-positive volume detected")
    if (numeric["High"] < numeric[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError("invalid OHLC relationship: high below open/close/low")
    if (numeric["Low"] > numeric[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("invalid OHLC relationship: low above open/close/high")

    abs_return_pct = numeric["Close"].pct_change().abs().max() * 100
    if bool(config.strict_data_quality) and abs_return_pct > float(config.max_abs_return_pct):
        raise ValueError(f"extreme adjusted return detected: {abs_return_pct:.2f}%")

    first_ts = pd.to_datetime(data.index.min(), utc=True)
    last_ts = pd.to_datetime(data.index.max(), utc=True)
    return {
        "symbol": symbol,
        "rows": int(len(data)),
        "first_timestamp": str(first_ts),
        "last_timestamp": str(last_ts),
        "duplicate_timestamps": 0,
        "max_abs_return_pct": round(float(abs_return_pct), 4),
        "latest_close": round(float(numeric["Close"].iloc[-1]), 4),
        "latest_volume": round(float(numeric["Volume"].iloc[-1]), 2),
    }


def _drop_stale_histories(
    data: dict[str, pd.DataFrame],
    quality: dict[str, dict[str, Any]],
    *,
    max_stale_days: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, str]]:
    if not data:
        return data, quality, {}
    latest_by_symbol = {
        symbol: pd.to_datetime(frame.index.max(), utc=True)
        for symbol, frame in data.items()
        if not frame.empty
    }
    freshest = max(latest_by_symbol.values())
    cutoff = freshest - pd.Timedelta(days=max(0, int(max_stale_days)))
    kept: dict[str, pd.DataFrame] = {}
    kept_quality: dict[str, dict[str, Any]] = {}
    stale: dict[str, str] = {}
    for symbol, frame in data.items():
        latest = latest_by_symbol.get(symbol)
        if latest is None or latest < cutoff:
            stale[symbol] = f"stale history: latest={latest} freshest={freshest}"
            continue
        kept[symbol] = frame
        kept_quality[symbol] = quality[symbol]
    return kept, kept_quality, stale


def fetch_yfinance_history(
    config: HistoricalLearningConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, dict[str, Any]]]:
    data: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    quality: dict[str, dict[str, Any]] = {}
    adapter = YFinanceMarketDataAdapter(period=config.period, interval=config.interval, prepost=config.prepost)
    for symbol in config.symbols:
        normalized = str(symbol).strip().upper()
        if not normalized:
            continue
        try:
            candles = adapter.get_ohlcv(normalized)
            quality[normalized] = _history_quality_report(normalized, candles, config)
            data[normalized] = candles
        except Exception as exc:
            failures[normalized] = str(exc)
    data, quality, stale = _drop_stale_histories(data, quality, max_stale_days=config.max_stale_days)
    failures.update(stale)
    return data, failures, quality


def _context_quality_report(symbol: str, candles: pd.DataFrame) -> dict[str, Any]:
    data = candles.copy().sort_index()
    validate_price_data(data)
    numeric = data[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    first_ts = pd.to_datetime(data.index.min(), utc=True)
    last_ts = pd.to_datetime(data.index.max(), utc=True)
    return {
        "symbol": symbol,
        "rows": int(len(data)),
        "first_timestamp": str(first_ts),
        "last_timestamp": str(last_ts),
        "duplicate_timestamps": int(data.index.duplicated().sum()),
        "latest_close": round(float(numeric["Close"].iloc[-1]), 4),
        "latest_volume": round(float(numeric["Volume"].iloc[-1]), 2),
    }


def fetch_yfinance_context(
    config: HistoricalLearningConfig,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict[str, Any]]]:
    context_data: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    quality: dict[str, dict[str, Any]] = {}
    adapter = YFinanceMarketDataAdapter(period=config.period, interval=config.interval, prepost=config.prepost)
    for symbol in config.context_symbols:
        normalized = str(symbol).strip().upper()
        if not normalized:
            continue
        try:
            candles = adapter.get_ohlcv(normalized)
            if len(candles) < max(60, int(config.min_rows_per_symbol) // 2):
                raise ValueError(f"not enough context candles: {len(candles)}")
            quality[normalized] = _context_quality_report(normalized, candles)
            context_data[normalized] = candles
        except Exception as exc:
            failures[normalized] = str(exc)
    return build_market_context_features(context_data), failures, quality


def build_historical_dataset(
    price_data: dict[str, pd.DataFrame],
    config: HistoricalLearningConfig,
    market_context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    samples: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    for symbol, prices in price_data.items():
        try:
            symbol_samples = build_labeled_chart_samples(
                symbol,
                prices,
                target_pct=config.target_pct,
                stop_pct=config.stop_pct,
                horizon_bars=config.horizon_bars,
                market_context=market_context,
            )
            if len(symbol_samples) < int(config.min_labeled_rows_per_symbol):
                raise ValueError(f"not enough labeled samples: {len(symbol_samples)}")
            samples.append(symbol_samples)
        except Exception as exc:
            failures[symbol] = str(exc)
    if not samples:
        return pd.DataFrame(), failures
    dataset = pd.concat(samples, ignore_index=True)
    dataset["timestamp_sort"] = pd.to_datetime(dataset["timestamp"], errors="coerce", utc=True)
    return dataset.sort_values(["timestamp_sort", "symbol"]).drop(columns=["timestamp_sort"]).reset_index(drop=True), failures


def _model_metrics(y_true: pd.Series, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
        "positive_rate": round(float(pd.Series(y_true).mean()), 4),
        "prediction_positive_rate": round(float(pd.Series(predictions).mean()), 4),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, probabilities)), 4)
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def _probability_bins(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    if predictions.empty:
        return []
    frame = predictions.copy()
    bins = np.linspace(0, 1, 11)
    frame["probability_bin"] = pd.cut(
        frame["target_hit_probability"],
        bins=bins,
        include_lowest=True,
        right=False,
    )
    rows: list[dict[str, Any]] = []
    for interval, group in frame.groupby("probability_bin", observed=True):
        if group.empty:
            continue
        rows.append(
            {
                "probability_bin": str(interval),
                "count": int(len(group)),
                "avg_predicted_probability": round(float(group["target_hit_probability"].mean()), 4),
                "actual_target_hit_rate": round(float(group["target_first"].mean()), 4),
                "avg_horizon_return_pct": round(float(group["horizon_return_pct"].mean()), 4),
                "avg_max_runup_pct": round(float(group["max_runup_pct"].mean()), 4),
                "avg_max_drawdown_pct": round(float(group["max_drawdown_pct"].mean()), 4),
            }
        )
    return rows


def train_target_hit_model(
    dataset: pd.DataFrame,
    config: HistoricalLearningConfig,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    if dataset.empty:
        raise ValueError("training dataset is empty")
    labels = dataset["target_first"].astype(int)
    if labels.nunique() < 2:
        raise ValueError("training dataset needs both positive and negative labels")

    split_at = int(len(dataset) * (1 - max(0.05, min(0.5, float(config.test_size)))))
    split_at = max(1, min(split_at, len(dataset) - 1))
    train = dataset.iloc[:split_at]
    test = dataset.iloc[split_at:]
    if train["target_first"].nunique() < 2:
        raise ValueError("training split needs both positive and negative labels")

    candidates: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=int(config.random_state),
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=25,
                        class_weight="balanced_subsample",
                        random_state=int(config.random_state),
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    candidate_metrics: dict[str, dict[str, Any]] = {}
    candidate_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, candidate in candidates.items():
        candidate.fit(train[FEATURE_COLUMNS], train["target_first"].astype(int))
        candidate_probabilities = candidate.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        candidate_predictions = (candidate_probabilities >= 0.5).astype(int)
        candidate_metrics[name] = _model_metrics(
            test["target_first"].astype(int),
            candidate_probabilities,
            candidate_predictions,
        )
        candidate_outputs[name] = (candidate_probabilities, candidate_predictions)

    def _rank_metric(item: tuple[str, dict[str, Any]]) -> tuple[float, float]:
        metric = item[1]
        auc = metric.get("roc_auc")
        auc_value = float(auc) if auc is not None else -1.0
        return auc_value, -float(metric.get("brier_score", 1.0))

    selected_model_name = max(candidate_metrics.items(), key=_rank_metric)[0]
    model = candidates[selected_model_name]
    probabilities, predictions = candidate_outputs[selected_model_name]
    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "all_rows": int(len(dataset)),
        "target_pct": float(config.target_pct),
        "stop_pct": float(config.stop_pct),
        "horizon_bars": int(config.horizon_bars),
        "selected_model": selected_model_name,
        "test": candidate_metrics[selected_model_name],
        "candidate_models": candidate_metrics,
    }
    predictions_frame = test[
        [
            "symbol",
            "timestamp",
            "outcome",
            "target_first",
            "bars_to_event",
            "horizon_return_pct",
            "max_runup_pct",
            "max_drawdown_pct",
        ]
    ].copy()
    predictions_frame["target_hit_probability"] = probabilities
    predictions_frame["prediction"] = predictions
    metrics["probability_bins"] = _probability_bins(predictions_frame)
    metrics["feature_count"] = int(len(FEATURE_COLUMNS))

    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        importance = sorted(
            [
                {
                    "feature": feature,
                    "coefficient": round(float(value), 6),
                    "abs_coefficient": round(abs(float(value)), 6),
                }
                for feature, value in zip(FEATURE_COLUMNS, values)
            ],
            key=lambda item: item["abs_coefficient"],
            reverse=True,
        )
    elif hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        importance = sorted(
            [
                {
                    "feature": feature,
                    "importance": round(float(value), 6),
                    "abs_coefficient": round(abs(float(value)), 6),
                }
                for feature, value in zip(FEATURE_COLUMNS, values)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )
    else:
        importance = []
    return model, metrics, predictions_frame, importance


def save_learning_artifacts(
    *,
    model: Pipeline,
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    result: HistoricalLearningResult,
    output_dir: str | Path,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "historical_target_hit_model.pkl"
    summary_path = output / "historical_learning_summary.json"
    dataset_path = output / "historical_chart_samples.csv"
    predictions_path = output / "historical_test_predictions.csv"

    with model_path.open("wb") as file:
        pickle.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "summary": result.config}, file)
    dataset.to_csv(dataset_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    files = {
        "model": str(model_path.resolve()),
        "summary": str(summary_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "test_predictions": str(predictions_path.resolve()),
    }
    result.files = files
    summary_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return files


def train_historical_chart_model(config: HistoricalLearningConfig) -> HistoricalLearningResult:
    price_data, fetch_failures, quality = fetch_yfinance_history(config)
    market_context, context_failures, context_quality = fetch_yfinance_context(config)
    dataset, dataset_failures = build_historical_dataset(price_data, config, market_context=market_context)
    skipped = {**fetch_failures, **dataset_failures}
    if dataset.empty:
        return HistoricalLearningResult(
            status="NO_DATA",
            config=asdict(config),
            rows=0,
            symbols=[],
            skipped_symbols=skipped,
            data_quality={
                "source": "Yahoo Finance via yfinance",
                "accepted_symbols": quality,
                "context_symbols": context_quality,
                "context_failures": context_failures,
            },
            positive_rate=0.0,
            metrics={},
            feature_importance=[],
            files={},
        )

    model, metrics, predictions, importance = train_target_hit_model(dataset, config)
    result = HistoricalLearningResult(
        status="TRAINED",
        config=asdict(config),
        rows=int(len(dataset)),
        symbols=sorted(dataset["symbol"].unique().tolist()),
        skipped_symbols=skipped,
        data_quality={
            "source": "Yahoo Finance via yfinance",
            "accepted_symbols": quality,
            "context_symbols": context_quality,
            "context_failures": context_failures,
        },
        positive_rate=round(float(dataset["target_first"].mean()), 4),
        metrics=metrics,
        feature_importance=importance,
        files={},
    )
    save_learning_artifacts(
        model=model,
        dataset=dataset,
        predictions=predictions,
        result=result,
        output_dir=config.output_dir,
    )
    return result
