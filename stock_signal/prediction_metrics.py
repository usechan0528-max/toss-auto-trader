from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, sqrt
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FormulaPrediction:
    score: float
    probability_up: float
    expected_move_pct: float
    projected_price: float
    projected_low: float
    projected_high: float
    atr_pct: float
    horizon_bars: int
    components: dict[str, float]
    diagnostics: dict[str, float]
    reasons: list[str]


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(low, min(high, value)))


def _sigmoid(value: float) -> float:
    return 1 / (1 + exp(-value))


def _latest(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def add_prediction_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["return_1"] = df["Close"].pct_change()
    df["return_5"] = df["Close"].pct_change(5)
    df["ema_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["volume_ma_20"] = df["Volume"].rolling(20, min_periods=5).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma_20"].replace(0, np.nan)
    volume_std_20 = df["Volume"].rolling(20, min_periods=5).std().replace(0, np.nan)
    df["volume_zscore_20"] = (df["Volume"] - df["volume_ma_20"]) / volume_std_20

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    df["tr"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = df["tr"].rolling(14, min_periods=5).mean()
    df["atr_pct"] = df["atr_14"] / df["Close"].replace(0, np.nan) * 100

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df.loc[(loss == 0) & (gain > 0), "rsi_14"] = 100
    df.loc[(loss == 0) & (gain == 0), "rsi_14"] = 50

    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    rolling_volume = df["Volume"].rolling(60, min_periods=10).sum().replace(0, np.nan)
    df["vwap_60"] = (typical * df["Volume"]).rolling(60, min_periods=10).sum() / rolling_volume
    df["vwap_distance_pct"] = (df["Close"] / df["vwap_60"].replace(0, np.nan) - 1) * 100
    df["ema_spread_pct"] = (df["ema_9"] / df["ema_21"].replace(0, np.nan) - 1) * 100
    df["ema_9_slope_5"] = df["ema_9"].pct_change(5) * 100
    df["ma_20_slope_5"] = df["Close"].rolling(20, min_periods=5).mean().pct_change(5) * 100
    boll_mid = df["Close"].rolling(20, min_periods=5).mean()
    boll_std = df["Close"].rolling(20, min_periods=5).std()
    df["boll_upper"] = boll_mid + boll_std * 2
    df["boll_lower"] = boll_mid - boll_std * 2
    boll_range = (df["boll_upper"] - df["boll_lower"]).replace(0, np.nan)
    df["boll_width_pct"] = boll_range / df["Close"].replace(0, np.nan) * 100
    df["boll_position"] = (df["Close"] - df["boll_lower"]) / boll_range
    low_14 = df["Low"].rolling(14, min_periods=5).min()
    high_14 = df["High"].rolling(14, min_periods=5).max()
    stoch_range = (high_14 - low_14).replace(0, np.nan)
    df["stoch_k"] = (df["Close"] - low_14) / stoch_range * 100
    df["stoch_d"] = df["stoch_k"].rolling(3, min_periods=2).mean()
    signed_volume = np.sign(df["Close"].diff().fillna(0)) * df["Volume"]
    df["obv"] = signed_volume.cumsum()
    df["obv_slope_5"] = (df["obv"] - df["obv"].shift(5)) / (df["volume_ma_20"] * 5).replace(0, np.nan)
    candle_range = (df["High"] - df["Low"]).replace(0, np.nan)
    body = df["Close"] - df["Open"]
    df["close_location"] = (df["Close"] - df["Low"]) / candle_range
    df["body_pct"] = body.abs() / df["Close"].replace(0, np.nan) * 100
    df["candle_body_direction"] = np.sign(body)
    df["resistance_20_prev"] = df["High"].rolling(20, min_periods=5).max().shift(1)
    df["support_20_prev"] = df["Low"].rolling(20, min_periods=5).min().shift(1)
    df["volatility_20"] = df["return_1"].rolling(20, min_periods=5).std()
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def _spread_pct(quote: Any, price: float) -> float:
    if quote is None:
        return 0.0
    if hasattr(quote, "spread_pct"):
        return float(getattr(quote, "spread_pct"))
    bid = getattr(quote, "bid", None)
    ask = getattr(quote, "ask", None)
    if bid is None or ask is None or price <= 0:
        return 0.0
    return max(0.0, (float(ask) - float(bid)) / price * 100)


def _current_session_slice(data: pd.DataFrame) -> pd.DataFrame:
    session = data[[column for column in ["High", "Low"] if column in data.columns]].dropna()
    if session.empty:
        return session
    if isinstance(session.index, pd.DatetimeIndex):
        index = session.index
        try:
            session_dates = index.tz_convert("Asia/Seoul").date if index.tz is not None else index.date
            latest_date = session_dates[-1]
            current_session = session.loc[session_dates == latest_date]
            if not current_session.empty:
                return current_session
        except (TypeError, ValueError):
            pass
    return session.tail(min(len(session), 390))


def _price_gap_pct(reference: float, price: float) -> float:
    if reference <= 0 or price <= 0:
        return 0.0
    return (float(price) / float(reference) - 1) * 100


def calculate_formula_prediction(data: pd.DataFrame, quote: Any = None, horizon_bars: int = 3) -> FormulaPrediction:
    features = add_prediction_features(data)
    if features.empty:
        raise ValueError("not enough data to calculate formula prediction")

    latest = features.iloc[-1]
    close = _latest(latest, "Close")
    quote_price = float(getattr(quote, "price", close) or close) if quote is not None else close
    price = quote_price if quote_price > 0 else close
    atr = max(_latest(latest, "atr_14", price * 0.003), price * 0.003)
    atr_pct = atr / price * 100 if price > 0 else 0.0
    volume_ratio = max(_latest(latest, "volume_ratio", 1.0), 0.0)
    volume_zscore = _latest(latest, "volume_zscore_20", 0.0)
    spread_pct = _spread_pct(quote, price)

    ema_spread_score = _clip(_latest(latest, "ema_spread_pct") / max(atr_pct, 0.1))
    slope_score = _clip((_latest(latest, "ema_9_slope_5") * 0.65 + _latest(latest, "ma_20_slope_5") * 0.35) / max(atr_pct, 0.1))
    trend_score = _clip(ema_spread_score * 0.58 + slope_score * 0.42)
    momentum_score = _clip(np.tanh((_latest(latest, "return_1") * 0.35 + _latest(latest, "return_5") * 0.65) * 45))
    vwap_score = _clip((price - _latest(latest, "vwap_60", price)) / atr)
    volume_score = _clip(np.tanh((volume_ratio - 1.0) * 0.55 + volume_zscore * 0.16))
    breakout_score = _clip(((price - _latest(latest, "resistance_20_prev", price)) / atr) * 0.72 + (_latest(latest, "boll_position", 0.5) - 0.5) * 0.56)
    obv_score = _clip(np.tanh(_latest(latest, "obv_slope_5", 0.0) * 1.25))
    close_location = _latest(latest, "close_location", 0.5)
    candle_direction = _latest(latest, "candle_body_direction", 0.0)
    candle_score = _clip((close_location - 0.5) * 1.45 + candle_direction * min(_latest(latest, "body_pct", 0.0) / max(atr_pct, 0.1), 1.0) * 0.22)

    rsi = _latest(latest, "rsi_14", 50.0)
    if 45 <= rsi <= 68:
        rsi_score = _clip((rsi - 50) / 30)
    elif rsi > 68:
        rsi_score = -_clip((rsi - 68) / 32, 0.0, 0.75)
    else:
        rsi_score = -_clip((45 - rsi) / 45, 0.0, 0.65)

    stoch_k = _latest(latest, "stoch_k", 50.0)
    stoch_d = _latest(latest, "stoch_d", 50.0)
    if 35 <= stoch_k <= 82:
        stochastic_score = _clip((stoch_k - 50) / 35 + (stoch_k - stoch_d) / 80)
    elif stoch_k > 82:
        stochastic_score = -_clip((stoch_k - 82) / 28, 0.0, 0.7)
    else:
        stochastic_score = -_clip((35 - stoch_k) / 35, 0.0, 0.65)

    volatility_pct = _latest(latest, "volatility_20") * 100
    boll_width_pct = _latest(latest, "boll_width_pct", atr_pct * 4)
    volatility_penalty = -_clip(max(volatility_pct - max(atr_pct * 0.65, 1.5), 0.0) / 6.0 + max(boll_width_pct - atr_pct * 8, 0.0) / 35.0, 0.0, 0.55)
    spread_penalty = -_clip(max(spread_pct - 0.5, 0.0) / 2.5, 0.0, 0.65)

    components = {
        "trend": round(trend_score, 4),
        "momentum": round(momentum_score, 4),
        "vwap": round(vwap_score, 4),
        "volume": round(volume_score, 4),
        "breakout": round(breakout_score, 4),
        "rsi": round(rsi_score, 4),
        "stochastic": round(stochastic_score, 4),
        "obv": round(obv_score, 4),
        "candle": round(candle_score, 4),
        "volatility_penalty": round(volatility_penalty, 4),
        "spread_penalty": round(spread_penalty, 4),
    }
    score = (
        trend_score * 0.19
        + momentum_score * 0.14
        + vwap_score * 0.15
        + volume_score * 0.11
        + breakout_score * 0.13
        + rsi_score * 0.05
        + stochastic_score * 0.05
        + obv_score * 0.08
        + candle_score * 0.05
        + volatility_penalty * 0.03
        + spread_penalty * 0.02
    )
    score = _clip(score)
    probability_up = max(0.05, min(0.95, _sigmoid(score * 2.35)))

    horizon = max(1, int(horizon_bars))
    directional_range_pct = atr_pct * sqrt(horizon) * (0.45 + min(volume_ratio, 4.0) * 0.08)
    expected_move_pct = score * directional_range_pct
    projected_price = price * (1 + expected_move_pct / 100)
    uncertainty_pct = max(atr_pct * sqrt(horizon) * 0.55, spread_pct * 0.5, 0.1)
    projected_low = projected_price * (1 - uncertainty_pct / 100)
    projected_high = projected_price * (1 + uncertainty_pct / 100)
    session = _current_session_slice(data)
    today_observed_high = float(session["High"].max()) if not session.empty and "High" in session else price
    today_observed_low = float(session["Low"].min()) if not session.empty and "Low" in session else price
    today_projected_high = max(today_observed_high, projected_high)
    today_projected_low = min(today_observed_low, projected_low)

    reasons: list[str] = []
    if trend_score > 0.25:
        reasons.append("short EMA is above long EMA")
    elif trend_score < -0.25:
        reasons.append("short EMA is below long EMA")
    if vwap_score > 0.25:
        reasons.append("price is above VWAP")
    elif vwap_score < -0.25:
        reasons.append("price is below VWAP")
    if breakout_score > 0.2:
        reasons.append("price is pressing recent resistance")
    if volume_score > 0.25:
        reasons.append("volume is expanding")
    elif volume_score < -0.25:
        reasons.append("volume confirmation is weak")
    if obv_score > 0.2:
        reasons.append("OBV confirms accumulation")
    elif obv_score < -0.2:
        reasons.append("OBV shows distribution pressure")
    if candle_score > 0.2:
        reasons.append("candle closes near the high")
    elif candle_score < -0.2:
        reasons.append("candle closes weak")
    if stochastic_score < -0.2:
        reasons.append("stochastic momentum is unfavorable")
    if volatility_penalty < -0.15:
        reasons.append("volatility expansion adds uncertainty")
    if spread_penalty < -0.1:
        reasons.append("spread is reducing signal quality")

    return FormulaPrediction(
        score=round(score, 4),
        probability_up=round(probability_up, 4),
        expected_move_pct=round(expected_move_pct, 4),
        projected_price=round(projected_price, 4),
        projected_low=round(projected_low, 4),
        projected_high=round(projected_high, 4),
        atr_pct=round(atr_pct, 4),
        horizon_bars=horizon,
        components=components,
        diagnostics={
            "volume_ratio": round(volume_ratio, 4),
            "volume_zscore_20": round(volume_zscore, 4),
            "vwap_distance_pct": round(_latest(latest, "vwap_distance_pct", 0.0), 4),
            "ema_spread_pct": round(_latest(latest, "ema_spread_pct", 0.0), 4),
            "boll_position": round(_latest(latest, "boll_position", 0.5), 4),
            "boll_width_pct": round(boll_width_pct, 4),
            "stoch_k": round(stoch_k, 4),
            "stoch_d": round(stoch_d, 4),
            "obv_slope_5": round(_latest(latest, "obv_slope_5", 0.0), 4),
            "close_location": round(close_location, 4),
            "directional_range_pct": round(directional_range_pct, 4),
            "projection_uncertainty_pct": round(uncertainty_pct, 4),
            "today_observed_high": round(today_observed_high, 4),
            "today_observed_low": round(today_observed_low, 4),
            "today_projected_high": round(today_projected_high, 4),
            "today_projected_low": round(today_projected_low, 4),
            "today_projected_high_gap_pct": round(_price_gap_pct(price, today_projected_high), 4),
            "today_projected_low_gap_pct": round(_price_gap_pct(price, today_projected_low), 4),
            "session_rows": round(float(len(session)), 0),
        },
        reasons=reasons,
    )


def formula_prediction_to_dict(prediction: FormulaPrediction) -> dict[str, Any]:
    return asdict(prediction)
