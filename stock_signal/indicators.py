from __future__ import annotations

import numpy as np
import pandas as pd


def add_intraday_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cumulative_volume = df["Volume"].replace(0, np.nan).cumsum()
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / cumulative_volume
    df["VolumeMA30"] = df["Volume"].rolling(30, min_periods=5).mean()
    df["VolumeRatio"] = df["Volume"] / df["VolumeMA30"].replace(0, np.nan)
    df["MA5"] = df["Close"].rolling(5, min_periods=5).mean()
    df["MA20"] = df["Close"].rolling(20, min_periods=10).mean()
    df["RecentHigh"] = df["High"].rolling(10, min_periods=3).max().shift(1)
    df["RecentLow"] = df["Low"].rolling(10, min_periods=3).min().shift(1)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))
    df.loc[(loss == 0) & (gain > 0), "RSI14"] = 100
    df.loc[(loss == 0) & (gain == 0), "RSI14"] = 50

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    df["TR"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = df["TR"].rolling(14, min_periods=5).mean()
    return df.dropna()
