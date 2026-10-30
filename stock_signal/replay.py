from __future__ import annotations

import time
from collections.abc import Callable

import pandas as pd


def replay_market_data(csv_path: str, strategy: Callable[[dict], object], delay_seconds: float = 0.1) -> None:
    data = pd.read_csv(csv_path)
    for _, row in data.iterrows():
        print(strategy(row.to_dict()))
        time.sleep(delay_seconds)
