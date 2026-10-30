from __future__ import annotations

import sys

import pandas as pd

from candle_validation import validate_candle, validate_price_data


def run_builtin_checks() -> None:
    validate_candle(
        {
            "open": 6.10,
            "high": 6.35,
            "low": 6.05,
            "close": 6.22,
            "volume": 120_000,
        }
    )

    invalid = {
        "open": 6.10,
        "high": 6.00,
        "low": 6.05,
        "close": 6.22,
        "volume": 120_000,
    }
    try:
        validate_candle(invalid)
    except AssertionError:
        return
    raise AssertionError("invalid candle was not rejected")


def main() -> None:
    run_builtin_checks()

    if len(sys.argv) > 1:
        path = sys.argv[1]
        data = pd.read_csv(path)
        data.columns = [str(col).strip().title() for col in data.columns]
        validate_price_data(data)
        print(f"OK: {len(data):,} candles validated from {path}")
    else:
        print("OK: built-in candle validation checks passed")


if __name__ == "__main__":
    main()
