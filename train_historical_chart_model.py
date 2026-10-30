from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from stock_signal.historical_learning import (
    DEFAULT_CONTEXT_SYMBOLS,
    DEFAULT_MAJOR_SYMBOLS,
    HistoricalLearningConfig,
    train_historical_chart_model,
)


def _symbols(value: str) -> list[str]:
    if not value.strip():
        return list(DEFAULT_MAJOR_SYMBOLS)
    return [symbol.strip().upper() for symbol in value.split(",") if symbol.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a target-before-stop model from historical major-stock charts.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_MAJOR_SYMBOLS))
    parser.add_argument("--context-symbols", default=",".join(DEFAULT_CONTEXT_SYMBOLS))
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--no-prepost", action="store_true")
    parser.add_argument("--target-pct", type=float, default=1.2)
    parser.add_argument("--stop-pct", type=float, default=0.6)
    parser.add_argument("--horizon-bars", type=int, default=6)
    parser.add_argument("--min-rows-per-symbol", type=int, default=120)
    parser.add_argument("--min-labeled-rows-per-symbol", type=int, default=120)
    parser.add_argument("--max-abs-return-pct", type=float, default=60.0)
    parser.add_argument("--max-stale-days", type=int, default=14)
    parser.add_argument("--allow-loose-data-quality", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--output-dir", default="model_artifacts/historical_chart")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = HistoricalLearningConfig(
        symbols=_symbols(args.symbols),
        context_symbols=_symbols(args.context_symbols),
        period=args.period,
        interval=args.interval,
        prepost=not args.no_prepost,
        target_pct=args.target_pct,
        stop_pct=args.stop_pct,
        horizon_bars=args.horizon_bars,
        min_rows_per_symbol=args.min_rows_per_symbol,
        min_labeled_rows_per_symbol=args.min_labeled_rows_per_symbol,
        strict_data_quality=not args.allow_loose_data_quality,
        max_abs_return_pct=args.max_abs_return_pct,
        max_stale_days=args.max_stale_days,
        test_size=args.test_size,
        output_dir=args.output_dir,
    )
    result = train_historical_chart_model(config)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.status == "TRAINED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
