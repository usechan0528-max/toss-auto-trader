from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stock_signal.toss_data_collector import (
    CollectionConfig,
    TossDataCollector,
    create_client_from_env,
    load_dotenv_file,
    load_symbols_from_local_state,
    parse_symbol_text,
    unique_symbols,
)


APP_DIR = Path(__file__).resolve().parent


def parse_csv_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the maximum read-only dataset available from Toss Invest OpenAPI.")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols to include first.")
    parser.add_argument("--output-dir", default="", help="Export directory. Defaults to toss_api_exports/<timestamp>.")
    parser.add_argument("--max-symbols", type=int, default=80, help="Maximum symbols for per-symbol endpoints. Use 0 for no cap.")
    parser.add_argument("--candle-intervals", default="1m,1d", help="Comma-separated candle intervals supported by Toss API.")
    parser.add_argument("--candle-count", type=int, default=200, help="Candles per page, max 200.")
    parser.add_argument("--max-candle-pages", type=int, default=2, help="Candle pages per symbol/interval.")
    parser.add_argument("--trades-count", type=int, default=50, help="Recent trades per symbol, max 50.")
    parser.add_argument("--order-statuses", default="OPEN,CLOSED", help="Order statuses to collect.")
    parser.add_argument("--max-order-pages", type=int, default=10, help="Closed-order pages to collect.")
    parser.add_argument("--max-order-details", type=int, default=200, help="Maximum order detail calls.")
    parser.add_argument("--order-from", default="", help="Order query start date, YYYY-MM-DD.")
    parser.add_argument("--order-to", default="", help="Order query end date, YYYY-MM-DD.")
    parser.add_argument("--calendar-days", type=int, default=7, help="Market calendar days from today.")
    parser.add_argument("--request-pause", type=float, default=0.2, help="Pause between API calls to reduce 429 errors.")
    parser.add_argument("--timeout-seconds", type=int, default=15)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--no-default-universe", action="store_true", help="Do not append built-in liquid US universe.")
    parser.add_argument("--no-official-docs", action="store_true", help="Skip saving official OpenAPI docs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv_file(APP_DIR / ".env")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else APP_DIR / "toss_api_exports" / timestamp
    cli_symbols = parse_symbol_text(args.symbols)
    local_symbols = load_symbols_from_local_state(APP_DIR, include_default_universe=not args.no_default_universe)
    symbols = unique_symbols(cli_symbols + local_symbols)
    client = create_client_from_env(
        request_pause_seconds=args.request_pause,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    config = CollectionConfig(
        app_dir=APP_DIR,
        output_dir=output_dir,
        symbols=symbols,
        include_default_universe=not args.no_default_universe,
        max_symbols=args.max_symbols,
        candle_intervals=parse_csv_values(args.candle_intervals.lower()),
        candle_count=args.candle_count,
        max_candle_pages=args.max_candle_pages,
        trades_count=args.trades_count,
        order_statuses=[status.strip().upper() for status in args.order_statuses.split(",") if status.strip()],
        max_order_pages=args.max_order_pages,
        max_order_details=args.max_order_details,
        order_from=args.order_from or None,
        order_to=args.order_to or None,
        calendar_days=args.calendar_days,
        request_pause_seconds=args.request_pause,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        save_official_docs=not args.no_official_docs,
    )
    summary = TossDataCollector(client, config).collect()
    print(f"export_dir={output_dir}")
    print(f"symbols={summary.get('counts', {}).get('symbols', 0)}")
    print(f"candle_rows={summary.get('counts', {}).get('candle_rows', 0)}")
    print(f"trade_rows={summary.get('counts', {}).get('trade_rows', 0)}")
    print(f"order_rows={summary.get('counts', {}).get('order_rows', 0)}")
    print(f"errors={summary.get('errors', 0)}")


if __name__ == "__main__":
    main()
