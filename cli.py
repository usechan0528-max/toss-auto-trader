from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)
logging.disable(logging.WARNING)

from app import (
    APP_DIR,
    TOSS_DATA_SOURCE,
    add_indicators,
    append_signal_event,
    build_trade_plan,
    calculate_position,
    fallback_quote,
    load_toss_price_data,
    normalize_toss_auth,
    normalize_ticker,
    save_latest_signal,
    secret_bool,
    secret_value,
    signal_payload,
    train_predictor,
)
logging.disable(logging.NOTSET)
from stock_signal.alert_priority import evaluate_priority
from stock_signal.execution import AccountSnapshot, OrderRequest
from stock_signal.market_data import TossMarketDataAdapter
from stock_signal.prediction_metrics import calculate_formula_prediction, formula_prediction_to_dict
from stock_signal.toss_broker import TossBrokerExecutionAdapter


load_dotenv(APP_DIR / ".env", override=False)


def _json_default(value: Any) -> str:
    return str(value)


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default))


def _toss_base_url() -> str:
    return secret_value("TOSS_BASE_URL", "https://openapi.tossinvest.com")


def _market_adapter(interval: str, count: int) -> TossMarketDataAdapter:
    token, client_id, client_secret = normalize_toss_auth(
        secret_value("TOSS_API_TOKEN"),
        secret_value("TOSS_CLIENT_ID"),
        secret_value("TOSS_CLIENT_SECRET"),
    )
    return TossMarketDataAdapter(
        token=token,
        client_id=client_id,
        client_secret=client_secret,
        account_seq=secret_value("TOSS_ACCOUNT_SEQ") or None,
        base_url=_toss_base_url(),
        interval=interval,
        count=count,
    )


def _broker(*, allow_live_orders: bool = False) -> TossBrokerExecutionAdapter:
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
        base_url=_toss_base_url(),
        currency=os.getenv("TOSS_ORDER_CURRENCY", "USD"),
        allow_live_orders=allow_live_orders,
    )


def run_analyze(args: argparse.Namespace) -> int:
    symbol = normalize_ticker(args.symbol)
    if not symbol:
        raise SystemExit("symbol is required")

    adapter = _market_adapter(args.interval, args.count)
    candles = adapter.get_ohlcv(symbol)
    features = add_indicators(candles)
    result, _model_data = train_predictor(features)
    plan = build_trade_plan(features, result.probability_up, args.reward_ratio)

    try:
        quote = adapter.get_quote(symbol)
    except Exception:
        quote = fallback_quote(symbol, features)

    position = calculate_position(
        account_krw=args.account_krw,
        usdkrw=args.usdkrw,
        entry=plan.breakout_entry if plan.direction == "롱 우위" else plan.pullback_entry,
        stop=plan.stop_loss,
        risk_pct=args.risk_pct,
    )
    metrics = formula_prediction_to_dict(calculate_formula_prediction(features, quote))
    payload = signal_payload(
        symbol=symbol,
        result=result,
        plan=plan,
        position=position,
        alerts=[],
        market_events=[],
        quote=quote,
        data_source=TOSS_DATA_SOURCE,
        data_warnings=[],
        prediction_metrics=metrics,
    )
    priority = evaluate_priority(payload, None)
    payload["priority"] = {
        "level": priority.level,
        "score": priority.score,
        "reasons": priority.reasons,
    }
    payload["cli_generated_at"] = datetime.now().isoformat(timespec="seconds")

    if args.save:
        save_latest_signal(payload)
        append_signal_event(payload)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        _print_json(payload)
    else:
        print(f"{symbol} | {payload['action']} | ${payload['current_price']}")
        print(f"ML 상승확률: {payload['confidence'] * 100:.1f}%")
        print(f"계산식 확률: {metrics.get('probability_up', 0) * 100:.1f}%")
        print(f"진입구간: {payload['entry_zone'][0]} ~ {payload['entry_zone'][1]}")
        print(f"손절: {payload['stop_loss']} | 익절: {', '.join(map(str, payload['take_profit']))}")
        print(f"스프레드: {payload['spread_pct']}% | 우선순위: {priority.level} ({priority.score})")
        if payload.get("risk_flags"):
            print("리스크:", ", ".join(payload["risk_flags"]))
        if args.save:
            print(f"저장: {APP_DIR / 'latest_signal.json'}")
        if args.output:
            print(f"출력: {Path(args.output).resolve()}")
    return 0


def run_account(args: argparse.Namespace) -> int:
    broker = _broker(allow_live_orders=False)
    snapshot = broker.get_account()
    holdings = broker.get_holdings() if args.holdings else None
    data = {
        "account_id": snapshot.account_id[-4:].rjust(len(snapshot.account_id), "*"),
        "currency": snapshot.currency,
        "cash_buying_power": snapshot.cash_buying_power,
        "equity": snapshot.equity,
        "holdings": holdings,
    }
    if args.json:
        _print_json(data)
    else:
        print(f"계좌: {data['account_id']}")
        print(f"통화: {snapshot.currency}")
        print(f"매수가능금액: {snapshot.cash_buying_power:.2f}")
        if holdings is not None:
            print(json.dumps(holdings, ensure_ascii=False, indent=2, default=_json_default))
    return 0


def run_order_payload(args: argparse.Namespace) -> int:
    symbol = normalize_ticker(args.symbol)
    order = OrderRequest(
        symbol=symbol,
        side=args.side,
        order_type=args.order_type,
        quantity=args.quantity,
        limit_price=args.price,
        time_in_force=args.time_in_force,
        client_order_id=args.client_order_id or f"cli-{symbol.lower()}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
    )
    broker = _broker(allow_live_orders=secret_bool("TOSS_LIVE_ORDER_ENABLED", False))
    payload = broker.order_payload(order, confirm_high_value_order=args.confirm_high_value_order)
    if args.submit:
        if not secret_bool("TOSS_LIVE_ORDER_ENABLED", False):
            raise SystemExit("TOSS_LIVE_ORDER_ENABLED=true일 때만 CLI 실주문 전송이 가능합니다.")
        result = broker.submit_order(order, confirm_high_value_order=args.confirm_high_value_order)
        _print_json(
            {
                "status": result.status,
                "broker_order_id": result.broker_order_id,
                "client_order_id": result.client_order_id,
                "message": result.message,
                "raw": result.raw,
            }
        )
    else:
        _print_json({"submit": False, "payload": payload})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="단타 분석기를 웹 없이 터미널에서 실행합니다.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Toss API 캔들/호가로 종목 분석")
    analyze.add_argument("symbol", help="예: AAPL, NVDA")
    analyze.add_argument("--interval", default="1m", choices=["1m", "1d"], help="Toss 캔들 간격")
    analyze.add_argument("--count", type=int, default=200, help="가져올 캔들 수")
    analyze.add_argument("--account-krw", type=float, default=10_000_000)
    analyze.add_argument("--usdkrw", type=float, default=1350)
    analyze.add_argument("--risk-pct", type=float, default=1.0)
    analyze.add_argument("--reward-ratio", type=float, default=1.8)
    analyze.add_argument("--save", action="store_true", help="latest_signal.json과 signal_events.jsonl에 저장")
    analyze.add_argument("--output", help="분석 JSON 저장 경로")
    analyze.add_argument("--json", action="store_true", help="전체 JSON 출력")
    analyze.set_defaults(func=run_analyze)

    account = subparsers.add_parser("account", help="Toss 계좌/매수가능금액 조회")
    account.add_argument("--holdings", action="store_true", help="보유종목도 함께 조회")
    account.add_argument("--json", action="store_true")
    account.set_defaults(func=run_account)

    order = subparsers.add_parser("order", help="주문 payload 생성, --submit 사용 시 전송")
    order.add_argument("symbol")
    order.add_argument("side", choices=["BUY", "SELL"])
    order.add_argument("quantity", type=int)
    order.add_argument("--order-type", default="LIMIT", choices=["LIMIT", "MARKET"])
    order.add_argument("--price", type=float)
    order.add_argument("--time-in-force", default="DAY")
    order.add_argument("--client-order-id")
    order.add_argument("--confirm-high-value-order", action="store_true")
    order.add_argument("--submit", action="store_true", help="실제 주문 API 호출")
    order.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    order.set_defaults(func=run_order_payload)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "HTTP"
        if status == 401:
            print("Toss API 인증 실패: .env의 API key/secret, 계좌 권한, 운영/테스트 base URL을 확인하세요.", file=sys.stderr)
        else:
            print(f"Toss API 요청 실패: {status} {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"실행 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
