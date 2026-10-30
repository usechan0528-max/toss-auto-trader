> **문서 상태 안내 (2026-10-30)**: 아래는 이전 상세 사용 설명을 보존한 문서입니다. 현재 `app.py`는 상태 모니터 중심이며, 여기에 설명된 모든 UI가 현재 소스에 제공되지는 않습니다. 현재 구현 범위와 알려진 오류는 루트 [README](../README.md)를 기준으로 확인하세요.

# 해외주식 단타 분석 프로그램

Streamlit 기반의 단타 참고용 대시보드입니다. 앱 화면의 시세 데이터는 토스증권 OpenAPI에서만 받아 분석합니다.

## 실행

```powershell
streamlit run app.py
```

기본 주소:

```text
http://localhost:8501
```

## 주요 기능

- 관심종목 스캐너
- 실시간 기준 상한선/하한선 계산
- 단기 지지/저항 계산
- 돌파 진입 후보가, 눌림 진입 후보가 계산
- 손절 기준, 1차/2차 익절 기준 계산
- 거래량 배율, RSI, EMA, VWAP, ATR 기반 보조 판단
- 사용자의 기존 진입가 입력 시 현재 손익률 표시
- 앱 내 종목 검색: 심볼/한글명/영문명 입력 후 토스 종목 기본 정보로 검증
- 실계좌 조회: 계좌 목록, 매수 가능 금액, 보유 주식, 수수료 조회
- 수동 실주문: 정수 수량 기반 지정가/시장가 주문, 미체결 조회, 주문 취소
- PC 워크스페이스 / 사이드패널 / 알림 모드 지원
- 장투 탭: 월 적립금, 투자 기간, 기대수익률 가정 기반 적립식 투자 계획 보조

## 실행 모드

앱 왼쪽 사이드바의 `실행 모드`에서 선택합니다.

- `워크스페이스`: PC 전체 화면용 대시보드입니다. 후보 랭킹, 차트, 진입 참고, 최근 본 종목을 한 화면에 배치합니다.
- `사이드패널`: 토스증권 화면 옆에 좁게 띄우는 보조 패널입니다. 선택 종목의 신호, 진입 구간, 손절, 익절, 최근 본 종목만 표시합니다.
- `알림 모드`: 최신 신호 JSON과 경고 이력을 확인하는 콘솔입니다.

워크스페이스에서는 사이드바의 `투자 탭`에서 `단타`와 `장투`를 전환할 수 있습니다. `장투`는 자동 주문 없이 적립식 투자 계획, 장기 추세, 분할 매수 구간, 장기 시나리오를 보여줍니다.

신호 파일은 아래 위치에 저장됩니다.

```text
latest_signal.json
signal_events.jsonl
app_settings.json
latest_trade_intent.json
trade_reviews.jsonl
llm_trade_review.md
latest_auto_trade.json
auto_trade_events.jsonl
latest_long_term_plan.json
live_order_events.jsonl
```

외부 알림 앱이나 별도 플러그인 프로세스는 이 파일을 읽어서 데스크톱 알림, 모바일 알림, 텔레그램 알림 등으로 확장할 수 있습니다.

별도 콘솔에서 최신 신호를 감시하려면 아래 명령을 실행합니다.

```powershell
python signal_watcher.py --path .\latest_signal.json --intent-path .\latest_trade_intent.json --event-log .\watch_events.jsonl --interval 2 --min-priority LOW
```

한 번만 확인하려면 아래처럼 실행합니다.

```powershell
python signal_watcher.py --path .\latest_signal.json --intent-path .\latest_trade_intent.json --once
```

현재 watcher는 콘솔 출력과 JSONL 이벤트 기록만 수행합니다. 주문 실행이나 토스증권 화면 조작은 하지 않습니다.

우선순위는 `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`로 계산됩니다. 콘솔 출력은 `--min-priority HIGH`처럼 필터링할 수 있고, JSONL 로그에는 모든 변경 이벤트의 우선순위가 저장됩니다.

## 장중 자동 세션 워커

장중에 자동 선정 기준과 자동매매 게이트를 반복 실행하려면 별도 콘솔에서 아래 명령을 실행합니다.

```powershell
python auto_session_trader.py --interval-seconds 60 --clear-stop
```

한 번만 검증하려면 아래처럼 실행합니다.

```powershell
python auto_session_trader.py --once
```

중지하려면 `auto_trader.stop` 파일을 만들면 다음 사이클에서 종료됩니다.

```powershell
New-Item -ItemType File .\auto_trader.stop -Force
```

워커는 `app_settings.json`과 `.env`의 현재 설정을 매 사이클 다시 읽습니다. 자동 선정 적격 종목이 없으면 `latest_trade_intent.json`은 차단 상태로 저장되고, 실주문 게이트도 주문을 보내지 않습니다. 실행 이력은 `auto_session_events.jsonl`, `latest_auto_trade.json`, `auto_trade_events.jsonl`에 기록됩니다.

안정성 장치:

- `auto_session_trader.lock`으로 워커 중복 실행을 차단합니다.
- 기본값으로 연속 오류 5회가 발생하면 워커가 종료됩니다. 필요하면 `--max-consecutive-errors 3`처럼 조정할 수 있습니다.
- `latest_*.json` 상태 파일은 임시 파일에 쓴 뒤 원자적으로 교체해, 감시 프로세스가 깨진 JSON을 읽을 가능성을 줄입니다.
- 자동 `SELL` 의도는 보유 수량이 확인되지 않으면 차단되고, 확인된 보유 수량을 초과하지 않게 캡핑됩니다.

## 매매 준비 모드

사이드바의 `매매 준비 모드`는 주문 실행이 아니라 주문 의도 파일을 만드는 준비 기능입니다.

- `disabled`: 기본값입니다. 매매 의도를 만들되 항상 `BLOCKED` 상태로 저장합니다.
- `dry_run`: 실주문 없이 의도 검증만 통과시키는 테스트 모드입니다.
- `live_ready`: 실계좌/주문 게이트와 함께 사용할 수 있는 준비 모드입니다. 실제 전송은 별도 확인을 통과해야 합니다.

생성 파일:

```text
latest_trade_intent.json
latest_execution_plan.json
latest_auto_trade.json
auto_trade_events.jsonl
live_order_events.jsonl
```

주의: 토스증권 실주문 API 호출 경로가 구현되어 있습니다. 기본값은 차단이며, `Toss 실주문 전송 허용`, 실계좌 조회, 주문 상한, 체크박스, 확인문 입력을 통과해야 수동 주문이 전송됩니다.

현재 추가된 매매 안전 하네스:

- 일일 손실 한도: 계좌 달러 환산 금액 기준 % 한도와 고정 USD 한도 중 더 엄격한 값을 사용합니다.
- 일일 수익 락: 목표 상단을 넘기면 그날 최고 실현수익률을 추적하고, 설정한 되돌림 폭만큼 하락하면 당일 거래를 중지합니다. 기본값은 `20%` 도달 후 최고점 대비 `10%p` 되돌림 차단입니다.
- 남은 손실 한도 검증: 새 진입의 예상 위험이 남은 손실 한도보다 크면 매매 준비를 차단합니다.
- 중복 매매 의도 차단: `종목 + 방향 + 가격대 + 거래일` 기준으로 같은 의도가 반복 준비되지 않게 막습니다.
- 데이터 품질 차단: 사용자가 켜면 최신 봉 지연 시간과 스프레드 기준을 넘는 신호를 `NO_TRADE`로 강제합니다.
  - 데이터 신선도는 `FRESH`, `DELAYED`, `STALE`, `UNKNOWN`으로 표시됩니다. 차단 옵션이 꺼져 있어도 오래된 데이터는 UI와 watcher에서 경고 등급으로 보입니다.
- 수동 승인 체크리스트: 손절가, 수량/예상손실, 자동주문 아님을 모두 확인해야 dry-run 매매 의도가 준비됩니다.
- 세션 진입 제한: 같은 거래일의 준비 완료 매매 의도 횟수를 세고, 설정한 최대 횟수에 닿으면 새 의도를 차단합니다.
- 토스 조회/주문 어댑터: OHLCV, 호가/스프레드, 계좌, 보유, 미체결, 주문 생성/취소 인터페이스를 준비했습니다.
- 실행 준비 레이어: `latest_execution_plan.json`에 브로커 주문 직전의 `OrderRequest`, 계좌 연결 여부, 실행 차단 사유를 저장합니다. 기본값은 항상 실행 차단입니다.
- 매매 의도 이력: `order_intent_events.jsonl`, `trade_intents.jsonl`에 준비 이력을 저장합니다.
- 자동매매 게이트: `latest_auto_trade.json`, `auto_trade_events.jsonl`에 자동 처리 상태를 저장합니다. 기본값은 `disabled`이며 킬스위치가 켜져 있습니다.
- 페이퍼 자동매매: `paper` 모드는 실제 주문 없이 `PaperBrokerExecutionAdapter`로 모의 제출 결과만 기록합니다.
- 라이브 자동매매: `live` 모드는 실계좌 조회, 사전 승인, 킬스위치 해제, 주문 상한, 중복 제출 차단을 통과해야 전송됩니다.

## 하네스 구조

핵심 계산은 `stock_signal/` 아래 모듈로 분리했습니다.

```text
stock_signal/
  market_data.py   # 토스 시세/종목 정보, CSV/mock 테스트 어댑터
  validators.py    # OHLCV/호가 검증
  filters.py       # 종목 필터
  indicators.py    # VWAP, 거래량 배율, RSI, ATR
  strategy.py      # BUY_WATCH / WAIT / NO_TRADE 판단
  risk.py          # 손실 한도 기반 수량 계산
  execution.py     # 계좌/주문 API 연결 전 실행 계획과 브로커 어댑터 계약
  autotrade.py     # 자동매매 게이트, 킬스위치, 페이퍼 브로커, 중복 제출 차단
  toss_broker.py   # 토스 계좌/주문 OpenAPI 어댑터, 기본 실주문 차단
  long_term.py     # 장투/적립식 투자 계획 계산
  backtest.py      # 신호 사후 평가
  output.py        # JSON/텍스트 출력
  trade_review.py  # 예측과 실제 청산 결과 비교, LLM 복기 문서 생성
  replay.py        # CSV 모의 재생
```

실주문 호출은 `stock_signal/toss_broker.py`의 `TossBrokerExecutionAdapter`에 분리했습니다. 기본값은 `allow_live_orders=False`라서 어댑터가 연결되어도 주문 생성 요청을 보내지 않습니다. 수동 실주문은 화면의 `실계좌 · 수동 주문` 패널에서 `Toss 실주문 전송 허용`, 실계좌 조회, 주문 상한, 확인 체크, 확인문 입력을 모두 통과해야만 전송됩니다. 자동매매 실주문은 별도로 `live` 모드, 사전 승인, 킬스위치 해제, Toss 계좌 조회 성공, 주문 상한을 통과해야만 시도됩니다.

## 캔들 검증 하네스

OHLCV 데이터가 정상인지 확인하는 하네스를 추가했습니다.

```powershell
python candle_harness.py
python candle_harness.py .\sample.csv
```

검증 조건:

- `High >= Low`
- `High >= Open`
- `High >= Close`
- `Low <= Open`
- `Low <= Close`
- `Volume >= 0`

## 테스트

```powershell
python -m unittest discover -s tests
```

## API 연결 위치

앱 화면의 데이터 입력은 `Toss API`로 고정되어 있습니다. 내부적으로는 `stock_signal/market_data.py`의 토스 어댑터를 사용합니다.

- `TossMarketDataAdapter`: 토스증권 OpenAPI 시세/참조 정보 어댑터입니다. `/oauth2/token`, `/api/v1/prices`, `/api/v1/orderbook`, `/api/v1/trades`, `/api/v1/price-limits`, `/api/v1/candles`, `/api/v1/stocks`, `/api/v1/stocks/{symbol}/warnings`, `/api/v1/exchange-rate`, `/api/v1/market-calendar/KR`, `/api/v1/market-calendar/US`를 사용합니다.
- `TossBrokerExecutionAdapter`: 토스증권 OpenAPI 계좌/주문 어댑터입니다. `/api/v1/accounts`, `/api/v1/buying-power`, `/api/v1/holdings`, `/api/v1/sellable-quantity`, `/api/v1/commissions`, `/api/v1/orders`, `/api/v1/orders/{orderId}/modify`, `/api/v1/orders/{orderId}/cancel`를 스펙에 맞춰 준비했습니다. 기본값은 실주문 차단입니다.

토스 조회용 환경 변수 예시는 아래와 같습니다.

```text
TOSS_API_TOKEN=your_toss_api_token_here
TOSS_CLIENT_ID=your_toss_client_id_here
TOSS_CLIENT_SECRET=your_toss_client_secret_here
TOSS_ACCOUNT_SEQ=your_toss_account_seq_here
TOSS_BASE_URL=https://openapi.tossinvest.com
```

토스에서 `API key`와 `Secret key`라는 이름으로 받은 경우에는 아래 이름도 사용할 수 있습니다.

```text
TOSS_API_KEY=your_toss_api_key_here
TOSS_SECRET_KEY=your_toss_secret_key_here
```

`TOSS_API_TOKEN`에 `tsck...` 형태의 API key를 넣고 `TOSS_CLIENT_SECRET`에 `tssk...` 형태의 secret key를 넣은 경우도 자동으로 `client_id/client_secret` 방식으로 처리합니다.

토스 quote는 현재가(`/prices`)와 최우선 bid/ask(`/orderbook`)를 합쳐 만듭니다. 스프레드가 `spread_pct`로 계산되어 데이터 품질 하네스에 자동 반영됩니다. quote 조회가 실패하면 차트 마지막 가격을 fallback으로 사용하고 실패 사유를 risk flag에 남깁니다.

사이드바의 `종목 검색`에 `AAPL`, `애플`, `Nvidia`처럼 입력하면 앱이 후보 심볼을 만들고 `/api/v1/stocks`로 종목 기본 정보를 확인합니다. 선택한 결과는 관심종목 목록에 추가할 수 있습니다.

토스증권 OpenAPI 주문 경로는 연결되어 있지만 기본 차단입니다. 앱의 `Toss API` 설정에서 `Toss 계좌 조회 연결`을 켜면 매수 가능 금액을 실행 계획과 수동 주문 게이트에 반영할 수 있고, `Toss 실주문 전송 허용`은 별도 체크로 분리되어 있습니다. 수동 실주문은 `내 주식 · 포지션 관리`의 `실계좌 · 수동 주문` 패널에서 전송합니다. 실전 연결 전에는 `paper` 모드와 소액 상한으로 충분히 검증해야 합니다.

## 계산식 기반 예측 레이어

실시간 API가 들어온다는 전제에서 `stock_signal/prediction_metrics.py`에 해석 가능한 계산식 기반 예측 점수를 추가했습니다. 이 값은 랜덤포레스트 확률과 별도로 `latest_signal.json`의 `prediction_metrics`에 저장됩니다.

계산 항목:

- 추세: `EMA9 - EMA21`을 ATR 기준으로 정규화
- 모멘텀: 1봉/5봉 수익률 조합
- VWAP 위치: 현재가와 60봉 VWAP의 거리
- 거래량 압력: 현재 거래량과 20봉 평균 거래량 비율
- 돌파 압력: 최근 20봉 저항선 대비 현재가 위치
- RSI 구간: 과열/약세 구간 감점
- 변동성/스프레드 패널티: ATR 대비 과도한 변동성과 넓은 bid/ask 스프레드 감점

출력값은 `score`, `probability_up`, `expected_move_pct`, `projected_price`, `projected_low`, `projected_high`, `components`, `reasons`입니다. 화면 사이드 패널의 `Formula Prediction`에서도 같은 값을 확인할 수 있습니다.

## 장투 / 적립식 탭

사이드바의 `투자 탭`에서 `장투`를 선택하면 선택 종목 기준으로 적립식 투자 보조 화면을 표시합니다.

입력값:

- 월 적립금
- 초기 투자금
- 투자 기간
- 연 기대수익률 가정

출력값:

- 월 예상 매수 주수
- 총 투입 원금과 예상 평가액
- 고점 대비 낙폭, 50일/200일선 위치, 연환산 변동성
- 현재가, -5%, -10%, -15% 기준 분할 매수 구간
- 장기 수익률 시나리오

최근 장투 계산 결과는 `latest_long_term_plan.json`에 저장됩니다.

## LLM 복기 데이터

실제 진입과 청산이 끝난 뒤 `내 주식 · 포지션 관리`의 `LLM 복기 데이터` 영역에 실제 진입가, 청산가, 수량, 수수료, 청산 사유를 입력하면 아래 파일이 생성됩니다.

```text
trade_reviews.jsonl
llm_trade_review.md
```

- `trade_reviews.jsonl`: 거래 1건당 JSON 1줄로 저장되는 원본 데이터입니다. 예측 당시 신호, 계산식 컴포넌트, 매매 의도, 실행 차단 사유, 실제 손익, R값, 방향 일치 여부가 함께 저장됩니다.
- `llm_trade_review.md`: Codex/LLM에게 바로 제공하기 쉬운 Markdown 요약 파일입니다. 누적 승률, 손익, 평균 R, 방향 정확도, 승리/손실 거래의 계산식 컴포넌트 평균 차이를 포함합니다.

이 문서를 바탕으로 계산식 개선을 검토할 때는 단일 거래 결과로 가중치를 바로 바꾸지 말고, 반복적으로 같은 패턴이 확인될 때만 필터나 가중치를 조정하는 것을 기본 원칙으로 둡니다.

## 주의

이 프로그램은 투자 참고용입니다. 주문 실행 지시가 아니며, 실제 매수/매도 판단과 책임은 사용자에게 있습니다. 실시간 API 지연, 프리마켓/애프터마켓 체결, 호가 공백, 뉴스 이벤트를 별도로 확인해야 합니다.
