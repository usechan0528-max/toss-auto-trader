from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

from app import normalize_toss_auth, secret_value
from auto_session_trader import _position_summary_rows
from stock_signal.toss_broker import TossBrokerExecutionAdapter


APP_DIR = Path(__file__).resolve().parent
SESSION_EVENT_PATH = APP_DIR / "auto_session_events.jsonl"
AUTO_TRADE_PATH = APP_DIR / "latest_auto_trade.json"
TRADE_INTENT_PATH = APP_DIR / "latest_trade_intent.json"
LATEST_SIGNAL_PATH = APP_DIR / "latest_signal.json"
LOCK_FILE = APP_DIR / "auto_session_trader.lock"
REALTIME_POSITIONS_PATH = APP_DIR / "latest_realtime_positions.json"
SETTINGS_PATH = APP_DIR / "app_settings.json"
POSITION_CACHE_TTL_SECONDS = 5.0
_POSITION_CACHE: dict[str, Any] = {"expires_at": 0.0, "payload": {}}

load_dotenv(APP_DIR / ".env", override=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path, limit: int = 30) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _setting_float(settings: dict[str, Any], key: str, default: float) -> float:
    return _as_float(settings.get(key), default)


def _env_or_setting_float(env_name: str, settings: dict[str, Any], key: str, default: float) -> float:
    return _as_float(secret_value(env_name, ""), _setting_float(settings, key, default))


def _symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _first_target(value: Any) -> float:
    if isinstance(value, list) and value:
        return _as_float(value[0])
    return _as_float(value)


def _collect_position_targets(*sources: Any) -> dict[str, dict[str, float]]:
    targets: dict[str, dict[str, float]] = {}

    def absorb(row: Any) -> None:
        if not isinstance(row, dict):
            return
        symbol = _symbol(row.get("symbol") or row.get("ticker"))
        if not symbol:
            return
        stop_loss = _as_float(row.get("stop_loss"))
        take_profit_1 = _as_float(row.get("take_profit_1"))
        if take_profit_1 <= 0:
            take_profit_1 = _first_target(row.get("take_profit"))
        if take_profit_1 <= 0:
            take_profit_1 = _as_float(row.get("target_price"))
        existing = targets.setdefault(symbol, {})
        if stop_loss > 0:
            existing["stop_loss"] = stop_loss
        if take_profit_1 > 0:
            existing["take_profit_1"] = take_profit_1

    def walk(source: Any) -> None:
        if isinstance(source, dict):
            absorb(source)
            for key in ("position_summary", "positions", "auto_exit", "trade_intent", "latest_trade_intent"):
                child = source.get(key)
                if isinstance(child, list):
                    for item in child:
                        absorb(item)
                else:
                    absorb(child)
        elif isinstance(source, list):
            for item in source:
                absorb(item)

    for source in sources:
        walk(source)
    return targets


def _enrich_position_targets(
    positions: list[dict[str, Any]],
    *,
    fallback_positions: list[dict[str, Any]],
    latest_signal: dict[str, Any],
    latest_event: dict[str, Any],
) -> list[dict[str, Any]]:
    settings = _read_json(SETTINGS_PATH)
    manual_stop_pct = abs(_env_or_setting_float("MANUAL_POSITION_STOP_LOSS_PCT", settings, "manual_position_stop_loss_pct", 2.0))
    manual_target_pct = abs(
        _env_or_setting_float("MANUAL_POSITION_TAKE_PROFIT_PCT", settings, "manual_position_take_profit_pct", 2.0)
    )
    target_lookup = _collect_position_targets(fallback_positions, latest_signal, latest_event)
    enriched: list[dict[str, Any]] = []
    for raw_position in positions:
        position = dict(raw_position)
        symbol = _symbol(position.get("symbol") or position.get("ticker"))
        entry_price = _as_float(position.get("entry_price"))
        current_price = _as_float(position.get("current_price"))
        basis_price = entry_price if entry_price > 0 else current_price
        stop_loss = _as_float(position.get("stop_loss"))
        take_profit_1 = _as_float(position.get("take_profit_1"))
        source = ""

        planned = target_lookup.get(symbol, {})
        planned_stop = _as_float(planned.get("stop_loss"))
        planned_target = _as_float(planned.get("take_profit_1"))
        if stop_loss <= 0 and planned_stop > 0:
            stop_loss = planned_stop
            source = "plan"
        if take_profit_1 <= 0 and planned_target > 0:
            take_profit_1 = planned_target
            source = "plan"

        if basis_price > 0:
            manual_stop_loss = basis_price * (1 - manual_stop_pct / 100)
            manual_target = basis_price * (1 + manual_target_pct / 100)
            if stop_loss <= 0 or manual_stop_loss < stop_loss:
                stop_loss = manual_stop_loss
                source = "manual" if source else "manual_default"
            if manual_target > take_profit_1:
                take_profit_1 = manual_target
                source = "manual"

        stop_gap_pct = (current_price / stop_loss - 1) * 100 if current_price > 0 and stop_loss > 0 else 0.0
        target_gap_pct = (take_profit_1 / current_price - 1) * 100 if current_price > 0 and take_profit_1 > 0 else 0.0
        position["stop_loss"] = round(stop_loss, 4)
        position["take_profit_1"] = round(take_profit_1, 4)
        position["stop_gap_pct"] = round(stop_gap_pct, 4)
        position["target_gap_pct"] = round(target_gap_pct, 4)
        position["target_source"] = source or str(position.get("target_source") or "")
        position["manual_stop_loss_pct"] = round(manual_stop_pct, 4)
        position["manual_take_profit_pct"] = round(manual_target_pct, 4)
        enriched.append(position)
    return enriched


def _toss_broker() -> TossBrokerExecutionAdapter:
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
        allow_live_orders=False,
    )


def _live_position_state(
    fallback_positions: list[dict[str, Any]],
    latest_signal: dict[str, Any],
    latest_event: dict[str, Any],
) -> dict[str, Any]:
    now = time.monotonic()
    cached = _POSITION_CACHE.get("payload")
    if isinstance(cached, dict) and now < float(_POSITION_CACHE.get("expires_at", 0.0)):
        return cached

    try:
        broker = _toss_broker()
        holdings = broker.get_holdings()
        try:
            commissions = broker.get_commissions()
        except Exception:
            commissions = []
        positions = _position_summary_rows(holdings, {}, commissions)
        positions = _enrich_position_targets(
            positions,
            fallback_positions=fallback_positions,
            latest_signal=latest_signal,
            latest_event=latest_event,
        )
        payload = {
            "generated_at": _now_iso(),
            "source": "toss_api",
            "positions": positions,
            "position_count": len(positions),
            "error": "",
        }
        _write_json(REALTIME_POSITIONS_PATH, payload)
    except Exception as exc:
        previous = _read_json(REALTIME_POSITIONS_PATH)
        previous_positions = previous.get("positions") if isinstance(previous.get("positions"), list) else []
        if previous_positions:
            positions = previous_positions
            source = "cached_toss_api"
            generated_at = str(previous.get("generated_at") or _now_iso())
        else:
            positions = fallback_positions
            source = "worker_event"
            generated_at = _now_iso()
        positions = _enrich_position_targets(
            positions,
            fallback_positions=fallback_positions,
            latest_signal=latest_signal,
            latest_event=latest_event,
        )
        payload = {
            "generated_at": generated_at,
            "source": source,
            "positions": positions,
            "position_count": len(positions),
            "error": str(exc),
        }

    _POSITION_CACHE["payload"] = payload
    _POSITION_CACHE["expires_at"] = now + POSITION_CACHE_TTL_SECONDS
    return payload


def _state() -> dict[str, Any]:
    events = _read_jsonl(SESSION_EVENT_PATH, limit=40)
    latest_event = events[-1] if events else {}
    latest_selection = latest_event.get("auto_selection") if isinstance(latest_event.get("auto_selection"), dict) else {}
    latest_signal = _read_json(LATEST_SIGNAL_PATH)
    top_surge = latest_event.get("top_surge_candidates") or latest_selection.get("top_surge_candidates") or []
    candidate_scores = latest_event.get("candidate_scores") or latest_selection.get("candidate_scores") or []
    fallback_positions = latest_event.get("position_summary") or latest_signal.get("position_summary") or []
    fallback_positions = fallback_positions if isinstance(fallback_positions, list) else []
    live_positions = _live_position_state(fallback_positions, latest_signal, latest_event)
    position_summary = live_positions.get("positions") if isinstance(live_positions.get("positions"), list) else []
    return {
        "generated_at": _now_iso(),
        "worker_lock": _read_json(LOCK_FILE),
        "latest_event": latest_event,
        "latest_signal": latest_signal,
        "latest_auto_trade": _read_json(AUTO_TRADE_PATH),
        "latest_trade_intent": _read_json(TRADE_INTENT_PATH),
        "top_surge_candidates": top_surge if isinstance(top_surge, list) else [],
        "candidate_scores": candidate_scores if isinstance(candidate_scores, list) else [],
        "position_summary": position_summary if isinstance(position_summary, list) else [],
        "position_summary_source": str(live_positions.get("source") or ""),
        "position_summary_generated_at": str(live_positions.get("generated_at") or ""),
        "position_summary_error": str(live_positions.get("error") or ""),
        "recent_events": events[-12:],
        "paths": {
            "session_events": str(SESSION_EVENT_PATH),
            "auto_trade": str(AUTO_TRADE_PATH),
            "trade_intent": str(TRADE_INTENT_PATH),
            "lock": str(LOCK_FILE),
        },
    }


HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>자동매매 실시간 관찰</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #080b10;
      --panel: #111821;
      --panel-2: #151f2b;
      --text: #eef4ff;
      --muted: #8ea0b5;
      --line: #223044;
      --blue: #5aa7ff;
      --green: #42d392;
      --red: #ff6673;
      --yellow: #ffd166;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: center;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: rgba(8, 11, 16, 0.94);
      backdrop-filter: blur(10px);
    }
    h1 { margin: 0; font-size: 20px; letter-spacing: 0; }
    main { padding: 20px 24px 36px; display: grid; gap: 16px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .wide { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 12px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }
    .label { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .value { font-size: 22px; font-weight: 700; overflow-wrap: anywhere; }
    .small { color: var(--muted); font-size: 12px; }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 3px 10px;
      border-radius: 999px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      font-size: 12px;
      color: var(--muted);
      margin: 2px 4px 2px 0;
    }
    .green { color: var(--green); }
    .red { color: var(--red); }
    .blue { color: var(--blue); }
    .yellow { color: var(--yellow); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 9px 8px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
    th { color: var(--muted); font-weight: 600; }
    td.num, th.num { text-align: right; }
    .status-dot {
      display: inline-block;
      width: 9px;
      height: 9px;
      border-radius: 50%;
      margin-right: 7px;
      background: var(--red);
    }
    .status-dot.live { background: var(--green); }
    @media (max-width: 980px) {
      .grid, .wide { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>자동매매 실시간 관찰</h1>
      <div class="small">2초마다 워커 이벤트만 읽어서 갱신합니다. 주문 로직은 실행하지 않습니다.</div>
    </div>
    <div class="small" id="clock">-</div>
  </header>
  <main>
    <section class="grid">
      <div class="panel"><div class="label">워커</div><div class="value" id="worker">-</div><div class="small" id="heartbeat">-</div></div>
      <div class="panel"><div class="label">자동 상태</div><div class="value" id="status">-</div><div class="small" id="state">-</div></div>
      <div class="panel"><div class="label">후보군</div><div class="value" id="candidates">-</div><div class="small" id="surge">-</div><div class="small" id="wideScan">wide scan -</div></div>
      <div class="panel"><div class="label">주문 의도</div><div class="value" id="intent">-</div><div class="small" id="order">-</div></div>
    </section>
    <section class="panel">
      <div class="label">판단 사유</div>
      <div id="reasons"></div>
    </section>
    <section class="panel">
      <div class="label">반등 여지</div>
      <div class="value" id="reboundStatus">-</div>
      <div class="small" id="reboundDetails">-</div>
      <div id="reboundReasons"></div>
    </section>
    <section class="panel">
      <div class="label">지지/저항 구조</div>
      <div class="value" id="structureStatus">-</div>
      <div class="small" id="structureDetails">-</div>
      <div id="structureReasons"></div>
    </section>
    <section class="panel">
      <div class="label">실계좌 포지션 <span id="positionCount"></span></div>
      <div style="overflow:auto;">
        <table>
          <thead>
            <tr>
              <th>종목</th><th class="num">수량</th><th class="num">진입가</th><th class="num">현재가</th>
              <th class="num">평가금액</th><th class="num">수수료반영손익</th><th class="num">수익률</th>
              <th class="num">예상수수료</th><th class="num">손절까지</th><th class="num">목표까지</th>
            </tr>
          </thead>
          <tbody id="positionRows"></tbody>
        </table>
      </div>
    </section>
    <section class="wide">
      <div class="panel">
        <div class="label">급등 관찰 상위</div>
        <table>
          <thead><tr><th>종목</th><th class="num">급등점수</th><th class="num">5분%</th><th class="num">직전봉%</th><th class="num">거래량배율</th><th class="num">거래량Z</th><th class="num">돌파%</th></tr></thead>
          <tbody id="surgeRows"></tbody>
        </table>
      </div>
      <div class="panel">
        <div class="label">최근 워커 이벤트</div>
        <table>
          <thead><tr><th>시간</th><th>상태</th><th>종목</th><th>의도</th><th class="num">수량</th></tr></thead>
          <tbody id="eventRows"></tbody>
        </table>
      </div>
    </section>
    <section class="panel">
      <div class="label">전체 후보 실시간 점수 <span id="candidateScoreCount"></span></div>
      <div style="overflow:auto; max-height:520px;">
        <table>
          <thead>
            <tr>
              <th class="num">#</th><th>종목</th><th class="num">자동점수</th><th class="num">승률%</th>
              <th>방향</th><th class="num">급등</th><th class="num">5분%</th><th class="num">직전%</th>
              <th class="num">거래량배율</th><th class="num">거래량Z</th><th>리스크</th><th>통과</th>
            </tr>
          </thead>
          <tbody id="candidateScoreRows"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const fmt = (value, digits = 2) => {
      const n = Number(value || 0);
      return Number.isFinite(n) ? n.toFixed(digits) : "-";
    };
    const text = (id, value) => { document.getElementById(id).textContent = value ?? "-"; };
    const pillHtml = (value) => `<span class="pill">${String(value)}</span>`;
    const clsForStatus = (status) => status === "LIVE_SUBMITTED" ? "green" : status === "AUTO_BLOCKED" ? "yellow" : status ? "blue" : "";

    async function refresh() {
      try {
        const response = await fetch("/api/state", { cache: "no-store" });
        const data = await response.json();
        const event = data.latest_event || {};
        const selection = event.auto_selection || {};
        const lock = data.worker_lock || {};
        const status = event.status || (lock.pid ? "RUNNING" : "WAITING");
        const symbol = event.symbol || selection.symbol || "-";
        const side = event.trade_intent_side || "-";
        const qty = event.trade_intent_quantity ?? 0;
        const dotClass = lock.pid ? "live" : "";

        text("clock", `갱신 ${new Date().toLocaleTimeString()} · 이벤트 ${event.logged_at || "-"}`);
        document.getElementById("worker").innerHTML = `<span class="status-dot ${dotClass}"></span>${lock.pid ? "RUNNING" : "STOPPED"}`;
        text("heartbeat", lock.last_heartbeat ? `heartbeat ${lock.last_heartbeat}` : "lock 없음");
        document.getElementById("status").innerHTML = `<span class="${clsForStatus(status)}">${status}</span>`;
        text("state", `${event.mode || "-"} · ${event.state || "-"}`);
        text("candidates", `${event.candidate_universe_size || 0} / ${event.liquidity_passed_count || 0}`);
        text("surge", `급등 모드 ${event.surge_focus_enabled ? "ON" : "OFF"} · 통과 ${event.surge_candidate_count || 0}`);
        const wide = event.wide_scan || selection.wide_scan || {};
        text("wideScan", wide.enabled ? `wide ${wide.scanned_count || 0}/${wide.universe_size || 0} · active ${wide.active_count || 0} · promoted ${wide.promoted_count || 0} · model ${wide.modeled_symbols_count || 0}` : "wide scan OFF");
        text("intent", `${side} ${qty}`);
        text("order", `${symbol} · 실행 ${event.execution_status || "-"}`);

        const reasons = event.reasons || selection.reason || [];
        document.getElementById("reasons").innerHTML = reasons.length ? reasons.slice(0, 10).map(pillHtml).join("") : pillHtml("관찰 중");

        const latestSignal = data.latest_signal || {};
        const rebound = event.rebound_potential || latestSignal.rebound_potential || {};
        const reboundReasons = Array.isArray(rebound.reason) ? rebound.reason : [];
        if (Object.keys(rebound).length) {
          text("reboundStatus", `${rebound.active ? "방어 ON" : "방어 OFF"} · ${fmt(rebound.score, 2)}점 · 표본 ${rebound.sample_count ?? 0}`);
          text("reboundDetails", `hit ${fmt(rebound.hit_rate_pct, 2)}% · 평균반등 ${fmt(rebound.avg_max_return_pct, 3)}% · 기준 ${fmt(rebound.threshold_pct, 3)}%`);
          document.getElementById("reboundReasons").innerHTML = reboundReasons.length ? reboundReasons.slice(0, 6).map(pillHtml).join("") : pillHtml("REBOUND_DATA_WAITING");
        } else {
          text("reboundStatus", "-");
          text("reboundDetails", "-");
          document.getElementById("reboundReasons").innerHTML = pillHtml("REBOUND_DATA_WAITING");
        }
        const structure = event.support_resistance || latestSignal.support_resistance || {};
        const structureReasons = Array.isArray(structure.reason) ? structure.reason : [];
        if (Object.keys(structure).length) {
          text("structureStatus", `${structure.state || "-"} · 하단 ${structure.lower_breakdown ? "이탈" : "유지"} · 상단 ${structure.upper_breakout ? "돌파" : "대기"}`);
          text("structureDetails", `지지 ${fmt(structure.support, 4)} (${fmt(structure.support_gap_pct, 3)}%) · 저항 ${fmt(structure.resistance, 4)} (${fmt(structure.resistance_gap_pct, 3)}%) · 버퍼 ${fmt(structure.break_buffer_pct, 3)}%`);
          document.getElementById("structureReasons").innerHTML = structureReasons.length ? structureReasons.slice(0, 6).map(pillHtml).join("") : pillHtml("STRUCTURE_DATA_WAITING");
        } else {
          text("structureStatus", "-");
          text("structureDetails", "-");
          document.getElementById("structureReasons").innerHTML = pillHtml("STRUCTURE_DATA_WAITING");
        }

        const positions = data.position_summary || event.position_summary || [];
        const posSource = data.position_summary_source || "event";
        const posUpdated = data.position_summary_generated_at ? String(data.position_summary_generated_at).slice(11, 19) : "-";
        text("positionCount", positions.length ? `(${positions.length} ${posSource} ${posUpdated})` : `(${posSource} ${posUpdated})`);
        const positionRows = document.getElementById("positionRows");
        positionRows.innerHTML = positions.length ? positions.map(item => {
          const pnl = Number(item.fee_adjusted_pnl_usd || 0);
          const colorClass = pnl > 0 ? "green" : pnl < 0 ? "red" : "";
          return `<tr>
            <td>${item.symbol || ""}</td>
            <td class="num">${fmt(item.quantity, 4)}</td>
            <td class="num">${fmt(item.entry_price, 4)}</td>
            <td class="num">${fmt(item.current_price, 4)}</td>
            <td class="num">$${fmt(item.market_value_usd, 2)}</td>
            <td class="num ${colorClass}">$${fmt(item.fee_adjusted_pnl_usd, 2)}</td>
            <td class="num ${colorClass}">${fmt(item.fee_adjusted_return_pct, 2)}%</td>
            <td class="num">$${fmt(item.estimated_roundtrip_fee_usd, 4)}</td>
            <td class="num">${fmt(item.stop_gap_pct, 2)}%</td>
            <td class="num">${fmt(item.target_gap_pct, 2)}%</td>
          </tr>`;
        }).join("") : `<tr><td colspan="10">현재 표시할 실계좌 포지션이 없습니다.</td></tr>`;

        const surgeRows = document.getElementById("surgeRows");
        const top = data.top_surge_candidates || [];
        surgeRows.innerHTML = top.length ? top.slice(0, 10).map(item => `
          <tr>
            <td>${item.symbol || ""}</td>
            <td class="num">${fmt(item.surge_score, 2)}</td>
            <td class="num">${fmt(item.lookback_change_pct, 3)}</td>
            <td class="num">${fmt(item.latest_bar_change_pct, 3)}</td>
            <td class="num">${fmt(item.volume_ratio, 2)}</td>
            <td class="num">${fmt(item.volume_zscore, 2)}</td>
            <td class="num">${fmt(item.breakout_pct, 3)}</td>
          </tr>`).join("") : `<tr><td colspan="7">표시할 급등 후보가 없습니다.</td></tr>`;

        const candidateScores = data.candidate_scores || event.candidate_scores || [];
        text("candidateScoreCount", candidateScores.length ? `(${candidateScores.length})` : "");
        const candidateRows = document.getElementById("candidateScoreRows");
        candidateRows.innerHTML = candidateScores.length ? candidateScores.map(item => `
          <tr>
            <td class="num">${item.rank ?? ""}</td>
            <td>${item.symbol || ""}</td>
            <td class="num">${fmt(item.auto_score, 2)}</td>
            <td class="num">${fmt(item.confidence_pct, 2)}</td>
            <td>${item.direction || ""}</td>
            <td class="num">${fmt(item.surge_score, 2)}</td>
            <td class="num">${fmt(item.lookback_change_pct, 3)}</td>
            <td class="num">${fmt(item.latest_bar_change_pct, 3)}</td>
            <td class="num">${fmt(item.surge_volume_ratio || item.volume_ratio, 2)}</td>
            <td class="num">${fmt(item.surge_volume_zscore || item.volume_zscore, 2)}</td>
            <td>${item.risk || ""}</td>
            <td>${item.auto_gate_passed ? "BUY" : item.surge_passed ? "WATCH" : ""}</td>
          </tr>`).join("") : `<tr><td colspan="12">아직 전체 후보 점수가 없습니다. 다음 워커 루프 이후 표시됩니다.</td></tr>`;

        const eventRows = document.getElementById("eventRows");
        const recent = data.recent_events || [];
        eventRows.innerHTML = recent.length ? recent.slice(-10).reverse().map(row => {
          const rowSelection = row.auto_selection || {};
          return `<tr>
            <td>${String(row.logged_at || "").slice(11, 19)}</td>
            <td>${row.status || ""}</td>
            <td>${row.symbol || rowSelection.symbol || ""}</td>
            <td>${row.trade_intent_side || ""}</td>
            <td class="num">${row.trade_intent_quantity ?? ""}</td>
          </tr>`;
        }).join("") : `<tr><td colspan="5">이벤트가 없습니다.</td></tr>`;
      } catch (error) {
        text("status", "MONITOR_ERROR");
        text("state", String(error));
      }
    }

    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


class MonitorHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(_state())
            return
        if path in {"/", "/index.html"}:
            self._send_text(HTML, content_type="text/html; charset=utf-8")
            return
        self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, *, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, int(args.port)), MonitorHandler)
    print(f"live monitor listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
