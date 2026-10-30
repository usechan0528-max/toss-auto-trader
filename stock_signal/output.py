from __future__ import annotations

import json
from dataclasses import asdict

from .strategy import Recommendation


SAFETY_NOTICE = (
    "이 결과는 자동 주문 신호가 아니라 판단 보조 정보입니다. "
    "실제 주문 여부와 수량은 사용자가 직접 결정해야 합니다. "
    "동전주는 급격한 가격 변동, 거래정지, 상장폐지, 유상증자, 역분할 위험이 있습니다."
)


def recommendation_to_json(recommendation: Recommendation) -> str:
    data = asdict(recommendation)
    data["safety_notice"] = SAFETY_NOTICE
    return json.dumps(data, ensure_ascii=False, indent=2)


def recommendation_to_text(recommendation: Recommendation) -> str:
    entry_low, entry_high = recommendation.entry_zone
    target_1, target_2 = recommendation.take_profit
    reasons = "\n".join(f"- {reason}" for reason in recommendation.reason) or "- 조건 미충족"
    invalid_if = "\n".join(f"- {condition}" for condition in recommendation.invalid_if)
    risk_flags = "\n".join(f"- {flag}" for flag in recommendation.risk_flags) or "- 없음"
    return f"""[종목] {recommendation.symbol}
[판단] {recommendation.action}
[현재가] ${recommendation.price:.4f}
[진입 구간] ${entry_low:.4f} ~ ${entry_high:.4f}
[손절가] ${recommendation.stop_loss:.4f}
[1차 익절] ${target_1:.4f}
[2차 익절] ${target_2:.4f}
[손익비] {recommendation.rr_ratio:.2f}
[추천 수량] {recommendation.recommended_shares}주 이하
[예상 손실] ${recommendation.expected_loss_usd:.2f}
[신뢰도] {recommendation.confidence:.2f}
[진입 근거]
{reasons}
[무효 조건]
{invalid_if}
[위험 플래그]
{risk_flags}
[주의]
{SAFETY_NOTICE}
"""
