from __future__ import annotations

import json

from .deepseek_helpers import make_client, parse_json_text, response_text


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_stance": {"type": "string"},
        "weighted_score_summary": {"type": "string"},
        "aggressive_summary": {"type": "string"},
        "conservative_summary": {"type": "string"},
        "decision": {"type": "string"},
        "risk_opportunity_ratio": {"type": "string"},
        "key_risks": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "tracking_signals": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    },
    "required": [
        "final_stance",
        "weighted_score_summary",
        "aggressive_summary",
        "conservative_summary",
        "decision",
        "risk_opportunity_ratio",
        "key_risks",
        "tracking_signals",
    ],
    "additionalProperties": False,
}


class JudgeDecisionAgent:
    """Summarizes aggressive and conservative agents into the final report paragraph."""

    def __init__(self, api_key: str, model: str = "deepseek-v4-flash"):
        self.client = make_client(api_key)
        self.model = model

    def judge(self, context: dict, aggressive: dict, conservative: dict) -> dict:
        if not self.client:
            raise RuntimeError("裁判Agent生成失败：未配置 DEEPSEEK_API_KEY 或 DeepSeek 客户端不可用")
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "system",
                    "content": (
                        "你是裁判型股票决策 Agent。你不重新编造数据，只总结激进派和保守派的分歧与共识。"
                        "你必须基于基本面40%、技术面30%、消息面20%、市场情绪10%的综合评分，给出最终倾向。"
                        "weighted_score_summary只写一句综合分，不解释计算过程；tracking_signals可以返回空数组。"
                        "risk_opportunity_ratio必须输出风险与机会的比例，格式为“风险:机会”，总和必须等于10，例如“4:6”；"
                        "风险越高前面的数字越大，机会越高后面的数字越大。"
                        "可以使用“进攻、持有、等待回调、降低仓位、暂不通过”等表述，"
                        "但不得写成确定性交易指令。结尾语义必须保留不构成投资建议。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"context": context, "aggressive_agent": aggressive, "conservative_agent": conservative},
                        ensure_ascii=False,
                    ),
                },
            ],
            text={"format": {"type": "json_schema", "name": "judge_decision", "schema": JUDGE_SCHEMA, "strict": True}},
        )
        return parse_json_text(response_text(response))


def market_sentiment_from_vix(vix: dict) -> dict:
    if not vix or vix.get("available") is False:
        return {"total_score": 50, "direction": "市场情绪数据不足", "components": []}
    value = float(vix.get("last") or 0)
    if value <= 15:
        score, direction = 80, "波动率偏低"
    elif value <= 20:
        score, direction = 65, "波动率温和"
    elif value <= 25:
        score, direction = 50, "波动率中性"
    elif value <= 30:
        score, direction = 35, "波动率偏高"
    else:
        score, direction = 20, "波动率高压"
    return {
        "total_score": score,
        "direction": direction,
        "components": [{"name": "VIX", "score": score, "reason": f"VIX={value:g}。"}],
    }


def weighted_core_score(
    *,
    fundamental_score: dict | None,
    technical_score: dict | None,
    news_score: dict | None,
    market_sentiment_score: dict | None,
) -> dict:
    f = _score_or_neutral(fundamental_score)
    t = _score_or_neutral(technical_score)
    n = _score_or_neutral(news_score)
    m = _score_or_neutral(market_sentiment_score)
    total = round(f * 0.4 + t * 0.3 + n * 0.2 + m * 0.1)
    direction = "偏强" if total >= 70 else "中性偏强" if total >= 55 else "中性偏弱" if total >= 40 else "偏弱"
    return {
        "total_score": total,
        "direction": direction,
        "weights": {
            "fundamental_40pct": f,
            "technical_30pct": t,
            "news_20pct": n,
            "market_sentiment_10pct": m,
        },
    }


def format_judge_block(result: dict, context: dict | None = None) -> str:
    risks = "；".join(result.get("key_risks") or [])
    weighted = _format_weighted_score(context, result)
    return (
        "三方决策总结：\n"
        f"- 最终倾向：{result.get('final_stance', '暂无结论')}\n"
        f"- 加权评分：{weighted}\n"
        f"- 激进派：{result.get('aggressive_summary', '')}\n"
        f"- 保守派：{result.get('conservative_summary', '')}\n"
        f"- 风险机会比：{result.get('risk_opportunity_ratio', '暂无')}（风险:机会，总和10）\n"
        f"- 裁判结论：{result.get('decision', '')}\n"
        f"- 主要风险：{risks or '暂无'}\n"
        "这不构成投资建议。"
    )


def _format_weighted_score(context: dict | None, result: dict) -> str:
    if not context:
        return result.get("weighted_score_summary", "") or "暂无"
    weighted = context.get("weighted_score") or {}
    parts = [
        f"综合{weighted.get('total_score', 0)}/100（{weighted.get('direction', '未知')}）",
        _agent_score_text("基本面", context.get("fundamental_agent_score")),
        _agent_score_text("技术", context.get("technical_agent_score")),
        _agent_score_text("消息", context.get("news_agent_score")),
        _agent_score_text("市场情绪", context.get("market_sentiment_score")),
    ]
    return "；".join(part for part in parts if part)


def _agent_score_text(label: str, score: dict | None) -> str:
    if not score:
        return f"{label}暂无"
    return f"{label}{score.get('total_score', 0)}/100（{score.get('direction', '未知')}）"

def _score_or_neutral(score: dict | None) -> float:
    if not score:
        return 50.0
    try:
        return float(score.get("total_score", 50))
    except (TypeError, ValueError):
        return 50.0
