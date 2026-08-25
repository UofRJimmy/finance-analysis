from __future__ import annotations

import json

from .deepseek_helpers import make_client, parse_json_text, response_text


CONSERVATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {"type": "string"},
        "score_view": {"type": "string"},
        "veto_risks": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "acceptable_conditions": {"type": "string"},
        "position_view": {"type": "string"},
        "risk_control": {"type": "string"},
    },
    "required": ["stance", "score_view", "veto_risks", "acceptable_conditions", "position_view", "risk_control"],
    "additionalProperties": False,
}


class ConservativeDecisionAgent:
    """Vetoes low-quality or badly positioned opportunities after core scores are ready."""

    def __init__(self, api_key: str, model: str = "deepseek-v4-flash"):
        self.client = make_client(api_key)
        self.model = model

    def analyze(self, context: dict) -> dict:
        if not self.client:
            raise RuntimeError("保守派Agent生成失败：未配置 DEEPSEEK_API_KEY 或 DeepSeek 客户端不可用")
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "system",
                    "content": (
                        "你是保守派股票交易分析 Agent。你宁愿错过上涨，也不愿在风险不清晰时追高。"
                        "你重视安全边际、买入位置、止损空间、基本面质量、成交量和市场环境。"
                        "以数周到数月的波段交易为主，长期交易为辅。必须结合新闻、技术指标、蜡烛图形态、"
                        "公司基本面和市场情绪。不允许追高式建议，必须优先保护本金，其次才是收益。"
                    ),
                },
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "conservative_decision",
                    "schema": CONSERVATIVE_SCHEMA,
                    "strict": True,
                }
            },
        )
        return parse_json_text(response_text(response))
