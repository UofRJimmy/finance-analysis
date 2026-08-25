from __future__ import annotations

import json

from .deepseek_helpers import make_client, parse_json_text, response_text


AGGRESSIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {"type": "string"},
        "score_view": {"type": "string"},
        "opportunity": {"type": "string"},
        "short_term_opportunity": {"type": "string"},
        "long_term_opportunity": {"type": "string"},
        "plan": {"type": "string"},
        "risk": {"type": "string"},
    },
    "required": ["stance", "score_view", "opportunity", "short_term_opportunity", "long_term_opportunity", "plan", "risk"],
    "additionalProperties": False,
}


NEWS_IMPACT_SCHEMA = {
    "type": "object",
    "properties": {
        "impact": {"type": "string"},
    },
    "required": ["impact"],
    "additionalProperties": False,
}


class AggressiveDecisionAgent:
    """Looks for asymmetric upside after the core agents have produced scores."""

    def __init__(self, api_key: str, model: str = "deepseek-v4-flash"):
        self.client = make_client(api_key)
        self.model = model

    def analyze(self, context: dict) -> dict:
        if not self.client:
            raise RuntimeError("激进派Agent生成失败：未配置 DEEPSEEK_API_KEY 或 DeepSeek 客户端不可用")
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "system",
                    "content": (
                        "你是激进派股票交易分析 Agent。你愿意为了更高收益承担更高波动，重点寻找趋势刚启动、"
                        "市场情绪快速变化、资金流入明显、新闻催化强、技术形态突破明显的机会。"
                        "以数天到数周的波段交易为主，长期交易为辅。必须结合新闻、技术指标、蜡烛图形态、"
                        "公司基本面和市场情绪，不允许只因单一指标下结论。必须区分短线情绪机会和长期基本面机会。"
                        "你可以提出进攻型交易计划，但必须写明风险，不得称为确定性机会。"
                    ),
                },
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            text={"format": {"type": "json_schema", "name": "aggressive_decision", "schema": AGGRESSIVE_SCHEMA, "strict": True}},
        )
        return parse_json_text(response_text(response))

    def analyze_news_impact(self, context: dict) -> dict:
        if not self.client:
            raise RuntimeError("激进派新闻影响分析生成失败：未配置 DEEPSEEK_API_KEY 或 DeepSeek 客户端不可用")
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "system",
                    "content": (
                        "你是激进派股票交易分析 Agent。现在只分析这条新闻对指定股票的影响。"
                        "只基于新闻和提供的公司背景，严格使用以下四步框架，每步一行："
                        "1. 定性：明确这是宏观类、产业类或公司类事件；"
                        "2. 定位：在利好、利空、传导利好利空、竞争利空中选最符合的一项；"
                        "3. 推演：构建“事件-变量-业绩-估值”链条。仅当新闻给出可验证数据时估算 EPS 增厚，"
                        "否则明确不可量化；区分讲故事的短暂影响、降成本/提销量的深远影响、短期利润与长期护城河的取舍，"
                        "并点出高开低走、内部人出货、非经常性损益等适用陷阱；"
                        "4. 交叉验证：判断消息符合预期、超预期或不及预期，并说明需要验证的关键事实。"
                        "最后追加“结论：利好”或“结论：利空”。保持偏进攻风格，但必须识别会令进攻计划失败的风险。"
                        "不得使用技术指标、量价、K线或蜡烛图；不输出来源、置信度、强度或 Agent 评分；"
                        "不提供买入、卖出、加仓或减仓等直接指令。总计控制在 360 个中文字符以内。"
                    ),
                },
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            text={"format": {"type": "json_schema", "name": "aggressive_news_impact", "schema": NEWS_IMPACT_SCHEMA, "strict": True}},
        )
        return parse_json_text(response_text(response))
