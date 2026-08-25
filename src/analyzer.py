from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from .models import ImpactAnalysis, NewsItem
from .deepseek_helpers import make_client, parse_json_text, response_text


# The model must return only these fields; code adds ids/timestamps afterwards.
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "magnitude": {"type": "string", "enum": ["large", "medium", "small"]},
        "time_horizon": {"type": "string", "enum": ["intraday", "short_term_days", "medium_long_term"]},
        "priced_in": {"type": "string", "enum": ["priced_in", "partially_priced_in", "not_priced_in", "unclear"]},
        "reasoning_zh": {"type": "string"},
    },
    "required": ["sentiment", "confidence", "magnitude", "time_horizon", "priced_in", "reasoning_zh"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """你是一名美股新闻摘要 Agent。你的主要任务是只总结新闻重点发生了什么，不做利好/利空理由分析，不做价格预测，不给交易建议。
reasoning_zh 只写新闻重点事件摘要，控制在 100 个中文字符以内。
sentiment、confidence、magnitude、time_horizon、priced_in 仅作为程序分流用的结构化标签，必须根据新闻和标的关系客观填写；不要在 reasoning_zh 里解释这些标签。
如果新闻对该标的没有明确关系，sentiment 必须返回 neutral。
严格按照给定的 JSON schema 输出，不要输出 schema 之外的任何文字或 markdown 标记。"""


class Analyzer:
    def __init__(self, api_key: str, model: str):
        self.client = make_client(api_key)
        self.model = model

    def analyze(
        self,
        news: NewsItem,
        ticker: str,
        meta: dict[str, Any],
        price_info: str,
        price_change_pct: float | None,
    ) -> ImpactAnalysis:
        if not self.client:
            raise RuntimeError("新闻分析生成失败：未配置 DEEPSEEK_API_KEY 或 DeepSeek 客户端不可用")
        data = self._call_model(news, ticker, meta, price_info)
        data["reasoning_zh"] = _limit_text(data["reasoning_zh"], 100)
        return ImpactAnalysis(
            news_id=news.id,
            ticker=ticker,
            company_name=str(meta.get("company_name") or ticker),
            sector=meta.get("sector"),
            sentiment=data["sentiment"],
            confidence=data["confidence"],
            magnitude=data["magnitude"],
            time_horizon=data["time_horizon"],
            priced_in=data["priced_in"],
            reasoning_zh=data["reasoning_zh"],
            price_change_pct_since_news=price_change_pct,
            created_at=datetime.now(timezone.utc),
        )

    def _call_model(self, news: NewsItem, ticker: str, meta: dict[str, Any], price_info: str) -> dict[str, str]:
        user_prompt = f"""新闻标题: {news.headline}
新闻摘要: {news.summary}
来源: {news.source}
TickerTick 新闻类型: {news.story_type or '未标注'}
发布时间(UTC): {news.published_at.isoformat()}
需要分析的标的: {ticker} ({meta.get('company_name') or ticker}, 所属行业: {meta.get('sector')})
该标的当前价格信息(可能为空): {price_info}

请只总结这条新闻重点发生了什么。reasoning_zh 不超过 100 个中文字符，不要写利好/利空理由。"""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "impact_analysis",
                            "schema": ANALYSIS_SCHEMA,
                            "strict": True,
                        }
                    },
                )
                return parse_json_text(response_text(response))
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.5)
        raise RuntimeError(f"DeepSeek analysis failed: {last_error}")

def _limit_text(value: str, max_chars: int) -> str:
    value = str(value).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1] + "…"
