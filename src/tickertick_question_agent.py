from __future__ import annotations

import json

from .deepseek_helpers import make_client, response_text
from .news_sources.tickertick_client import TickerTickClient


class TickerTickQuestionAgent:
    """Answer out-of-scope market questions from a fresh TickerTick feed."""

    def __init__(self, api_key: str, model: str):
        self.client = make_client(api_key)
        self.model = model
        self.news = TickerTickClient()

    def answer(self, question: str, tickers: list[str]) -> str:
        if not self.client:
            raise RuntimeError("未配置 DeepSeek API，无法整理 TickerTick 新闻")
        stories = self.news.fetch_latest_ticker_news(tickers)
        if not stories:
            return "TickerTick 当前没有返回与该问题相关的最新股票新闻。"
        payload = [
            {
                "ticker_tags": story.tagged_tickers,
                "headline": story.headline,
                "summary": story.summary,
                "published_at": story.published_at.isoformat(),
                "source": story.source,
                "url": story.url,
            }
            for story in stories
        ]
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "你是 TickerTick 新闻问答 Agent。只能根据提供的最新新闻回答金融市场问题，"
                        "不能使用本地持仓、历史对话或训练记忆补充近期事实。"
                        "先用一句话回答问题，再说明新闻依据和不确定性；不提供买卖指令。"
                        "若新闻不足以回答，明确说数据不足。使用中文，简洁。"
                    ),
                },
                {"role": "user", "content": json.dumps({"question": question, "news": payload}, ensure_ascii=False)},
            ],
        )
        answer = response_text(response).strip()
        if not answer:
            raise RuntimeError("TickerTick 新闻问答模型未返回内容")
        sources = "\n".join(f"- {story.headline}: {story.url}" for story in stories[:6])
        return f"TickerTick 新闻查询结果（非本地持仓数据库）：\n\n{answer}\n\n新闻来源：\n{sources}"
