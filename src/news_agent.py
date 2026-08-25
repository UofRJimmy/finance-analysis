from __future__ import annotations

from typing import Any

from .agent_scoring import score_news_agent
from .analyzer import Analyzer
from .models import ImpactAnalysis, NewsItem


class NewsAgent(Analyzer):
    """Agent responsible for news impact analysis."""

    def analyze(
        self,
        news: NewsItem,
        ticker: str,
        meta: dict[str, Any],
        price_info: str,
        price_change_pct: float | None,
    ) -> ImpactAnalysis:
        analysis = super().analyze(news, ticker, meta, price_info, price_change_pct)
        analysis.agent_scores = {"news_agent": score_news_agent(analysis)}
        return analysis
