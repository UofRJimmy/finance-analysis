from __future__ import annotations

from .agent_scoring import score_technical_agent
from .close_risk import CloseRiskAnalyzer
from .models import ImpactAnalysis, NewsItem
from .technical_analysis import TechnicalAnalyzer


class TechnicalAgent(TechnicalAnalyzer):
    """Agent responsible for intraday and daily technical analysis."""

    def get_snapshot(self, ticker: str) -> dict:
        snapshot = super().get_snapshot(ticker)
        snapshot["agent_score"] = score_technical_agent(snapshot)
        return snapshot

    def analyze_news(self, news: NewsItem, impact: ImpactAnalysis) -> dict:
        result = super().analyze_news(news, impact)
        result["snapshot"]["agent_score"] = score_technical_agent(result["snapshot"])
        return result


class CloseRiskTechnicalAgent(CloseRiskAnalyzer):
    """Agent responsible for close-session daily/weekly risk analysis."""

    def analyze(self, ticker: str) -> dict:
        snapshot = super().analyze(ticker)
        snapshot["agent_score"] = score_technical_agent(snapshot)
        return snapshot
