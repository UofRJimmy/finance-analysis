from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal, Optional


NewsCategory = Literal["company", "general", "macro"]
Sentiment = Literal["bullish", "bearish", "neutral"]
Confidence = Literal["high", "medium", "low"]
Magnitude = Literal["large", "medium", "small"]
TimeHorizon = Literal["intraday", "short_term_days", "medium_long_term"]
PricedIn = Literal["priced_in", "partially_priced_in", "not_priced_in", "unclear"]


@dataclass
class NewsItem:
    id: str
    source: str
    headline: str
    summary: str
    url: str
    published_at: datetime
    category: NewsCategory
    tagged_tickers: list[str]
    story_type: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["published_at"] = self.published_at.isoformat()
        return data


@dataclass
class ImpactAnalysis:
    news_id: str
    ticker: str
    company_name: str
    sector: Optional[str]
    sentiment: Sentiment
    confidence: Confidence
    magnitude: Magnitude
    time_horizon: TimeHorizon
    priced_in: PricedIn
    reasoning_zh: str
    price_change_pct_since_news: Optional[float]
    created_at: datetime
    technical_analysis: Optional[dict] = None
    combined_conclusion_zh: Optional[str] = None
    agent_scores: Optional[dict] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


@dataclass
class Quote:
    symbol: str
    current: Optional[float]
    previous_close: Optional[float]
    change_pct: Optional[float]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
