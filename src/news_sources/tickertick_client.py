from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from ..models import NewsItem


try:
    import requests
except Exception:
    requests = None


FEED_URL = "https://api.tickertick.com/feed"
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z])?$")
LIVE_STORY_TYPES = ("curated", "market", "earning")
STORY_TYPE_CATEGORIES = {
    "market": "macro",
    "industry": "macro",
    "curated": "company",
    "earning": "company",
    "all": "company",
}


class TickerTickClient:
    """Fetch the newest TickerTick stories for the current watchlist in one request."""

    def __init__(self, timeout_seconds: int = 15):
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session() if requests else None
        if self.session:
            self.session.headers.update({"User-Agent": "finance-analysis-agent/1.0"})

    def fetch_watchlist_news(self, tickers: list[str], story_type: str) -> list[NewsItem]:
        active = sorted({ticker.upper() for ticker in tickers if TICKER_PATTERN.fullmatch(ticker.upper())})
        if not active:
            return []
        if story_type not in {*LIVE_STORY_TYPES, "industry"}:
            raise ValueError(f"Unsupported TickerTick story type: {story_type}")
        if not self.session:
            raise RuntimeError("requests package is unavailable")

        query = _watchlist_query(active, story_type)
        limit = min(200, max(30, len(active) * 20))
        response = self.session.get(FEED_URL, params={"q": query, "n": limit}, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            raise RuntimeError(f"TickerTick feed HTTP {response.status_code}: {response.text[:200]}")
        payload = response.json()
        stories = payload.get("stories") if isinstance(payload, dict) else payload
        if not isinstance(stories, list):
            raise RuntimeError("TickerTick feed returned an unexpected payload")
        return [item for story in stories if (item := _news_from_story(story, active, story_type))]

    def fetch_latest_ticker_news(self, tickers: list[str], limit: int = 12) -> list[NewsItem]:
        """Fetch one compact, unfiltered latest-news feed for a question fallback."""
        active = sorted({ticker.upper() for ticker in tickers if TICKER_PATTERN.fullmatch(ticker.upper())})
        if not active:
            return []
        if not self.session:
            raise RuntimeError("requests package is unavailable")
        ticker_query = _ticker_query(active)
        response = self.session.get(
            FEED_URL,
            params={"q": ticker_query, "n": max(1, min(30, limit))},
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"TickerTick feed HTTP {response.status_code}: {response.text[:200]}")
        payload = response.json()
        stories = payload.get("stories") if isinstance(payload, dict) else payload
        if not isinstance(stories, list):
            raise RuntimeError("TickerTick feed returned an unexpected payload")
        return [item for story in stories if (item := _news_from_story(story, active, "all"))]


class TickerTickNewsPoller:
    """Poll curated, market, and earnings stories for the current watchlist."""

    def __init__(self, client: TickerTickClient, interval_seconds: int):
        self.client = client
        self.interval_seconds = max(60, interval_seconds)
        self._last_fetch = 0.0
        self._last_watchlist: tuple[str, ...] = ()

    def fetch(self, tickers: list[str]) -> list[NewsItem]:
        import time

        active = tuple(sorted({ticker.upper() for ticker in tickers if TICKER_PATTERN.fullmatch(ticker.upper())}))
        now = time.monotonic()
        if active == self._last_watchlist and now - self._last_fetch < self.interval_seconds:
            return []
        self._last_fetch = now
        self._last_watchlist = active
        items: list[NewsItem] = []
        for story_type in LIVE_STORY_TYPES:
            try:
                items.extend(self.client.fetch_watchlist_news(list(active), story_type))
            except Exception:
                # A single story type must not stop the remaining TickerTick queries.
                continue
        return _merge_story_types(items)


def _watchlist_query(tickers: list[str], story_type: str) -> str:
    return f"(and {_ticker_query(tickers)} T:{story_type})"


def _ticker_query(tickers: list[str]) -> str:
    terms = [f"tt:{ticker.lower()}" for ticker in tickers]
    return terms[0] if len(terms) == 1 else f"(or {' '.join(terms)})"


def _news_from_story(story: Any, active_tickers: list[str], story_type: str) -> NewsItem | None:
    if not isinstance(story, dict):
        return None
    headline = str(story.get("title") or "").strip()
    url = str(story.get("url") or "").strip()
    if not headline or not url:
        return None
    tagged = {
        str(ticker).upper()
        for ticker in (story.get("tags") or story.get("tickers") or [])
        if str(ticker).upper() in active_tickers
    }
    if not tagged:
        return None
    timestamp = story.get("time")
    try:
        published_at = datetime.fromtimestamp(float(timestamp) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        published_at = datetime.now(timezone.utc)
    story_id = str(story.get("id") or _hash(f"{url}|{headline}"))
    return NewsItem(
        id=f"tickertick:{story_id}",
        source=f"tickertick:{str(story.get('site') or 'unknown')}",
        headline=headline,
        summary=str(story.get("description") or "").strip(),
        url=url,
        published_at=published_at,
        category=STORY_TYPE_CATEGORIES[story_type],  # type: ignore[arg-type]
        tagged_tickers=sorted(tagged),
        story_type=story_type,
    )


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _merge_story_types(items: list[NewsItem]) -> list[NewsItem]:
    merged: dict[str, NewsItem] = {}
    for item in items:
        existing = merged.get(item.id)
        if not existing:
            merged[item.id] = item
            continue
        types = {
            part
            for value in [existing.story_type, item.story_type]
            for part in str(value or "").split(",")
            if part
        }
        existing.story_type = ",".join(sorted(types))
        existing.tagged_tickers = sorted(set(existing.tagged_tickers) | set(item.tagged_tickers))
    return list(merged.values())
