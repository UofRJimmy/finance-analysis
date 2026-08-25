from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from .config import Settings
from .dedup_store import DedupStore
from .models import ImpactAnalysis, NewsItem
from .market_data import YahooMarketData
from .news_agent import NewsAgent
from .news_sources.finnhub_client import FinnhubClient
from .news_sources.tickertick_client import TickerTickClient, TickerTickNewsPoller
from .reporter import Reporter
from .technical_agent import TechnicalAgent
from .ticker_meta import TickerMetaCache
from .watchlist import Watchlist


class AgentPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.watchlist = Watchlist(settings.watchlist_file)
        self.finnhub = FinnhubClient(
            settings.finnhub_api_key,
            min_request_interval=settings.finnhub_min_request_interval_seconds,
        )
        self.tickertick_news = TickerTickNewsPoller(
            TickerTickClient(), settings.tickertick_company_news_interval_seconds
        )
        self.store = DedupStore(settings.sqlite_path, max_news_items=settings.news_db_max_items)
        self.meta_cache = TickerMetaCache(settings.ticker_meta_path, self.finnhub)
        self.news_agent = NewsAgent(settings.deepseek_api_key, settings.deepseek_model_analyze)
        self.technical_agent = TechnicalAgent(
            YahooMarketData(),
            settings.deepseek_api_key,
            settings.deepseek_model_analyze,
        )
        self.reporter = Reporter(
            settings.report_dir,
            settings.display_timezone,
            settings.dingtalk_webhook_url,
            settings.dingtalk_keyword,
            settings.dingtalk_secret,
        )

    def run_once(self) -> int:
        tickers = sorted(set(self.watchlist.get()))
        if not tickers:
            return 0
        # The profile cache gives the relevance gate company and sector context.
        meta = self.meta_cache.warm(tickers)
        news_items = self._fetch_news(tickers)
        emitted = 0
        for item in news_items:
            try:
                # A single story can be tagged to several holdings.  Completion is
                # tracked per story and ticker so one success cannot hide another
                # ticker's transient model failure.
                self.store.is_new(item)
                story_tickers = [ticker for ticker in item.tagged_tickers if ticker in tickers]
                candidates = [
                    ticker
                    for ticker in story_tickers
                    if not self.store.has_analysis(item.id, ticker)
                ]
                if not candidates:
                    self.store.advance_checkpoint(item)
                    continue
                completed_tickers: list[str] = []
                # Each ticker uses independent model calls, so analyze them in
                # parallel and combine this batch into one output panel.
                with ThreadPoolExecutor(max_workers=min(3, len(candidates))) as executor:
                    futures = {
                        executor.submit(self._analyze_ticker, item, ticker, meta.get(ticker, {})): ticker
                        for ticker in candidates
                    }
                    for future in as_completed(futures):
                        ticker = futures[future]
                        try:
                            analysis = future.result()
                            self.store.save_analysis(analysis)
                            completed_tickers.append(ticker)
                        except Exception as exc:
                            print(
                                f"[analysis] pending retry ticker={ticker} news={item.id}: {exc}",
                                flush=True,
                            )

                # A temporary failure must not delay the other stocks' analysis.
                if completed_tickers:
                    analyses = self.store.get_analyses(item.id, candidates)
                    visible_analyses = [analysis for analysis in analyses if analysis.sentiment != "neutral"]
                    if visible_analyses:
                        self.reporter.emit_news_analyses(item, visible_analyses)
                        emitted += 1
                if len(completed_tickers) == len(candidates):
                    self.store.advance_checkpoint(item)
            except Exception as exc:
                print(f"[pipeline] failed news={item.id}: {exc}", flush=True)
        self.store.cleanup_if_due(
            retention_hours=self.settings.news_retention_hours,
            interval_hours=self.settings.db_cleanup_interval_hours,
            neutral_retention_hours=self.settings.neutral_news_retention_hours,
        )
        return emitted

    def _analyze_ticker(self, item: NewsItem, ticker: str, meta: dict) -> ImpactAnalysis:
        analysis = self.news_agent.analyze(item, ticker, meta, "", None)
        if analysis.sentiment == "neutral":
            # Persist neutral items for deduplication, but do not push them.
            return analysis
        technical_result = self.technical_agent.analyze_news(item, analysis)
        analysis.technical_analysis = technical_result["snapshot"]
        analysis.combined_conclusion_zh = technical_result["combined_conclusion"]
        return analysis

    def _fetch_news(self, tickers: list[str]) -> list[NewsItem]:
        items = self.tickertick_news.fetch(tickers)
        recent_items = [item for item in items if item.headline and self._is_recent(item)]
        # Chronological processing makes the persisted restart checkpoint monotonic.
        recent_items.sort(key=lambda item: item.published_at)
        return recent_items

    def close(self) -> None:
        self.store.close()

    def _is_recent(self, item: NewsItem) -> bool:
        # RSS feeds may include older entries on every poll; ignore anything outside the freshness window.
        published_at = item.published_at
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - published_at.astimezone(timezone.utc)
        return timedelta(0) <= age <= timedelta(hours=self.settings.news_max_age_hours)
