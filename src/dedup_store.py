from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import ImpactAnalysis, NewsItem


class DedupStore:
    def __init__(self, path: Path, max_news_items: int = 500):
        self.path = path
        self.max_news_items = max_news_items
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=20)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=20000")
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS seen_news (
                id TEXT PRIMARY KEY,
                first_seen_at TEXT NOT NULL,
                headline TEXT NOT NULL,
                news_json TEXT
            );
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                news_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                created_at TEXT NOT NULL,
                analysis_json TEXT NOT NULL,
                UNIQUE(news_id, ticker)
            );
            CREATE TABLE IF NOT EXISTS agent_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_seen_news_first_seen ON seen_news(first_seen_at);
            CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at);
            """
        )
        self.conn.commit()

    def is_new(self, item: NewsItem) -> bool:
        row = self.conn.execute("SELECT 1 FROM seen_news WHERE id = ?", (item.id,)).fetchone()
        if row:
            return False
        self.conn.execute(
            "INSERT INTO seen_news(id, first_seen_at, headline, news_json) VALUES (?, ?, ?, ?)",
            (
                item.id,
                datetime.now(timezone.utc).isoformat(),
                item.headline,
                json.dumps(item.to_dict(), ensure_ascii=False),
            ),
        )
        self._trim_to_limit(self.max_news_items)
        self.conn.commit()
        return True

    def save_analysis(self, analysis: ImpactAnalysis) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO analyses(news_id, ticker, created_at, analysis_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                analysis.news_id,
                analysis.ticker,
                analysis.created_at.isoformat(),
                json.dumps(analysis.to_dict(), ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def has_analysis(self, news_id: str, ticker: str) -> bool:
        """Return whether this specific ticker has completed this news item."""
        row = self.conn.execute(
            "SELECT 1 FROM analyses WHERE news_id = ? AND ticker = ?",
            (news_id, ticker),
        ).fetchone()
        return row is not None

    def get_analyses(self, news_id: str, tickers: list[str]) -> list[ImpactAnalysis]:
        """Load completed analyses in the same order as the news ticker tags."""
        if not tickers:
            return []
        placeholders = ", ".join("?" for _ in tickers)
        rows = self.conn.execute(
            f"SELECT ticker, analysis_json FROM analyses WHERE news_id = ? AND ticker IN ({placeholders})",
            (news_id, *tickers),
        ).fetchall()
        stored = {row["ticker"]: json.loads(row["analysis_json"]) for row in rows}
        return [_analysis_from_dict(stored[ticker]) for ticker in tickers if ticker in stored]

    def discard_unanalyzed_news(self, news_id: str) -> None:
        """Allow a transient model failure to be retried on the next source poll."""
        existing = self.conn.execute("SELECT 1 FROM analyses WHERE news_id = ? LIMIT 1", (news_id,)).fetchone()
        if existing:
            return
        self.conn.execute("DELETE FROM seen_news WHERE id = ?", (news_id,))
        self.conn.commit()

    def recent_records(self, since: datetime) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT seen_news.news_json, analyses.analysis_json
            FROM analyses
            JOIN seen_news ON seen_news.id = analyses.news_id
            WHERE analyses.created_at >= ?
            ORDER BY analyses.created_at DESC
            """,
            (since.isoformat(),),
        ).fetchall()
        return [
            {"news": json.loads(row["news_json"]), "analysis": json.loads(row["analysis_json"])}
            for row in rows
            if row["news_json"] and row["analysis_json"]
        ]

    def get_checkpoint(self) -> tuple[datetime | None, set[str]]:
        timestamp_text = self._get_state("last_processed_published_at")
        ids_text = self._get_state("last_processed_ids")
        if timestamp_text:
            return _parse_datetime(timestamp_text), set(json.loads(ids_text or "[]"))
        return self._derive_checkpoint_from_history()

    def is_after_checkpoint(self, item: NewsItem, checkpoint: datetime | None, checkpoint_ids: set[str]) -> bool:
        if checkpoint is None:
            return True
        published = _as_utc(item.published_at)
        return published > checkpoint or (published == checkpoint and item.id not in checkpoint_ids)

    def advance_checkpoint(self, item: NewsItem) -> None:
        published = _as_utc(item.published_at)
        current, current_ids = self.get_checkpoint()
        if current is not None and published < current:
            return
        if current is None or published > current:
            current_ids = {item.id}
        else:
            current_ids.add(item.id)
        self._set_state("last_processed_published_at", published.isoformat())
        self._set_state("last_processed_ids", json.dumps(sorted(current_ids)))
        self.conn.commit()

    def cleanup_if_due(
        self,
        retention_hours: int = 24,
        interval_hours: int = 24,
        neutral_retention_hours: int = 24 * 7,
    ) -> int:
        now = datetime.now(timezone.utc)
        last_cleanup = _parse_datetime(self._get_state("last_cleanup_at"))
        if last_cleanup and now - last_cleanup < timedelta(hours=interval_hours):
            return 0
        neutral_cutoff = now - timedelta(hours=neutral_retention_hours)
        neutral_rows = self.conn.execute(
            "SELECT id, news_id, analysis_json FROM analyses WHERE created_at < ?",
            (neutral_cutoff.isoformat(),),
        ).fetchall()
        neutral_ids: list[int] = []
        neutral_news_ids: set[str] = set()
        for row in neutral_rows:
            try:
                is_neutral = json.loads(row["analysis_json"]).get("sentiment") == "neutral"
            except (TypeError, json.JSONDecodeError):
                is_neutral = False
            if is_neutral:
                neutral_ids.append(row["id"])
                neutral_news_ids.add(row["news_id"])
        if neutral_ids:
            placeholders = ",".join("?" for _ in neutral_ids)
            self.conn.execute(f"DELETE FROM analyses WHERE id IN ({placeholders})", neutral_ids)

        # 仅含过期中性分析的新闻正文一起删除；若同一新闻另有非中性结论则保留。
        orphaned_neutral_news = [
            news_id
            for news_id in neutral_news_ids
            if self.conn.execute("SELECT 1 FROM analyses WHERE news_id = ? LIMIT 1", (news_id,)).fetchone() is None
        ]
        self._delete_news(orphaned_neutral_news)

        cutoff = now - timedelta(hours=retention_hours)
        old_ids = [
            row["id"]
            for row in self.conn.execute("SELECT id FROM seen_news WHERE first_seen_at < ?", (cutoff.isoformat(),)).fetchall()
        ]
        self._delete_news(old_ids)
        self._set_state("last_cleanup_at", now.isoformat())
        self.conn.commit()
        # Reclaim WAL pages without running a heavy VACUUM on every cleanup.
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return len(neutral_ids) + len(old_ids)

    def _trim_to_limit(self, max_items: int) -> None:
        if max_items <= 0:
            return
        count = self.conn.execute("SELECT COUNT(*) AS n FROM seen_news").fetchone()["n"]
        excess = count - max_items
        if excess <= 0:
            return
        oldest = [
            row["id"]
            for row in self.conn.execute(
                "SELECT id FROM seen_news ORDER BY first_seen_at ASC LIMIT ?",
                (excess,),
            ).fetchall()
        ]
        self._delete_news(oldest)

    def _delete_news(self, news_ids: list[str]) -> None:
        if not news_ids:
            return
        placeholders = ",".join("?" for _ in news_ids)
        self.conn.execute(f"DELETE FROM analyses WHERE news_id IN ({placeholders})", news_ids)
        self.conn.execute(f"DELETE FROM seen_news WHERE id IN ({placeholders})", news_ids)

    def _derive_checkpoint_from_history(self) -> tuple[datetime | None, set[str]]:
        rows = self.conn.execute("SELECT id, news_json FROM seen_news").fetchall()
        latest: datetime | None = None
        latest_ids: set[str] = set()
        for row in rows:
            try:
                published = _parse_datetime(json.loads(row["news_json"])["published_at"])
            except Exception:
                continue
            if latest is None or published > latest:
                latest, latest_ids = published, {row["id"]}
            elif published == latest:
                latest_ids.add(row["id"])
        if latest is not None:
            self._set_state("last_processed_published_at", latest.isoformat())
            self._set_state("last_processed_ids", json.dumps(sorted(latest_ids)))
            self.conn.commit()
        return latest, latest_ids

    def _get_state(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM agent_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO agent_state(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def close(self) -> None:
        self.conn.close()


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return _as_utc(parsed)


def _analysis_from_dict(data: dict) -> ImpactAnalysis:
    """Rebuild a stored analysis for a later combined multi-ticker alert."""
    data = dict(data)
    data["created_at"] = _parse_datetime(data.get("created_at")) or datetime.now(timezone.utc)
    return ImpactAnalysis(**data)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
