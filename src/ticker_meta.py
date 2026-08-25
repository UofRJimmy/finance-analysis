from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .news_sources.finnhub_client import FinnhubClient


class TickerMetaCache:
    def __init__(self, path: Path, finnhub: FinnhubClient):
        self.path = path
        self.finnhub = finnhub
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def warm(self, tickers: list[str]) -> dict[str, dict[str, Any]]:
        today = date.today().isoformat()
        changed = False
        for ticker in tickers:
            row = self._data.get(ticker)
            if row and row.get("updated_at") == today:
                continue
            profile: dict[str, Any] = {}
            if self.finnhub.enabled:
                try:
                    profile = self.finnhub.get_profile(ticker)
                except Exception as exc:
                    print(f"[meta] profile failed {ticker}: {exc}", flush=True)
            self._data[ticker] = {
                "ticker": ticker,
                "company_name": profile.get("name") or profile.get("ticker") or ticker,
                "sector": profile.get("finnhubIndustry") or profile.get("sector"),
                "updated_at": today,
            }
            changed = True
        if changed:
            self._save()
        return self._data

    def get(self, ticker: str) -> dict[str, Any]:
        return self._data.get(ticker, {"ticker": ticker, "company_name": ticker, "sector": None})

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
