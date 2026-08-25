from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

try:
    import requests
except Exception:
    requests = None

from .news_sources.finnhub_client import FinnhubClient


class EarningsCalendar:
    """Fetch previous and next earnings dates for close-report context."""

    def __init__(self, finnhub_api_key: str = "", min_request_interval: float = 1.1, timeout_seconds: int = 15):
        self.finnhub = FinnhubClient(
            finnhub_api_key,
            timeout=timeout_seconds,
            min_request_interval=min_request_interval,
        )
        self.timeout_seconds = timeout_seconds

    def dates_for(self, symbol: str) -> dict[str, Any]:
        ticker = symbol.strip().upper()
        result = {
            "ticker": ticker,
            "previous_earnings_date": None,
            "next_earnings_date": None,
            "source": None,
            "status": "unavailable",
        }
        if not ticker:
            return result

        if self.finnhub.enabled:
            try:
                finnhub_result = self._from_finnhub(ticker)
                if finnhub_result["previous_earnings_date"] or finnhub_result["next_earnings_date"]:
                    return finnhub_result
            except Exception as exc:
                result["status"] = f"finnhub_error: {type(exc).__name__}"

        try:
            yahoo_next = self._next_from_yahoo(ticker)
            if yahoo_next:
                result.update(
                    {
                        "next_earnings_date": yahoo_next,
                        "source": "yahoo_quote_summary",
                        "status": "partial_next_only",
                    }
                )
        except Exception as exc:
            if result["status"] == "unavailable":
                result["status"] = f"yahoo_error: {type(exc).__name__}"
        return result

    def _from_finnhub(self, symbol: str) -> dict[str, Any]:
        today = date.today()
        start = today - timedelta(days=370)
        end = today + timedelta(days=370)
        rows = self.finnhub.get_earnings_calendar(symbol, start, end)
        dated_rows = sorted(
            (row for row in rows if _parse_date(row.get("date"))),
            key=lambda row: _parse_date(row.get("date")) or date.min,
        )
        previous = None
        next_date = None
        for row in dated_rows:
            row_date = _parse_date(row.get("date"))
            if not row_date:
                continue
            if row_date <= today:
                previous = row_date
            elif next_date is None:
                next_date = row_date
        return {
            "ticker": symbol,
            "previous_earnings_date": previous.isoformat() if previous else None,
            "next_earnings_date": next_date.isoformat() if next_date else None,
            "source": "finnhub_earnings_calendar",
            "status": "ok" if previous or next_date else "not_found",
        }

    def _next_from_yahoo(self, symbol: str) -> str | None:
        if not requests:
            return None
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        response = requests.get(
            url,
            params={"modules": "calendarEvents"},
            timeout=self.timeout_seconds,
            headers={"User-Agent": "finance-analysis-agent/1.0"},
        )
        response.raise_for_status()
        results = response.json().get("quoteSummary", {}).get("result") or []
        calendar = (results[0] if results else {}).get("calendarEvents", {})
        earnings = calendar.get("earnings", {})
        candidates = earnings.get("earningsDate") or []
        dates = [_parse_timestamp(row.get("raw")) for row in candidates if isinstance(row, dict)]
        dates = sorted(row for row in dates if row)
        return dates[0].isoformat() if dates else None


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _parse_timestamp(value: Any) -> date | None:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).date()
    except (TypeError, ValueError, OSError):
        return None
