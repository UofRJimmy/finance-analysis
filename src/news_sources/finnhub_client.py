from __future__ import annotations

import hashlib
import threading
import time
from datetime import date
from typing import Any

from ..models import Quote


try:
    import requests
except Exception:
    requests = None


BASE_URL = "https://finnhub.io/api/v1"


class FinnhubClient:
    _rate_lock = threading.Lock()
    _last_request_by_key: dict[str, float] = {}

    def __init__(self, api_key: str, timeout: int = 15, min_request_interval: float = 1.1):
        self.api_key = api_key
        self.timeout = timeout
        self.min_request_interval = max(0.0, min_request_interval)
        self.session = requests.Session() if requests else None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        if not self.api_key:
            raise RuntimeError("FINNHUB_API_KEY is not configured")
        params = {**params, "token": self.api_key}
        url = f"{BASE_URL}{path}"
        for attempt in range(2):
            self._wait_for_rate_slot()
            try:
                if self.session:
                    response = self.session.get(url, params=params, timeout=self.timeout)
                    if response.status_code == 429:
                        if attempt == 0:
                            retry_after = min(float(response.headers.get("Retry-After", 2)), 15.0)
                            time.sleep(max(1.0, retry_after))
                            continue
                        raise RuntimeError(f"Finnhub {path} HTTP 429: 请求频率受限")
                    if response.status_code >= 500 and attempt == 0:
                        time.sleep(1)
                        continue
                    if response.status_code >= 400:
                        body = response.text[:300].replace(self.api_key, "***")
                        raise RuntimeError(f"Finnhub {path} HTTP {response.status_code}: {body}")
                    return response.json()
                from urllib.request import urlopen
                from urllib.parse import urlencode

                with urlopen(f"{url}?{urlencode(params)}", timeout=self.timeout) as response:
                    import json

                    return json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                # 明确的 4xx（除上面单独处理的 429）不会因重试而恢复。
                if " HTTP 4" in str(exc):
                    safe_error = str(exc).replace(self.api_key, "***")
                    raise RuntimeError(safe_error) from None
                if attempt == 0:
                    time.sleep(1)
                    continue
                safe_error = str(exc).replace(self.api_key, "***")
                raise RuntimeError(f"Finnhub {path} request failed: {safe_error}") from None
        return None

    def _wait_for_rate_slot(self) -> None:
        """同一 API Key 的多个客户端共享节流，避免免费额度瞬时超限。"""
        if not self.min_request_interval:
            return
        key_hash = hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()
        with self._rate_lock:
            now = time.monotonic()
            last = self._last_request_by_key.get(key_hash, 0.0)
            delay = self.min_request_interval - (now - last)
            if delay > 0:
                time.sleep(delay)
            self._last_request_by_key[key_hash] = time.monotonic()

    def get_quote(self, symbol: str) -> Quote:
        data = self._get("/quote", {"symbol": symbol}) or {}
        current = _num(data.get("c"))
        previous_close = _num(data.get("pc"))
        change_pct = None
        if current is not None and previous_close:
            change_pct = (current - previous_close) / previous_close * 100
        return Quote(symbol=symbol, current=current, previous_close=previous_close, change_pct=change_pct)

    def get_profile(self, symbol: str) -> dict[str, Any]:
        return self._get("/stock/profile2", {"symbol": symbol}) or {}

    def search_symbol(self, query: str) -> list[dict[str, Any]]:
        data = self._get("/search", {"q": query}) or {}
        return list(data.get("result") or [])

    def get_basic_financials(self, symbol: str) -> dict[str, Any]:
        return self._get("/stock/metric", {"symbol": symbol, "metric": "all"}) or {}

    def get_earnings_calendar(self, symbol: str, from_date: date, to_date: date) -> list[dict[str, Any]]:
        data = self._get(
            "/calendar/earnings",
            {"symbol": symbol, "from": from_date.isoformat(), "to": to_date.isoformat()},
        ) or {}
        return list(data.get("earningsCalendar") or [])

def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
