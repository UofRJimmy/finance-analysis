from __future__ import annotations

from .models import Quote
from .news_sources.finnhub_client import FinnhubClient


class PriceChecker:
    def __init__(self, finnhub: FinnhubClient):
        self.finnhub = finnhub

    def get_price_context(self, ticker: str) -> tuple[str, float | None]:
        if not self.finnhub.enabled:
            return "无 FINNHUB_API_KEY，无法获取盘前/盘后行情，仅供参考。", None
        try:
            quote = self.finnhub.get_quote(ticker)
        except Exception as exc:
            return f"行情获取失败: {exc}", None
        return _format_quote(quote), quote.change_pct


def _format_quote(quote: Quote) -> str:
    if quote.current is None:
        return "无可用 quote 数据。"
    pct = "N/A" if quote.change_pct is None else f"{quote.change_pct:.2f}%"
    pc = "N/A" if quote.previous_close is None else f"{quote.previous_close:.2f}"
    return f"current={quote.current:.2f}, previous_close={pc}, change_since_prev_close={pct}"
