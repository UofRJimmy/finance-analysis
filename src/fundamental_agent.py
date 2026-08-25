from __future__ import annotations

from typing import Any

from .agent_scoring import score_fundamental_agent
from .config import Settings
from .edgar_financials import EdgarFinancialAnalyzer
from .market_data import YahooMarketData
from .news_sources.finnhub_client import FinnhubClient


class FundamentalAgent:
    """Agent responsible for company fundamentals and financial-report analysis."""

    def __init__(self, settings: Settings, finnhub: FinnhubClient | None = None):
        self.settings = settings
        self.edgar = EdgarFinancialAnalyzer(settings)
        self.finnhub = finnhub or FinnhubClient(settings.finnhub_api_key)
        self.market_data = YahooMarketData()

    def analyze_report(self, ticker: str, question: str) -> str:
        report = self.edgar.analyze(ticker, question)
        try:
            payload = self.basic_financials(ticker)
            score = payload.get("agent_score") or {}
        except Exception as exc:
            score = {
                "total_score": 0,
                "direction": "数据不足",
                "components": [{"name": "基本面数据获取失败", "score": 0, "reason": str(exc)}],
            }
        return f"{report}\n\n{_format_fundamental_score(score)}"

    def basic_financials(self, ticker: str, prefer_edgar: bool = False) -> dict[str, Any]:
        if prefer_edgar:
            try:
                payload = self.edgar.fundamentals(ticker)
            except Exception as edgar_exc:
                try:
                    payload = self._finnhub_basic_financials(ticker)
                except Exception as finnhub_exc:
                    raise RuntimeError(
                        f"EDGAR fundamentals failed: {edgar_exc}; Finnhub fallback failed: {finnhub_exc}"
                    ) from None
                payload.setdefault("source", {})["fallback_reason"] = str(edgar_exc)
        else:
            try:
                payload = self._finnhub_basic_financials(ticker)
            except Exception as finnhub_exc:
                try:
                    payload = self.edgar.fundamentals(ticker)
                except Exception as edgar_exc:
                    raise RuntimeError(
                        f"Finnhub basic financials failed: {finnhub_exc}; EDGAR fundamentals failed: {edgar_exc}"
                    ) from None
                payload.setdefault("source", {})["fallback_reason"] = str(finnhub_exc)
        payload["agent_score"] = score_fundamental_agent(payload)
        _append_source_component(payload["agent_score"], payload.get("source") or {})
        return payload

    def _finnhub_basic_financials(self, ticker: str) -> dict[str, Any]:
        try:
            payload = self.finnhub.get_basic_financials(ticker)
            if not _has_fundamental_metrics(payload):
                raise RuntimeError("Finnhub basic financials returned empty metrics")
            payload["source"] = {"provider": "Finnhub stock metric", "ticker": ticker}
            return payload
        except Exception as exc:
            raise RuntimeError(str(exc)) from None

    def get_vix(self) -> dict[str, Any]:
        frame = self.market_data.get_history("^VIX", period="5d", interval="1d")
        latest = frame.iloc[-1]
        previous = frame.iloc[-2] if len(frame) >= 2 else latest
        last = float(latest["close"])
        previous_close = float(previous["close"])
        return {
            "symbol": "^VIX",
            "name": "CBOE Volatility Index",
            "last": round(last, 4),
            "previous_close": round(previous_close, 4),
            "change_pct": round(last / previous_close - 1, 6) if previous_close else None,
            "bar_time_utc": latest["timestamp"].isoformat(),
            "source": "Yahoo Finance chart",
        }


def _format_fundamental_score(score: dict[str, Any]) -> str:
    lines = [f"## 基本面Agent评分", f"{score.get('total_score', 0)}/100（{score.get('direction', '未知')}）"]
    components = score.get("components") or []
    if components:
        lines.append("分项：")
        lines.extend(
            f"- {item.get('name')} {item.get('score'):+g}：{item.get('reason', '')}"
            for item in components
            if item.get("score") is not None
        )
    return "\n".join(lines)


def _has_fundamental_metrics(payload: dict[str, Any]) -> bool:
    metric = payload.get("metric") or {}
    required = [
        "freeCashFlowPerShareTTM",
        "cashFlowPerShareTTM",
        "netIncomePerShareTTM",
        "currentRatioAnnual",
        "totalDebt/totalEquityAnnual",
    ]
    return any(metric.get(key) not in (None, "", "N/A") for key in required)


def _append_source_component(score: dict[str, Any], source: dict[str, Any]) -> None:
    provider = source.get("fundamental_source") or source.get("provider") or "unknown"
    year = source.get("latest_fiscal_year")
    reason = f"基本面评分来源：{provider}" + (f"，最新财年 {year}。" if year else "。")
    score.setdefault("components", []).append({"name": "基本面数据来源", "score": 0, "reason": reason})
