from __future__ import annotations

from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class YahooMarketData:
    """通过 Yahoo Finance 的只读 Chart JSON 接口获取历史 K 线。"""

    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(self, timeout_seconds: int = 15):
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({"User-Agent": "finance-analysis-agent/1.0"})

    def get_history(self, ticker: str, *, period: str, interval: str) -> pd.DataFrame:
        # ticker 经过 URL 编码，所有请求都发往固定 HTTPS 域名，不会携带任何 API Key。
        symbol = quote(ticker.strip().upper(), safe=".-^=")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        try:
            response = self.session.get(
                url,
                params={"range": period, "interval": interval, "events": "div,splits"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            chart = response.json().get("chart", {})
        except Exception as exc:
            raise RuntimeError(f"Yahoo Finance 获取 {ticker} {interval} K线失败: {exc}") from exc

        if chart.get("error"):
            description = chart["error"].get("description", "未知错误")
            raise RuntimeError(f"Yahoo Finance 获取 {ticker} {interval} K线失败: {description}")
        results = chart.get("result") or []
        if not results:
            raise RuntimeError(f"Yahoo Finance 暂无 {ticker} {interval} K线数据")

        result = results[0]
        timestamps = result.get("timestamp") or []
        quotes = (result.get("indicators", {}).get("quote") or [{}])[0]
        frame = pd.DataFrame({"timestamp": timestamps, **quotes})
        missing = self.REQUIRED_COLUMNS.difference(frame.columns)
        if not timestamps or missing:
            raise RuntimeError(f"Yahoo Finance 返回的 {ticker} {interval} K线字段不完整")

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        frame = frame[columns].dropna(subset=["timestamp", "close"])
        return frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
