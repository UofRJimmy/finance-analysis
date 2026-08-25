from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .aggressive_decision_agent import AggressiveDecisionAgent
from .candlestick_patterns import analyze_candlestick_patterns
from .models import ImpactAnalysis, NewsItem
from .market_data import YahooMarketData


class TechnicalAnalyzer:
    def __init__(self, market_data: YahooMarketData, deepseek_api_key: str, model: str, cache_seconds: int = 300):
        self.market_data = market_data
        self.aggressive_agent = AggressiveDecisionAgent(deepseek_api_key)
        self.model = model
        self.cache_seconds = cache_seconds
        self._cache: dict[str, tuple[float, pd.DataFrame, dict]] = {}

    def get_snapshot(self, ticker: str) -> dict:
        _, _, snapshot = self._market_data(ticker)
        return snapshot

    def analyze_news(self, news: NewsItem, impact: ImpactAnalysis) -> dict:
        combined = self._combine(news, impact)
        return {"snapshot": {}, "combined_conclusion": combined}

    def _market_data(self, ticker: str) -> tuple[float, pd.DataFrame, dict]:
        cached = self._cache.get(ticker)
        if cached and time.monotonic() - cached[0] < self.cache_seconds:
            return cached
        # 两年日线足以计算 200 日均线；新闻分析不再抓取分时线，减少噪音和请求量。
        daily = self._fetch_frame(ticker, period="2y", interval="1d")
        snapshot = {
            "available": not daily.empty,
            "daily": _indicator_snapshot(daily, "日线"),
        }
        result = (time.monotonic(), daily, snapshot)
        self._cache[ticker] = result
        return result

    def _fetch_frame(self, ticker: str, *, period: str, interval: str) -> pd.DataFrame:
        try:
            return self.market_data.get_history(ticker, period=period, interval=interval)
        except Exception as exc:
            frame = pd.DataFrame()
            frame.attrs["error"] = str(exc)
            return frame

    def _combine(self, news: NewsItem, impact: ImpactAnalysis) -> str:
        context = {
            "news": news.to_dict(),
            "message_analysis": impact.to_dict(),
            "task": "分析这条新闻对指定股票的影响",
        }
        result = self.aggressive_agent.analyze_news_impact(context)
        aggressive_impact = str(result.get("impact") or "").strip()
        if not aggressive_impact:
            raise RuntimeError("激进Agent返回了空的新闻影响分析")
        return _format_news_structure(
            event_summary=impact.reasoning_zh,
            aggressive_impact=aggressive_impact,
        )


def _indicator_snapshot(frame: pd.DataFrame, timeframe: str) -> dict:
    if frame.empty or len(frame) < 21:
        return {"available": False, "timeframe": timeframe, "reason": frame.attrs.get("error", "K线数量不足")}
    data = _with_indicators(frame)
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    prior_high = data["high"].shift(1).rolling(20).max().iloc[-1]
    volume_ratio = _safe_float(latest["volume_ratio"])
    price_change = _safe_float(latest["close"] / previous["close"] - 1)
    breakout = bool(latest["close"] > prior_high) if pd.notna(prior_high) else False
    effective_breakout = breakout and volume_ratio is not None and volume_ratio >= 1.5
    weak_rise = price_change is not None and price_change > 0 and volume_ratio is not None and volume_ratio < 0.8

    return {
        "available": True,
        "timeframe": timeframe,
        "bar_time_utc": latest["timestamp"].isoformat(),
        "close": _safe_float(latest["close"]),
        "price_change_last_bar": price_change,
        "moving_averages": {period: _safe_float(latest[f"sma_{period}"]) for period in [7, 21, 50, 200]},
        "trend": _trend_label(latest),
        "cross_signal": _cross_signal(data),
        "macd": {
            "value": _safe_float(latest["macd"]),
            "signal": _safe_float(latest["macd_signal"]),
            "histogram": _safe_float(latest["macd_hist"]),
            "momentum": _macd_direction(latest, previous),
        },
        "rsi_14": _safe_float(latest["rsi_14"]),
        "rsi_state": _rsi_state(latest["rsi_14"]),
        "bollinger": {
            "upper": _safe_float(latest["bb_upper"]),
            "lower": _safe_float(latest["bb_lower"]),
            "bandwidth": _safe_float(latest["bb_width"]),
            "abnormal_expansion": bool(latest["bb_expansion"]),
        },
        "atr_14": _safe_float(latest["atr_14"]),
        "atr_expansion_ratio": _safe_float(latest["atr_ratio"]),
        "volume_ratio_to_20_bar_average": volume_ratio,
        "kdj": {
            "k": _safe_float(latest["kdj_k"]),
            "d": _safe_float(latest["kdj_d"]),
            "j": _safe_float(latest["kdj_j"]),
            "state": _kdj_state(latest["kdj_k"], latest["kdj_d"], latest["kdj_j"]),
        },
        "breakout": {
            "above_recent_20_bar_high": breakout,
            "effective_breakout_with_volume": effective_breakout,
            "weak_rise_on_low_volume": weak_rise,
        },
        "candlestick_patterns": analyze_candlestick_patterns(data, timeframe),
    }


def _with_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float)
    for period in [7, 21, 50, 200]:
        data[f"sma_{period}"] = close.rolling(period).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi_14"] = 100 - 100 / (1 + rs)

    middle = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    data["bb_upper"] = middle + 2 * std
    data["bb_lower"] = middle - 2 * std
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / middle.replace(0, np.nan)
    data["bb_expansion"] = data["bb_width"] > data["bb_width"].rolling(20).mean() * 1.5

    true_range = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    data["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    data["atr_ratio"] = data["atr_14"] / data["atr_14"].rolling(50).mean()
    data["volume_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)

    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
    data["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    data["kdj_d"] = data["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
    data["kdj_j"] = 3 * data["kdj_k"] - 2 * data["kdj_d"]
    return data


def _trend_label(row) -> str:
    values = [row.get(f"sma_{period}") for period in [7, 21, 50, 200]]
    if any(pd.isna(value) for value in values):
        return "长期均线数据不足"
    sma7, sma21, sma50, sma200 = values
    if sma7 > sma21 > sma50 > sma200:
        return "多头排列"
    if sma7 < sma21 < sma50 < sma200:
        return "空头排列"
    return "均线纠缠/趋势过渡"


def _cross_signal(data: pd.DataFrame) -> str:
    latest, previous = data.iloc[-1], data.iloc[-2]
    signals = []
    for short, long, name in [(7, 21, "短期"), (50, 200, "长期")]:
        current_short, current_long = latest[f"sma_{short}"], latest[f"sma_{long}"]
        prev_short, prev_long = previous[f"sma_{short}"], previous[f"sma_{long}"]
        if any(pd.isna(value) for value in [current_short, current_long, prev_short, prev_long]):
            continue
        if prev_short <= prev_long and current_short > current_long:
            signals.append(f"{name}金叉")
        elif prev_short >= prev_long and current_short < current_long:
            signals.append(f"{name}死叉")
    return "、".join(signals) or "无新金叉/死叉"


def _macd_direction(latest, previous) -> str:
    current, prior = latest["macd_hist"], previous["macd_hist"]
    if pd.isna(current) or pd.isna(prior):
        return "数据不足"
    if current > prior:
        return "动能转强" if current >= 0 else "空头动能减弱"
    return "动能转弱" if current <= 0 else "多头动能减弱"


def _rsi_state(value) -> str:
    if pd.isna(value):
        return "数据不足"
    if value > 70:
        return "超买"
    if value < 30:
        return "超卖"
    return "中性"


def _kdj_state(k, d, j) -> str:
    if any(pd.isna(value) for value in [k, d, j]):
        return "数据不足"
    if max(k, d, j) > 80:
        return "超买"
    if min(k, d, j) < 20:
        return "超卖"
    return "中性"


def _format_news_structure(*, event_summary: str, aggressive_impact: str) -> str:
    return "\n".join(
        [
            f"新闻Agent：{event_summary}",
            f"激进Agent：{aggressive_impact}",
        ]
    )


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value) or np.isinf(value):
        return None
    return round(float(value), 6)
