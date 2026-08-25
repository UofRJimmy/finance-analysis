from __future__ import annotations

import pandas as pd

from .candlestick_patterns import analyze_candlestick_patterns
from .market_data import YahooMarketData
from .technical_analysis import _indicator_snapshot, _safe_float, _with_indicators


class CloseRiskAnalyzer:
    """仅使用日线和周线，为盘后报告生成可复现的技术风险数据。"""

    def __init__(self, market_data: YahooMarketData | None = None):
        self.market_data = market_data or YahooMarketData()

    def analyze(self, ticker: str) -> dict:
        try:
            daily = self.market_data.get_history(ticker, period="2y", interval="1d")
            weekly = self.market_data.get_history(ticker, period="5y", interval="1wk")
        except Exception as exc:
            return {"available": False, "reason": str(exc)}
        if len(daily) < 50 or len(weekly) < 21:
            return {"available": False, "reason": "日线或周线数量不足"}

        daily_data = _with_indicators(daily)
        weekly_data = _with_indicators(weekly)
        daily_snapshot = _indicator_snapshot(daily, "日线")
        weekly_snapshot = _indicator_snapshot(weekly, "周线")
        levels = _price_zones(daily_data, weekly_data)
        divergence = _confirmed_divergence(daily_data)
        daily_candlestick = analyze_candlestick_patterns(daily_data, "日线")
        weekly_candlestick = analyze_candlestick_patterns(weekly_data, "周线")
        technical_risk, components = _technical_risk_score(daily_data, daily_snapshot, weekly_snapshot, levels, divergence)
        return {
            "available": True,
            "close": _safe_float(daily_data.iloc[-1]["close"]),
            "bar_time_utc": daily_data.iloc[-1]["timestamp"].isoformat(),
            "support_zone": levels["support_zone"],
            "resistance_zone": levels["resistance_zone"],
            "distance_to_support_atr": levels["distance_to_support_atr"],
            "distance_to_resistance_atr": levels["distance_to_resistance_atr"],
            "divergence": divergence,
            "daily": daily_snapshot,
            "weekly": weekly_snapshot,
            "candlestick_patterns": {
                "daily": daily_candlestick,
                "weekly": weekly_candlestick,
            },
            "technical_risk_score": technical_risk,
            "technical_risk_components": components,
        }


def _price_zones(daily: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    latest = daily.iloc[-1]
    close = float(latest["close"])
    atr = float(latest["atr_14"]) if pd.notna(latest["atr_14"]) and latest["atr_14"] > 0 else close * 0.02
    daily_window = daily.tail(120)
    weekly_window = weekly.tail(104)
    supports = _swing_values(daily_window, "low", 3, "low") + _swing_values(weekly_window, "low", 2, "low")
    resistances = _swing_values(daily_window, "high", 3, "high") + _swing_values(weekly_window, "high", 2, "high")
    support_center = _nearest_cluster(supports, close, atr * 0.5, below=True)
    resistance_center = _nearest_cluster(resistances, close, atr * 0.5, below=False)
    if support_center is None:
        support_center = float(daily_window["low"].tail(20).min())
    if resistance_center is None:
        resistance_center = float(daily_window["high"].tail(20).max())
    zone_half_width = atr * 0.25
    return {
        "support_zone": [round(support_center - zone_half_width, 4), round(support_center + zone_half_width, 4)],
        "resistance_zone": [round(resistance_center - zone_half_width, 4), round(resistance_center + zone_half_width, 4)],
        "distance_to_support_atr": round((close - support_center) / atr, 3),
        "distance_to_resistance_atr": round((resistance_center - close) / atr, 3),
    }


def _swing_values(frame: pd.DataFrame, column: str, radius: int, kind: str) -> list[float]:
    values = frame[column].astype(float).reset_index(drop=True)
    result = []
    for index in range(radius, len(values) - radius):
        value = values.iloc[index]
        neighborhood = values.iloc[index - radius : index + radius + 1]
        if (kind == "high" and value == neighborhood.max()) or (kind == "low" and value == neighborhood.min()):
            result.append(float(value))
    return result


def _nearest_cluster(values: list[float], close: float, tolerance: float, below: bool) -> float | None:
    candidates = sorted(value for value in values if (value < close if below else value > close))
    if not candidates:
        return None
    clusters: list[list[float]] = []
    for value in candidates:
        if clusters and abs(value - sum(clusters[-1]) / len(clusters[-1])) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    centers = [sum(cluster) / len(cluster) for cluster in clusters]
    return max(centers) if below else min(centers)


def _confirmed_divergence(data: pd.DataFrame) -> dict:
    window = data.tail(100).reset_index(drop=True)
    highs = _swing_indices(window["close"], radius=3, kind="high")
    lows = _swing_indices(window["close"], radius=3, kind="low")
    signals = []
    if len(highs) >= 2:
        first, second = highs[-2:]
        price_higher = window.loc[second, "close"] > window.loc[first, "close"] * 1.003
        rsi_lower = window.loc[second, "rsi_14"] < window.loc[first, "rsi_14"]
        macd_lower = window.loc[second, "macd"] < window.loc[first, "macd"]
        if price_higher and (rsi_lower or macd_lower):
            signals.append("日线顶背离")
    if len(lows) >= 2:
        first, second = lows[-2:]
        price_lower = window.loc[second, "close"] < window.loc[first, "close"] * 0.997
        rsi_higher = window.loc[second, "rsi_14"] > window.loc[first, "rsi_14"]
        macd_higher = window.loc[second, "macd"] > window.loc[first, "macd"]
        if price_lower and (rsi_higher or macd_higher):
            signals.append("日线底背离")
    return {"signals": signals, "state": "、".join(signals) if signals else "未发现已确认日线背离"}


def _swing_indices(series: pd.Series, radius: int, kind: str) -> list[int]:
    values = series.astype(float).reset_index(drop=True)
    indices = []
    for index in range(radius, len(values) - radius):
        neighborhood = values.iloc[index - radius : index + radius + 1]
        if (kind == "high" and values.iloc[index] == neighborhood.max()) or (
            kind == "low" and values.iloc[index] == neighborhood.min()
        ):
            indices.append(index)
    return indices


def _technical_risk_score(daily_data, daily: dict, weekly: dict, levels: dict, divergence: dict) -> tuple[int, dict]:
    structure = 0
    resistance_distance = levels["distance_to_resistance_atr"]
    if resistance_distance <= 0.5:
        structure = 20
    elif resistance_distance <= 1.0:
        structure = 14
    elif resistance_distance <= 2.0:
        structure = 7
    prior_low = daily_data["low"].shift(1).rolling(20).min().iloc[-1]
    if pd.notna(prior_low) and daily_data.iloc[-1]["close"] < prior_low:
        structure = 25

    trend = 0
    if "空头" in str(weekly.get("trend")):
        trend += 12
    if "空头" in str(daily.get("trend")):
        trend += 8
    trend = min(20, trend)

    divergence_score = 15 if "日线顶背离" in divergence["signals"] else -5 if "日线底背离" in divergence["signals"] else 0
    momentum = 0
    if daily.get("rsi_state") == "超买":
        momentum += 7
    if "减弱" in str((daily.get("macd") or {}).get("momentum")):
        momentum += 6
    if (daily.get("kdj") or {}).get("state") == "超买":
        momentum += 4
    momentum = min(15, momentum)

    volatility = 0
    atr_ratio = daily.get("atr_expansion_ratio")
    if atr_ratio is not None and atr_ratio >= 1.5:
        volatility += 7
    elif atr_ratio is not None and atr_ratio >= 1.2:
        volatility += 4
    if (daily.get("bollinger") or {}).get("abnormal_expansion"):
        volatility += 3
    volatility = min(10, volatility)

    latest, previous = daily_data.iloc[-1], daily_data.iloc[-2]
    volume = 0
    volume_ratio = latest.get("volume_ratio")
    if pd.notna(volume_ratio) and volume_ratio >= 1.5 and latest["close"] < previous["close"]:
        volume = 10
    elif pd.notna(volume_ratio) and volume_ratio < 0.8 and latest["close"] > previous["close"]:
        volume = 5

    score = max(0, min(100, round(structure + trend + divergence_score + momentum + volatility + volume)))
    return score, {
        "price_structure": structure,
        "daily_weekly_trend": trend,
        "divergence": divergence_score,
        "momentum": momentum,
        "volatility": volatility,
        "volume": volume,
    }
