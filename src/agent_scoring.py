from __future__ import annotations

from .models import ImpactAnalysis


BULLISH_PATTERN_SCORES = {
    "放量长阳突破": 80,
    "红三兵": 70,
    "三白兵": 70,
    "早晨之星": 70,
    "早晨十字星": 75,
    "看涨吞没": 60,
    "突破缺口": 55,
    "三外升": 55,
    "三内升": 45,
    "放量突破后缩量回踩": 45,
    "双底": 40,
    "W底": 40,
    "头肩底": 50,
    "低位长下影放量": 35,
    "锤子线": 15,
    "长下影线": 15,
}

BEARISH_PATTERN_SCORES = {
    "放量长阴破位": 80,
    "三只乌鸦": 70,
    "黑三兵": 70,
    "黄昏之星": 70,
    "黄昏十字星": 75,
    "看跌吞没": 60,
    "岛形反转": 65,
    "三外降": 55,
    "三内降": 45,
    "双顶": 40,
    "M顶": 40,
    "头肩顶": 50,
    "缩量突破": 35,
    "高位长上影放量": 50,
    "射击之星": 25,
    "长上影线": 15,
}


def score_news_agent(analysis: ImpactAnalysis) -> dict:
    direction = {"bullish": "偏多", "bearish": "偏空", "neutral": "中性"}.get(analysis.sentiment, "中性")
    if analysis.sentiment == "neutral":
        return _score(0, direction, [{"name": "新闻影响不明确", "score": 0, "reason": analysis.reasoning_zh}])

    magnitude_score = {"large": 45, "medium": 30, "small": 15}.get(analysis.magnitude, 15)
    confidence_score = {"high": 30, "medium": 20, "low": 10}.get(analysis.confidence, 10)
    horizon_score = {"intraday": 10, "short_term_days": 15, "medium_long_term": 20}.get(analysis.time_horizon, 10)
    priced_in_adjustment = {"not_priced_in": 10, "partially_priced_in": 0, "priced_in": -15, "unclear": -5}.get(analysis.priced_in, 0)
    components = [
        {"name": f"消息强度：{analysis.magnitude}", "score": magnitude_score, "reason": "衡量新闻本身冲击大小。"},
        {"name": f"模型置信度：{analysis.confidence}", "score": confidence_score, "reason": "衡量判断可靠度。"},
        {"name": f"影响周期：{analysis.time_horizon}", "score": horizon_score, "reason": "中期影响比纯日内噪音权重更高。"},
        {"name": f"Price in：{analysis.priced_in}", "score": priced_in_adjustment, "reason": "已反映会降低后续信号分。"},
    ]
    return _score(magnitude_score + confidence_score + horizon_score + priced_in_adjustment, direction, components)


def score_technical_agent(snapshot: dict) -> dict:
    if not snapshot.get("available"):
        return _score(0, "数据不足", [{"name": "技术数据不可用", "score": 0, "reason": snapshot.get("reason", "暂无原因")}])

    timeframe_scores = {}
    top_level_candles = snapshot.get("candlestick_patterns") or {}
    for timeframe in ["daily", "weekly"]:
        block = snapshot.get(timeframe) or {}
        candle = block.get("candlestick_patterns") or top_level_candles.get(timeframe) or {}
        timeframe_scores[timeframe] = _score_timeframe_patterns(timeframe, candle)

    daily = snapshot.get("daily") or {}
    if "多头" in str(daily.get("trend")):
        _add_to_timeframe_score(timeframe_scores["daily"], "日线趋势：多头排列", "bullish", 15, "趋势结构偏多。")
    if "空头" in str(daily.get("trend")):
        _add_to_timeframe_score(timeframe_scores["daily"], "日线趋势：空头排列", "bearish", 15, "趋势结构偏空。")
    divergence = snapshot.get("divergence") or {}
    signals = divergence.get("signals", [])
    if "日线顶背离" in signals:
        _add_to_timeframe_score(timeframe_scores["daily"], "日线顶背离", "bearish", 20, "价格创新高但动能没有同步。")
    if "日线底背离" in signals:
        _add_to_timeframe_score(timeframe_scores["daily"], "日线底背离", "bullish", 20, "价格创新低但动能没有同步走弱。")

    available = {
        key: value
        for key, value in timeframe_scores.items()
        if value.get("components") or key in snapshot or (top_level_candles.get(key) or {}).get("available")
    }
    weights = {"daily": 1.0, "weekly": 0.5}
    total_weight = sum(weights[key] for key in available) or 1
    total = round(sum(value["total_score"] * weights[key] for key, value in available.items()) / total_weight)
    directions = [value["direction"] for value in available.values()]
    direction = _aggregate_direction(directions)
    components = []
    for key in ["daily", "weekly"]:
        components.extend(timeframe_scores[key].get("components", [])[:3])
    result = _score(total, direction, components)
    result["timeframe_scores"] = {
        _timeframe_name(key): _public_score(value) for key, value in available.items()
    }
    return result


def _score_timeframe_patterns(timeframe: str, candle: dict) -> dict:
    components = []
    bullish_total = 0
    bearish_total = 0
    for item in candle.get("patterns", [])[:5]:
        name = str(item.get("name") or "")
        bias = str(item.get("bias") or "")
        score = _pattern_score(name, bias)
        if score <= 0:
            continue
        label = _pattern_label(name, bias)
        signed_score = -score if "bearish" in bias else score
        components.append(
            {
                "name": f"{_timeframe_name(timeframe)}：{label}",
                "score": signed_score,
                "reason": item.get("reason", "蜡烛图形态信号。"),
            }
        )
        if "bearish" in bias:
            bearish_total += score
        elif "bullish" in bias:
            bullish_total += score
    result = _directional_score(bullish_total, bearish_total, components)
    result["_bullish_total"] = bullish_total
    result["_bearish_total"] = bearish_total
    return result


def _add_to_timeframe_score(score_data: dict, name: str, bias: str, score: int, reason: str) -> None:
    signed_score = -score if bias == "bearish" else score
    score_data.setdefault("components", []).append({"name": name, "score": signed_score, "reason": reason})
    bullish = score_data.get("_bullish_total", 0)
    bearish = score_data.get("_bearish_total", 0)
    if bias == "bullish":
        bullish += score
    elif bias == "bearish":
        bearish += score
    updated = _directional_score(bullish, bearish, score_data["components"])
    updated["_bullish_total"] = bullish
    updated["_bearish_total"] = bearish
    score_data.update(updated)


def _directional_score(bullish_total: int, bearish_total: int, components: list[dict]) -> dict:
    total = 50 + bullish_total - bearish_total
    if bullish_total > bearish_total + 10:
        return _score(total, "偏多", components)
    if bearish_total > bullish_total + 10:
        return _score(total, "偏空", components)
    return _score(total, "多空混合", components)


def _aggregate_direction(directions: list[str]) -> str:
    if not directions:
        return "无明确信号"
    bullish = directions.count("偏多")
    bearish = directions.count("偏空")
    if bullish > bearish:
        return "偏多"
    if bearish > bullish:
        return "偏空"
    return "多空混合"


def score_fundamental_agent(metrics: dict) -> dict:
    metric = metrics.get("metric") or metrics
    components = []
    score = 0
    fcf = _num(metric.get("freeCashFlowPerShareTTM"))
    cash_flow = _num(metric.get("cashFlowPerShareTTM"))
    net_income = _num(metric.get("netIncomePerShareTTM"))
    current_ratio = _num(metric.get("currentRatioAnnual"))
    debt_equity = _num(metric.get("totalDebt/totalEquityAnnual"))

    if fcf is not None and fcf > 0:
        score += 30
        components.append({"name": "自由现金流为正", "score": 30, "reason": "现金流质量优先于会计利润。"})
    if cash_flow is not None and net_income not in (None, 0):
        ratio = cash_flow / abs(net_income)
        if ratio >= 1:
            score += 25
            components.append({"name": "经营现金流覆盖利润", "score": 25, "reason": "利润有现金流支撑。"})
        elif ratio < 0.8:
            score -= 15
            components.append({"name": "现金流弱于利润", "score": -15, "reason": "利润质量需要警惕。"})
    if current_ratio is not None and current_ratio >= 1.5:
        score += 15
        components.append({"name": "流动比率健康", "score": 15, "reason": "短期偿债压力较低。"})
    if debt_equity is not None:
        if debt_equity <= 1:
            score += 15
            components.append({"name": "杠杆压力可控", "score": 15, "reason": "债务权益比不高。"})
        elif debt_equity >= 2:
            score -= 20
            components.append({"name": "杠杆偏高", "score": -20, "reason": "资产负债表风险上升。"})
    direction = "基本面偏强" if score >= 50 else "基本面偏弱" if score <= 20 else "基本面中性"
    return _score(score, direction, components)


def _pattern_score(name: str, bias: str) -> int:
    if "bullish" in bias:
        return BULLISH_PATTERN_SCORES.get(name, 0)
    if "bearish" in bias:
        return BEARISH_PATTERN_SCORES.get(name, 0)
    return 0


def _pattern_label(name: str, bias: str) -> str:
    if name == "锤子线" and "bullish" in bias:
        return "低位锤子线"
    return name


def _timeframe_name(key: str) -> str:
    return {"daily": "日线", "weekly": "周线"}.get(key, key)


def _public_score(score: dict) -> dict:
    return {key: value for key, value in score.items() if not str(key).startswith("_")}


def _score(total: int | float, direction: str, components: list[dict]) -> dict:
    return {
        "total_score": max(0, min(100, round(float(total)))),
        "direction": direction,
        "components": components,
    }


def _num(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
