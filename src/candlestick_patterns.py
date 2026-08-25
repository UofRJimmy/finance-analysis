from __future__ import annotations

import pandas as pd


def analyze_candlestick_patterns(frame: pd.DataFrame, timeframe: str) -> dict:
    """Detect common candlestick and price-action patterns from OHLCV bars."""
    if frame.empty or len(frame) < 3:
        return {"available": False, "timeframe": timeframe, "patterns": [], "names": [], "primary_pattern": "数据不足"}

    data = frame.copy().tail(160).reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = data[column].astype(float)

    patterns: list[dict] = []
    _single_bar_patterns(data, patterns)
    _two_bar_patterns(data, patterns)
    _three_bar_patterns(data, patterns)
    _multi_bar_patterns(data, patterns)
    _gap_patterns(data, patterns)
    _top_bottom_patterns(data, patterns)
    _volume_price_patterns(data, patterns)

    unique = []
    seen = set()
    for item in sorted(patterns, key=lambda row: row["priority"], reverse=True):
        if item["name"] in seen:
            continue
        seen.add(item["name"])
        unique.append({key: value for key, value in item.items() if key != "priority"})

    names = [item["name"] for item in unique[:8]]
    bias = _decision_bias(unique)
    return {
        "available": True,
        "timeframe": timeframe,
        "patterns": unique[:8],
        "names": names,
        "primary_pattern": "、".join(names[:3]) if names else "未发现明确蜡烛图形态",
        "bias": bias["bias"],
        "bias_reason": bias["reason"],
    }


def _single_bar_patterns(data: pd.DataFrame, patterns: list[dict]) -> None:
    row = data.iloc[-1]
    feature = _bar(row)
    prior_trend = _trend(data.iloc[:-1])
    high_position = _position_in_range(data, row["close"], "high")
    low_position = _position_in_range(data, row["close"], "low")

    if feature["range_pct"] < 0.001:
        _add(patterns, "一字线", "neutral", 68, "几乎没有波动，流动性或等待方向特征明显。")
        return

    if feature["is_bull"] and feature["body_ratio"] >= 0.6:
        _add(patterns, "大阳线", "bullish", 76, "长实体收涨，买方主导。")
    elif feature["is_bear"] and feature["body_ratio"] >= 0.6:
        _add(patterns, "大阴线", "bearish", 76, "长实体收跌，卖方主导。")
    elif feature["is_bull"] and feature["body_ratio"] < 0.35:
        _add(patterns, "小阳线", "slightly_bullish", 48, "小实体收涨，买方略占优势但力度有限。")
    elif feature["is_bear"] and feature["body_ratio"] < 0.35:
        _add(patterns, "小阴线", "slightly_bearish", 48, "小实体收跌，卖方略占优势但抛压不重。")

    if feature["body_ratio"] <= 0.10:
        _add(patterns, "十字星", "neutral", 62, "开收盘接近，多空暂时平衡。")
        if feature["upper_ratio"] >= 0.35 and feature["lower_ratio"] >= 0.35:
            _add(patterns, "长腿十字星", "neutral", 70, "上下影线都长，盘中分歧剧烈。")
        if feature["lower_ratio"] >= 0.6 and feature["upper_ratio"] <= 0.12:
            _add(patterns, "蜻蜓十字星", "bullish", 74, "下影线很长，杀跌后被买方拉回。")
        if feature["upper_ratio"] >= 0.6 and feature["lower_ratio"] <= 0.12:
            _add(patterns, "墓碑十字星", "bearish", 74, "上影线很长，冲高后被空方打回。")

    if feature["lower_ratio"] >= 0.55 and feature["upper_ratio"] <= 0.2 and feature["body_ratio"] <= 0.35:
        _add(patterns, "锤子线" if prior_trend == "down" else "上吊线", "bullish" if prior_trend == "down" else "bearish", 80, "长下影显示下方承接。")
    if feature["upper_ratio"] >= 0.55 and feature["lower_ratio"] <= 0.2 and feature["body_ratio"] <= 0.35:
        _add(patterns, "倒锤子线" if prior_trend == "down" else "射击之星", "bullish" if prior_trend == "down" else "bearish", 80, "长上影显示冲高尝试或上方抛压。")
    if feature["body_ratio"] <= 0.35 and feature["upper_ratio"] >= 0.2 and feature["lower_ratio"] >= 0.2:
        _add(patterns, "纺锤线", "neutral", 50, "上下都有试探，趋势可能停顿。")

    if feature["is_bull"] and feature["upper_ratio"] <= 0.05:
        _add(patterns, "光头阳线", "bullish", 64, "收盘接近最高价，买方收盘仍强。")
    if feature["is_bull"] and feature["lower_ratio"] <= 0.05:
        _add(patterns, "光脚阳线", "bullish", 60, "开盘接近最低价，买方从开盘主导。")
    if feature["is_bear"] and feature["upper_ratio"] <= 0.05:
        _add(patterns, "光头阴线", "bearish", 60, "开盘接近最高价，卖方一路压制。")
    if feature["is_bear"] and feature["lower_ratio"] <= 0.05:
        _add(patterns, "光脚阴线", "bearish", 64, "收盘接近最低价，卖压持续到收盘。")
    if feature["upper_ratio"] >= 0.5:
        _add(patterns, "长上影线", "bearish" if high_position else "neutral", 68, "冲高回落，上方抛压较明显。")
    if feature["lower_ratio"] >= 0.5:
        _add(patterns, "长下影线", "bullish" if low_position else "neutral", 68, "杀跌后收回，下方承接较明显。")


def _two_bar_patterns(data: pd.DataFrame, patterns: list[dict]) -> None:
    prev, row = data.iloc[-2], data.iloc[-1]
    a, b = _bar(prev), _bar(row)
    trend = _trend(data.iloc[:-2])
    prev_low_close = min(prev["open"], prev["close"])
    prev_high_close = max(prev["open"], prev["close"])
    cur_low_close = min(row["open"], row["close"])
    cur_high_close = max(row["open"], row["close"])
    tolerance = max(row["close"] * 0.003, b["range"] * 0.15)

    if a["is_bear"] and b["is_bull"] and cur_low_close <= prev_low_close and cur_high_close >= prev_high_close:
        _add(patterns, "看涨吞没", "bullish", 86, "阳线实体吞没前一根阴线实体。")
    if a["is_bull"] and b["is_bear"] and cur_low_close <= prev_low_close and cur_high_close >= prev_high_close:
        _add(patterns, "看跌吞没", "bearish", 86, "阴线实体吞没前一根阳线实体。")
    if trend == "down" and a["is_bear"] and b["is_bull"] and row["close"] > (prev["open"] + prev["close"]) / 2:
        _add(patterns, "刺透形态", "bullish", 78, "低开高走并收回前阴线实体一半以上。")
    if trend == "up" and a["is_bull"] and b["is_bear"] and row["close"] < (prev["open"] + prev["close"]) / 2:
        _add(patterns, "乌云盖顶", "bearish", 78, "高开低走并深入前阳线实体。")
    if a["is_bear"] and a["body_ratio"] >= 0.5 and b["body_ratio"] <= 0.35 and cur_low_close >= prev_low_close and cur_high_close <= prev_high_close:
        _add(patterns, "看涨孕线", "bullish", 70, "小实体被前一根大阴线包住，下跌动能减弱。")
    if a["is_bull"] and a["body_ratio"] >= 0.5 and b["body_ratio"] <= 0.35 and cur_low_close >= prev_low_close and cur_high_close <= prev_high_close:
        _add(patterns, "看跌孕线", "bearish", 70, "小实体被前一根大阳线包住，上涨动能减弱。")
    if b["body_ratio"] <= 0.1 and cur_low_close >= prev_low_close and cur_high_close <= prev_high_close:
        _add(patterns, "十字孕线", "neutral", 72, "十字星被前一根实体包住，趋势犹豫增强。")
    if abs(prev["low"] - row["low"]) <= tolerance:
        _add(patterns, "平头底", "bullish", 62, "两根K线低点接近，支撑被反复守住。")
        _add(patterns, "镊子底", "bullish", 64, "低位两次下探相近区域失败。")
    if abs(prev["high"] - row["high"]) <= tolerance:
        _add(patterns, "平头顶", "bearish", 62, "两根K线高点接近，压力被反复验证。")
        _add(patterns, "镊子顶", "bearish", 64, "高位两次冲高相近区域失败。")
    if a["body_ratio"] >= 0.55 and a["is_bull"] != b["is_bull"] and abs(prev["close"] - row["close"]) <= tolerance:
        _add(patterns, "反击线", "neutral", 66, "反向K线收盘接近前收盘，原趋势可能失效。")
    if abs(prev["open"] - row["open"]) <= tolerance and a["is_bull"] != b["is_bull"]:
        _add(patterns, "分手线", "neutral", 58, "相近开盘区域给出相反结果，方向重新选择。")


def _three_bar_patterns(data: pd.DataFrame, patterns: list[dict]) -> None:
    a, b, c = data.iloc[-3], data.iloc[-2], data.iloc[-1]
    fa, fb, fc = _bar(a), _bar(b), _bar(c)
    if fa["is_bear"] and fa["body_ratio"] >= 0.45 and fb["body_ratio"] <= 0.3 and fc["is_bull"] and c["close"] > (a["open"] + a["close"]) / 2:
        _add(patterns, "早晨十字星" if fb["body_ratio"] <= 0.1 else "早晨之星", "bullish", 88, "下跌、犹豫后出现阳线反攻。")
    if fa["is_bull"] and fa["body_ratio"] >= 0.45 and fb["body_ratio"] <= 0.3 and fc["is_bear"] and c["close"] < (a["open"] + a["close"]) / 2:
        _add(patterns, "黄昏十字星" if fb["body_ratio"] <= 0.1 else "黄昏之星", "bearish", 88, "上涨、犹豫后出现阴线反击。")
    last3 = data.tail(3)
    if all(row["close"] > row["open"] for _, row in last3.iterrows()) and last3["close"].is_monotonic_increasing:
        _add(patterns, "三白兵", "bullish", 82, "连续三根阳线且收盘逐步抬高。")
        _add(patterns, "红三兵", "bullish", 78, "买盘连续进入，趋势可能延续。")
    if all(row["close"] < row["open"] for _, row in last3.iterrows()) and last3["close"].is_monotonic_decreasing:
        _add(patterns, "三只乌鸦", "bearish", 82, "连续三根阴线且收盘逐步降低。")
        _add(patterns, "黑三兵", "bearish", 78, "卖盘持续释放，情绪转弱。")
    if fa["is_bear"] and fb["body_ratio"] <= 0.35 and fc["is_bull"] and c["close"] > b["high"]:
        _add(patterns, "三内升", "bullish", 76, "孕线后第三根向上确认。")
    if fa["is_bull"] and fb["body_ratio"] <= 0.35 and fc["is_bear"] and c["close"] < b["low"]:
        _add(patterns, "三内降", "bearish", 76, "孕线后第三根向下确认。")
    if fa["is_bear"] and fb["is_bull"] and b["close"] > a["open"] and fc["is_bull"]:
        _add(patterns, "三外升", "bullish", 80, "看涨吞没后继续上涨确认。")
    if fa["is_bull"] and fb["is_bear"] and b["close"] < a["open"] and fc["is_bear"]:
        _add(patterns, "三外降", "bearish", 80, "看跌吞没后继续下跌确认。")
    if all(_bar(row)["body_ratio"] <= 0.1 for _, row in last3.iterrows()):
        _add(patterns, "三星形态", "neutral", 74, "连续三根十字星，方向选择临近。")


def _multi_bar_patterns(data: pd.DataFrame, patterns: list[dict]) -> None:
    window = data.tail(20)
    highs = window["high"]
    lows = window["low"]
    closes = window["close"]
    if len(window) >= 8:
        if highs.iloc[-1] > highs.iloc[-6:-1].max() and lows.tail(6).min() > lows.iloc[-12:-6].min():
            _add(patterns, "上升趋势排列", "bullish", 72, "高点和低点整体抬高。")
        if highs.tail(6).max() < highs.iloc[-12:-6].max() and lows.iloc[-1] < lows.iloc[-6:-1].min():
            _add(patterns, "下跌趋势排列", "bearish", 72, "高点和低点整体降低。")
    range_pct = (highs.max() - lows.min()) / max(closes.iloc[-1], 0.01)
    if range_pct <= 0.08:
        _add(patterns, "横盘震荡", "neutral", 58, "价格在窄区间反复波动。")
        _add(patterns, "箱体整理", "neutral", 58, "上沿和下沿较固定，等待突破。")
        _add(patterns, "矩形整理", "neutral", 54, "水平支撑压力之间震荡。")
    recent_range = (window["high"] - window["low"]).rolling(5).mean()
    if len(recent_range.dropna()) >= 2 and recent_range.iloc[-1] < recent_range.dropna().iloc[0] * 0.65:
        _add(patterns, "三角形整理", "neutral", 62, "波动逐渐收窄，等待突破。")
        slope = closes.tail(10).iloc[-1] - closes.tail(10).iloc[0]
        _add(patterns, "楔形整理", "bearish" if slope > 0 else "bullish", 58, "收敛结构显示趋势动能衰减。")

    if len(data) >= 6:
        first = _bar(data.iloc[-6])
        middle = data.iloc[-5:-1]
        last = _bar(data.iloc[-1])
        if first["is_bull"] and first["body_ratio"] > 0.55 and all(row["close"] < row["open"] for _, row in middle.tail(3).iterrows()) and last["is_bull"]:
            if middle["low"].min() >= data.iloc[-6]["low"] and data.iloc[-1]["close"] > data.iloc[-6]["high"]:
                _add(patterns, "上升三法", "bullish", 84, "大阳后小阴调整不破范围，最后再上攻。")
        if first["is_bear"] and first["body_ratio"] > 0.55 and all(row["close"] > row["open"] for _, row in middle.tail(3).iterrows()) and last["is_bear"]:
            if middle["high"].max() <= data.iloc[-6]["high"] and data.iloc[-1]["close"] < data.iloc[-6]["low"]:
                _add(patterns, "下跌三法", "bearish", 84, "大阴后弱反弹不破范围，最后再下压。")


def _gap_patterns(data: pd.DataFrame, patterns: list[dict]) -> None:
    if len(data) < 3:
        return
    prev, row = data.iloc[-2], data.iloc[-1]
    trend = _trend(data.iloc[:-1])
    gap_up = row["low"] > prev["high"]
    gap_down = row["high"] < prev["low"]
    if not gap_up and not gap_down:
        return
    bias = "bullish" if gap_up else "bearish"
    _add(patterns, "普通缺口", bias, 60, "价格跳空，短期供需突然失衡。")
    if gap_up and row["close"] > data["high"].shift(1).rolling(20).max().iloc[-1]:
        _add(patterns, "突破缺口", "bullish", 86, "跳空突破近期压力。")
    if gap_down and row["close"] < data["low"].shift(1).rolling(20).min().iloc[-1]:
        _add(patterns, "突破缺口", "bearish", 86, "跳空跌破近期支撑。")
    if (gap_up and trend == "up") or (gap_down and trend == "down"):
        _add(patterns, "持续缺口", bias, 74, "趋势中跳空，原方向情绪加速。")
    if (gap_up and row["close"] < row["open"]) or (gap_down and row["close"] > row["open"]):
        _add(patterns, "衰竭缺口", "bearish" if gap_up else "bullish", 78, "跳空后反向收盘，趋势末端风险升高。")
    if len(data) >= 6:
        prior_gap_up = data.iloc[-5]["low"] > data.iloc[-6]["high"]
        prior_gap_down = data.iloc[-5]["high"] < data.iloc[-6]["low"]
        if (prior_gap_up and gap_down) or (prior_gap_down and gap_up):
            _add(patterns, "岛形反转", "bearish" if gap_down else "bullish", 90, "前后反向缺口形成孤岛结构。")


def _top_bottom_patterns(data: pd.DataFrame, patterns: list[dict]) -> None:
    window = data.tail(80).reset_index(drop=True)
    highs = _swing_points(window, "high", "high")
    lows = _swing_points(window, "low", "low")
    close = window.iloc[-1]["close"]
    tolerance = close * 0.025
    if len(highs) >= 2 and abs(highs[-1][1] - highs[-2][1]) <= tolerance:
        _add(patterns, "双顶", "bearish", 76, "两次冲击相近高点失败。")
        _add(patterns, "M顶", "bearish", 74, "双顶结构，买方动能不足。")
    if len(highs) >= 3 and max(value for _, value in highs[-3:]) - min(value for _, value in highs[-3:]) <= tolerance:
        _add(patterns, "三重顶", "bearish", 82, "三次冲击相近高点失败。")
    if len(lows) >= 2 and abs(lows[-1][1] - lows[-2][1]) <= tolerance:
        _add(patterns, "双底", "bullish", 76, "两次下探相近低点后被承接。")
        _add(patterns, "W底", "bullish", 74, "双底结构，支撑区域被验证。")
    if len(lows) >= 3 and max(value for _, value in lows[-3:]) - min(value for _, value in lows[-3:]) <= tolerance:
        _add(patterns, "三重底", "bullish", 82, "底部支撑多次验证。")
    if len(highs) >= 3 and highs[-2][1] > highs[-3][1] and highs[-2][1] > highs[-1][1]:
        _add(patterns, "头肩顶", "bearish", 80, "中间高点最高，右肩未能创新高。")
    if len(lows) >= 3 and lows[-2][1] < lows[-3][1] and lows[-2][1] < lows[-1][1]:
        _add(patterns, "头肩底", "bullish", 80, "中间低点最低，右肩不再创新低。")
    pct_5 = close / window.iloc[-6]["close"] - 1 if len(window) >= 6 else 0
    pct_15 = close / window.iloc[-16]["close"] - 1 if len(window) >= 16 else 0
    if pct_5 < -0.08 and pct_15 > 0.08:
        _add(patterns, "V形反转顶", "bearish", 72, "快速上涨后快速回落。")
        _add(patterns, "尖顶", "bearish", 68, "顶部停留很短后快速下跌。")
    if pct_5 > 0.08 and pct_15 < -0.08:
        _add(patterns, "V形反转底", "bullish", 72, "快速下跌后快速反弹。")
        _add(patterns, "尖底", "bullish", 68, "底部停留很短后快速反弹。")
    if _rounded_turn(window["close"], top=True):
        _add(patterns, "圆弧顶", "bearish", 64, "价格从上涨逐渐转为走平再转弱。")
    if _rounded_turn(window["close"], top=False):
        _add(patterns, "圆弧底", "bullish", 64, "价格从下跌逐渐企稳再转强。")


def _volume_price_patterns(data: pd.DataFrame, patterns: list[dict]) -> None:
    row = data.iloc[-1]
    feature = _bar(row)
    avg_volume = data["volume"].tail(21).iloc[:-1].mean()
    volume_ratio = row["volume"] / avg_volume if avg_volume and avg_volume > 0 else 1.0
    prior_high = data["high"].shift(1).rolling(20).max().iloc[-1]
    prior_low = data["low"].shift(1).rolling(20).min().iloc[-1]
    close = row["close"]
    if pd.notna(prior_high) and close > prior_high and feature["is_bull"] and volume_ratio >= 1.5:
        _add(patterns, "放量长阳突破", "bullish", 92, "长阳突破压力且成交量明显放大。")
    if pd.notna(prior_high) and close > prior_high and volume_ratio < 0.9:
        _add(patterns, "缩量突破", "bearish", 72, "突破缺少量能配合，假突破风险较高。")
    if pd.notna(prior_low) and close < prior_low and feature["is_bear"] and volume_ratio >= 1.5:
        _add(patterns, "放量长阴破位", "bearish", 92, "长阴跌破支撑且放量。")
    if feature["is_bear"] and volume_ratio < 0.8:
        _add(patterns, "缩量下跌", "neutral", 56, "下跌但量能不足，可能只是调整。")
    if feature["is_bull"] and feature["body_ratio"] < 0.35 and volume_ratio >= 1.5:
        _add(patterns, "放量滞涨", "bearish", 76, "放量但上涨有限，上方卖盘较重。")
    if feature["is_bear"] and feature["lower_ratio"] >= 0.45 and volume_ratio >= 1.5:
        _add(patterns, "放量下跌后长下影", "bullish", 80, "恐慌抛压被承接。")
    support = data["low"].tail(20).min()
    if abs(close - support) / max(close, 0.01) <= 0.02 and volume_ratio < 0.85:
        _add(patterns, "缩量回踩支撑", "bullish", 72, "回踩支撑时抛压减轻。")
    if len(data) >= 5:
        breakout_recent = data.iloc[-5:-1]["close"].max() > data["high"].shift(1).rolling(20).max().iloc[-5:-1].max()
        if breakout_recent and close > prior_high * 0.97 and volume_ratio < 0.9:
            _add(patterns, "放量突破后缩量回踩", "bullish", 76, "突破后缩量回踩，结构相对健康。")
    if _position_in_range(data, close, "high") and feature["upper_ratio"] >= 0.45 and volume_ratio >= 1.5:
        _add(patterns, "高位长上影放量", "bearish", 84, "高位冲高回落且放量，卖压明显。")
    if _position_in_range(data, close, "low") and feature["lower_ratio"] >= 0.45 and volume_ratio >= 1.5:
        _add(patterns, "低位长下影放量", "bullish", 84, "低位杀跌后放量拉回，承接增强。")


def _bar(row) -> dict:
    open_, high, low, close = row["open"], row["high"], row["low"], row["close"]
    range_ = max(high - low, 1e-9)
    body = abs(close - open_)
    upper = high - max(open_, close)
    lower = min(open_, close) - low
    return {
        "range": range_,
        "range_pct": range_ / max(close, 0.01),
        "body": body,
        "body_ratio": body / range_,
        "upper_ratio": upper / range_,
        "lower_ratio": lower / range_,
        "is_bull": close > open_,
        "is_bear": close < open_,
    }


def _trend(data: pd.DataFrame) -> str:
    if len(data) < 12:
        return "flat"
    close = data["close"].tail(20)
    pct = close.iloc[-1] / close.iloc[0] - 1
    if pct > 0.04:
        return "up"
    if pct < -0.04:
        return "down"
    return "flat"


def _position_in_range(data: pd.DataFrame, price: float, side: str) -> bool:
    window = data.tail(60)
    high, low = window["high"].max(), window["low"].min()
    if high <= low:
        return False
    percentile = (price - low) / (high - low)
    return percentile >= 0.75 if side == "high" else percentile <= 0.25


def _swing_points(frame: pd.DataFrame, column: str, kind: str) -> list[tuple[int, float]]:
    values = frame[column].astype(float).reset_index(drop=True)
    result = []
    for index in range(2, len(values) - 2):
        segment = values.iloc[index - 2 : index + 3]
        if kind == "high" and values.iloc[index] == segment.max():
            result.append((index, float(values.iloc[index])))
        if kind == "low" and values.iloc[index] == segment.min():
            result.append((index, float(values.iloc[index])))
    return result


def _rounded_turn(series: pd.Series, *, top: bool) -> bool:
    if len(series) < 45:
        return False
    s = series.tail(45).reset_index(drop=True)
    first, middle, last = s.iloc[:15].mean(), s.iloc[15:30].mean(), s.iloc[30:].mean()
    return first < middle and last < middle * 0.98 if top else first > middle and last > middle * 1.02


def _decision_bias(patterns: list[dict]) -> dict:
    score = 0
    for item in patterns[:6]:
        weight = item.get("confidence", 60) / 100
        if item["bias"] == "bullish":
            score += 2 * weight
        elif item["bias"] == "slightly_bullish":
            score += weight
        elif item["bias"] == "bearish":
            score -= 2 * weight
        elif item["bias"] == "slightly_bearish":
            score -= weight
    if score >= 1.5:
        return {"bias": "偏向可考虑买入", "reason": "蜡烛图形态整体偏多。"}
    if score <= -1.5:
        return {"bias": "偏向可考虑卖出", "reason": "蜡烛图形态整体偏空。"}
    return {"bias": "偏向持有观察", "reason": "蜡烛图信号分歧或力度不足。"}


def _add(patterns: list[dict], name: str, bias: str, confidence: int, reason: str) -> None:
    priority = confidence
    if name in {"放量长阳突破", "放量长阴破位", "岛形反转", "早晨之星", "黄昏之星", "早晨十字星", "黄昏十字星"}:
        priority += 15
    patterns.append({"name": name, "bias": bias, "confidence": confidence, "reason": reason, "priority": priority})
