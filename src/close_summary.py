from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from .aggressive_decision_agent import AggressiveDecisionAgent
from .config import load_settings
from .conservative_decision_agent import ConservativeDecisionAgent
from .dedup_store import DedupStore
from .earnings_calendar import EarningsCalendar
from .fundamental_agent import FundamentalAgent
from .judge_decision_agent import JudgeDecisionAgent, format_judge_block, market_sentiment_from_vix, weighted_core_score
from .market_calendar import is_us_market_session, market_closed_reason
from .deepseek_helpers import make_client, response_text
from .portfolio import PortfolioStore
from .reporter import Reporter
from .technical_agent import CloseRiskTechnicalAgent
from .watchlist import Watchlist


def generate_close_summary(display: bool = False) -> str:
    """生成新的盘后持仓风险报告；不再使用旧盘后总结模板。"""
    settings = load_settings()
    if not is_us_market_session():
        return f"盘后报告跳过：{market_closed_reason()}，等开市再生成。"
    watchlist = Watchlist(settings.watchlist_file).get()
    reporter = Reporter(
        settings.report_dir,
        settings.display_timezone,
        settings.dingtalk_webhook_url,
        settings.dingtalk_keyword,
        settings.dingtalk_secret,
    )
    if not watchlist:
        path = reporter.write_close("# 盘后持仓风险报告\n\nwatchlist 为空。", display=display)
        return f"盘后报告已生成: {path}"

    store = DedupStore(settings.sqlite_path)
    try:
        records = store.recent_records(datetime.now(timezone.utc) - timedelta(hours=24))
    finally:
        store.close()

    risk_analyzer = CloseRiskTechnicalAgent()
    snapshots = {ticker: risk_analyzer.analyze(ticker) for ticker in watchlist}
    earnings_calendar = EarningsCalendar(settings.finnhub_api_key, settings.finnhub_min_request_interval_seconds)
    earnings_dates = {ticker: earnings_calendar.dates_for(ticker) for ticker in watchlist}
    fundamental_agent = FundamentalAgent(settings)
    fundamental_scores = _fundamental_scores(fundamental_agent, watchlist)
    market_sentiment = market_sentiment_from_vix(_vix_indicator(fundamental_agent))
    weights = _portfolio_weights(settings, watchlist, snapshots)
    ticker_contexts = _build_ticker_contexts(
        watchlist,
        records,
        snapshots,
        weights,
        earnings_dates,
        fundamental_scores,
        market_sentiment,
    )
    _attach_decision_debate(settings, ticker_contexts)
    context = {
        "report_type": "盘后持仓风险报告",
        "technical_timeframes": ["1d", "1wk"],
        "news_window": "过去24小时截至报告生成时",
        "allowed_watchlist": watchlist,
        "tickers": ticker_contexts,
    }
    if settings.deepseek_api_key:
        try:
            content = _llm_close_risk_report(settings, context)
            missing = [ticker for ticker in watchlist if ticker not in content]
            if missing:
                raise RuntimeError(f"模型报告遗漏标的: {', '.join(missing)}")
        except Exception as exc:
            message = f"盘后报告生成失败：{type(exc).__name__}: {exc}"
            print(f"[盘后报告] {message}", flush=True)
            return message
    else:
        return "盘后报告生成失败：未配置 DEEPSEEK_API_KEY，无法调用模型。"
    path = reporter.write_close(content, display=display)
    return f"盘后报告已生成: {path}"


def _llm_close_risk_report(settings, context: dict) -> str:
    client = make_client(settings.deepseek_api_key)
    if not client:
        raise RuntimeError("DeepSeek 客户端不可用或 API key 未配置")
    response = client.responses.create(
        model=settings.deepseek_model_summary,
        input=[
            {
                "role": "system",
                "content": (
                    "你是私人高级美股风险分析师。生成盘后持仓风险报告，必须覆盖数据中的每一个watchlist标的。"
                    "报告只能分析allowed_watchlist里的标的；任何不在allowed_watchlist里的股票都必须忽略。"
                    "只能使用提供的日线、周线、过去24小时消息和持仓权重；禁止引用分时指标。"
                    "每个标的严格按顺序输出：风险系数与等级、价格/压力区/支撑区、消息面总结、"
                    "日线与周线技术风险、蜡烛图形态名称、背离、三个主要风险、下一交易日两个验证信号。"
                    "每个标的必须输出技术Agent评分：使用technical_agent_score.total_score和direction，并把timeframe_scores里的日线、周线评分分开列出。"
                    "财报日期必须使用earnings_calendar字段；查不到时写暂无数据，禁止编造日期。"
                    "风险系数必须原样使用final_risk_score，不得自行修改；"
                    "每个标的必须原样输出decision_debate.formatted中的三方决策总结，不得遗漏。"
                    "这只是决策倾向，不是交易指令，必须附上不构成投资建议。"
                    "没有重大新闻时明确写消息面无新增重大驱动。不得写成“现在立刻交易”的执行指令。"
                ),
            },
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
    )
    content = response_text(response).strip()
    if not content:
        raise RuntimeError("模型返回了空的盘后风险报告")
    return content


def _build_ticker_contexts(
    watchlist: list[str],
    records: list[dict],
    snapshots: dict,
    weights: dict,
    earnings_dates: dict | None = None,
    fundamental_scores: dict | None = None,
    market_sentiment: dict | None = None,
) -> list[dict]:
    earnings_dates = earnings_dates or {}
    fundamental_scores = fundamental_scores or {}
    market_sentiment = market_sentiment or {}
    grouped: dict[str, list[dict]] = {ticker: [] for ticker in watchlist}
    for row in records:
        analysis = row["analysis"]
        ticker = analysis.get("ticker")
        if ticker not in grouped or analysis.get("sentiment") == "neutral":
            continue
        grouped[ticker].append(
            {
                "news_id": analysis.get("news_id"),
                "headline": row["news"].get("headline"),
                "published_at": row["news"].get("published_at"),
                "sentiment": analysis.get("sentiment"),
                "confidence": analysis.get("confidence"),
                "magnitude": analysis.get("magnitude"),
                "reason": analysis.get("reasoning_zh"),
                "message_technical_conclusion": analysis.get("combined_conclusion_zh"),
                "agent_scores": analysis.get("agent_scores") or {},
            }
        )

    contexts = []
    for ticker in watchlist:
        news_events = _deduplicate_events(grouped[ticker])
        news_risk = _news_risk_score(news_events)
        technical = snapshots[ticker]
        technical_risk = int(technical.get("technical_risk_score", 50)) if technical.get("available") else 50
        weight_pct = float(weights.get(ticker, 0))
        concentration_risk = min(100, round(weight_pct * 4))
        final_risk = round(technical_risk * 0.55 + news_risk * 0.30 + concentration_risk * 0.15)
        technical_agent_score = technical.get("agent_score", {})
        news_agent_score = _aggregate_news_agent_score(news_events)
        fundamental_agent_score = fundamental_scores.get(ticker, {})
        weighted_score = weighted_core_score(
            fundamental_score=fundamental_agent_score,
            technical_score=technical_agent_score,
            news_score=news_agent_score,
            market_sentiment_score=market_sentiment,
        )
        contexts.append(
            {
                "ticker": ticker,
                "portfolio_weight_pct": round(weight_pct, 2),
                "technical_risk_score": technical_risk,
                "news_risk_score": news_risk,
                "concentration_risk_score": concentration_risk,
                "final_risk_score": max(0, min(100, final_risk)),
                "risk_level": _risk_level(final_risk),
                "daily_weekly_technical": technical,
                "technical_agent_score": technical_agent_score,
                "news_agent_score": news_agent_score,
                "fundamental_agent_score": fundamental_agent_score,
                "market_sentiment_score": market_sentiment,
                "weighted_score": weighted_score,
                "earnings_calendar": earnings_dates.get(ticker, {}),
                "material_news_events": news_events[:5],
                "material_news_count": len(news_events),
                "news_event_summary": _news_event_summary(news_events),
            }
        )
    return contexts


def _attach_decision_debate(settings, ticker_contexts: list[dict]) -> None:
    aggressive = AggressiveDecisionAgent(settings.deepseek_api_key)
    conservative = ConservativeDecisionAgent(settings.deepseek_api_key)
    judge = JudgeDecisionAgent(settings.deepseek_api_key)
    for row in ticker_contexts:
        context = {
            "scope": "close_report_holding",
            "ticker": row["ticker"],
            "weights": "基本面40%，技术面30%，消息面20%，市场情绪10%",
            "weighted_score": row.get("weighted_score", {}),
            "fundamental_agent_score": row.get("fundamental_agent_score", {}),
            "technical_agent_score": row.get("technical_agent_score", {}),
            "news_agent_score": row.get("news_agent_score", {}),
            "market_sentiment_score": row.get("market_sentiment_score", {}),
            "technical": row.get("daily_weekly_technical", {}),
            "news": row.get("material_news_events", []),
            "risk": {
                "final_risk_score": row.get("final_risk_score"),
                "risk_level": row.get("risk_level"),
                "portfolio_weight_pct": row.get("portfolio_weight_pct"),
            },
        }
        try:
            aggressive_view = aggressive.analyze(context)
            conservative_view = conservative.analyze(context)
            judge_view = judge.judge(context, aggressive_view, conservative_view)
            formatted = format_judge_block(judge_view, context)
        except Exception as exc:
            aggressive_view = {}
            conservative_view = {}
            judge_view = {}
            formatted = f"三方决策总结：生成失败（{type(exc).__name__}: {exc}）。这不构成投资建议。"
        row["decision_debate"] = {
            "aggressive": aggressive_view,
            "conservative": conservative_view,
            "judge": judge_view,
            "formatted": formatted,
        }


def _deduplicate_events(events: list[dict]) -> list[dict]:
    unique = {}
    for event in sorted(events, key=lambda row: row.get("published_at") or "", reverse=True):
        key = event.get("news_id") or event.get("headline")
        unique.setdefault(key, event)
    magnitude_order = {"large": 3, "medium": 2, "small": 1}
    return sorted(
        unique.values(),
        key=lambda row: (magnitude_order.get(row.get("magnitude"), 0), row.get("published_at") or ""),
        reverse=True,
    )


def _fundamental_scores(agent: FundamentalAgent, watchlist: list[str]) -> dict[str, dict]:
    scores = {}
    for ticker in watchlist:
        try:
            scores[ticker] = agent.basic_financials(ticker, prefer_edgar=True).get("agent_score", {})
        except Exception as exc:
            scores[ticker] = {
                "total_score": 0,
                "direction": "数据不足",
                "components": [{"name": "基本面数据获取失败", "score": 0, "reason": str(exc)}],
            }
    return scores


def _vix_indicator(agent: FundamentalAgent) -> dict:
    try:
        return agent.get_vix()
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _aggregate_news_agent_score(events: list[dict]) -> dict:
    scores = [
        ((event.get("agent_scores") or {}).get("news_agent") or {})
        for event in events
        if ((event.get("agent_scores") or {}).get("news_agent") or {}).get("total_score") is not None
    ]
    if not scores:
        return {"total_score": 50, "direction": "消息面中性", "components": []}
    total = round(sum(float(score.get("total_score", 0)) for score in scores[:5]) / min(len(scores), 5))
    directions = [score.get("direction") for score in scores[:5]]
    direction = max(set(directions), key=directions.count) if directions else "消息面中性"
    return {
        "total_score": max(0, min(100, total)),
        "direction": direction,
        "components": [{"name": "近24小时相关新闻评分均值", "score": total, "reason": "按重要新闻的新闻Agent评分聚合。"}],
    }


def _news_risk_score(events: list[dict]) -> int:
    score = 50.0
    magnitude = {"large": 12, "medium": 7, "small": 3}
    confidence = {"high": 1.0, "medium": 0.75, "low": 0.5}
    for event in events[:5]:
        impact = magnitude.get(event.get("magnitude"), 3) * confidence.get(event.get("confidence"), 0.75)
        score += impact if event.get("sentiment") == "bearish" else -impact
    return max(0, min(100, round(score)))


def _news_event_summary(events: list[dict]) -> dict:
    return {
        "bullish": sum(event.get("sentiment") == "bullish" for event in events),
        "bearish": sum(event.get("sentiment") == "bearish" for event in events),
        "large_magnitude": sum(event.get("magnitude") == "large" for event in events),
    }


def _portfolio_weights(settings, watchlist: list[str], snapshots: dict) -> dict[str, float]:
    portfolio = PortfolioStore(settings.portfolio_file, settings.trade_log_path).load()
    allowed = {ticker.upper() for ticker in watchlist}
    values = {}
    for holding in portfolio.get("holdings", []):
        ticker = str(holding.get("symbol") or "").upper()
        if ticker not in allowed:
            continue
        quantity = float(holding.get("quantity") or 0)
        snapshot = snapshots.get(ticker) or {}
        close = float(snapshot.get("close") or 0)
        value = quantity * close
        if value <= 0:
            value = float(holding.get("manual_market_value") or 0)
        if value <= 0:
            value = quantity * float(holding.get("average_cost") or 0)
        values[ticker] = value
    total = sum(values.values()) + float(portfolio.get("cash") or 0)
    total += sum(float(row.get("value") or 0) for row in portfolio.get("other_assets", []))
    return {ticker: value / total * 100 if total > 0 else 0 for ticker, value in values.items()}


def _risk_level(score: int | float) -> str:
    if score >= 85:
        return "极高"
    if score >= 70:
        return "高"
    if score >= 50:
        return "较高"
    if score >= 30:
        return "中低"
    return "低"
