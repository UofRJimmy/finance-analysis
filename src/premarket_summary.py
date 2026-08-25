from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from .config import load_settings
from .market_calendar import is_us_market_session, market_closed_reason
from .deepseek_helpers import make_client, response_text
from .news_sources.tickertick_client import TickerTickClient
from .reporter import Reporter
from .watchlist import Watchlist


PREMARKET_MODEL = "deepseek-v4-flash"


def generate_summary(display: bool = False) -> str:
    settings = load_settings()
    if not is_us_market_session():
        return f"盘前总结跳过：{market_closed_reason()}，等开市再生成。"
    reporter = Reporter(
        settings.report_dir,
        settings.display_timezone,
        settings.dingtalk_webhook_url,
        settings.dingtalk_keyword,
        settings.dingtalk_secret,
    )
    if settings.deepseek_api_key:
        try:
            content = _llm_summary(settings)
        except Exception as exc:
            message = f"deepseek-v4-flash 盘前总结生成失败: {type(exc).__name__}: {exc}"
            print(f"[盘前总结] {message}", flush=True)
            return message
    else:
        return "盘前总结生成失败：未配置 DEEPSEEK_API_KEY，无法调用 deepseek-v4-flash。"
    path = reporter.write_premarket(content, display=display)
    return f"盘前总结已生成: {path}"


def _llm_summary(settings) -> str:
    client = make_client(settings.deepseek_api_key)
    if not client:
        raise RuntimeError("DeepSeek 客户端不可用或 API key 未配置")
    today = datetime.now().date().isoformat()
    watchlist = Watchlist(settings.watchlist_file).get()
    try:
        stories = TickerTickClient().fetch_watchlist_news(watchlist, "industry")
    except Exception:
        stories = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    industry_news = [
        {
            "ticker": ",".join(item.tagged_tickers),
            "headline": item.headline,
            "summary": item.summary,
            "source": item.source,
            "published_at": item.published_at.isoformat(),
        }
        for item in stories
        if item.published_at >= cutoff
    ][:40]
    response = client.responses.create(
        model=PREMARKET_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "你是高级美股盘前分析师。只基于提供的 TickerTick industry 新闻生成报告，"
                    "没有数据时明确写行业新闻不足，不得编造或自行网页搜索。只输出中文。"
                    "不要给买入、卖出、加仓、减仓等直接交易指令。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"今天日期是 {today}。行业新闻：{industry_news}\n\n请生成一份美股盘前总结，严格按以下结构：\n"
                    "1. 一句话盘前结论：整体偏利多/利空/中性、力度，并说明是否钝化或可能已 price in。\n"
                    "2. 当日新闻热点：总结2-4个最重要热点，每个热点用标题加一句解释，不要堆新闻列表。\n"
                    "3. 热点解读与产业映射：说明可能影响黄金、美元、美债、半导体、AI、电力、能源、金融、消费、军工等哪些资产或产业；"
                    "写清谁可能受益、谁可能承压、哪些产业可能有潜在动能。\n"
                    "4. 预测市场资金走向：判断目前市场资金正在炒作的主题和板块. \n"
                    "5. 逻辑判断：判断是新增驱动、旧消息延续、利多/利空钝化，还是已被市场 price in；给出开盘后验证信号。\n"
                    "6. 风险与观察：给出2个开盘后最该观察的量价或宏观信号。"
                ),
            },
        ],
    )
    content = response_text(response).strip()
    if not content:
        raise RuntimeError("模型返回了空的盘前总结")
    return content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-now", action="store_true", help="immediately generate a premarket report")
    args = parser.parse_args()
    if args.run_now:
        print(generate_summary(display=True))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
