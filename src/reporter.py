from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .dingtalk import DingTalkNotifier
from .models import ImpactAnalysis, NewsItem
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
except Exception:
    Console = None
    Panel = None
    Text = None


class Reporter:
    def __init__(
        self,
        report_dir: Path,
        display_timezone: str = "Asia/Shanghai",
        dingtalk_webhook_url: str = "",
        dingtalk_keyword: str = "",
        dingtalk_secret: str = "",
    ):
        self.report_dir = report_dir
        self.intraday_dir = report_dir / "intraday_alerts"
        self.premarket_dir = report_dir / "premarket"
        self.close_dir = report_dir / "close"
        self.intraday_dir.mkdir(parents=True, exist_ok=True)
        self.premarket_dir.mkdir(parents=True, exist_ok=True)
        self.close_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console() if Console else None
        self.display_timezone = ZoneInfo(display_timezone)
        self.notifier = DingTalkNotifier(dingtalk_webhook_url, dingtalk_keyword, dingtalk_secret)

    def emit_news_analyses(self, news: NewsItem, analyses: list[ImpactAnalysis]) -> None:
        """Publish one alert for a story, even when it affects several holdings."""
        content = self._render_news_analyses(news, analyses)
        title = f"{', '.join(analysis.ticker for analysis in analyses)} 新闻分析"
        self.display_summary(content, title)
        self.notifier.send_markdown(title, content)
        path = self.intraday_dir / f"{news.published_at.astimezone(self.display_timezone).date().isoformat()}.md"
        with path.open("a", encoding="utf-8") as file:
            file.write(content)
            file.write("\n\n---\n\n")

    def write_premarket(self, content: str, display: bool = True) -> Path:
        path = self.premarket_dir / f"{datetime.now().date().isoformat()}.md"
        return self._write_summary(path, content, "盘前报告", display)

    def write_close(self, content: str, display: bool = True) -> Path:
        path = self.close_dir / f"{datetime.now().date().isoformat()}.md"
        return self._write_summary(path, content, "盘后报告", display)

    def display_summary(self, content: str, title: str) -> None:
        if self.console:
            self.console.print(
                Panel(Text(content) if Text else content, title=title, border_style="cyan", expand=True),
                overflow="fold",
                crop=False,
            )
        else:
            print(content)

    def _write_summary(self, path: Path, content: str, title: str, display: bool) -> Path:
        path.write_text(content, encoding="utf-8")
        self.notifier.send_markdown(title, content)
        if display:
            self.display_summary(content, title)
        return path

    def _format_news_time(self, news: NewsItem) -> str:
        local_time = news.published_at.astimezone(self.display_timezone)
        return local_time.strftime("%Y-%m-%d %H:%M:%S %Z")

    def _render_news_analyses(self, news: NewsItem, analyses: list[ImpactAnalysis]) -> str:
        sections = [
            "### 新闻",
            "",
            f"- 标题: [{news.headline}]({news.url})",
            f"- 发布时间: {self._format_news_time(news)}",
            f"- 类型: {news.story_type or '未标注'}",
        ]
        for analysis in analyses:
            sections.extend(
                [
                    "",
                    f"### {analysis.ticker} 分析",
                    "",
                    f"新闻Agent：{analysis.reasoning_zh}",
                    "",
                    analysis.combined_conclusion_zh or "激进Agent分析暂未生成。",
                ]
            )
        return "\n".join(sections)
