from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    import yaml
except Exception:
    yaml = None


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    root_dir: Path = ROOT
    watchlist_file: Path = ROOT / "watchlist.txt"
    data_dir: Path = ROOT / "data"
    report_dir: Path = ROOT / "reports"
    sqlite_path: Path = ROOT / "data" / "news_history.sqlite3"
    ticker_meta_path: Path = ROOT / "data" / "ticker_meta.json"
    portfolio_file: Path = ROOT / "portfolio.yaml"
    portfolio_text_file: Path = ROOT / "portfolio.txt"
    portfolio_text_state_path: Path = ROOT / "data" / "portfolio_text_state.json"
    trade_log_path: Path = ROOT / "data" / "trades.jsonl"
    output_language: str = "zh"
    finnhub_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_model_analyze: str = "deepseek-v4-flash"
    deepseek_model_summary: str = "deepseek-v4-flash"
    poll_interval_seconds: int = 90
    tickertick_company_news_interval_seconds: int = 300
    finnhub_min_request_interval_seconds: float = 1.1
    news_max_age_hours: int = 2
    news_retention_hours: int = 24 * 30
    neutral_news_retention_hours: int = 24 * 7
    db_cleanup_interval_hours: int = 24
    news_db_max_items: int = 0
    display_timezone: str = "Asia/Shanghai"
    premarket_summary_hour: int = 9
    premarket_summary_timezone: str = "America/New_York"
    close_summary_hour: int = 16
    close_summary_minute: int = 10
    edgar_identity: str = ""
    edgar_data_dir: Path = ROOT / "data" / "edgar"
    dcf_discount_rate: float = 0.10
    dingtalk_webhook_url: str = ""
    dingtalk_keyword: str = ""
    dingtalk_secret: str = ""

    def ensure_dirs(self) -> None:
        for path in [
            self.data_dir,
            self.report_dir,
            self.report_dir / "intraday_alerts",
            self.report_dir / "premarket",
            self.report_dir / "close",
            self.edgar_data_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def _simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition(":")
        data[key.strip()] = value.strip()
    return data


def load_settings(config_path: Path | None = None) -> Settings:
    if load_dotenv:
        load_dotenv(ROOT / ".env")

    config_file = config_path or ROOT / "config.yaml"
    raw: dict[str, Any] = {}
    if config_file.exists():
        text = config_file.read_text(encoding="utf-8")
        raw = yaml.safe_load(text) if yaml else _simple_yaml(text)

    settings = Settings()
    settings.watchlist_file = ROOT / raw.get("watchlist_file", "watchlist.txt")
    settings.report_dir = ROOT / raw.get("report_dir", "reports")
    settings.output_language = raw.get("output_language", "zh")
    settings.data_dir = ROOT / "data"
    settings.sqlite_path = settings.data_dir / "news_history.sqlite3"
    settings.ticker_meta_path = settings.data_dir / "ticker_meta.json"
    settings.portfolio_file = ROOT / raw.get("portfolio_file", "portfolio.yaml")
    settings.portfolio_text_file = ROOT / raw.get("portfolio_text_file", "portfolio.txt")
    settings.portfolio_text_state_path = settings.data_dir / "portfolio_text_state.json"
    settings.trade_log_path = settings.data_dir / "trades.jsonl"
    settings.finnhub_api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    settings.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    settings.deepseek_model_analyze = os.getenv("DEEPSEEK_MODEL_ANALYZE", settings.deepseek_model_analyze).strip()
    settings.deepseek_model_summary = os.getenv("DEEPSEEK_MODEL_SUMMARY", settings.deepseek_model_summary).strip()
    settings.poll_interval_seconds = int(os.getenv("POLL_INTERVAL_SECONDS", settings.poll_interval_seconds))
    tickertick_news_interval = raw.get(
        "tickertick_company_news_interval_seconds", settings.tickertick_company_news_interval_seconds
    )
    min_request_interval = raw.get(
        "finnhub_min_request_interval_seconds", settings.finnhub_min_request_interval_seconds
    )
    settings.tickertick_company_news_interval_seconds = int(
        os.getenv("TICKERTICK_COMPANY_NEWS_INTERVAL_SECONDS", tickertick_news_interval)
    )
    settings.finnhub_min_request_interval_seconds = float(
        os.getenv("FINNHUB_MIN_REQUEST_INTERVAL_SECONDS", min_request_interval)
    )
    settings.news_max_age_hours = int(os.getenv("NEWS_MAX_AGE_HOURS", settings.news_max_age_hours))
    settings.news_retention_hours = int(os.getenv("NEWS_RETENTION_HOURS", settings.news_retention_hours))
    settings.neutral_news_retention_hours = int(
        os.getenv("NEUTRAL_NEWS_RETENTION_HOURS", settings.neutral_news_retention_hours)
    )
    settings.db_cleanup_interval_hours = int(
        os.getenv("DB_CLEANUP_INTERVAL_HOURS", settings.db_cleanup_interval_hours)
    )
    settings.news_db_max_items = int(os.getenv("NEWS_DB_MAX_ITEMS", settings.news_db_max_items))
    settings.display_timezone = os.getenv("DISPLAY_TIMEZONE", settings.display_timezone).strip()
    settings.premarket_summary_hour = int(os.getenv("PREMARKET_SUMMARY_HOUR", settings.premarket_summary_hour))
    settings.premarket_summary_timezone = os.getenv(
        "PREMARKET_SUMMARY_TIMEZONE", settings.premarket_summary_timezone
    )
    settings.close_summary_hour = int(os.getenv("CLOSE_SUMMARY_HOUR", settings.close_summary_hour))
    settings.close_summary_minute = int(os.getenv("CLOSE_SUMMARY_MINUTE", settings.close_summary_minute))
    settings.edgar_identity = os.getenv("EDGAR_IDENTITY", "").strip()
    settings.edgar_data_dir = ROOT / "data" / "edgar"
    settings.dcf_discount_rate = float(os.getenv("DCF_DISCOUNT_RATE", settings.dcf_discount_rate))
    settings.dingtalk_webhook_url = os.getenv("DINGTALK_WEBHOOK_URL", "").strip()
    settings.dingtalk_keyword = os.getenv("DINGTALK_KEYWORD", "").strip()
    settings.dingtalk_secret = os.getenv("DINGTALK_SECRET", "").strip()
    settings.ensure_dirs()
    return settings
