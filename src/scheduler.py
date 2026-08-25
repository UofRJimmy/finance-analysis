from __future__ import annotations

import signal
import sys
import threading
import time
from zoneinfo import ZoneInfo

from .close_summary import generate_close_summary
from .config import Settings
from .interactive import InteractiveAssistant
from .market_calendar import is_us_market_session, market_closed_reason
from .pipeline import AgentPipeline
from .premarket_summary import generate_summary


try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except Exception:
    BackgroundScheduler = None
    CronTrigger = None


class ExecutionCoordinator:
    """将新闻、盘前和盘后分析串行化，并让定时总结优先于新新闻批次。"""

    def __init__(self):
        self._analysis_lock = threading.Lock()
        self._summary_pending = threading.Event()

    def run_news(self, callback) -> bool:
        if self._summary_pending.is_set():
            return False
        with self._analysis_lock:
            # 总结可能在新闻等待锁期间到点；此时让总结先执行。
            if self._summary_pending.is_set():
                return False
            callback()
            return True

    def run_summary(self, callback):
        self._summary_pending.set()
        try:
            with self._analysis_lock:
                return callback()
        finally:
            self._summary_pending.clear()


def run_agent(settings: Settings) -> None:
    assistant = InteractiveAssistant(settings)
    try:
        assistant.prepare_portfolio()
    except Exception as exc:
        # 配置读取失败不应阻止新闻监控启动，但必须把原因明确显示出来。
        print(f"启动资产配置读取失败: {type(exc).__name__}: {exc}", flush=True)
    pipeline = AgentPipeline(settings)
    coordinator = ExecutionCoordinator()
    stop_event = threading.Event()
    _install_signal_handlers(stop_event)
    scheduler = None
    try:
        # Poll serially: each new interval starts only after the previous run has finished.
        threading.Thread(
            target=_poll_loop,
            args=(settings, pipeline, coordinator, stop_event),
            daemon=True,
            name="news-poll-loop",
        ).start()

        if BackgroundScheduler is not None:
            scheduler = _build_scheduler(settings, coordinator)
            scheduler.start()

        print(
            f"Agent 已启动：监控 {len(pipeline.watchlist.get())} 个标的，"
            f"只分析最近 {settings.news_max_age_hours} 小时内的新新闻。",
            flush=True,
        )
        if sys.stdin.isatty():
            assistant.run(stop_event)
        else:
            _wait_for_stop(stop_event)
    finally:
        stop_event.set()
        if scheduler is not None:
            scheduler.shutdown(wait=False)
        pipeline.close()


def _build_scheduler(settings: Settings, coordinator: ExecutionCoordinator | None = None):
    coordinator = coordinator or ExecutionCoordinator()
    scheduler = BackgroundScheduler()
    # 盘前、盘后是独立通道；协调器确保它们不会与新闻分析同时调用模型。
    scheduler.add_job(
        _run_summary_job,
        CronTrigger(
            timezone=ZoneInfo(settings.premarket_summary_timezone),
            hour=settings.premarket_summary_hour,
            minute=0,
        ),
        args=(coordinator, "盘前", generate_summary),
        id="premarket_summary",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    scheduler.add_job(
        _run_summary_job,
        CronTrigger(
            timezone=ZoneInfo(settings.premarket_summary_timezone),
            hour=settings.close_summary_hour,
            minute=settings.close_summary_minute,
        ),
        args=(coordinator, "盘后", generate_close_summary),
        id="close_summary",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    return scheduler


def _run_summary_job(coordinator: ExecutionCoordinator, name: str, callback) -> None:
    try:
        if not is_us_market_session():
            print(f"{name}报告跳过：{market_closed_reason()}，等开市再生成。", flush=True)
            return
        result = coordinator.run_summary(lambda: callback(display=True))
        if isinstance(result, str) and "生成失败" in result:
            print(result, flush=True)
            return
        print(result, flush=True)
    except Exception as exc:
        print(f"[{name}报告] 整理失败: {type(exc).__name__}: {exc}", flush=True)


def _poll_loop(
    settings: Settings,
    pipeline: AgentPipeline,
    coordinator: ExecutionCoordinator,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            ran = coordinator.run_news(pipeline.run_once)
        except Exception as exc:
            print(f"[news-poll-loop] 本轮执行失败，将继续下一轮: {type(exc).__name__}: {exc}", flush=True)
            stop_event.wait(min(30, settings.poll_interval_seconds))
            continue
        if not ran:
            # 总结在等待或执行时，新闻仍由下一轮继续抓取，不与总结争抢模型额度。
            stop_event.wait(min(5, settings.poll_interval_seconds))
            continue
        stop_event.wait(settings.poll_interval_seconds)


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def handle_stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)


def _wait_for_stop(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        time.sleep(0.5)
