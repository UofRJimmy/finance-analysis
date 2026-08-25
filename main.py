from __future__ import annotations

import builtins
import logging

from src.config import load_settings
from src.dingtalk import DingTalkNotifier


def main() -> None:
    # Keep third-party scheduler noise out of the terminal after the agent is stable.
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    try:
        print("Agent 正在加载数据与技术指标模块...", flush=True)
        # Delay the heavier pandas/technical imports until after the startup message is visible.
        from src.scheduler import run_agent

        settings = load_settings()
        _install_dingtalk_print_tee(
            settings.dingtalk_webhook_url,
            settings.dingtalk_keyword,
            settings.dingtalk_secret,
        )
        run_agent(settings)
    except Exception as exc:
        # A fatal startup error must remain visible even though routine INFO logs are disabled.
        print(f"Agent 启动失败: {type(exc).__name__}: {exc}", flush=True)
        raise


def _install_dingtalk_print_tee(webhook_url: str, keyword: str = "", secret: str = "") -> None:
    notifier = DingTalkNotifier(webhook_url, keyword, secret)
    if not notifier.enabled:
        return
    original_print = builtins.print

    def tee_print(*args, **kwargs):
        original_print(*args, **kwargs)
        try:
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            message = sep.join(str(arg) for arg in args) + end
            notifier.send_text(message.rstrip())
        except Exception:
            pass

    builtins.print = tee_print


if __name__ == "__main__":
    main()
