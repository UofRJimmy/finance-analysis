from __future__ import annotations

import base64
import hashlib
import hmac
import threading
import time
from typing import Any
from urllib.parse import quote_plus

try:
    import requests
except Exception:
    requests = None


class DingTalkNotifier:
    def __init__(self, webhook_url: str = "", keyword: str = "", secret: str = "", timeout_seconds: int = 10):
        self.webhook_url = webhook_url.strip()
        self.keyword = keyword.strip()
        self.secret = secret.strip()
        self.timeout_seconds = timeout_seconds
        self._local = threading.local()

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url and requests)

    def send_text(self, content: str, title: str | None = None) -> None:
        if not self.enabled:
            return
        text = str(content or "").strip()
        if not text:
            return
        if title:
            text = f"{title}\n{text}"
        text = self._with_keyword(text)
        for chunk in _chunks(text, 3500):
            self._post({"msgtype": "text", "text": {"content": chunk}})

    def send_markdown(self, title: str, content: str) -> None:
        if not self.enabled:
            return
        title = str(title or "Finance Agent")
        text = str(content or "").strip()
        if not text:
            return
        text = self._with_keyword(text)
        for index, chunk in enumerate(_chunks(text, 3500), 1):
            chunk_title = title if index == 1 else f"{title} ({index})"
            self._post({"msgtype": "markdown", "markdown": {"title": chunk_title, "text": chunk}})

    def _with_keyword(self, text: str) -> str:
        if self.keyword and self.keyword not in text:
            return f"{self.keyword}\n{text}"
        return text

    def _post(self, payload: dict[str, Any]) -> None:
        if getattr(self._local, "sending", False):
            return
        self._local.sending = True
        try:
            requests.post(self._signed_url(), json=payload, timeout=self.timeout_seconds)
        except Exception:
            # Do not let notification failures interrupt trading/news analysis.
            pass
        finally:
            self._local.sending = False

    def _signed_url(self) -> str:
        if not self.secret:
            return self.webhook_url
        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self.secret}".encode("utf-8")
        digest = hmac.new(self.secret.encode("utf-8"), string_to_sign, hashlib.sha256).digest()
        sign = quote_plus(base64.b64encode(digest))
        separator = "&" if "?" in self.webhook_url else "?"
        return f"{self.webhook_url}{separator}timestamp={timestamp}&sign={sign}"


def _chunks(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    return [text[index:index + size] for index in range(0, len(text), size)]
