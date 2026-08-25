from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


DEEPSEEK_RESPONSES_URL = "https://api.deepseek.com/responses"


@dataclass
class DeepSeekResponse:
    """Small wrapper used by the existing agents' text extraction helpers."""

    output_text: str
    output: list[dict[str, Any]]


class _ResponsesClient:
    """Provides the existing `client.responses.create` interface over DeepSeek Responses."""

    def __init__(self, client: "DeepSeekClient"):
        self._client = client

    def create(
        self,
        *,
        model: str,
        input: str | list[dict[str, Any]],
        text: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        reasoning: dict[str, Any] | None = None,
        **_: Any,
    ) -> DeepSeekResponse:
        return self._client.create(
            model=model,
            input=input,
            text=text,
            tools=tools,
            reasoning=reasoning,
        )


class DeepSeekClient:
    """Minimal client for DeepSeek's official Responses API endpoint."""

    def __init__(self, api_key: str, timeout_seconds: int = 60):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.responses = _ResponsesClient(self)

    def create(
        self,
        *,
        model: str,
        input: str | list[dict[str, Any]],
        text: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> DeepSeekResponse:
        payload: dict[str, Any] = {"model": model, "input": input, "store": False}
        if text:
            payload["text"] = text
        if tools:
            payload["tools"] = tools
        if reasoning:
            payload["reasoning"] = reasoning

        response = requests.post(
            DEEPSEEK_RESPONSES_URL,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=(10, self.timeout_seconds),
        )
        if not response.ok:
            raise RuntimeError(f"DeepSeek API request failed ({response.status_code}): {_error_detail(response)}")
        data = response.json()
        output = data.get("output") or []
        output_text = _extract_output_text(data)
        if not output_text:
            raise RuntimeError(f"DeepSeek API returned no output text: {data}")
        return DeepSeekResponse(output_text=output_text, output=output)


def make_client(api_key: str) -> DeepSeekClient | None:
    return DeepSeekClient(api_key=api_key) if api_key else None


def parse_json_text(text: str) -> dict[str, Any]:
    text = text.lstrip("\ufeff").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if len(lines) >= 2 else lines[1:]).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if not text:
        raise ValueError("DeepSeek returned an empty structured response")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # Some providers prepend a short status line despite JSON-schema mode.
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start : end + 1])
        preview = text[:160].replace("\n", " ")
        raise ValueError(f"DeepSeek returned invalid structured JSON: {preview!r}") from exc


def response_text(response: Any) -> str:
    return str(getattr(response, "output_text", "") or "")


def _extract_output_text(data: dict[str, Any]) -> str:
    direct = data.get("output_text")
    if direct and str(direct).strip():
        return str(direct).strip()
    fragments: list[str] = []
    for item in data.get("output") or []:
        for content in item.get("content") or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                fragments.append(str(content["text"]))
    return "\n".join(fragments).strip()


def _error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error", payload)
        return str(error.get("message") or error) if isinstance(error, dict) else str(error)
    except ValueError:
        return response.text[:500]
