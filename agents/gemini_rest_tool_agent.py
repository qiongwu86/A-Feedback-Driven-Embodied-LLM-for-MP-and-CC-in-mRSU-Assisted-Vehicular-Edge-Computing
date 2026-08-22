import json
import time
from typing import Any, Dict

import requests

from agents.llm_tool_agent import BaseToolAgent, LLMToolAgent, ToolDecision, _extract_json_object


class GeminiRestToolAgent(BaseToolAgent):
    """Gemini-native REST adapter for gateways using x-goog-api-key."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.gptsapi.net/v1beta",
        model_name: str = "gemini-3.5-flash",
        max_context_chars: int = 24000,
        max_retries: int = 3,
        retry_sleep_seconds: float = 3.0,
        timeout_seconds: float = 120.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model_name
        self.max_retries = int(max_retries)
        self.retry_sleep_seconds = float(retry_sleep_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self._prompt_builder = LLMToolAgent(
            api_key=api_key or "prompt-builder-only",
            model_name=model_name,
            max_context_chars=max_context_chars,
        )

    def decide_tools(self, context: Dict[str, Any]) -> ToolDecision:
        prompt = self._prompt_builder._build_prompt(context)
        data = self._call_json(prompt)
        return ToolDecision(
            selected_hotspot_id=int(data.get("selected_hotspot_id", 0)),
            lambda_smooth=0.0,
            update_mrsu_cache=bool(data.get("update_mrsu_cache", data.get("cache_update", True))),
            update_frsu_cache=bool(data.get("update_frsu_cache", data.get("cache_update", True))),
            reason=str(data.get("reason", ""))[:300],
        )

    def _endpoint(self) -> str:
        if ":generateContent" in self.base_url:
            return self.base_url
        if self.base_url.endswith("/v1beta"):
            return f"{self.base_url}/models/{self.model}:generateContent"
        return f"{self.base_url}/v1beta/models/{self.model}:generateContent"

    def _call_json(self, prompt: str) -> dict:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "topP": 1,
            },
        }
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self._endpoint(),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key,
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
                return _extract_json_object(_gemini_text_from_response(response.json()))
            except (requests.exceptions.RequestException, RuntimeError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep_seconds * (attempt + 1))
        raise RuntimeError(f"Gemini REST request failed: {last_error}") from last_error


def _gemini_text_from_response(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts = [str(part.get("text", "")) for part in parts if isinstance(part, dict)]
    return "\n".join(text for text in texts if text)
