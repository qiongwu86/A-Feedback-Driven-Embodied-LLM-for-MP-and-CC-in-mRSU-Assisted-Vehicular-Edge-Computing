import os
import time
from typing import Any, Dict

from agents.llm_tool_agent import BaseToolAgent, LLMToolAgent, ToolDecision, _extract_json_object


GEMINI_API_KEY_ENV_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


class GeminiToolAgent(BaseToolAgent):
    """Official Google Gemini API adapter for the tool-agent interface."""

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.5-pro",
        max_context_chars: int = 24000,
        api_key_env: str = "",
        max_retries: int = 3,
        retry_sleep_seconds: float = 3.0,
    ):
        self.api_key_env = api_key_env
        self.api_key = api_key or _read_gemini_api_key(api_key_env)
        self.mock_mode = not self.api_key
        self.model = model_name
        self.max_context_chars = max_context_chars
        self.max_retries = int(max_retries)
        self.retry_sleep_seconds = float(retry_sleep_seconds)
        self._client = None
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

    def _client_or_raise(self):
        if self.mock_mode:
            raise RuntimeError(
                "No Gemini API key is set. Set GEMINI_API_KEY or GOOGLE_API_KEY, "
                "or pass --api-key-env with your key variable name."
            )
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "Official Gemini mode requires the google-genai package. "
                    "Install it with: pip install -U google-genai"
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _call_json(self, prompt: str) -> dict:
        client = self._client_or_raise()
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": 0, "top_p": 1},
                )
                return _extract_json_object(getattr(response, "text", "") or "")
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep_seconds * (attempt + 1))
        raise last_error


def _read_gemini_api_key(api_key_env: str = "") -> str:
    if api_key_env:
        return os.getenv(api_key_env, "")
    for env_name in GEMINI_API_KEY_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    return ""
