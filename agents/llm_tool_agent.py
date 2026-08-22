import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict


DEFAULT_API_KEY_ENV_VARS = (
    "DASHSCOPE_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "GPTSAPI_API_KEY",
)


@dataclass
class ToolDecision:
    selected_hotspot_id: int
    lambda_smooth: float
    update_mrsu_cache: bool
    update_frsu_cache: bool
    reason: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["lambda_smooth_source"] = "system_auto_rule"
        return data

    @property
    def cache_update(self) -> bool:
        return self.update_mrsu_cache or self.update_frsu_cache


class BaseToolAgent:
    """Agent that decides tool calls instead of directly producing cache IDs."""

    def decide_tools(self, context: Dict[str, Any]) -> ToolDecision:
        raise NotImplementedError


class LLMToolAgent(BaseToolAgent):
    """DashScope-compatible tool-calling planner for mRSU caching.

    The LLM chooses a hotspot and whether each RSU cache should be updated.
    It does not generate cache content IDs or tune low-level path parameters.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "qwen3-vl-flash",
        max_context_chars: int = 24000,
        api_key_env: str = "",
    ):
        self.api_key_env = api_key_env
        self.api_key = api_key or _read_api_key(api_key_env)
        self.mock_mode = not self.api_key
        self.model = model_name
        self.base_url = base_url
        self.max_context_chars = max_context_chars
        self._client = None

    def decide_tools(self, context: Dict[str, Any]) -> ToolDecision:
        prompt = self._build_prompt(context)
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
                "No API key is set. Use MockToolAgent or set one of: "
                + ", ".join(DEFAULT_API_KEY_ENV_VARS)
            )
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _call_json(self, prompt: str) -> dict:
        client = self._client_or_raise()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
        )
        return _extract_json_object(response.choices[0].message.content)

    def _compact_context(self, context: Dict[str, Any]) -> str:
        text = json.dumps(context, ensure_ascii=False)
        if len(text) <= self.max_context_chars:
            return text
        return text[: self.max_context_chars] + "\n...TRUNCATED..."

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        path_planning_disabled = bool(context.get("path_planning_disabled", False))
        planner_tool_text = (
            "- QPPathPlanner: disabled in this ablation. The mRSU stays fixed, so selected_hotspot_id "
            "is only a demand-reference signal and will not trigger movement.\n"
            if path_planning_disabled
            else "- QPPathPlanner: generates the physically feasible mRSU movement after you choose a hotspot. "
            "The path-smoothness parameter is computed automatically by the system rule, not by you.\n"
        )
        hotspot_principles = (
            "- Path planning is disabled. Focus on whether the fixed mRSU/fRSU caches should be updated under the current static coverage.\n"
            "- selected_hotspot_id is diagnostic only in this ablation; it will not move the mRSU.\n"
            if path_planning_disabled
            else "- Prefer cache-service hotspots with high potential_cache_gain and clear dominant_contents.\n"
            "- Consider distance_to_mrsu as forward circular travel distance and current mRSU speed; do not chase a far hotspot blindly.\n"
            "- If not_covered_count is high, prioritize request-active vehicles that are not covered.\n"
        )
        return (
            "You are an embodied tool-using mRSU planner in a one-dimensional circular-road VEC caching system.\n\n"
            "Your role is high-level embodied decision making. Do not generate movie/content cache IDs. "
            "You must decide where the mRSU should track and whether the system should update the mRSU cache "
            "and/or the fRSU cache.\n\n"
            "Road topology: the road is a closed circular one-way road with periodic boundary conditions. "
            "Vehicles and the mRSU move along the fixed travel direction. Coverage uses shortest circular distance, "
            "while movement reachability uses forward circular distance; distance_to_mrsu is this forward distance.\n\n"
            "Important information timing: all demand, hotspot, cache-fit, and feedback fields "
            "in the compact CONTEXT are decision-time prediction signals built from MovieLens training history and "
            "previous-round feedback. Current-round true requests are hidden from you and are used only "
            "after your decision for evaluation and next-round feedback.\n\n"
            "Available tools:\n"
            f"{planner_tool_text}"
            "- DemandAwareCooperativeCacheTool: if you set an update flag, the system directly rebuilds the corresponding RSU cache. "
            "The mRSU cache prioritizes the current post-movement coverage region's predicted demand, predicted missing contents, "
            "and covered-vehicle feedback before selected-hotspot lookahead demand. "
            "For fRSU, demand from vehicles not covered or weakly covered by mRSU is prioritized before all-road demand.\n\n"
            "Important: you only decide selected_hotspot_id, update_mrsu_cache, and update_frsu_cache. "
            "Do not output cache lists.\n\n"
            "Hotspot decision principles:\n"
            f"{hotspot_principles}\n"
            "Cache update decision principles:\n"
            "- Use cache_fit_analysis. estimated_keep_chr is the predicted CHR if current caches are kept.\n"
            "- estimated_tool_update_chr is the predicted CHR if both caches are updated by DemandAwareCooperativeCacheTool.\n"
            "- If estimated_tool_update_chr is meaningfully higher than estimated_keep_chr, update the relevant cache.\n"
            "- update_mrsu_cache should be true when selected_hotspot demand or mRSU-side missing contents are poorly covered by current mRSU cache.\n"
            "- update_frsu_cache should be true when fRSU-side demand or fRSU-side missing contents are poorly covered by current fRSU cache.\n"
            "- If current caches already fit demand, keep them to avoid unnecessary replacement.\n\n"
            "Return strict JSON only. Do not output markdown or extra text.\n"
            "Required JSON format:\n"
            "{\n"
            '  "selected_hotspot_id": <integer>,\n'
            '  "update_mrsu_cache": <boolean>,\n'
            '  "update_frsu_cache": <boolean>,\n'
            '  "reason": "<brief reason>"\n'
            "}\n\n"
            f"CONTEXT:\n{self._compact_context(context)}"
        )


def _extract_json_object(text: str) -> dict:
    if not text:
        return {}
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _read_api_key(api_key_env: str = "") -> str:
    if api_key_env:
        return os.getenv(api_key_env, "")
    for env_name in DEFAULT_API_KEY_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    return ""
