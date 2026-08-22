import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class HotspotDecision:
    selected_hotspot_id: int
    lambda_smooth: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CacheDecision:
    mrsu_cache_list: List[int]
    frsu_cache_list: List[int]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def content_priority_list(self) -> List[int]:
        return self.mrsu_cache_list + self.frsu_cache_list


@dataclass
class EmbodiedDecision:
    hotspot_decision: HotspotDecision
    cache_decision: CacheDecision


class BaseLLMEmbodiedAgent:
    """Common interface for real and mock embodied LLM agents."""

    def select_hotspot(self, context: Dict[str, Any]) -> HotspotDecision:
        raise NotImplementedError

    def decide_cache(self, context: Dict[str, Any]) -> CacheDecision:
        raise NotImplementedError


class LLMEmbodiedAgent(BaseLLMEmbodiedAgent):
    """DashScope-compatible LLM agent.

    The same interface is used by MockLLMEmbodiedAgent, so experiments can run
    without an API key and later switch to real LLM calls.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "qwen3-vl-flash",
        max_context_chars: int = 24000,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.mock_mode = not self.api_key
        self.model = model_name
        self.base_url = base_url
        self.max_context_chars = max_context_chars
        self._client = None

    def _client_or_raise(self):
        if self.mock_mode:
            raise RuntimeError("DASHSCOPE_API_KEY is not set; use MockLLMEmbodiedAgent or set the env var.")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def select_hotspot(self, context: Dict[str, Any]) -> HotspotDecision:
        prompt = self._build_hotspot_prompt(context)
        data = self._call_json(prompt)
        lambda_smooth = float(data.get("lambda_smooth", 1.0))
        lambda_smooth = min(5.0, max(0.1, lambda_smooth))
        return HotspotDecision(
            selected_hotspot_id=int(data.get("selected_hotspot_id", 0)),
            lambda_smooth=lambda_smooth,
            reason=str(data.get("reason", ""))[:240],
        )

    def decide_cache(self, context: Dict[str, Any]) -> CacheDecision:
        prompt = self._build_cache_prompt(context)
        data = self._call_json(prompt)
        mrsu_cache = data.get("mrsu_cache_list", data.get("mrsu_cache", []))
        frsu_cache = data.get("frsu_cache_list", data.get("frsu_cache", []))
        if not mrsu_cache and not frsu_cache and data.get("content_priority_list"):
            priority = [int(x) for x in data.get("content_priority_list", []) if _is_int_like(x)]
            mrsu_capacity = int(context.get("mrsu_cache_capacity", 0))
            frsu_capacity = int(context.get("frsu_cache_capacity", 0))
            mrsu_cache = priority[:mrsu_capacity]
            frsu_cache = priority[mrsu_capacity : mrsu_capacity + frsu_capacity]
        return CacheDecision(
            mrsu_cache_list=[int(x) for x in mrsu_cache if _is_int_like(x)],
            frsu_cache_list=[int(x) for x in frsu_cache if _is_int_like(x)],
            reason=str(data.get("reason", ""))[:240],
        )

    def _call_json(self, prompt: str) -> dict:
        client = self._client_or_raise()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
        )
        content = response.choices[0].message.content
        return _extract_json_object(content)

    def _compact_context(self, context: Dict[str, Any]) -> str:
        text = json.dumps(context, ensure_ascii=False)
        if len(text) <= self.max_context_chars:
            return text
        return text[: self.max_context_chars] + "\n...TRUNCATED..."

    def _build_hotspot_prompt(self, context: Dict[str, Any]) -> str:
        return (
            "You are an embodied mRSU decision agent in a one-dimensional circular-road vehicular edge caching system.\n\n"
            "Your task is to select the cache-service hotspot that the mRSU should track in this round. "
            "The goal is to improve cache hit ratio, not simply to move toward the region with the most vehicles.\n\n"
            "Road topology: the road is a closed circular one-way road with periodic boundary conditions. "
            "Coverage uses shortest circular distance, while movement reachability uses forward circular distance.\n\n"
            "Information timing: all demand-related fields in CONTEXT are prediction signals from MovieLens training history "
            "and previous-round feedback. Current-round true requests are hidden until after your decision and are used only "
            "for evaluation and next-round feedback.\n\n"
            "The mRSU is a physical mobile node. It cannot jump to a hotspot directly. "
            "Its motion is generated by a QP path planner.\n\n"
            "Motion model used by the QP planner:\n"
            "- Position update: x_m(t+1) = (x_m(t) + v_m(t) * dt) mod road_length.\n"
            "- The default time step is dt = 1.0 s.\n"
            "- Speed constraint: v_min <= v_m(t) <= v_max.\n"
            "- Acceleration constraint: a_min <= (v_m(t+1) - v_m(t)) / dt <= a_max.\n"
            "- The QP planner predicts a short future trajectory, but the simulator only executes the first step in the current round. "
            "The next round will perceive the environment and replan again.\n\n"
            "QP objective:\n"
            "minimize tracking error to the selected hotspot and velocity variation:\n"
            "sum_t (x_m(t) - c)^2 + lambda_smooth * sum_t (v_m(t) - v_m(t-1))^2\n\n"
            "where c is the selected hotspot position.\n\n"
            "Meaning of lambda_smooth:\n"
            "- Smaller lambda_smooth: the mRSU tracks the selected hotspot more aggressively.\n"
            "- Larger lambda_smooth: the mRSU moves more smoothly and avoids abrupt speed changes.\n"
            "- Use a small value when the hotspot has high cache-service value and is reasonably reachable.\n"
            "- Use a large value when the hotspot is far away, uncertain, or when the current service state is already stable.\n"
            "- Recommended range: 0.1 to 5.0.\n\n"
            "Selection principles:\n"
            "- Prefer hotspots with high potential_cache_gain and clear dominant_contents.\n"
            "- Prefer hotspots that can serve request-active vehicles, not merely many vehicles.\n"
            "- Consider the current mRSU position and speed. A faraway hotspot with high gain may not be useful if it cannot be tracked effectively.\n"
            "- If not_covered_count is high, prefer a hotspot that improves coverage of request-active vehicles.\n"
            "- If not_cached_count is high, prefer a hotspot whose dominant contents can be well served by mRSU caching.\n"
            "- Do not blindly choose the hotspot with the largest covered_vehicle_count or largest potential_cache_gain.\n\n"
            "Return strict JSON only. Do not output markdown or extra text.\n\n"
            "Required JSON format:\n"
            "{\n"
            '  "selected_hotspot_id": <integer>,\n'
            '  "lambda_smooth": <float>,\n'
            '  "reason": "<brief reason>"\n'
            "}\n\n"
            "Field definitions:\n"
            "- candidate_hotspots: candidate positions that the mRSU may track.\n"
            "- potential_cache_gain: predicted cache-hit benefit if the mRSU tracks this hotspot. "
            "It is computed from predicted request demand of vehicles covered by this hotspot.\n"
            "- dominant_contents: the most predicted-requested content IDs among vehicles covered by this hotspot.\n"
            "- demand_summary: predicted request counts of dominant contents in this hotspot.\n"
            "- distance_to_mrsu: forward circular travel distance from the current mRSU position to this hotspot.\n"
            "- not_covered_count: missed requests caused by vehicles not being covered by either mRSU or fRSU.\n"
            "- not_cached_count: missed requests where the vehicle was covered by at least one RSU, but the requested content was not cached there.\n"
            "- mbs_miss_count: requests finally served by MBS/Cloud because edge caches missed them.\n"
            "- last_round_missed_contents: popular contents that were missed in the previous round.\n"
            "- cache_fit_summary: estimated match between current caches and predicted near-future requests.\n\n"
            "Decision hints:\n"
            "- If not_covered_count is high, coverage is the main bottleneck. Prefer hotspots that cover more request-active vehicles.\n"
            "- If not_cached_count is high, cache-content mismatch is the main bottleneck. Prefer hotspots with clear dominant_contents and high potential_cache_gain.\n"
            "- Do not choose a hotspot only because it covers the most vehicles.\n"
            "- Do not choose a faraway hotspot only because it has the highest potential_cache_gain.\n\n"
            f"CONTEXT:\n{self._compact_context(context)}"
        )

    def _build_hotspot_prompt_legacy(self, context: Dict[str, Any]) -> str:
        return (
            "You are an embodied mRSU agent in a one-dimensional circular-road VEC caching system. "
            "Select the cache-service hotspot that the mRSU should track this round. "
            "Use candidate_hotspots and feedback, not only vehicle density. "
            "Return JSON only with keys: selected_hotspot_id, lambda_smooth, reason.\n"
            f"CONTEXT:\n{self._compact_context(context)}"
        )

    def _build_cache_prompt(self, context: Dict[str, Any]) -> str:
        mrsu_capacity = int(context["mrsu_cache_capacity"])
        frsu_capacity = int(context["frsu_cache_capacity"])
        return (
            "You are an embodied LLM cooperative caching agent. "
            "Your goal is to maximize near-future cache hit ratio using predicted demand, not hidden current-round true requests. "
            "Generate two ordered cache lists directly: mrsu_cache_list for the mobile RSU and frsu_cache_list for the fixed RSU. "
            "The two RSUs may cache the same movie/content because they serve different vehicle groups and locations, "
            "but do not blindly duplicate content across RSUs. Cross-RSU duplication is useful only when the same content has strong demand "
            "in both the mRSU-served area and the fRSU-served area. "
            "Each RSU's own cache list must not contain duplicate content IDs. "
            "For mRSU, prioritize mrsu_group_popularity, selected_hotspot dominant_contents, requests from vehicles currently or soon covered by mRSU, "
            "popular missed contents in the mRSU area, and use global_popularity only as a tie-breaker. "
            "For fRSU, prioritize frsu_group_popularity, fRSU-only vehicle demand, requests from vehicles not covered by mRSU, "
            "popular missed contents in the fRSU area, and use global_popularity only as a tie-breaker. "
            "If a content is mainly requested by overlap vehicles and is already in mrsu_cache_list, avoid putting it in frsu_cache_list "
            "unless it is also popular among fRSU-only vehicles. "
            "Output order matters: earlier content IDs are higher cache priority. "
            f"mrsu_cache_list must contain at least {mrsu_capacity} integer content IDs; "
            f"frsu_cache_list must contain at least {frsu_capacity} integer content IDs. "
            "If you cannot fully satisfy the requested length, the simulator will repair and fill with candidate contents plus global Top-K, "
            "but you should output enough IDs whenever possible. "
            "Return strict JSON only, no markdown and no extra text. "
            "Required JSON keys: mrsu_cache_list, frsu_cache_list, reason.\n\n"
            "Information timing: candidate demand statistics in CONTEXT are prediction signals from MovieLens training history "
            "and previous-round feedback. Current-round true requests are hidden until evaluation.\n\n"
            "Field definitions:\n"
            "- selected_hotspot: the hotspot selected in the previous decision step. It represents the region that the mRSU is expected to serve.\n"
            "- dominant_contents: the most predicted-requested content IDs among vehicles covered by the selected hotspot.\n"
            "- demand_summary: predicted request counts of dominant contents in the selected hotspot. It should guide mRSU caching for the hotspot region.\n"
            "- candidate_contents: candidate content items for caching decisions. Each item contains global popularity and region-specific demand statistics for mRSU and fRSU.\n"
            "- global_popularity: historical/global popularity of a content item. Use it mainly as a fallback or tie-breaker, not as the only caching criterion.\n"
            "- mrsu_group_popularity: predicted request strength of this content among vehicles currently served or expected to be served by mRSU.\n"
            "- frsu_group_popularity: predicted request strength of this content among vehicles served by fRSU.\n"
            "- mrsu_only_popularity: request strength from vehicles covered only by mRSU. A high value means the content is especially suitable for mRSU.\n"
            "- frsu_only_popularity: request strength from vehicles covered only by fRSU. A high value means the content is especially suitable for fRSU.\n"
            "- overlap_popularity: request strength from vehicles covered by both mRSU and fRSU. Use it to decide whether cross-RSU duplicate caching is necessary.\n"
            "- last_round_missed_contents: popular contents missed in the previous round.\n"
            "- cache_fit_summary: estimated match between current caches and current or near-future requests.\n"
            "- not_covered_count: missed requests caused by vehicles not being covered by either mRSU or fRSU.\n"
            "- not_cached_count: missed requests where the vehicle was covered by at least one RSU, but the requested content was not cached in the available RSU cache.\n"
            "- mbs_miss_count: requests finally served by MBS/Cloud because edge caches missed them.\n\n"
            "Decision hints:\n"
            "- mRSU should prioritize selected_hotspot.dominant_contents, selected_hotspot.demand_summary, mrsu_group_popularity, and mrsu_only_popularity.\n"
            "- fRSU should prioritize frsu_group_popularity and frsu_only_popularity.\n"
            "- Cross-RSU duplicate caching is allowed, but only when both mRSU-side and fRSU-side demand are strong.\n"
            "- If overlap_popularity is high, avoid wasting both caches on the same content unless it also has strong mRSU-only or fRSU-only demand.\n"
            "- If not_cached_count is high, the main problem is cache-content mismatch. Prioritize contents with strong local demand.\n"
            "- If not_covered_count is high, the main problem is spatial coverage. For caching, focus on contents requested by vehicles in the selected hotspot and covered regions.\n\n"
            f"CONTEXT:\n{self._compact_context(context)}"
        )

    def _build_cache_prompt_legacy(self, context: Dict[str, Any]) -> str:
        mrsu_capacity = int(context["mrsu_cache_capacity"])
        frsu_capacity = int(context["frsu_cache_capacity"])
        return (
            "You are an embodied LLM cooperative caching agent. "
            "Generate two cache lists independently: one for the mobile RSU (mRSU) and one for the fixed RSU (fRSU). "
            "The two RSUs may cache the same movie/content because they serve different locations and vehicle groups. "
            "Do not intentionally repeat the same content ID inside one RSU's own cache list. "
            "Prefer mRSU-covered vehicle demand for mrsu_cache_list and fRSU-covered vehicle demand for frsu_cache_list. "
            f"Return JSON only with keys: mrsu_cache_list, frsu_cache_list, reason. "
            f"mrsu_cache_list should contain at least {mrsu_capacity} integer content IDs; "
            f"frsu_cache_list should contain at least {frsu_capacity} integer content IDs.\n"
            f"CONTEXT:\n{self._compact_context(context)}"
        )


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def _extract_json_object(text: str) -> dict:
    if not text:
        return {}
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
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
