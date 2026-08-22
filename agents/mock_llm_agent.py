from typing import Any, Dict, List

from agents.llm_embodied_agent import BaseLLMEmbodiedAgent, CacheDecision, HotspotDecision


class MockLLMEmbodiedAgent(BaseLLMEmbodiedAgent):
    """Deterministic drop-in replacement for the real embodied LLM agent."""

    def select_hotspot(self, context: Dict[str, Any]) -> HotspotDecision:
        candidates = context.get("candidate_hotspots", [])
        if not candidates:
            return HotspotDecision(0, 1.0, "no candidate hotspot available")

        last_chr = float(context.get("last_round_hit_ratio", 0.0) or 0.0)
        selected = max(
            candidates,
            key=lambda item: (
                float(item.get("potential_cache_gain", 0.0)),
                int(item.get("covered_vehicle_count", 0)),
            ),
        )
        lambda_smooth = 0.6 if last_chr < 0.25 else 1.2
        return HotspotDecision(
            selected_hotspot_id=int(selected.get("hotspot_id", 0)),
            lambda_smooth=lambda_smooth,
            reason="selected the candidate with highest potential cache gain",
        )

    def decide_cache(self, context: Dict[str, Any]) -> CacheDecision:
        items = context.get("candidate_contents", [])
        mrsu_scored = []
        frsu_scored = []
        for item in items:
            content_id = int(item["content_id"])
            global_popularity = 0.05 * float(item.get("global_popularity", 0))
            mrsu_score = 3.0 * float(item.get("mrsu_group_popularity", 0)) + global_popularity
            frsu_score = 3.0 * float(item.get("frsu_group_popularity", 0)) + global_popularity
            mrsu_scored.append((mrsu_score, content_id))
            frsu_scored.append((frsu_score, content_id))
        mrsu_scored.sort(reverse=True)
        frsu_scored.sort(reverse=True)
        return CacheDecision(
            mrsu_cache_list=_unique([content_id for _, content_id in mrsu_scored]),
            frsu_cache_list=_unique([content_id for _, content_id in frsu_scored]),
            reason="generated independent mRSU/fRSU cache lists from covered-group demand and global popularity",
        )


def _unique(items: List[int]) -> List[int]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
