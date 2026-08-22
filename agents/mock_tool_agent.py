from typing import Any, Dict

from agents.llm_tool_agent import BaseToolAgent, ToolDecision


class MockToolAgent(BaseToolAgent):
    """Deterministic tool-agent used when no LLM API key is available."""

    def __init__(self, update_gain_threshold: float = 0.015):
        self.update_gain_threshold = update_gain_threshold

    def decide_tools(self, context: Dict[str, Any]) -> ToolDecision:
        candidates = context.get("candidate_hotspots", [])
        if candidates:
            selected = max(
                candidates,
                key=lambda item: (
                    float(item.get("potential_cache_gain", 0.0)),
                    int(item.get("covered_vehicle_count", 0)),
                ),
            )
            selected_hotspot_id = int(selected.get("hotspot_id", 0))
        else:
            selected_hotspot_id = 0

        fit = context.get("cache_fit_analysis", {})
        estimated_gain = float(fit.get("estimated_gain", 0.0))
        mrsu_gain = float(fit.get("acr_update_mrsu_hit", 0)) - float(fit.get("keep_mrsu_hit", 0))
        frsu_gain = float(fit.get("acr_update_frsu_hit", 0)) - float(fit.get("keep_frsu_hit", 0))
        update_mrsu_cache = estimated_gain >= self.update_gain_threshold and mrsu_gain >= 0
        update_frsu_cache = estimated_gain >= self.update_gain_threshold and frsu_gain >= 0

        reason = (
            f"estimated_gain={estimated_gain:.4f}; "
            f"threshold={self.update_gain_threshold:.4f}; "
            f"update_mrsu_cache={update_mrsu_cache}; "
            f"update_frsu_cache={update_frsu_cache}"
        )
        return ToolDecision(
            selected_hotspot_id=selected_hotspot_id,
            lambda_smooth=0.0,
            update_mrsu_cache=update_mrsu_cache,
            update_frsu_cache=update_frsu_cache,
            reason=reason,
        )
