from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from caching.cache_repair import CacheRepair
from caching.greedy_cache import greedy_cooperative_cache
from simulation.config import MRSUSimulationConfig
from simulation.metrics import evaluate_cache_hit_ratio


class CacheUpdateEvaluator:
    """Cache-update tool for the LLM tool-agent.

    The LLM decides whether to update each RSU cache. This evaluator maps
    predicted regional demand, selected-hotspot demand, and cache-miss signals
    into complete RSU cache lists. Cross-RSU duplicate caching is allowed.
    """

    def __init__(
        self,
        repair: CacheRepair,
        global_top_contents: List[int],
        config: MRSUSimulationConfig,
        global_popularity: Counter = None,
        valid_content_ids: Iterable[int] = None,
        update_candidate_limit: int = 30,
    ):
        self.repair = repair
        self.global_top_contents = [int(x) for x in global_top_contents]
        self.config = config
        self.global_popularity = Counter(global_popularity or {})
        self.valid_content_ids = set(
            int(x)
            for x in (valid_content_ids or range(1, self.config.movie_num + 1))
        )
        self.update_candidate_limit = int(update_candidate_limit)

    def update_with_acr_tool(
        self,
        vehicle_requests: Dict[int, List[int]],
        coverage,
        selected_hotspot: dict,
        content_features: List[dict],
        fit_summary: dict,
        update_mrsu: bool,
        update_frsu: bool,
        current_mrsu_cache: List[int],
        current_frsu_cache: List[int],
    ) -> Tuple[List[int], List[int], dict]:
        mrsu_candidates = self.mrsu_update_candidates(selected_hotspot, content_features, fit_summary)
        frsu_candidates = self.frsu_update_candidates(content_features, fit_summary)
        if update_mrsu:
            mrsu_cache = self._repair_single_cache(
                mrsu_candidates,
                self.config.mrsu_cache_capacity,
                selected_hotspot.get("dominant_contents", []) if selected_hotspot else [],
            )
        else:
            mrsu_cache = self._repair_single_cache(
                current_mrsu_cache,
                self.config.mrsu_cache_capacity,
                selected_hotspot.get("dominant_contents", []) if selected_hotspot else [],
            )
        if update_frsu:
            frsu_cache = self._repair_single_cache(
                frsu_candidates,
                self.config.frsu_cache_capacity,
                [],
            )
        else:
            frsu_cache = self._repair_single_cache(
                current_frsu_cache,
                self.config.frsu_cache_capacity,
                [],
            )
        details = {
            "tool": "DemandAwareCooperativeCache",
            "update_mrsu": bool(update_mrsu),
            "update_frsu": bool(update_frsu),
            "mrsu_candidate_head": mrsu_candidates[:20],
            "frsu_candidate_head": frsu_candidates[:20],
            "policy": "direct_priority_rebuild",
            "mrsu_priority_order": [
                "current_mrsu_covered_predicted_demand",
                "current_mrsu_predicted_missing_contents",
                "current_mrsu_covered_vehicle_feedback",
                "selected_hotspot_lookahead_demand",
                "global_popularity_fallback",
            ],
            "frsu_priority_order": [
                "mrsu_uncovered_or_weakly_covered_predicted_demand",
                "frsu_predicted_missing_contents",
                "frsu_all_road_predicted_demand",
                "global_popularity_fallback",
            ],
        }
        return mrsu_cache, frsu_cache, details

    def mrsu_update_candidates(
        self,
        selected_hotspot: dict,
        content_features: List[dict],
        fit_summary: dict,
    ) -> List[int]:
        candidates: List[int] = []
        candidates.extend(
            int(item["content_id"])
            for item in sorted(
                [
                    item
                    for item in content_features
                    if int(item.get("mrsu_group_popularity", 0)) > 0
                    or int(item.get("mrsu_only_popularity", 0)) > 0
                ],
                key=lambda x: (
                    int(x.get("mrsu_group_popularity", 0)),
                    int(x.get("mrsu_only_popularity", 0)),
                    int(x.get("global_popularity", 0)),
                ),
                reverse=True,
            )
        )
        candidates.extend(
            int(item["content_id"])
            for item in fit_summary.get("mrsu_top_missing_contents", [])
        )
        candidates.extend(
            int(item["content_id"])
            for item in fit_summary.get("mrsu_covered_vehicle_feedback_contents", [])
        )
        candidates.extend(_sorted_keys_by_value(selected_hotspot.get("demand_summary") or {}))
        candidates.extend(int(x) for x in selected_hotspot.get("dominant_contents", []))
        candidates.extend(self.global_top_contents)
        return _unique_valid(candidates, self.valid_content_ids)

    def frsu_update_candidates(self, content_features: List[dict], fit_summary: dict) -> List[int]:
        candidates: List[int] = []
        # Prioritize vehicles covered by the fixed RSU but not the mobile RSU,
        # so fRSU complements the moving hotspot service when their regions differ.
        candidates.extend(
            int(item["content_id"])
            for item in sorted(
                [
                    item
                    for item in content_features
                    if int(item.get("frsu_only_popularity", 0)) > 0
                ],
                key=lambda x: (
                    int(x.get("frsu_only_popularity", 0)),
                    int(x.get("frsu_group_popularity", 0)),
                    int(x.get("global_popularity", 0)),
                ),
                reverse=True,
            )
        )
        candidates.extend(
            int(item["content_id"])
            for item in fit_summary.get("frsu_top_missing_contents", [])
        )
        candidates.extend(
            int(item["content_id"])
            for item in sorted(
                [
                    item
                    for item in content_features
                    if int(item.get("frsu_group_popularity", 0)) > 0
                ],
                key=lambda x: (
                    int(x.get("frsu_group_popularity", 0)),
                    int(x.get("global_popularity", 0)),
                ),
                reverse=True,
            )
        )
        candidates.extend(self.global_top_contents)
        return _unique_valid(candidates, self.valid_content_ids)

    def build_acr_cache_fit_analysis(
        self,
        current_mrsu_cache: List[int],
        current_frsu_cache: List[int],
        vehicle_requests: Dict[int, List[int]],
        coverage,
        selected_hotspot: dict,
        content_features: List[dict],
        fit_summary: dict,
    ) -> dict:
        keep_mrsu = self._repair_single_cache(
            current_mrsu_cache,
            self.config.mrsu_cache_capacity,
            selected_hotspot.get("dominant_contents", []) if selected_hotspot else [],
        )
        keep_frsu = self._repair_single_cache(
            current_frsu_cache,
            self.config.frsu_cache_capacity,
            [],
        )
        update_mrsu, update_frsu, _ = self.simulate_acr_update(
            keep_mrsu,
            keep_frsu,
            vehicle_requests,
            coverage,
            selected_hotspot,
            content_features,
            fit_summary,
            update_mrsu=True,
            update_frsu=True,
        )
        keep_metrics = evaluate_cache_hit_ratio(
            vehicle_requests,
            coverage.mrsu_covered,
            coverage.frsu_covered,
            keep_mrsu,
            keep_frsu,
        )
        update_metrics = evaluate_cache_hit_ratio(
            vehicle_requests,
            coverage.mrsu_covered,
            coverage.frsu_covered,
            update_mrsu,
            update_frsu,
        )
        dominant = set(int(x) for x in selected_hotspot.get("dominant_contents", [])[:20]) if selected_hotspot else set()
        current_cache = set(keep_mrsu).union(keep_frsu)
        overlap = len(dominant.intersection(current_cache))
        return {
            "estimated_keep_chr": keep_metrics.chr,
            "estimated_tool_update_chr": update_metrics.chr,
            "estimated_acr_update_chr": update_metrics.chr,
            "estimated_gain": update_metrics.chr - keep_metrics.chr,
            "dominant_content_overlap": overlap,
            "dominant_content_count": len(dominant),
            "dominant_content_overlap_ratio": overlap / len(dominant) if dominant else 0.0,
            "keep_mrsu_hit": keep_metrics.mrsu_hit_count,
            "keep_frsu_hit": keep_metrics.frsu_hit_count,
            "tool_update_mrsu_hit": update_metrics.mrsu_hit_count,
            "tool_update_frsu_hit": update_metrics.frsu_hit_count,
            "acr_update_mrsu_hit": update_metrics.mrsu_hit_count,
            "acr_update_frsu_hit": update_metrics.frsu_hit_count,
            "mrsu_candidate_head": self.mrsu_update_candidates(selected_hotspot, content_features, fit_summary)[:20],
            "frsu_candidate_head": self.frsu_update_candidates(content_features, fit_summary)[:20],
            "cache_update_tool": "DemandAwareCooperativeCache",
        }

    def simulate_acr_update(
        self,
        current_mrsu_cache: List[int],
        current_frsu_cache: List[int],
        vehicle_requests: Dict[int, List[int]],
        coverage,
        selected_hotspot: dict,
        content_features: List[dict],
        fit_summary: dict,
        update_mrsu: bool,
        update_frsu: bool,
    ) -> Tuple[List[int], List[int], dict]:
        mrsu_candidates = self.mrsu_update_candidates(selected_hotspot, content_features, fit_summary)
        frsu_candidates = self.frsu_update_candidates(content_features, fit_summary)
        if update_mrsu:
            repaired_mrsu = self._repair_single_cache(
                mrsu_candidates,
                self.config.mrsu_cache_capacity,
                selected_hotspot.get("dominant_contents", []) if selected_hotspot else [],
            )
        else:
            repaired_mrsu = self._repair_single_cache(
                current_mrsu_cache,
                self.config.mrsu_cache_capacity,
                selected_hotspot.get("dominant_contents", []) if selected_hotspot else [],
            )
        if update_frsu:
            repaired_frsu = self._repair_single_cache(
                frsu_candidates,
                self.config.frsu_cache_capacity,
                [],
            )
        else:
            repaired_frsu = self._repair_single_cache(
                current_frsu_cache,
                self.config.frsu_cache_capacity,
                [],
            )
        details = {
            "mrsu_candidate_head": mrsu_candidates[:20],
            "frsu_candidate_head": frsu_candidates[:20],
            "tool": "DemandAwareCooperativeCache",
        }
        return repaired_mrsu, repaired_frsu, details

    def greedy_repaired_cache(
        self,
        vehicle_requests: Dict[int, List[int]],
        mrsu_covered: List[int],
        frsu_covered: List[int],
        selected_hotspot: dict = None,
    ) -> Tuple[List[int], List[int]]:
        mrsu_cache, frsu_cache = greedy_cooperative_cache(
            vehicle_requests=vehicle_requests,
            mrsu_covered=mrsu_covered,
            frsu_covered=frsu_covered,
            mrsu_capacity=self.config.mrsu_cache_capacity,
            frsu_capacity=self.config.frsu_cache_capacity,
            global_top_contents=self.global_top_contents,
        )
        local_fallback = selected_hotspot.get("dominant_contents", []) if selected_hotspot else []
        return self.repair.repair(
            mrsu_cache,
            frsu_cache,
            self.config.mrsu_cache_capacity,
            self.config.frsu_cache_capacity,
            local_fallback=local_fallback,
        )

    def build_cache_fit_analysis(
        self,
        current_mrsu_cache: List[int],
        current_frsu_cache: List[int],
        vehicle_requests: Dict[int, List[int]],
        coverage,
        selected_hotspot: dict,
    ) -> dict:
        keep_mrsu, keep_frsu = self.repair.repair(
            current_mrsu_cache,
            current_frsu_cache,
            self.config.mrsu_cache_capacity,
            self.config.frsu_cache_capacity,
            local_fallback=selected_hotspot.get("dominant_contents", []) if selected_hotspot else [],
        )
        greedy_mrsu, greedy_frsu = self.greedy_repaired_cache(
            vehicle_requests=vehicle_requests,
            mrsu_covered=coverage.mrsu_covered,
            frsu_covered=coverage.frsu_covered,
            selected_hotspot=selected_hotspot,
        )
        keep_metrics = evaluate_cache_hit_ratio(
            vehicle_requests,
            coverage.mrsu_covered,
            coverage.frsu_covered,
            keep_mrsu,
            keep_frsu,
        )
        update_metrics = evaluate_cache_hit_ratio(
            vehicle_requests,
            coverage.mrsu_covered,
            coverage.frsu_covered,
            greedy_mrsu,
            greedy_frsu,
        )
        dominant = set(int(x) for x in selected_hotspot.get("dominant_contents", [])[:20]) if selected_hotspot else set()
        current_cache = set(keep_mrsu).union(keep_frsu)
        overlap = len(dominant.intersection(current_cache))
        return {
            "estimated_keep_chr": keep_metrics.chr,
            "estimated_update_chr": update_metrics.chr,
            "estimated_gain": update_metrics.chr - keep_metrics.chr,
            "dominant_content_overlap": overlap,
            "dominant_content_count": len(dominant),
            "dominant_content_overlap_ratio": overlap / len(dominant) if dominant else 0.0,
            "keep_mrsu_hit": keep_metrics.mrsu_hit_count,
            "keep_frsu_hit": keep_metrics.frsu_hit_count,
            "update_mrsu_hit": update_metrics.mrsu_hit_count,
            "update_frsu_hit": update_metrics.frsu_hit_count,
        }

    def _repair_single_cache(
        self,
        cache_items: Sequence[int],
        capacity: int,
        local_fallback: Sequence[int],
    ) -> List[int]:
        repaired = _unique_valid(cache_items, self.valid_content_ids)[:capacity]
        for content_id in list(local_fallback or []) + self.global_top_contents:
            if len(repaired) >= capacity:
                break
            content_id = int(content_id)
            if content_id in repaired or content_id not in self.valid_content_ids:
                continue
            repaired.append(content_id)
        return repaired[:capacity]


def _sorted_keys_by_value(mapping: Dict) -> List[int]:
    return [
        int(key)
        for key, _ in sorted(
            mapping.items(),
            key=lambda item: int(item[1]),
            reverse=True,
        )
    ]


def _unique_valid(items: Iterable[int], valid_content_ids: Iterable[int]) -> List[int]:
    valid = set(int(x) for x in valid_content_ids)
    seen = set()
    result = []
    for item in items:
        try:
            content_id = int(item)
        except (TypeError, ValueError):
            continue
        if content_id in seen or content_id not in valid:
            continue
        seen.add(content_id)
        result.append(content_id)
    return result
