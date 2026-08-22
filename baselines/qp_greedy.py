from typing import Dict, List, Tuple

from caching.greedy_cache import greedy_cooperative_cache
from mobility.candidate_hotspot import CandidateHotspotGenerator
from mobility.qp_path_planner import QPPathPlanner, PathPlan


def select_best_hotspot(candidate_hotspots: List[dict]) -> dict:
    if not candidate_hotspots:
        return {"position": 0.0, "hotspot_id": 0}
    return max(
        candidate_hotspots,
        key=lambda item: (
            float(item.get("potential_cache_gain", 0.0)),
            int(item.get("covered_vehicle_count", 0)),
        ),
    )


def qp_greedy_decision(
    planner: QPPathPlanner,
    candidate_hotspots: List[dict],
    current_position: float,
    current_speed: float,
    lambda_smooth: float,
) -> Tuple[dict, PathPlan]:
    hotspot = select_best_hotspot(candidate_hotspots)
    plan = planner.plan(
        current_position=current_position,
        current_speed=current_speed,
        target_position=float(hotspot["position"]),
        lambda_smooth=lambda_smooth,
    )
    return hotspot, plan


def qp_greedy_cache(
    vehicle_requests: Dict[int, List[int]],
    mrsu_covered: List[int],
    frsu_covered: List[int],
    mrsu_capacity: int,
    frsu_capacity: int,
    global_top_contents: List[int],
) -> Tuple[List[int], List[int]]:
    return greedy_cooperative_cache(
        vehicle_requests=vehicle_requests,
        mrsu_covered=mrsu_covered,
        frsu_covered=frsu_covered,
        mrsu_capacity=mrsu_capacity,
        frsu_capacity=frsu_capacity,
        global_top_contents=global_top_contents,
    )

