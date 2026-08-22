from typing import List, Tuple

import numpy as np

from mobility.qp_path_planner import QPPathPlanner, PathPlan
from simulation.coverage import CoverageModel


def random_cache(
    valid_content_ids: List[int],
    capacity: int,
    rng: np.random.Generator,
    fallback_content_ids: List[int] = None,
) -> List[int]:
    """Sample one RSU cache without duplicates inside the RSU.

    Different RSUs should call this function independently, so cross-RSU
    duplicate content is allowed.
    """

    pool = _unique_ints(valid_content_ids + list(fallback_content_ids or []))
    if not pool or capacity <= 0:
        return []
    sample_size = min(capacity, len(pool))
    selected = rng.choice(pool, size=sample_size, replace=False).tolist()
    return [int(item) for item in selected]


def random_target_position(
    candidate_hotspots: List[dict],
    road_length: float,
    rng: np.random.Generator,
) -> float:
    if candidate_hotspots:
        idx = int(rng.integers(0, len(candidate_hotspots)))
        return float(candidate_hotspots[idx].get("position", 0.0))
    return float(rng.uniform(0.0, road_length))


def random_path_decision(
    planner: QPPathPlanner,
    candidate_hotspots: List[dict],
    current_position: float,
    current_speed: float,
    road_length: float,
    rng: np.random.Generator,
    lambda_smooth: float = 1.0,
) -> Tuple[dict, PathPlan]:
    target_position = random_target_position(candidate_hotspots, road_length, rng)
    selected_hotspot = _nearest_hotspot(candidate_hotspots, target_position, road_length)
    if selected_hotspot is None:
        selected_hotspot = {
            "hotspot_id": None,
            "position": target_position,
            "covered_vehicle_ids": [],
            "covered_vehicle_count": 0,
            "potential_cache_gain": 0.0,
            "dominant_contents": [],
            "demand_summary": {},
        }
    plan = planner.plan(
        current_position=current_position,
        current_speed=current_speed,
        target_position=target_position,
        lambda_smooth=lambda_smooth,
    )
    return selected_hotspot, plan


def _nearest_hotspot(candidate_hotspots: List[dict], target_position: float, road_length: float):
    if not candidate_hotspots:
        return None
    return min(
        candidate_hotspots,
        key=lambda hotspot: CoverageModel.circular_distance(
            float(hotspot.get("position", 0.0)),
            target_position,
            road_length,
        ),
    )


def _unique_ints(items: List[int]) -> List[int]:
    seen = set()
    result = []
    for item in items:
        content_id = int(item)
        if content_id in seen:
            continue
        seen.add(content_id)
        result.append(content_id)
    return result
