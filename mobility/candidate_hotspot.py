from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import numpy as np

from simulation.coverage import CoverageModel


@dataclass
class CandidateHotspot:
    hotspot_id: int
    position: float
    covered_vehicle_ids: List[int]
    covered_vehicle_count: int
    potential_cache_gain: float
    dominant_contents: List[int]
    demand_summary: Dict[int, int]

    def to_dict(self) -> dict:
        return asdict(self)


class CandidateHotspotGenerator:
    """Generate circular-road hotspots by cache service value, not just vehicle density."""

    def __init__(
        self,
        road_length: float,
        grid_step: float,
        mrsu_radius: float,
        mrsu_cache_capacity: int,
        candidate_count: int = 8,
    ):
        self.road_length = road_length
        self.grid_step = grid_step
        self.mrsu_radius = mrsu_radius
        self.mrsu_cache_capacity = mrsu_cache_capacity
        self.candidate_count = candidate_count

    def generate(
        self,
        vehicle_positions: Iterable[float],
        vehicle_demands: Dict[int, Counter],
    ) -> List[CandidateHotspot]:
        positions = np.arange(0.0, self.road_length, self.grid_step)
        candidates: List[CandidateHotspot] = []

        for idx, z in enumerate(positions):
            covered = CoverageModel.covered_by_point(
                vehicle_positions,
                z,
                self.mrsu_radius,
                self.road_length,
            )
            demand = Counter()
            for vehicle_id in covered:
                demand.update(vehicle_demands.get(vehicle_id, Counter()))

            top_items = demand.most_common(max(1, self.mrsu_cache_capacity))
            potential_gain = float(sum(count for _, count in top_items))
            dominant = [int(content_id) for content_id, _ in top_items[:10]]
            summary = {int(content_id): int(count) for content_id, count in top_items[:10]}
            candidates.append(
                CandidateHotspot(
                    hotspot_id=idx,
                    position=float(z),
                    covered_vehicle_ids=covered,
                    covered_vehicle_count=len(covered),
                    potential_cache_gain=potential_gain,
                    dominant_contents=dominant,
                    demand_summary=summary,
                )
            )

        candidates.sort(
            key=lambda h: (h.potential_cache_gain, h.covered_vehicle_count),
            reverse=True,
        )
        selected = candidates[: self.candidate_count]
        for new_id, hotspot in enumerate(selected):
            hotspot.hotspot_id = new_id
        return selected
