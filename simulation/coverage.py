from dataclasses import dataclass
from typing import Iterable, List, Set


@dataclass
class CoverageSnapshot:
    mrsu_covered: List[int]
    frsu_covered: List[int]
    overlap: List[int]


class CoverageModel:
    """One-dimensional circular-road coverage model for one mRSU and one fRSU."""

    @staticmethod
    def circular_distance(a: float, b: float, road_length: float) -> float:
        length = max(float(road_length), 1e-9)
        direct = abs((float(a) % length) - (float(b) % length))
        return min(direct, length - direct)

    @staticmethod
    def forward_distance(start: float, target: float, road_length: float) -> float:
        length = max(float(road_length), 1e-9)
        return (float(target) % length - float(start) % length) % length

    @staticmethod
    def covered_by_point(
        vehicle_positions: Iterable[float],
        center: float,
        radius: float,
        road_length: float,
    ) -> List[int]:
        return [
            idx
            for idx, pos in enumerate(vehicle_positions)
            if CoverageModel.circular_distance(float(pos), center, road_length) <= radius
        ]

    @classmethod
    def snapshot(
        cls,
        vehicle_positions: Iterable[float],
        mrsu_position: float,
        mrsu_radius: float,
        frsu_position: float,
        frsu_radius: float,
        road_length: float,
    ) -> CoverageSnapshot:
        mrsu = cls.covered_by_point(vehicle_positions, mrsu_position, mrsu_radius, road_length)
        frsu = cls.covered_by_point(vehicle_positions, frsu_position, frsu_radius, road_length)
        overlap: Set[int] = set(mrsu).intersection(frsu)
        return CoverageSnapshot(
            mrsu_covered=mrsu,
            frsu_covered=frsu,
            overlap=sorted(overlap),
        )
