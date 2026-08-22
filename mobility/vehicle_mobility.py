from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class VehicleState:
    vehicle_id: int
    user_id: int
    position: float
    speed: float
    platoon_id: int = -1


class OneDimensionalVehicleMobility:
    """Simple single-direction circular-road mobility model.

    Vehicles move on [0, road_length). When they pass the road end, they wrap
    around by periodic boundary condition to keep a stable density.
    """

    def __init__(
        self,
        road_length: float,
        vehicle_num: int,
        user_ids: List[int],
        min_speed: float,
        max_speed: float,
        rng: np.random.Generator,
        speed_noise_std: float = 0.0,
        platoon_cluster_count: int = 0,
        platoon_cluster_std: float = 0.0,
        platoon_speed_std: float = 1.0,
    ):
        self.road_length = road_length
        self.vehicle_num = vehicle_num
        self.user_ids = user_ids
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.rng = rng
        self.speed_noise_std = max(0.0, float(speed_noise_std))
        self.platoon_cluster_count = max(0, int(platoon_cluster_count))
        self.platoon_cluster_std = max(0.0, float(platoon_cluster_std))
        self.platoon_speed_std = max(0.0, float(platoon_speed_std))
        self.vehicles = self._init_vehicles()

    def _init_vehicles(self) -> List[VehicleState]:
        cluster_count = min(self.platoon_cluster_count, self.vehicle_num)
        if cluster_count >= 2 and self.platoon_cluster_std > 0:
            centers = self.rng.uniform(0.0, self.road_length, size=cluster_count)
            platoon_ids = np.arange(self.vehicle_num, dtype=int) % cluster_count
            self.rng.shuffle(platoon_ids)
            positions = (
                centers[platoon_ids]
                + self.rng.normal(0.0, self.platoon_cluster_std, size=self.vehicle_num)
            ) % self.road_length
            platoon_speeds = self.rng.uniform(self.min_speed, self.max_speed, size=cluster_count)
            speeds = platoon_speeds[platoon_ids] + self.rng.normal(
                0.0,
                self.platoon_speed_std,
                size=self.vehicle_num,
            )
            speeds = np.clip(speeds, self.min_speed, self.max_speed)
        else:
            positions = np.linspace(0, self.road_length, self.vehicle_num, endpoint=False)
            self.rng.shuffle(positions)
            speeds = self.rng.uniform(self.min_speed, self.max_speed, size=self.vehicle_num)
            platoon_ids = np.full(self.vehicle_num, -1, dtype=int)
        vehicles = []
        for vehicle_id in range(self.vehicle_num):
            user_id = self.user_ids[vehicle_id % len(self.user_ids)]
            vehicles.append(
                VehicleState(
                    vehicle_id=vehicle_id,
                    user_id=int(user_id),
                    position=float(positions[vehicle_id]),
                    speed=float(speeds[vehicle_id]),
                    platoon_id=int(platoon_ids[vehicle_id]),
                )
            )
        return vehicles

    def step(self, dt: float) -> None:
        for vehicle in self.vehicles:
            if self.speed_noise_std > 0:
                vehicle.speed = float(
                    np.clip(
                        vehicle.speed + self.rng.normal(0.0, self.speed_noise_std),
                        self.min_speed,
                        self.max_speed,
                    )
                )
            vehicle.position = (vehicle.position + vehicle.speed * dt) % self.road_length

    def positions(self) -> np.ndarray:
        return np.array([v.position for v in self.vehicles], dtype=float)

    def speeds(self) -> np.ndarray:
        return np.array([v.speed for v in self.vehicles], dtype=float)

    def user_to_vehicle(self) -> dict:
        return {v.user_id: v.vehicle_id for v in self.vehicles}
