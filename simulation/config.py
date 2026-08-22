from dataclasses import dataclass
from typing import Optional


@dataclass
class MRSUSimulationConfig:
    """Centralized parameters for the one-dimensional circular-road mRSU/fRSU scenario."""

    seed: int = 42
    rounds: int = 100
    road_length: float = 1000.0
    vehicle_num: int = 50
    user_num: int = 50
    movie_num: int = 2000
    request_min: int = 670
    request_max: int = 690
    decision_interval: int = 10

    min_vehicle_speed: float = 12.0
    max_vehicle_speed: float = 20.0
    vehicle_speed_noise_std: float = 1.0
    platoon_cluster_count: int = 4
    platoon_cluster_std: float = 35.0
    platoon_speed_std: float = 1.0
    dt: float = 1.0

    mrsu_initial_position: float = 100.0
    mrsu_initial_speed: float = 15.0
    mrsu_radius: float = 200.0
    mrsu_cache_capacity: int = 200
    mrsu_v_min: float = 0.0
    mrsu_v_max: float = 30.0
    mrsu_a_min: float = -4.0
    mrsu_a_max: float = 4.0

    frsu_position: float = 500.0
    frsu_radius: Optional[float] = None
    frsu_full_coverage: bool = False
    frsu_cache_capacity: int = 200

    grid_step: float = 50.0
    candidate_count: int = 8
    planner_horizon: int = 10
    default_lambda_smooth: float = 1.0

    request_count_smoothing: float = 0.9
    demand_profile_smoothing: float = 0.88
    prediction_feedback_weight: float = 1.0
    prediction_prior_weight: float = 0.02
    true_demand_prior_weight: float = 0.12
    prediction_noise_scale: float = 0.03
    true_demand_noise_scale: float = 0.4

    global_topk_for_prompt: int = 120
    output_dir: str = "results"

    def __post_init__(self) -> None:
        self.rounds = max(1, int(self.rounds))
        self.decision_interval = max(1, int(self.decision_interval))
        self.planner_horizon = max(int(self.planner_horizon), int(self.decision_interval))
        self.vehicle_num = max(1, int(self.vehicle_num))
        self.user_num = max(1, int(self.user_num))
        self.platoon_cluster_count = max(0, int(self.platoon_cluster_count))
        self.platoon_cluster_std = max(0.0, float(self.platoon_cluster_std))
        self.platoon_speed_std = max(0.0, float(self.platoon_speed_std))
        self.vehicle_speed_noise_std = max(0.0, float(self.vehicle_speed_noise_std))
        self.true_demand_noise_scale = max(0.0, float(self.true_demand_noise_scale))
        if self.frsu_radius is None:
            self.frsu_radius = float(self.mrsu_radius)

    @property
    def decision_rounds(self) -> int:
        return max(1, (int(self.rounds) + int(self.decision_interval) - 1) // int(self.decision_interval))
