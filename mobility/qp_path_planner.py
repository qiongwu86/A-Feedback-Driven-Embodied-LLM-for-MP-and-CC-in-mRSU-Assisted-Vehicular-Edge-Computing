from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class PathPlan:
    positions: List[float]
    velocities: List[float]
    target_position: float
    lambda_smooth: float
    solver: str
    status: str


def auto_lambda_smooth(
    current_position: float,
    target_position: float,
    potential_cache_gain: float,
    road_length: float,
    request_budget: float,
    default_lambda: float = 1.0,
    min_lambda: float = 0.1,
    max_lambda: float = 5.0,
) -> float:
    """Compute a deterministic path-smoothness weight for the selected hotspot.

    On the circular road, reachability is measured by forward travel distance:
    high-value nearby hotspots should be tracked more aggressively, while
    faraway or low-value targets should be approached more smoothly.
    """

    distance_ratio = min(
        1.0,
        _forward_distance(current_position, target_position, road_length) / max(float(road_length), 1e-9),
    )
    gain_ratio = min(1.0, max(0.0, float(potential_cache_gain)) / max(float(request_budget), 1.0))
    value = float(default_lambda) * (1.2 + 1.3 * distance_ratio - 1.0 * gain_ratio)
    return float(np.clip(value, float(min_lambda), float(max_lambda)))


def _normalize_position(position: float, road_length: float) -> float:
    return float(position) % max(float(road_length), 1e-9)


def _forward_distance(start: float, target: float, road_length: float) -> float:
    length = max(float(road_length), 1e-9)
    return (_normalize_position(target, length) - _normalize_position(start, length)) % length


class QPPathPlanner:
    """Smooth one-way circular-road mRSU path planner with cvxpy/OSQP fallback."""

    def __init__(
        self,
        road_length: float,
        dt: float,
        horizon: int,
        v_min: float,
        v_max: float,
        a_min: float,
        a_max: float,
    ):
        self.road_length = road_length
        self.dt = dt
        self.horizon = horizon
        self.v_min = v_min
        self.v_max = v_max
        self.a_min = a_min
        self.a_max = a_max

    def plan(
        self,
        current_position: float,
        current_speed: float,
        target_position: float,
        lambda_smooth: float,
    ) -> PathPlan:
        current_position = _normalize_position(current_position, self.road_length)
        target_position = _normalize_position(target_position, self.road_length)
        try:
            return self._plan_with_cvxpy(
                current_position=current_position,
                current_speed=current_speed,
                target_position=target_position,
                lambda_smooth=lambda_smooth,
            )
        except Exception as exc:
            fallback = self._plan_fallback(
                current_position=current_position,
                current_speed=current_speed,
                target_position=target_position,
                lambda_smooth=lambda_smooth,
            )
            fallback.status = f"fallback: {exc}"
            return fallback

    def _target_sequence(self, current_position: float, target_position: float) -> np.ndarray:
        target_unwrapped = float(current_position) + _forward_distance(
            current_position,
            target_position,
            self.road_length,
        )
        return np.full(self.horizon + 1, target_unwrapped, dtype=float)

    def _plan_with_cvxpy(
        self,
        current_position: float,
        current_speed: float,
        target_position: float,
        lambda_smooth: float,
    ) -> PathPlan:
        import cvxpy as cp

        x = cp.Variable(self.horizon + 1)
        v = cp.Variable(self.horizon)
        target = self._target_sequence(current_position, target_position)

        tracking = cp.sum_squares(x[1:] - target[1:])
        smooth_terms = [cp.square(v[0] - current_speed)]
        if self.horizon > 1:
            smooth_terms.append(cp.sum_squares(v[1:] - v[:-1]))
        objective = cp.Minimize(tracking + lambda_smooth * cp.sum(smooth_terms))

        constraints = [x[0] == current_position]
        for t in range(self.horizon):
            constraints.append(x[t + 1] == x[t] + v[t] * self.dt)
            constraints.append(v[t] >= self.v_min)
            constraints.append(v[t] <= self.v_max)
        for t in range(self.horizon - 1):
            acceleration = (v[t + 1] - v[t]) / self.dt
            constraints.append(acceleration >= self.a_min)
            constraints.append(acceleration <= self.a_max)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if x.value is None or v.value is None:
            raise RuntimeError(f"OSQP failed with status {problem.status}")

        return PathPlan(
            positions=[_normalize_position(float(value), self.road_length) for value in x.value.tolist()],
            velocities=[float(value) for value in v.value.tolist()],
            target_position=float(target_position),
            lambda_smooth=float(lambda_smooth),
            solver="cvxpy.OSQP",
            status=str(problem.status),
        )

    def _plan_fallback(
        self,
        current_position: float,
        current_speed: float,
        target_position: float,
        lambda_smooth: float,
    ) -> PathPlan:
        unwrapped_positions = [float(current_position)]
        velocities: List[float] = []
        speed = float(current_speed)
        smooth_scale = max(1.0, float(lambda_smooth))
        target_unwrapped = float(current_position) + _forward_distance(
            current_position,
            target_position,
            self.road_length,
        )

        for _ in range(self.horizon):
            distance = target_unwrapped - unwrapped_positions[-1]
            desired_speed = np.clip(distance / max(self.dt, 1e-6), self.v_min, self.v_max)
            max_delta = max(abs(self.a_min), abs(self.a_max)) * self.dt / smooth_scale
            speed_delta = np.clip(desired_speed - speed, -max_delta, max_delta)
            speed = float(np.clip(speed + speed_delta, self.v_min, self.v_max))
            next_position = float(unwrapped_positions[-1] + speed * self.dt)
            velocities.append(speed)
            unwrapped_positions.append(next_position)

        return PathPlan(
            positions=[_normalize_position(position, self.road_length) for position in unwrapped_positions],
            velocities=velocities,
            target_position=float(target_position),
            lambda_smooth=float(lambda_smooth),
            solver="heuristic",
            status="fallback",
        )
