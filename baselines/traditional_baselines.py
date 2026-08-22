from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from communication.latency_model import CV2XLatencyModel, summarize_round_latencies
from mobility.candidate_hotspot import CandidateHotspotGenerator
from mobility.qp_path_planner import QPPathPlanner, auto_lambda_smooth
from simulation.config import MRSUSimulationConfig
from simulation.environment import MRSUEnvironment
from simulation.metrics import RoundMetrics, local_rsu_chr_from_counts, summarize_metrics


METHOD_THOMPSON = "thompson_sampling"
METHOD_CPSAT = "cp_sat"

METHOD_LABELS = {
    METHOD_THOMPSON: "Thompson Sampling",
    METHOD_CPSAT: "CP-SAT",
}

METHOD_NOTES = {
    METHOD_THOMPSON: (
        "Maintains alpha/beta estimates for content and hotspot value; "
        "uses non-feedback predicted demand for decisions, applies a weakened "
        "regional-demand cache score, and updates only cached-content and hotspot "
        "bandit estimates after post-execution feedback."
    ),
    METHOD_CPSAT: (
        "Uses OR-Tools CP-SAT to jointly select one hotspot and mRSU/fRSU caches "
        "from non-feedback predicted demand; falls back to a deterministic regional-demand cache "
        "repair if the solver is unavailable or fails."
    ),
}

SUPPORTED_METHODS = (
    METHOD_THOMPSON,
    METHOD_CPSAT,
)


@dataclass
class BaselineDecision:
    selected_hotspot: dict
    mrsu_cache: List[int] | None = None
    frsu_cache: List[int] | None = None
    details: dict | None = None


class ContentHotspotThompsonController:
    """Content-level and hotspot-level Thompson Sampling controller."""

    def __init__(
        self,
        movie_num: int,
        road_length: float,
        seed: int,
        demand_scale: float = 1.0,
        global_scale: float = 0.15,
        miss_alpha_scale: float = 0.0,
    ):
        self.movie_num = int(movie_num)
        self.road_length = float(road_length)
        self.rng = np.random.default_rng(int(seed))
        self.demand_scale = float(demand_scale)
        self.global_scale = float(global_scale)
        self.miss_alpha_scale = float(miss_alpha_scale)
        self.content_alpha = np.ones(self.movie_num + 1, dtype=float)
        self.content_beta = np.ones(self.movie_num + 1, dtype=float)
        self.hotspot_alpha: Dict[str, float] = defaultdict(lambda: 1.0)
        self.hotspot_beta: Dict[str, float] = defaultdict(lambda: 1.0)

    def select_hotspot(self, candidate_hotspots: List[dict]) -> Tuple[dict, dict]:
        if not candidate_hotspots:
            fallback = _fallback_hotspot()
            return fallback, {"policy": "thompson_hotspot", "fallback": "no_candidate_hotspot"}

        sampled_rows = []
        for hotspot in candidate_hotspots:
            key = self._hotspot_key(hotspot)
            sample = float(self.rng.beta(self.hotspot_alpha[key], self.hotspot_beta[key]))
            sampled_rows.append((sample, hotspot, key))
        sample, selected, key = max(
            sampled_rows,
            key=lambda item: (
                item[0],
                float(item[1].get("potential_cache_gain", 0.0)),
                int(item[1].get("covered_vehicle_count", 0)),
            ),
        )
        return selected, {
            "policy": "thompson_hotspot",
            "selected_hotspot_key": key,
            "selected_hotspot_sample": sample,
            "hotspot_alpha": float(self.hotspot_alpha[key]),
            "hotspot_beta": float(self.hotspot_beta[key]),
        }

    def select_cache(
        self,
        demand: Counter,
        capacity: int,
        global_popularity: Counter,
        global_top_contents: Sequence[int],
    ) -> List[int]:
        if capacity <= 0:
            return []
        scores = self.rng.beta(self.content_alpha[1:], self.content_beta[1:])
        ranked: List[Tuple[float, int]] = []
        max_demand = max(demand.values(), default=0)
        max_global = max(global_popularity.values(), default=0)
        for content_id in range(1, self.movie_num + 1):
            demand_weight = 1.0
            if max_demand > 0:
                demand_weight += self.demand_scale * float(demand.get(content_id, 0)) / float(max_demand)
            if max_global > 0:
                demand_weight += self.global_scale * float(global_popularity.get(content_id, 0)) / float(max_global)
            ranked.append((float(scores[content_id - 1]) * demand_weight, content_id))
        ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return _fill_unique([content_id for _, content_id in ranked], capacity, global_top_contents)

    def update(
        self,
        selected_hotspot: dict,
        true_requests: Dict[int, List[int]],
        coverage,
        mrsu_cache: Sequence[int],
        frsu_cache: Sequence[int],
        metrics: RoundMetrics,
    ) -> dict:
        true_mrsu_demand = _demand_for_vehicles(coverage.mrsu_covered, true_requests)
        true_frsu_demand = _demand_for_vehicles(coverage.frsu_covered, true_requests)
        mrsu_cache_set = set(int(content_id) for content_id in mrsu_cache)
        frsu_cache_set = set(int(content_id) for content_id in frsu_cache)
        cached_contents = set(mrsu_cache_set).union(frsu_cache_set)

        content_update_count = 0
        for content_id in cached_contents:
            if content_id < 1 or content_id > self.movie_num:
                continue
            observed_demand = 0
            if content_id in mrsu_cache_set:
                observed_demand += int(true_mrsu_demand.get(content_id, 0))
            if content_id in frsu_cache_set:
                observed_demand += int(true_frsu_demand.get(content_id, 0))
            if observed_demand > 0:
                self.content_alpha[content_id] += float(observed_demand)
            else:
                self.content_beta[content_id] += 1.0
            content_update_count += 1

        miss_alpha_update_count = 0
        if self.miss_alpha_scale > 0:
            missed_demand = Counter()
            missed_demand.update(true_mrsu_demand)
            missed_demand.update(true_frsu_demand)
            for content_id in cached_contents:
                missed_demand.pop(int(content_id), None)
            for content_id, count in missed_demand.most_common(50):
                content_id = int(content_id)
                if 1 <= content_id <= self.movie_num:
                    self.content_alpha[content_id] += self.miss_alpha_scale * float(count)
                    miss_alpha_update_count += 1

        hotspot_key = self._hotspot_key(selected_hotspot)
        mrsu_request_count = sum(len(true_requests.get(int(vehicle_id), [])) for vehicle_id in coverage.mrsu_covered)
        hotspot_reward = float(metrics.mrsu_hit_count) / max(float(mrsu_request_count), 1.0)
        hotspot_reward = float(np.clip(hotspot_reward, 0.0, 1.0))
        self.hotspot_alpha[hotspot_key] += hotspot_reward
        self.hotspot_beta[hotspot_key] += 1.0 - hotspot_reward
        return {
            "content_update_count": int(content_update_count),
            "miss_alpha_update_count": int(miss_alpha_update_count),
            "demand_scale": float(self.demand_scale),
            "global_scale": float(self.global_scale),
            "miss_alpha_scale": float(self.miss_alpha_scale),
            "hotspot_key": hotspot_key,
            "hotspot_reward": hotspot_reward,
            "hotspot_alpha_after": float(self.hotspot_alpha[hotspot_key]),
            "hotspot_beta_after": float(self.hotspot_beta[hotspot_key]),
        }

    def _hotspot_key(self, hotspot: dict) -> str:
        position = float(hotspot.get("position", 0.0)) % max(self.road_length, 1e-9)
        return f"{position:.6f}"


def run_traditional_baseline(
    config: MRSUSimulationConfig,
    method: str,
    verbose: bool = True,
    cp_sat_time_limit: float = 5.0,
    latency_model: CV2XLatencyModel | None = None,
) -> dict:
    method = _normalize_method(method)
    env = MRSUEnvironment(config)
    valid_content_ids = list(range(1, int(config.movie_num) + 1))
    hotspot_generator = CandidateHotspotGenerator(
        road_length=config.road_length,
        grid_step=config.grid_step,
        mrsu_radius=config.mrsu_radius,
        mrsu_cache_capacity=config.mrsu_cache_capacity,
        candidate_count=config.candidate_count,
    )
    planner = QPPathPlanner(
        road_length=config.road_length,
        dt=config.dt,
        horizon=config.planner_horizon,
        v_min=config.mrsu_v_min,
        v_max=config.mrsu_v_max,
        a_min=config.mrsu_a_min,
        a_max=config.mrsu_a_max,
    )
    latency_model = latency_model or CV2XLatencyModel()
    thompson = (
        ContentHotspotThompsonController(
            movie_num=config.movie_num,
            road_length=config.road_length,
            seed=int(config.seed) + 9176,
        )
        if method == METHOD_THOMPSON
        else None
    )

    round_metrics: List[RoundMetrics] = []
    round_logs: List[dict] = []
    for round_index in range(config.decision_rounds):
        window_ticks = env.decision_window_ticks(round_index)
        decision_requests = env.predict_round_requests(round_index, use_feedback=False)
        vehicle_demands = env.vehicle_request_counters(decision_requests)
        candidate_hotspots = [
            hotspot.to_dict()
            for hotspot in hotspot_generator.generate(env.mobility.positions(), vehicle_demands)
        ]

        decision = _decide_before_motion(
            method=method,
            env=env,
            config=config,
            planner=planner,
            candidate_hotspots=candidate_hotspots,
            decision_requests=decision_requests,
            valid_content_ids=valid_content_ids,
            thompson=thompson,
            cp_sat_time_limit=cp_sat_time_limit,
            window_ticks=window_ticks,
        )
        selected_hotspot = decision.selected_hotspot
        decision_request_budget = max(1, sum(len(requests) for requests in decision_requests.values()))
        lambda_smooth = auto_lambda_smooth(
            current_position=env.mrsu.position,
            target_position=float(selected_hotspot.get("position", 0.0)),
            potential_cache_gain=float(selected_hotspot.get("potential_cache_gain", 0.0)),
            road_length=config.road_length,
            request_budget=decision_request_budget,
            default_lambda=config.default_lambda_smooth,
        )
        path_plan = planner.plan(
            current_position=env.mrsu.position,
            current_speed=env.mrsu.speed,
            target_position=float(selected_hotspot.get("position", 0.0)),
            lambda_smooth=lambda_smooth,
        )
        decision_coverage = env.project_service_window_coverage(path_plan, ticks=window_ticks)

        if decision.mrsu_cache is not None and decision.frsu_cache is not None:
            mrsu_cache = decision.mrsu_cache
            frsu_cache = decision.frsu_cache
        else:
            mrsu_cache, frsu_cache = _decide_cache_after_motion(
                method=method,
                env=env,
                config=config,
                coverage=decision_coverage,
                decision_requests=decision_requests,
                thompson=thompson,
            )

        env.set_cache(mrsu_cache, frsu_cache)
        coverage = env.execute_service_window(path_plan, ticks=window_ticks)
        true_requests = env.sample_round_requests(round_index)
        metrics = env.evaluate(true_requests, coverage, mrsu_cache, frsu_cache)
        latency = latency_model.evaluate_round(
            vehicle_requests=true_requests,
            vehicle_positions=env.mobility.positions(),
            coverage=coverage,
            mrsu_position=env.mrsu.position,
            frsu_position=env.frsu.position,
            road_length=config.road_length,
            mrsu_radius=config.mrsu_radius,
            frsu_radius=config.frsu_radius,
            mrsu_cache=mrsu_cache,
            frsu_cache=frsu_cache,
        ).to_dict()
        ts_update = None
        if thompson is not None:
            ts_update = thompson.update(
                selected_hotspot=selected_hotspot,
                true_requests=true_requests,
                coverage=coverage,
                mrsu_cache=mrsu_cache,
                frsu_cache=frsu_cache,
                metrics=metrics,
            )

        round_metrics.append(metrics)
        details = dict(decision.details or {})
        if ts_update:
            details["thompson_update"] = ts_update
        round_logs.append(
            {
                "round": int(round_index),
                "physical_tick_start": int(round_index * config.decision_interval),
                "physical_tick_end": int(round_index * config.decision_interval + window_ticks - 1),
                "decision_interval_ticks": int(window_ticks),
                "chr": float(metrics.chr),
                "round_delay_ms": float(latency.get("average_delay_ms", 0.0)),
                "latency": latency,
                "request_count": int(metrics.request_count),
                "decision_request_count": int(sum(len(requests) for requests in decision_requests.values())),
                "evaluation_request_count": int(metrics.request_count),
                "decision_request_source": "predicted_history_signal_no_proposed_feedback",
                "uses_proposed_miss_feedback": False,
                "evaluation_request_source": "current_service_window_true_requests",
                "coverage_source": "service_window_swept_coverage",
                "selected_hotspot": selected_hotspot,
                "lambda_smooth": float(lambda_smooth),
                "lambda_smooth_source": "system_auto_rule",
                "path_plan_status": path_plan.status,
                "path_plan_solver": path_plan.solver,
                "mrsu_position": float(env.mrsu.position),
                "mrsu_speed": float(env.mrsu.speed),
                "mrsu_cache": [int(content_id) for content_id in mrsu_cache],
                "frsu_cache": [int(content_id) for content_id in frsu_cache],
                "mrsu_covered": [int(vehicle_id) for vehicle_id in coverage.mrsu_covered],
                "frsu_covered": [int(vehicle_id) for vehicle_id in coverage.frsu_covered],
                "decision_mrsu_covered": [int(vehicle_id) for vehicle_id in decision_coverage.mrsu_covered],
                "decision_frsu_covered": [int(vehicle_id) for vehicle_id in decision_coverage.frsu_covered],
                "overlap": [int(vehicle_id) for vehicle_id in coverage.overlap],
                "metrics": metrics.to_dict(),
                "decision_details": details,
            }
        )
        if verbose:
            fallback_text = ""
            if method == METHOD_CPSAT and details.get("fallback_to_topk"):
                fallback_text = f" fallback={details.get('fallback_reason', '')}"
            print(
                f"[{METHOD_LABELS[method]}] round={round_index:02d} "
                f"LocalCHR={metrics.local_rsu_chr:.4f} "
                f"mRSU_hit={metrics.mrsu_hit_count} "
                f"fRSU_hit={metrics.frsu_hit_count} "
                f"MBS_miss={metrics.mbs_miss_count} "
                f"Delay={latency.get('average_delay_ms', 0.0):.2f}ms"
                f"{fallback_text}"
            )

    summary = summarize_metrics(round_metrics)
    summary.update(summarize_round_latencies(log.get("latency") for log in round_logs))
    summary.update(
        {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "method_note": METHOD_NOTES[method],
            "decision_request_source": "predicted_history_signal_no_proposed_feedback",
            "uses_proposed_miss_feedback": False,
            "physical_rounds": int(config.rounds),
            "decision_interval_ticks": int(config.decision_interval),
            "decision_rounds": int(config.decision_rounds),
            "latency_model": latency_model.config.to_dict(),
            "round_chr": [float(item.chr) for item in round_metrics],
            "round_local_rsu_chr": [
                local_rsu_chr_from_counts(item.hit_count, item.not_cached_count)
                for item in round_metrics
            ],
        }
    )
    return {"summary": summary, "round_logs": round_logs}


def _decide_before_motion(
    method: str,
    env: MRSUEnvironment,
    config: MRSUSimulationConfig,
    planner: QPPathPlanner,
    candidate_hotspots: List[dict],
    decision_requests: Dict[int, List[int]],
    valid_content_ids: List[int],
    thompson: ContentHotspotThompsonController | None,
    cp_sat_time_limit: float,
    window_ticks: int,
) -> BaselineDecision:
    if method == METHOD_THOMPSON:
        if thompson is None:
            raise RuntimeError("Thompson controller is not initialized.")
        selected, details = thompson.select_hotspot(candidate_hotspots)
        return BaselineDecision(selected_hotspot=selected, details=details)

    if method == METHOD_CPSAT:
        decision = _cp_sat_decision(
            env=env,
            config=config,
            planner=planner,
            candidate_hotspots=candidate_hotspots,
            decision_requests=decision_requests,
            valid_content_ids=valid_content_ids,
            cp_sat_time_limit=cp_sat_time_limit,
            window_ticks=window_ticks,
        )
        return decision

    raise ValueError(f"Unknown traditional baseline method: {method}")


def _decide_cache_after_motion(
    method: str,
    env: MRSUEnvironment,
    config: MRSUSimulationConfig,
    coverage,
    decision_requests: Dict[int, List[int]],
    thompson: ContentHotspotThompsonController | None,
) -> Tuple[List[int], List[int]]:
    if method == METHOD_CPSAT:
        return _topk_regional_caches(
            env=env,
            coverage=coverage,
            decision_requests=decision_requests,
            mrsu_capacity=config.mrsu_cache_capacity,
            frsu_capacity=config.frsu_cache_capacity,
        )

    if method == METHOD_THOMPSON:
        if thompson is None:
            raise RuntimeError("Thompson controller is not initialized.")
        mrsu_demand = env.demand_for_vehicles(coverage.mrsu_covered, decision_requests)
        frsu_demand = env.demand_for_vehicles(coverage.frsu_covered, decision_requests)
        return (
            thompson.select_cache(
                demand=mrsu_demand,
                capacity=config.mrsu_cache_capacity,
                global_popularity=env.global_popularity,
                global_top_contents=env.global_top_contents,
            ),
            thompson.select_cache(
                demand=frsu_demand,
                capacity=config.frsu_cache_capacity,
                global_popularity=env.global_popularity,
                global_top_contents=env.global_top_contents,
            ),
        )

    raise ValueError(f"Unknown traditional baseline method: {method}")


def _cp_sat_decision(
    env: MRSUEnvironment,
    config: MRSUSimulationConfig,
    planner: QPPathPlanner,
    candidate_hotspots: List[dict],
    decision_requests: Dict[int, List[int]],
    valid_content_ids: List[int],
    cp_sat_time_limit: float,
    window_ticks: int,
) -> BaselineDecision:
    if not candidate_hotspots:
        return BaselineDecision(
            selected_hotspot=_fallback_hotspot(),
            details={
                "policy": "cp_sat",
                "fallback_to_topk": True,
                "fallback_reason": "no_candidate_hotspot",
            },
        )

    try:
        from ortools.sat.python import cp_model
    except Exception as exc:
        return BaselineDecision(
            selected_hotspot=_best_hotspot(candidate_hotspots),
            details={
                "policy": "cp_sat",
                "fallback_to_topk": True,
                "fallback_reason": f"ortools_unavailable: {type(exc).__name__}: {exc}",
            },
        )

    estimated_rows = []
    request_budget = max(1, sum(len(requests) for requests in decision_requests.values()))
    for hotspot in candidate_hotspots:
        lambda_smooth = auto_lambda_smooth(
            current_position=env.mrsu.position,
            target_position=float(hotspot.get("position", 0.0)),
            potential_cache_gain=float(hotspot.get("potential_cache_gain", 0.0)),
            road_length=config.road_length,
            request_budget=request_budget,
            default_lambda=config.default_lambda_smooth,
        )
        plan = planner.plan(
            current_position=env.mrsu.position,
            current_speed=env.mrsu.speed,
            target_position=float(hotspot.get("position", 0.0)),
            lambda_smooth=lambda_smooth,
        )
        coverage = env.project_service_window_coverage(plan, ticks=window_ticks)
        mrsu_ids = set(int(vehicle_id) for vehicle_id in coverage.mrsu_covered)
        frsu_ids = set(int(vehicle_id) for vehicle_id in coverage.frsu_covered)
        estimated_rows.append(
            {
                "hotspot": hotspot,
                "mrsu_demand": env.demand_for_vehicles(sorted(mrsu_ids), decision_requests),
                "frsu_non_mrsu_demand": env.demand_for_vehicles(sorted(frsu_ids - mrsu_ids), decision_requests),
            }
        )

    try:
        model = cp_model.CpModel()
        z = [model.NewBoolVar(f"z_{idx}") for idx in range(len(estimated_rows))]
        x = {content_id: model.NewBoolVar(f"x_{content_id}") for content_id in valid_content_ids}
        y = {content_id: model.NewBoolVar(f"y_{content_id}") for content_id in valid_content_ids}
        model.Add(sum(z) == 1)
        model.Add(sum(x.values()) <= int(config.mrsu_cache_capacity))
        model.Add(sum(y.values()) <= int(config.frsu_cache_capacity))

        objective_terms = []
        for k, row in enumerate(estimated_rows):
            mrsu_demand = row["mrsu_demand"]
            frsu_non_mrsu_demand = row["frsu_non_mrsu_demand"]
            relevant_contents = set(mrsu_demand.keys()).union(frsu_non_mrsu_demand.keys())
            for content_id in relevant_contents:
                content_id = int(content_id)
                if content_id not in x:
                    continue
                wx = model.NewBoolVar(f"z{k}_x_{content_id}")
                model.Add(wx <= z[k])
                model.Add(wx <= x[content_id])
                model.Add(wx >= z[k] + x[content_id] - 1)

                wy = model.NewBoolVar(f"z{k}_y_{content_id}")
                model.Add(wy <= z[k])
                model.Add(wy <= y[content_id])
                model.Add(wy >= z[k] + y[content_id] - 1)

                wy_not_x = model.NewBoolVar(f"z{k}_y_not_x_{content_id}")
                model.Add(wy_not_x <= z[k])
                model.Add(wy_not_x <= y[content_id])
                model.Add(wy_not_x <= 1 - x[content_id])
                model.Add(wy_not_x >= z[k] + y[content_id] - x[content_id] - 1)

                mrsu_coeff = int(mrsu_demand.get(content_id, 0))
                frsu_coeff = int(frsu_non_mrsu_demand.get(content_id, 0))
                if mrsu_coeff > 0:
                    objective_terms.append(mrsu_coeff * wx)
                    objective_terms.append(mrsu_coeff * wy_not_x)
                if frsu_coeff > 0:
                    objective_terms.append(frsu_coeff * wy)

        model.Maximize(sum(objective_terms) if objective_terms else 0)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(cp_sat_time_limit)
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return BaselineDecision(
                selected_hotspot=_best_hotspot(candidate_hotspots),
                details={
                    "policy": "cp_sat",
                    "solver_status": status_name,
                    "fallback_to_topk": True,
                    "fallback_reason": f"cp_sat_status_{status_name}",
                },
            )

        selected_index = next((idx for idx, var in enumerate(z) if solver.Value(var) == 1), 0)
        selected_hotspot = estimated_rows[selected_index]["hotspot"]
        raw_mrsu_cache = [content_id for content_id in valid_content_ids if solver.Value(x[content_id]) == 1]
        raw_frsu_cache = [content_id for content_id in valid_content_ids if solver.Value(y[content_id]) == 1]
        mrsu_cache = _fill_unique(raw_mrsu_cache, config.mrsu_cache_capacity, env.global_top_contents)
        frsu_cache = _fill_unique(raw_frsu_cache, config.frsu_cache_capacity, env.global_top_contents)
        return BaselineDecision(
            selected_hotspot=selected_hotspot,
            mrsu_cache=mrsu_cache,
            frsu_cache=frsu_cache,
            details={
                "policy": "cp_sat",
                "solver_status": status_name,
                "objective_value": float(solver.ObjectiveValue()),
                "fallback_to_topk": False,
                "candidate_count": len(candidate_hotspots),
                "content_variable_count": len(valid_content_ids),
            },
        )
    except Exception as exc:
        return BaselineDecision(
            selected_hotspot=_best_hotspot(candidate_hotspots),
            details={
                "policy": "cp_sat",
                "fallback_to_topk": True,
                "fallback_reason": f"cp_sat_exception: {type(exc).__name__}: {exc}",
            },
        )


def _topk_regional_caches(
    env: MRSUEnvironment,
    coverage,
    decision_requests: Dict[int, List[int]],
    mrsu_capacity: int,
    frsu_capacity: int,
) -> Tuple[List[int], List[int]]:
    mrsu_demand = env.demand_for_vehicles(coverage.mrsu_covered, decision_requests)
    frsu_demand = env.demand_for_vehicles(coverage.frsu_covered, decision_requests)
    mrsu_cache = _fill_unique(
        [content_id for content_id, _ in mrsu_demand.most_common()],
        mrsu_capacity,
        env.global_top_contents,
    )
    frsu_cache = _fill_unique(
        [content_id for content_id, _ in frsu_demand.most_common()],
        frsu_capacity,
        env.global_top_contents,
    )
    return mrsu_cache, frsu_cache


def _best_hotspot(candidate_hotspots: List[dict]) -> dict:
    if not candidate_hotspots:
        return _fallback_hotspot()
    return max(
        candidate_hotspots,
        key=lambda hotspot: (
            float(hotspot.get("potential_cache_gain", 0.0)),
            int(hotspot.get("covered_vehicle_count", 0)),
        ),
    )


def _fill_unique(
    priority_items: Iterable[int],
    capacity: int,
    fallback_items: Iterable[int],
) -> List[int]:
    result = []
    seen = set()
    for item in list(priority_items) + list(fallback_items):
        try:
            content_id = int(item)
        except (TypeError, ValueError):
            continue
        if content_id in seen:
            continue
        seen.add(content_id)
        result.append(content_id)
        if len(result) >= int(capacity):
            break
    return result


def _demand_for_vehicles(vehicle_ids: Sequence[int], vehicle_requests: Dict[int, List[int]]) -> Counter:
    demand = Counter()
    for vehicle_id in vehicle_ids:
        demand.update(int(content_id) for content_id in vehicle_requests.get(int(vehicle_id), []))
    return demand


def _fallback_hotspot() -> dict:
    return {
        "hotspot_id": 0,
        "position": 0.0,
        "covered_vehicle_ids": [],
        "covered_vehicle_count": 0,
        "potential_cache_gain": 0.0,
        "dominant_contents": [],
        "demand_summary": {},
    }


def _normalize_method(method: str) -> str:
    method = str(method).strip()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported traditional baseline method: {method}")
    return method
