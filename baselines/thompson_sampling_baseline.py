from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from caching.cache_repair import CacheRepair
from mobility.candidate_hotspot import CandidateHotspotGenerator
from mobility.qp_path_planner import QPPathPlanner
from simulation.config import MRSUSimulationConfig
from simulation.environment import MRSUEnvironment
from simulation.metrics import RoundMetrics, summarize_metrics


HOTSPOT_CHOICES = 5
LAMBDA_CHOICES = (0.2, 1.0, 3.0)
CACHE_POLICY_CHOICES = (
    "global_topk",
    "local_demand",
    "hotspot_cache",
    "demand_aware_duplicate",
)
ACTION_DIM = HOTSPOT_CHOICES * len(LAMBDA_CHOICES) * len(CACHE_POLICY_CHOICES)
METHOD_NAME = "thompson_sampling_joint"
METHOD_LABEL = "Thompson Sampling"


@dataclass
class TSAction:
    action_id: int
    hotspot_choice: int
    lambda_smooth: float
    cache_policy_choice: str

    def to_dict(self) -> dict:
        return {
            "action_id": int(self.action_id),
            "hotspot_choice": int(self.hotspot_choice),
            "lambda_smooth": float(self.lambda_smooth),
            "cache_policy_choice": self.cache_policy_choice,
        }


def decode_ts_action(action_id: int) -> TSAction:
    action_id = int(action_id) % ACTION_DIM
    policy_index = action_id % len(CACHE_POLICY_CHOICES)
    lambda_index = (action_id // len(CACHE_POLICY_CHOICES)) % len(LAMBDA_CHOICES)
    hotspot_index = action_id // (len(CACHE_POLICY_CHOICES) * len(LAMBDA_CHOICES))
    return TSAction(
        action_id=action_id,
        hotspot_choice=int(hotspot_index),
        lambda_smooth=float(LAMBDA_CHOICES[lambda_index]),
        cache_policy_choice=CACHE_POLICY_CHOICES[policy_index],
    )


class ThompsonSamplingJointController:
    """Thompson Sampling over the same high-level joint action space as DQN."""

    def __init__(self, action_dim: int = ACTION_DIM, seed: int = 42):
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(self.action_dim, dtype=float)
        self.beta = np.ones(self.action_dim, dtype=float)

    def select_action(self) -> Tuple[int, TSAction, List[float]]:
        samples = self.rng.beta(self.alpha, self.beta)
        action_id = int(np.argmax(samples))
        return action_id, decode_ts_action(action_id), samples.tolist()

    def update(self, action_id: int, reward: float) -> None:
        reward = float(np.clip(reward, 0.0, 1.0))
        self.alpha[int(action_id)] += reward
        self.beta[int(action_id)] += 1.0 - reward


def run_thompson_sampling_baseline(
    config: MRSUSimulationConfig,
    rounds: int = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    config.rounds = int(rounds or config.rounds)
    env = MRSUEnvironment(config)
    valid_content_ids = list(range(1, config.movie_num + 1))
    repair = CacheRepair(valid_content_ids, env.global_top_contents)
    controller = ThompsonSamplingJointController(seed=seed)
    hotspot_generator = CandidateHotspotGenerator(
        road_length=config.road_length,
        grid_step=config.grid_step,
        mrsu_radius=config.mrsu_radius,
        mrsu_cache_capacity=config.mrsu_cache_capacity,
        candidate_count=max(config.candidate_count, HOTSPOT_CHOICES),
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

    round_metrics: List[RoundMetrics] = []
    round_logs: List[dict] = []
    for round_index in range(config.decision_rounds):
        window_ticks = env.decision_window_ticks(round_index)
        decision_requests = env.predict_round_requests(round_index)
        vehicle_demands = env.vehicle_request_counters(decision_requests)
        candidate_hotspots = [
            hotspot.to_dict()
            for hotspot in hotspot_generator.generate(env.mobility.positions(), vehicle_demands)
        ]
        action_id, action, action_samples = controller.select_action()
        selected_hotspot = _select_hotspot(candidate_hotspots, action.hotspot_choice, env.mrsu.position)
        path_plan = planner.plan(
            current_position=env.mrsu.position,
            current_speed=env.mrsu.speed,
            target_position=float(selected_hotspot["position"]),
            lambda_smooth=action.lambda_smooth,
        )
        decision_coverage = env.project_service_window_coverage(path_plan, ticks=window_ticks)
        content_features = env.content_features(
            decision_coverage.mrsu_covered,
            decision_coverage.frsu_covered,
            decision_requests,
            config.global_topk_for_prompt,
        )
        fit_summary = env.cache_fit_summary(decision_coverage, decision_requests)
        mrsu_cache, frsu_cache, cache_policy_detail = build_cache_by_policy(
            policy=action.cache_policy_choice,
            config=config,
            repair=repair,
            global_top_contents=env.global_top_contents,
            vehicle_requests=decision_requests,
            coverage=decision_coverage,
            selected_hotspot=selected_hotspot,
            content_features=content_features,
            fit_summary=fit_summary,
        )
        env.set_cache(mrsu_cache, frsu_cache)
        coverage = env.execute_service_window(path_plan, ticks=window_ticks)
        vehicle_requests = env.sample_round_requests(round_index)
        metrics = env.evaluate(vehicle_requests, coverage, mrsu_cache, frsu_cache)
        controller.update(action_id, metrics.chr)
        round_metrics.append(metrics)
        round_logs.append(
            {
                "round": round_index,
                "physical_tick_start": int(round_index * config.decision_interval),
                "physical_tick_end": int(round_index * config.decision_interval + window_ticks - 1),
                "decision_interval_ticks": int(window_ticks),
                "chr": metrics.chr,
                "decision_request_count": sum(len(requests) for requests in decision_requests.values()),
                "evaluation_request_count": metrics.request_count,
                "decision_request_source": "predicted_history_signal",
                "evaluation_request_source": "current_service_window_true_requests",
                "coverage_source": "service_window_swept_coverage",
                "selected_action": action.to_dict(),
                "selected_hotspot_id": selected_hotspot.get("hotspot_id"),
                "selected_hotspot": selected_hotspot,
                "lambda_smooth": action.lambda_smooth,
                "cache_policy_choice": action.cache_policy_choice,
                "cache_policy_detail": cache_policy_detail,
                "path_plan_status": path_plan.status,
                "path_plan_solver": path_plan.solver,
                "mrsu_position": env.mrsu.position,
                "mrsu_speed": env.mrsu.speed,
                "mrsu_covered": coverage.mrsu_covered,
                "frsu_covered": coverage.frsu_covered,
                "decision_mrsu_covered": decision_coverage.mrsu_covered,
                "decision_frsu_covered": decision_coverage.frsu_covered,
                "metrics": metrics.to_dict(),
                "alpha_selected_after": float(controller.alpha[action_id]),
                "beta_selected_after": float(controller.beta[action_id]),
                "sampled_value_selected": float(action_samples[action_id]),
            }
        )
        if verbose:
            print(
                f"[{METHOD_NAME}] round={round_index:02d} "
                f"CHR={metrics.chr:.4f} "
                f"action={action_id} "
                f"hotspot={selected_hotspot.get('hotspot_id')} "
                f"lambda={action.lambda_smooth:.1f} "
                f"policy={action.cache_policy_choice} "
                f"mRSU_hit={metrics.mrsu_hit_count} "
                f"fRSU_hit={metrics.frsu_hit_count} "
                f"MBS_miss={metrics.mbs_miss_count}"
            )

    summary = summarize_metrics(round_metrics)
    summary.update(
        {
            "method": METHOD_NAME,
            "method_label": METHOD_LABEL,
            "physical_rounds": int(config.rounds),
            "decision_interval_ticks": int(config.decision_interval),
            "decision_rounds": int(config.decision_rounds),
            "round_chr": [item.chr for item in round_metrics],
            "final_alpha": controller.alpha.tolist(),
            "final_beta": controller.beta.tolist(),
        }
    )
    return {"summary": summary, "round_logs": round_logs}


def build_cache_by_policy(
    policy: str,
    config: MRSUSimulationConfig,
    repair: CacheRepair,
    global_top_contents: Sequence[int],
    vehicle_requests: Dict[int, List[int]],
    coverage,
    selected_hotspot: dict,
    content_features: List[dict],
    fit_summary: dict,
) -> Tuple[List[int], List[int], dict]:
    if policy == "global_topk":
        mrsu_raw = list(global_top_contents[: config.mrsu_cache_capacity])
        frsu_raw = list(global_top_contents[: config.frsu_cache_capacity])
        detail = {"policy": policy}
        mrsu_cache, frsu_cache = _repair_independent(repair, mrsu_raw, frsu_raw, config, list(global_top_contents))
        return mrsu_cache, frsu_cache, detail

    if policy == "local_demand":
        mrsu_raw = [
            int(item["content_id"])
            for item in sorted(
                content_features,
                key=lambda item: (
                    int(item.get("mrsu_group_popularity", 0)),
                    int(item.get("mrsu_only_popularity", 0)),
                    int(item.get("global_popularity", 0)),
                ),
                reverse=True,
            )
        ]
        frsu_raw = [
            int(item["content_id"])
            for item in sorted(
                content_features,
                key=lambda item: (
                    int(item.get("frsu_group_popularity", 0)),
                    int(item.get("frsu_only_popularity", 0)),
                    int(item.get("global_popularity", 0)),
                ),
                reverse=True,
            )
        ]
        detail = {"policy": policy, "mrsu_candidate_head": mrsu_raw[:20], "frsu_candidate_head": frsu_raw[:20]}
        mrsu_cache, frsu_cache = _repair_independent(repair, mrsu_raw, frsu_raw, config, list(global_top_contents))
        return mrsu_cache, frsu_cache, detail

    if policy == "hotspot_cache":
        hotspot_candidates = _sorted_keys_by_value(selected_hotspot.get("demand_summary") or {})
        hotspot_candidates += [int(x) for x in selected_hotspot.get("dominant_contents", [])]
        mrsu_raw = hotspot_candidates + [
            int(item.get("content_id"))
            for item in fit_summary.get("mrsu_top_missing_contents", [])
        ]
        frsu_raw = [
            int(item["content_id"])
            for item in sorted(
                content_features,
                key=lambda item: (
                    int(item.get("frsu_group_popularity", 0)),
                    int(item.get("frsu_only_popularity", 0)),
                    int(item.get("global_popularity", 0)),
                ),
                reverse=True,
            )
        ]
        detail = {"policy": policy, "hotspot_candidate_head": hotspot_candidates[:20]}
        mrsu_cache, frsu_cache = _repair_independent(repair, mrsu_raw, frsu_raw, config, list(global_top_contents))
        return mrsu_cache, frsu_cache, detail

    if policy == "demand_aware_duplicate":
        mrsu_raw = []
        frsu_raw = []
        for item in sorted(
            content_features,
            key=lambda row: (
                int(row.get("mrsu_group_popularity", 0)) + int(row.get("frsu_group_popularity", 0)),
                int(row.get("overlap_popularity", 0)),
                int(row.get("global_popularity", 0)),
            ),
            reverse=True,
        ):
            content_id = int(item["content_id"])
            mrsu_score = int(item.get("mrsu_group_popularity", 0)) + int(item.get("mrsu_only_popularity", 0))
            frsu_score = int(item.get("frsu_group_popularity", 0)) + int(item.get("frsu_only_popularity", 0))
            overlap_score = int(item.get("overlap_popularity", 0))
            if mrsu_score > 0:
                mrsu_raw.append(content_id)
            if frsu_score > 0:
                frsu_raw.append(content_id)
            if overlap_score > 0 and mrsu_score > 0 and frsu_score > 0:
                mrsu_raw.append(content_id)
                frsu_raw.append(content_id)
        mrsu_raw += [int(item.get("content_id")) for item in fit_summary.get("mrsu_top_missing_contents", [])]
        frsu_raw += [int(item.get("content_id")) for item in fit_summary.get("frsu_top_missing_contents", [])]
        detail = {"policy": policy, "mrsu_candidate_head": mrsu_raw[:20], "frsu_candidate_head": frsu_raw[:20]}
        mrsu_cache, frsu_cache = _repair_independent(repair, mrsu_raw, frsu_raw, config, list(global_top_contents))
        return mrsu_cache, frsu_cache, detail

    raise ValueError(f"Unknown Thompson Sampling cache policy: {policy}")


def _repair_independent(
    repair: CacheRepair,
    mrsu_raw: Sequence[int],
    frsu_raw: Sequence[int],
    config: MRSUSimulationConfig,
    fallback: Sequence[int],
) -> Tuple[List[int], List[int]]:
    mrsu_cache = _fill_single_cache(repair.clean_priority_list(mrsu_raw), config.mrsu_cache_capacity, fallback)
    frsu_cache = _fill_single_cache(repair.clean_priority_list(frsu_raw), config.frsu_cache_capacity, fallback)
    return mrsu_cache, frsu_cache


def _fill_single_cache(items: Sequence[int], capacity: int, fallback: Sequence[int]) -> List[int]:
    seen = set()
    result = []
    for content_id in list(items) + list(fallback):
        content_id = int(content_id)
        if content_id in seen:
            continue
        seen.add(content_id)
        result.append(content_id)
        if len(result) >= capacity:
            break
    return result[:capacity]


def _select_hotspot(candidate_hotspots: List[dict], hotspot_choice: int, fallback_position: float) -> dict:
    if not candidate_hotspots:
        return {
            "hotspot_id": None,
            "position": float(fallback_position),
            "covered_vehicle_ids": [],
            "covered_vehicle_count": 0,
            "potential_cache_gain": 0.0,
            "dominant_contents": [],
            "demand_summary": {},
        }
    index = min(max(int(hotspot_choice), 0), min(len(candidate_hotspots), HOTSPOT_CHOICES) - 1)
    return candidate_hotspots[index]


def _sorted_keys_by_value(mapping: Dict) -> List[int]:
    return [
        int(key)
        for key, _ in sorted(
            mapping.items(),
            key=lambda item: int(item[1]),
            reverse=True,
        )
    ]
