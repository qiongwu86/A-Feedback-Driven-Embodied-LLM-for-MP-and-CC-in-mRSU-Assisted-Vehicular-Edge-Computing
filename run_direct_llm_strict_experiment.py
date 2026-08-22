from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from agents.llm_tool_agent import DEFAULT_API_KEY_ENV_VARS, _extract_json_object
from communication.latency_model import CV2XLatencyModel, summarize_round_latencies
from mobility.candidate_hotspot import CandidateHotspotGenerator
from run_ablation_experiment import (
    METHOD_DIRECT_LLM,
    METHOD_LABELS,
    aggregate_summaries,
    apply_direct_motion,
    build_config,
    build_latency_model,
    fill_single_cache,
    limit_unique_ints,
    normalize_summary,
    parse_capacities,
    plot_capacity_curve,
    plot_capacity_delay_curve,
    plot_round_curve,
    plot_round_delay_curve,
    save_json,
    write_aggregate_csv,
    write_delay_plot_data_csv,
    write_method_csv,
    write_plot_data_csv,
)
from simulation.config import MRSUSimulationConfig
from simulation.coverage import CoverageModel
from simulation.environment import MRSUEnvironment
from simulation.metrics import RoundMetrics, summarize_metrics


STRICT_METHOD_NOTE = (
    "Strict Direct LLM directly outputs target_position and mRSU/fRSU cache rankings. "
    "The simulator converts target_position into a feasible direct speed without QP path planning. "
    "The prompt hides tool-like cache summaries, regional predicted content tables, dominant hotspot contents, "
    "and content-level miss feedback; only compact motion state, hotspot load hints, aggregate feedback, "
    "and a neutral training/global candidate pool are provided."
)


@dataclass
class StrictDirectLLMDecision:
    target_position: float
    next_speed: float
    mrsu_cache_rank: List[int]
    frsu_cache_rank: List[int]

    @property
    def mrsu_cache_list(self) -> List[int]:
        return self.mrsu_cache_rank

    @property
    def frsu_cache_list(self) -> List[int]:
        return self.frsu_cache_rank

    def to_dict(self) -> dict:
        return {
            "target_position": float(self.target_position),
            "mrsu_cache_rank": [int(x) for x in self.mrsu_cache_rank],
            "frsu_cache_rank": [int(x) for x in self.frsu_cache_rank],
            "mrsu_cache_list": [int(x) for x in self.mrsu_cache_rank],
            "frsu_cache_list": [int(x) for x in self.frsu_cache_rank],
        }


class MockStrictDirectLLMAgent:
    model = "mock-strict-direct-llm"

    def __init__(self, cache_rank_limit: int = 400):
        self.cache_rank_limit = int(cache_rank_limit)

    def decide(self, context: Dict[str, Any]) -> StrictDirectLLMDecision:
        hotspots = context.get("hotspots") or []
        best = max(
            hotspots,
            key=lambda item: (
                float(item.get("potential_cache_gain", 0.0)),
                int(item.get("covered_vehicle_count", 0)),
            ),
            default={},
        )
        target_position = float(best.get("position", context.get("mrsu", {}).get("position", 0.0)))
        pool = context.get("candidate_pool") or []
        return StrictDirectLLMDecision(
            target_position=target_position,
            next_speed=speed_toward_target(context, target_position),
            mrsu_cache_rank=limit_unique_ints(pool, self.cache_rank_limit),
            frsu_cache_rank=limit_unique_ints(pool, self.cache_rank_limit),
        )


class StrictDirectLLMAgent:
    """Compact direct-output LLM used to avoid giving Direct LLM tool-like evidence."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_context_chars: int = 12000,
        cache_rank_limit: int = 400,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model_name
        self.max_context_chars = int(max_context_chars)
        self.cache_rank_limit = int(cache_rank_limit)
        self._client = None

    def decide(self, context: Dict[str, Any]) -> StrictDirectLLMDecision:
        prompt = self._build_prompt(context)
        data = self._call_json(prompt)
        target_position = float(data.get("target_position", context.get("mrsu", {}).get("position", 0.0)))
        return StrictDirectLLMDecision(
            target_position=target_position,
            next_speed=speed_toward_target(context, target_position),
            mrsu_cache_rank=limit_unique_ints(_as_int_list(data.get("mrsu_cache_rank", [])), self.cache_rank_limit),
            frsu_cache_rank=limit_unique_ints(_as_int_list(data.get("frsu_cache_rank", [])), self.cache_rank_limit),
        )

    def _client_or_raise(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _call_json(self, prompt: str) -> dict:
        response = self._client_or_raise().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
        )
        return _extract_json_object(response.choices[0].message.content)

    def _compact_context(self, context: Dict[str, Any]) -> str:
        text = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= self.max_context_chars:
            return text
        return text[: self.max_context_chars] + "\n...TRUNCATED..."

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        limit = int(context.get("cache_rank_limit", self.cache_rank_limit))
        return (
            "Direct LLM ablation. No tools. Return JSON only.\n"
            "Road is circular one-way; choose a reachable target position.\n"
            "Use candidate_pool content IDs for cache rankings; each list should be unique.\n"
            f"Return about {limit} IDs per ranking when possible; simulator truncates to current capacity.\n"
            'Schema:{"target_position":number,"mrsu_cache_rank":[int],"frsu_cache_rank":[int]}\n'
            f"CONTEXT:{self._compact_context(context)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a stricter Direct LLM ablation with a compact prompt and no tool-like cache evidence."
    )
    parser.add_argument("--rounds", type=int, default=100, help="Total physical simulation ticks.")
    parser.add_argument("--decision-interval", type=int, default=10, help="Physical ticks per decision window.")
    parser.add_argument(
        "--seed",
        type=str,
        default="42",
        help="Single seed or comma-separated seeds. Ignored when --seeds is set.",
    )
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds, e.g. 7,42,2026.")
    parser.add_argument("--rsu-cache", type=int, default=200)
    parser.add_argument("--capacities", type=str, default="", help="Comma-separated synchronized mRSU/fRSU capacities.")
    parser.add_argument("--plot-capacity", type=int, default=200)

    parser.add_argument("--vehicle-num", type=int, default=50)
    parser.add_argument("--user-num", type=int, default=50)
    parser.add_argument("--movie-num", type=int, default=2000)
    parser.add_argument("--road-length", type=float, default=1000.0)
    parser.add_argument("--request-min", type=int, default=670)
    parser.add_argument("--request-max", type=int, default=690)
    parser.add_argument("--min-vehicle-speed", type=float, default=12.0)
    parser.add_argument("--max-vehicle-speed", type=float, default=20.0)
    parser.add_argument("--vehicle-speed-noise-std", type=float, default=1.0)
    parser.add_argument("--platoon-cluster-count", type=int, default=4)
    parser.add_argument("--platoon-cluster-std", type=float, default=35.0)
    parser.add_argument("--platoon-speed-std", type=float, default=1.0)
    parser.add_argument("--true-demand-noise-scale", type=float, default=0.4)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--mrsu-initial-position", type=float, default=100.0)
    parser.add_argument("--mrsu-initial-speed", type=float, default=15.0)
    parser.add_argument("--mrsu-radius", type=float, default=200.0)
    parser.add_argument("--frsu-radius", type=float, default=None, help="Defaults to --mrsu-radius when omitted.")
    parser.add_argument("--mrsu-v-min", type=float, default=0.0)
    parser.add_argument("--mrsu-v-max", type=float, default=30.0)
    parser.add_argument("--mrsu-a-min", type=float, default=-4.0)
    parser.add_argument("--mrsu-a-max", type=float, default=4.0)
    parser.add_argument("--frsu-position", type=float, default=500.0)
    parser.add_argument("--grid-step", type=float, default=50.0)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--planner-horizon", type=int, default=10)
    parser.add_argument("--default-lambda-smooth", type=float, default=1.0)
    parser.add_argument("--global-topk-for-prompt", type=int, default=120)

    parser.add_argument("--agent", choices=["auto", "mock", "llm"], default="auto")
    parser.add_argument("--base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", type=str, default="")
    parser.add_argument("--model-name", type=str, default="qwen3.6-flash")
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument(
        "--strict-cache-rank-limit",
        type=int,
        default=400,
        help="Number of ordered cache candidates Strict Direct LLM should return before capacity truncation.",
    )

    parser.add_argument("--latency-content-size-kbit", type=float, default=800.0)
    parser.add_argument("--latency-bandwidth-mhz", type=float, default=10.0)
    parser.add_argument("--latency-rsu-distance-loss", type=float, default=16.0)
    parser.add_argument("--latency-cloud-backhaul-rate-mbps", type=float, default=80.0)
    parser.add_argument("--latency-cloud-extra-ms", type=float, default=20.0)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [METHOD_DIRECT_LLM]
    seeds = parse_seed_values(args.seeds or args.seed)
    capacities = parse_capacities(args.capacities, args.rsu_cache)
    round_plot_capacity = int(args.plot_capacity) if int(args.plot_capacity) in capacities else int(capacities[0])
    output_dir = create_output_dir(args.output_dir)
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=False)
    direct_agent = build_strict_direct_agent(args)
    latency_model = build_latency_model(args)

    print("Strict Direct LLM experiment config:")
    print(
        json.dumps(
            {
                "method": METHOD_DIRECT_LLM,
                "method_label": METHOD_LABELS[METHOD_DIRECT_LLM],
                "method_note": STRICT_METHOD_NOTE,
                "seeds": seeds,
                "capacities": capacities,
                "physical_rounds": args.rounds,
                "decision_interval": args.decision_interval,
                "decision_rounds": (args.rounds + args.decision_interval - 1) // max(args.decision_interval, 1),
                "plot_capacity": round_plot_capacity,
                "road_topology": "circular_one_way",
                "latency_model": latency_model.config.to_dict(),
                "output_dir": output_dir,
                "data_dir": str(data_dir),
                "direct_llm_agent": describe_strict_direct_agent(direct_agent),
                "strict_cache_rank_limit": int(args.strict_cache_rank_limit),
                "prompt_mode": "strict_compact_no_tool_like_cache_evidence",
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    results: Dict[str, Dict[str, Dict[str, dict]]] = {METHOD_DIRECT_LLM: {}}
    summaries: Dict[str, Dict[str, Dict[str, dict]]] = {METHOD_DIRECT_LLM: {}}
    configs: Dict[str, Dict[str, dict]] = {}

    for seed in seeds:
        configs[str(seed)] = {}
        print(f"\n=== seed={seed} ===")
        for capacity in capacities:
            config = build_config(args, seed=seed, capacity=capacity, output_dir=output_dir)
            configs[str(seed)][str(capacity)] = asdict(config)
            print(f"\nRunning Strict Direct LLM seed={seed} C={capacity}...")
            result = run_strict_direct_llm_method(
                config=config,
                direct_agent=direct_agent,
                latency_model=latency_model,
                verbose=not args.quiet,
            )
            summary = normalize_summary(result, METHOD_DIRECT_LLM, config)
            summary["method_note"] = STRICT_METHOD_NOTE
            summary["prompt_mode"] = "strict_compact_no_tool_like_cache_evidence"
            result["summary"] = summary
            results[METHOD_DIRECT_LLM].setdefault(str(seed), {})[str(capacity)] = result
            summaries[METHOD_DIRECT_LLM].setdefault(str(seed), {})[str(capacity)] = summary
            write_method_csv(data_dir / "direct_llm.csv", METHOD_DIRECT_LLM, results[METHOD_DIRECT_LLM])
            save_partial_json(output_dir, seeds, capacities, configs, summaries, results, latency_model)
            print(
                f"Finished Strict Direct LLM seed={seed} C={capacity}: "
                f"LocalACHR={float(summary.get('local_rsu_achr', summary['achr'])):.4f} "
                f"ACHR={float(summary.get('achr', 0.0)):.4f} "
                f"Delay={float(summary.get('average_delay_ms', 0.0)):.2f}ms "
                f"fallback={int(summary.get('direct_llm_fallback_count', 0))}"
            )

    aggregate = aggregate_summaries(summaries, methods, capacities, seeds)
    write_aggregate_csv(data_dir / "aggregate_summary.csv", aggregate, methods, capacities)
    write_plot_data_csv(data_dir / "capacity_achr_mean.csv", aggregate, methods, capacities)
    write_delay_plot_data_csv(data_dir / "capacity_delay_mean.csv", aggregate, methods, capacities)
    plot_capacity_curve(aggregate, methods, capacities, Path(output_dir) / "direct_llm_achr_vs_capacity.svg")
    plot_capacity_delay_curve(aggregate, methods, capacities, Path(output_dir) / "direct_llm_delay_vs_capacity.svg")
    plot_round_curve(
        results,
        methods,
        round_plot_capacity,
        Path(output_dir) / f"direct_llm_round_chr_capacity_{round_plot_capacity}.svg",
    )
    plot_round_delay_curve(
        results,
        methods,
        round_plot_capacity,
        Path(output_dir) / f"direct_llm_round_delay_capacity_{round_plot_capacity}.svg",
    )
    payload = build_result_payload(seeds, capacities, configs, aggregate, summaries, results, data_dir, latency_model)
    save_json(payload, str(Path(output_dir) / "direct_llm_experiment_results.json"))
    print("\nStrict Direct LLM experiment finished.")
    print(f"Results saved to: {os.path.abspath(output_dir)}")


def run_strict_direct_llm_method(
    config: MRSUSimulationConfig,
    direct_agent,
    latency_model: CV2XLatencyModel = None,
    verbose: bool = True,
) -> dict:
    env = MRSUEnvironment(config)
    latency_model = latency_model or CV2XLatencyModel()
    valid_content_ids = list(range(1, config.movie_num + 1))
    cache_rank_limit = int(getattr(direct_agent, "cache_rank_limit", 400) or 400)
    hotspot_generator = CandidateHotspotGenerator(
        road_length=config.road_length,
        grid_step=config.grid_step,
        mrsu_radius=config.mrsu_radius,
        mrsu_cache_capacity=config.mrsu_cache_capacity,
        candidate_count=config.candidate_count,
    )

    round_metrics: List[RoundMetrics] = []
    round_logs: List[dict] = []

    for round_index in range(config.decision_rounds):
        window_ticks = env.decision_window_ticks(round_index)
        decision_requests = env.predict_round_requests(round_index, use_feedback=True)
        vehicle_demands = env.vehicle_request_counters(decision_requests)
        candidate_hotspots = [
            hotspot.to_dict()
            for hotspot in hotspot_generator.generate(env.mobility.positions(), vehicle_demands)
        ]
        context = build_strict_direct_context(
            env=env,
            config=config,
            round_index=round_index,
            decision_requests=decision_requests,
            candidate_hotspots=candidate_hotspots,
            cache_rank_limit=cache_rank_limit,
        )
        direct_error = None
        direct_fallback = False
        try:
            direct_decision = direct_agent.decide(context)
        except Exception as exc:
            direct_error = f"{type(exc).__name__}: {exc}"
            direct_fallback = True
            direct_decision = fallback_strict_direct_decision(env, config, candidate_hotspots, context)

        motion_details = apply_direct_motion(env, config, direct_decision, advance_position=False)
        decision_coverage = env.project_service_window_coverage(ticks=window_ticks)
        fallback_pool = build_neutral_candidate_pool(env, cache_rank_limit=max(config.movie_num, cache_rank_limit))
        mrsu_cache = fill_single_cache(
            direct_decision.mrsu_cache_rank,
            config.mrsu_cache_capacity,
            fallback_pool,
            valid_content_ids,
        )
        frsu_cache = fill_single_cache(
            direct_decision.frsu_cache_rank,
            config.frsu_cache_capacity,
            fallback_pool,
            valid_content_ids,
        )

        env.set_cache(mrsu_cache, frsu_cache)
        coverage = env.execute_service_window(path_plan=None, ticks=window_ticks)
        vehicle_requests = env.sample_round_requests(round_index)
        metrics = env.evaluate(vehicle_requests, coverage, mrsu_cache, frsu_cache)
        latency = latency_model.evaluate_round(
            vehicle_requests=vehicle_requests,
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
        selected_hotspot = _find_best_hotspot(candidate_hotspots)
        round_metrics.append(metrics)
        round_logs.append(
            {
                "round": int(round_index),
                "physical_tick_start": int(round_index * config.decision_interval),
                "physical_tick_end": int(round_index * config.decision_interval + window_ticks - 1),
                "decision_interval_ticks": int(window_ticks),
                "chr": float(metrics.chr),
                "local_rsu_chr": float(metrics.local_rsu_chr),
                "round_delay_ms": float(latency.get("average_delay_ms", 0.0)),
                "latency": latency,
                "request_count": int(metrics.request_count),
                "decision_request_count": int(sum(len(requests) for requests in decision_requests.values())),
                "evaluation_request_count": int(metrics.request_count),
                "decision_request_source": "strict_direct_predicted_history_feedback_signal",
                "uses_miss_feedback": True,
                "evaluation_request_source": "current_service_window_true_requests",
                "coverage_source": "service_window_swept_coverage",
                "selected_hotspot": selected_hotspot,
                "motion_details": motion_details,
                "path_plan_status": "direct_llm_no_qp_planner",
                "path_plan_solver": "none",
                "tool_decision": None,
                "tool_decision_error": None,
                "llm_context_chars": int(len(json.dumps(context, ensure_ascii=False, separators=(',', ':')))),
                "llm_context_schema": "strict_direct_llm_compact_v1",
                "cache_fit_analysis": None,
                "cache_update_used": False,
                "direct_llm_decision": direct_decision.to_dict(),
                "direct_llm_error": direct_error,
                "direct_llm_fallback": bool(direct_fallback),
                "direct_cache_rank_limit": int(cache_rank_limit),
                "direct_llm_mrsu_rank_length": len(direct_decision.mrsu_cache_rank),
                "direct_llm_frsu_rank_length": len(direct_decision.frsu_cache_rank),
                "mrsu_position": float(env.mrsu.position),
                "mrsu_speed": float(env.mrsu.speed),
                "mrsu_covered": [int(x) for x in coverage.mrsu_covered],
                "frsu_covered": [int(x) for x in coverage.frsu_covered],
                "decision_mrsu_covered": [int(x) for x in decision_coverage.mrsu_covered],
                "decision_frsu_covered": [int(x) for x in decision_coverage.frsu_covered],
                "overlap": [int(x) for x in coverage.overlap],
                "mrsu_cache": [int(x) for x in mrsu_cache],
                "frsu_cache": [int(x) for x in frsu_cache],
                "cache_tool_details": {
                    "policy": "strict_direct_llm_rank_truncated",
                    "fallback_used": bool(direct_fallback),
                    "strict_prompt": True,
                    "tool_like_content_evidence_hidden": True,
                    "truncation_rule": "first valid unique ranked contents, then neutral training/global fallback",
                },
                "metrics": metrics.to_dict(),
            }
        )
        if verbose:
            fallback_text = " direct_fallback=True" if direct_fallback else ""
            print(
                f"[Strict Direct LLM] round={round_index:02d} "
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
            "method": METHOD_DIRECT_LLM,
            "method_label": METHOD_LABELS[METHOD_DIRECT_LLM],
            "method_note": STRICT_METHOD_NOTE,
            "physical_rounds": int(config.rounds),
            "decision_interval_ticks": int(config.decision_interval),
            "decision_rounds": int(config.decision_rounds),
            "latency_model": latency_model.config.to_dict(),
            "round_chr": [float(item.chr) for item in round_metrics],
            "round_local_rsu_chr": [float(item.local_rsu_chr) for item in round_metrics],
            "round_delay_ms": [float(log.get("latency", {}).get("average_delay_ms", 0.0)) for log in round_logs],
            "cache_update_count": 0,
            "direct_llm_fallback_count": sum(1 for log in round_logs if log.get("direct_llm_fallback")),
            "uses_miss_feedback": True,
            "prompt_mode": "strict_compact_no_tool_like_cache_evidence",
        }
    )
    return {"summary": summary, "round_logs": round_logs}


def build_strict_direct_context(
    env: MRSUEnvironment,
    config: MRSUSimulationConfig,
    round_index: int,
    decision_requests: Dict[int, List[int]],
    candidate_hotspots: List[dict],
    cache_rank_limit: int,
) -> dict:
    request_counts = {int(vid): int(len(reqs)) for vid, reqs in decision_requests.items()}
    vehicles = [
        {
            "id": int(vehicle.vehicle_id),
            "x": round(float(vehicle.position), 1),
            "v": round(float(vehicle.speed), 1),
            "req_n": request_counts.get(int(vehicle.vehicle_id), 0),
        }
        for vehicle in env.mobility.vehicles
    ]
    hotspots = [
        {
            "id": int(hotspot.get("hotspot_id", idx)),
            "position": round(float(hotspot.get("position", 0.0)), 1),
            "covered_vehicle_count": int(hotspot.get("covered_vehicle_count", 0)),
            "potential_cache_gain": round(float(hotspot.get("potential_cache_gain", 0.0)), 2),
            "forward_distance": round(
                CoverageModel.forward_distance(
                    float(env.mrsu.position),
                    float(hotspot.get("position", 0.0)),
                    float(config.road_length),
                ),
                1,
            ),
        }
        for idx, hotspot in enumerate(candidate_hotspots[: config.candidate_count])
    ]
    return {
        "round": int(round_index),
        "road": {
            "topology": "circular_one_way",
            "length": float(config.road_length),
            "dt": float(config.dt),
            "decision_interval_ticks": int(config.decision_interval),
            "service_window": "decision is held; vehicles covered at least once can be served",
        },
        "mrsu": {
            "position": round(float(env.mrsu.position), 1),
            "speed": round(float(env.mrsu.speed), 1),
            "radius": float(config.mrsu_radius),
            "cache_capacity": int(config.mrsu_cache_capacity),
            "v_min": float(config.mrsu_v_min),
            "v_max": float(config.mrsu_v_max),
            "a_min": float(config.mrsu_a_min),
            "a_max": float(config.mrsu_a_max),
        },
        "frsu": {
            "position": round(float(env.frsu.position), 1),
            "radius": float(config.frsu_radius),
            "cache_capacity": int(config.frsu_cache_capacity),
        },
        "feedback": {
            "last_chr": round(float(env.last_metrics.chr), 4),
            "last_not_covered_count": int(env.last_metrics.not_covered_count),
            "last_not_cached_count": int(env.last_metrics.not_cached_count),
            "last_mbs_miss_count": int(env.last_metrics.mbs_miss_count),
        },
        "vehicles": vehicles,
        "hotspots": hotspots,
        "candidate_pool": build_neutral_candidate_pool(env, cache_rank_limit),
        "cache_rank_limit": int(cache_rank_limit),
        "valid_content_id_range": [1, int(config.movie_num)],
    }


def build_neutral_candidate_pool(env: MRSUEnvironment, cache_rank_limit: int) -> List[int]:
    items: List[int] = []
    items.extend(int(x) for x in env.global_top_contents)
    for user_id in env.user_ids:
        history = env.user_history.get(int(user_id), [])
        for row in history:
            if isinstance(row, (list, tuple)) and row:
                items.append(int(row[0]))
            else:
                try:
                    items.append(int(row))
                except (TypeError, ValueError):
                    continue
    return _unique_valid(items, range(1, env.config.movie_num + 1), int(cache_rank_limit))


def speed_toward_target(context: Dict[str, Any], target_position: float) -> float:
    road = context.get("road") or {}
    mrsu = context.get("mrsu") or {}
    road_length = max(float(road.get("length", 1000.0)), 1e-9)
    dt = max(float(road.get("dt", 1.0)), 1e-9)
    interval = max(int(road.get("decision_interval_ticks", 1)), 1)
    current_position = float(mrsu.get("position", 0.0))
    distance = CoverageModel.forward_distance(current_position, float(target_position), road_length)
    requested_speed = float(distance) / max(float(interval) * dt, 1e-9)
    v_min = float(mrsu.get("v_min", 0.0))
    v_max = float(mrsu.get("v_max", max(requested_speed, 0.0)))
    return float(np.clip(requested_speed, v_min, v_max))


def fallback_strict_direct_decision(
    env: MRSUEnvironment,
    config: MRSUSimulationConfig,
    candidate_hotspots: List[dict],
    context: Dict[str, Any],
) -> StrictDirectLLMDecision:
    best = _find_best_hotspot(candidate_hotspots)
    target_position = float(best.get("position", env.mrsu.position)) % float(config.road_length)
    distance = CoverageModel.forward_distance(env.mrsu.position, target_position, config.road_length)
    requested_speed = float(distance) / max(float(config.decision_interval) * float(config.dt), 1e-9)
    pool = context.get("candidate_pool") or build_neutral_candidate_pool(env, 400)
    limit = int(context.get("cache_rank_limit", 400))
    return StrictDirectLLMDecision(
        target_position=target_position,
        next_speed=float(np.clip(requested_speed, config.mrsu_v_min, config.mrsu_v_max)),
        mrsu_cache_rank=limit_unique_ints(pool, limit),
        frsu_cache_rank=limit_unique_ints(pool, limit),
    )


def save_partial_json(
    output_dir: str,
    seeds: List[int],
    capacities: List[int],
    configs: Dict[str, Dict[str, dict]],
    summaries: Dict[str, Dict[str, Dict[str, dict]]],
    results: Dict[str, Dict[str, Dict[str, dict]]],
    latency_model,
) -> None:
    payload = build_result_payload(
        seeds=seeds,
        capacities=capacities,
        configs=configs,
        aggregate={},
        summaries=summaries,
        results=results,
        data_dir=Path(output_dir) / "data",
        latency_model=latency_model,
    )
    save_json(payload, str(Path(output_dir) / "direct_llm_experiment_partial.json"))


def build_result_payload(
    seeds: List[int],
    capacities: List[int],
    configs: Dict[str, Dict[str, dict]],
    aggregate: Dict[str, Dict[str, dict]],
    summaries: Dict[str, Dict[str, Dict[str, dict]]],
    results: Dict[str, Dict[str, Dict[str, dict]]],
    data_dir: Path,
    latency_model,
) -> dict:
    return {
        "experiment": "direct_llm_experiment_strict",
        "methods": [METHOD_DIRECT_LLM],
        "method_labels": {METHOD_DIRECT_LLM: METHOD_LABELS[METHOD_DIRECT_LLM]},
        "method_notes": {METHOD_DIRECT_LLM: STRICT_METHOD_NOTE},
        "seeds": seeds,
        "capacities": capacities,
        "latency_model": latency_model.config.to_dict(),
        "config_by_seed_capacity": configs,
        "aggregate_summaries": aggregate,
        "summaries_by_seed_capacity": summaries,
        "results_by_seed_capacity": results,
        "data_dir": str(data_dir),
    }


def build_strict_direct_agent(args: argparse.Namespace):
    api_key = resolve_api_key(args.api_key_env)
    cache_rank_limit = int(args.strict_cache_rank_limit)
    if args.agent == "mock":
        return MockStrictDirectLLMAgent(cache_rank_limit=cache_rank_limit)
    if args.agent == "auto" and not api_key:
        print("No API key is set; using MockStrictDirectLLMAgent.")
        return MockStrictDirectLLMAgent(cache_rank_limit=cache_rank_limit)
    if args.agent == "llm" and not api_key:
        print("No API key is set; using MockStrictDirectLLMAgent.")
        return MockStrictDirectLLMAgent(cache_rank_limit=cache_rank_limit)
    return StrictDirectLLMAgent(
        api_key=api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        max_context_chars=args.max_context_chars,
        cache_rank_limit=cache_rank_limit,
    )


def describe_strict_direct_agent(agent) -> str:
    if agent is None:
        return "none"
    return str(getattr(agent, "model", type(agent).__name__))


def create_output_dir(base_dir: str) -> str:
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"direct_llm_experiment_strict_{timestamp}"
    counter = 1
    while path.exists():
        path = Path(base_dir) / f"direct_llm_experiment_strict_{timestamp}_{counter:02d}"
        counter += 1
    path.mkdir(parents=True, exist_ok=False)
    return str(path)


def parse_seed_values(text: str) -> List[int]:
    seeds: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        seed = int(item)
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def resolve_api_key(api_key_env: str = "") -> str:
    if api_key_env:
        return os.getenv(api_key_env, "")
    for env_name in DEFAULT_API_KEY_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    return ""


def _as_int_list(values: Any) -> List[int]:
    if not isinstance(values, list):
        return []
    result: List[int] = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _unique_valid(items: Iterable[int], valid_content_ids: Iterable[int], limit: int) -> List[int]:
    valid = set(int(x) for x in valid_content_ids)
    seen = set()
    result: List[int] = []
    for item in items:
        try:
            content_id = int(item)
        except (TypeError, ValueError):
            continue
        if content_id in seen or content_id not in valid:
            continue
        seen.add(content_id)
        result.append(content_id)
        if len(result) >= int(limit):
            break
    return result


def _find_best_hotspot(candidate_hotspots: Sequence[dict]) -> dict:
    return max(
        candidate_hotspots or [],
        key=lambda item: (
            float(item.get("potential_cache_gain", 0.0)),
            int(item.get("covered_vehicle_count", 0)),
        ),
        default={},
    )


if __name__ == "__main__":
    main()
