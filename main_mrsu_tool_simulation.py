import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from agents.llm_tool_agent import DEFAULT_API_KEY_ENV_VARS, LLMToolAgent
from agents.gemini_tool_agent import GEMINI_API_KEY_ENV_VARS, GeminiToolAgent
from agents.gemini_rest_tool_agent import GeminiRestToolAgent
from agents.mock_tool_agent import MockToolAgent
from baselines.no_mrsu_topk import no_mrsu_frsu_topk_cache
from baselines.qp_greedy import qp_greedy_decision
from communication.latency_model import CV2XLatencyModel, LatencyModelConfig, summarize_round_latencies
from baselines.uniform_mrsu_topk import uniform_move, uniform_topk_cache
from caching.cache_repair import CacheRepair
from caching.cache_update_evaluator import CacheUpdateEvaluator
from mobility.candidate_hotspot import CandidateHotspotGenerator
from mobility.qp_path_planner import QPPathPlanner, auto_lambda_smooth
from run_traditional_baselines import _write_svg_line_chart, save_json
from simulation.config import MRSUSimulationConfig
from simulation.environment import MRSUEnvironment
from simulation.metrics import RoundMetrics, local_rsu_chr_from_counts, summarize_metrics


TOOL_AGENT_METHOD = "tool_agent"
TOOL_AGENT_LABEL = "FD-EMC"
METHOD_LABELS = {
    TOOL_AGENT_METHOD: TOOL_AGENT_LABEL,
}
METHOD_NOTES = {
    TOOL_AGENT_METHOD: "LLM tool-agent main method with hotspot selection and cache-update tool calls.",
}
METHODS = [
    TOOL_AGENT_METHOD,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mRSU/sRSU tool-agent caching simulation")
    parser.add_argument("--rounds", type=int, default=100, help="Total physical simulation ticks.")
    parser.add_argument("--decision-interval", type=int, default=10, help="Physical ticks per request/cache decision window.")
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--mrsu-cache", type=int, default=200)
    parser.add_argument("--frsu-cache", type=int, default=200)
    parser.add_argument(
        "--capacities",
        type=str,
        default="",
        help="Comma-separated synchronized mRSU/sRSU cache capacities. Overrides --mrsu-cache and --frsu-cache.",
    )
    parser.add_argument("--mrsu-radius", type=float, default=200.0)
    parser.add_argument(
        "--frsu-radius",
        type=float,
        default=None,
        help="sRSU coverage radius. Defaults to --mrsu-radius when omitted.",
    )
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
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--agent", choices=["auto", "mock", "llm", "gemini", "gemini-rest"], default="auto")
    parser.add_argument("--base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", type=str, default="")
    parser.add_argument("--model-name", type=str, default="qwen3.6-flash") # # qwen3.5-flash deepseek-v4-flash  deepseek-v4-pro qwen3.6-plus qwen3.6-flash
    parser.add_argument("--update-threshold", type=float, default=0.015)
    parser.add_argument("--cache-update-candidate-limit", type=int, default=30)
    parser.add_argument("--latency-content-size-kbit", type=float, default=800.0)
    parser.add_argument("--latency-bandwidth-mhz", type=float, default=10.0)
    parser.add_argument(
        "--latency-rsu-distance-loss",
        type=float,
        default=16.0,
        help="RSU path-loss distance coefficient in dB per log10(distance).",
    )
    parser.add_argument("--latency-cloud-backhaul-rate-mbps", type=float, default=80.0)
    parser.add_argument("--latency-cloud-extra-ms", type=float, default=20.0)
    return parser.parse_args()


def build_config(args: argparse.Namespace, cache_capacity: int = None) -> MRSUSimulationConfig:
    mrsu_cache_capacity = int(cache_capacity) if cache_capacity is not None else int(args.mrsu_cache)
    frsu_cache_capacity = int(cache_capacity) if cache_capacity is not None else int(args.frsu_cache)
    return MRSUSimulationConfig(
        seed=args.seed,
        rounds=args.rounds,
        decision_interval=args.decision_interval,
        road_length=args.road_length,
        vehicle_num=args.vehicle_num,
        user_num=args.user_num,
        movie_num=args.movie_num,
        request_min=args.request_min,
        request_max=args.request_max,
        min_vehicle_speed=args.min_vehicle_speed,
        max_vehicle_speed=args.max_vehicle_speed,
        vehicle_speed_noise_std=args.vehicle_speed_noise_std,
        platoon_cluster_count=args.platoon_cluster_count,
        platoon_cluster_std=args.platoon_cluster_std,
        platoon_speed_std=args.platoon_speed_std,
        true_demand_noise_scale=args.true_demand_noise_scale,
        dt=args.dt,
        mrsu_initial_position=args.mrsu_initial_position,
        mrsu_initial_speed=args.mrsu_initial_speed,
        mrsu_cache_capacity=mrsu_cache_capacity,
        frsu_cache_capacity=frsu_cache_capacity,
        mrsu_radius=args.mrsu_radius,
        frsu_radius=args.frsu_radius,
        mrsu_v_min=args.mrsu_v_min,
        mrsu_v_max=args.mrsu_v_max,
        mrsu_a_min=args.mrsu_a_min,
        mrsu_a_max=args.mrsu_a_max,
        frsu_position=args.frsu_position,
        grid_step=args.grid_step,
        candidate_count=args.candidate_count,
        planner_horizon=args.planner_horizon,
        default_lambda_smooth=args.default_lambda_smooth,
        global_topk_for_prompt=args.global_topk_for_prompt,
        output_dir=args.output_dir,
    )


def build_latency_model(args: argparse.Namespace) -> CV2XLatencyModel:
    return CV2XLatencyModel(
        LatencyModelConfig(
            content_size_kbit=float(args.latency_content_size_kbit),
            bandwidth_mhz=float(args.latency_bandwidth_mhz),
            rsu_distance_loss_db_per_decade=float(args.latency_rsu_distance_loss),
            cloud_backhaul_rate_mbps=float(args.latency_cloud_backhaul_rate_mbps),
            cloud_extra_latency_ms=float(args.latency_cloud_extra_ms),
        )
    )


def build_agent(args: argparse.Namespace):
    api_key = _resolve_api_key(args.api_key_env)
    gemini_api_key = _resolve_gemini_api_key(args.api_key_env)
    if args.agent == "gemini":
        if not gemini_api_key:
            print("No Gemini API key is set; falling back to MockToolAgent.")
            return MockToolAgent(update_gain_threshold=args.update_threshold)
        return GeminiToolAgent(
            api_key=gemini_api_key,
            model_name=args.model_name,
            api_key_env=args.api_key_env,
        )
    if args.agent == "gemini-rest":
        if not api_key:
            print("No API key is set; falling back to MockToolAgent.")
            return MockToolAgent(update_gain_threshold=args.update_threshold)
        return GeminiRestToolAgent(
            api_key=api_key,
            base_url=args.base_url,
            model_name=args.model_name,
        )
    if args.agent == "auto":
        if api_key:
            return LLMToolAgent(api_key=api_key, base_url=args.base_url, model_name=args.model_name)
        print("No API key is set; using MockToolAgent.")
        return MockToolAgent(update_gain_threshold=args.update_threshold)
    if args.agent == "llm":
        if not api_key:
            print("No API key is set; falling back to MockToolAgent.")
            return MockToolAgent(update_gain_threshold=args.update_threshold)
        return LLMToolAgent(api_key=api_key, base_url=args.base_url, model_name=args.model_name)
    return MockToolAgent(update_gain_threshold=args.update_threshold)


def _resolve_api_key(api_key_env: str = "") -> str:
    if api_key_env:
        return os.getenv(api_key_env, "")
    for env_name in DEFAULT_API_KEY_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    return ""


def _resolve_gemini_api_key(api_key_env: str = "") -> str:
    if api_key_env:
        return os.getenv(api_key_env, "")
    for env_name in GEMINI_API_KEY_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    return ""


def describe_agent(agent) -> str:
    if isinstance(agent, LLMToolAgent):
        return f"llm_tool_agent ({agent.model})"
    if isinstance(agent, GeminiToolAgent):
        return f"gemini_tool_agent ({agent.model})"
    if isinstance(agent, GeminiRestToolAgent):
        return f"gemini_rest_tool_agent ({agent.model})"
    return "mock_tool_agent"


def plot_round_chr(results: Dict[str, dict], output_path: str) -> None:
    series: Dict[str, List[float]] = {}
    max_rounds = 0
    for method, result in results.items():
        summary = result.get("summary", {})
        label = summary.get("method_label", METHOD_LABELS.get(method, method))
        values = [float(value) * 100.0 for value in summary.get("round_local_rsu_chr", summary.get("round_chr", []))]
        if values:
            series[label] = values
            max_rounds = max(max_rounds, len(values))
    _write_svg_line_chart(
        series,
        output_path,
        "Per-round Average Cache Hit Ratio",
        "Round",
        "Average Cache Hit Ratio (%)",
        x_labels=[str(idx) for idx in range(max_rounds)],
    )


def run_method(
    config: MRSUSimulationConfig,
    method: str,
    agent=None,
    verbose: bool = True,
    output_dir: str = None,
    cache_update_candidate_limit: int = 30,
    latency_model: CV2XLatencyModel = None,
) -> dict:
    env = MRSUEnvironment(config)
    latency_model = latency_model or CV2XLatencyModel()
    valid_content_ids = list(range(1, config.movie_num + 1))
    repair = CacheRepair(valid_content_ids, env.global_top_contents)
    cache_evaluator = CacheUpdateEvaluator(
        repair,
        env.global_top_contents,
        config,
        global_popularity=env.global_popularity,
        valid_content_ids=valid_content_ids,
    )
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

        selected_hotspot = None
        path_plan = None
        tool_decision = None
        cache_fit_analysis = None
        cache_tool_details = None
        cache_update_used = False
        tool_decision_error = None

        if method == "no_mrsu_frsu_topk":
            mrsu_cache, frsu_cache = no_mrsu_frsu_topk_cache(
                env.global_top_contents,
                config.frsu_cache_capacity,
            )
            decision_coverage = env.project_service_window_coverage(
                ticks=window_ticks,
                hold_mrsu_position=True,
            )
            decision_coverage.mrsu_covered = []
            decision_coverage.overlap = []

        elif method == "uniform_topk":
            decision_coverage = env.project_service_window_coverage(ticks=window_ticks)
            mrsu_cache, frsu_cache = uniform_topk_cache(
                env.global_top_contents,
                config.mrsu_cache_capacity,
                config.frsu_cache_capacity,
            )

        elif method == "qp_greedy":
            selected_hotspot, path_plan = qp_greedy_decision(
                planner=planner,
                candidate_hotspots=candidate_hotspots,
                current_position=env.mrsu.position,
                current_speed=env.mrsu.speed,
                lambda_smooth=config.default_lambda_smooth,
            )
            decision_coverage = env.project_service_window_coverage(path_plan, ticks=window_ticks)
            mrsu_cache, frsu_cache = cache_evaluator.greedy_repaired_cache(
                decision_requests,
                decision_coverage.mrsu_covered,
                decision_coverage.frsu_covered,
                selected_hotspot,
            )

        elif method == "tool_agent":
            if agent is None:
                agent = MockToolAgent()
            selected_for_estimate = _find_hotspot(candidate_hotspots, _best_hotspot_id(candidate_hotspots))
            decision_request_budget = max(1, sum(len(requests) for requests in decision_requests.values()))
            estimate_lambda_smooth = auto_lambda_smooth(
                current_position=env.mrsu.position,
                target_position=float(selected_for_estimate["position"]),
                potential_cache_gain=float(selected_for_estimate.get("potential_cache_gain", 0.0)),
                road_length=config.road_length,
                request_budget=decision_request_budget,
                default_lambda=config.default_lambda_smooth,
            )
            estimate_plan = planner.plan(
                current_position=env.mrsu.position,
                current_speed=env.mrsu.speed,
                target_position=float(selected_for_estimate["position"]),
                lambda_smooth=estimate_lambda_smooth,
            )
            estimated_coverage = env.project_service_window_coverage(
                estimate_plan,
                ticks=window_ticks,
            )
            estimate_content_features = env.content_features(
                estimated_coverage.mrsu_covered,
                estimated_coverage.frsu_covered,
                decision_requests,
                config.global_topk_for_prompt,
            )
            estimate_fit_summary = env.cache_fit_summary(estimated_coverage, decision_requests)
            cache_fit_analysis = cache_evaluator.build_acr_cache_fit_analysis(
                current_mrsu_cache=env.mrsu_cache,
                current_frsu_cache=env.frsu_cache,
                vehicle_requests=decision_requests,
                coverage=estimated_coverage,
                selected_hotspot=selected_for_estimate,
                content_features=estimate_content_features,
                fit_summary=estimate_fit_summary,
            )
            tool_context = env.build_tool_decision_context(
                round_index=round_index,
                vehicle_requests=decision_requests,
                candidate_hotspots=candidate_hotspots,
                selected_hotspot=selected_for_estimate,
                coverage=estimated_coverage,
                request_source="predicted_history_signal",
                cache_fit_analysis=cache_fit_analysis,
                fit_summary=estimate_fit_summary,
                content_features=estimate_content_features,
            )
            llm_context_chars = len(json.dumps(tool_context, ensure_ascii=False))
            tool_decision_error = None
            try:
                tool_decision = agent.decide_tools(tool_context)
            except Exception as exc:
                tool_decision_error = f"{type(exc).__name__}: {exc}"
                print(f"[tool_agent] LLM decision failed; using MockToolAgent fallback. {tool_decision_error}")
                tool_decision = MockToolAgent(update_gain_threshold=0.0).decide_tools(tool_context)
            selected_hotspot = _find_hotspot(candidate_hotspots, tool_decision.selected_hotspot_id)
            lambda_smooth = auto_lambda_smooth(
                current_position=env.mrsu.position,
                target_position=float(selected_hotspot["position"]),
                potential_cache_gain=float(selected_hotspot.get("potential_cache_gain", 0.0)),
                road_length=config.road_length,
                request_budget=decision_request_budget,
                default_lambda=config.default_lambda_smooth,
            )
            tool_decision.lambda_smooth = lambda_smooth
            path_plan = planner.plan(
                current_position=env.mrsu.position,
                current_speed=env.mrsu.speed,
                target_position=float(selected_hotspot["position"]),
                lambda_smooth=lambda_smooth,
            )
            decision_coverage = env.project_service_window_coverage(
                path_plan,
                ticks=window_ticks,
            )

            content_features = env.content_features(
                decision_coverage.mrsu_covered,
                decision_coverage.frsu_covered,
                decision_requests,
                config.global_topk_for_prompt,
            )
            fit_summary = env.cache_fit_summary(decision_coverage, decision_requests)
            need_mrsu_update = tool_decision.update_mrsu_cache or not env.mrsu_cache
            need_frsu_update = tool_decision.update_frsu_cache or not env.frsu_cache
            if need_mrsu_update or need_frsu_update:
                mrsu_cache, frsu_cache, cache_tool_details = cache_evaluator.update_with_acr_tool(
                    vehicle_requests=decision_requests,
                    coverage=decision_coverage,
                    selected_hotspot=selected_hotspot,
                    content_features=content_features,
                    fit_summary=fit_summary,
                    update_mrsu=need_mrsu_update,
                    update_frsu=need_frsu_update,
                    current_mrsu_cache=env.mrsu_cache,
                    current_frsu_cache=env.frsu_cache,
                )
                cache_update_used = True
            else:
                mrsu_cache, frsu_cache, cache_tool_details = cache_evaluator.update_with_acr_tool(
                    vehicle_requests=decision_requests,
                    coverage=decision_coverage,
                    selected_hotspot=selected_hotspot,
                    content_features=content_features,
                    fit_summary=fit_summary,
                    update_mrsu=False,
                    update_frsu=False,
                    current_mrsu_cache=env.mrsu_cache,
                    current_frsu_cache=env.frsu_cache,
                )

        else:
            raise ValueError(f"Unknown method: {method}")

        env.set_cache(mrsu_cache, frsu_cache)
        if method == "no_mrsu_frsu_topk":
            coverage = env.execute_service_window(
                ticks=window_ticks,
                hold_mrsu_position=True,
            )
            coverage.mrsu_covered = []
            coverage.overlap = []
        else:
            coverage = env.execute_service_window(
                path_plan,
                ticks=window_ticks,
                hold_mrsu_position=(method == "static_mrsu"),
            )
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
        round_metrics.append(metrics)
        round_logs.append(
            {
                "round": round_index,
                "physical_tick_start": int(round_index * config.decision_interval),
                "physical_tick_end": int(round_index * config.decision_interval + window_ticks - 1),
                "decision_interval_ticks": int(window_ticks),
                "chr": metrics.chr,
                "round_delay_ms": float(latency.get("average_delay_ms", 0.0)),
                "latency": latency,
                "request_count": metrics.request_count,
                "decision_request_count": sum(len(requests) for requests in decision_requests.values()),
                "evaluation_request_count": metrics.request_count,
                "decision_request_source": "predicted_history_signal",
                "evaluation_request_source": "current_service_window_true_requests",
                "coverage_source": "service_window_swept_coverage",
                "mrsu_position": env.mrsu.position,
                "mrsu_speed": env.mrsu.speed,
                "mrsu_covered": coverage.mrsu_covered,
                "frsu_covered": coverage.frsu_covered,
                "decision_mrsu_covered": getattr(decision_coverage, "mrsu_covered", []),
                "decision_frsu_covered": getattr(decision_coverage, "frsu_covered", []),
                "selected_hotspot": selected_hotspot,
                "lambda_smooth": tool_decision.lambda_smooth if tool_decision else None,
                "lambda_smooth_source": "system_auto_rule" if tool_decision else None,
                "path_plan_status": path_plan.status if path_plan else None,
                "path_plan_solver": path_plan.solver if path_plan else None,
                "tool_decision": tool_decision.to_dict() if tool_decision else None,
                "tool_decision_error": tool_decision_error,
                "llm_context_chars": llm_context_chars if method == "tool_agent" else None,
                "llm_context_schema": "compact_tool_decision_v1" if method == "tool_agent" else None,
                "cache_fit_analysis": cache_fit_analysis,
                "cache_tool_details": cache_tool_details,
                "cache_update_used": cache_update_used,
                "metrics": metrics.to_dict(),
            }
        )
        if verbose:
            update_text = f" cache_update={cache_update_used}" if method == "tool_agent" else ""
            print(
                f"[{method}] round={round_index:02d} "
                f"LocalCHR={metrics.local_rsu_chr:.4f} "
                f"mRSU_hit={metrics.mrsu_hit_count} "
                f"sRSU_hit={metrics.frsu_hit_count} "
                f"MBS_miss={metrics.mbs_miss_count}"
                f" Delay={latency.get('average_delay_ms', 0.0):.2f}ms"
                f"{update_text}"
            )

    summary = summarize_metrics(round_metrics)
    summary.update(summarize_round_latencies(log.get("latency") for log in round_logs))
    summary.update(
        {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "method_note": METHOD_NOTES.get(method, ""),
            "physical_rounds": int(config.rounds),
            "decision_interval_ticks": int(config.decision_interval),
            "decision_rounds": int(config.decision_rounds),
            "round_chr": [item.chr for item in round_metrics],
            "round_local_rsu_chr": [
                local_rsu_chr_from_counts(item.hit_count, item.not_cached_count)
                for item in round_metrics
            ],
            "cache_update_count": sum(1 for item in round_logs if item.get("cache_update_used")),
            "latency_model": latency_model.config.to_dict(),
        }
    )
    return {"summary": summary, "round_logs": round_logs}


def run_all_methods(
    config: MRSUSimulationConfig,
    agent,
    verbose: bool = True,
    output_dir: str = None,
    cache_update_candidate_limit: int = 30,
    latency_model: CV2XLatencyModel = None,
) -> Dict[str, dict]:
    latency_model = latency_model or CV2XLatencyModel()
    results = {}
    for method in METHODS:
        results[method] = run_method(
            config,
            method,
            agent=agent if method == TOOL_AGENT_METHOD else None,
            verbose=verbose,
            output_dir=output_dir,
            cache_update_candidate_limit=cache_update_candidate_limit,
            latency_model=latency_model,
        )
    return results


def _best_hotspot_id(candidate_hotspots: List[dict]) -> int:
    if not candidate_hotspots:
        return 0
    best = max(
        candidate_hotspots,
        key=lambda item: (
            float(item.get("potential_cache_gain", 0.0)),
            int(item.get("covered_vehicle_count", 0)),
        ),
    )
    return int(best.get("hotspot_id", 0))


def _find_hotspot(candidate_hotspots: List[dict], hotspot_id: int) -> dict:
    for hotspot in candidate_hotspots:
        if int(hotspot.get("hotspot_id", -1)) == int(hotspot_id):
            return hotspot
    if candidate_hotspots:
        return candidate_hotspots[0]
    return {
        "hotspot_id": 0,
        "position": 0.0,
        "covered_vehicle_ids": [],
        "covered_vehicle_count": 0,
        "potential_cache_gain": 0.0,
        "dominant_contents": [],
        "demand_summary": {},
    }


def parse_capacities(args: argparse.Namespace) -> List[int]:
    if args.capacities.strip():
        capacities = [int(item.strip()) for item in args.capacities.split(",") if item.strip()]
        if not capacities:
            raise ValueError("At least one synchronized RSU cache capacity is required.")
        return capacities
    if int(args.mrsu_cache) == int(args.frsu_cache):
        return [int(args.mrsu_cache)]
    return []


def create_embodied_output_dir(base_dir: str) -> str:
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"具身智能{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = Path(base_dir) / f"具身智能{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir)


def save_embodied_summary_csv(summary_rows: List[dict], output_path: str) -> None:
    fieldnames = [
        "rsu_cache_capacity",
        "mrsu_cache_capacity",
        "frsu_cache_capacity",
        "method",
        "method_label",
        "achr",
        "local_rsu_achr",
        "mrsu_hit_count",
        "frsu_hit_count",
        "mbs_miss_count",
        "not_covered_count",
        "not_cached_count",
        "average_delay_ms",
        "latency_scope",
        "latency_request_count",
        "excluded_not_covered_request_count",
        "mrsu_average_delay_ms",
        "frsu_average_delay_ms",
        "mbs_average_delay_ms",
        "mrsu_average_rate_mbps",
        "frsu_average_rate_mbps",
        "mbs_average_rate_mbps",
        "average_service_distance_m",
        "mrsu_average_distance_m",
        "frsu_average_distance_m",
        "mbs_average_distance_m",
        "mrsu_request_count",
        "frsu_request_count",
        "mbs_request_count",
        "cache_update_count",
        "fallback_to_topk_count",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_capacity_achr(summary_rows: List[dict], output_path: str) -> None:
    capacities = sorted({int(row["rsu_cache_capacity"]) for row in summary_rows})
    series: Dict[str, List[float]] = {}
    for method in METHODS:
        label = METHOD_LABELS.get(method, method)
        by_capacity = {
            int(row["rsu_cache_capacity"]): float(row.get("local_rsu_achr", row.get("achr", 0.0)))
            for row in summary_rows
            if row.get("method") == method
        }
        if by_capacity:
            series[label] = [float(by_capacity.get(capacity, 0.0)) * 100.0 for capacity in capacities]
    _write_svg_line_chart(
        series,
        output_path,
        "FD-EMC ACHR vs Synchronized RSU Cache Capacity",
        "mRSU/sRSU Cache Capacity",
        "Average Cache Hit Ratio (%)",
        x_labels=[str(capacity) for capacity in capacities],
    )


def summarize_for_print(method: str, summary: dict) -> str:
    label = summary.get("method_label", METHOD_LABELS.get(method, method))
    extra = f" cache_updates={summary.get('cache_update_count', 0)}" if method == TOOL_AGENT_METHOD else ""
    return (
        f"{label:24s} LocalACHR={float(summary.get('local_rsu_achr', summary['achr'])):.4f} "
        f"mRSU_hit={summary['mrsu_hit_count']} "
        f"sRSU_hit={summary['frsu_hit_count']} "
        f"MBS_miss={summary['mbs_miss_count']}"
        f" Delay={float(summary.get('average_delay_ms', 0.0)):.2f}ms"
        f"{extra}"
    )


def main() -> None:
    args = parse_args()
    capacities = parse_capacities(args)
    capacities_from_arg = bool(args.capacities.strip())
    capacity_runs = capacities if capacities else [None]
    os.makedirs(args.output_dir, exist_ok=True)
    run_output_dir = create_embodied_output_dir(args.output_dir)
    agent = build_agent(args)
    latency_model = build_latency_model(args)

    print("mRSU VEC embodied-intelligence simulation config:")
    print(
        json.dumps(
            {
                "methods": [METHOD_LABELS.get(method, method) for method in METHODS],
                "physical_rounds": args.rounds,
                "decision_interval": args.decision_interval,
                "decision_rounds": (args.rounds + args.decision_interval - 1) // max(args.decision_interval, 1),
                "seed": args.seed,
                "capacities": capacities if capacities_from_arg else [],
                "single_mrsu_cache": args.mrsu_cache,
                "single_frsu_cache": args.frsu_cache,
                "capacity_rule": (
                    "mrsu_cache_capacity = frsu_cache_capacity = each value in --capacities"
                    if capacities_from_arg
                    else "single run uses --mrsu-cache and --frsu-cache"
                ),
                "latency_model": latency_model.config.to_dict(),
                "output_dir": run_output_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Agent: {describe_agent(agent)}")
    print(f"Active methods: {', '.join(METHOD_LABELS.get(method, method) for method in METHODS)}")

    results_by_capacity: Dict[str, dict] = {}
    config_by_capacity: Dict[str, dict] = {}
    summary_rows: List[dict] = []
    for cache_capacity in capacity_runs:
        config = build_config(args, cache_capacity)
        capacity_label = (
            str(int(cache_capacity))
            if cache_capacity is not None
            else f"{config.mrsu_cache_capacity}_{config.frsu_cache_capacity}"
        )
        print(
            f"\n=== running mRSU cache={config.mrsu_cache_capacity}, "
            f"sRSU cache={config.frsu_cache_capacity}, "
            f"physical_rounds={config.rounds}, decision_rounds={config.decision_rounds} ==="
        )
        results = run_all_methods(
            config,
            agent,
            verbose=True,
            output_dir=run_output_dir,
            cache_update_candidate_limit=args.cache_update_candidate_limit,
            latency_model=latency_model,
        )
        summary_table = {}
        for method, result in results.items():
            summary = dict(result["summary"])
            summary.update(
                {
                    "rsu_cache_capacity": (
                        int(config.mrsu_cache_capacity)
                        if int(config.mrsu_cache_capacity) == int(config.frsu_cache_capacity)
                        else ""
                    ),
                    "mrsu_cache_capacity": int(config.mrsu_cache_capacity),
                    "frsu_cache_capacity": int(config.frsu_cache_capacity),
                }
            )
            result["summary"] = summary
            summary_table[method] = {
                key: value
                for key, value in summary.items()
                if key not in ("round_chr", "round_local_rsu_chr")
            }
            summary_rows.append(summary_table[method])

        print("\nFinal ACHR summary:")
        for method, summary in summary_table.items():
            if summary.get("skipped"):
                print(f"{method:24s} skipped: {summary.get('skip_reason', '')}")
                continue
            print(summarize_for_print(method, summary))

        plot_results = {
            method: result
            for method, result in results.items()
            if result.get("summary", {}).get("round_chr")
        }
        round_plot_name = (
            "具身智能每轮CHR曲线.svg"
            if len(capacity_runs) == 1
            else f"具身智能每轮CHR曲线_容量{capacity_label}.svg"
        )
        plot_round_chr(plot_results, os.path.join(run_output_dir, round_plot_name))
        results_by_capacity[capacity_label] = results
        config_by_capacity[capacity_label] = asdict(config)

    save_json(
        {
            "config_by_capacity": config_by_capacity,
            "results_by_capacity": results_by_capacity,
        },
        os.path.join(run_output_dir, "具身智能完整运行结果.json"),
    )
    save_embodied_summary_csv(summary_rows, os.path.join(run_output_dir, "具身智能ACHR汇总表.csv"))
    if len(capacity_runs) > 1:
        plot_capacity_achr(summary_rows, os.path.join(run_output_dir, "具身智能容量ACHR曲线.svg"))
    print(f"\nResults saved under: {os.path.abspath(run_output_dir)}")


if __name__ == "__main__":
    main()
