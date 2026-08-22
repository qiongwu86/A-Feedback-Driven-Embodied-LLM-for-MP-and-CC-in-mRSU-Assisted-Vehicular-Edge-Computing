from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from baselines.dqn_baseline import METHOD_DQN, METHOD_LABEL, METHOD_NOTE, evaluate_dqn
from communication.latency_model import CV2XLatencyModel, LatencyModelConfig
from run_traditional_baselines import _write_svg_line_chart, save_json
from simulation.config import MRSUSimulationConfig
from simulation.metrics import local_rsu_chr_from_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained conventional DQN baseline without online learning."
    )
    parser.add_argument("--model-dir", type=str, default="", help="Directory from train_dqn_baseline.py.")
    parser.add_argument("--model-path", type=str, default="", help="Single DQN .pt model path for one-capacity tests.")
    parser.add_argument("--rounds", type=int, default=100, help="Total physical simulation ticks.")
    parser.add_argument("--decision-interval", type=int, default=10, help="Physical ticks per DQN action window.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated evaluation seeds, e.g. 7,42,2026.")
    parser.add_argument("--rsu-cache", type=int, default=200)
    parser.add_argument("--capacities", type=str, default="")

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
    parser.add_argument(
        "--frsu-radius",
        type=float,
        default=None,
        help="fRSU coverage radius. Defaults to --mrsu-radius when omitted.",
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

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
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
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capacities = parse_capacities(args.capacities, args.rsu_cache)
    seeds = parse_seeds(args.seeds, args.seed)
    model_dir = Path(args.model_dir) if args.model_dir else find_latest_training_dir(args.output_dir)
    if not args.model_path and not valid_dir(model_dir):
        raise FileNotFoundError("No DQN training directory was found. Run train_dqn_baseline.py first or pass --model-path.")
    output_dir = create_output_dir(args.output_dir)
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=False)
    latency_model = build_latency_model(args)

    print("DQN baseline evaluation config:")
    print(
        json.dumps(
            {
                "method": METHOD_DQN,
                "method_label": METHOD_LABEL,
                "model_dir": str(model_dir) if model_dir else "",
                "model_path": args.model_path,
                "capacities": capacities,
                "seeds": seeds,
                "physical_rounds": int(args.rounds),
                "decision_interval": int(args.decision_interval),
                "decision_rounds": (args.rounds + args.decision_interval - 1) // max(args.decision_interval, 1),
                "online_learning": False,
                "uses_content_level_feedback": False,
                "uses_predicted_content_demands": False,
                "reward_includes_delay": False,
                "latency_model": latency_model.config.to_dict(),
                "cache_content_source": "training_history_only",
                "output_dir": output_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    results: Dict[str, Dict[str, dict]] = {}
    summaries: Dict[str, Dict[str, dict]] = {}
    configs: Dict[str, Dict[str, dict]] = {}
    model_by_capacity: Dict[str, str] = {}
    for seed in seeds:
        results[str(seed)] = {}
        summaries[str(seed)] = {}
        configs[str(seed)] = {}
        print(f"\n=== DQN evaluation seed={seed} ===")
        for capacity in capacities:
            model_path = resolve_model_path(args.model_path, model_dir, capacity, len(capacities))
            model_by_capacity[str(capacity)] = str(model_path)
            config = build_config(args, capacity, seed, output_dir)
            configs[str(seed)][str(capacity)] = asdict(config)
            print(f"\nRunning DQN seed={seed} capacity={capacity} model={model_path}")
            result = evaluate_dqn(
                config=config,
                model_path=model_path,
                seed=seed,
                device=args.device,
                latency_model=latency_model,
                verbose=not args.quiet,
            )
            summary = normalize_summary(result, config)
            result["summary"] = summary
            results[str(seed)][str(capacity)] = result
            summaries[str(seed)][str(capacity)] = summary
            write_dqn_csv(data_dir / "dqn.csv", results)
            print(
                f"Finished DQN seed={seed} C={capacity}: "
                f"LocalACHR={float(summary.get('local_rsu_achr', summary['achr'])):.4f} "
                f"mRSU_hit={summary['mrsu_hit_count']} "
                f"sRSU_hit={summary['frsu_hit_count']} "
                f"MBS_miss={summary['mbs_miss_count']} "
                f"Delay={float(summary.get('average_delay_ms', 0.0)):.2f}ms"
            )

    aggregate = aggregate_by_capacity(summaries, capacities, seeds)
    write_aggregate_csv(data_dir / "aggregate_summary.csv", aggregate, capacities)
    write_capacity_csv(data_dir / "capacity_achr_mean.csv", aggregate, capacities)
    write_capacity_delay_csv(data_dir / "capacity_delay_mean.csv", aggregate, capacities)
    plot_round_curves(results, capacities, output_dir)
    plot_round_delay_curves(results, capacities, output_dir)
    plot_capacity_curve(aggregate, capacities, Path(output_dir) / "dqn_achr_vs_capacity.svg")
    plot_capacity_delay_curve(aggregate, capacities, Path(output_dir) / "dqn_delay_vs_capacity.svg")

    save_json(
        {
            "experiment": "dqn_baseline_evaluation",
            "method": METHOD_DQN,
            "method_label": METHOD_LABEL,
            "method_note": METHOD_NOTE,
            "seeds": seeds,
            "capacities": capacities,
            "model_dir": str(model_dir) if model_dir else "",
            "model_by_capacity": model_by_capacity,
            "latency_model": latency_model.config.to_dict(),
            "reward_includes_delay": False,
            "config_by_seed_capacity": configs,
            "aggregate_summaries": aggregate,
            "summaries_by_seed_capacity": summaries,
            "results_by_seed_capacity": results,
            "data_dir": str(data_dir),
        },
        str(Path(output_dir) / "dqn_baseline_results.json"),
    )
    print("\nDQN evaluation finished.")
    print(f"Results saved to: {os.path.abspath(output_dir)}")


def build_config(args: argparse.Namespace, capacity: int, seed: int, output_dir: str) -> MRSUSimulationConfig:
    return MRSUSimulationConfig(
        seed=int(seed),
        rounds=int(args.rounds),
        decision_interval=int(args.decision_interval),
        road_length=float(args.road_length),
        vehicle_num=int(args.vehicle_num),
        user_num=int(args.user_num),
        movie_num=int(args.movie_num),
        request_min=int(args.request_min),
        request_max=int(args.request_max),
        min_vehicle_speed=float(args.min_vehicle_speed),
        max_vehicle_speed=float(args.max_vehicle_speed),
        vehicle_speed_noise_std=float(args.vehicle_speed_noise_std),
        platoon_cluster_count=int(args.platoon_cluster_count),
        platoon_cluster_std=float(args.platoon_cluster_std),
        platoon_speed_std=float(args.platoon_speed_std),
        true_demand_noise_scale=float(args.true_demand_noise_scale),
        dt=float(args.dt),
        mrsu_initial_position=float(args.mrsu_initial_position),
        mrsu_initial_speed=float(args.mrsu_initial_speed),
        mrsu_cache_capacity=int(capacity),
        frsu_cache_capacity=int(capacity),
        mrsu_radius=float(args.mrsu_radius),
        frsu_radius=args.frsu_radius,
        mrsu_v_min=float(args.mrsu_v_min),
        mrsu_v_max=float(args.mrsu_v_max),
        mrsu_a_min=float(args.mrsu_a_min),
        mrsu_a_max=float(args.mrsu_a_max),
        frsu_position=float(args.frsu_position),
        grid_step=float(args.grid_step),
        candidate_count=int(args.candidate_count),
        planner_horizon=int(args.planner_horizon),
        default_lambda_smooth=float(args.default_lambda_smooth),
        global_topk_for_prompt=int(args.global_topk_for_prompt),
        output_dir=output_dir,
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


def normalize_summary(result: dict, config: MRSUSimulationConfig) -> dict:
    summary = dict(result.get("summary", {}))
    summary.update(
        {
            "method": METHOD_DQN,
            "method_label": METHOD_LABEL,
            "method_note": METHOD_NOTE,
            "seed": int(config.seed),
            "rounds": int(config.rounds),
            "physical_rounds": int(config.rounds),
            "decision_interval_ticks": int(config.decision_interval),
            "decision_rounds": int(config.decision_rounds),
            "rsu_cache_capacity": int(config.mrsu_cache_capacity),
            "mrsu_cache_capacity": int(config.mrsu_cache_capacity),
            "frsu_cache_capacity": int(config.frsu_cache_capacity),
            "decision_request_source": "mobility_and_training_history_only",
            "uses_proposed_miss_feedback": False,
            "uses_content_level_feedback": False,
            "uses_predicted_content_demands": False,
            "reward_includes_delay": bool(summary.get("reward_includes_delay", False)),
        }
    )
    return summary


def write_dqn_csv(output_path: Path, results: Dict[str, Dict[str, dict]]) -> None:
    fieldnames = [
        "seed",
        "average_chr",
        "average_local_rsu_chr",
        "method",
        "method_label",
        "rsu_cache_capacity",
        "mrsu_cache_capacity",
        "frsu_cache_capacity",
        "round",
        "physical_tick_start",
        "physical_tick_end",
        "decision_interval_ticks",
        "chr",
        "local_rsu_chr",
        "round_delay_ms",
        "average_delay_ms",
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
        "mrsu_latency_request_count",
        "frsu_latency_request_count",
        "mbs_latency_request_count",
        "request_count",
        "hit_count",
        "mrsu_hit_count",
        "frsu_hit_count",
        "mbs_miss_count",
        "not_covered_count",
        "not_cached_count",
        "mrsu_position",
        "mrsu_speed",
        "mrsu_covered_count",
        "frsu_covered_count",
        "selected_hotspot_id",
        "selected_hotspot_position",
        "lambda_smooth",
        "path_plan_status",
        "path_plan_solver",
        "decision_request_count",
        "evaluation_request_count",
        "decision_request_source",
        "uses_proposed_miss_feedback",
        "uses_content_level_feedback",
        "uses_predicted_content_demands",
        "decision_policy",
        "action_id",
        "mrsu_cache_policy",
        "frsu_cache_policy",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed in sorted(results, key=lambda item: int(item)):
            for capacity in sorted(results[seed], key=lambda item: int(item)):
                result = results[seed][capacity]
                summary = result.get("summary", {})
                average_chr = float(summary.get("achr", 0.0))
                average_local_rsu_chr = float(summary.get("local_rsu_achr", average_chr))
                for round_index, log in enumerate(result.get("round_logs", [])):
                    metrics = log.get("metrics") or {}
                    hotspot = log.get("selected_hotspot") or {}
                    latency = log.get("latency") or {}
                    details = log.get("decision_details") or {}
                    action = details.get("action") or {}
                    writer.writerow(
                        {
                            "seed": int(summary.get("seed", seed)),
                            "average_chr": average_chr,
                            "average_local_rsu_chr": average_local_rsu_chr,
                            "method": METHOD_DQN,
                            "method_label": METHOD_LABEL,
                            "rsu_cache_capacity": int(capacity),
                            "mrsu_cache_capacity": int(summary.get("mrsu_cache_capacity", capacity)),
                            "frsu_cache_capacity": int(summary.get("frsu_cache_capacity", capacity)),
                            "round": int(log.get("round", round_index)),
                            "physical_tick_start": log.get("physical_tick_start", ""),
                            "physical_tick_end": log.get("physical_tick_end", ""),
                            "decision_interval_ticks": log.get("decision_interval_ticks", ""),
                            "chr": float(metrics.get("chr", log.get("chr", 0.0))),
                            "local_rsu_chr": local_rsu_chr_from_row(metrics, log),
                            "round_delay_ms": float(
                                latency.get("average_delay_ms", log.get("round_delay_ms", 0.0))
                            ),
                            "average_delay_ms": float(summary.get("average_delay_ms", 0.0)),
                            "mrsu_average_delay_ms": latency.get("mrsu_average_delay_ms", ""),
                            "frsu_average_delay_ms": latency.get("frsu_average_delay_ms", ""),
                            "mbs_average_delay_ms": latency.get("mbs_average_delay_ms", ""),
                            "mrsu_average_rate_mbps": latency.get("mrsu_average_rate_mbps", ""),
                            "frsu_average_rate_mbps": latency.get("frsu_average_rate_mbps", ""),
                            "mbs_average_rate_mbps": latency.get("mbs_average_rate_mbps", ""),
                            "average_service_distance_m": latency.get("average_service_distance_m", ""),
                            "mrsu_average_distance_m": latency.get("mrsu_average_distance_m", ""),
                            "frsu_average_distance_m": latency.get("frsu_average_distance_m", ""),
                            "mbs_average_distance_m": latency.get("mbs_average_distance_m", ""),
                            "mrsu_latency_request_count": latency.get("mrsu_request_count", ""),
                            "frsu_latency_request_count": latency.get("frsu_request_count", ""),
                            "mbs_latency_request_count": latency.get("mbs_request_count", ""),
                            "request_count": int(metrics.get("request_count", log.get("request_count", 0))),
                            "hit_count": int(metrics.get("hit_count", 0)),
                            "mrsu_hit_count": int(metrics.get("mrsu_hit_count", 0)),
                            "frsu_hit_count": int(metrics.get("frsu_hit_count", 0)),
                            "mbs_miss_count": int(metrics.get("mbs_miss_count", 0)),
                            "not_covered_count": int(metrics.get("not_covered_count", 0)),
                            "not_cached_count": int(metrics.get("not_cached_count", 0)),
                            "mrsu_position": log.get("mrsu_position", ""),
                            "mrsu_speed": log.get("mrsu_speed", ""),
                            "mrsu_covered_count": len(log.get("mrsu_covered") or []),
                            "frsu_covered_count": len(log.get("frsu_covered") or []),
                            "selected_hotspot_id": hotspot.get("hotspot_id", ""),
                            "selected_hotspot_position": hotspot.get("position", ""),
                            "lambda_smooth": log.get("lambda_smooth", ""),
                            "path_plan_status": log.get("path_plan_status", ""),
                            "path_plan_solver": log.get("path_plan_solver", ""),
                            "decision_request_count": log.get("decision_request_count", ""),
                            "evaluation_request_count": log.get("evaluation_request_count", ""),
                            "decision_request_source": log.get("decision_request_source", ""),
                            "uses_proposed_miss_feedback": log.get("uses_proposed_miss_feedback", False),
                            "uses_content_level_feedback": log.get("uses_content_level_feedback", False),
                            "uses_predicted_content_demands": log.get("uses_predicted_content_demands", False),
                            "decision_policy": details.get("policy", ""),
                            "action_id": action.get("action_id", ""),
                            "mrsu_cache_policy": action.get("mrsu_cache_policy", ""),
                            "frsu_cache_policy": action.get("frsu_cache_policy", ""),
                        }
                    )


def aggregate_by_capacity(
    summaries: Dict[str, Dict[str, dict]],
    capacities: List[int],
    seeds: List[int],
) -> Dict[str, dict]:
    aggregate: Dict[str, dict] = {}
    metric_fields = [
        "achr",
        "local_rsu_achr",
        "request_count",
        "hit_count",
        "mrsu_hit_count",
        "frsu_hit_count",
        "mbs_miss_count",
        "not_covered_count",
        "not_cached_count",
        "covered_request_count",
        "total_delay_ms",
        "average_delay_ms",
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
    ]
    for capacity in capacities:
        rows = [
            summaries.get(str(seed), {}).get(str(capacity))
            for seed in seeds
            if summaries.get(str(seed), {}).get(str(capacity))
        ]
        if not rows:
            continue
        row = {
            "method": METHOD_DQN,
            "method_label": METHOD_LABEL,
            "rsu_cache_capacity": int(capacity),
            "mrsu_cache_capacity": int(capacity),
            "frsu_cache_capacity": int(capacity),
            "seed_count": len(rows),
            "seeds": [int(item.get("seed", seeds[idx])) for idx, item in enumerate(rows)],
            "decision_request_source": "mobility_and_training_history_only",
            "uses_proposed_miss_feedback": False,
            "uses_content_level_feedback": False,
            "uses_predicted_content_demands": False,
        }
        for field in metric_fields:
            values = [float(item.get(field, 0.0)) for item in rows if field in item]
            if not values:
                continue
            row[field] = mean(values)
            row[f"{field}_mean"] = mean(values)
            row[f"{field}_std"] = sample_std(values)
            row[f"{field}_min"] = min(values)
            row[f"{field}_max"] = max(values)
        row["round_chr"] = mean_round_series([
            [float(value) for value in item.get("round_chr", [])]
            for item in rows
            if item.get("round_chr")
        ])
        row["round_local_rsu_chr"] = mean_round_series([
            [float(value) for value in item.get("round_local_rsu_chr", [])]
            for item in rows
            if item.get("round_local_rsu_chr")
        ])
        if "local_rsu_achr_mean" not in row:
            hit_mean = float(row.get("hit_count_mean", 0.0))
            not_cached_mean = float(row.get("not_cached_count_mean", 0.0))
            row["local_rsu_achr_mean"] = local_rsu_chr_from_counts(hit_mean, not_cached_mean)
        row["round_delay_ms"] = mean_round_series([
            [float(value) for value in item.get("round_delay_ms", [])]
            for item in rows
            if item.get("round_delay_ms")
        ])
        aggregate[str(capacity)] = row
    return aggregate


def write_aggregate_csv(path: Path, aggregate: Dict[str, dict], capacities: List[int]) -> None:
    fieldnames = [
        "method",
        "method_label",
        "rsu_cache_capacity",
        "mrsu_cache_capacity",
        "frsu_cache_capacity",
        "seed_count",
        "seeds",
        "achr_mean",
        "local_rsu_achr_mean",
        "achr_std",
        "achr_min",
        "achr_max",
        "average_delay_ms_mean",
        "average_delay_ms_std",
        "mrsu_average_delay_ms_mean",
        "frsu_average_delay_ms_mean",
        "mbs_average_delay_ms_mean",
        "mrsu_average_rate_mbps_mean",
        "frsu_average_rate_mbps_mean",
        "mbs_average_rate_mbps_mean",
        "average_service_distance_m_mean",
        "mrsu_average_distance_m_mean",
        "frsu_average_distance_m_mean",
        "mbs_average_distance_m_mean",
        "mrsu_hit_count_mean",
        "frsu_hit_count_mean",
        "mbs_miss_count_mean",
        "not_covered_count_mean",
        "not_cached_count_mean",
        "decision_request_source",
        "uses_proposed_miss_feedback",
        "uses_content_level_feedback",
        "uses_predicted_content_demands",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for capacity in capacities:
            row = aggregate.get(str(capacity))
            if not row:
                continue
            writer.writerow(
                {
                    "method": METHOD_DQN,
                    "method_label": METHOD_LABEL,
                    "rsu_cache_capacity": int(capacity),
                    "mrsu_cache_capacity": int(row.get("mrsu_cache_capacity", capacity)),
                    "frsu_cache_capacity": int(row.get("frsu_cache_capacity", capacity)),
                    "seed_count": int(row.get("seed_count", 0)),
                    "seeds": ",".join(str(seed) for seed in row.get("seeds", [])),
                    "achr_mean": row.get("achr_mean", ""),
                    "local_rsu_achr_mean": row.get("local_rsu_achr_mean", ""),
                    "achr_std": row.get("achr_std", ""),
                    "achr_min": row.get("achr_min", ""),
                    "achr_max": row.get("achr_max", ""),
                    "average_delay_ms_mean": row.get("average_delay_ms_mean", ""),
                    "average_delay_ms_std": row.get("average_delay_ms_std", ""),
                    "mrsu_average_delay_ms_mean": row.get("mrsu_average_delay_ms_mean", ""),
                    "frsu_average_delay_ms_mean": row.get("frsu_average_delay_ms_mean", ""),
                    "mbs_average_delay_ms_mean": row.get("mbs_average_delay_ms_mean", ""),
                    "mrsu_average_rate_mbps_mean": row.get("mrsu_average_rate_mbps_mean", ""),
                    "frsu_average_rate_mbps_mean": row.get("frsu_average_rate_mbps_mean", ""),
                    "mbs_average_rate_mbps_mean": row.get("mbs_average_rate_mbps_mean", ""),
                    "average_service_distance_m_mean": row.get("average_service_distance_m_mean", ""),
                    "mrsu_average_distance_m_mean": row.get("mrsu_average_distance_m_mean", ""),
                    "frsu_average_distance_m_mean": row.get("frsu_average_distance_m_mean", ""),
                    "mbs_average_distance_m_mean": row.get("mbs_average_distance_m_mean", ""),
                    "mrsu_hit_count_mean": row.get("mrsu_hit_count_mean", ""),
                    "frsu_hit_count_mean": row.get("frsu_hit_count_mean", ""),
                    "mbs_miss_count_mean": row.get("mbs_miss_count_mean", ""),
                    "not_covered_count_mean": row.get("not_covered_count_mean", ""),
                    "not_cached_count_mean": row.get("not_cached_count_mean", ""),
                    "decision_request_source": row.get("decision_request_source", ""),
                    "uses_proposed_miss_feedback": False,
                    "uses_content_level_feedback": False,
                    "uses_predicted_content_demands": False,
                }
            )


def write_capacity_csv(path: Path, aggregate: Dict[str, dict], capacities: List[int]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "method_label",
                "rsu_cache_capacity",
                "achr_mean",
                "local_rsu_achr_mean",
                "achr_std",
            ],
        )
        writer.writeheader()
        for capacity in capacities:
            row = aggregate.get(str(capacity))
            if row:
                writer.writerow(
                    {
                        "method": METHOD_DQN,
                        "method_label": METHOD_LABEL,
                        "rsu_cache_capacity": int(capacity),
                        "achr_mean": row.get("achr_mean", ""),
                        "local_rsu_achr_mean": row.get("local_rsu_achr_mean", ""),
                        "achr_std": row.get("achr_std", ""),
                    }
                )


def write_capacity_delay_csv(path: Path, aggregate: Dict[str, dict], capacities: List[int]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "method_label",
                "rsu_cache_capacity",
                "average_delay_ms_mean",
                "average_delay_ms_std",
            ],
        )
        writer.writeheader()
        for capacity in capacities:
            row = aggregate.get(str(capacity))
            if row:
                writer.writerow(
                    {
                        "method": METHOD_DQN,
                        "method_label": METHOD_LABEL,
                        "rsu_cache_capacity": int(capacity),
                        "average_delay_ms_mean": row.get("average_delay_ms_mean", ""),
                        "average_delay_ms_std": row.get("average_delay_ms_std", ""),
                    }
                )


def plot_round_curves(results: Dict[str, Dict[str, dict]], capacities: List[int], output_dir: str) -> None:
    for capacity in capacities:
        rows = []
        for seed_results in results.values():
            result = seed_results.get(str(capacity))
            if result:
                summary = result.get("summary", {})
                round_chr = summary.get("round_local_rsu_chr", summary.get("round_chr", []))
                if round_chr:
                    rows.append([float(value) for value in round_chr])
        if not rows:
            continue
        series = {METHOD_LABEL: [float(value) * 100.0 for value in mean_round_series(rows)]}
        _write_svg_line_chart(
            series,
            str(Path(output_dir) / f"dqn_round_chr_capacity_{int(capacity)}.svg"),
            f"DQN Per-round Average Cache Hit Ratio at RSU Cache Capacity {int(capacity)}",
            "Round",
            "Average Cache Hit Ratio (%)",
            x_labels=[str(idx) for idx in range(len(series[METHOD_LABEL]))],
        )


def plot_round_delay_curves(results: Dict[str, Dict[str, dict]], capacities: List[int], output_dir: str) -> None:
    for capacity in capacities:
        rows = []
        for seed_results in results.values():
            result = seed_results.get(str(capacity))
            if result:
                round_delay = result.get("summary", {}).get("round_delay_ms", [])
                if round_delay:
                    rows.append([float(value) for value in round_delay])
        if not rows:
            continue
        series = {METHOD_LABEL: mean_round_series(rows)}
        _write_svg_line_chart(
            series,
            str(Path(output_dir) / f"dqn_round_delay_capacity_{int(capacity)}.svg"),
            f"DQN Per-round Average Delay at RSU Cache Capacity {int(capacity)}",
            "Round",
            "Average Delay (ms)",
            x_labels=[str(idx) for idx in range(len(series[METHOD_LABEL]))],
        )


def plot_capacity_curve(aggregate: Dict[str, dict], capacities: List[int], output_path: Path) -> None:
    values = [
        float(
            aggregate.get(str(capacity), {}).get(
                "local_rsu_achr_mean",
                aggregate.get(str(capacity), {}).get("achr_mean", 0.0),
            )
        )
        for capacity in capacities
    ]
    _write_svg_line_chart(
        {METHOD_LABEL: [float(value) * 100.0 for value in values]},
        str(output_path),
        "DQN ACHR vs Synchronized RSU Cache Capacity",
        "mRSU/sRSU Cache Capacity",
        "Average Cache Hit Ratio (%)",
        x_labels=[str(capacity) for capacity in capacities],
    )


def plot_capacity_delay_curve(aggregate: Dict[str, dict], capacities: List[int], output_path: Path) -> None:
    values = [float(aggregate.get(str(capacity), {}).get("average_delay_ms_mean", 0.0)) for capacity in capacities]
    _write_svg_line_chart(
        {METHOD_LABEL: values},
        str(output_path),
        "DQN Average Delay vs Synchronized RSU Cache Capacity",
        "mRSU/sRSU Cache Capacity",
        "Average Delay (ms)",
        x_labels=[str(capacity) for capacity in capacities],
    )


def resolve_model_path(model_path: str, model_dir: Path, capacity: int, capacity_count: int) -> Path:
    if model_path:
        path = Path(model_path)
        if capacity_count > 1:
            capacity_path = path.parent / f"dqn_model_capacity_{int(capacity)}.pt"
            if capacity_path.exists():
                return capacity_path
        if not path.exists():
            raise FileNotFoundError(f"DQN model not found: {path}")
        return path
    candidates = [
        model_dir / f"dqn_model_capacity_{int(capacity)}.pt",
        model_dir / "dqn_model.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No DQN model found for capacity {capacity} in {model_dir}")


def find_latest_training_dir(results_root: str) -> Path:
    root = Path(results_root)
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("dqn_training_")
        and (path / "dqn_training_summary.json").exists()
    ] if root.exists() else []
    if not candidates:
        return Path("")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def valid_dir(path: Path) -> bool:
    return bool(path.name) and path.exists() and path.is_dir()


def parse_capacities(text: str, single_capacity: int) -> List[int]:
    if text.strip():
        raw = [int(item.strip()) for item in text.split(",") if item.strip()]
    else:
        raw = [int(single_capacity)]
    capacities: List[int] = []
    for capacity in raw:
        if capacity not in capacities:
            capacities.append(capacity)
    if not capacities:
        raise ValueError("At least one capacity is required.")
    return capacities


def parse_seeds(text: str, single_seed: int) -> List[int]:
    if text.strip():
        raw = [int(item.strip()) for item in text.split(",") if item.strip()]
    else:
        raw = [int(single_seed)]
    seeds: List[int] = []
    for seed in raw:
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def create_output_dir(base_dir: str) -> str:
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"dqn_baseline_{timestamp}"
    counter = 1
    while path.exists():
        path = Path(base_dir) / f"dqn_baseline_{timestamp}_{counter:02d}"
        counter += 1
    path.mkdir(parents=True, exist_ok=False)
    return str(path)


def mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def sample_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return float((sum((value - avg) ** 2 for value in values) / float(len(values) - 1)) ** 0.5)


def mean_round_series(round_rows: List[List[float]]) -> List[float]:
    if not round_rows:
        return []
    max_len = max(len(row) for row in round_rows)
    result = []
    for idx in range(max_len):
        values = [float(row[idx]) for row in round_rows if idx < len(row)]
        result.append(mean(values))
    return result


def local_rsu_chr_from_row(metrics: dict, log: dict = None) -> float:
    log = log or {}
    direct = metrics.get("local_rsu_chr", log.get("local_rsu_chr"))
    if direct not in ("", None):
        return float(direct)
    hit = metrics.get("hit_count")
    if hit in ("", None):
        hit = float(metrics.get("mrsu_hit_count", 0.0)) + float(metrics.get("frsu_hit_count", 0.0))
    not_cached = metrics.get("not_cached_count", log.get("not_cached_count", 0.0))
    return local_rsu_chr_from_counts(float(hit or 0.0), float(not_cached or 0.0))


if __name__ == "__main__":
    main()
