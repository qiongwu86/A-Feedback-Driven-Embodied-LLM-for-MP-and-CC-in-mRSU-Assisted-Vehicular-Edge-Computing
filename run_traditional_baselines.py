from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from communication.latency_model import CV2XLatencyModel, LatencyModelConfig
from baselines.traditional_baselines import (
    METHOD_CPSAT,
    METHOD_LABELS,
    METHOD_NOTES,
    METHOD_THOMPSON,
    SUPPORTED_METHODS,
    run_traditional_baseline,
)
from simulation.config import MRSUSimulationConfig
from simulation.metrics import local_rsu_chr_from_counts


DEFAULT_METHODS = (METHOD_THOMPSON, METHOD_CPSAT)
DEFAULT_SINGLE_CAPACITY = 200

METHOD_ALIASES = {
    "thompson": METHOD_THOMPSON,
    "thompson_sampling": METHOD_THOMPSON,
    "ts": METHOD_THOMPSON,
    "cp-sat": METHOD_CPSAT,
    "cpsat": METHOD_CPSAT,
    "cp_sat": METHOD_CPSAT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Thompson Sampling and CP-SAT baselines without LLM/DQN."
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods: thompson, cp-sat.",
    )
    parser.add_argument(
        "--capacities",
        type=str,
        default="",
        help=(
            "Comma-separated synchronized mRSU/fRSU cache capacities. "
            "If omitted, the script runs one capacity from --rsu-cache, default 200."
        ),
    )
    parser.add_argument(
        "--rsu-cache",
        type=int,
        default=DEFAULT_SINGLE_CAPACITY,
        help="Single synchronized capacity used when --capacities is omitted.",
    )
    parser.add_argument("--rounds", type=int, default=100, help="Total physical simulation ticks.")
    parser.add_argument("--decision-interval", type=int, default=10, help="Physical ticks per request/cache decision window.")
    parser.add_argument("--seed", type=int, default=42, help="Single random seed used when --seeds is omitted.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated random seeds, for example 7,42,2026. Overrides --seed when provided.",
    )
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
    parser.add_argument("--cp-sat-time-limit", type=float, default=10.0)
    parser.add_argument("--latency-content-size-kbit", type=float, default=800.0)
    parser.add_argument("--latency-bandwidth-mhz", type=float, default=10.0)
    parser.add_argument(
        "--latency-rsu-distance-loss",
        type=float,
        default=16.0,
        help="RSU path-loss distance coefficient in dB per log10(distance). Larger means stronger distance sensitivity.",
    )
    parser.add_argument("--latency-cloud-backhaul-rate-mbps", type=float, default=80.0)
    parser.add_argument("--latency-cloud-extra-ms", type=float, default=20.0)
    parser.add_argument(
        "--ours-result-dir",
        type=str,
        default="",
        help="Directory containing ours_capacity_scan_summary.csv. If omitted, latest Ours scan is auto-detected.",
    )
    parser.add_argument("--skip-ours", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace, capacity: int, output_dir: str, seed: int = None) -> MRSUSimulationConfig:
    seed_value = int(args.seed if seed is None else seed)
    return MRSUSimulationConfig(
        seed=seed_value,
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
        mrsu_cache_capacity=int(capacity),
        frsu_cache_capacity=int(capacity),
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


def save_json(data: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_svg_line_chart(
    series: Dict[str, List[float]],
    output_path: str,
    title: str,
    x_label: str,
    y_label: str,
    x_labels: List[str] = None,
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 180, 50, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    all_values = [value for values in series.values() for value in values]
    y_min = 0.0
    y_max = max(all_values) if all_values else 1.0
    y_max = max(0.05, y_max * 1.08)
    max_len = max((len(values) for values in series.values()), default=1)

    def x_coord(idx: int) -> float:
        if max_len <= 1:
            return left
        return left + idx * plot_w / (max_len - 1)

    def y_coord(value: float) -> float:
        return top + (y_max - value) * plot_h / max(y_max - y_min, 1e-9)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_coord(value)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ddd"/>')
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="12">{value:.2f}</text>'
        )

    if x_labels:
        for idx, label in enumerate(x_labels):
            x = x_coord(idx)
            lines.append(
                f'<text x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle" '
                f'font-family="Arial" font-size="12">{label}</text>'
            )

    for s_idx, (name, values) in enumerate(series.items()):
        color = colors[s_idx % len(colors)]
        points = " ".join(f"{x_coord(idx):.2f},{y_coord(value):.2f}" for idx, value in enumerate(values))
        if points:
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
            for idx, value in enumerate(values):
                lines.append(f'<circle cx="{x_coord(idx):.2f}" cy="{y_coord(value):.2f}" r="3" fill="{color}"/>')
        legend_y = top + 20 + s_idx * 22
        legend_x = left + plot_w + 25
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="2"/>'
        )
        lines.append(f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-family="Arial" font-size="12">{name}</text>')

    lines.append(
        f'<text x="{left + plot_w/2}" y="{height - 20}" text-anchor="middle" '
        f'font-family="Arial" font-size="14">{x_label}</text>'
    )
    lines.append(
        f'<text x="22" y="{top + plot_h/2}" text-anchor="middle" '
        f'transform="rotate(-90 22 {top + plot_h/2})" '
        f'font-family="Arial" font-size="14">{y_label}</text>'
    )
    lines.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    capacities = parse_capacities(args.capacities, args.rsu_cache)
    seeds = parse_seeds(args.seeds, args.seed)
    run_output_dir = create_output_dir(args.output_dir)
    data_dir = Path(run_output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=False)

    print("Traditional baseline config:")
    print(
        json.dumps(
            {
                "methods": methods,
                "method_labels": {method: METHOD_LABELS[method] for method in methods},
                "capacities": capacities,
                "capacity_rule": "mrsu_cache_capacity = frsu_cache_capacity = capacity",
                "physical_rounds": args.rounds,
                "decision_interval": args.decision_interval,
                "decision_rounds": (args.rounds + args.decision_interval - 1) // max(args.decision_interval, 1),
                "seeds": seeds,
                "seed": seeds[0] if len(seeds) == 1 else None,
                "road_topology": "circular_one_way",
                "decision_request_source": "predicted_history_signal_no_proposed_feedback",
                "uses_proposed_miss_feedback": False,
                "latency_model": build_latency_model(args).config.to_dict(),
                "output_dir": run_output_dir,
                "data_dir": str(data_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("No LLM API calls or DQN evaluation will be made.")

    results: Dict[str, Dict[str, Dict[str, dict]]] = {method: {} for method in methods}
    summaries: Dict[str, Dict[str, Dict[str, dict]]] = {method: {} for method in methods}
    configs: Dict[str, dict] = {}
    for seed in seeds:
        configs[str(seed)] = {}
        print(f"\n=== random seed: {seed} ===")
        for capacity in capacities:
            config = build_config(args, capacity, run_output_dir, seed=seed)
            configs[str(seed)][str(capacity)] = asdict(config)
            print(f"\n=== synchronized RSU cache capacity: {capacity} ===")
            for method in methods:
                label = METHOD_LABELS[method]
                print(f"Running {label} with seed={seed}...")
                result = run_traditional_baseline(
                    config=config,
                    method=method,
                    verbose=not args.quiet,
                    cp_sat_time_limit=args.cp_sat_time_limit,
                    latency_model=build_latency_model(args),
                )
                summary = normalize_summary(result, method, config)
                result["summary"] = summary
                results[method].setdefault(str(seed), {})[str(capacity)] = result
                summaries[method].setdefault(str(seed), {})[str(capacity)] = summary
                write_method_csv(data_dir / f"{method}.csv", method, results[method])
                print(
                    f"Finished {label} seed={seed} C={capacity}: "
                    f"LocalACHR={float(summary.get('local_rsu_achr', summary['achr'])):.4f} "
                    f"Delay={float(summary.get('average_delay_ms', 0.0)):.2f}ms "
                    f"mRSU_hit={summary['mrsu_hit_count']} "
                    f"sRSU_hit={summary['frsu_hit_count']} "
                    f"MBS_miss={summary['mbs_miss_count']}"
                )

    aggregate_summaries = aggregate_summaries_by_capacity(summaries, methods, capacities, seeds)
    write_aggregate_summary_csv(data_dir / "aggregate_summary.csv", aggregate_summaries, methods, capacities)
    baseline_series = build_achr_series(aggregate_summaries, capacities)
    plot_capacity_curve(
        baseline_series,
        capacities,
        str(Path(run_output_dir) / "traditional_baselines_achr_vs_capacity.svg"),
        title="Baseline ACHR vs Synchronized RSU Cache Capacity",
    )
    plot_round_curves(results, capacities, run_output_dir)
    delay_series = build_delay_series(aggregate_summaries, capacities)
    if delay_series:
        plot_delay_capacity_curve(
            delay_series,
            capacities,
            str(Path(run_output_dir) / "traditional_baselines_delay_vs_capacity.svg"),
            title="Baseline Average Delay vs Synchronized RSU Cache Capacity",
        )
        plot_round_delay_curves(results, capacities, run_output_dir)

    ours_series, ours_note = ({}, "")
    if not args.skip_ours:
        ours_series, ours_note = read_ours_capacity_series(args.ours_result_dir, args.output_dir)
        if ours_note:
            print(ours_note)
        write_with_ours_curve(
            baseline_series=baseline_series,
            ours_series=ours_series,
            capacities=capacities,
            output_dir=run_output_dir,
        )

    save_json(
        {
            "experiment": "traditional_baselines",
            "seeds": seeds,
            "config_by_seed_capacity": configs,
            "methods": methods,
            "method_labels": METHOD_LABELS,
            "method_notes": METHOD_NOTES,
            "aggregate_summaries": aggregate_summaries,
            "summaries_by_seed_capacity": summaries,
            "results_by_seed_capacity": results,
            "ours_note": ours_note,
            "data_dir": str(data_dir),
        },
        str(Path(run_output_dir) / "traditional_baselines_results.json"),
    )
    save_json(
        {
            "capacities": capacities,
            "seeds": seeds,
            "baseline_series": baseline_series,
            "baseline_delay_series": delay_series,
            "aggregate_summaries": aggregate_summaries,
            "ours_series": [
                {"capacity": capacity, "achr": ours_series[capacity]}
                for capacity in sorted(ours_series)
            ],
        },
        str(Path(run_output_dir) / "traditional_baselines_plot_series.json"),
    )

    print("\nExperiment finished.")
    print(f"Results saved to: {os.path.abspath(run_output_dir)}")


def normalize_summary(result: dict, method: str, config: MRSUSimulationConfig) -> dict:
    summary = dict(result.get("summary", {}))
    summary.update(
        {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "method_note": METHOD_NOTES[method],
            "decision_request_source": summary.get(
                "decision_request_source",
                "predicted_history_signal_no_proposed_feedback",
            ),
            "uses_proposed_miss_feedback": bool(summary.get("uses_proposed_miss_feedback", False)),
            "seed": int(config.seed),
            "rounds": int(config.rounds),
            "physical_rounds": int(config.rounds),
            "decision_interval_ticks": int(config.decision_interval),
            "decision_rounds": int(config.decision_rounds),
            "mrsu_cache_capacity": int(config.mrsu_cache_capacity),
            "frsu_cache_capacity": int(config.frsu_cache_capacity),
        }
    )
    return summary


def write_method_csv(
    output_path: Path,
    method: str,
    method_results: Dict[str, Dict[str, dict]],
) -> None:
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
        "decision_policy",
        "fallback_to_topk",
        "fallback_reason",
        "solver_status",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed in sorted(method_results, key=lambda item: int(item)):
            capacity_results = method_results[seed]
            for capacity in sorted(capacity_results, key=lambda item: int(item)):
                result = capacity_results[capacity]
                summary = result.get("summary", {})
                average_chr = float(summary.get("achr", 0.0))
                average_local_rsu_chr = float(summary.get("local_rsu_achr", average_chr))
                seed_value = int(summary.get("seed", seed))
                for round_index, log in enumerate(result.get("round_logs", [])):
                    metrics = log.get("metrics") or {}
                    latency = log.get("latency") or {}
                    selected_hotspot = log.get("selected_hotspot") or {}
                    details = log.get("decision_details") or {}
                    writer.writerow(
                        {
                            "seed": seed_value,
                            "average_chr": average_chr,
                            "average_local_rsu_chr": average_local_rsu_chr,
                            "method": method,
                            "method_label": summary.get("method_label", METHOD_LABELS.get(method, method)),
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
                            "latency_scope": latency.get("latency_scope", summary.get("latency_scope", "")),
                            "latency_request_count": latency.get("request_count", ""),
                            "excluded_not_covered_request_count": latency.get(
                                "excluded_not_covered_request_count", ""
                            ),
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
                            "selected_hotspot_id": selected_hotspot.get("hotspot_id", ""),
                            "selected_hotspot_position": selected_hotspot.get("position", ""),
                            "lambda_smooth": log.get("lambda_smooth", ""),
                            "path_plan_status": log.get("path_plan_status", ""),
                            "path_plan_solver": log.get("path_plan_solver", ""),
                            "decision_request_count": log.get("decision_request_count", ""),
                            "evaluation_request_count": log.get("evaluation_request_count", ""),
                            "decision_request_source": log.get(
                                "decision_request_source",
                                summary.get("decision_request_source", ""),
                            ),
                            "uses_proposed_miss_feedback": log.get(
                                "uses_proposed_miss_feedback",
                                summary.get("uses_proposed_miss_feedback", False),
                            ),
                            "decision_policy": details.get("policy", ""),
                            "fallback_to_topk": details.get("fallback_to_topk", ""),
                            "fallback_reason": details.get("fallback_reason", ""),
                            "solver_status": details.get("solver_status", ""),
                        }
                    )


def aggregate_summaries_by_capacity(
    summaries: Dict[str, Dict[str, Dict[str, dict]]],
    methods: List[str],
    capacities: List[int],
    seeds: List[int],
) -> Dict[str, Dict[str, dict]]:
    aggregate: Dict[str, Dict[str, dict]] = {method: {} for method in methods}
    metric_fields = [
        "achr",
        "local_rsu_achr",
        "request_count",
        "hit_count",
        "covered_request_count",
        "mrsu_hit_count",
        "frsu_hit_count",
        "mbs_miss_count",
        "not_covered_count",
        "not_cached_count",
        "total_delay_ms",
        "latency_request_count",
        "excluded_not_covered_request_count",
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
        "cache_update_count",
        "fallback_to_topk_count",
    ]
    for method in methods:
        for capacity in capacities:
            row_pairs = [
                (seed, summaries.get(method, {}).get(str(seed), {}).get(str(capacity)))
                for seed in seeds
            ]
            row_pairs = [(seed, row) for seed, row in row_pairs if row]
            rows = [row for _, row in row_pairs]
            if not rows:
                continue
            seed_values = [int(row.get("seed", seed)) for seed, row in row_pairs]
            aggregate_row = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "method_note": METHOD_NOTES[method],
                "rsu_cache_capacity": int(capacity),
                "mrsu_cache_capacity": int(capacity),
                "frsu_cache_capacity": int(capacity),
                "seed_count": len(rows),
                "seeds": seed_values,
                "decision_request_source": "predicted_history_signal_no_proposed_feedback",
                "uses_proposed_miss_feedback": False,
            }
            for field in metric_fields:
                values = [float(row.get(field, 0.0)) for row in rows if field in row]
                if not values:
                    continue
                mean_value = _mean(values)
                aggregate_row[field] = mean_value
                aggregate_row[f"{field}_mean"] = mean_value
                aggregate_row[f"{field}_std"] = _sample_std(values)
                aggregate_row[f"{field}_min"] = min(values)
                aggregate_row[f"{field}_max"] = max(values)
            round_rows = [
                [float(value) for value in row.get("round_chr", [])]
                for row in rows
                if row.get("round_chr")
            ]
            aggregate_row["round_chr"] = _mean_round_series(round_rows)
            local_round_rows = [
                [float(value) for value in row.get("round_local_rsu_chr", [])]
                for row in rows
                if row.get("round_local_rsu_chr")
            ]
            aggregate_row["round_local_rsu_chr"] = _mean_round_series(local_round_rows)
            if "local_rsu_achr_mean" not in aggregate_row:
                hit_mean = float(aggregate_row.get("hit_count_mean", 0.0))
                not_cached_mean = float(aggregate_row.get("not_cached_count_mean", 0.0))
                aggregate_row["local_rsu_achr_mean"] = local_rsu_chr_from_counts(hit_mean, not_cached_mean)
            delay_rows = [
                [float(value) for value in row.get("round_delay_ms", [])]
                for row in rows
                if row.get("round_delay_ms")
            ]
            aggregate_row["round_delay_ms"] = _mean_round_series(delay_rows)
            aggregate[method][str(capacity)] = aggregate_row
    return aggregate


def write_aggregate_summary_csv(
    output_path: Path,
    aggregate_summaries: Dict[str, Dict[str, dict]],
    methods: List[str],
    capacities: List[int],
) -> None:
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
        "latency_request_count_mean",
        "excluded_not_covered_request_count_mean",
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
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            for capacity in capacities:
                row = aggregate_summaries.get(method, {}).get(str(capacity))
                if not row:
                    continue
                writer.writerow(
                    {
                        "method": method,
                        "method_label": row.get("method_label", METHOD_LABELS.get(method, method)),
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
                        "latency_request_count_mean": row.get("latency_request_count_mean", ""),
                        "excluded_not_covered_request_count_mean": row.get(
                            "excluded_not_covered_request_count_mean", ""
                        ),
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
                        "uses_proposed_miss_feedback": row.get("uses_proposed_miss_feedback", False),
                    }
                )


def build_achr_series(
    summaries: Dict[str, Dict[str, dict]],
    capacities: List[int],
) -> Dict[str, List[float]]:
    series: Dict[str, List[float]] = {}
    for method in DEFAULT_METHODS:
        if method not in summaries:
            continue
        label = METHOD_LABELS[method]
        series[label] = [
            float(
                summaries[method]
                .get(str(capacity), {})
                .get("local_rsu_achr_mean", summaries[method].get(str(capacity), {}).get("achr_mean", 0.0))
            )
            for capacity in capacities
        ]
    return series


def build_delay_series(
    summaries: Dict[str, Dict[str, dict]],
    capacities: List[int],
) -> Dict[str, List[float]]:
    series: Dict[str, List[float]] = {}
    for method in DEFAULT_METHODS:
        if method not in summaries:
            continue
        values = []
        has_value = False
        for capacity in capacities:
            value = summaries[method].get(str(capacity), {}).get("average_delay_ms_mean")
            if value not in ("", None):
                has_value = True
            values.append(float(value or 0.0))
        if has_value:
            series[METHOD_LABELS[method]] = values
    return series


def plot_capacity_curve(
    series: Dict[str, List[float]],
    capacities: List[int],
    output_path: str,
    title: str,
) -> None:
    _write_svg_line_chart(
        percent_series(series),
        output_path,
        title,
        "mRSU/sRSU Cache Capacity",
        "Average Cache Hit Ratio (%)",
        x_labels=[str(capacity) for capacity in capacities],
    )


def plot_delay_capacity_curve(
    series: Dict[str, List[float]],
    capacities: List[int],
    output_path: str,
    title: str,
) -> None:
    _write_svg_line_chart(
        series,
        output_path,
        title,
        "mRSU/sRSU Cache Capacity",
        "Average Delay (ms)",
        x_labels=[str(capacity) for capacity in capacities],
    )


def plot_round_curves(
    results: Dict[str, Dict[str, Dict[str, dict]]],
    capacities: List[int],
    output_dir: str,
) -> None:
    for capacity in capacities:
        series = {}
        max_rounds = 0
        for method in DEFAULT_METHODS:
            round_rows = []
            for seed_results in results.get(method, {}).values():
                result = seed_results.get(str(capacity))
                if not result:
                    continue
                round_chr = result.get("summary", {}).get("round_local_rsu_chr", result.get("summary", {}).get("round_chr", []))
                if round_chr:
                    round_rows.append([float(value) for value in round_chr])
            if not round_rows:
                continue
            mean_round_chr = _mean_round_series(round_rows)
            if mean_round_chr:
                series[METHOD_LABELS[method]] = [float(value) * 100.0 for value in mean_round_chr]
                max_rounds = max(max_rounds, len(mean_round_chr))
        if not series:
            continue
        _write_svg_line_chart(
            series,
            str(Path(output_dir) / f"traditional_baselines_round_chr_capacity_{capacity}.svg"),
            f"Per-round Average Cache Hit Ratio at RSU Cache Capacity {capacity}",
            "Round",
            "Average Cache Hit Ratio (%)",
            x_labels=[str(idx) for idx in range(max_rounds)],
        )


def plot_round_delay_curves(
    results: Dict[str, Dict[str, Dict[str, dict]]],
    capacities: List[int],
    output_dir: str,
) -> None:
    for capacity in capacities:
        series = {}
        max_rounds = 0
        for method in DEFAULT_METHODS:
            round_rows = []
            for seed_results in results.get(method, {}).values():
                result = seed_results.get(str(capacity))
                if not result:
                    continue
                round_delay = result.get("summary", {}).get("round_delay_ms", [])
                if round_delay:
                    round_rows.append([float(value) for value in round_delay])
            if not round_rows:
                continue
            mean_round_delay = _mean_round_series(round_rows)
            if mean_round_delay:
                series[METHOD_LABELS[method]] = mean_round_delay
                max_rounds = max(max_rounds, len(mean_round_delay))
        if not series:
            continue
        _write_svg_line_chart(
            series,
            str(Path(output_dir) / f"traditional_baselines_round_delay_capacity_{capacity}.svg"),
            f"Per-round Average Delay at RSU Cache Capacity {capacity}",
            "Round",
            "Average Delay (ms)",
            x_labels=[str(idx) for idx in range(max_rounds)],
        )


def write_with_ours_curve(
    baseline_series: Dict[str, List[float]],
    ours_series: Dict[int, float],
    capacities: List[int],
    output_dir: str,
) -> None:
    common_capacities = [capacity for capacity in capacities if capacity in ours_series]
    if not common_capacities:
        print("Ours ACHR curve was not plotted because no matching Ours capacities were found.")
        return
    capacity_index = {capacity: idx for idx, capacity in enumerate(capacities)}
    combined = {
        label: [values[capacity_index[capacity]] for capacity in common_capacities]
        for label, values in baseline_series.items()
    }
    combined["Ours"] = [float(ours_series[capacity]) for capacity in common_capacities]
    plot_capacity_curve(
        combined,
        common_capacities,
        str(Path(output_dir) / "traditional_baselines_with_ours_achr_vs_capacity.svg"),
        "ACHR vs Synchronized RSU Cache Capacity",
    )


def percent_series(series: Dict[str, List[float]]) -> Dict[str, List[float]]:
    return {
        label: [float(value) * 100.0 for value in values]
        for label, values in series.items()
    }


def read_ours_capacity_series(ours_result_dir: str, results_root: str) -> Tuple[Dict[int, float], str]:
    summary_path = Path(ours_result_dir) / "ours_capacity_scan_summary.csv" if ours_result_dir else None
    if summary_path is None or not summary_path.exists():
        latest = find_latest_ours_result_dir(results_root)
        summary_path = Path(latest) / "ours_capacity_scan_summary.csv" if latest else None
    if summary_path is None or not summary_path.exists():
        latest = find_latest_embodied_result_dir(results_root)
        summary_path = Path(latest) / "具身智能ACHR汇总表.csv" if latest else None
    if summary_path is None or not summary_path.exists():
        return {}, "Ours capacity summary not found; only baseline curves were plotted."

    rows: Dict[int, float] = {}
    with open(summary_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                capacity = int(float(row.get("mrsu_cache_capacity") or row.get("rsu_cache_capacity")))
                if row.get("method") and row.get("method") not in ("tool_agent", "ours"):
                    continue
                value = row.get("local_rsu_achr")
                if value in ("", None):
                    hit = float(row.get("mrsu_hit_count", 0.0)) + float(row.get("frsu_hit_count", 0.0))
                    not_cached = float(row.get("not_cached_count", 0.0))
                    value = local_rsu_chr_from_counts(hit, not_cached)
                rows[capacity] = float(value)
            except (KeyError, TypeError, ValueError):
                continue
    return rows, f"Read Ours ACHR curve from: {summary_path}"


def find_latest_ours_result_dir(results_root: str) -> str:
    root = Path(results_root)
    if not root.exists():
        return ""
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("ours_capacity_scan")
        and (path / "ours_capacity_scan_summary.csv").exists()
    ]
    if not candidates:
        return ""
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])


def find_latest_embodied_result_dir(results_root: str) -> str:
    root = Path(results_root)
    if not root.exists():
        return ""
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("具身智能")
        and (path / "具身智能ACHR汇总表.csv").exists()
    ]
    if not candidates:
        return ""
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])


def parse_methods(text: str) -> List[str]:
    methods: List[str] = []
    for item in text.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in METHOD_ALIASES:
            raise ValueError(f"Unknown method '{item}'. Use thompson or cp-sat.")
        method = METHOD_ALIASES[key]
        if method not in methods:
            methods.append(method)
    if not methods:
        raise ValueError("At least one baseline method is required.")
    for method in methods:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported baseline method: {method}")
    return methods


def parse_capacities(text: str, single_capacity: int) -> List[int]:
    if text.strip():
        capacities = [int(item.strip()) for item in text.split(",") if item.strip()]
    else:
        capacities = [int(single_capacity)]
    if not capacities:
        raise ValueError("At least one synchronized RSU cache capacity is required.")
    return capacities


def parse_seeds(text: str, single_seed: int) -> List[int]:
    if text.strip():
        raw_seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    else:
        raw_seeds = [int(single_seed)]
    seeds: List[int] = []
    for seed in raw_seeds:
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _sample_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    variance = sum((float(value) - avg) ** 2 for value in values) / float(len(values) - 1)
    return float(variance ** 0.5)


def _mean_round_series(round_rows: List[List[float]]) -> List[float]:
    if not round_rows:
        return []
    max_len = max(len(row) for row in round_rows)
    result = []
    for idx in range(max_len):
        values = [float(row[idx]) for row in round_rows if idx < len(row)]
        result.append(_mean(values))
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


def create_output_dir(base_dir: str) -> str:
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"traditional_baselines_{timestamp}"
    counter = 1
    while path.exists():
        path = Path(base_dir) / f"traditional_baselines_{timestamp}_{counter:02d}"
        counter += 1
    path.mkdir(parents=True, exist_ok=False)
    return str(path)


if __name__ == "__main__":
    main()
