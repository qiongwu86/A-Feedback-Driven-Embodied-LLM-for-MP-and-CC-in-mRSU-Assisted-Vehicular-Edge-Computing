from __future__ import annotations

import argparse
import csv
import html
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from main_mrsu_tool_simulation import (
    TOOL_AGENT_LABEL,
    TOOL_AGENT_METHOD,
    build_agent,
    build_latency_model,
    describe_agent,
    run_method,
)
from simulation.config import MRSUSimulationConfig


DEFAULT_VEHICLE_COUNTS = [50]
DEFAULT_CACHE_CAPACITY = 200
VEHICLE_SCAN_CHART_FILENAME = "\u8f66\u8f86\u6570\u626b\u63cfCHR\u65f6\u5ef6\u53cc\u8f74\u67f1\u72b6\u56fe.svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan the number of vehicles using only the FD-EMC main method."
    )
    parser.add_argument(
        "--vehicle-counts",
        type=str,
        default=",".join(str(value) for value in DEFAULT_VEHICLE_COUNTS),
        help=(
            "Comma-separated vehicle counts. Defaults to 50. "
            "Use 10,20,30,40,50 for the vehicle-count scan."
        ),
    )
    parser.add_argument(
        "--user-num",
        type=int,
        default=0,
        help="Number of users. Defaults to the current vehicle count when omitted or <= 0.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Single seed used when --seeds is omitted.")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds, for example 7,42,2026.")
    parser.add_argument("--rounds", type=int, default=100, help="Total physical simulation ticks.")
    parser.add_argument("--decision-interval", type=int, default=10, help="Physical ticks per FD-EMC decision window.")

    parser.add_argument("--rsu-cache", type=int, default=DEFAULT_CACHE_CAPACITY)
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

    parser.add_argument("--agent", choices=["auto", "mock", "llm", "gemini", "gemini-rest"], default="auto")
    parser.add_argument("--base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", type=str, default="")
    parser.add_argument("--model-name", type=str, default="qwen3.7-flash")
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

    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vehicle_counts = parse_int_list(args.vehicle_counts, "vehicle-counts")
    seeds = parse_seeds(args.seeds, args.seed)
    output_dir = create_output_dir(args.output_dir)
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=False)
    agent = build_agent(args)
    latency_model = build_latency_model(args)

    print("Vehicle-count scan config:")
    print(
        json.dumps(
            {
                "method": TOOL_AGENT_METHOD,
                "method_label": TOOL_AGENT_LABEL,
                "vehicle_counts": vehicle_counts,
                "seeds": seeds,
                "rsu_cache_capacity": int(args.rsu_cache),
                "user_num_rule": "user_num = vehicle_num" if int(args.user_num) <= 0 else "fixed user_num",
                "fixed_user_num": int(args.user_num) if int(args.user_num) > 0 else None,
                "physical_rounds": int(args.rounds),
                "decision_interval": int(args.decision_interval),
                "decision_rounds": (args.rounds + args.decision_interval - 1) // max(args.decision_interval, 1),
                "latency_model": latency_model.config.to_dict(),
                "agent": describe_agent(agent),
                "output_dir": output_dir,
                "data_dir": str(data_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    results: Dict[str, Dict[str, dict]] = {}
    summaries: Dict[str, Dict[str, dict]] = {}
    configs: Dict[str, Dict[str, dict]] = {}

    for seed in seeds:
        results[str(seed)] = {}
        summaries[str(seed)] = {}
        configs[str(seed)] = {}
        print(f"\n=== seed={seed} ===")
        for vehicle_count in vehicle_counts:
            config = build_config(args, seed=seed, vehicle_count=vehicle_count, output_dir=output_dir)
            configs[str(seed)][str(vehicle_count)] = asdict(config)
            print(
                f"\n=== FD-EMC vehicle_num={config.vehicle_num}, "
                f"user_num={config.user_num}, C={config.mrsu_cache_capacity} ==="
            )
            result = run_method(
                config=config,
                method=TOOL_AGENT_METHOD,
                agent=agent,
                verbose=not args.quiet,
                output_dir=output_dir,
                cache_update_candidate_limit=args.cache_update_candidate_limit,
                latency_model=latency_model,
            )
            summary = normalize_summary(result, config)
            result["summary"] = summary
            results[str(seed)][str(vehicle_count)] = result
            summaries[str(seed)][str(vehicle_count)] = summary
            write_run_csv(data_dir / "vehicle_count_scan_runs.csv", summaries, vehicle_counts, seeds)
            write_round_csv(data_dir / "vehicle_count_scan_rounds.csv", results, vehicle_counts, seeds)
            print(
                f"Finished FD-EMC seed={seed} vehicles={vehicle_count}: "
                f"ACHR={summary['achr']:.4f} "
                f"Delay={float(summary.get('average_delay_ms', 0.0)):.2f}ms "
                f"mRSU_hit={summary['mrsu_hit_count']} "
                f"sRSU_hit={summary['frsu_hit_count']} "
                f"MBS_miss={summary['mbs_miss_count']}"
            )

    aggregate = aggregate_by_vehicle_count(summaries, vehicle_counts, seeds)
    write_summary_csv(data_dir / "vehicle_count_scan_summary.csv", aggregate, vehicle_counts)
    write_plot_csv(data_dir / "vehicle_count_scan_plot_data.csv", aggregate, vehicle_counts)
    chart_path = Path(output_dir) / VEHICLE_SCAN_CHART_FILENAME
    write_dual_axis_bar_chart(
        vehicle_counts=vehicle_counts,
        chr_values=[
            float(aggregate.get(str(vehicle_count), {}).get("achr_mean", 0.0))
            for vehicle_count in vehicle_counts
        ],
        delay_values=[
            float(aggregate.get(str(vehicle_count), {}).get("average_delay_ms_mean", 0.0))
            for vehicle_count in vehicle_counts
        ],
        output_path=chart_path,
        title="FD-EMC under Different Numbers of Vehicles",
    )

    save_json(
        {
            "experiment": "vehicle_count_scan",
            "method": TOOL_AGENT_METHOD,
            "method_label": TOOL_AGENT_LABEL,
            "vehicle_counts": vehicle_counts,
            "seeds": seeds,
            "rsu_cache_capacity": int(args.rsu_cache),
            "latency_model": latency_model.config.to_dict(),
            "config_by_seed_vehicle_count": configs,
            "aggregate_summaries": aggregate,
            "summaries_by_seed_vehicle_count": summaries,
            "results_by_seed_vehicle_count": results,
            "data_dir": str(data_dir),
            "plot": str(chart_path),
        },
        str(Path(output_dir) / "vehicle_count_scan_results.json"),
    )
    print("\nVehicle-count scan finished.")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print(f"Dual-axis bar chart: {chart_path}")


def build_config(
    args: argparse.Namespace,
    seed: int,
    vehicle_count: int,
    output_dir: str,
) -> MRSUSimulationConfig:
    user_num = int(args.user_num) if int(args.user_num) > 0 else int(vehicle_count)
    return MRSUSimulationConfig(
        seed=int(seed),
        rounds=int(args.rounds),
        decision_interval=int(args.decision_interval),
        road_length=float(args.road_length),
        vehicle_num=int(vehicle_count),
        user_num=int(user_num),
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
        mrsu_cache_capacity=int(args.rsu_cache),
        frsu_cache_capacity=int(args.rsu_cache),
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


def normalize_summary(result: dict, config: MRSUSimulationConfig) -> dict:
    summary = dict(result.get("summary", {}))
    summary.update(
        {
            "method": TOOL_AGENT_METHOD,
            "method_label": TOOL_AGENT_LABEL,
            "seed": int(config.seed),
            "rounds": int(config.rounds),
            "physical_rounds": int(config.rounds),
            "decision_interval_ticks": int(config.decision_interval),
            "decision_rounds": int(config.decision_rounds),
            "vehicle_num": int(config.vehicle_num),
            "user_num": int(config.user_num),
            "rsu_cache_capacity": int(config.mrsu_cache_capacity),
            "mrsu_cache_capacity": int(config.mrsu_cache_capacity),
            "frsu_cache_capacity": int(config.frsu_cache_capacity),
        }
    )
    return summary


def aggregate_by_vehicle_count(
    summaries: Dict[str, Dict[str, dict]],
    vehicle_counts: List[int],
    seeds: List[int],
) -> Dict[str, dict]:
    aggregate: Dict[str, dict] = {}
    metric_fields = [
        "achr",
        "request_count",
        "hit_count",
        "mrsu_hit_count",
        "frsu_hit_count",
        "mbs_miss_count",
        "not_covered_count",
        "not_cached_count",
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
    ]
    for vehicle_count in vehicle_counts:
        rows = [
            summaries.get(str(seed), {}).get(str(vehicle_count))
            for seed in seeds
            if summaries.get(str(seed), {}).get(str(vehicle_count))
        ]
        if not rows:
            continue
        row = {
            "method": TOOL_AGENT_METHOD,
            "method_label": TOOL_AGENT_LABEL,
            "vehicle_num": int(vehicle_count),
            "user_num": int(rows[0].get("user_num", vehicle_count)),
            "seed_count": len(rows),
            "seeds": [int(item.get("seed", seeds[idx])) for idx, item in enumerate(rows)],
            "rsu_cache_capacity": int(rows[0].get("rsu_cache_capacity", DEFAULT_CACHE_CAPACITY)),
        }
        for field in metric_fields:
            values = [float(item.get(field, 0.0)) for item in rows if field in item]
            if not values:
                continue
            avg = mean(values)
            row[field] = avg
            row[f"{field}_mean"] = avg
            row[f"{field}_std"] = sample_std(values)
            row[f"{field}_min"] = min(values)
            row[f"{field}_max"] = max(values)
        round_chr_rows = [
            [float(value) for value in item.get("round_chr", [])]
            for item in rows
            if item.get("round_chr")
        ]
        round_delay_rows = [
            [float(value) for value in item.get("round_delay_ms", [])]
            for item in rows
            if item.get("round_delay_ms")
        ]
        row["round_chr"] = mean_round_series(round_chr_rows)
        row["round_delay_ms"] = mean_round_series(round_delay_rows)
        aggregate[str(vehicle_count)] = row
    return aggregate


def write_run_csv(
    output_path: Path,
    summaries: Dict[str, Dict[str, dict]],
    vehicle_counts: List[int],
    seeds: List[int],
) -> None:
    fieldnames = [
        "seed",
        "vehicle_num",
        "user_num",
        "method",
        "method_label",
        "rsu_cache_capacity",
        "achr",
        "average_delay_ms",
        "average_service_distance_m",
        "mrsu_average_distance_m",
        "frsu_average_distance_m",
        "mbs_average_distance_m",
        "mrsu_hit_count",
        "frsu_hit_count",
        "mbs_miss_count",
        "not_covered_count",
        "not_cached_count",
        "cache_update_count",
        "decision_rounds",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed in seeds:
            for vehicle_count in vehicle_counts:
                row = summaries.get(str(seed), {}).get(str(vehicle_count))
                if not row:
                    continue
                writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_round_csv(
    output_path: Path,
    results: Dict[str, Dict[str, dict]],
    vehicle_counts: List[int],
    seeds: List[int],
) -> None:
    fieldnames = [
        "seed",
        "vehicle_num",
        "user_num",
        "rsu_cache_capacity",
        "round",
        "physical_tick_start",
        "physical_tick_end",
        "decision_interval_ticks",
        "chr",
        "round_delay_ms",
        "average_service_distance_m",
        "mrsu_average_distance_m",
        "frsu_average_distance_m",
        "mbs_average_distance_m",
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
        "cache_update_used",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed in seeds:
            for vehicle_count in vehicle_counts:
                result = results.get(str(seed), {}).get(str(vehicle_count))
                if not result:
                    continue
                summary = result.get("summary", {})
                for round_index, log in enumerate(result.get("round_logs", [])):
                    metrics = log.get("metrics") or {}
                    latency = log.get("latency") or {}
                    writer.writerow(
                        {
                            "seed": int(seed),
                            "vehicle_num": int(summary.get("vehicle_num", vehicle_count)),
                            "user_num": int(summary.get("user_num", vehicle_count)),
                            "rsu_cache_capacity": int(summary.get("rsu_cache_capacity", DEFAULT_CACHE_CAPACITY)),
                            "round": int(log.get("round", round_index)),
                            "physical_tick_start": log.get("physical_tick_start", ""),
                            "physical_tick_end": log.get("physical_tick_end", ""),
                            "decision_interval_ticks": log.get("decision_interval_ticks", ""),
                            "chr": float(log.get("chr", metrics.get("chr", 0.0))),
                            "round_delay_ms": float(log.get("round_delay_ms", 0.0)),
                            "average_service_distance_m": latency.get("average_service_distance_m", ""),
                            "mrsu_average_distance_m": latency.get("mrsu_average_distance_m", ""),
                            "frsu_average_distance_m": latency.get("frsu_average_distance_m", ""),
                            "mbs_average_distance_m": latency.get("mbs_average_distance_m", ""),
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
                            "cache_update_used": bool(log.get("cache_update_used", False)),
                        }
                    )


def write_summary_csv(output_path: Path, aggregate: Dict[str, dict], vehicle_counts: List[int]) -> None:
    fieldnames = [
        "vehicle_num",
        "user_num",
        "seed_count",
        "seeds",
        "rsu_cache_capacity",
        "achr_mean",
        "achr_std",
        "average_delay_ms_mean",
        "average_delay_ms_std",
        "average_service_distance_m_mean",
        "mrsu_average_distance_m_mean",
        "frsu_average_distance_m_mean",
        "mbs_average_distance_m_mean",
        "mrsu_hit_count_mean",
        "frsu_hit_count_mean",
        "mbs_miss_count_mean",
        "not_covered_count_mean",
        "not_cached_count_mean",
        "cache_update_count_mean",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for vehicle_count in vehicle_counts:
            row = aggregate.get(str(vehicle_count))
            if not row:
                continue
            writer.writerow(
                {
                    "vehicle_num": int(row.get("vehicle_num", vehicle_count)),
                    "user_num": int(row.get("user_num", vehicle_count)),
                    "seed_count": int(row.get("seed_count", 0)),
                    "seeds": ",".join(str(seed) for seed in row.get("seeds", [])),
                    "rsu_cache_capacity": int(row.get("rsu_cache_capacity", DEFAULT_CACHE_CAPACITY)),
                    "achr_mean": row.get("achr_mean", ""),
                    "achr_std": row.get("achr_std", ""),
                    "average_delay_ms_mean": row.get("average_delay_ms_mean", ""),
                    "average_delay_ms_std": row.get("average_delay_ms_std", ""),
                    "average_service_distance_m_mean": row.get("average_service_distance_m_mean", ""),
                    "mrsu_average_distance_m_mean": row.get("mrsu_average_distance_m_mean", ""),
                    "frsu_average_distance_m_mean": row.get("frsu_average_distance_m_mean", ""),
                    "mbs_average_distance_m_mean": row.get("mbs_average_distance_m_mean", ""),
                    "mrsu_hit_count_mean": row.get("mrsu_hit_count_mean", ""),
                    "frsu_hit_count_mean": row.get("frsu_hit_count_mean", ""),
                    "mbs_miss_count_mean": row.get("mbs_miss_count_mean", ""),
                    "not_covered_count_mean": row.get("not_covered_count_mean", ""),
                    "not_cached_count_mean": row.get("not_cached_count_mean", ""),
                    "cache_update_count_mean": row.get("cache_update_count_mean", ""),
                }
            )


def write_plot_csv(output_path: Path, aggregate: Dict[str, dict], vehicle_counts: List[int]) -> None:
    fieldnames = ["vehicle_num", "achr", "achr_percent", "average_delay_ms"]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for vehicle_count in vehicle_counts:
            row = aggregate.get(str(vehicle_count), {})
            achr = float(row.get("achr_mean", 0.0))
            writer.writerow(
                {
                    "vehicle_num": int(vehicle_count),
                    "achr": achr,
                    "achr_percent": achr * 100.0,
                    "average_delay_ms": float(row.get("average_delay_ms_mean", 0.0)),
                }
            )


def write_dual_axis_bar_chart(
    vehicle_counts: List[int],
    chr_values: List[float],
    delay_values: List[float],
    output_path: Path,
    title: str,
) -> None:
    width, height = 660, 560
    left, right, top, bottom = 90, 100, 56, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    chr_percent = [float(value) * 100.0 for value in chr_values]
    delay = [float(value) for value in delay_values]
    chr_min, chr_max = padded_range(chr_percent, floor_zero=True)
    delay_min, delay_max = padded_range(delay, floor_zero=False)
    group_count = max(len(vehicle_counts), 1)
    group_w = plot_w / group_count
    bar_w = min(28.0, group_w * 0.30)
    bar_gap = 0.0
    blue = "#4169e1"
    pink = "#f35db2"
    grid_color = "#c4c4c4"

    def x_center(idx: int) -> float:
        return left + group_w * idx + group_w / 2.0

    def y_left(value: float) -> float:
        return top + (chr_max - value) * plot_h / max(chr_max - chr_min, 1e-9)

    def y_right(value: float) -> float:
        return top + (delay_max - value) * plot_h / max(delay_max - delay_min, 1e-9)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" '
            f'font-family="Times New Roman, Arial" font-size="21" fill="#111">{escape_xml(title)}</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111" stroke-width="1.2"/>',
        f'<line x1="{left + plot_w}" y1="{top}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left + plot_w}" y2="{top}" stroke="#111" stroke-width="1.2"/>',
    ]

    for tick in range(6):
        left_value = chr_min + (chr_max - chr_min) * tick / 5.0
        y = y_left(left_value)
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            f'stroke="{grid_color}" stroke-dasharray="3 2"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Times New Roman, Arial" font-size="13" fill="{blue}">{left_value:.1f}</text>'
        )
        right_value = delay_min + (delay_max - delay_min) * tick / 5.0
        lines.append(
            f'<text x="{left + plot_w + 10}" y="{y + 4:.2f}" text-anchor="start" '
            f'font-family="Times New Roman, Arial" font-size="13" fill="{pink}">{right_value:.1f}</text>'
        )

    for idx in range(group_count):
        x = x_center(idx)
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            f'stroke="{grid_color}" stroke-dasharray="3 2"/>'
        )

    for idx, vehicle_count in enumerate(vehicle_counts):
        center = x_center(idx)
        chr_top = y_left(chr_percent[idx])
        delay_top = y_right(delay[idx])
        baseline = top + plot_h
        cluster_x = center - (2.0 * bar_w + bar_gap) / 2.0
        lines.append(
            f'<rect x="{cluster_x:.2f}" y="{chr_top:.2f}" width="{bar_w:.2f}" '
            f'height="{max(0.0, baseline - chr_top):.2f}" fill="{blue}"/>'
        )
        lines.append(
            f'<rect x="{cluster_x + bar_w + bar_gap:.2f}" y="{delay_top:.2f}" width="{bar_w:.2f}" '
            f'height="{max(0.0, baseline - delay_top):.2f}" fill="{pink}"/>'
        )
        lines.append(
            f'<text x="{center:.2f}" y="{baseline + 24}" text-anchor="middle" '
            f'font-family="Times New Roman, Arial" font-size="15">{vehicle_count}</text>'
        )

    lines.extend(
        [
            (
                f'<text x="{left + plot_w / 2:.1f}" y="{height - 25}" text-anchor="middle" '
                f'font-family="Times New Roman, Arial" font-size="22" fill="#111">Number of vehicles</text>'
            ),
            (
                f'<text x="28" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
                f'transform="rotate(-90 28 {top + plot_h / 2:.1f})" '
                f'font-family="Times New Roman, Arial" font-size="21" fill="{blue}">Average Cache Hit Ratio (%)</text>'
            ),
            (
                f'<text x="{width - 28}" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
                f'transform="rotate(90 {width - 28} {top + plot_h / 2:.1f})" '
                f'font-family="Times New Roman, Arial" font-size="21" fill="{pink}">Average delay (ms)</text>'
            ),
            f'<rect x="{left + plot_w - 180}" y="{top + 12}" width="14" height="14" fill="{blue}"/>',
            f'<text x="{left + plot_w - 160}" y="{top + 24}" font-family="Times New Roman, Arial" font-size="14">ACHR</text>',
            f'<rect x="{left + plot_w - 105}" y="{top + 12}" width="14" height="14" fill="{pink}"/>',
            f'<text x="{left + plot_w - 85}" y="{top + 24}" font-family="Times New Roman, Arial" font-size="14">Delay</text>',
            "</svg>",
        ]
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def padded_range(values: List[float], floor_zero: bool) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-9:
        pad = max(abs(high) * 0.08, 1.0)
    else:
        pad = (high - low) * 0.16
    y_min = low - pad
    y_max = high + pad
    if floor_zero:
        y_min = max(0.0, y_min)
    if y_max <= y_min:
        y_max = y_min + 1.0
    return y_min, y_max


def parse_int_list(text: str, name: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    result: List[int] = []
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} must contain positive integers.")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError(f"{name} must contain at least one value.")
    return result


def parse_seeds(text: str, single_seed: int) -> List[int]:
    raw = [int(item.strip()) for item in text.split(",") if item.strip()] if text.strip() else [int(single_seed)]
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
    path = Path(base_dir) / f"vehicle_count_scan_{timestamp}"
    counter = 1
    while path.exists():
        path = Path(base_dir) / f"vehicle_count_scan_{timestamp}_{counter:02d}"
        counter += 1
    path.mkdir(parents=True, exist_ok=False)
    return str(path)


def save_json(data: dict, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def sample_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((float(value) - avg) ** 2 for value in values) / float(len(values) - 1)
    return float(variance ** 0.5)


def mean_round_series(round_rows: List[List[float]]) -> List[float]:
    if not round_rows:
        return []
    max_len = max(len(row) for row in round_rows)
    result = []
    for idx in range(max_len):
        values = [float(row[idx]) for row in round_rows if idx < len(row)]
        result.append(mean(values))
    return result


def escape_xml(text) -> str:
    return html.escape(str(text), quote=False)


if __name__ == "__main__":
    main()
