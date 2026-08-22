from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from main_mrsu_tool_simulation import (
    TOOL_AGENT_METHOD,
    build_agent,
    build_config,
    build_latency_model,
    describe_agent,
    parse_capacities,
    run_method,
)
from simulation.metrics import local_rsu_chr_from_counts


METHOD_PREFIX = "da_elc_model"
DEFAULT_MODEL_LABEL_PREFIX = "FD-EMC w/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent FD-EMC model-comparison experiments."
    )
    parser.add_argument("--rounds", type=int, default=100, help="Total physical simulation ticks.")
    parser.add_argument("--decision-interval", type=int, default=10, help="Physical ticks per request/cache decision window.")
    parser.add_argument("--seed", type=str, default="42", help="Single seed. Comma-separated values are also accepted.")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds. Overrides --seed.")
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
        help="Comma-separated synchronized mRSU/fRSU cache capacities. Overrides --mrsu-cache and --frsu-cache.",
    )
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
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--agent", choices=["auto", "mock", "llm", "gemini", "gemini-rest"], default="auto")
    parser.add_argument("--base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", type=str, default="")
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen-flash",
        help="Single comparison model name. Ignored when --models is provided.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated comparison model names, e.g. qwen-flash,qwen3.6-flash.",
    )
    parser.add_argument(
        "--model-label-prefix",
        type=str,
        default=DEFAULT_MODEL_LABEL_PREFIX,
        help="Legend label prefix used in saved data.",
    )
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
    parser.add_argument("--quiet", action="store_true", help="Suppress per-round logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds or args.seed, name="seed")
    models = parse_str_list(args.models or args.model_name)
    if not seeds:
        raise ValueError("At least one seed is required.")
    if not models:
        raise ValueError("At least one model name is required.")

    capacities = parse_capacities(args)
    capacity_runs: List[Optional[int]] = capacities if capacities else [None]
    output_dir = create_output_dir(args.output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    latency_model = build_latency_model(args)

    all_results: Dict[str, Dict[str, Dict[str, dict]]] = {}
    configs: Dict[str, Dict[str, Dict[str, dict]]] = {}

    print("LLM model-comparison experiment config:")
    print(
        json.dumps(
            {
                "method": "FD-EMC with different LLM backbones",
                "models": models,
                "seeds": seeds,
                "capacities": capacities if capacities else [],
                "single_mrsu_cache": args.mrsu_cache,
                "single_frsu_cache": args.frsu_cache,
                "physical_rounds": args.rounds,
                "decision_interval": args.decision_interval,
                "latency_model": latency_model.config.to_dict(),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    for model_name in models:
        args.model_name = model_name
        agent = build_agent(args)
        model_key = safe_name(model_name)
        method_id = method_id_for_model(model_name)
        method_label = make_method_label(args.model_label_prefix, model_name)
        all_results.setdefault(model_key, {})
        configs.setdefault(model_key, {})
        print(f"\n=== model={model_name} label='{method_label}' agent={describe_agent(agent)} ===")

        for seed in seeds:
            args.seed = int(seed)
            seed_key = str(int(seed))
            all_results[model_key].setdefault(seed_key, {})
            configs[model_key].setdefault(seed_key, {})
            for cache_capacity in capacity_runs:
                config = build_config(args, cache_capacity)
                capacity_key = (
                    str(int(cache_capacity))
                    if cache_capacity is not None
                    else f"{config.mrsu_cache_capacity}_{config.frsu_cache_capacity}"
                )
                print(
                    f"\n--- running model={model_name}, seed={seed}, "
                    f"mRSU cache={config.mrsu_cache_capacity}, "
                    f"fRSU cache={config.frsu_cache_capacity}, "
                    f"decision_rounds={config.decision_rounds} ---"
                )
                result = run_method(
                    config=config,
                    method=TOOL_AGENT_METHOD,
                    agent=agent,
                    verbose=not args.quiet,
                    output_dir=str(output_dir),
                    cache_update_candidate_limit=args.cache_update_candidate_limit,
                    latency_model=latency_model,
                )
                result["summary"] = normalize_summary(
                    result=result,
                    method_id=method_id,
                    method_label=method_label,
                    model_name=model_name,
                    agent_description=describe_agent(agent),
                    config=config,
                )
                all_results[model_key][seed_key][capacity_key] = result
                configs[model_key][seed_key][capacity_key] = asdict(config)
                write_outputs(
                    output_dir=output_dir,
                    data_dir=data_dir,
                    args=args,
                    models=models,
                    seeds=seeds,
                    all_results=all_results,
                    configs=configs,
                )
                print(summarize_for_print(result["summary"]))

    print(f"\nResults saved under: {output_dir.resolve()}")


def normalize_summary(
    result: dict,
    method_id: str,
    method_label: str,
    model_name: str,
    agent_description: str,
    config,
) -> dict:
    summary = dict(result.get("summary") or {})
    summary.update(
        {
            "method": method_id,
            "method_label": method_label,
            "method_note": (
                "FD-EMC framework with the LLM backbone replaced by the specified comparison model."
            ),
            "model_name": model_name,
            "agent_description": agent_description,
            "seed": int(config.seed),
            "rounds": int(config.rounds),
            "physical_rounds": int(config.rounds),
            "decision_interval_ticks": int(config.decision_interval),
            "decision_rounds": int(config.decision_rounds),
            "rsu_cache_capacity": (
                int(config.mrsu_cache_capacity)
                if int(config.mrsu_cache_capacity) == int(config.frsu_cache_capacity)
                else ""
            ),
            "mrsu_cache_capacity": int(config.mrsu_cache_capacity),
            "frsu_cache_capacity": int(config.frsu_cache_capacity),
        }
    )
    return summary


def write_outputs(
    output_dir: Path,
    data_dir: Path,
    args: argparse.Namespace,
    models: List[str],
    seeds: List[int],
    all_results: Dict[str, Dict[str, Dict[str, dict]]],
    configs: Dict[str, Dict[str, Dict[str, dict]]],
) -> None:
    write_json(
        {
            "experiment": "llm_model_comparison",
            "base_method": "FD-EMC",
            "args": vars(args),
            "models": models,
            "seeds": seeds,
            "config_by_model_seed_capacity": configs,
            "results_by_model_seed_capacity": all_results,
        },
        output_dir / "llm_model_comparison_results.json",
    )

    aggregate_rows = aggregate_rows_by_model_capacity(all_results)
    write_aggregate_csv(data_dir / "aggregate_summary.csv", aggregate_rows)
    write_summary_csv(output_dir / "llm_model_comparison_summary.csv", aggregate_rows)

    for model_key, seed_results in sorted(all_results.items()):
        rows = []
        for seed_key, capacity_results in sorted(seed_results.items(), key=lambda item: int(item[0])):
            for capacity_key, result in sorted(
                capacity_results.items(),
                key=lambda item: parse_capacity_sort_key(item[0]),
            ):
                rows.extend(round_rows_for_result(seed_key, capacity_key, result))
        if rows:
            write_round_csv(data_dir / f"{model_key}.csv", rows)


def round_rows_for_result(seed_key: str, capacity_key: str, result: dict) -> List[dict]:
    summary = result.get("summary") or {}
    average_chr = float(summary.get("achr", 0.0))
    average_local_rsu_chr = float(summary.get("local_rsu_achr", average_chr))
    capacity = summary.get("rsu_cache_capacity") or summary.get("mrsu_cache_capacity") or capacity_key
    rows: List[dict] = []
    for round_index, log in enumerate(result.get("round_logs") or []):
        metrics = log.get("metrics") or {}
        latency = log.get("latency") or {}
        selected_hotspot = log.get("selected_hotspot") or {}
        local_chr = metrics.get("local_rsu_chr")
        if local_chr in ("", None):
            hit = float(metrics.get("hit_count", 0.0))
            if not hit:
                hit = float(metrics.get("mrsu_hit_count", 0.0)) + float(metrics.get("frsu_hit_count", 0.0))
            local_chr = local_rsu_chr_from_counts(hit, float(metrics.get("not_cached_count", 0.0)))
        rows.append(
            {
                "seed": int(summary.get("seed", seed_key)),
                "average_chr": average_chr,
                "average_local_rsu_chr": average_local_rsu_chr,
                "method": summary.get("method", ""),
                "method_label": summary.get("method_label", ""),
                "model_name": summary.get("model_name", ""),
                "agent_description": summary.get("agent_description", ""),
                "rsu_cache_capacity": int(float(capacity)) if str(capacity).strip() else "",
                "mrsu_cache_capacity": int(summary.get("mrsu_cache_capacity", capacity)),
                "frsu_cache_capacity": int(summary.get("frsu_cache_capacity", capacity)),
                "round": int(log.get("round", round_index)),
                "physical_tick_start": log.get("physical_tick_start", ""),
                "physical_tick_end": log.get("physical_tick_end", ""),
                "decision_interval_ticks": log.get("decision_interval_ticks", ""),
                "chr": float(metrics.get("chr", log.get("chr", 0.0))),
                "local_rsu_chr": float(local_chr),
                "round_delay_ms": float(latency.get("average_delay_ms", log.get("round_delay_ms", 0.0))),
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
                "selected_hotspot_position": selected_hotspot.get("position", ""),
                "path_plan_status": log.get("path_plan_status", ""),
                "path_plan_solver": log.get("path_plan_solver", ""),
                "decision_request_count": log.get("decision_request_count", ""),
                "decision_request_source": log.get("decision_request_source", ""),
                "evaluation_request_count": log.get("evaluation_request_count", ""),
                "evaluation_request_source": log.get("evaluation_request_source", ""),
                "cache_update_count": int(summary.get("cache_update_count", 0)),
            }
        )
    return rows


def aggregate_rows_by_model_capacity(
    all_results: Dict[str, Dict[str, Dict[str, dict]]],
) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for _model_key, seed_results in all_results.items():
        for _seed_key, capacity_results in seed_results.items():
            for capacity_key, result in capacity_results.items():
                summary = result.get("summary") or {}
                method = str(summary.get("method", ""))
                capacity = str(summary.get("rsu_cache_capacity") or summary.get("mrsu_cache_capacity") or capacity_key)
                grouped.setdefault((method, capacity), []).append(summary)

    rows: List[dict] = []
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
    ]
    for (_method, _capacity), summaries in sorted(grouped.items(), key=lambda item: (item[0][0], parse_capacity_sort_key(item[0][1]))):
        first = summaries[0]
        row = {
            "method": first.get("method", ""),
            "method_label": first.get("method_label", ""),
            "model_name": first.get("model_name", ""),
            "agent_description": first.get("agent_description", ""),
            "rsu_cache_capacity": int(float(_capacity)) if str(_capacity).strip() else "",
            "mrsu_cache_capacity": int(first.get("mrsu_cache_capacity", _capacity)),
            "frsu_cache_capacity": int(first.get("frsu_cache_capacity", _capacity)),
            "seed_count": len(summaries),
            "seeds": ",".join(str(int(item.get("seed", 0))) for item in summaries),
        }
        for field in metric_fields:
            values = [float(item[field]) for item in summaries if item.get(field) not in ("", None)]
            if not values:
                continue
            row[f"{field}_mean"] = mean(values)
            row[f"{field}_std"] = sample_std(values)
            row[f"{field}_min"] = min(values)
            row[f"{field}_max"] = max(values)
        if "local_rsu_achr_mean" not in row:
            row["local_rsu_achr_mean"] = local_rsu_chr_from_counts(
                float(row.get("hit_count_mean", 0.0)),
                float(row.get("not_cached_count_mean", 0.0)),
            )
        rows.append(row)
    return rows


def write_round_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "seed",
        "average_chr",
        "average_local_rsu_chr",
        "method",
        "method_label",
        "model_name",
        "agent_description",
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
        "selected_hotspot_position",
        "path_plan_status",
        "path_plan_solver",
        "decision_request_count",
        "decision_request_source",
        "evaluation_request_count",
        "evaluation_request_source",
        "cache_update_count",
    ]
    write_csv(path, fieldnames, rows)


def write_aggregate_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "method",
        "method_label",
        "model_name",
        "agent_description",
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
        "cache_update_count_mean",
    ]
    write_csv(path, fieldnames, rows)


def write_summary_csv(path: Path, rows: List[dict]) -> None:
    write_aggregate_csv(path, rows)


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_output_dir(base_dir: str) -> Path:
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"llm_model_comparison_{timestamp}"
    suffix = 1
    while path.exists():
        path = root / f"llm_model_comparison_{timestamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def parse_int_list(text: str, name: str) -> List[int]:
    values: List[int] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid {name} value: {item}") from exc
        if value not in values:
            values.append(value)
    return values


def parse_str_list(text: str) -> List[str]:
    values: List[str] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if item and item not in values:
            values.append(item)
    return values


def make_method_label(prefix: str, model_name: str) -> str:
    prefix = str(prefix or DEFAULT_MODEL_LABEL_PREFIX).strip()
    if not prefix:
        return str(model_name)
    return f"{prefix} {model_name}"


def method_id_for_model(model_name: str) -> str:
    return f"{METHOD_PREFIX}_{safe_name(model_name)}"


def safe_name(text: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", str(text).strip()).strip("_").lower()
    return value or "model"


def parse_capacity_sort_key(value) -> int:
    try:
        return int(float(str(value).split("_")[0]))
    except (TypeError, ValueError):
        return 0


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / len(values)) if values else 0.0


def sample_std(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return float(math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1)))


def summarize_for_print(summary: dict) -> str:
    return (
        f"{summary.get('method_label', ''):28s} "
        f"seed={summary.get('seed')} "
        f"capacity={summary.get('rsu_cache_capacity') or summary.get('mrsu_cache_capacity')} "
        f"LocalACHR={float(summary.get('local_rsu_achr', summary.get('achr', 0.0))):.4f} "
        f"SystemACHR={float(summary.get('achr', 0.0)):.4f} "
        f"Delay={float(summary.get('average_delay_ms', 0.0)):.2f}ms"
    )


if __name__ == "__main__":
    main()
