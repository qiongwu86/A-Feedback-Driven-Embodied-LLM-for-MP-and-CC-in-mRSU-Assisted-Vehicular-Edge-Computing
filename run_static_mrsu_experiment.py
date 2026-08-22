from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from run_ablation_experiment import (
    METHOD_LABELS,
    METHOD_NOTES,
    METHOD_STATIC_LLM,
    aggregate_summaries,
    build_config,
    build_latency_model,
    build_tool_agent,
    describe_tool_agent,
    normalize_summary,
    parse_capacities,
    plot_capacity_curve,
    plot_capacity_delay_curve,
    plot_round_curve,
    plot_round_delay_curve,
    run_ablation_method,
    save_json,
    write_aggregate_csv,
    write_delay_plot_data_csv,
    write_method_csv,
    write_plot_data_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Static mRSU ablation as an independent multi-seed experiment."
    )
    parser.add_argument("--rounds", type=int, default=100, help="Total physical simulation ticks.")
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=10,
        help="Physical ticks per request/cache decision window.",
    )
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

    parser.add_argument("--agent", choices=["auto", "mock", "llm"], default="auto")
    parser.add_argument("--base-url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", type=str, default="")
    parser.add_argument("--model-name", type=str, default="qwen3.6-flash")
    parser.add_argument("--max-context-chars", type=int, default=24000)

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
    methods = [METHOD_STATIC_LLM]
    seeds = parse_seed_values(args.seeds or args.seed)
    capacities = parse_capacities(args.capacities, args.rsu_cache)
    round_plot_capacity = int(args.plot_capacity) if int(args.plot_capacity) in capacities else int(capacities[0])
    output_dir = create_static_output_dir(args.output_dir)
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=False)
    tool_agent = build_tool_agent(args)
    latency_model = build_latency_model(args)

    print("Static mRSU experiment config:")
    print(
        json.dumps(
            {
                "method": METHOD_STATIC_LLM,
                "method_label": METHOD_LABELS[METHOD_STATIC_LLM],
                "seeds": seeds,
                "capacities": capacities,
                "physical_rounds": args.rounds,
                "decision_interval": args.decision_interval,
                "decision_rounds": (args.rounds + args.decision_interval - 1)
                // max(args.decision_interval, 1),
                "plot_capacity": round_plot_capacity,
                "road_topology": "circular_one_way",
                "latency_model": latency_model.config.to_dict(),
                "output_dir": output_dir,
                "data_dir": str(data_dir),
                "tool_llm_agent": describe_tool_agent(tool_agent),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    results: Dict[str, Dict[str, Dict[str, dict]]] = {METHOD_STATIC_LLM: {}}
    summaries: Dict[str, Dict[str, Dict[str, dict]]] = {METHOD_STATIC_LLM: {}}
    configs: Dict[str, Dict[str, dict]] = {}

    for seed in seeds:
        configs[str(seed)] = {}
        print(f"\n=== seed={seed} ===")
        for capacity in capacities:
            config = build_config(args, seed=seed, capacity=capacity, output_dir=output_dir)
            configs[str(seed)][str(capacity)] = asdict(config)
            print(f"\nRunning Static mRSU seed={seed} C={capacity}...")
            result = run_ablation_method(
                config=config,
                method=METHOD_STATIC_LLM,
                direct_agent=None,
                tool_agent=tool_agent,
                latency_model=latency_model,
                verbose=not args.quiet,
            )
            summary = normalize_summary(result, METHOD_STATIC_LLM, config)
            result["summary"] = summary
            results[METHOD_STATIC_LLM].setdefault(str(seed), {})[str(capacity)] = result
            summaries[METHOD_STATIC_LLM].setdefault(str(seed), {})[str(capacity)] = summary
            write_method_csv(data_dir / "static_mrsu_llm.csv", METHOD_STATIC_LLM, results[METHOD_STATIC_LLM])
            save_partial_json(output_dir, seeds, capacities, configs, summaries, results, latency_model)
            print(
                f"Finished Static mRSU seed={seed} C={capacity}: "
                f"LocalACHR={float(summary.get('local_rsu_achr', summary['achr'])):.4f} "
                f"ACHR={float(summary.get('achr', 0.0)):.4f} "
                f"Delay={float(summary.get('average_delay_ms', 0.0)):.2f}ms "
                f"cache_updates={int(summary.get('cache_update_count', 0))}"
            )

    aggregate = aggregate_summaries(summaries, methods, capacities, seeds)
    write_aggregate_csv(data_dir / "aggregate_summary.csv", aggregate, methods, capacities)
    write_plot_data_csv(data_dir / "capacity_achr_mean.csv", aggregate, methods, capacities)
    write_delay_plot_data_csv(data_dir / "capacity_delay_mean.csv", aggregate, methods, capacities)
    plot_capacity_curve(
        aggregate,
        methods,
        capacities,
        Path(output_dir) / "static_mrsu_achr_vs_capacity.svg",
    )
    plot_capacity_delay_curve(
        aggregate,
        methods,
        capacities,
        Path(output_dir) / "static_mrsu_delay_vs_capacity.svg",
    )
    plot_round_curve(
        results,
        methods,
        round_plot_capacity,
        Path(output_dir) / f"static_mrsu_round_chr_capacity_{round_plot_capacity}.svg",
    )
    plot_round_delay_curve(
        results,
        methods,
        round_plot_capacity,
        Path(output_dir) / f"static_mrsu_round_delay_capacity_{round_plot_capacity}.svg",
    )
    payload = build_result_payload(seeds, capacities, configs, aggregate, summaries, results, data_dir, latency_model)
    save_json(payload, str(Path(output_dir) / "static_mrsu_experiment_results.json"))
    print("\nStatic mRSU experiment finished.")
    print(f"Results saved to: {os.path.abspath(output_dir)}")


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
    save_json(payload, str(Path(output_dir) / "static_mrsu_experiment_partial.json"))


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
        "experiment": "static_mrsu_experiment",
        "methods": [METHOD_STATIC_LLM],
        "method_labels": {METHOD_STATIC_LLM: METHOD_LABELS[METHOD_STATIC_LLM]},
        "method_notes": {METHOD_STATIC_LLM: METHOD_NOTES[METHOD_STATIC_LLM]},
        "seeds": seeds,
        "capacities": capacities,
        "latency_model": latency_model.config.to_dict(),
        "config_by_seed_capacity": configs,
        "aggregate_summaries": aggregate,
        "summaries_by_seed_capacity": summaries,
        "results_by_seed_capacity": results,
        "data_dir": str(data_dir),
    }


def create_static_output_dir(base_dir: str) -> str:
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"static_mrsu_experiment_{timestamp}"
    counter = 1
    while path.exists():
        path = Path(base_dir) / f"static_mrsu_experiment_{timestamp}_{counter:02d}"
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


if __name__ == "__main__":
    main()
