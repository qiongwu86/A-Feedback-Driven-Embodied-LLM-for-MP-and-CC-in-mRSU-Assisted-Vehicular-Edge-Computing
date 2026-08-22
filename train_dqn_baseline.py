from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from baselines.dqn_baseline import METHOD_DQN, METHOD_LABEL, train_dqn
from run_traditional_baselines import _write_svg_line_chart, save_json
from simulation.config import MRSUSimulationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the conventional DQN baseline with mobility/history-only state features."
    )
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--rounds", type=int, default=100, help="Total physical simulation ticks per episode.")
    parser.add_argument("--decision-interval", type=int, default=10, help="Physical ticks per DQN action window.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rsu-cache", type=int, default=200)
    parser.add_argument(
        "--capacities",
        type=str,
        default="",
        help="Comma-separated synchronized mRSU/fRSU capacities. If omitted, uses --rsu-cache.",
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

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.99)
    parser.add_argument("--target-update-steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--reward-not-covered-penalty", type=float, default=0.0)
    parser.add_argument("--reward-not-cached-penalty", type=float, default=0.0)
    parser.add_argument(
        "--plot-window",
        type=int,
        default=50,
        help="Sliding window size for the DQN training reward/CHR curve.",
    )

    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capacities = parse_capacities(args.capacities, args.rsu_cache)
    output_dir = create_output_dir(args.output_dir)
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=False)

    print("DQN baseline training config:")
    print(
        json.dumps(
            {
                "method": METHOD_DQN,
                "method_label": METHOD_LABEL,
                "capacities": capacities,
                "episodes": int(args.episodes),
                "physical_rounds_per_episode": int(args.rounds),
                "decision_interval": int(args.decision_interval),
                "decision_steps_per_episode": (args.rounds + args.decision_interval - 1) // max(args.decision_interval, 1),
                "seed": int(args.seed),
                "plot_window": int(args.plot_window),
                "state_content_policy": (
                    "DQN state excludes movie IDs, predicted content demands, demand_summary, "
                    "dominant_contents, and vehicle-level miss-content feedback."
                ),
                "cache_content_source": "MovieLens training history and global training popularity only",
                "output_dir": output_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    model_infos: Dict[str, dict] = {}
    config_by_capacity: Dict[str, dict] = {}
    for capacity in capacities:
        config = build_config(args, capacity, output_dir)
        config_by_capacity[str(capacity)] = asdict(config)
        log_csv = data_dir / f"dqn_training_log_capacity_{int(capacity)}.csv"
        print(f"\n=== Train DQN capacity={capacity} ===")
        info = train_dqn(
            config=config,
            output_dir=output_dir,
            episodes=args.episodes,
            rounds=args.rounds,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            target_update_steps=args.target_update_steps,
            device=args.device,
            reward_not_covered_penalty=args.reward_not_covered_penalty,
            reward_not_cached_penalty=args.reward_not_cached_penalty,
            log_csv_path=log_csv,
            verbose=not args.quiet,
        )
        info["training_log_path"] = str(log_csv)
        model_infos[str(capacity)] = info
        plot_training_curve(
            read_training_log(log_csv),
            Path(output_dir) / f"dqn_training_curve_capacity_{int(capacity)}.svg",
            window=args.plot_window,
        )
        print(f"Saved DQN model for capacity={capacity}: {os.path.abspath(info['model_path'])}")

    save_json(
        {
            "experiment": "dqn_baseline_training",
            "method": METHOD_DQN,
            "method_label": METHOD_LABEL,
            "capacities": capacities,
            "episodes": int(args.episodes),
            "physical_rounds": int(args.rounds),
            "decision_interval": int(args.decision_interval),
            "decision_rounds": (args.rounds + args.decision_interval - 1) // max(args.decision_interval, 1),
            "seed": int(args.seed),
            "plot_window": int(args.plot_window),
            "config_by_capacity": config_by_capacity,
            "model_infos": model_infos,
            "data_dir": str(data_dir),
        },
        str(Path(output_dir) / "dqn_training_summary.json"),
    )
    print("\nDQN training finished.")
    print(f"Results saved to: {os.path.abspath(output_dir)}")


def build_config(args: argparse.Namespace, capacity: int, output_dir: str) -> MRSUSimulationConfig:
    return MRSUSimulationConfig(
        seed=int(args.seed),
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


def create_output_dir(base_dir: str) -> str:
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"dqn_training_{timestamp}"
    counter = 1
    while path.exists():
        path = Path(base_dir) / f"dqn_training_{timestamp}_{counter:02d}"
        counter += 1
    path.mkdir(parents=True, exist_ok=False)
    return str(path)


def read_training_log(path: Path) -> List[dict]:
    import csv

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def plot_training_curve(rows: List[dict], output_path: Path, window: int = 50) -> None:
    rewards, x_labels = windowed_curve_points(
        [float(row.get("episode_reward", 0.0)) for row in rows],
        window,
    )
    achr, _ = windowed_curve_points(
        [
            float(row.get("episode_average_local_rsu_chr") or row.get("episode_average_chr", 0.0))
            for row in rows
        ],
        window,
    )
    _write_svg_line_chart(
        {
            f"Reward MA-{max(int(window), 1)}": rewards,
            f"ACHR MA-{max(int(window), 1)}": achr,
        },
        str(output_path),
        f"DQN Training Curve with ACHR (Sliding Window={max(int(window), 1)})",
        "Episode",
        "Smoothed Reward / ACHR",
        x_labels=x_labels,
    )


def moving_average(values: List[float], window: int) -> List[float]:
    window = max(int(window), 1)
    result: List[float] = []
    rolling_sum = 0.0
    for idx, value in enumerate(values):
        rolling_sum += float(value)
        if idx >= window:
            rolling_sum -= float(values[idx - window])
            denominator = window
        else:
            denominator = idx + 1
        result.append(float(rolling_sum) / float(max(denominator, 1)))
    return result


def windowed_curve_points(values: List[float], window: int) -> tuple[List[float], List[str]]:
    if not values:
        return [], []
    window = max(int(window), 1)
    averaged = moving_average(values, window)
    indices = list(range(window - 1, len(values), window))
    if not indices or indices[-1] != len(values) - 1:
        indices.append(len(values) - 1)
    points = [float(averaged[idx]) for idx in indices]
    labels = [str(idx) for idx in indices]
    return points, labels


if __name__ == "__main__":
    main()
