from __future__ import annotations

import argparse
import csv
import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


OURS_BASE_MODEL = "qwen3.7-flash"
OURS_LABEL = f"FD-EMC({OURS_BASE_MODEL})"
DEFAULT_LLM_COMPARISON_LABELS = ["FD-EMC(deepseek-v4-flash)"]
METHOD_ORDER = [OURS_LABEL] + DEFAULT_LLM_COMPARISON_LABELS + ["Thompson Sampling", "CP-SAT", "DQN"]
BASELINE_METHOD_LABELS = {"Thompson Sampling", "CP-SAT", "DQN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot saved FD-EMC and baseline results without rerunning simulations."
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="",
        help="Directory from run_traditional_baselines.py. Auto-detects the latest one if omitted.",
    )
    parser.add_argument(
        "--ours-dir",
        type=str,
        default="",
        help="Directory from main_mrsu_tool_simulation.py, named like 具身智能YYYYMMDD_HHMMSS. Auto-detects latest if omitted.",
    )
    parser.add_argument(
        "--dqn-dir",
        type=str,
        default="",
        help="Directory from test_dqn_baseline.py. Auto-detects latest dqn_baseline_* if omitted.",
    )
    parser.add_argument(
        "--llm-comparison-dir",
        type=str,
        default="",
        help="Directory from run_llm_model_comparison_experiment.py. Auto-detects latest llm_model_comparison_* if omitted; skipped when absent.",
    )
    parser.add_argument("--capacity", type=int, default=200, help="Capacity used for per-round CHR plot.")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else find_latest_baseline_dir(args.results_root)
    ours_dir = Path(args.ours_dir) if args.ours_dir else find_latest_ours_dir(args.results_root)
    dqn_dir = Path(args.dqn_dir) if args.dqn_dir else find_latest_dqn_dir(args.results_root)
    llm_comparison_dir = (
        Path(args.llm_comparison_dir)
        if args.llm_comparison_dir
        else find_latest_llm_comparison_dir(args.results_root)
    )
    has_baseline = _valid_result_dir(baseline_dir)
    has_ours = _valid_result_dir(ours_dir)
    has_dqn = _valid_result_dir(dqn_dir)
    has_llm_comparison = is_usable_llm_comparison_dir(llm_comparison_dir)
    output_dir = create_output_dir(args.output_dir)

    if not has_baseline and not has_dqn:
        raise FileNotFoundError("No traditional or DQN baseline result directory was found.")
    if not has_ours:
        raise FileNotFoundError("No FD-EMC result directory was found.")

    round_series = {}
    round_series.update(read_ours_round_series(ours_dir, args.capacity))
    if has_llm_comparison:
        round_series.update(read_llm_comparison_round_series(llm_comparison_dir, args.capacity))
    if has_baseline:
        round_series.update(read_baseline_round_series(baseline_dir, args.capacity))
    if has_dqn:
        round_series.update(read_baseline_round_series(dqn_dir, args.capacity))
    round_series = order_series(round_series)
    if not round_series:
        raise RuntimeError(f"No per-round ACHR data found for capacity {args.capacity}.")

    capacity_series = {}
    capacity_series.update(read_ours_capacity_series(ours_dir))
    if has_llm_comparison:
        capacity_series.update(read_llm_comparison_capacity_series(llm_comparison_dir))
    if has_baseline:
        capacity_series.update(read_baseline_capacity_series(baseline_dir))
    if has_dqn:
        capacity_series.update(read_baseline_capacity_series(dqn_dir))
    capacity_series = order_series(capacity_series)
    if not capacity_series:
        raise RuntimeError("No capacity-scan ACHR data was found.")

    capacity_axis = sorted(
        {
            int(capacity)
            for series in capacity_series.values()
            for capacity in series.keys()
        }
    )

    round_svg = output_dir / f"{args.capacity}容量每轮CHR对比.svg"
    capacity_svg = output_dir / "容量扫描ACHR对比.svg"
    write_svg_line_chart(
        round_series,
        round_svg,
        f"Per-round Average Cache Hit Ratio at RSU Cache Capacity {args.capacity}",
        "Round",
        "Average Cache Hit Ratio (%)",
        x_labels=[str(idx) for idx in range(max(len(values) for values in round_series.values()))],
        legend_position="bottom_right",
        y_floor_zero=False,
        y_scale=100.0,
        y_tick_decimals=0,
    )
    write_svg_line_chart(
        series_dicts_to_lists(capacity_series, capacity_axis),
        capacity_svg,
        "ACHR vs Synchronized RSU Cache Capacity",
        "mRSU/sRSU Cache Capacity",
        "Average Cache Hit Ratio (%)",
        x_labels=[str(capacity) for capacity in capacity_axis],
        legend_position="bottom_right",
        y_floor_zero=False,
        y_scale=100.0,
        y_tick_decimals=0,
    )

    save_round_csv(round_series, output_dir / f"{args.capacity}容量每轮CHR对比数据.csv")
    save_capacity_csv(capacity_series, capacity_axis, output_dir / "容量扫描ACHR对比数据.csv")
    output_files = {
        "round_chr_plot": str(round_svg),
        "capacity_scan_plot": str(capacity_svg),
    }
    round_delay_series = {}
    round_delay_series.update(read_ours_round_delay_series(ours_dir, args.capacity))
    if has_llm_comparison:
        round_delay_series.update(read_llm_comparison_round_delay_series(llm_comparison_dir, args.capacity))
    if has_baseline:
        round_delay_series.update(read_baseline_round_delay_series(baseline_dir, args.capacity))
    if has_dqn:
        round_delay_series.update(read_baseline_round_delay_series(dqn_dir, args.capacity))
    round_delay_series = order_series(round_delay_series)
    if round_delay_series:
        round_delay_svg = output_dir / f"{args.capacity}_capacity_round_delay_ms_compare.svg"
        write_svg_line_chart(
            round_delay_series,
            round_delay_svg,
            f"Per-round Average Delay at RSU Cache Capacity {args.capacity}",
            "Round",
            "Average Delay (ms)",
            x_labels=[str(idx) for idx in range(max(len(values) for values in round_delay_series.values()))],
            legend_position="top_right",
            y_floor_zero=False,
            y_tick_decimals=0,
        )
        save_round_csv(round_delay_series, output_dir / f"{args.capacity}_capacity_round_delay_ms_compare.csv")
        output_files["round_delay_plot"] = str(round_delay_svg)

    capacity_delay_series = {}
    capacity_delay_series.update(read_ours_capacity_delay_series(ours_dir))
    if has_llm_comparison:
        capacity_delay_series.update(read_llm_comparison_capacity_delay_series(llm_comparison_dir))
    if has_baseline:
        capacity_delay_series.update(read_baseline_capacity_delay_series(baseline_dir))
    if has_dqn:
        capacity_delay_series.update(read_baseline_capacity_delay_series(dqn_dir))
    capacity_delay_series = order_series(capacity_delay_series)
    if capacity_delay_series:
        delay_capacity_axis = sorted(
            {
                int(capacity)
                for series in capacity_delay_series.values()
                for capacity in series.keys()
            }
        )
        capacity_delay_svg = output_dir / "容量扫描delay对比.svg"
        write_svg_line_chart(
            series_dicts_to_lists(capacity_delay_series, delay_capacity_axis),
            capacity_delay_svg,
            "Average Delay vs Synchronized RSU Cache Capacity",
            "mRSU/sRSU Cache Capacity",
            "Average Delay (ms)",
            x_labels=[str(capacity) for capacity in delay_capacity_axis],
            legend_position="top_right",
            y_floor_zero=False,
            y_tick_decimals=0,
        )
        save_capacity_csv(
            capacity_delay_series,
            delay_capacity_axis,
            output_dir / "容量扫描delay对比数据.csv",
        )
        output_files["capacity_delay_plot"] = str(capacity_delay_svg)

    save_json(
        {
            "baseline_dir": str(baseline_dir) if has_baseline else "",
            "dqn_dir": str(dqn_dir) if has_dqn else "",
            "llm_comparison_dir": str(llm_comparison_dir) if has_llm_comparison else "",
            "ours_dir": str(ours_dir),
            "round_capacity": int(args.capacity),
            "round_series": round_series,
            "round_delay_series": round_delay_series,
            "capacity_axis": capacity_axis,
            "capacity_series": capacity_series,
            "capacity_delay_series": capacity_delay_series,
            "outputs": output_files,
        },
        output_dir / "基线绘图数据.json",
    )

    print("Plotting finished.")
    print(f"Baseline source: {baseline_dir if has_baseline else ''}")
    print(f"DQN source: {dqn_dir if has_dqn else ''}")
    print(f"LLM comparison source: {llm_comparison_dir if has_llm_comparison else ''}")
    print(f"{OURS_LABEL} source: {ours_dir}")
    print(f"Saved to: {output_dir.resolve()}")


def read_baseline_round_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    data_dir = result_dir / "data"
    series = {}
    if not data_dir.exists():
        return series
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not _is_baseline_method_csv(rows):
            continue
        selected = [
            row
            for row in rows
            if parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity")) == int(capacity)
        ]
        if not selected:
            continue
        label = selected[0].get("method_label") or csv_path.stem
        if label not in BASELINE_METHOD_LABELS:
            continue
        by_round: Dict[int, List[float]] = {}
        for row in selected:
            round_index = parse_int(row.get("round"), default=0)
            value = row_local_rsu_chr(row)
            if value is None:
                value = parse_float(row.get("chr"))
            by_round.setdefault(round_index, []).append(float(value))
        series[label] = [
            _mean(values)
            for _, values in sorted(by_round.items())
        ]
    return series


def read_baseline_round_delay_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    data_dir = result_dir / "data"
    series = {}
    if not data_dir.exists():
        return series
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not _is_baseline_method_csv(rows):
            continue
        selected = [
            row
            for row in rows
            if parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity")) == int(capacity)
        ]
        if not selected:
            continue
        label = selected[0].get("method_label") or csv_path.stem
        if label not in BASELINE_METHOD_LABELS:
            continue
        by_round: Dict[int, List[float]] = {}
        for row in selected:
            value = parse_float(row.get("round_delay_ms"), default=None)
            if value is None:
                continue
            round_index = parse_int(row.get("round"), default=0)
            by_round.setdefault(round_index, []).append(float(value))
        if by_round:
            series[label] = [
                _mean(values)
                for _, values in sorted(by_round.items())
            ]
    return series


def read_baseline_capacity_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    data_dir = result_dir / "data"
    series: Dict[str, Dict[int, float]] = {}
    if not data_dir.exists():
        return series
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not _is_baseline_method_csv(rows):
            continue
        average_by_seed_capacity: Dict[Tuple[int, int], float] = {}
        count_by_seed_capacity: Dict[Tuple[int, int], Tuple[float, float]] = {}
        round_values_by_seed_capacity: Dict[Tuple[int, int], List[float]] = {}
        label = ""
        for row in rows:
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            if capacity is None:
                continue
            seed = parse_int(row.get("seed"), default=0)
            label = row.get("method_label") or label or csv_path.stem
            key = (int(capacity), int(seed))
            count_value = row_local_rsu_counts(row)
            if count_value is not None:
                hit, not_cached = count_value
                old_hit, old_not_cached = count_by_seed_capacity.get(key, (0.0, 0.0))
                count_by_seed_capacity[key] = (old_hit + hit, old_not_cached + not_cached)
            elif row.get("average_local_rsu_chr") not in ("", None):
                average_by_seed_capacity[key] = parse_float(row.get("average_local_rsu_chr"))
            elif row.get("average_chr") not in ("", None):
                average_by_seed_capacity[key] = parse_float(row.get("average_chr"))
            else:
                value = row_local_rsu_chr(row)
                if value is None:
                    value = parse_float(row.get("chr"))
                round_values_by_seed_capacity.setdefault(key, []).append(float(value))
        by_capacity: Dict[int, List[float]] = {}
        for (capacity, seed), (hit, not_cached) in count_by_seed_capacity.items():
            by_capacity.setdefault(int(capacity), []).append(local_rsu_chr_from_counts(hit, not_cached))
        for (capacity, seed), value in average_by_seed_capacity.items():
            if (capacity, seed) in count_by_seed_capacity:
                continue
            by_capacity.setdefault(int(capacity), []).append(float(value))
        for (capacity, seed), values in round_values_by_seed_capacity.items():
            if (capacity, seed) in count_by_seed_capacity or (capacity, seed) in average_by_seed_capacity:
                continue
            by_capacity.setdefault(int(capacity), []).append(_mean(values))
        if not by_capacity:
            continue
        label = label or csv_path.stem
        if label not in BASELINE_METHOD_LABELS:
            continue
        series[label] = {
            int(capacity): _mean(values)
            for capacity, values in sorted(by_capacity.items())
        }
    return series


def read_baseline_capacity_delay_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    data_dir = result_dir / "data"
    series: Dict[str, Dict[int, float]] = {}
    if not data_dir.exists():
        return series
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not _is_baseline_method_csv(rows):
            continue
        average_by_seed_capacity: Dict[Tuple[int, int], float] = {}
        round_values_by_seed_capacity: Dict[Tuple[int, int], List[float]] = {}
        label = ""
        for row in rows:
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            if capacity is None:
                continue
            seed = parse_int(row.get("seed"), default=0)
            label = row.get("method_label") or label or csv_path.stem
            average_delay = parse_float(row.get("average_delay_ms"), default=None)
            round_delay = parse_float(row.get("round_delay_ms"), default=None)
            if average_delay is not None:
                average_by_seed_capacity[(int(capacity), int(seed))] = float(average_delay)
            elif round_delay is not None:
                round_values_by_seed_capacity.setdefault((int(capacity), int(seed)), []).append(float(round_delay))
        by_capacity: Dict[int, List[float]] = {}
        for (capacity, seed), value in average_by_seed_capacity.items():
            by_capacity.setdefault(int(capacity), []).append(float(value))
        for (capacity, seed), values in round_values_by_seed_capacity.items():
            if (capacity, seed) in average_by_seed_capacity:
                continue
            by_capacity.setdefault(int(capacity), []).append(_mean(values))
        if not by_capacity:
            continue
        label = label or csv_path.stem
        if label not in BASELINE_METHOD_LABELS:
            continue
        series[label] = {
            int(capacity): _mean(values)
            for capacity, values in sorted(by_capacity.items())
        }
    return series


def read_llm_comparison_round_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    data_dir = result_dir / "data"
    series = {}
    if not data_dir.exists():
        return series
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not _is_baseline_method_csv(rows):
            continue
        selected = [
            row
            for row in rows
            if parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity")) == int(capacity)
        ]
        if not selected:
            continue
        label = normalize_llm_comparison_label(selected[0])
        if should_skip_llm_comparison_label(label):
            continue
        by_round: Dict[int, List[float]] = {}
        for row in selected:
            value = row_local_rsu_chr(row)
            if value is None:
                value = parse_float(row.get("chr"))
            round_index = parse_int(row.get("round"), default=0)
            by_round.setdefault(round_index, []).append(float(value))
        if by_round:
            series[label] = [_mean(values) for _, values in sorted(by_round.items())]
    return series


def read_llm_comparison_round_delay_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    data_dir = result_dir / "data"
    series = {}
    if not data_dir.exists():
        return series
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not _is_baseline_method_csv(rows):
            continue
        selected = [
            row
            for row in rows
            if parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity")) == int(capacity)
        ]
        if not selected:
            continue
        label = normalize_llm_comparison_label(selected[0])
        if should_skip_llm_comparison_label(label):
            continue
        by_round: Dict[int, List[float]] = {}
        for row in selected:
            value = parse_float(row.get("round_delay_ms"), default=None)
            if value is None:
                continue
            round_index = parse_int(row.get("round"), default=0)
            by_round.setdefault(round_index, []).append(float(value))
        if by_round:
            series[label] = [_mean(values) for _, values in sorted(by_round.items())]
    return series


def read_llm_comparison_capacity_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    aggregate_csv = result_dir / "data" / "aggregate_summary.csv"
    if aggregate_csv.exists():
        series: Dict[str, Dict[int, float]] = {}
        for row in read_csv(aggregate_csv):
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            if capacity is None:
                continue
            label = normalize_llm_comparison_label(row)
            if should_skip_llm_comparison_label(label):
                continue
            value = row.get("local_rsu_achr_mean")
            if value in ("", None):
                count_value = row_local_rsu_counts(row)
                value = local_rsu_chr_from_counts(*count_value) if count_value is not None else row.get("achr_mean")
            if value not in ("", None):
                series.setdefault(label, {})[int(capacity)] = float(value)
        if series:
            return series
    return read_llm_comparison_capacity_series_from_method_csvs(result_dir)


def read_llm_comparison_capacity_series_from_method_csvs(result_dir: Path) -> Dict[str, Dict[int, float]]:
    data_dir = result_dir / "data"
    series: Dict[str, Dict[int, float]] = {}
    if not data_dir.exists():
        return series
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not _is_baseline_method_csv(rows):
            continue
        label = normalize_llm_comparison_label(rows[0])
        if should_skip_llm_comparison_label(label):
            continue
        values_by_seed_capacity: Dict[Tuple[int, int], List[float]] = {}
        average_by_seed_capacity: Dict[Tuple[int, int], float] = {}
        count_by_seed_capacity: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for row in rows:
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            if capacity is None:
                continue
            seed = parse_int(row.get("seed"), default=0)
            key = (int(capacity), int(seed))
            count_value = row_local_rsu_counts(row)
            if count_value is not None:
                hit, not_cached = count_value
                old_hit, old_not_cached = count_by_seed_capacity.get(key, (0.0, 0.0))
                count_by_seed_capacity[key] = (old_hit + hit, old_not_cached + not_cached)
            elif row.get("average_local_rsu_chr") not in ("", None):
                average_by_seed_capacity[key] = parse_float(row.get("average_local_rsu_chr"))
            elif row.get("average_chr") not in ("", None):
                average_by_seed_capacity[key] = parse_float(row.get("average_chr"))
            else:
                value = row_local_rsu_chr(row)
                if value is None:
                    value = parse_float(row.get("chr"))
                values_by_seed_capacity.setdefault(key, []).append(float(value))
        by_capacity: Dict[int, List[float]] = {}
        for (capacity, seed), (hit, not_cached) in count_by_seed_capacity.items():
            by_capacity.setdefault(int(capacity), []).append(local_rsu_chr_from_counts(hit, not_cached))
        for (capacity, seed), value in average_by_seed_capacity.items():
            if (capacity, seed) in count_by_seed_capacity:
                continue
            by_capacity.setdefault(int(capacity), []).append(float(value))
        for (capacity, seed), values in values_by_seed_capacity.items():
            if (capacity, seed) in count_by_seed_capacity or (capacity, seed) in average_by_seed_capacity:
                continue
            by_capacity.setdefault(int(capacity), []).append(_mean(values))
        if by_capacity:
            series[label] = {int(capacity): _mean(values) for capacity, values in sorted(by_capacity.items())}
    return series


def read_llm_comparison_capacity_delay_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    aggregate_csv = result_dir / "data" / "aggregate_summary.csv"
    if aggregate_csv.exists():
        series: Dict[str, Dict[int, float]] = {}
        for row in read_csv(aggregate_csv):
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            delay = parse_float(row.get("average_delay_ms_mean"), default=None)
            if capacity is None or delay is None:
                continue
            label = normalize_llm_comparison_label(row)
            if should_skip_llm_comparison_label(label):
                continue
            series.setdefault(label, {})[int(capacity)] = float(delay)
        if series:
            return series
    return read_llm_comparison_capacity_delay_series_from_method_csvs(result_dir)


def read_llm_comparison_capacity_delay_series_from_method_csvs(result_dir: Path) -> Dict[str, Dict[int, float]]:
    data_dir = result_dir / "data"
    series: Dict[str, Dict[int, float]] = {}
    if not data_dir.exists():
        return series
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not _is_baseline_method_csv(rows):
            continue
        label = normalize_llm_comparison_label(rows[0])
        if should_skip_llm_comparison_label(label):
            continue
        average_by_seed_capacity: Dict[Tuple[int, int], float] = {}
        round_values_by_seed_capacity: Dict[Tuple[int, int], List[float]] = {}
        for row in rows:
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            if capacity is None:
                continue
            seed = parse_int(row.get("seed"), default=0)
            average_delay = parse_float(row.get("average_delay_ms"), default=None)
            round_delay = parse_float(row.get("round_delay_ms"), default=None)
            key = (int(capacity), int(seed))
            if average_delay is not None:
                average_by_seed_capacity[key] = float(average_delay)
            elif round_delay is not None:
                round_values_by_seed_capacity.setdefault(key, []).append(float(round_delay))
        by_capacity: Dict[int, List[float]] = {}
        for (capacity, seed), value in average_by_seed_capacity.items():
            by_capacity.setdefault(int(capacity), []).append(float(value))
        for (capacity, seed), values in round_values_by_seed_capacity.items():
            if (capacity, seed) in average_by_seed_capacity:
                continue
            by_capacity.setdefault(int(capacity), []).append(_mean(values))
        if by_capacity:
            series[label] = {int(capacity): _mean(values) for capacity, values in sorted(by_capacity.items())}
    return series


def read_ours_round_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    embodied_json = result_dir / "具身智能完整运行结果.json"
    if embodied_json.exists():
        data = read_json(embodied_json)
        by_capacity = data.get("results_by_capacity") or {}
        result = find_capacity_result(by_capacity, capacity)
        if result:
            ours = result.get("tool_agent") or result.get("Ours")
            summary = ((ours or {}).get("summary") or {})
            round_chr = summary.get("round_local_rsu_chr", summary.get("round_chr", [])) or []
            if not round_chr:
                round_chr = round_local_rsu_chr_from_logs((ours or {}).get("round_logs") or [])
            if round_chr:
                return {OURS_LABEL: [float(value) for value in round_chr]}
    return {}


def read_ours_round_delay_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    embodied_json = result_dir / "具身智能完整运行结果.json"
    if embodied_json.exists():
        data = read_json(embodied_json)
        by_capacity = data.get("results_by_capacity") or {}
        result = find_capacity_result(by_capacity, capacity)
        if result:
            ours = result.get("tool_agent") or result.get("Ours")
            summary = ((ours or {}).get("summary") or {})
            round_delay = summary.get("round_delay_ms") or []
            if round_delay:
                return {OURS_LABEL: [float(value) for value in round_delay]}
            logs = (ours or {}).get("round_logs") or []
            values = []
            for log in logs:
                latency = log.get("latency") or {}
                value = latency.get("average_delay_ms", log.get("round_delay_ms"))
                if value not in ("", None):
                    values.append(float(value))
            if values:
                return {OURS_LABEL: values}
    return {}


def read_ours_capacity_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    embodied_csv = result_dir / "具身智能ACHR汇总表.csv"
    if embodied_csv.exists():
        rows = read_csv(embodied_csv)
        values = {}
        for row in rows:
            if row.get("method") != "tool_agent" and row.get("method_label") not in ("Ours", OURS_LABEL):
                continue
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            if capacity is None:
                continue
            value = row.get("local_rsu_achr")
            if value in ("", None):
                count_value = row_local_rsu_counts(row)
                value = local_rsu_chr_from_counts(*count_value) if count_value is not None else row.get("achr")
            values[int(capacity)] = float(value)
        if values:
            return {OURS_LABEL: values}

    embodied_json = result_dir / "具身智能完整运行结果.json"
    if embodied_json.exists():
        data = read_json(embodied_json)
        values = {}
        for capacity_key, result in (data.get("results_by_capacity") or {}).items():
            ours = (result or {}).get("tool_agent") or {}
            summary = ours.get("summary") or {}
            capacity = parse_int(summary.get("rsu_cache_capacity") or capacity_key, default=None)
            if capacity is not None:
                value = summary.get("local_rsu_achr")
                if value in ("", None):
                    value = summary.get("achr")
                if value not in ("", None):
                    values[int(capacity)] = float(value)
        if values:
            return {OURS_LABEL: values}
    return {}


def read_ours_capacity_delay_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    embodied_csv = result_dir / "具身智能ACHR汇总表.csv"
    if embodied_csv.exists():
        rows = read_csv(embodied_csv)
        values = {}
        for row in rows:
            if row.get("method") != "tool_agent" and row.get("method_label") not in ("Ours", OURS_LABEL):
                continue
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            delay = parse_float(row.get("average_delay_ms"), default=None)
            if capacity is not None and delay is not None:
                values[int(capacity)] = float(delay)
        if values:
            return {OURS_LABEL: values}

    embodied_json = result_dir / "具身智能完整运行结果.json"
    if embodied_json.exists():
        data = read_json(embodied_json)
        values = {}
        for capacity_key, result in (data.get("results_by_capacity") or {}).items():
            ours = (result or {}).get("tool_agent") or {}
            summary = ours.get("summary") or {}
            capacity = parse_int(summary.get("rsu_cache_capacity") or capacity_key, default=None)
            delay = parse_float(summary.get("average_delay_ms"), default=None)
            if capacity is not None and delay is not None:
                values[int(capacity)] = float(delay)
        if values:
            return {OURS_LABEL: values}
    return {}


def find_capacity_result(results: Dict[str, dict], capacity: int) -> Optional[dict]:
    if str(capacity) in results:
        return results[str(capacity)]
    for key, value in results.items():
        if parse_int(key, default=None) == int(capacity):
            return value
        summary = ((value or {}).get("summary") or {}) if isinstance(value, dict) else {}
        if parse_int(summary.get("rsu_cache_capacity") or summary.get("mrsu_cache_capacity"), default=None) == int(capacity):
            return value
    return None


def order_series(series: Dict[str, object]) -> Dict[str, object]:
    ordered = {}
    for label in METHOD_ORDER:
        if label in series:
            ordered[label] = series[label]
    for label in sorted(series):
        if label not in ordered:
            ordered[label] = series[label]
    return ordered


def normalize_llm_comparison_label(row: dict) -> str:
    model_name = str(row.get("model_name") or "").strip()
    if model_name:
        return f"FD-EMC({model_name})"
    label = str(row.get("method_label") or row.get("method") or "").strip()
    for prefix in ("FD-EMC w/", "FD-EMC with ", "FD-EMC w/ ", "DA-ELC w/", "DA-ELC with ", "DA-ELC w/ "):
        if label.startswith(prefix):
            return f"FD-EMC({label[len(prefix):].strip()})"
    if label.startswith("FD-EMC("):
        return label
    if label.startswith("DA-ELC("):
        return "FD-EMC(" + label[len("DA-ELC("):]
    return label or "FD-EMC"


def should_skip_llm_comparison_label(label: str) -> bool:
    return str(label).strip() in {"", OURS_LABEL}


def series_dicts_to_lists(series: Dict[str, Dict[int, float]], axis: List[int]) -> Dict[str, List[Optional[float]]]:
    return {
        label: [values.get(capacity) for capacity in axis]
        for label, values in series.items()
    }


def write_svg_line_chart(
    series: Dict[str, List[Optional[float]]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    x_labels: List[str],
    legend_position: str = "bottom_right",
    y_floor_zero: bool = True,
    y_label_x: float = 26.0,
    y_scale: float = 1.0,
    y_tick_decimals: int = 2,
) -> None:
    width, height = 980, 560
    left, right, top, bottom = 82, 48, 34, 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#2563eb", "#22c55e", "#e11d48", "#8b5cf6", "#06b6d4", "#facc15", "#ec4899"]
    markers = ["cross", "triangle", "star", "diamond", "square", "plus", "down_triangle"]
    grid_color = "#c4c4c4"
    values = [
        float(value) * float(y_scale)
        for row in series.values()
        for value in row
        if value is not None
    ]
    y_min, y_max = padded_range(values, floor_zero=y_floor_zero)
    max_len = max((len(row) for row in series.values()), default=1)

    def x_coord(idx: int) -> float:
        if max_len <= 1:
            return left + plot_w / 2
        return left + idx * plot_w / (max_len - 1)

    def y_coord(value: float) -> float:
        return top + (y_max - value) * plot_h / max(y_max - y_min, 1e-9)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_coord(value)
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            f'stroke="{grid_color}" stroke-dasharray="3 2"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{y + 7:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="22" font-weight="700">{format_tick(value, y_tick_decimals)}</text>'
        )

    for idx in range(1, max_len):
        x = x_coord(idx)
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            f'stroke="{grid_color}" stroke-dasharray="3 2"/>'
        )

    for idx, label in enumerate(x_labels):
        x = x_coord(idx)
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 38}" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape_xml(label)}</text>')

    legend_items = []
    plot_items = []
    for s_idx, (label, row) in enumerate(series.items()):
        color, marker = style_for_series(label, s_idx, colors, markers)
        legend_items.append((label, color, marker))
        plot_items.append((label, row, color, marker))

    plot_items.sort(key=lambda item: 1 if is_primary_ours_series(item[0]) else 0)
    for label, row, color, marker in plot_items:
        valid_points = [
            (idx, float(value) * float(y_scale))
            for idx, value in enumerate(row)
            if value is not None
        ]
        if valid_points:
            points = " ".join(f"{x_coord(idx):.2f},{y_coord(value):.2f}" for idx, value in valid_points)
            line_width = series_line_width(label)
            marker_size = series_marker_size(label)
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{line_width:.1f}"/>')
            for idx, value in valid_points:
                lines.extend(render_marker(x_coord(idx), y_coord(value), color, marker, marker_size))

    lines.extend(render_inside_legend(legend_items, left, top, plot_w, plot_h, legend_position))

    lines.append(f'<text x="{left + plot_w/2:.1f}" y="{height - 12}" text-anchor="middle" font-family="Arial" font-size="28" font-weight="700">{escape_xml(x_label)}</text>')
    lines.append(f'<text x="{y_label_x:.1f}" y="{top + plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 {y_label_x:.1f} {top + plot_h/2:.1f})" font-family="Arial" font-size="28" font-weight="700">{escape_xml(y_label)}</text>')
    lines.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def style_for_series(label: str, index: int, colors: List[str], markers: List[str]) -> Tuple[str, str]:
    normalized = str(label).strip().lower()
    if is_primary_ours_series(label):
        return "#f97316", "circle"
    if is_deepseek_comparison_series(label):
        return "#1d4ed8", "diamond"
    if normalized.startswith(("fd-emc(", "da-elc(")):
        return "#0f766e", "diamond"
    return colors[index % len(colors)], markers[index % len(markers)]


def is_primary_ours_series(label: str) -> bool:
    normalized = str(label).strip().lower()
    return normalized in {OURS_LABEL.lower(), "fd-emc", "da-elc", "ours", "da-elc(qwen3.7-flash)"}


def is_deepseek_comparison_series(label: str) -> bool:
    return str(label).strip().lower() in {
        "fd-emc(deepseek-v4-flash)",
        "da-elc(deepseek-v4-flash)",
    }


def series_line_width(label: str) -> float:
    return 2.1 if is_deepseek_comparison_series(label) else 2.8


def series_marker_size(label: str) -> float:
    return 4.2 if is_deepseek_comparison_series(label) else 5.2


def render_marker(cx: float, cy: float, color: str, marker: str, size: float) -> List[str]:
    if marker == "circle":
        return [
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{size:.2f}" fill="{color}" stroke="white" stroke-width="1.1"/>'
        ]
    if marker == "cross":
        return [
            f'<line x1="{cx - size:.2f}" y1="{cy - size:.2f}" x2="{cx + size:.2f}" y2="{cy + size:.2f}" stroke="{color}" stroke-width="2.2"/>',
            f'<line x1="{cx - size:.2f}" y1="{cy + size:.2f}" x2="{cx + size:.2f}" y2="{cy - size:.2f}" stroke="{color}" stroke-width="2.2"/>',
        ]
    if marker == "triangle":
        return [
            f'<polygon points="{cx:.2f},{cy - size:.2f} {cx + size:.2f},{cy + size:.2f} {cx - size:.2f},{cy + size:.2f}" fill="{color}" stroke="white" stroke-width="0.8"/>'
        ]
    if marker == "down_triangle":
        return [
            f'<polygon points="{cx - size:.2f},{cy - size:.2f} {cx + size:.2f},{cy - size:.2f} {cx:.2f},{cy + size:.2f}" fill="{color}" stroke="white" stroke-width="0.8"/>'
        ]
    if marker == "diamond":
        return [
            f'<polygon points="{cx:.2f},{cy - size:.2f} {cx + size:.2f},{cy:.2f} {cx:.2f},{cy + size:.2f} {cx - size:.2f},{cy:.2f}" fill="{color}" stroke="white" stroke-width="0.8"/>'
        ]
    if marker == "square":
        return [
            f'<rect x="{cx - size:.2f}" y="{cy - size:.2f}" width="{2 * size:.2f}" height="{2 * size:.2f}" fill="{color}" stroke="white" stroke-width="0.8"/>'
        ]
    if marker == "plus":
        return [
            f'<line x1="{cx - size:.2f}" y1="{cy:.2f}" x2="{cx + size:.2f}" y2="{cy:.2f}" stroke="{color}" stroke-width="2.2"/>',
            f'<line x1="{cx:.2f}" y1="{cy - size:.2f}" x2="{cx:.2f}" y2="{cy + size:.2f}" stroke="{color}" stroke-width="2.2"/>',
        ]
    return [
        f'<text x="{cx:.2f}" y="{cy + size * 0.58:.2f}" text-anchor="middle" font-family="Arial" font-size="{size * 3.0:.1f}" font-weight="700" fill="{color}">*</text>'
    ]


def padded_range(values: List[float], floor_zero: bool = True) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-9:
        pad = max(abs(high) * 0.08, 0.05 if floor_zero else 1.0)
    else:
        pad = (high - low) * 0.16
    y_min = low - pad
    y_max = high + pad
    if floor_zero:
        y_min = 0.0
        y_max = max(0.05, y_max)
    if y_max <= y_min:
        y_max = y_min + 1.0
    return float(y_min), float(y_max)


def render_inside_legend(
    legend_items: List[Tuple[str, str, str]],
    left: float,
    top: float,
    plot_w: float,
    plot_h: float,
    position: str,
) -> List[str]:
    if not legend_items:
        return []
    item_h = 40
    padding_x = 14
    padding_y = 12
    legend_w = max(200, min(360, max(len(str(label)) for label, _, _ in legend_items) * 11 + 82))
    legend_h = padding_y * 2 + item_h * len(legend_items)
    legend_x = left + plot_w - legend_w - 14
    if position == "top_right":
        legend_y = top + 12
    else:
        legend_y = top + plot_h - legend_h - 12
    lines = [
        (
            f'<rect x="{legend_x:.2f}" y="{legend_y:.2f}" width="{legend_w:.2f}" height="{legend_h:.2f}" '
            f'fill="white" fill-opacity="0.86" stroke="#cfcfcf" stroke-width="0.8" rx="4"/>'
        )
    ]
    for idx, (label, color, marker) in enumerate(legend_items):
        y = legend_y + padding_y + 22 + idx * item_h
        x = legend_x + padding_x
        lines.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x + 24:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="2.8"/>')
        lines.extend(render_marker(x + 12, y, color, marker, 4.8))
        lines.append(f'<text x="{x + 38:.2f}" y="{y + 8:.2f}" font-family="Arial" font-size="22" font-weight="700" fill="#111">{escape_xml(label)}</text>')
    return lines


def save_round_csv(series: Dict[str, List[float]], output_path: Path) -> None:
    max_len = max((len(values) for values in series.values()), default=0)
    fieldnames = ["round"] + list(series.keys())
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(max_len):
            row = {"round": idx}
            for label, values in series.items():
                row[label] = values[idx] if idx < len(values) else ""
            writer.writerow(row)


def save_capacity_csv(series: Dict[str, Dict[int, float]], axis: List[int], output_path: Path) -> None:
    fieldnames = ["rsu_cache_capacity"] + list(series.keys())
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for capacity in axis:
            row = {"rsu_cache_capacity": capacity}
            for label, values in series.items():
                row[label] = values.get(capacity, "")
            writer.writerow(row)


def read_csv(path: Path) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def escape_xml(text) -> str:
    return html.escape(str(text), quote=False)


def format_tick(value: float, decimals: int) -> str:
    decimals = max(int(decimals), 0)
    if decimals == 0:
        return str(int(round(float(value))))
    return f"{float(value):.{decimals}f}"


def local_rsu_chr_from_counts(hit_count: float, not_cached_count: float) -> float:
    denominator = float(hit_count) + float(not_cached_count)
    return float(hit_count) / denominator if denominator > 0.0 else 0.0


def row_local_rsu_counts(row: dict) -> Optional[Tuple[float, float]]:
    not_cached = parse_float(row.get("not_cached_count"), default=None)
    if not_cached is None:
        not_cached = parse_float(row.get("not_cached_count_mean"), default=None)
    if not_cached is None:
        return None
    hit = parse_float(row.get("hit_count"), default=None)
    if hit is None:
        hit = parse_float(row.get("hit_count_mean"), default=None)
    if hit is None:
        mrsu_hit = parse_float(row.get("mrsu_hit_count"), default=None)
        if mrsu_hit is None:
            mrsu_hit = parse_float(row.get("mrsu_hit_count_mean"), default=0.0)
        frsu_hit = parse_float(row.get("frsu_hit_count"), default=None)
        if frsu_hit is None:
            frsu_hit = parse_float(row.get("frsu_hit_count_mean"), default=0.0)
        hit = float(mrsu_hit or 0.0) + float(frsu_hit or 0.0)
    return float(hit or 0.0), float(not_cached or 0.0)


def row_local_rsu_chr(row: dict) -> Optional[float]:
    direct = parse_float(row.get("local_rsu_chr"), default=None)
    if direct is not None:
        return float(direct)
    counts = row_local_rsu_counts(row)
    if counts is None:
        return None
    return local_rsu_chr_from_counts(*counts)


def round_local_rsu_chr_from_logs(logs: List[dict]) -> List[float]:
    values: List[float] = []
    for log in logs:
        metrics = log.get("metrics") or {}
        direct = parse_float(metrics.get("local_rsu_chr", log.get("local_rsu_chr")), default=None)
        if direct is not None:
            values.append(float(direct))
            continue
        counts = row_local_rsu_counts(metrics)
        if counts is not None:
            values.append(local_rsu_chr_from_counts(*counts))
    return values


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _is_baseline_method_csv(rows: List[dict]) -> bool:
    if not rows:
        return False
    columns = set(rows[0].keys())
    return {"seed", "average_chr", "round", "chr", "method_label"}.issubset(columns)


def _valid_result_dir(path: Path) -> bool:
    return bool(path.name) and path.exists() and path.is_dir()


def is_usable_llm_comparison_dir(path: Path) -> bool:
    return (
        isinstance(path, Path)
        and path.name.startswith("llm_model_comparison_")
        and (path / "data").exists()
    )


def find_latest_baseline_dir(results_root: str) -> Path:
    root = Path(results_root)
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("traditional_baselines_")
        and (path / "data").exists()
    ] if root.exists() else []
    if not candidates:
        return Path("")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def find_latest_dqn_dir(results_root: str) -> Path:
    root = Path(results_root)
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("dqn_baseline_")
        and (path / "data").exists()
    ] if root.exists() else []
    if not candidates:
        return Path("")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def find_latest_llm_comparison_dir(results_root: str) -> Path:
    root = Path(results_root)
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("llm_model_comparison_")
        and (path / "data").exists()
    ] if root.exists() else []
    if not candidates:
        return Path("")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def find_latest_ours_dir(results_root: str) -> Path:
    root = Path(results_root)
    if not root.exists():
        return Path("")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("具身智能")
        and (path / "具身智能完整运行结果.json").exists()
    ]
    if not candidates:
        return Path("")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def create_output_dir(base_dir: str) -> Path:
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"基线绘图{timestamp}"
    counter = 1
    while path.exists():
        path = root / f"基线绘图{timestamp}_{counter:02d}"
        counter += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


if __name__ == "__main__":
    main()
