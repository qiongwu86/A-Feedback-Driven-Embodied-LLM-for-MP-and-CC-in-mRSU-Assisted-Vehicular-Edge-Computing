from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_ABLATION_METHODS = [
    "direct_llm",
    "open_loop_llm",
    "static_mrsu_llm",
]
DEFAULT_LABEL_ORDER = [
    "FD-EMC(qwen3.7-flash)",
    "Direct LLM",
    "Open-loop FD-EMC",
    "Static mRSU",
]
DEFAULT_OURS_MODEL_NAME = "qwen3.7-flash"
DEFAULT_OURS_LABEL = f"FD-EMC({DEFAULT_OURS_MODEL_NAME})"

LABEL_ALIASES = {
    "Static mRSU + LLM": "Static mRSU",
    "Static mRSU w/ LLM": "Static mRSU",
    "Static mRSU with LLM": "Static mRSU",
    "Static LLM": "Static mRSU",
    "Open-loop DA-ELC": "Open-loop FD-EMC",
    "Open-loop LLM": "Open-loop FD-EMC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot saved FD-EMC and ablation results without rerunning simulations."
    )
    parser.add_argument(
        "--ours-dir",
        type=str,
        default="",
        help="Directory from main_mrsu_tool_simulation.py. Auto-detects latest 具身智能* if omitted.",
    )
    parser.add_argument(
        "--ablation-dir",
        type=str,
        default="",
        help="Directory from run_ablation_experiment.py. Auto-detects latest ablation_experiment_* if omitted.",
    )
    parser.add_argument(
        "--direct-llm-dir",
        type=str,
        default="",
        help="Directory from run_direct_llm_experiment.py. Auto-detects latest direct_llm_experiment_* if omitted; skipped when absent.",
    )
    parser.add_argument(
        "--static-mrsu-dir",
        type=str,
        default="",
        help="Directory from run_static_mrsu_experiment.py. Auto-detects latest static_mrsu_experiment_* if omitted; skipped when absent.",
    )
    parser.add_argument(
        "--llm-comparison-dir",
        type=str,
        default="",
        help=(
            "Directory from run_llm_model_comparison_experiment.py. "
            "Auto-detects latest llm_model_comparison_* if omitted."
        ),
    )
    parser.add_argument(
        "--ours-model-name",
        type=str,
        default=DEFAULT_OURS_MODEL_NAME,
        help=(
            "Preferred FD-EMC backbone to draw as the main FD-EMC curve from "
            "llm_model_comparison_*; falls back to --ours-dir if unavailable."
        ),
    )
    parser.add_argument("--capacity", type=int, default=200, help="Capacity used for per-round CHR plot.")
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_ABLATION_METHODS),
        help="Comma-separated ablation method ids to plot.",
    )
    parser.add_argument("--ours-label", type=str, default=DEFAULT_OURS_LABEL)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_methods = parse_method_filter(args.methods)
    ours_dir = Path(args.ours_dir) if args.ours_dir else find_latest_ours_dir(args.results_root)
    ablation_dir = (
        Path(args.ablation_dir)
        if args.ablation_dir
        else find_latest_ablation_dir(args.results_root)
    )
    direct_llm_dir = (
        Path(args.direct_llm_dir)
        if args.direct_llm_dir
        else find_latest_direct_llm_dir(args.results_root)
    )
    static_mrsu_dir = (
        Path(args.static_mrsu_dir)
        if args.static_mrsu_dir
        else find_latest_static_mrsu_dir(args.results_root)
    )
    llm_comparison_dir = (
        Path(args.llm_comparison_dir)
        if args.llm_comparison_dir
        else find_latest_llm_comparison_dir(args.results_root)
    )
    if not ours_dir:
        raise FileNotFoundError("No 具身智能 result directory was found.")
    if not ablation_dir:
        raise FileNotFoundError("No ablation_experiment result directory was found.")

    output_dir = create_output_dir(args.output_dir)

    ours_model_source = describe_preferred_ours_source(
        ours_dir=ours_dir,
        llm_comparison_dir=llm_comparison_dir,
        model_name=args.ours_model_name,
    )

    round_series: Dict[str, List[Optional[float]]] = {}
    round_series.update(
        read_preferred_ours_round_series(
            ours_dir,
            llm_comparison_dir,
            args.capacity,
            args.ours_label,
            args.ours_model_name,
        )
    )
    round_series.update(read_ablation_round_series(ablation_dir, args.capacity, allowed_methods))
    round_series.update(read_direct_llm_round_series(direct_llm_dir, args.capacity))
    round_series.update(read_static_mrsu_round_series(static_mrsu_dir, args.capacity))
    round_series = order_series(round_series, args.ours_label)
    if not round_series:
        raise RuntimeError(f"No per-round ACHR data found for capacity {args.capacity}.")

    capacity_series: Dict[str, Dict[int, float]] = {}
    capacity_series.update(
        read_preferred_ours_capacity_series(
            ours_dir,
            llm_comparison_dir,
            args.ours_label,
            args.ours_model_name,
        )
    )
    capacity_series.update(read_ablation_capacity_series(ablation_dir, allowed_methods))
    capacity_series.update(read_direct_llm_capacity_series(direct_llm_dir))
    capacity_series.update(read_static_mrsu_capacity_series(static_mrsu_dir))
    capacity_series = order_series(capacity_series, args.ours_label)
    if not capacity_series:
        raise RuntimeError("No capacity-scan ACHR data was found.")

    capacity_axis = sorted(
        {
            int(capacity)
            for values in capacity_series.values()
            for capacity in values.keys()
        }
    )

    round_svg = output_dir / f"{args.capacity}容量消融每轮CHR对比.svg"
    capacity_svg = output_dir / "容量扫描消融ACHR对比.svg"
    round_csv = output_dir / f"{args.capacity}容量消融每轮CHR对比数据.csv"
    capacity_csv = output_dir / "容量扫描消融ACHR对比数据.csv"
    metadata_json = output_dir / "消融绘图数据.json"

    write_svg_line_chart(
        round_series,
        round_svg,
        f"Per-round Average Cache Hit Ratio at RSU Cache Capacity {args.capacity}",
        "Round",
        "Average Cache Hit Ratio (%)",
        x_labels=[str(idx) for idx in range(max(len(values) for values in round_series.values()))],
        legend_position="bottom_right",
        y_scale=100.0,
        y_tick_decimals=0,
    )
    write_svg_line_chart(
        series_dicts_to_lists(capacity_series, capacity_axis),
        capacity_svg,
        "Ablation ACHR vs Synchronized RSU Cache Capacity",
        "mRSU/sRSU Cache Capacity",
        "Average Cache Hit Ratio (%)",
        x_labels=[str(capacity) for capacity in capacity_axis],
        legend_position="bottom_right",
        y_scale=100.0,
        y_tick_decimals=0,
    )
    save_round_csv(round_series, round_csv)
    save_capacity_csv(capacity_series, capacity_axis, capacity_csv)

    system_round_series: Dict[str, List[Optional[float]]] = {}
    system_round_series.update(
        read_preferred_ours_round_system_series(
            ours_dir,
            llm_comparison_dir,
            args.capacity,
            args.ours_label,
            args.ours_model_name,
        )
    )
    system_round_series.update(read_ablation_round_system_series(ablation_dir, args.capacity, allowed_methods))
    system_round_series.update(read_direct_llm_round_system_series(direct_llm_dir, args.capacity))
    system_round_series.update(read_static_mrsu_round_system_series(static_mrsu_dir, args.capacity))
    system_round_series = order_series(system_round_series, args.ours_label)
    system_round_svg = output_dir / f"system_chr_round_capacity_{args.capacity}.svg"
    system_round_csv = output_dir / f"system_chr_round_capacity_{args.capacity}.csv"
    if system_round_series:
        write_svg_line_chart(
            system_round_series,
            system_round_svg,
            f"Per-round Average Cache Hit Ratio at RSU Cache Capacity {args.capacity}",
            "Round",
            "Average Cache Hit Ratio (%)",
            x_labels=[str(idx) for idx in range(max(len(values) for values in system_round_series.values()))],
            legend_position="bottom_right",
            y_scale=100.0,
            y_tick_decimals=0,
        )
        save_round_csv(system_round_series, system_round_csv)

    system_capacity_series: Dict[str, Dict[int, float]] = {}
    system_capacity_series.update(
        read_preferred_ours_capacity_system_series(
            ours_dir,
            llm_comparison_dir,
            args.ours_label,
            args.ours_model_name,
        )
    )
    system_capacity_series.update(read_ablation_capacity_system_series(ablation_dir, allowed_methods))
    system_capacity_series.update(read_direct_llm_capacity_system_series(direct_llm_dir))
    system_capacity_series.update(read_static_mrsu_capacity_system_series(static_mrsu_dir))
    system_capacity_series = order_series(system_capacity_series, args.ours_label)
    system_capacity_axis = sorted(
        {
            int(capacity)
            for values in system_capacity_series.values()
            for capacity in values.keys()
        }
    )
    system_capacity_svg = output_dir / "system_achr_capacity_scan.svg"
    system_capacity_csv = output_dir / "system_achr_capacity_scan.csv"
    if system_capacity_series:
        write_svg_line_chart(
            series_dicts_to_lists(system_capacity_series, system_capacity_axis),
            system_capacity_svg,
            "Ablation ACHR vs Synchronized RSU Cache Capacity",
            "mRSU/sRSU Cache Capacity",
            "Average Cache Hit Ratio (%)",
            x_labels=[str(capacity) for capacity in system_capacity_axis],
            legend_position="bottom_right",
            y_scale=100.0,
            y_tick_decimals=0,
        )
        save_capacity_csv(system_capacity_series, system_capacity_axis, system_capacity_csv)

    round_delay_series: Dict[str, List[Optional[float]]] = {}
    round_delay_series.update(
        read_preferred_ours_round_delay_series(
            ours_dir,
            llm_comparison_dir,
            args.capacity,
            args.ours_label,
            args.ours_model_name,
        )
    )
    round_delay_series.update(read_ablation_round_delay_series(ablation_dir, args.capacity, allowed_methods))
    round_delay_series.update(read_direct_llm_round_delay_series(direct_llm_dir, args.capacity))
    round_delay_series.update(read_static_mrsu_round_delay_series(static_mrsu_dir, args.capacity))
    round_delay_series = order_series(round_delay_series, args.ours_label)
    round_delay_svg = output_dir / f"{args.capacity}容量消融每轮delay对比.svg"
    round_delay_csv = output_dir / f"{args.capacity}容量消融每轮delay对比数据.csv"
    if round_delay_series:
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
        save_round_csv(round_delay_series, round_delay_csv)

    capacity_delay_series: Dict[str, Dict[int, float]] = {}
    capacity_delay_series.update(
        read_preferred_ours_capacity_delay_series(
            ours_dir,
            llm_comparison_dir,
            args.ours_label,
            args.ours_model_name,
        )
    )
    capacity_delay_series.update(read_ablation_capacity_delay_series(ablation_dir, allowed_methods))
    capacity_delay_series.update(read_direct_llm_capacity_delay_series(direct_llm_dir))
    capacity_delay_series.update(read_static_mrsu_capacity_delay_series(static_mrsu_dir))
    capacity_delay_series = order_series(capacity_delay_series, args.ours_label)
    delay_capacity_axis = sorted(
        {
            int(capacity)
            for values in capacity_delay_series.values()
            for capacity in values.keys()
        }
    )
    capacity_delay_svg = output_dir / "容量扫描消融delay对比.svg"
    capacity_delay_csv = output_dir / "容量扫描消融delay对比数据.csv"
    if capacity_delay_series:
        write_svg_line_chart(
            series_dicts_to_lists(capacity_delay_series, delay_capacity_axis),
            capacity_delay_svg,
            "Ablation Average Delay vs Synchronized RSU Cache Capacity",
            "mRSU/sRSU Cache Capacity",
            "Average Delay (ms)",
            x_labels=[str(capacity) for capacity in delay_capacity_axis],
            legend_position="top_right",
            y_floor_zero=False,
            y_tick_decimals=0,
        )
        save_capacity_csv(capacity_delay_series, delay_capacity_axis, capacity_delay_csv)
    save_json(
        {
            "ours_dir": str(ours_dir),
            "ours_model_name": str(args.ours_model_name),
            "ours_model_source": ours_model_source,
            "llm_comparison_dir": str(llm_comparison_dir) if is_usable_llm_comparison_dir(llm_comparison_dir) else "",
            "ablation_dir": str(ablation_dir),
            "direct_llm_dir": str(direct_llm_dir) if is_usable_direct_llm_dir(direct_llm_dir) else "",
            "static_mrsu_dir": str(static_mrsu_dir) if is_usable_static_mrsu_dir(static_mrsu_dir) else "",
            "round_capacity": int(args.capacity),
            "allowed_ablation_methods": allowed_methods,
            "round_series": round_series,
            "system_round_series": system_round_series,
            "round_delay_series": round_delay_series,
            "capacity_axis": capacity_axis,
            "system_capacity_axis": system_capacity_axis,
            "delay_capacity_axis": delay_capacity_axis,
            "capacity_series": capacity_series,
            "system_capacity_series": system_capacity_series,
            "capacity_delay_series": capacity_delay_series,
            "outputs": {
                "round_chr_plot": str(round_svg),
                "capacity_scan_plot": str(capacity_svg),
                "round_chr_data": str(round_csv),
                "capacity_scan_data": str(capacity_csv),
                "system_round_chr_plot": str(system_round_svg) if system_round_series else "",
                "system_capacity_scan_plot": str(system_capacity_svg) if system_capacity_series else "",
                "system_round_chr_data": str(system_round_csv) if system_round_series else "",
                "system_capacity_scan_data": str(system_capacity_csv) if system_capacity_series else "",
                "round_delay_plot": str(round_delay_svg) if round_delay_series else "",
                "capacity_delay_plot": str(capacity_delay_svg) if capacity_delay_series else "",
                "round_delay_data": str(round_delay_csv) if round_delay_series else "",
                "capacity_delay_data": str(capacity_delay_csv) if capacity_delay_series else "",
            },
        },
        metadata_json,
    )

    print("Ablation plotting finished.")
    print(f"FD-EMC plotted source: {ours_model_source}")
    print(f"LLM comparison source: {llm_comparison_dir if is_usable_llm_comparison_dir(llm_comparison_dir) else ''}")
    print(f"FD-EMC source: {ours_dir}")
    print(f"Ablation source: {ablation_dir}")
    if is_usable_direct_llm_dir(direct_llm_dir):
        print(f"Direct LLM source: {direct_llm_dir}")
    if is_usable_static_mrsu_dir(static_mrsu_dir):
        print(f"Static mRSU source: {static_mrsu_dir}")
    print(f"Saved to: {output_dir.resolve()}")


def read_ours_round_series(result_dir: Path, capacity: int, label: str) -> Dict[str, List[float]]:
    result_json = result_dir / "具身智能完整运行结果.json"
    if not result_json.exists():
        return {}
    data = read_json(result_json)
    result = find_capacity_result(data.get("results_by_capacity") or {}, capacity)
    if not result:
        return {}
    ours = result.get("tool_agent") or result.get("Ours") or {}
    summary = ((ours or {}).get("summary") or {})
    round_chr = summary.get("round_local_rsu_chr", summary.get("round_chr", [])) or []
    if not round_chr:
        round_chr = round_local_rsu_chr_from_logs((ours or {}).get("round_logs") or [])
    if not round_chr:
        return {}
    return {label: [float(value) for value in round_chr]}


def read_ours_capacity_series(result_dir: Path, label: str) -> Dict[str, Dict[int, float]]:
    summary_csv = result_dir / "具身智能ACHR汇总表.csv"
    values: Dict[int, float] = {}
    if summary_csv.exists():
        for row in read_csv(summary_csv):
            if row.get("method") != "tool_agent" and row.get("method_label") != "Ours":
                continue
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            if capacity is None:
                continue
            value = row.get("local_rsu_achr")
            if value in ("", None):
                counts = row_local_rsu_counts(row)
                value = local_rsu_chr_from_counts(*counts) if counts is not None else row.get("achr")
            values[int(capacity)] = float(value)
    if values:
        return {label: values}

    result_json = result_dir / "具身智能完整运行结果.json"
    if not result_json.exists():
        return {}
    data = read_json(result_json)
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
    return {label: values} if values else {}


def read_ablation_round_series(
    result_dir: Path,
    capacity: int,
    allowed_methods: List[str],
) -> Dict[str, List[float]]:
    data_dir = result_dir / "data"
    if not data_dir.exists():
        return {}
    series: Dict[str, List[float]] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not is_method_round_csv(rows):
            continue
        method = rows[0].get("method") or csv_path.stem
        if allowed_methods and method not in allowed_methods:
            continue
        selected = [
            row
            for row in rows
            if parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            == int(capacity)
        ]
        if not selected:
            continue
        label = normalize_method_label(selected[0].get("method_label") or method, method)
        values_by_round: Dict[int, List[float]] = {}
        for row in selected:
            round_index = parse_int(row.get("round"), default=0)
            value = row_local_rsu_chr(row)
            if value is None:
                value = parse_float(row.get("chr"))
            values_by_round.setdefault(round_index, []).append(float(value))
        if values_by_round:
            series[label] = [
                mean(values)
                for _, values in sorted(values_by_round.items())
            ]
    return series


def read_ablation_capacity_series(
    result_dir: Path,
    allowed_methods: List[str],
) -> Dict[str, Dict[int, float]]:
    aggregate_csv = result_dir / "data" / "aggregate_summary.csv"
    if aggregate_csv.exists():
        series: Dict[str, Dict[int, float]] = {}
        for row in read_csv(aggregate_csv):
            method = row.get("method") or ""
            if allowed_methods and method not in allowed_methods:
                continue
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            if capacity is None:
                continue
            label = normalize_method_label(row.get("method_label") or method, method)
            value = row.get("local_rsu_achr_mean")
            if value in ("", None):
                counts = row_local_rsu_counts(row)
                value = local_rsu_chr_from_counts(*counts) if counts is not None else row.get("achr_mean")
            series.setdefault(label, {})[int(capacity)] = float(value)
        if series:
            return series
    return read_ablation_capacity_series_from_method_csvs(result_dir, allowed_methods)


def read_ablation_capacity_series_from_method_csvs(
    result_dir: Path,
    allowed_methods: List[str],
) -> Dict[str, Dict[int, float]]:
    data_dir = result_dir / "data"
    if not data_dir.exists():
        return {}
    series: Dict[str, Dict[int, float]] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not is_method_round_csv(rows):
            continue
        method = rows[0].get("method") or csv_path.stem
        if allowed_methods and method not in allowed_methods:
            continue
        label = normalize_method_label(rows[0].get("method_label") or method, method)
        average_by_seed_capacity: Dict[Tuple[int, int], float] = {}
        count_by_seed_capacity: Dict[Tuple[int, int], Tuple[float, float]] = {}
        round_values_by_seed_capacity: Dict[Tuple[int, int], List[float]] = {}
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
                round_values_by_seed_capacity.setdefault(key, []).append(float(value))
        by_capacity: Dict[int, List[float]] = {}
        for (capacity, seed), (hit, not_cached) in count_by_seed_capacity.items():
            by_capacity.setdefault(int(capacity), []).append(local_rsu_chr_from_counts(hit, not_cached))
        for (capacity, _seed), value in average_by_seed_capacity.items():
            if (capacity, _seed) in count_by_seed_capacity:
                continue
            by_capacity.setdefault(int(capacity), []).append(float(value))
        for (capacity, seed), values in round_values_by_seed_capacity.items():
            if (capacity, seed) in count_by_seed_capacity or (capacity, seed) in average_by_seed_capacity:
                continue
            by_capacity.setdefault(int(capacity), []).append(mean(values))
        if by_capacity:
            series[label] = {
                int(capacity): mean(values)
                for capacity, values in sorted(by_capacity.items())
            }
    return series


def read_ours_round_system_series(result_dir: Path, capacity: int, label: str) -> Dict[str, List[float]]:
    result_json = _embodied_json_path(result_dir)
    if not result_json.exists():
        return {}
    data = read_json(result_json)
    result = find_capacity_result(data.get("results_by_capacity") or {}, capacity)
    if not result:
        return {}
    ours = result.get("tool_agent") or result.get("Ours") or {}
    summary = (ours or {}).get("summary") or {}
    values = [float(value) for value in (summary.get("round_chr") or [])]
    if not values:
        values = round_system_chr_from_logs((ours or {}).get("round_logs") or [])
    return {label: values} if values else {}


def read_ours_capacity_system_series(result_dir: Path, label: str) -> Dict[str, Dict[int, float]]:
    summary_csv = _embodied_summary_csv_path(result_dir)
    values: Dict[int, float] = {}
    if summary_csv.exists():
        for row in read_csv(summary_csv):
            if row.get("method") != "tool_agent" and row.get("method_label") not in ("Ours", label):
                continue
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            value = parse_float(row.get("achr"), default=None)
            if capacity is not None and value is not None:
                values[int(capacity)] = float(value)
    if values:
        return {label: values}

    result_json = _embodied_json_path(result_dir)
    if not result_json.exists():
        return {}
    data = read_json(result_json)
    for capacity_key, result in (data.get("results_by_capacity") or {}).items():
        ours = (result or {}).get("tool_agent") or {}
        summary = ours.get("summary") or {}
        capacity = parse_int(summary.get("rsu_cache_capacity") or capacity_key, default=None)
        value = parse_float(summary.get("achr"), default=None)
        if capacity is not None and value is not None:
            values[int(capacity)] = float(value)
    return {label: values} if values else {}


def read_ablation_round_system_series(
    result_dir: Path,
    capacity: int,
    allowed_methods: List[str],
) -> Dict[str, List[float]]:
    data_dir = result_dir / "data"
    if not data_dir.exists():
        return {}
    series: Dict[str, List[float]] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not is_method_round_csv(rows):
            continue
        method = rows[0].get("method") or csv_path.stem
        if allowed_methods and method not in allowed_methods:
            continue
        selected = [
            row
            for row in rows
            if parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            == int(capacity)
        ]
        if not selected:
            continue
        label = normalize_method_label(selected[0].get("method_label") or method, method)
        values_by_round: Dict[int, List[float]] = {}
        for row in selected:
            value = parse_float(row.get("chr"), default=None)
            if value is None:
                continue
            round_index = parse_int(row.get("round"), default=0)
            values_by_round.setdefault(round_index, []).append(float(value))
        if values_by_round:
            series[label] = [mean(values) for _, values in sorted(values_by_round.items())]
    return series


def read_ablation_capacity_system_series(
    result_dir: Path,
    allowed_methods: List[str],
) -> Dict[str, Dict[int, float]]:
    aggregate_csv = result_dir / "data" / "aggregate_summary.csv"
    if aggregate_csv.exists():
        series: Dict[str, Dict[int, float]] = {}
        for row in read_csv(aggregate_csv):
            method = row.get("method") or ""
            if allowed_methods and method not in allowed_methods:
                continue
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            value = parse_float(row.get("achr_mean"), default=None)
            if capacity is None or value is None:
                continue
            label = normalize_method_label(row.get("method_label") or method, method)
            series.setdefault(label, {})[int(capacity)] = float(value)
        if series:
            return series

    data_dir = result_dir / "data"
    if not data_dir.exists():
        return {}
    series: Dict[str, Dict[int, float]] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not is_method_round_csv(rows):
            continue
        method = rows[0].get("method") or csv_path.stem
        if allowed_methods and method not in allowed_methods:
            continue
        label = normalize_method_label(rows[0].get("method_label") or method, method)
        average_by_seed_capacity: Dict[Tuple[int, int], float] = {}
        round_values_by_seed_capacity: Dict[Tuple[int, int], List[float]] = {}
        for row in rows:
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            if capacity is None:
                continue
            seed = parse_int(row.get("seed"), default=0)
            key = (int(capacity), int(seed))
            average_chr = parse_float(row.get("average_chr"), default=None)
            round_chr = parse_float(row.get("chr"), default=None)
            if average_chr is not None:
                average_by_seed_capacity[key] = float(average_chr)
            elif round_chr is not None:
                round_values_by_seed_capacity.setdefault(key, []).append(float(round_chr))
        values_by_capacity: Dict[int, List[float]] = {}
        for (capacity, _seed), value in average_by_seed_capacity.items():
            values_by_capacity.setdefault(int(capacity), []).append(float(value))
        for (capacity, seed), values in round_values_by_seed_capacity.items():
            if (capacity, seed) in average_by_seed_capacity:
                continue
            values_by_capacity.setdefault(int(capacity), []).append(mean(values))
        if values_by_capacity:
            series[label] = {
                int(capacity): mean(values)
                for capacity, values in sorted(values_by_capacity.items())
            }
    return series


def round_system_chr_from_logs(logs: List[dict]) -> List[float]:
    values: List[float] = []
    for log in logs:
        metrics = log.get("metrics") or {}
        value = parse_float(metrics.get("chr", log.get("chr")), default=None)
        if value is not None:
            values.append(float(value))
    return values


def find_capacity_result(results: Dict[str, dict], capacity: int) -> Optional[dict]:
    if str(capacity) in results:
        return results[str(capacity)]
    for key, value in results.items():
        if parse_int(key, default=None) == int(capacity):
            return value
        summary = ((value or {}).get("summary") or {}) if isinstance(value, dict) else {}
        found = parse_int(summary.get("rsu_cache_capacity") or summary.get("mrsu_cache_capacity"), default=None)
        if found == int(capacity):
            return value
    return None


def parse_method_filter(text: str) -> List[str]:
    methods = []
    for item in text.split(","):
        key = item.strip()
        if key and key not in methods:
            methods.append(key)
    return methods


def normalize_method_label(label: str, method: str = "") -> str:
    if method == "static_mrsu_llm":
        return "Static mRSU"
    text = str(label or method)
    return LABEL_ALIASES.get(text, text)


def order_series(series: Dict[str, object], ours_label: str) -> Dict[str, object]:
    llm_model_labels = sorted(
        label
        for label in series
        if str(label).startswith(f"{ours_label} w/")
        or str(label).startswith(f"{ours_label} with ")
    )
    order = (
        [ours_label]
        + llm_model_labels
        + [label for label in DEFAULT_LABEL_ORDER if label != ours_label]
    )
    ordered = {}
    for label in order:
        if label in series:
            ordered[label] = series[label]
    for label in sorted(series):
        if label not in ordered:
            ordered[label] = series[label]
    return ordered


def series_dicts_to_lists(
    series: Dict[str, Dict[int, float]],
    axis: List[int],
) -> Dict[str, List[Optional[float]]]:
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
    y_label_x: float = 22.0,
    y_scale: float = 1.0,
    y_tick_decimals: int = 2,
) -> None:
    width, height = 980, 560
    left, right, top, bottom = 82, 48, 34, 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = [
        "#2563eb",
        "#22c55e",
        "#e11d48",
        "#8b5cf6",
        "#06b6d4",
        "#facc15",
        "#ec4899",
    ]
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
            f'<text x="{left - 10}" y="{y + 6:.2f}" text-anchor="end" '
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
        lines.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 33}" text-anchor="middle" '
            f'font-family="Arial" font-size="22" font-weight="700">{escape_xml(label)}</text>'
        )

    legend_items = []
    for s_idx, (label, row) in enumerate(series.items()):
        color, marker = style_for_series(label, s_idx, colors, markers)
        legend_items.append((label, color, marker))
        valid_points = [
            (idx, float(value) * float(y_scale))
            for idx, value in enumerate(row)
            if value is not None
        ]
        if valid_points:
            points = " ".join(
                f"{x_coord(idx):.2f},{y_coord(value):.2f}"
                for idx, value in valid_points
            )
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.8"/>')
            for idx, value in valid_points:
                lines.extend(render_marker(x_coord(idx), y_coord(value), color, marker, 5.2))

    lines.extend(render_inside_legend(legend_items, left, top, plot_w, plot_h, legend_position))

    lines.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" '
        f'font-family="Arial" font-size="28" font-weight="700">{escape_xml(x_label)}</text>'
    )
    lines.append(
        f'<text x="{y_label_x:.1f}" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 {y_label_x:.1f} {top + plot_h / 2:.1f})" '
        f'font-family="Arial" font-size="28" font-weight="700">{escape_xml(y_label)}</text>'
    )
    lines.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def style_for_series(label: str, index: int, colors: List[str], markers: List[str]) -> Tuple[str, str]:
    normalized = str(label).strip().lower()
    if normalized in {"fd-emc", "da-elc", "ours"} or normalized.startswith(("fd-emc(", "da-elc(")):
        return "#f97316", "circle"
    return colors[index % len(colors)], markers[index % len(markers)]


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
    legend_w = max(200, min(380, max(len(str(label)) for label, _, _ in legend_items) * 11 + 82))
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
        lines.append(
            f'<text x="{x + 38:.2f}" y="{y + 8:.2f}" '
            f'font-family="Arial" font-size="22" font-weight="700" fill="#111">{escape_xml(label)}</text>'
        )
    return lines


def save_round_csv(series: Dict[str, List[Optional[float]]], output_path: Path) -> None:
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


def save_capacity_csv(
    series: Dict[str, Dict[int, float]],
    axis: List[int],
    output_path: Path,
) -> None:
    fieldnames = ["rsu_cache_capacity"] + list(series.keys())
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for capacity in axis:
            row = {"rsu_cache_capacity": capacity}
            for label, values in series.items():
                row[label] = values.get(capacity, "")
            writer.writerow(row)


def is_method_round_csv(rows: List[dict]) -> bool:
    if not rows:
        return False
    columns = set(rows[0].keys())
    return {"seed", "average_chr", "method", "method_label", "round", "chr"}.issubset(columns)


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


def find_latest_ablation_dir(results_root: str) -> Path:
    root = Path(results_root)
    if not root.exists():
        return Path("")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("ablation_experiment_")
        and (path / "data").exists()
    ]
    if not candidates:
        return Path("")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def create_output_dir(base_dir: str) -> Path:
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"消融绘图{timestamp}"
    suffix = 1
    while path.exists():
        path = root / f"消融绘图{timestamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


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


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def escape_xml(text) -> str:
    return html.escape(str(text), quote=False)


def _embodied_json_path(result_dir: Path) -> Path:
    return result_dir / "具身智能完整运行结果.json"


def _embodied_summary_csv_path(result_dir: Path) -> Path:
    return result_dir / "具身智能ACHR汇总表.csv"


def read_ours_round_series(result_dir: Path, capacity: int, label: str) -> Dict[str, List[float]]:
    result_json = _embodied_json_path(result_dir)
    if not result_json.exists():
        return {}
    data = read_json(result_json)
    result = find_capacity_result(data.get("results_by_capacity") or {}, capacity)
    if not result:
        return {}
    ours = result.get("tool_agent") or result.get("Ours") or {}
    summary = ((ours or {}).get("summary") or {})
    round_chr = summary.get("round_local_rsu_chr", summary.get("round_chr", [])) or []
    if not round_chr:
        round_chr = round_local_rsu_chr_from_logs((ours or {}).get("round_logs") or [])
    return {label: [float(value) for value in round_chr]} if round_chr else {}


def read_ours_capacity_series(result_dir: Path, label: str) -> Dict[str, Dict[int, float]]:
    summary_csv = _embodied_summary_csv_path(result_dir)
    values: Dict[int, float] = {}
    if summary_csv.exists():
        for row in read_csv(summary_csv):
            if row.get("method") != "tool_agent" and row.get("method_label") not in ("Ours", label):
                continue
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            if capacity is not None:
                value = row.get("local_rsu_achr")
                if value in ("", None):
                    counts = row_local_rsu_counts(row)
                    value = local_rsu_chr_from_counts(*counts) if counts is not None else row.get("achr")
                values[int(capacity)] = float(value)
    if values:
        return {label: values}

    result_json = _embodied_json_path(result_dir)
    if not result_json.exists():
        return {}
    data = read_json(result_json)
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
    return {label: values} if values else {}


def read_ours_round_delay_series(result_dir: Path, capacity: int, label: str) -> Dict[str, List[float]]:
    result_json = _embodied_json_path(result_dir)
    if not result_json.exists():
        return {}
    data = read_json(result_json)
    result = find_capacity_result(data.get("results_by_capacity") or {}, capacity)
    if not result:
        return {}
    ours = result.get("tool_agent") or result.get("Ours") or {}
    summary = ((ours or {}).get("summary") or {})
    round_delay = summary.get("round_delay_ms") or []
    if round_delay:
        return {label: [float(value) for value in round_delay]}
    values = []
    for log in (ours or {}).get("round_logs") or []:
        latency = log.get("latency") or {}
        value = latency.get("average_delay_ms", log.get("round_delay_ms"))
        if value not in ("", None):
            values.append(float(value))
    return {label: values} if values else {}


def read_ours_capacity_delay_series(result_dir: Path, label: str) -> Dict[str, Dict[int, float]]:
    summary_csv = _embodied_summary_csv_path(result_dir)
    values: Dict[int, float] = {}
    if summary_csv.exists():
        for row in read_csv(summary_csv):
            if row.get("method") != "tool_agent" and row.get("method_label") not in ("Ours", label):
                continue
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            delay = parse_float(row.get("average_delay_ms"), default=None)
            if capacity is not None and delay is not None:
                values[int(capacity)] = float(delay)
    if values:
        return {label: values}

    result_json = _embodied_json_path(result_dir)
    if not result_json.exists():
        return {}
    data = read_json(result_json)
    for capacity_key, result in (data.get("results_by_capacity") or {}).items():
        ours = (result or {}).get("tool_agent") or {}
        summary = ours.get("summary") or {}
        capacity = parse_int(summary.get("rsu_cache_capacity") or capacity_key, default=None)
        delay = parse_float(summary.get("average_delay_ms"), default=None)
        if capacity is not None and delay is not None:
            values[int(capacity)] = float(delay)
    return {label: values} if values else {}


def read_ablation_round_delay_series(
    result_dir: Path,
    capacity: int,
    allowed_methods: List[str],
) -> Dict[str, List[float]]:
    data_dir = result_dir / "data"
    if not data_dir.exists():
        return {}
    series: Dict[str, List[float]] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not is_method_round_csv(rows):
            continue
        method = rows[0].get("method") or csv_path.stem
        if allowed_methods and method not in allowed_methods:
            continue
        selected = [
            row
            for row in rows
            if parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            == int(capacity)
        ]
        if not selected:
            continue
        label = normalize_method_label(selected[0].get("method_label") or method, method)
        values_by_round: Dict[int, List[float]] = {}
        for row in selected:
            value = parse_float(row.get("round_delay_ms"), default=None)
            if value is None:
                continue
            round_index = parse_int(row.get("round"), default=0)
            values_by_round.setdefault(round_index, []).append(float(value))
        if values_by_round:
            series[label] = [mean(values) for _, values in sorted(values_by_round.items())]
    return series


def read_ablation_capacity_delay_series(
    result_dir: Path,
    allowed_methods: List[str],
) -> Dict[str, Dict[int, float]]:
    aggregate_csv = result_dir / "data" / "aggregate_summary.csv"
    if aggregate_csv.exists():
        series: Dict[str, Dict[int, float]] = {}
        for row in read_csv(aggregate_csv):
            method = row.get("method") or ""
            if allowed_methods and method not in allowed_methods:
                continue
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            delay = parse_float(row.get("average_delay_ms_mean"), default=None)
            if capacity is None or delay is None:
                continue
            label = normalize_method_label(row.get("method_label") or method, method)
            series.setdefault(label, {})[int(capacity)] = float(delay)
        if series:
            return series

    data_dir = result_dir / "data"
    if not data_dir.exists():
        return {}
    series: Dict[str, Dict[int, float]] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        rows = read_csv(csv_path)
        if not is_method_round_csv(rows):
            continue
        method = rows[0].get("method") or csv_path.stem
        if allowed_methods and method not in allowed_methods:
            continue
        label = normalize_method_label(rows[0].get("method_label") or method, method)
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
        values_by_capacity: Dict[int, List[float]] = {}
        for (capacity, _seed), value in average_by_seed_capacity.items():
            values_by_capacity.setdefault(int(capacity), []).append(float(value))
        for (capacity, seed), values in round_values_by_seed_capacity.items():
            if (capacity, seed) in average_by_seed_capacity:
                continue
            values_by_capacity.setdefault(int(capacity), []).append(mean(values))
        if values_by_capacity:
            series[label] = {
                int(capacity): mean(values)
                for capacity, values in sorted(values_by_capacity.items())
            }
    return series


def read_llm_comparison_round_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    if not is_usable_llm_comparison_dir(result_dir):
        return {}
    return read_ablation_round_series(result_dir, capacity, [])


def read_llm_comparison_capacity_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    if not is_usable_llm_comparison_dir(result_dir):
        return {}
    return read_ablation_capacity_series(result_dir, [])


def read_llm_comparison_round_system_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    if not is_usable_llm_comparison_dir(result_dir):
        return {}
    return read_ablation_round_system_series(result_dir, capacity, [])


def read_llm_comparison_capacity_system_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    if not is_usable_llm_comparison_dir(result_dir):
        return {}
    return read_ablation_capacity_system_series(result_dir, [])


def read_llm_comparison_round_delay_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    if not is_usable_llm_comparison_dir(result_dir):
        return {}
    return read_ablation_round_delay_series(result_dir, capacity, [])


def read_llm_comparison_capacity_delay_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    if not is_usable_llm_comparison_dir(result_dir):
        return {}
    return read_ablation_capacity_delay_series(result_dir, [])


def read_preferred_ours_round_series(
    ours_dir: Path,
    llm_comparison_dir: Path,
    capacity: int,
    label: str,
    model_name: str,
) -> Dict[str, List[float]]:
    preferred = read_llm_model_round_series(llm_comparison_dir, capacity, model_name, label)
    return preferred or read_ours_round_series(ours_dir, capacity, label)


def read_preferred_ours_capacity_series(
    ours_dir: Path,
    llm_comparison_dir: Path,
    label: str,
    model_name: str,
) -> Dict[str, Dict[int, float]]:
    preferred = read_llm_model_capacity_series(llm_comparison_dir, model_name, label)
    return preferred or read_ours_capacity_series(ours_dir, label)


def read_preferred_ours_round_system_series(
    ours_dir: Path,
    llm_comparison_dir: Path,
    capacity: int,
    label: str,
    model_name: str,
) -> Dict[str, List[float]]:
    preferred = read_llm_model_round_system_series(llm_comparison_dir, capacity, model_name, label)
    return preferred or read_ours_round_system_series(ours_dir, capacity, label)


def read_preferred_ours_capacity_system_series(
    ours_dir: Path,
    llm_comparison_dir: Path,
    label: str,
    model_name: str,
) -> Dict[str, Dict[int, float]]:
    preferred = read_llm_model_capacity_system_series(llm_comparison_dir, model_name, label)
    return preferred or read_ours_capacity_system_series(ours_dir, label)


def read_preferred_ours_round_delay_series(
    ours_dir: Path,
    llm_comparison_dir: Path,
    capacity: int,
    label: str,
    model_name: str,
) -> Dict[str, List[float]]:
    preferred = read_llm_model_round_delay_series(llm_comparison_dir, capacity, model_name, label)
    return preferred or read_ours_round_delay_series(ours_dir, capacity, label)


def read_preferred_ours_capacity_delay_series(
    ours_dir: Path,
    llm_comparison_dir: Path,
    label: str,
    model_name: str,
) -> Dict[str, Dict[int, float]]:
    preferred = read_llm_model_capacity_delay_series(llm_comparison_dir, model_name, label)
    return preferred or read_ours_capacity_delay_series(ours_dir, label)


def read_llm_model_round_series(
    result_dir: Path,
    capacity: int,
    model_name: str,
    label: str,
) -> Dict[str, List[float]]:
    return read_llm_model_round_metric_series(
        result_dir=result_dir,
        capacity=capacity,
        model_name=model_name,
        label=label,
        metric="local",
    )


def read_llm_model_round_system_series(
    result_dir: Path,
    capacity: int,
    model_name: str,
    label: str,
) -> Dict[str, List[float]]:
    return read_llm_model_round_metric_series(
        result_dir=result_dir,
        capacity=capacity,
        model_name=model_name,
        label=label,
        metric="system",
    )


def read_llm_model_round_delay_series(
    result_dir: Path,
    capacity: int,
    model_name: str,
    label: str,
) -> Dict[str, List[float]]:
    return read_llm_model_round_metric_series(
        result_dir=result_dir,
        capacity=capacity,
        model_name=model_name,
        label=label,
        metric="delay",
    )


def read_llm_model_round_metric_series(
    result_dir: Path,
    capacity: int,
    model_name: str,
    label: str,
    metric: str,
) -> Dict[str, List[float]]:
    rows = llm_model_rows(result_dir, model_name)
    if not rows:
        return {}
    selected = [
        row
        for row in rows
        if parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
        == int(capacity)
    ]
    if not selected:
        return {}
    values_by_round: Dict[int, List[float]] = {}
    for row in selected:
        round_index = parse_int(row.get("round"), default=0)
        if metric == "delay":
            value = parse_float(row.get("round_delay_ms"), default=None)
        elif metric == "system":
            value = parse_float(row.get("chr"), default=None)
        else:
            value = row_local_rsu_chr(row)
            if value is None:
                value = parse_float(row.get("chr"), default=None)
        if value is None:
            continue
        values_by_round.setdefault(round_index, []).append(float(value))
    if not values_by_round:
        return {}
    return {label: [mean(values) for _, values in sorted(values_by_round.items())]}


def read_llm_model_capacity_series(
    result_dir: Path,
    model_name: str,
    label: str,
) -> Dict[str, Dict[int, float]]:
    return read_llm_model_capacity_metric_series(result_dir, model_name, label, metric="local")


def read_llm_model_capacity_system_series(
    result_dir: Path,
    model_name: str,
    label: str,
) -> Dict[str, Dict[int, float]]:
    return read_llm_model_capacity_metric_series(result_dir, model_name, label, metric="system")


def read_llm_model_capacity_delay_series(
    result_dir: Path,
    model_name: str,
    label: str,
) -> Dict[str, Dict[int, float]]:
    return read_llm_model_capacity_metric_series(result_dir, model_name, label, metric="delay")


def read_llm_model_capacity_metric_series(
    result_dir: Path,
    model_name: str,
    label: str,
    metric: str,
) -> Dict[str, Dict[int, float]]:
    aggregate_csv = result_dir / "data" / "aggregate_summary.csv"
    values: Dict[int, float] = {}
    if is_usable_llm_comparison_dir(result_dir) and aggregate_csv.exists():
        for row in read_csv(aggregate_csv):
            if not llm_model_row_matches(row, model_name):
                continue
            capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
            if capacity is None:
                continue
            if metric == "delay":
                value = parse_float(row.get("average_delay_ms_mean"), default=None)
            elif metric == "system":
                value = parse_float(row.get("achr_mean"), default=None)
            else:
                value = parse_float(row.get("local_rsu_achr_mean"), default=None)
                if value is None:
                    counts = row_local_rsu_counts(row)
                    value = (
                        local_rsu_chr_from_counts(*counts)
                        if counts is not None
                        else parse_float(row.get("achr_mean"), default=None)
                    )
            if value is not None:
                values[int(capacity)] = float(value)
    if values:
        return {label: values}
    return read_llm_model_capacity_metric_series_from_rounds(result_dir, model_name, label, metric)


def read_llm_model_capacity_metric_series_from_rounds(
    result_dir: Path,
    model_name: str,
    label: str,
    metric: str,
) -> Dict[str, Dict[int, float]]:
    rows = llm_model_rows(result_dir, model_name)
    if not rows:
        return {}
    average_by_seed_capacity: Dict[Tuple[int, int], float] = {}
    count_by_seed_capacity: Dict[Tuple[int, int], Tuple[float, float]] = {}
    round_values_by_seed_capacity: Dict[Tuple[int, int], List[float]] = {}
    for row in rows:
        capacity = parse_int(row.get("rsu_cache_capacity") or row.get("mrsu_cache_capacity"), default=None)
        if capacity is None:
            continue
        seed = parse_int(row.get("seed"), default=0)
        key = (int(capacity), int(seed))
        if metric == "delay":
            average = parse_float(row.get("average_delay_ms"), default=None)
            value = parse_float(row.get("round_delay_ms"), default=None)
        elif metric == "system":
            average = parse_float(row.get("average_chr"), default=None)
            value = parse_float(row.get("chr"), default=None)
        else:
            counts = row_local_rsu_counts(row)
            if counts is not None:
                hit, not_cached = counts
                old_hit, old_not_cached = count_by_seed_capacity.get(key, (0.0, 0.0))
                count_by_seed_capacity[key] = (old_hit + hit, old_not_cached + not_cached)
                continue
            average = parse_float(row.get("average_local_rsu_chr"), default=None)
            value = row_local_rsu_chr(row)
        if average is not None:
            average_by_seed_capacity[key] = float(average)
        elif value is not None:
            round_values_by_seed_capacity.setdefault(key, []).append(float(value))
    values_by_capacity: Dict[int, List[float]] = {}
    for (capacity, _seed), (hit, not_cached) in count_by_seed_capacity.items():
        values_by_capacity.setdefault(int(capacity), []).append(local_rsu_chr_from_counts(hit, not_cached))
    for (capacity, seed), value in average_by_seed_capacity.items():
        if (capacity, seed) in count_by_seed_capacity:
            continue
        values_by_capacity.setdefault(int(capacity), []).append(float(value))
    for (capacity, seed), values in round_values_by_seed_capacity.items():
        if (capacity, seed) in count_by_seed_capacity or (capacity, seed) in average_by_seed_capacity:
            continue
        values_by_capacity.setdefault(int(capacity), []).append(mean(values))
    if not values_by_capacity:
        return {}
    return {
        label: {
            int(capacity): mean(values)
            for capacity, values in sorted(values_by_capacity.items())
        }
    }


def llm_model_rows(result_dir: Path, model_name: str) -> List[dict]:
    if not is_usable_llm_comparison_dir(result_dir):
        return []
    data_dir = result_dir / "data"
    if not data_dir.exists():
        return []
    rows: List[dict] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        csv_rows = read_csv(csv_path)
        if not csv_rows or not is_method_round_csv(csv_rows):
            continue
        if not any(llm_model_row_matches(row, model_name, csv_path.stem) for row in csv_rows[:3]):
            continue
        rows.extend(row for row in csv_rows if llm_model_row_matches(row, model_name, csv_path.stem))
    return rows


def llm_model_row_matches(row: dict, model_name: str, file_stem: str = "") -> bool:
    wanted = normalize_model_token(model_name)
    if not wanted:
        return False
    candidates = [
        row.get("model_name"),
        row.get("method_label"),
        row.get("method"),
        file_stem,
    ]
    return any(wanted in normalize_model_token(candidate) for candidate in candidates if candidate not in ("", None))


def normalize_model_token(value: object) -> str:
    text = str(value or "").lower().strip()
    for prefix in ("fd-emc w/", "fd-emc with ", "fd-emc(", "da-elc w/", "da-elc with ", "da-elc("):
        text = text.replace(prefix, "")
    return "".join(ch for ch in text if ch.isalnum())


def describe_preferred_ours_source(ours_dir: Path, llm_comparison_dir: Path, model_name: str) -> str:
    if read_llm_model_capacity_series(llm_comparison_dir, model_name, "FD-EMC"):
        return f"llm_model_comparison:{model_name}"
    return f"embodied_dir:{ours_dir}"


def read_direct_llm_round_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    if not is_usable_direct_llm_dir(result_dir):
        return {}
    return read_ablation_round_series(result_dir, capacity, ["direct_llm"])


def read_direct_llm_capacity_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    if not is_usable_direct_llm_dir(result_dir):
        return {}
    return read_ablation_capacity_series(result_dir, ["direct_llm"])


def read_direct_llm_round_system_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    if not is_usable_direct_llm_dir(result_dir):
        return {}
    return read_ablation_round_system_series(result_dir, capacity, ["direct_llm"])


def read_direct_llm_capacity_system_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    if not is_usable_direct_llm_dir(result_dir):
        return {}
    return read_ablation_capacity_system_series(result_dir, ["direct_llm"])


def read_direct_llm_round_delay_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    if not is_usable_direct_llm_dir(result_dir):
        return {}
    return read_ablation_round_delay_series(result_dir, capacity, ["direct_llm"])


def read_direct_llm_capacity_delay_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    if not is_usable_direct_llm_dir(result_dir):
        return {}
    return read_ablation_capacity_delay_series(result_dir, ["direct_llm"])


def read_static_mrsu_round_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    if not is_usable_static_mrsu_dir(result_dir):
        return {}
    return read_ablation_round_series(result_dir, capacity, ["static_mrsu_llm"])


def read_static_mrsu_capacity_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    if not is_usable_static_mrsu_dir(result_dir):
        return {}
    return read_ablation_capacity_series(result_dir, ["static_mrsu_llm"])


def read_static_mrsu_round_system_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    if not is_usable_static_mrsu_dir(result_dir):
        return {}
    return read_ablation_round_system_series(result_dir, capacity, ["static_mrsu_llm"])


def read_static_mrsu_capacity_system_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    if not is_usable_static_mrsu_dir(result_dir):
        return {}
    return read_ablation_capacity_system_series(result_dir, ["static_mrsu_llm"])


def read_static_mrsu_round_delay_series(result_dir: Path, capacity: int) -> Dict[str, List[float]]:
    if not is_usable_static_mrsu_dir(result_dir):
        return {}
    return read_ablation_round_delay_series(result_dir, capacity, ["static_mrsu_llm"])


def read_static_mrsu_capacity_delay_series(result_dir: Path) -> Dict[str, Dict[int, float]]:
    if not is_usable_static_mrsu_dir(result_dir):
        return {}
    return read_ablation_capacity_delay_series(result_dir, ["static_mrsu_llm"])


def is_usable_llm_comparison_dir(path: Path) -> bool:
    return (
        isinstance(path, Path)
        and path.name.startswith("llm_model_comparison_")
        and (path / "data").exists()
    )


def is_usable_direct_llm_dir(path: Path) -> bool:
    return (
        isinstance(path, Path)
        and path.name.startswith("direct_llm_experiment_")
        and (path / "data").exists()
    )


def is_usable_static_mrsu_dir(path: Path) -> bool:
    return (
        isinstance(path, Path)
        and path.name.startswith("static_mrsu_experiment_")
        and (path / "data").exists()
    )


def find_latest_ours_dir(results_root: str) -> Path:
    root = Path(results_root)
    if not root.exists():
        return Path("")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("具身智能")
        and _embodied_json_path(path).exists()
    ]
    if not candidates:
        return Path("")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def find_latest_llm_comparison_dir(results_root: str) -> Path:
    root = Path(results_root)
    if not root.exists():
        return Path("")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("llm_model_comparison_")
        and (path / "data").exists()
    ]
    if not candidates:
        return Path("")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def find_latest_direct_llm_dir(results_root: str) -> Path:
    root = Path(results_root)
    if not root.exists():
        return Path("")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("direct_llm_experiment_")
        and (path / "data").exists()
    ]
    if not candidates:
        return Path("")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def find_latest_static_mrsu_dir(results_root: str) -> Path:
    root = Path(results_root)
    if not root.exists():
        return Path("")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("static_mrsu_experiment_")
        and (path / "data").exists()
    ]
    if not candidates:
        return Path("")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def create_output_dir(base_dir: str) -> Path:
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"消融绘图{timestamp}"
    suffix = 1
    while path.exists():
        path = root / f"消融绘图{timestamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


if __name__ == "__main__":
    main()
