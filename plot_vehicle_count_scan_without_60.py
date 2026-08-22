from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_VEHICLE_COUNTS = [10, 20, 30, 40, 50]
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DATA_FILENAMES = [
    "vehicle_count_scan_plot_data.csv",
    "vehicle_count_scan_plot_data_without_60.csv",
    "vehicle_count_scan_summary.csv",
]
FILTERED_CHART_FILENAME = "\u8f66\u8f86\u6570\u626b\u63cfCHR\u65f6\u5ef6\u53cc\u8f74\u67f1\u72b6\u56fe_\u65e060.svg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replot an existing vehicle-count scan without rerunning simulation. "
            "By default it keeps vehicle counts 10,20,30,40,50 and filters out 60."
        )
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="",
        help="Existing vehicle_count_scan_* result directory. Defaults to the newest usable one under --results-root.",
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument(
        "--vehicle-counts",
        type=str,
        default=",".join(str(value) for value in DEFAULT_VEHICLE_COUNTS),
        help="Comma-separated vehicle counts to keep. Defaults to 10,20,30,40,50.",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--title", type=str, default="FD-EMC under Different Numbers of Vehicles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vehicle_counts = parse_int_list(args.vehicle_counts, "vehicle-counts")
    source_dir = resolve_project_path(args.result_dir) if args.result_dir else find_latest_vehicle_scan(resolve_project_path(args.results_root))
    if not source_dir:
        raise FileNotFoundError(
            "No usable vehicle_count_scan_* directory was found. "
            "Pass --result-dir to an existing result directory."
        )

    rows = read_vehicle_count_rows(source_dir)
    filtered_rows = [row for row in rows if int(row["vehicle_num"]) in vehicle_counts]
    filtered_rows.sort(key=lambda row: vehicle_counts.index(int(row["vehicle_num"])))
    plotted_counts = [int(row["vehicle_num"]) for row in filtered_rows]
    missing_counts = [count for count in vehicle_counts if count not in plotted_counts]
    if not filtered_rows:
        raise ValueError(f"No requested vehicle counts were found in {source_dir}.")

    output_dir = create_output_dir(resolve_project_path(args.output_dir))
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=False)

    filtered_csv = data_dir / "vehicle_count_scan_plot_data_without_60.csv"
    write_filtered_csv(filtered_csv, filtered_rows)
    chart_path = output_dir / FILTERED_CHART_FILENAME
    write_dual_axis_bar_chart(
        vehicle_counts=plotted_counts,
        chr_values=[float(row["achr"]) for row in filtered_rows],
        delay_values=[float(row["average_delay_ms"]) for row in filtered_rows],
        output_path=chart_path,
        title=args.title,
    )

    metadata = {
        "experiment": "vehicle_count_scan_replot_without_60",
        "source_result_dir": str(source_dir),
        "requested_vehicle_counts": vehicle_counts,
        "plotted_vehicle_counts": plotted_counts,
        "missing_vehicle_counts": missing_counts,
        "data_csv": str(filtered_csv),
        "plot": str(chart_path),
    }
    with open(output_dir / "vehicle_count_scan_replot_without_60.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Vehicle-count scan replot finished.")
    print(f"Source: {source_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Plot: {chart_path.resolve()}")
    if missing_counts:
        print(f"Missing requested vehicle counts skipped: {missing_counts}")


def find_latest_vehicle_scan(results_root: Path) -> Optional[Path]:
    if not results_root.exists():
        return None
    candidates = []
    for path in results_root.iterdir():
        if not path.is_dir() or not path.name.startswith("vehicle_count_scan_"):
            continue
        data_dir = path / "data"
        if any((data_dir / filename).exists() for filename in SOURCE_DATA_FILENAMES):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def read_vehicle_count_rows(result_dir: Path) -> List[Dict[str, float]]:
    plot_csv = result_dir / "data" / "vehicle_count_scan_plot_data.csv"
    filtered_plot_csv = result_dir / "data" / "vehicle_count_scan_plot_data_without_60.csv"
    summary_csv = result_dir / "data" / "vehicle_count_scan_summary.csv"
    if plot_csv.exists():
        return read_plot_csv(plot_csv)
    if filtered_plot_csv.exists():
        return read_plot_csv(filtered_plot_csv)
    if summary_csv.exists():
        return read_summary_csv(summary_csv)
    raise FileNotFoundError(
        f"No usable vehicle-count CSV exists under {result_dir / 'data'}."
    )


def read_plot_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vehicle_num = int(float(row["vehicle_num"]))
            achr = float(row.get("achr", 0.0))
            rows.append(
                {
                    "vehicle_num": vehicle_num,
                    "achr": achr,
                    "achr_percent": float(row.get("achr_percent", achr * 100.0)),
                    "average_delay_ms": float(row.get("average_delay_ms", 0.0)),
                }
            )
    return rows


def read_summary_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vehicle_num = int(float(row["vehicle_num"]))
            achr = float(row.get("achr_mean", 0.0))
            rows.append(
                {
                    "vehicle_num": vehicle_num,
                    "achr": achr,
                    "achr_percent": achr * 100.0,
                    "average_delay_ms": float(row.get("average_delay_ms_mean", 0.0)),
                }
            )
    return rows


def write_filtered_csv(output_path: Path, rows: List[Dict[str, float]]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["vehicle_num", "achr", "achr_percent", "average_delay_ms"])
        writer.writeheader()
        for row in rows:
            achr = float(row["achr"])
            writer.writerow(
                {
                    "vehicle_num": int(row["vehicle_num"]),
                    "achr": achr,
                    "achr_percent": float(row.get("achr_percent", achr * 100.0)),
                    "average_delay_ms": float(row["average_delay_ms"]),
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
    left, right, top, bottom = 90, 100, 36, 82
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
            f'font-family="Times New Roman, Arial" font-size="22" font-weight="700" fill="{blue}">{int(round(left_value))}</text>'
        )
        right_value = delay_min + (delay_max - delay_min) * tick / 5.0
        lines.append(
            f'<text x="{left + plot_w + 10}" y="{y + 4:.2f}" text-anchor="start" '
            f'font-family="Times New Roman, Arial" font-size="22" font-weight="700" fill="{pink}">{int(round(right_value))}</text>'
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
            f'font-family="Times New Roman, Arial" font-size="24" font-weight="700">{vehicle_count}</text>'
        )

    lines.extend(
        [
            (
                f'<text x="{left + plot_w / 2:.1f}" y="{height - 25}" text-anchor="middle" '
                f'font-family="Times New Roman, Arial" font-size="34" font-weight="700" fill="#111">Number of vehicles</text>'
            ),
            (
                f'<text x="28" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
                f'transform="rotate(-90 28 {top + plot_h / 2:.1f})" '
                f'font-family="Times New Roman, Arial" font-size="34" font-weight="700" fill="{blue}">Average Cache Hit Ratio (%)</text>'
            ),
            (
                f'<text x="{width - 28}" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
                f'transform="rotate(90 {width - 28} {top + plot_h / 2:.1f})" '
                f'font-family="Times New Roman, Arial" font-size="34" font-weight="700" fill="{pink}">Average delay (ms)</text>'
            ),
            f'<rect x="{left + plot_w - 180}" y="{top + 12}" width="14" height="14" fill="{blue}"/>',
            f'<text x="{left + plot_w - 160}" y="{top + 30}" font-family="Times New Roman, Arial" font-size="23" font-weight="700">ACHR</text>',
            f'<rect x="{left + plot_w - 105}" y="{top + 12}" width="14" height="14" fill="{pink}"/>',
            f'<text x="{left + plot_w - 85}" y="{top + 30}" font-family="Times New Roman, Arial" font-size="23" font-weight="700">Delay</text>',
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


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def create_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base_dir / f"vehicle_count_scan_replot_{timestamp}"
    counter = 1
    while path.exists():
        path = base_dir / f"vehicle_count_scan_replot_{timestamp}_{counter:02d}"
        counter += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def escape_xml(text) -> str:
    return html.escape(str(text), quote=False)


if __name__ == "__main__":
    main()
