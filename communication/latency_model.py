from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, List, Sequence

from simulation.coverage import CoverageModel, CoverageSnapshot


@dataclass
class LatencyModelConfig:
    """CV2X-style link-rate and content-delay parameters.

    The rate model follows the legacy project structure: path loss plus noise and
    load-dependent interference are converted to Mbps with a Shannon-like formula.
    Content delay is then content_size_kbit / rate_mbps, which yields milliseconds.
    """

    content_size_kbit: float = 800.0
    bandwidth_mhz: float = 10.0
    carrier_frequency_ghz: float = 5.9
    rsu_tx_power_dbm: float = 23.0
    mbs_tx_power_dbm: float = 43.0
    rsu_distance_loss_db_per_decade: float = 16.0
    noise_figure_db: float = 9.0
    spectral_efficiency: float = 0.25
    min_distance_m: float = 1.0
    min_rate_mbps: float = 0.1
    cloud_backhaul_rate_mbps: float = 80.0
    cloud_extra_latency_ms: float = 20.0
    local_processing_latency_ms: float = 2.0
    cloud_processing_latency_ms: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RequestLatencySummary:
    request_count: int
    total_delay_ms: float
    average_delay_ms: float
    excluded_not_covered_request_count: int
    mrsu_request_count: int
    frsu_request_count: int
    mbs_request_count: int
    mrsu_average_delay_ms: float
    frsu_average_delay_ms: float
    mbs_average_delay_ms: float
    mrsu_average_rate_mbps: float
    frsu_average_rate_mbps: float
    mbs_average_rate_mbps: float
    average_service_distance_m: float
    mrsu_average_distance_m: float
    frsu_average_distance_m: float
    mbs_average_distance_m: float

    def to_dict(self) -> dict:
        return asdict(self)


class CV2XLatencyModel:
    """Estimate per-request transmission latency for mRSU/fRSU/cache-miss service."""

    def __init__(self, config: LatencyModelConfig | None = None):
        self.config = config or LatencyModelConfig()
        self.bandwidth_hz = max(float(self.config.bandwidth_mhz), 1e-9) * 1e6
        self.noise_power_dbm = (
            -174.0
            + 10.0 * math.log10(max(self.bandwidth_hz, 1e-9))
            + float(self.config.noise_figure_db)
        )

    def evaluate_round(
        self,
        vehicle_requests: Dict[int, List[int]],
        vehicle_positions: Sequence[float],
        coverage: CoverageSnapshot,
        mrsu_position: float,
        frsu_position: float,
        road_length: float,
        mrsu_radius: float,
        frsu_radius: float,
        mrsu_cache: Iterable[int],
        frsu_cache: Iterable[int],
    ) -> RequestLatencySummary:
        mrsu_covered = set(int(x) for x in coverage.mrsu_covered)
        frsu_covered = set(int(x) for x in coverage.frsu_covered)
        mrsu_cache_set = set(int(x) for x in mrsu_cache)
        frsu_cache_set = set(int(x) for x in frsu_cache)
        load = max(len(mrsu_covered.union(frsu_covered)), 1)

        totals = {
            "mrsu": {"delay": 0.0, "count": 0, "rate": 0.0, "distance": 0.0},
            "frsu": {"delay": 0.0, "count": 0, "rate": 0.0, "distance": 0.0},
            "mbs": {"delay": 0.0, "count": 0, "rate": 0.0, "distance": 0.0},
        }
        excluded_not_covered = 0

        for vehicle_id, requests in vehicle_requests.items():
            vehicle_id = int(vehicle_id)
            position = self._vehicle_position(vehicle_positions, vehicle_id)
            in_mrsu = vehicle_id in mrsu_covered
            in_frsu = vehicle_id in frsu_covered
            for content_id in requests:
                content_id = int(content_id)
                if not in_mrsu and not in_frsu:
                    excluded_not_covered += 1
                    continue
                if in_mrsu and content_id in mrsu_cache_set:
                    distance = self._covered_distance(
                        position,
                        mrsu_position,
                        road_length,
                        mrsu_radius,
                    )
                    rate = self.rsu_rate_mbps(distance, load)
                    delay = self.content_delay_ms(rate) + self.config.local_processing_latency_ms
                    bucket = "mrsu"
                elif in_frsu and content_id in frsu_cache_set:
                    distance = self._covered_distance(
                        position,
                        frsu_position,
                        road_length,
                        frsu_radius,
                    )
                    rate = self.rsu_rate_mbps(distance, load)
                    delay = self.content_delay_ms(rate) + self.config.local_processing_latency_ms
                    bucket = "frsu"
                else:
                    distance = max(
                        self.config.min_distance_m,
                        CoverageModel.circular_distance(position, frsu_position, road_length),
                    )
                    rate = self.mbs_rate_mbps(distance, load)
                    delay = (
                        self.content_delay_ms(rate)
                        + self.content_delay_ms(self.config.cloud_backhaul_rate_mbps)
                        + self.config.cloud_extra_latency_ms
                        + self.config.cloud_processing_latency_ms
                    )
                    bucket = "mbs"

                totals[bucket]["delay"] += float(delay)
                totals[bucket]["rate"] += float(rate)
                totals[bucket]["distance"] += float(distance)
                totals[bucket]["count"] += 1

        request_count = sum(int(item["count"]) for item in totals.values())
        total_delay = sum(float(item["delay"]) for item in totals.values())
        total_distance = sum(float(item["distance"]) for item in totals.values())
        return RequestLatencySummary(
            request_count=int(request_count),
            total_delay_ms=float(total_delay),
            average_delay_ms=_safe_average(total_delay, request_count),
            excluded_not_covered_request_count=int(excluded_not_covered),
            mrsu_request_count=int(totals["mrsu"]["count"]),
            frsu_request_count=int(totals["frsu"]["count"]),
            mbs_request_count=int(totals["mbs"]["count"]),
            mrsu_average_delay_ms=_safe_average(totals["mrsu"]["delay"], totals["mrsu"]["count"]),
            frsu_average_delay_ms=_safe_average(totals["frsu"]["delay"], totals["frsu"]["count"]),
            mbs_average_delay_ms=_safe_average(totals["mbs"]["delay"], totals["mbs"]["count"]),
            mrsu_average_rate_mbps=_safe_average(totals["mrsu"]["rate"], totals["mrsu"]["count"]),
            frsu_average_rate_mbps=_safe_average(totals["frsu"]["rate"], totals["frsu"]["count"]),
            mbs_average_rate_mbps=_safe_average(totals["mbs"]["rate"], totals["mbs"]["count"]),
            average_service_distance_m=_safe_average(total_distance, request_count),
            mrsu_average_distance_m=_safe_average(totals["mrsu"]["distance"], totals["mrsu"]["count"]),
            frsu_average_distance_m=_safe_average(totals["frsu"]["distance"], totals["frsu"]["count"]),
            mbs_average_distance_m=_safe_average(totals["mbs"]["distance"], totals["mbs"]["count"]),
        )

    def rsu_rate_mbps(self, distance_m: float, active_vehicle_count: int = 1) -> float:
        path_loss_db = self._local_rsu_path_loss(distance_m)
        return self._rate_from_path_loss(
            tx_power_dbm=self.config.rsu_tx_power_dbm,
            path_loss_db=path_loss_db,
            active_vehicle_count=active_vehicle_count,
        )

    def mbs_rate_mbps(self, distance_m: float, active_vehicle_count: int = 1) -> float:
        path_loss_db = self._macro_v2i_path_loss(distance_m)
        return self._rate_from_path_loss(
            tx_power_dbm=self.config.mbs_tx_power_dbm,
            path_loss_db=path_loss_db,
            active_vehicle_count=active_vehicle_count,
        )

    def content_delay_ms(self, rate_mbps: float) -> float:
        return float(self.config.content_size_kbit) / max(float(rate_mbps), float(self.config.min_rate_mbps))

    def _rate_from_path_loss(
        self,
        tx_power_dbm: float,
        path_loss_db: float,
        active_vehicle_count: int,
    ) -> float:
        interference_db = self._interference_db(active_vehicle_count)
        received_dbm = float(tx_power_dbm) - float(path_loss_db)
        snr_db = received_dbm - interference_db - self.noise_power_dbm
        snr_linear = max(0.0, 10.0 ** (snr_db / 10.0))
        rate = (
            self.bandwidth_hz
            * math.log2(1.0 + snr_linear)
            * float(self.config.spectral_efficiency)
            / 1e6
        )
        return max(float(self.config.min_rate_mbps), float(rate))

    def _local_rsu_path_loss(self, distance_m: float) -> float:
        distance = max(float(distance_m), float(self.config.min_distance_m))
        fc = max(float(self.config.carrier_frequency_ghz), 1e-9)
        distance_loss = float(self.config.rsu_distance_loss_db_per_decade) * math.log10(distance)
        return 36.85 + 18.9 * math.log10(fc) + distance_loss

    def _macro_v2i_path_loss(self, distance_m: float) -> float:
        distance_km = max(float(distance_m), float(self.config.min_distance_m)) / 1000.0
        return 128.1 + 37.6 * math.log10(max(distance_km, 1e-6))

    def _interference_db(self, active_vehicle_count: int) -> float:
        return math.log10(1.0 + max(int(active_vehicle_count), 0) / 20.0)

    def _covered_distance(
        self,
        vehicle_position: float,
        rsu_position: float,
        road_length: float,
        radius: float,
    ) -> float:
        distance = CoverageModel.circular_distance(vehicle_position, rsu_position, road_length)
        capped = min(float(distance), max(float(radius), float(self.config.min_distance_m)))
        return max(capped, float(self.config.min_distance_m))

    def _vehicle_position(self, positions: Sequence[float], vehicle_id: int) -> float:
        if 0 <= int(vehicle_id) < len(positions):
            return float(positions[int(vehicle_id)])
        return 0.0


def summarize_round_latencies(round_latencies: Iterable[dict]) -> dict:
    rows = [dict(row or {}) for row in round_latencies]
    total_delay = sum(float(row.get("total_delay_ms", 0.0)) for row in rows)
    request_count = sum(int(row.get("request_count", 0)) for row in rows)
    summary = {
        "latency_scope": "rsu_covered_requests_only",
        "latency_request_count": int(request_count),
        "total_delay_ms": float(total_delay),
        "average_delay_ms": _safe_average(total_delay, request_count),
        "excluded_not_covered_request_count": sum(
            int(row.get("excluded_not_covered_request_count", 0)) for row in rows
        ),
        "round_delay_ms": [float(row.get("average_delay_ms", 0.0)) for row in rows],
    }
    for prefix in ("mrsu", "frsu", "mbs"):
        count_key = f"{prefix}_request_count"
        delay_key = f"{prefix}_average_delay_ms"
        rate_key = f"{prefix}_average_rate_mbps"
        distance_key = f"{prefix}_average_distance_m"
        count = sum(int(row.get(count_key, 0)) for row in rows)
        delay_total = sum(
            float(row.get(delay_key, 0.0)) * int(row.get(count_key, 0))
            for row in rows
        )
        rate_total = sum(
            float(row.get(rate_key, 0.0)) * int(row.get(count_key, 0))
            for row in rows
        )
        distance_total = sum(
            float(row.get(distance_key, 0.0)) * int(row.get(count_key, 0))
            for row in rows
        )
        summary[count_key] = int(count)
        summary[delay_key] = _safe_average(delay_total, count)
        summary[rate_key] = _safe_average(rate_total, count)
        summary[distance_key] = _safe_average(distance_total, count)
    service_distance_total = sum(
        float(row.get("average_service_distance_m", 0.0)) * int(row.get("request_count", 0))
        for row in rows
    )
    summary["average_service_distance_m"] = _safe_average(service_distance_total, request_count)
    return summary


def _safe_average(total: float, count: int) -> float:
    return float(total) / max(int(count), 1)
