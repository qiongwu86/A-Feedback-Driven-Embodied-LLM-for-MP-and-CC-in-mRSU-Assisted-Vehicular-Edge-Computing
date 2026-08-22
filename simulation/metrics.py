from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Set


@dataclass
class RoundMetrics:
    hit_count: int
    request_count: int
    chr: float
    mrsu_hit_count: int
    frsu_hit_count: int
    mbs_miss_count: int
    not_covered_count: int
    not_cached_count: int
    local_rsu_chr: float = 0.0
    covered_request_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_cache_hit_ratio(
    vehicle_requests: Dict[int, List[int]],
    mrsu_covered: Iterable[int],
    frsu_covered: Iterable[int],
    mrsu_cache: Iterable[int],
    frsu_cache: Iterable[int],
) -> RoundMetrics:
    """Evaluate one-round CHR with mRSU priority over fRSU."""

    mrsu_covered_set: Set[int] = set(mrsu_covered)
    frsu_covered_set: Set[int] = set(frsu_covered)
    mrsu_cache_set: Set[int] = set(int(x) for x in mrsu_cache)
    frsu_cache_set: Set[int] = set(int(x) for x in frsu_cache)

    hit_count = 0
    request_count = 0
    mrsu_hits = 0
    frsu_hits = 0
    mbs_misses = 0
    not_covered = 0
    not_cached = 0

    for vehicle_id, requests in vehicle_requests.items():
        in_mrsu = vehicle_id in mrsu_covered_set
        in_frsu = vehicle_id in frsu_covered_set
        for content_id in requests:
            request_count += 1
            content_id = int(content_id)
            if in_mrsu and content_id in mrsu_cache_set:
                hit_count += 1
                mrsu_hits += 1
            elif in_frsu and content_id in frsu_cache_set:
                hit_count += 1
                frsu_hits += 1
            else:
                mbs_misses += 1
                if not in_mrsu and not in_frsu:
                    not_covered += 1
                else:
                    not_cached += 1

    chr_value = hit_count / request_count if request_count else 0.0
    covered_request_count = hit_count + not_cached
    local_chr_value = local_rsu_chr_from_counts(hit_count, not_cached)
    return RoundMetrics(
        hit_count=hit_count,
        request_count=request_count,
        chr=chr_value,
        mrsu_hit_count=mrsu_hits,
        frsu_hit_count=frsu_hits,
        mbs_miss_count=mbs_misses,
        not_covered_count=not_covered,
        not_cached_count=not_cached,
        local_rsu_chr=local_chr_value,
        covered_request_count=covered_request_count,
    )


def summarize_metrics(round_metrics: List[RoundMetrics]) -> dict:
    if not round_metrics:
        return {
            "achr": 0.0,
            "local_rsu_achr": 0.0,
            "mrsu_hit_count": 0,
            "frsu_hit_count": 0,
            "mbs_miss_count": 0,
            "not_covered_count": 0,
            "not_cached_count": 0,
            "covered_request_count": 0,
        }

    total_hit_count = sum(item.hit_count for item in round_metrics)
    total_not_cached_count = sum(item.not_cached_count for item in round_metrics)
    return {
        "achr": sum(item.chr for item in round_metrics) / len(round_metrics),
        "local_rsu_achr": local_rsu_chr_from_counts(total_hit_count, total_not_cached_count),
        "mrsu_hit_count": sum(item.mrsu_hit_count for item in round_metrics),
        "frsu_hit_count": sum(item.frsu_hit_count for item in round_metrics),
        "mbs_miss_count": sum(item.mbs_miss_count for item in round_metrics),
        "not_covered_count": sum(item.not_covered_count for item in round_metrics),
        "not_cached_count": sum(item.not_cached_count for item in round_metrics),
        "covered_request_count": sum(item.covered_request_count for item in round_metrics),
    }


def count_requests_by_vehicle(vehicle_requests: Dict[int, List[int]]) -> Dict[int, Counter]:
    return {
        vehicle_id: Counter(int(content_id) for content_id in requests)
        for vehicle_id, requests in vehicle_requests.items()
    }


def local_rsu_chr_from_counts(hit_count: int | float, not_cached_count: int | float) -> float:
    denominator = float(hit_count) + float(not_cached_count)
    return float(hit_count) / denominator if denominator > 0.0 else 0.0
