from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple


def topk_from_counter(counter: Counter, k: int, exclude: Iterable[int] = None) -> List[int]:
    excluded = set(int(x) for x in (exclude or []))
    items = []
    for content_id, _ in counter.most_common():
        content_id = int(content_id)
        if content_id in excluded:
            continue
        items.append(content_id)
        if len(items) >= k:
            break
    return items


def greedy_cooperative_cache(
    vehicle_requests: Dict[int, List[int]],
    mrsu_covered: List[int],
    frsu_covered: List[int],
    mrsu_capacity: int,
    frsu_capacity: int,
    global_top_contents: List[int],
) -> Tuple[List[int], List[int]]:
    """Greedily maximize predicted hit count with mRSU priority over fRSU.

    Each step evaluates every feasible placement action, i.e. caching one
    content at either mRSU or fRSU, and applies the action with the largest
    marginal gain. Duplicate contents across the two RSUs are allowed; duplicate
    contents inside one RSU are not.
    """

    mrsu_capacity = max(0, int(mrsu_capacity))
    frsu_capacity = max(0, int(frsu_capacity))
    if mrsu_capacity <= 0 and frsu_capacity <= 0:
        return [], []

    mrsu_set = set(int(vehicle_id) for vehicle_id in mrsu_covered)
    frsu_set = set(int(vehicle_id) for vehicle_id in frsu_covered)
    overlap_set = mrsu_set.intersection(frsu_set)
    mrsu_only_set = mrsu_set - frsu_set
    frsu_only_set = frsu_set - mrsu_set

    mrsu_only_demand = _demand_for_vehicles(vehicle_requests, mrsu_only_set)
    frsu_only_demand = _demand_for_vehicles(vehicle_requests, frsu_only_set)
    overlap_demand = _demand_for_vehicles(vehicle_requests, overlap_set)
    total_demand = Counter()
    total_demand.update(mrsu_only_demand)
    total_demand.update(frsu_only_demand)
    total_demand.update(overlap_demand)

    candidate_contents = _candidate_contents(
        [mrsu_only_demand, frsu_only_demand, overlap_demand],
        global_top_contents,
    )

    mrsu_cache: List[int] = []
    frsu_cache: List[int] = []
    mrsu_cache_set = set()
    frsu_cache_set = set()

    while len(mrsu_cache) < mrsu_capacity or len(frsu_cache) < frsu_capacity:
        best_action = None
        best_score = (0, 0, 0, 0)

        for content_id in candidate_contents:
            content_id = int(content_id)
            total_value = int(total_demand.get(content_id, 0))
            if len(mrsu_cache) < mrsu_capacity and content_id not in mrsu_cache_set:
                gain = int(mrsu_only_demand.get(content_id, 0))
                if content_id not in frsu_cache_set:
                    gain += int(overlap_demand.get(content_id, 0))
                score = (gain, total_value, 1, -content_id)
                if score > best_score:
                    best_score = score
                    best_action = ("mrsu", content_id)

            if len(frsu_cache) < frsu_capacity and content_id not in frsu_cache_set:
                gain = int(frsu_only_demand.get(content_id, 0))
                if content_id not in mrsu_cache_set:
                    gain += int(overlap_demand.get(content_id, 0))
                score = (gain, total_value, 0, -content_id)
                if score > best_score:
                    best_score = score
                    best_action = ("frsu", content_id)

        if best_action is None or best_score[0] <= 0:
            break

        rsu, selected_content = best_action
        if rsu == "mrsu":
            mrsu_cache.append(selected_content)
            mrsu_cache_set.add(selected_content)
        else:
            frsu_cache.append(selected_content)
            frsu_cache_set.add(selected_content)

    fallback_items = list(global_top_contents) + [
        content_id
        for content_id, _ in total_demand.most_common()
    ]
    _fill_cache(mrsu_cache, mrsu_capacity, fallback_items)
    _fill_cache(frsu_cache, frsu_capacity, fallback_items)

    return mrsu_cache[:mrsu_capacity], frsu_cache[:frsu_capacity]


def _demand_for_vehicles(
    vehicle_requests: Dict[int, List[int]],
    vehicle_ids: Iterable[int],
) -> Counter:
    demand = Counter()
    for vehicle_id in vehicle_ids:
        demand.update(int(content_id) for content_id in vehicle_requests.get(int(vehicle_id), []))
    return demand


def _candidate_contents(
    demand_counters: Sequence[Counter],
    global_top_contents: Iterable[int],
) -> List[int]:
    seen = set()
    result: List[int] = []
    combined = Counter()
    for counter in demand_counters:
        combined.update(counter)
    for content_id, _ in combined.most_common():
        content_id = int(content_id)
        if content_id in seen:
            continue
        seen.add(content_id)
        result.append(content_id)
    for content_id in global_top_contents:
        content_id = int(content_id)
        if content_id in seen:
            continue
        seen.add(content_id)
        result.append(content_id)
    return result


def _fill_cache(cache: List[int], capacity: int, fallback_items: Iterable[int]) -> None:
    seen = set(int(content_id) for content_id in cache)
    for content_id in fallback_items:
        content_id = int(content_id)
        if content_id in seen:
            continue
        cache.append(content_id)
        seen.add(content_id)
        if len(cache) >= int(capacity):
            break
