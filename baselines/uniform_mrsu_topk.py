from typing import List, Tuple

import numpy as np


def uniform_move(position: float, speed: float, dt: float, road_length: float) -> float:
    return float((position + speed * dt) % road_length)


def uniform_topk_cache(
    global_top_contents: List[int],
    mrsu_capacity: int,
    frsu_capacity: int,
) -> Tuple[List[int], List[int]]:
    mrsu_cache = [int(x) for x in global_top_contents[:mrsu_capacity]]
    frsu_cache = [
        int(x)
        for x in global_top_contents[mrsu_capacity : mrsu_capacity + frsu_capacity]
    ]
    return mrsu_cache, frsu_cache


def uniform_random_cache(
    valid_content_ids: List[int],
    mrsu_capacity: int,
    frsu_capacity: int,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int]]:
    total = min(len(valid_content_ids), mrsu_capacity + frsu_capacity)
    selected = rng.choice(valid_content_ids, size=total, replace=False).tolist()
    selected = [int(x) for x in selected]
    return selected[:mrsu_capacity], selected[mrsu_capacity : mrsu_capacity + frsu_capacity]

