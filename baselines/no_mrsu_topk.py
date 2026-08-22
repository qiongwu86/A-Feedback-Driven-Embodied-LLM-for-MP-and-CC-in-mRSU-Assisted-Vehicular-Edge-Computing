from typing import List, Tuple


def no_mrsu_frsu_topk_cache(global_top_contents: List[int], frsu_capacity: int) -> Tuple[List[int], List[int]]:
    """No mRSU baseline: only fRSU caches global Top-K contents."""

    return [], [int(x) for x in global_top_contents[:frsu_capacity]]

