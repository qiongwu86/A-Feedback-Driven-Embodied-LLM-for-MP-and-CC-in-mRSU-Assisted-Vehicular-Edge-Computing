from typing import Iterable, List, Sequence, Tuple

from caching.cache_repair import CacheRepair


class CacheMapper:
    """Deterministically map a priority list into mRSU and fRSU caches."""

    def __init__(self, repair: CacheRepair):
        self.repair = repair

    def map_priority_list(
        self,
        content_priority_list: Iterable[int],
        mrsu_capacity: int,
        frsu_capacity: int,
        local_fallback: Sequence[int] = None,
    ) -> Tuple[List[int], List[int]]:
        cleaned = self.repair.clean_priority_list(content_priority_list)
        mrsu_cache = cleaned[:mrsu_capacity]
        frsu_cache = cleaned[mrsu_capacity : mrsu_capacity + frsu_capacity]
        return self.repair.repair(
            mrsu_cache=mrsu_cache,
            frsu_cache=frsu_cache,
            mrsu_capacity=mrsu_capacity,
            frsu_capacity=frsu_capacity,
            local_fallback=local_fallback,
        )

