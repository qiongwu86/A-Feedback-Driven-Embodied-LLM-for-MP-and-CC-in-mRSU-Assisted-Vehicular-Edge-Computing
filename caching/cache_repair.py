from typing import Iterable, List, Sequence, Tuple


class CacheRepair:
    """Repair cache outputs to satisfy content and capacity constraints.

    Duplicates are removed inside each single RSU cache only. mRSU and fRSU are
    allowed to cache the same content because their service regions may both
    strongly demand it.
    """

    def __init__(self, valid_content_ids: Iterable[int], fallback_top_contents: Sequence[int]):
        self.valid_content_ids = set(int(x) for x in valid_content_ids)
        self.fallback_top_contents = [int(x) for x in fallback_top_contents]

    def clean_priority_list(self, content_priority_list: Iterable[int]) -> List[int]:
        seen = set()
        cleaned: List[int] = []
        for item in content_priority_list or []:
            try:
                content_id = int(item)
            except (TypeError, ValueError):
                continue
            if content_id not in self.valid_content_ids or content_id in seen:
                continue
            seen.add(content_id)
            cleaned.append(content_id)
        return cleaned

    def repair(
        self,
        mrsu_cache: Iterable[int],
        frsu_cache: Iterable[int],
        mrsu_capacity: int,
        frsu_capacity: int,
        local_fallback: Sequence[int] = None,
    ) -> Tuple[List[int], List[int]]:
        local_fallback = [int(x) for x in (local_fallback or [])]
        fallback = local_fallback + self.fallback_top_contents

        repaired_mrsu = self._fill_single_cache(mrsu_cache, mrsu_capacity, fallback)
        repaired_frsu = self._fill_single_cache(frsu_cache, frsu_capacity, fallback)
        return repaired_mrsu, repaired_frsu

    def _fill_single_cache(
        self,
        cache_items: Iterable[int],
        capacity: int,
        fallback: Sequence[int],
    ) -> List[int]:
        repaired = self.clean_priority_list(cache_items)[:capacity]
        seen = set(repaired)
        for item in fallback:
            if len(repaired) >= capacity:
                break
            try:
                content_id = int(item)
            except (TypeError, ValueError):
                continue
            if content_id not in self.valid_content_ids or content_id in seen:
                continue
            repaired.append(content_id)
            seen.add(content_id)
        return repaired[:capacity]
