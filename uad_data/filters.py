"""Per-row inclusion filters applied during row generation.

A `RowFilter.include_row(row)` decides whether an expanded row is kept.
Filters are selected by name in a run config's optional `row_filter` field
(`from_config`); the default is `AllPassFilter` (keep everything).
"""

import abc
import random

from .row import Row

class RowFilter(abc.ABC):
    """Use this filter to determine whether a row should be included in a dataset or not."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def include_row(self, row: Row) -> bool:
        raise NotImplementedError

class AllPassFilter(RowFilter):
    """All rows are filtered in."""

    def include_row(self, row: Row) -> bool:
        return True


class RandomFilter(RowFilter):
    """All rows are selected randomly."""

    def __init__(self, random_seed: int = 42):
        super().__init__()
        random.seed(random_seed)

    def include_row(self, row: Row) -> bool:
        if random.random() > 0.5:
            return True
        else:
            return False


FILTER_REGISTRY: dict[str, type[RowFilter]] = {
    'all_pass': AllPassFilter,
    'random': RandomFilter,
}


def from_config(name: str, **kwargs) -> RowFilter:
    if name not in FILTER_REGISTRY:
        raise ValueError(
            f'Unknown row_filter "{name}". Must be one of: {list(FILTER_REGISTRY)}')
    return FILTER_REGISTRY[name](**kwargs)
