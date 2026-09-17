"""A resolved selection of internal datasets to generate rows for.

This replaces the `UniversalAudioUnderstandingConfig` builder-config that used to
live in the HF loading script. It is a plain data holder -- no `datasets`
GeneratorBasedBuilder machinery -- describing which internal datasets, tasks and
splits a generation/eval run should cover.
"""
from typing import List

from . import filters as filters_lib
from .internal_dataset import InternalDataset


class UadCollection:

    def __init__(
        self,
        name: str = 'UniversalAudioDataset',
        internal_datasets: InternalDataset | list[InternalDataset] | None = None,
        randomize_prompt_format: bool = False,
        row_filter: filters_lib.RowFilter | None = None,
    ):
        self._internal_datasets = internal_datasets if internal_datasets is not None else []
        self.name = name
        self.randomize_prompt_format = randomize_prompt_format
        self.row_filter = row_filter or filters_lib.AllPassFilter()

    @property
    def internal_datasets(self) -> list[InternalDataset]:
        if not isinstance(self._internal_datasets, List):
            self._internal_datasets = [self._internal_datasets]
        return self._internal_datasets

    @property
    def splits(self):
        return list(set(
            split for d in self.internal_datasets for split in d.get_splits()))

    @property
    def tasks(self):
        return list(set(
            task for d in self.internal_datasets for task in d.tasks))

    def get_datasets_with_splits(self, split) -> list[InternalDataset]:
        return [d for d in self.internal_datasets if split in d.get_splits()]

    def is_random_prompt_format_selection(self) -> bool:
        return self.randomize_prompt_format
