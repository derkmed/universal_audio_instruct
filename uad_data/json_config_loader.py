"""Captures the expected JSON fields when reading a Universal Dataset Config from JSON.

Usage:
```
python3 -m uad_data.json_config_loader --config_path=JSON_CONFIG.json
```

Data should appear similar to as follows:
```
{
    name: "UNIVERSAL_NAME"
    datasets: [
        {
            name: "INTERNAL_NAME1",
            tasks: ["task1", "task2"],
            splits: ["split1", "split2"]
        },
        {
            name: "INTERNAL_NAME2",
            tasks: ["task1"],
            splits: ["split1"]
        },
    ]
}
```
"""
import argparse
import copy
import datasets
import json
import logging

from . import filters as filters_lib
from . import internal_datasets
from .collection import UadCollection
from .internal_dataset import InternalDataset
from .tasks import Task


logger = logging.getLogger(__name__)


class InternalDatasetJsonConfig:
    """Dataclass to map JSON-specifiable fields to an InternalDataset."""

    # Expected JSON fields apper below along with their type.
    def __init__(self,
                 *,
                 name: str,
                 tasks: list[str] = [],
                 splits: list[str] = []
                 ):
        self.name = name
        if self.name not in internal_datasets.DATASETS_DIRECTORY:
            # Datasets must be listed in the DATASETS_DIRECTORY.
            raise ValueError(
                f'Unsupported dataset: {self.name} was specified.'
                'Try specifying an onboarded dataset.')

        complete_dataset = internal_datasets.DATASETS_DIRECTORY[self.name]

        self.tasks = [Task(t) for t in tasks]
        if not self.tasks:
            raise ValueError(f'Tasks unspecified for dataset: {self.name}.')

        self.splits = [datasets.Split(s) for s in splits]
        if not self.splits:
            logger.info(f'Splits unspecified for dataset: {self.name}.'
                        ' Defaulting to all splits.')
            # Copy, so this config never holds (or hands out) the registry's own list.
            self.splits = list(complete_dataset.get_splits())

    def toInternalDataset(self) -> InternalDataset:

        # All configurations are contextualized on the directory listing of this dataset.
        # Tasks and Splits must be a subset of those defined in this listing.
        complete_dataset = internal_datasets.DATASETS_DIRECTORY[self.name]

        # Verify that specified tasks are a subset of those available.
        invalid_tasks = [t for t in self.tasks if t not in complete_dataset.tasks]
        if invalid_tasks:
            raise ValueError(f'Task: {invalid_tasks} requested of {complete_dataset.name}, which '
                             f'only contains tasks: {complete_dataset.tasks}')
        # Verify that specified splits are a subset of those available.
        invalid_splits = [s for s in self.splits if s not in complete_dataset.get_splits()]
        if invalid_splits:
            raise ValueError(f'Splits: {invalid_splits} requested of {complete_dataset.name}, which '
                             f'only contains splits: {complete_dataset.get_splits()}')

        # Narrow a copy: the registry entry is process-wide, and narrowing it in place
        # would leak this config's tasks/splits into every config loaded after it.
        internal_dataset = copy.copy(complete_dataset)
        internal_dataset.set_tasks(list(self.tasks))
        internal_dataset.set_splits(list(self.splits))

        return internal_dataset


class UniversalJsonConfig:
    """Dataclass to map JSON-specifiable fields to a UadCollection.

    Expected JSON fields:
    * name: Name of the collection of data.
    * datasets: Which internal datasets to include in this collection.

    Optional JSON fields:
    * randomize_prompt_format: boolean field to indicate random selection of prompt/expected
      outputs. Setting to False defaults to inclusion of the cross-product of all prompt-output
      format pairings.
    * row_filter: name of a filter to apply to rows. One of: "all_pass" (default), "random".
    """

    # Expected JSON fields apper below along with their type.
    name: str
    internal_datasets: list[InternalDatasetJsonConfig]
    randomize_prompt_format: bool = False
    row_filter: filters_lib.RowFilter | None = None

    def __init__(self, *, filepath: str):
        with open(filepath) as f:
            data = json.loads(f.read())

            # Parse the name of this collection of data.
            if 'name' not in data.keys():
                raise ValueError('JSON configuration must contain "name" field.')
            self.name = data['name']

            # Parse selected dataset metadata (e.g. dataset name & split information).
            if 'datasets' not in data.keys():
                raise ValueError('JSON configuration must contain "datasets" field.')
            self.internal_datasets = [
                InternalDatasetJsonConfig(**d) for d in data['datasets']]

            # Parse the optional indication that a random prompt format should be selected.
            if 'randomize_prompt_format' in data.keys():
                self.randomize_prompt_format = bool(data['randomize_prompt_format'])

            # The filter key used to be `sample_filter`. Refuse it rather than
            # ignore it, so an old config can't silently lose its filter.
            if 'sample_filter' in data.keys():
                raise ValueError(
                    'The run config key "sample_filter" is now "row_filter"; rename it.')

            # Parse the optional row filter.
            if 'row_filter' in data.keys():
                self.row_filter = filters_lib.from_config(data['row_filter'])

    def toCollection(self) -> UadCollection:
        return UadCollection(
            name=self.name,
            internal_datasets=[d.toInternalDataset() for d in self.internal_datasets],
            randomize_prompt_format=self.randomize_prompt_format,
            row_filter=self.row_filter,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='json_config_loader.py',
        description=(
            'Loads a JSON configuration of the UniversalAudioUnderstanding dataset.'
        ),
    )
    parser.add_argument('--config_path', help='JSON filepath configuration.',)
    args = vars(parser.parse_args())
    filepath = args['config_path']
    loaded_config = UniversalJsonConfig(filepath=filepath)

    # You can view one of the internal dataset configurations as follows:
    collection = loaded_config.toCollection()
    print(collection.internal_datasets[0])
