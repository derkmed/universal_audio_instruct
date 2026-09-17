"""Offline test: loading a run config must not narrow the shared dataset registry.

`InternalDatasetJsonConfig.toInternalDataset` used to call `set_tasks` /
`set_splits` directly on the `internal_datasets.DATASETS_DIRECTORY` entry, so a
config that narrowed Clotho to `caption` made a later config in the same process
fail with "Task ... requested of Clotho, which only contains tasks: [caption]",
and made unspecified splits default to the earlier config's splits.

Also checks the `row_filter` key, and that the old `sample_filter` key raises.

Runnable directly (`python tests/test_json_config_loader.py`) or under pytest.
Only requires `datasets`, `jinja2`, `huggingface_hub` -- not the heavy eval deps.
"""
import json
import os
import sys
import tempfile

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datasets  # noqa: E402

from uad_data import filters, internal_datasets  # noqa: E402
from uad_data.json_config_loader import UniversalJsonConfig  # noqa: E402
from uad_data.tasks import Task  # noqa: E402

# Narrows Clotho to one task and one split, like the Hub's complete-1.json.
NARROW_CONFIG = {
    "name": "Clotho Caption Only",
    "datasets": [{"name": "Clotho", "splits": ["test"], "tasks": ["caption"]}],
}

# Asks for a task the first config left out, with splits unspecified.
COMMONSENSE_CONFIG = {
    "name": "Clotho Commonsense",
    "datasets": [{"name": "Clotho", "tasks": ["commonsense"]}],
}


def _write_config(root: str, filename: str, config: dict) -> str:
    path = os.path.join(root, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f)
    return path


def _load_clotho(config_path: str):
    collection = UniversalJsonConfig(filepath=config_path).toCollection()
    assert len(collection.internal_datasets) == 1, collection.internal_datasets
    return collection.internal_datasets[0]


def test_sequential_configs_do_not_mutate_registry() -> None:
    registry_clotho = internal_datasets.DATASETS_DIRECTORY["Clotho"]
    registry_tasks = list(registry_clotho.tasks)
    registry_splits = list(registry_clotho.get_splits())
    assert Task.COMMONSENSE in registry_tasks, registry_tasks
    assert len(registry_splits) > 1, registry_splits

    with tempfile.TemporaryDirectory() as root:
        narrow_path = _write_config(root, "narrow.json", NARROW_CONFIG)
        commonsense_path = _write_config(root, "commonsense.json", COMMONSENSE_CONFIG)

        narrow = _load_clotho(narrow_path)
        # Used to raise ValueError: the registry had been narrowed to [caption].
        commonsense = _load_clotho(commonsense_path)

    # Each load gets its own narrowed copy, never the registry object itself.
    assert narrow is not registry_clotho
    assert commonsense is not registry_clotho
    assert narrow is not commonsense

    assert narrow.tasks == [Task.CAPTION], narrow.tasks
    assert narrow.get_splits() == [datasets.Split.TEST], narrow.get_splits()
    assert commonsense.tasks == [Task.COMMONSENSE], commonsense.tasks
    # Unspecified splits default to the full registry listing, not the first config's.
    assert commonsense.get_splits() == registry_splits, commonsense.get_splits()

    # The registry itself is untouched.
    assert registry_clotho.tasks == registry_tasks, registry_clotho.tasks
    assert registry_clotho.get_splits() == registry_splits, registry_clotho.get_splits()

    print("PASS: sequential config loads narrowed copies and left the Clotho registry entry intact.")


def test_row_filter_key_is_read() -> None:
    config = {**COMMONSENSE_CONFIG, "row_filter": "random"}
    with tempfile.TemporaryDirectory() as root:
        path = _write_config(root, "row_filter.json", config)
        collection = UniversalJsonConfig(filepath=path).toCollection()
    assert isinstance(collection.row_filter, filters.RandomFilter), collection.row_filter
    print("PASS: row_filter selects the named filter.")


def test_sample_filter_key_is_rejected() -> None:
    # The key was renamed; dropping an old filter silently would change a run.
    config = {**COMMONSENSE_CONFIG, "sample_filter": "random"}
    with tempfile.TemporaryDirectory() as root:
        path = _write_config(root, "sample_filter.json", config)
        try:
            UniversalJsonConfig(filepath=path)
        except ValueError as e:
            assert "row_filter" in str(e), e
        else:
            raise AssertionError("a config with sample_filter should raise ValueError")
    print("PASS: sample_filter raises a ValueError naming row_filter.")


if __name__ == "__main__":
    test_sequential_configs_do_not_mutate_registry()
    test_row_filter_key_is_read()
    test_sample_filter_key_is_rejected()
