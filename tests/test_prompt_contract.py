"""Offline tests: every registered task can actually render a row.

Two ways a run dies on the first clip of a task, both of which used to reach a
real run before anything noticed:

  - the task has no prompt file, so `_get_prompt_templates` raises
    `PromptTemplateError` -- the one error a smoke run does not step over
    (#57: slurp_real registered `intent_detection` and `action_classification`,
    neither of which the Hub has a file for);
  - the task has a file whose templates read a name `Task.render_context` never
    supplies, so jinja2's default `Undefined` renders it as an empty string and
    the row is generated silently wrong (`intonation_detection.json` read
    `{{intonation}}` when the feature is `category`; `sentiment_analysis.json`
    read `{{sentiment}}` when it is `Sentiment`).

`prompts/` lives on the Hub, not in this repo, and this suite never touches the
network, so `tests/hub_prompts.json` stands in for it: per task, its file and
the placeholders its templates read. `python -m uad_data.check_hub_prompts`
re-checks that file against the real Hub, and `--write` refreshes it.

Runnable directly (`python tests/test_prompt_contract.py`) or under pytest.
"""
import json
import os

from uad_data.internal_datasets import DATASETS_DIRECTORY
from uad_data.tasks import Task

CONTRACT_PATH = os.path.join(os.path.dirname(__file__), "hub_prompts.json")

# Tasks with a prompt file that no internal dataset registers. Both label
# something `classification` now carries instead (ADR-0008), so their files are
# dead weight -- but the members have to stay in `Task` while the files are on
# the Hub, because `_get_prompt_templates` builds a `PromptFilepath` for every
# file it globs and `Task(...)` would raise on an unknown one, taking every
# other task's lookup down with it.
UNREGISTERED_TASKS = {"sentiment_analysis", "intonation_detection"}


def _contract() -> dict:
    with open(CONTRACT_PATH, encoding="utf-8") as f:
        return json.load(f)


def _registered_tasks() -> set[Task]:
    return {task for dataset in DATASETS_DIRECTORY.values() for task in dataset.tasks}


def _render_context_keys(task: Task) -> set[str]:
    """The names `Task.render_context` puts in a row's template context.

    asr_timestamp_search renders one utterance, so its context is the utterance's
    fields, not the record's `transcriptions` list.
    """
    features = task.features
    if task == Task.ASR_TIMESTAMP_SEARCH:
        return set(features["transcriptions"][0])
    return set(features)


def test_every_registered_task_has_a_prompt_file() -> None:
    contract = _contract()
    missing = sorted(
        task.value for task in _registered_tasks() if task.value not in contract)
    assert not missing, (
        f"{missing} are registered in uad_data/internal_datasets.py but have no "
        f"prompt file, so _get_prompt_templates raises PromptTemplateError on the "
        f"first clip. Author prompts/<task>.json on the Hub and re-run "
        f"`python -m uad_data.check_hub_prompts --write`, or drop the task.")
    print(f"PASS: all {len(_registered_tasks())} registered tasks have a prompt file.")


def test_every_prompt_file_renders_from_its_task_features() -> None:
    for task_value, entry in sorted(_contract().items()):
        keys = _render_context_keys(Task(task_value))
        unrenderable = sorted(set(entry["placeholders"]) - keys)
        assert not unrenderable, (
            f"{entry['file']} reads {unrenderable}, which Task.{Task(task_value).name}"
            f".render_context never supplies (it gives {sorted(keys)}). jinja2 renders "
            f"those as empty strings instead of failing, so every row of the task is "
            f"silently wrong.")
    print("PASS: every prompt file's placeholders come from its task's features.")


def test_prompt_files_without_a_dataset_are_the_known_two() -> None:
    """A new orphan means someone dropped a registration without its prompt file."""
    registered = {task.value for task in _registered_tasks()}
    orphans = set(_contract()) - registered
    assert orphans == UNREGISTERED_TASKS, (
        f"prompt files with no internal dataset: {sorted(orphans)}, expected "
        f"{sorted(UNREGISTERED_TASKS)}. Delete the file from the Hub (and its Task "
        f"member) if the task is gone for good, or add it to UNREGISTERED_TASKS.")
    print(f"PASS: the only prompt files without a dataset are {sorted(orphans)}.")


def test_slurp_real_does_its_intent_labels_as_classification() -> None:
    """#57: the two prompt-less tasks are gone, and the label survives as `category`."""
    tasks = set(DATASETS_DIRECTORY["slurp_real"].tasks)
    assert Task.CLASSIFICATION in tasks, tasks
    assert not {t.value for t in tasks} & {"intent_detection", "action_classification"}, tasks
    print("PASS: slurp_real's intent labels come through classification.")


if __name__ == "__main__":
    test_every_registered_task_has_a_prompt_file()
    test_every_prompt_file_renders_from_its_task_features()
    test_prompt_files_without_a_dataset_are_the_known_two()
    test_slurp_real_does_its_intent_labels_as_classification()
