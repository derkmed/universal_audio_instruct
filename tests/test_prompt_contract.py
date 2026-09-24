"""Offline tests: every registered task can actually render a row.

Two ways a run dies on the first clip of a task, both of which used to reach a
real run before anything noticed:

  - the task has no prompt file, so `_get_prompt_templates` raises
    `PromptTemplateError` -- the one error a smoke run does not step over
    (#57: slurp_real registered `intent_detection` and `action_classification`,
    neither of which the Hub has a file for);
  - the task has a file whose templates read a name `Task.render_context` never
    supplies, so jinja2's default `Undefined` renders it as an empty string and
    the row is generated silently wrong.

`prompts/` lives on the Hub, not in this repo, and this suite never touches the
network, so `tests/hub_prompts.json` stands in for it: per task, its file and the
placeholders its templates read. It records what the Hub **has**, defects
included -- `python -m uad_data.check_hub_prompts` re-checks it against the real
Hub and `--write` refreshes it, so a contract recording a fix that had not landed
would leave this suite green while runs stayed broken, and would go red the next
time anyone refreshed it.

Runnable directly (`python tests/test_prompt_contract.py`) or under pytest.
"""
import json

from uad_data.check_hub_prompts import contract_path
from uad_data.internal_datasets import DATASETS_DIRECTORY
from uad_data.tasks import Task

# Tasks with a prompt file that no internal dataset registers. Both label
# something `classification` now carries instead (ADR-0008), so their files are
# dead weight -- but the members have to stay in `Task` while the files are on
# the Hub, because `_get_prompt_templates` builds a `PromptFilepath` for every
# file it globs and `Task(...)` would raise on an unknown one, taking every
# other task's lookup down with it.
UNREGISTERED_TASKS = {"sentiment_analysis", "intonation_detection"}

# Prompt files on the Hub that read a name their task's `render_context` never
# supplies, so every row they generate carries an empty string where the label
# belongs. Recorded rather than asserted away: the fix is a Hub edit, not a code
# change, and until it lands the contract has to say so out loud.
#
# Both are prompt files no internal dataset registers, so nothing renders them
# today -- which is why this went unnoticed. Registering either without fixing
# its file first would ship blank labels.
KNOWN_BAD_PLACEHOLDERS = {
    # `{{intonation}}`; Task.INTONATION_DETECTION.features gives `category`.
    "intonation_detection": {"intonation"},
    # `{{sentiment}}`; Task.SENTIMENT_ANALYSIS.features gives `Sentiment`.
    "sentiment_analysis": {"sentiment"},
}


def _contract() -> dict:
    with open(contract_path(), encoding="utf-8") as f:
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


def _unrenderable(task_value: str, entry: dict) -> set[str]:
    """Placeholders the task's render context has no value for."""
    return set(entry["placeholders"]) - _render_context_keys(Task(task_value))


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


def test_no_registered_task_renders_a_blank() -> None:
    """The invariant that matters: a task a dataset uses renders every placeholder."""
    for task in sorted(_registered_tasks()):
        entry = _contract()[task.value]
        unrenderable = sorted(_unrenderable(task.value, entry))
        assert not unrenderable, (
            f"{entry['file']} reads {unrenderable}, which Task.{task.name}.render_context "
            f"never supplies (it gives {sorted(_render_context_keys(task))}). jinja2 "
            f"renders those as empty strings instead of failing, so every row of the "
            f"task is silently wrong. Fix the file on the Hub, then re-run "
            f"`python -m uad_data.check_hub_prompts --write`.")
    print("PASS: no registered task's prompt file renders a blank.")


def test_the_only_unrenderable_placeholders_are_the_known_defects() -> None:
    """Catches a new broken file, and clears itself when a known one is fixed."""
    for task_value, entry in sorted(_contract().items()):
        unrenderable = _unrenderable(task_value, entry)
        recorded = KNOWN_BAD_PLACEHOLDERS.get(task_value, set())
        assert unrenderable == recorded, (
            f"{entry['file']} reads {sorted(unrenderable) or 'nothing'} that "
            f"Task.{Task(task_value).name}.render_context does not supply; "
            f"KNOWN_BAD_PLACEHOLDERS records {sorted(recorded) or 'nothing'}. "
            f"If the Hub file was just fixed, drop the entry; if it was just broken, "
            f"fix the file rather than recording it.")
    print(f"PASS: the only unrenderable placeholders are {sorted(KNOWN_BAD_PLACEHOLDERS)}.")


def test_known_bad_placeholders_are_all_unregistered_tasks() -> None:
    """A dataset must not be pointed at a prompt file known to render a blank."""
    registered = {task.value for task in _registered_tasks()}
    shipped = sorted(set(KNOWN_BAD_PLACEHOLDERS) & registered)
    assert not shipped, (
        f"{shipped} are registered in uad_data/internal_datasets.py and their prompt "
        f"files are recorded as rendering a blank. Fix the file on the Hub before "
        f"registering the task.")
    print("PASS: no dataset registers a task whose prompt file is known broken.")


def test_prompt_files_without_a_dataset_are_known() -> None:
    """A new orphan means someone dropped a registration without its prompt file.

    A subset, not an equality: a dataset that later registers `sentiment_analysis`
    (ADR-0008 names this as the plausible next step for MELD's second axis) is a
    legitimate change and should not fail here.
    """
    registered = {task.value for task in _registered_tasks()}
    orphans = set(_contract()) - registered
    unexpected = sorted(orphans - UNREGISTERED_TASKS)
    assert not unexpected, (
        f"prompt files with no internal dataset: {unexpected}. Delete the file from "
        f"the Hub (and its Task member) if the task is gone for good, or add it to "
        f"UNREGISTERED_TASKS.")
    print(f"PASS: the only prompt files without a dataset are {sorted(orphans)}.")


def test_slurp_real_carries_its_intent_labels_as_classification() -> None:
    """#57: the two prompt-less tasks are gone, and the label survives as `category`."""
    tasks = set(DATASETS_DIRECTORY["slurp_real"].tasks)
    assert Task.CLASSIFICATION in tasks, tasks
    assert not {t.value for t in tasks} & {"intent_detection", "action_classification"}, tasks
    print("PASS: slurp_real's intent labels come through classification.")


if __name__ == "__main__":
    test_every_registered_task_has_a_prompt_file()
    test_no_registered_task_renders_a_blank()
    test_the_only_unrenderable_placeholders_are_the_known_defects()
    test_known_bad_placeholders_are_all_unregistered_tasks()
    test_prompt_files_without_a_dataset_are_known()
    test_slurp_real_carries_its_intent_labels_as_classification()
