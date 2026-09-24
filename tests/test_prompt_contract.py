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
placeholders its templates read. What it records is what the Hub **has**, never
what someone means the Hub to have -- `python -m uad_data.check_hub_prompts`
re-checks it against the real Hub and `--write` refreshes it, so a recording that
described a fix nobody had pushed would leave this suite green while real runs
stayed broken, and would go red the next time anyone refreshed it.

That cuts both ways, and the second way is why there is no carve-out below. A
failure here is a statement about the Hub, so it is answered on the Hub -- edit
or delete the file, then `--write` -- and not by teaching the test to expect the
defect. The two files that made that tempting (`sentiment_analysis.json` and
`intonation_detection.json`, each reading a name its task's features never
supplied) are gone from the Hub and their tasks are gone from `Task`, so the
invariants are unconditional again.

Runnable directly (`python tests/test_prompt_contract.py`) or under pytest.
"""
import json

from uad_data.check_hub_prompts import contract_path
from uad_data.internal_datasets import DATASETS_DIRECTORY
from uad_data.tasks import Task


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


def _task(task_value: str) -> Task:
    """`Task(task_value)`, but failing as an assertion rather than a ValueError.

    `test_every_recorded_prompt_file_names_a_real_task` is the test that explains
    an unknown task, and running the whole file reaches it first. A selective run
    (`pytest -k renders_a_blank`) does not, so without this the helper would raise
    a bare `ValueError` from three frames down and say none of it.
    """
    try:
        return Task(task_value)
    except ValueError:
        raise AssertionError(
            f"tests/hub_prompts.json records {task_value!r}, which is not in Task. "
            f"See test_every_recorded_prompt_file_names_a_real_task: while such a "
            f"file is on the Hub, _get_prompt_templates raises for every task."
        ) from None


def _unrenderable(task_value: str, entry: dict) -> set[str]:
    """Placeholders the task's render context has no value for."""
    return set(entry["placeholders"]) - _render_context_keys(_task(task_value))


def test_every_recorded_prompt_file_names_a_real_task() -> None:
    """A prompt file for a task this code lacks breaks *every* task, not just its own.

    `_get_prompt_templates` builds a `PromptFilepath` for every file it globs out
    of `prompts/`, and that constructor calls `Task(...)`, so one file naming an
    unknown task raises before any task's templates are found. That is why a
    `Task` member and its prompt file have to be added and removed in one move --
    and why this is checked before the tests that call `Task(...)` themselves.
    A selective run can skip that ordering, so `_task` repeats the explanation
    rather than letting a helper raise a bare ValueError.
    """
    unknown = sorted(t for t in _contract() if t not in {task.value for task in Task})
    assert not unknown, (
        f"tests/hub_prompts.json records {unknown}, which are not in Task. While such "
        f"a file is on the Hub, _get_prompt_templates raises for every task. Add the "
        f"Task member, or delete the file from the Hub and re-run "
        f"`python -m uad_data.check_hub_prompts --write`.")
    print(f"PASS: all {len(_contract())} recorded prompt files name a real Task.")


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


def test_no_prompt_file_renders_a_blank() -> None:
    """Every placeholder on the Hub is a name its task's render context supplies.

    Every file, not only the registered ones. A file nobody renders yet is the
    one that gets this wrong and stays wrong -- that is the whole history of
    `sentiment_analysis.json` and `intonation_detection.json` -- and the moment a
    dataset registers it, the blank ships.
    """
    for task_value, entry in sorted(_contract().items()):
        unrenderable = sorted(_unrenderable(task_value, entry))
        task = _task(task_value)
        assert not unrenderable, (
            f"{entry['file']} reads {unrenderable}, which Task.{task.name}.render_context "
            f"never supplies (it gives {sorted(_render_context_keys(task))}). jinja2 "
            f"renders those as empty strings instead of failing, so every row of the "
            f"task is silently wrong. Fix the file on the Hub, then re-run "
            f"`python -m uad_data.check_hub_prompts --write`.")
    print(f"PASS: none of the {len(_contract())} prompt files renders a blank.")


def test_every_prompt_file_has_a_dataset() -> None:
    """A prompt file no dataset registers is dead weight, and dead weight rots.

    Strict, where this used to permit a recorded set of known orphans. That
    allowlist is what made an equality check wrong: a dataset legitimately
    registering one of the listed tasks *shrank* the orphan set and failed the
    test, with a message telling you to undo the registration. Emptying the
    allowlist collapses the distinction -- `orphans <= set()` and
    `orphans == set()` are the same assertion, because an empty set cannot shrink
    -- so the strict form costs nothing and only the guidance had to change.

    What still fires here is a prompt file uploaded ahead of the dataset that
    registers its task, and that is worth a failure rather than an exemption:
    it is exactly the state the two removed files sat in, unrendered by anything
    and therefore unchecked by anyone, both reading a placeholder their task
    never supplied. Land the file and the registration together.
    """
    registered = {task.value for task in _registered_tasks()}
    orphans = sorted(set(_contract()) - registered)
    assert not orphans, (
        f"prompt files with no internal dataset: {orphans}. Register the task on a "
        f"dataset in uad_data/internal_datasets.py, or -- if it is gone for good -- "
        f"delete the file from the Hub, drop its Task member, and re-run "
        f"`python -m uad_data.check_hub_prompts --write`.")
    print(f"PASS: all {len(_contract())} prompt files have a dataset that registers them.")


def test_slurp_real_carries_its_intent_labels_as_classification() -> None:
    """#57: the two prompt-less tasks are gone, and the label survives as `category`."""
    tasks = set(DATASETS_DIRECTORY["slurp_real"].tasks)
    assert Task.CLASSIFICATION in tasks, tasks
    assert not {t.value for t in tasks} & {"intent_detection", "action_classification"}, tasks
    print("PASS: slurp_real's intent labels come through classification.")


if __name__ == "__main__":
    test_every_recorded_prompt_file_names_a_real_task()
    test_every_registered_task_has_a_prompt_file()
    test_no_prompt_file_renders_a_blank()
    test_every_prompt_file_has_a_dataset()
    test_slurp_real_carries_its_intent_labels_as_classification()
