"""Offline tests for the command that checks the Hub's `prompts/` folder.

`build_contract` reads files the loader will later read through
`prompts.PromptFilepath`, and the two must agree about what a usable prompt file
is. They briefly did not: to report a file naming a task this code lacks --
rather than dying on it, which is the one thing the command exists to catch --
`build_contract` stopped going through `PromptFilepath` at all, and lost the rest
of that constructor's validation with it. A file the loader rejects then recorded
clean, and the guard said so while every run of that task raised.

So the rule these tests pin down is: `build_contract` tolerates exactly one thing
`PromptFilepath` will not, the unknown task, and nothing else.

Runnable directly (`python tests/test_check_hub_prompts.py`) or under pytest.
"""
import json
import os
import tempfile

from uad_data import check_hub_prompts, prompts as prompts_lib
from uad_data.tasks import Task


def _prompts_dir(**files: dict) -> str:
    directory = tempfile.mkdtemp()
    for name, body in files.items():
        with open(os.path.join(directory, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(body, f)
    return directory


def test_a_file_the_loader_rejects_is_not_recorded_clean() -> None:
    """`outputs` alone: PromptFilepath raises, so recording it would be a lie.

    `_get_prompt_templates` builds a `PromptFilepath` for the file before it ever
    reads a template, so every row of the task dies on
    `must contain at least one of "system_instructions" or "prompts"`. A contract
    that recorded it would report no drift while the task was unrunnable.
    """
    directory = _prompts_dir(asr={"task": "asr", "outputs": ["{{transcription}}"]})
    try:
        check_hub_prompts.build_contract(directory)
    except ValueError as error:
        assert prompts_lib.SYSTEM_INSTRUCTIONS_COLUMN in str(error), error
        print("PASS: a file with no system_instructions or prompts is rejected.")
        return
    raise AssertionError("build_contract recorded a file PromptFilepath rejects.")


def test_a_file_that_yields_no_templates_is_not_recorded_clean() -> None:
    """Empty arrays pass PromptFilepath and still leave the task unrunnable.

    `all_templates` drops every combination whose system instruction and prompt
    are both None, so empty arrays give an empty cross product and
    `_get_prompt_templates` raises `PromptTemplateError`. The key is present, so
    the constructor's own check does not see it.
    """
    directory = _prompts_dir(
        asr={"task": "asr", "system_instructions": [], "outputs": ["{{transcription}}"]})
    try:
        check_hub_prompts.build_contract(directory)
    except ValueError as error:
        assert "template" in str(error).lower(), error
        print("PASS: a file that yields no templates is rejected.")
        return
    raise AssertionError("build_contract recorded a file that gives no templates.")


def test_an_unknown_task_is_recorded_rather_than_raised_on() -> None:
    """The one tolerance: report the file, because raising reproduces the outage.

    A prompt file naming a task `Task` does not have makes `Task(...)` raise for
    *every* task, so this command has to survive reading it in order to say which
    file is at fault.
    """
    directory = _prompts_dir(
        ghost={"task": "ghost_task", "system_instructions": ["hi {{category}}"]})
    contract = check_hub_prompts.build_contract(directory)
    assert contract == {
        "ghost_task": {"file": "ghost.json", "placeholders": ["category"]}
    }, contract
    print("PASS: a file naming an unknown task is recorded, not raised on.")


def test_drift_warns_that_recording_an_unregistered_task_fails_the_suite() -> None:
    """`--write` is the advice for drift, and here it swaps one red for another.

    The task is a real `Task` that no dataset registers, so recording it would
    fail `test_every_prompt_file_has_a_dataset`. The message has to say so, or it
    walks the operator into that.

    Every registered task currently has a dataset -- that is what
    `test_every_prompt_file_has_a_dataset` enforces -- so the registry is
    narrowed here to produce the state rather than waiting for one to appear.
    """
    live = {"english_translation": {"file": "english_translation.json",
                                    "placeholders": ["english_translation"]}}
    original = check_hub_prompts.DATASETS_DIRECTORY
    check_hub_prompts.DATASETS_DIRECTORY = {
        name: dataset for name, dataset in original.items()
        if Task.ENGLISH_TRANSLATION not in dataset.tasks
    }
    try:
        lines = check_hub_prompts.drift({}, live)
    finally:
        check_hub_prompts.DATASETS_DIRECTORY = original
    assert len(lines) == 1, lines
    assert "no dataset registers it" in lines[0], lines[0]
    print("PASS: drift warns before --write records an unregistered task.")


def test_drift_stays_quiet_for_a_task_that_is_simply_new() -> None:
    """A registered task's new file is ordinary drift: --write is the whole fix."""
    live = {"asr": {"file": "asr.json", "placeholders": ["transcription"]}}
    lines = check_hub_prompts.drift({}, live)
    assert len(lines) == 1 and lines[0].endswith("not recorded"), lines
    print("PASS: ordinary drift carries no extra warning.")


def test_drift_names_an_unknown_task_as_the_wider_outage() -> None:
    live = {"ghost_task": {"file": "ghost.json", "placeholders": []}}
    lines = check_hub_prompts.drift({}, live)
    assert "is not a Task" in lines[0], lines[0]
    print("PASS: drift says an unknown task takes every lookup down.")


def test_a_missing_recording_says_so_instead_of_FileNotFoundError() -> None:
    """`tests/` exists but the file does not -- the case the guard used to miss."""
    original = check_hub_prompts.contract_path
    check_hub_prompts.contract_path = lambda: os.path.join(
        tempfile.mkdtemp(), "hub_prompts.json")
    try:
        check_hub_prompts.read_contract()
    except check_hub_prompts.NotACheckoutError as error:
        assert "--write" in str(error), error
        print("PASS: a missing recording raises NotACheckoutError, not FileNotFoundError.")
        return
    finally:
        check_hub_prompts.contract_path = original
    raise AssertionError("read_contract did not report the missing recording.")


if __name__ == "__main__":
    test_a_file_the_loader_rejects_is_not_recorded_clean()
    test_a_file_that_yields_no_templates_is_not_recorded_clean()
    test_an_unknown_task_is_recorded_rather_than_raised_on()
    test_drift_warns_that_recording_an_unregistered_task_fails_the_suite()
    test_drift_stays_quiet_for_a_task_that_is_simply_new()
    test_drift_names_an_unknown_task_as_the_wider_outage()
    test_a_missing_recording_says_so_instead_of_FileNotFoundError()
