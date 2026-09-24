"""Verify the Hub's `prompts/` against the contract the offline suite tests.

The prompt templates live on the Hub, not in this repo (ADR-0001), and the test
suite runs offline, so `tests/hub_prompts.json` records what `prompts/` holds:
per task, its file and the jinja2 placeholders its templates use.
`tests/test_prompt_contract.py` reads that file and checks it against the task
registry -- which is how #57 (a registered task with no prompt file) is caught
without a network call.

This module is the other half: it fetches the real `prompts/` and compares, so
the recorded contract cannot quietly drift away from the Hub.

    python -m uad_data.check_hub_prompts            # verify, exit 1 on drift
    python -m uad_data.check_hub_prompts --write    # refresh after a Hub change

`tests/hub_prompts.json` records what the Hub **has**, not what it should have.
A contract edited by hand to describe a fix nobody had pushed would leave the
offline suite green while real runs stayed broken, and the next `--write` would
turn it red for a reason the runner did not cause. So the only way to change the
recording is to change the Hub and re-run `--write`: when the offline suite fails
on a prompt file, fix the file on the Hub rather than the recording of it.

`.github/workflows/hub-prompts.yml` runs the verify path on a schedule. Run it by
hand after editing anything in `prompts/` on the Hub and commit the refreshed
JSON; nothing in the offline suite can notice a Hub edit on its own.

This is a development command: it reads and writes `tests/hub_prompts.json` in a
checkout, and an installed copy of the package has no `tests/` to find.
"""
import argparse
import glob
import json
import os
import sys
from typing import Any

import jinja2
import jinja2.meta

from . import hub
from . import prompts as prompts_lib
from . import tasks

CONTRACT_RELPATH = os.path.join("tests", "hub_prompts.json")

TEMPLATE_COLUMNS = (
    prompts_lib.SYSTEM_INSTRUCTIONS_COLUMN,
    prompts_lib.PROMPTS_COLUMN,
    prompts_lib.OUTPUTS_COLUMN,
)


class NotACheckoutError(RuntimeError):
    """`tests/hub_prompts.json` is not where the package sits.

    Raised rather than letting the read fail with a bare `FileNotFoundError` --
    or, worse, letting `--write` create the file under `site-packages/` and
    report success for a contract nobody will ever read.
    """


def contract_path() -> str:
    """Locate `tests/hub_prompts.json` from this module's place in a checkout.

    `tests/` is not in `pyproject.toml`'s `packages`, so it is there for the
    editable install the suite and CI use and absent from a plain `pip install .`.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, CONTRACT_RELPATH)
    if not os.path.isdir(os.path.dirname(path)):
        raise NotACheckoutError(
            f"{CONTRACT_RELPATH} is not under {root}: this command reads and writes "
            f"the repo's recorded contract, so run it from a checkout (pip install -e .), "
            f"not from an installed copy of uad_data.")
    return path


def placeholder_names(template: str) -> set[str]:
    """The names a jinja2 template reads, e.g. `{{category}}` -> `{"category"}`.

    `io_templates.Template.make` renders with jinja2's default `Undefined`, so a
    name the render context has no value for becomes an empty string and the row
    is generated anyway, silently wrong. Listing the names is what lets the
    offline test compare them against `Task.features`.
    """
    return jinja2.meta.find_undeclared_variables(jinja2.Environment().parse(template))


def build_contract(prompts_dir: str) -> dict[str, Any]:
    """Read a `prompts/` directory into the contract `tests/hub_prompts.json` holds.

    Keyed by task value so the file diffs readably, one block per task. Raises if
    two files claim the same task: `_get_prompt_templates` treats that as a config
    error, and the contract should not paper over it.

    A file naming a task this code does not have is recorded, not raised on. It is
    the most important thing this command can find -- `_get_prompt_templates`
    builds a `PromptFilepath` for every file it globs, so one such file makes
    `Task(...)` raise for *every* task and takes a whole run down, which is why
    a prompt file and its `Task` member have to be added and removed together.
    Reporting it as drift names the file; raising here would only reproduce the
    outage, with a traceback in place of an explanation.
    """
    contract: dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(prompts_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        task = data.get(prompts_lib.TASK_COLUMN)
        if not isinstance(task, str):
            raise ValueError(
                f"{os.path.basename(path)} has no {prompts_lib.TASK_COLUMN!r} string; "
                f"PromptFilepath rejects it and so does this.")
        if task in contract:
            raise ValueError(
                f"{os.path.basename(path)} and {contract[task]['file']} both claim "
                f"task {task!r}; _get_prompt_templates accepts only one.")
        names: set[str] = set()
        for column in TEMPLATE_COLUMNS:
            for template in data.get(column, []):
                names |= placeholder_names(template)
        contract[task] = {
            "file": os.path.basename(path),
            "placeholders": sorted(names),
        }
    return contract


def read_contract() -> dict[str, Any]:
    with open(contract_path(), encoding="utf-8") as f:
        return json.load(f)


def write_contract(contract: dict[str, Any]) -> str:
    path = contract_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _is_a_task(value: str) -> bool:
    try:
        tasks.Task(value)
    except ValueError:
        return False
    return True


def drift(recorded: dict[str, Any], live: dict[str, Any]) -> list[str]:
    """One line per task the recorded contract and the Hub disagree about."""
    lines = []
    for task in sorted(set(recorded) | set(live)):
        if task not in live:
            lines.append(f"  {task}: recorded as {recorded[task]['file']}, gone from the Hub")
        elif task not in recorded:
            known = "" if _is_a_task(task) else " -- and is not a Task, so every task's lookup raises while it is there"
            lines.append(
                f"  {task}: {live[task]['file']} is on the Hub, not recorded{known}")
        elif recorded[task] != live[task]:
            lines.append(f"  {task}: recorded {recorded[task]}, Hub has {live[task]}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uad_data.check_hub_prompts",
        description="Check tests/hub_prompts.json against the Hub's prompts/ folder.")
    parser.add_argument("--write", action="store_true",
                        help="Overwrite tests/hub_prompts.json with what the Hub holds.")
    parser.add_argument("--repo-id", default=hub.DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None, help="Hub revision to read prompts/ at.")
    parser.add_argument("--token", default=None,
                        help="HF token (default: the saved Hugging Face login).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # Fail before the download when there is nowhere to read or write the contract.
    path = contract_path()

    prompts_dir = hub.download_prompts_dir(
        repo_id=args.repo_id, revision=args.revision, token=args.token)
    live = build_contract(prompts_dir)

    if args.write:
        print(f"Wrote {len(live)} task(s) to {write_contract(live)}")
        return

    differences = drift(read_contract(), live)
    if differences:
        print(f"{path} no longer matches {args.repo_id}:")
        print("\n".join(differences))
        print("\nRe-run with --write to record what the Hub has. If what it has is "
              "wrong, fix it on the Hub first: this file records the Hub, never the "
              "intention.")
        raise SystemExit(1)
    print(f"{len(live)} prompt file(s) match {path}")


if __name__ == "__main__":
    sys.exit(main())
