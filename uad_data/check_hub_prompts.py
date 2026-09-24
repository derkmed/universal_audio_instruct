"""Verify the Hub's `prompts/` against the contract the offline suite tests.

The prompt templates live on the Hub, not in this repo (ADR-0001), and the test
suite runs offline, so `tests/hub_prompts.json` records what `prompts/` holds:
per task, its file and the jinja2 placeholders its templates use.
`tests/test_prompt_contract.py` reads that file and checks it against the task
registry -- which is how #57 (a registered task with no prompt file) is caught
without a network call.

This module is the other half: it fetches the real `prompts/` and compares.

    python -m uad_data.check_hub_prompts            # verify, exit 1 on drift
    python -m uad_data.check_hub_prompts --write    # refresh after a Hub change

Run it after editing anything in `prompts/` on the Hub, and commit the refreshed
JSON with a note of what changed. Nothing in the offline suite can notice a Hub
edit on its own.
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

CONTRACT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "hub_prompts.json")

TEMPLATE_COLUMNS = (
    prompts_lib.SYSTEM_INSTRUCTIONS_COLUMN,
    prompts_lib.PROMPTS_COLUMN,
    prompts_lib.OUTPUTS_COLUMN,
)


def placeholders(template: str) -> set[str]:
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
    """
    contract: dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(prompts_dir, "*.json"))):
        prompt_file = prompts_lib.PromptFilepath(filepath=path)
        task = prompt_file.task.value
        if task in contract:
            raise ValueError(
                f"{os.path.basename(path)} and {contract[task]['file']} both claim "
                f"task {task!r}; _get_prompt_templates accepts only one.")
        names: set[str] = set()
        for column in TEMPLATE_COLUMNS:
            for template in prompt_file.data.get(column, []):
                names |= placeholders(template)
        contract[task] = {
            "file": os.path.basename(path),
            "placeholders": sorted(names),
        }
    return contract


def read_contract() -> dict[str, Any]:
    with open(CONTRACT_PATH, encoding="utf-8") as f:
        return json.load(f)


def _diff(recorded: dict[str, Any], live: dict[str, Any]) -> list[str]:
    lines = []
    for task in sorted(set(recorded) | set(live)):
        if task not in live:
            lines.append(f"  {task}: recorded as {recorded[task]['file']}, gone from the Hub")
        elif task not in recorded:
            lines.append(f"  {task}: {live[task]['file']} is on the Hub, not recorded")
        elif recorded[task] != live[task]:
            lines.append(f"  {task}: recorded {recorded[task]}, Hub has {live[task]}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write", action="store_true",
        help="overwrite tests/hub_prompts.json with what the Hub holds")
    parser.add_argument("--repo-id", default=hub.DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)

    prompts_dir = hub.download_prompts_dir(
        repo_id=args.repo_id, revision=args.revision, token=os.environ.get("HF_TOKEN"))
    live = build_contract(prompts_dir)

    if args.write:
        with open(CONTRACT_PATH, "w", encoding="utf-8") as f:
            json.dump(live, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote {len(live)} task(s) to {CONTRACT_PATH}")
        return 0

    drift = _diff(read_contract(), live)
    if drift:
        print(f"{CONTRACT_PATH} no longer matches {args.repo_id}:")
        print("\n".join(drift))
        print("\nRe-run with --write once the Hub is the way you want it.")
        return 1
    print(f"{len(live)} prompt file(s) match {CONTRACT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
