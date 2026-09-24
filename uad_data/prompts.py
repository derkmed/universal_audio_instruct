"""Loading and validation of prompt-template files (`prompts/*.json`).

Each JSON file declares a `task` plus `system_instructions` / `prompts` /
`outputs` arrays of jinja2 templates. `PromptFilepath` validates one such file
and exposes `all_templates` -- the cross-product of those arrays -- which is how
one audio clip fans out into multiple rows.

`PROMPTS_DIR` is the directory those files are read from. It defaults to a
relative `./prompts` but is repointed at runtime by `loader.load_uad_dataset`,
which downloads the repo's `prompts/` folder from the Hub and sets this to the
local snapshot path (the plain-data replacement for the old loading script's
`_ensure_hub_resources` shim).
"""
import functools
import itertools
import json
import random
from typing import Any

from . import io_templates
from . import tasks as tasks_lib

PROMPTS_DIR = './prompts'
SYSTEM_INSTRUCTIONS_COLUMN = 'system_instructions'
PROMPTS_COLUMN = 'prompts'
OUTPUTS_COLUMN = 'outputs'
TASK_COLUMN = 'task'

def validate_shape(filepath: str, data: dict[str, Any]) -> None:
    """The checks `PromptFilepath` makes that do not need a known task.

    Split out so `check_hub_prompts.build_contract` can apply exactly these to a
    file naming a task this code does not have. That command must tolerate the
    unknown task -- reporting which file carries it is the whole point, and
    raising would only reproduce the outage it is looking for -- but tolerating
    the task must not mean skipping everything else the loader will enforce
    later. Sharing the rules is what keeps the two from drifting apart.
    """
    if not filepath.endswith('.json'):
        raise ValueError('PromptFilepath only supports JSON files.')
    if TASK_COLUMN not in data:
        raise ValueError(
            f'File {filepath} is missing JSON string column "{TASK_COLUMN}".'
            ' This task should match the value of a registered task in tasks.py.')
    if SYSTEM_INSTRUCTIONS_COLUMN not in data and PROMPTS_COLUMN not in data:
        raise ValueError(
            f'File {filepath} must contain at least one of '
            f'"{SYSTEM_INSTRUCTIONS_COLUMN}" or "{PROMPTS_COLUMN}".')


class PromptFilepath:
    """Wrapper for filepath to validate and parse out prompt templates.

    A valid prompt JSON file should contain:
    * a task: defined in tasks_lib.Task
    * system_instructions (optional): a list of jinja2 str templates for system instructions
    * prompts (optional): a list of jinja2 str templates for user-turn messages
    * outputs (optional): a list of jinja2 str templates

    At least one of system_instructions or prompts must be present.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        validate_shape(self.filepath, self.data)
        self.task = tasks_lib.Task(self.data[TASK_COLUMN])

    @functools.cached_property
    def data(self) -> dict[str, Any]:
        with open(self.filepath) as f:
            try:
                return json.loads(f.read())
            except json.decoder.JSONDecodeError as e:
                raise RuntimeError(f'{self.filepath} is misconfigured: {e}')

    @functools.cached_property
    def system_instruction_templates(self) -> list[io_templates.SystemInstructionTemplate]:
        templates = self.data.get(SYSTEM_INSTRUCTIONS_COLUMN, [])
        return [
            io_templates.SystemInstructionTemplate(task=self.task, template=template)
            for template in templates
        ]

    @functools.cached_property
    def prompt_templates(self) -> list[io_templates.PromptTemplate]:
        templates = self.data.get(PROMPTS_COLUMN, [])
        return [
            io_templates.PromptTemplate(task=self.task, template=template)
            for template in templates
        ]

    @functools.cached_property
    def output_templates(self) -> list[io_templates.OutputTemplate]:
        templates = self.data.get(OUTPUTS_COLUMN, [])
        return [
            io_templates.OutputTemplate(task=self.task, template=template)
            for template in templates
        ]

    @functools.cached_property
    def all_templates(self) -> list[tuple[
        io_templates.SystemInstructionTemplate | None,
        io_templates.PromptTemplate | None,
        io_templates.OutputTemplate | None
    ]]:
        """Returns the cross product of system_instruction, prompt, and output templates.

        At least one of system_instruction or prompt in each returned tuple is non-None.
        """
        si_templates = self.system_instruction_templates or [None]
        p_templates = self.prompt_templates or [None]
        o_templates = self.output_templates or [None]
        combos = list(itertools.product(si_templates, p_templates, o_templates))
        return [(si, p, o) for si, p, o in combos if si is not None or p is not None]

    def random_template_selection(self, rng: random.Random) -> tuple[
        io_templates.SystemInstructionTemplate | None,
        io_templates.PromptTemplate | None,
        io_templates.OutputTemplate | None
    ]:
        return rng.choice(self.all_templates)

    def accepts_task(self, task: tasks_lib.Task) -> bool:
        return self.data[TASK_COLUMN] == task.value
