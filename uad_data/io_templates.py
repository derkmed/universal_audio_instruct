"""Jinja2 template wrappers for the three text fields of a generated row.

Each `Template` binds a task to a jinja2 string; `make(context)` renders it with
the values `Task.render_context` reads from a row's metadata (for
asr_timestamp_search, from one utterance). `prompts.PromptFilepath` builds these
from the `system_instructions` / `prompts` / `outputs` arrays in a
`prompts/*.json` file, one subclass per field:
  - `SystemInstructionTemplate` -> the row's `system_instruction`
  - `PromptTemplate`            -> the row's `prompt`
  - `OutputTemplate`            -> the row's `output` (ground truth)
"""
import abc
import dataclasses
import jinja2
import re

from . import tasks as tasks_lib


@dataclasses.dataclass(kw_only=True, frozen=True)
class Template(abc.ABC):
    # Task to generate for.
    task: tasks_lib.Task
    # String template to interpolate from.
    template: str

    def make(self, context: dict[str, str]) -> str:
        template = jinja2.Template(self.template)
        return template.render(context)

    def get_expected_kwargs(self):
        kwargs = {}
        matches = re.findall(r"\{(\w+)\}", self.template)
        for match in matches:
            kwargs[match] = None
        return kwargs


@dataclasses.dataclass(kw_only=True, frozen=True)
class SystemInstructionTemplate(Template):
    pass


@dataclasses.dataclass(kw_only=True, frozen=True)
class PromptTemplate(Template):
    pass


@dataclasses.dataclass(kw_only=True, frozen=True)
class OutputTemplate(Template):
    pass
