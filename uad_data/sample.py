"""Encapsulation of outputted data sample."""

import dataclasses
import os
from typing import Any

from . import io_templates
from .tasks import Task

AUDIO_DATA_BASEPATH = "./data"

@dataclasses.dataclass(frozen=True, kw_only=True)
class Sample:

    audio_path: str
    dataset_name: str
    split: str
    task: Task
    audio_data: bytes
    metadata: dict[str, str]
    system_instruction_template: io_templates.SystemInstructionTemplate | None = None
    prompt_template: io_templates.PromptTemplate | None = None
    output_template: io_templates.OutputTemplate | None = None
    # Which utterance in `metadata['transcriptions']` this row renders
    # (asr_timestamp_search only; see `Task.utterance_indices`). Rendering
    # raises ValueError if it isn't a valid index there, or isn't None for any
    # other task (see `Task.render_context`).
    utterance_index: int | None = None

    def __post_init__(self):
        if self.system_instruction_template is None and self.prompt_template is None:
            raise ValueError(
                'At least one of system_instruction_template or prompt_template must be provided.')

    def to_output(self) -> dict[str, Any]:
        # NOTE: Every field of the metadata record is copied into the row
        # (including the loader's `tasks` list); the fields below are added or
        # overwritten on top of it.
        #
        # A shallow copy of the record is taken so that each generated row is an
        # independent dict. The same `metadata` record is reused across every
        # (task, utterance, prompt-template) expansion of one audio clip; without
        # the copy, materialising the generator into a list would alias every row
        # to the last-written state. (The old HF loading script serialised each
        # yield to Arrow immediately, so aliasing was invisible there.)
        example = dict(self.metadata)
        example["task"] = self.task.value
        example["split"] = self.split
        example["originating_dataset"] = self.dataset_name
        example["audio"] = {
            "path": (os.path.join(AUDIO_DATA_BASEPATH, self.dataset_name, self.audio_path)),
            "bytes": self.audio_data
        }
        if self.utterance_index is not None:
            example["utterance_index"] = self.utterance_index
        example['system_instruction'] = self.build_system_instruction() if self.system_instruction_template else ''
        example['prompt'] = self.build_prompt() if self.prompt_template else ''
        if self.output_template:
            example['output'] = self.build_output()
        return example

    def _context(self) -> dict[str, Any]:
        return self.task.render_context(self.metadata, self.utterance_index)

    def build_system_instruction(self) -> str:
        if not self.system_instruction_template:
            raise ValueError('Sample does not contain a system instruction template.')
        return self.system_instruction_template.make(self._context())

    def build_prompt(self) -> str:
        if not self.prompt_template:
            raise ValueError('Sample does not contain a prompt template.')
        return self.prompt_template.make(self._context())

    def build_output(self) -> str:
        if not self.output_template:
            raise ValueError('Sample does not contain output.')
        return self.output_template.make(self._context())
