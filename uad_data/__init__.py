"""Universal Audio Understanding dataset library.

This package holds the dataset-generation logic that used to live inside the
HuggingFace dataset repo as a `trust_remote_code` loading script. It is now a
plain importable library: `loader.load_uad_dataset(...)` downloads the audio,
metadata, prompts and config from the (private) HF Hub repo via `huggingface_hub`
and expands them into evaluation rows -- no loading script, no trust_remote_code.
"""

from .loader import load_uad_dataset, PromptTemplateError
from .load_report import LoadedRows, LoadFailure, LoadReport, RenderFailure, SplitReport

__all__ = [
    "load_uad_dataset",
    "LoadedRows",
    "LoadReport",
    "SplitReport",
    "LoadFailure",
    "RenderFailure",
    "PromptTemplateError",
]
