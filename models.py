"""The model registry shared by the eval and train harnesses.

Both `EvalConfig` and `TrainConfig` take `--model GEMMA-4` style choices and an
optional checkpoint override, and both must resolve them the same way, so the
registry and the resolution rule live here rather than in either harness. Model
ids are not a dataset concern, so this stays out of `uad_data`: the loader
imports with no model config present.
"""
from typing import Optional


# Human-friendly model choice -> default HuggingFace model id.
DEFAULT_MODEL_PATHS: dict[str, str] = {
    "GEMMA-4": "google/gemma-4-e2b-it",
    "QWEN3-Omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
}


def resolve_model_path(choice: str, override: Optional[str]) -> str:
    """Resolve a model choice to the HuggingFace id to load.

    Args:
        choice: A key of `DEFAULT_MODEL_PATHS`, e.g. "GEMMA-4".
        override: An explicit model id or local path. Any non-empty value wins
            over the registry, so a finetuned checkpoint needs no new choice.

    Returns:
        The HuggingFace model id (or local path) to load.

    Raises:
        ValueError: `choice` is not in the registry and no override was given.
    """
    if override:
        return override
    if choice not in DEFAULT_MODEL_PATHS:
        raise ValueError(
            f"Unknown model_choice '{choice}'. "
            f"Either set model_path or use one of: {list(DEFAULT_MODEL_PATHS)}"
        )
    return DEFAULT_MODEL_PATHS[choice]
