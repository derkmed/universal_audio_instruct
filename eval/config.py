"""Configuration for an evaluation run.

`EvalConfig` is the single settings object threaded through the harness: it
selects the model backend, tells `uad_data.load_uad_dataset` which slice of the
dataset to build, and tunes batching / preprocessing / output. `main.py`
populates it from CLI flags; the Colab notebook populates it from form fields.
"""
from dataclasses import dataclass
from typing import Optional


# Human-friendly model choice -> default HuggingFace model id.
DEFAULT_MODEL_PATHS: dict[str, str] = {
    "GEMMA-4": "google/gemma-4-e2b-it",
    "QWEN3-Omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
}


def resolve_dataset_split(
    dataset_split: Optional[str], *, clips_per_split: Optional[int], uncapped_default: str,
) -> str:
    """Fill in an unset `dataset_split`, and reject a cap below 1.

    Both config classes resolve here, so a command line and the notebook read an
    unset split the same way: a smoke run wants every registered split, and a
    regular run wants the one split its harness is for.
    """
    if clips_per_split is not None and clips_per_split < 1:
        raise ValueError(
            f"clips_per_split must be a positive integer, got {clips_per_split}.")
    if dataset_split is not None:
        return dataset_split
    return "all" if clips_per_split is not None else uncapped_default


@dataclass
class EvalConfig:
    """All knobs for one evaluation run (model, dataset slice, batching, output)."""

    model_choice: str  # "GEMMA-4" | "QWEN3-Omni"

    # Dataset
    dataset_name: str = "AudioInstruct/Universal-Audio-Understanding"  # HF Hub repo_id passed to the loader
    # The splits to load, written the HuggingFace way: one name, several joined
    # with "+", or "all". Unset resolves in __post_init__.
    dataset_split: Optional[str] = None
    # Local path to a UAD JSON config, or the name of one hosted in the repo's
    # universal_audio_dataset_configs/ folder (see uad_data.loader).
    json_config_path: str = "configs/clotho_config.json"

    # Model
    model_path: Optional[str] = None  # overrides DEFAULT_MODEL_PATHS if set
    max_new_tokens: int = 256

    # Performance
    batch_size: int = 4
    num_preprocessing_workers: int = 4  # threads for parallel audio preprocessing

    # Evaluation
    clips_per_split: Optional[int] = None  # first N clips of each selected split; None = every clip
    seed: int = 42  # seeds the loader's prompt-template picks
    output_dir: Optional[str] = None   # directory for results.jsonl + summary.json

    # Auth
    hf_token: Optional[str] = None

    # Audio preprocessing
    target_sr: int = 16_000
    max_audio_seconds: int = 30

    def __post_init__(self):
        self.dataset_split = resolve_dataset_split(
            self.dataset_split, clips_per_split=self.clips_per_split, uncapped_default="test")

    @property
    def resolved_model_path(self) -> str:
        if self.model_path:
            return self.model_path
        if self.model_choice not in DEFAULT_MODEL_PATHS:
            raise ValueError(
                f"Unknown model_choice '{self.model_choice}'. "
                f"Either set model_path or use one of: {list(DEFAULT_MODEL_PATHS)}"
            )
        return DEFAULT_MODEL_PATHS[self.model_choice]
