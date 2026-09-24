"""Configuration for an evaluation run.

`EvalConfig` is the single settings object threaded through the harness: it
selects the model backend, tells `uad_data.load_uad_dataset` which slice of the
dataset to build, and tunes batching / preprocessing / output. `main.py`
populates it from CLI flags; the Colab notebook populates it from form fields.
"""
from dataclasses import dataclass
from typing import Optional

import models
from uad_data import run_options


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
    model_path: Optional[str] = None  # overrides models.DEFAULT_MODEL_PATHS if set
    max_new_tokens: int = 256

    # Performance
    batch_size: int = 4
    num_preprocessing_workers: int = 4  # threads for parallel audio preprocessing

    # Evaluation
    clips_per_split: Optional[int] = None  # first N clips of each selected split; None = every clip
    seed: int = run_options.DEFAULT_SEED  # seeds the loader's prompt-template picks
    output_dir: Optional[str] = None   # directory for results.jsonl + summary.json

    # Auth
    hf_token: Optional[str] = None

    # Audio preprocessing
    target_sr: int = 16_000
    max_audio_seconds: int = 30

    def __post_init__(self):
        run_options.validate_clips_per_split(self.clips_per_split)
        self.dataset_split = run_options.resolve_dataset_split(
            self.dataset_split, clips_per_split=self.clips_per_split, uncapped_default="test")

    @property
    def is_smoke_run(self) -> bool:
        """Whether this run caps clips per split, and so tolerates load failures."""
        return self.clips_per_split is not None

    @property
    def resolved_model_path(self) -> str:
        return models.resolve_model_path(self.model_choice, self.model_path)
