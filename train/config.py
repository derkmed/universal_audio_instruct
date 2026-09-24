"""Configuration for a finetuning run.

`TrainConfig` is the training-side twin of `eval.config.EvalConfig`: it selects
the model backend, tells `uad_data.load_uad_dataset` which slice to build, and
holds the QLoRA + HF Trainer hyperparameters. `train.main` populates it from CLI
flags.
"""
from dataclasses import dataclass, field
from typing import Optional

# The same shared homes eval reads, so --model and --split mean the same thing
# in both harnesses without either importing the other.
import models
from uad_data import run_options


@dataclass
class TrainConfig:
    """All knobs for one finetuning run (model, dataset slice, QLoRA, trainer)."""

    model_choice: str  # "GEMMA-4" | "QWEN3-Omni"

    # Dataset
    dataset_name: str = "AudioInstruct/Universal-Audio-Understanding"  # HF Hub repo_id
    # See EvalConfig.dataset_split. Unset resolves in __post_init__.
    dataset_split: Optional[str] = None
    json_config_path: str = "configs/clotho_config.json"
    clips_per_split: Optional[int] = None  # first N clips of each selected split; None = every clip

    # Model
    model_path: Optional[str] = None  # overrides models.DEFAULT_MODEL_PATHS if set

    # Audio preprocessing for Gemma (must match eval so train/eval see identical
    # inputs). The Qwen backend reads raw audio bytes and ignores these.
    target_sr: int = 16_000
    max_audio_seconds: int = 30

    # Finetuning mode: two independent knobs spanning three modes.
    #   load_in_4bit=True,  use_lora=True   -> QLoRA   (default; fits Qwen-30B on one A100)
    #   load_in_4bit=False, use_lora=True   -> LoRA    (bf16 base; faster, no quant noise)
    #   load_in_4bit=False, use_lora=False  -> full finetune (all params; most VRAM)
    # (4-bit without LoRA is rejected: a quantized base can't be trained directly.)
    load_in_4bit: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Attention projections exist under these names in both Gemma and Qwen3-Omni;
    # extend with gate_proj/up_proj/down_proj to also adapt the MLPs.
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # Trainer
    output_dir: str = "outputs/finetune"
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    gradient_checkpointing: bool = True
    seed: int = run_options.DEFAULT_SEED  # seeds the Trainer and the loader's prompt-template picks

    # Auth
    hf_token: Optional[str] = None

    @property
    def is_smoke_run(self) -> bool:
        """Whether this run caps clips per split, and so tolerates load failures."""
        return self.clips_per_split is not None

    def __post_init__(self):
        run_options.validate_clips_per_split(self.clips_per_split)
        self.dataset_split = run_options.resolve_dataset_split(
            self.dataset_split, clips_per_split=self.clips_per_split, uncapped_default="train")
        if self.load_in_4bit and not self.use_lora:
            raise ValueError(
                "load_in_4bit without use_lora is not supported: a 4-bit quantized "
                "base cannot be trained directly. Use LoRA, or disable 4-bit for a "
                "full finetune (--no-4bit --no-lora).")

    @property
    def resolved_model_path(self) -> str:
        return models.resolve_model_path(self.model_choice, self.model_path)
