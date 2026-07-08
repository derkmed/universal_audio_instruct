"""Abstract base for training backends.

A `TrainBackend` mirrors the eval `ModelBackend` split: model-specific code
(processor usage, chat templating, audio handling) lives in one subclass per
model family, everything else is shared. A training backend owns:

  * the loaded model + processor. Two independent config knobs pick the mode:
    `load_in_4bit` (NF4-quantize the frozen base) and `use_lora` (train adapters
    instead of all weights). Both on = QLoRA, the recipe from
    https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora;
    LoRA-only keeps the base in bf16; both off = full finetune.
  * `collate(rows)` -- the HF Trainer `data_collator`. It receives raw
    `uad_data` row dicts and returns a padded batch of tensors with `labels`.

Label masking uses the standard prompt/full two-pass recipe: the batch is
processed twice through the processor -- once with just the prompt (system +
user turn + generation header) and once with the full conversation including
the assistant answer. The prompt token count per sample (with right padding,
`attention_mask.sum()`) gives the prefix to mask with -100, so only answer
tokens contribute to the loss. Processing the prompt with the *same audio*
matters: processors expand the audio placeholder into a variable number of
tokens based on the audio features, so a text-only tokenize would undercount.
"""
from abc import ABC, abstractmethod
from typing import Any, List

import torch

from ..config import TrainConfig


class TrainBackend(ABC):
    """Loads a (QLoRA-wrapped) model+processor and collates uad_data rows into batches."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.processor = self._load_processor()
        self.model = self._load_model()
        if config.use_lora:
            self.model = self._apply_lora(self.model)
        if config.gradient_checkpointing:
            self.model.config.use_cache = False  # incompatible with checkpointing

    # ------------------------------------------------------------------
    # Model-specific hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_processor(self):
        ...

    @abstractmethod
    def _load_model(self):
        ...

    @abstractmethod
    def collate(self, rows: List[dict]) -> dict[str, torch.Tensor]:
        """Turn raw uad_data rows into a padded training batch with `labels`."""
        ...

    # ------------------------------------------------------------------
    # Shared quantization / LoRA plumbing
    # ------------------------------------------------------------------

    def _quantization_config(self):
        """4-bit NF4 quantization for the frozen base model (the Q in QLoRA)."""
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def _apply_lora(self, model):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if self.config.load_in_4bit:
            # k-bit prep (norm upcasting, input grads) only applies to quantized bases.
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=self.config.gradient_checkpointing)
        lora = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        return model

    # ------------------------------------------------------------------
    # Shared label masking
    # ------------------------------------------------------------------

    @staticmethod
    def mask_labels(
        full_input_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build labels: -100 on padding and on each sample's prompt prefix.

        Requires right padding so that a sample's prompt occupies positions
        [0, prompt_len) of its full sequence.
        """
        labels = full_input_ids.clone()
        labels[full_attention_mask == 0] = -100
        prompt_lens = prompt_attention_mask.sum(dim=-1)
        for i, plen in enumerate(prompt_lens.tolist()):
            labels[i, :plen] = -100
        return labels
