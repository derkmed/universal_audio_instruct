"""Gemma training backend.

Mirrors `eval.backends.gemma.GemmaBackend`'s batched path exactly -- messages
with an audio content part, `apply_chat_template(tokenize=False)`, then one
`processor(text=[...], audio=[...], padding=True)` call for the whole batch --
extended with the assistant answer turn and prompt-masked labels for training.
"""
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from uad_data.audio_utils import preprocess_audio
from .base import TrainBackend


class GemmaTrainBackend(TrainBackend):

    def _load_processor(self):
        model_id = self.config.resolved_model_path
        print(f"Loading Gemma processor: {model_id}")
        return AutoProcessor.from_pretrained(model_id, token=self.config.hf_token)

    def _load_model(self):
        model_id = self.config.resolved_model_path
        print(f"Loading Gemma model: {model_id}")
        kwargs = dict(
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=self.config.hf_token,
        )
        if self.config.load_in_4bit:
            kwargs["quantization_config"] = self._quantization_config()
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        return model

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    def _build_user_messages(self, row: dict, audio_array) -> list:
        # Same message shape as eval's _build_messages: audio content part carries
        # the array (the chat template renders the audio placeholder from it) and
        # system instruction + prompt share the user turn.
        full_text = f"{(row.get('system_instruction') or '').strip()}\n\n" \
                    f"{(row.get('prompt') or '').strip()}".strip()
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": full_text},
                ],
            }
        ]

    def collate(self, rows: List[dict]) -> dict[str, torch.Tensor]:
        audio_arrays = [
            preprocess_audio(
                r["audio"]["bytes"],
                target_sr=self.config.target_sr,
                max_seconds=self.config.max_audio_seconds,
            )
            for r in rows
        ]

        prompt_texts, full_texts = [], []
        for row, audio_array in zip(rows, audio_arrays):
            user_messages = self._build_user_messages(row, audio_array)
            # Prompt = user turn + generation header (same rendering eval uses
            # before generate); full = prompt + the assistant answer.
            prompt_texts.append(self.processor.apply_chat_template(
                user_messages, add_generation_prompt=True, tokenize=False))
            full_messages = user_messages + [{
                "role": "assistant",
                "content": [{"type": "text", "text": (row.get("output") or "").strip()}],
            }]
            full_texts.append(self.processor.apply_chat_template(
                full_messages, add_generation_prompt=False, tokenize=False))

        # Right padding so each sample's prompt is a prefix (required by mask_labels).
        self.processor.tokenizer.padding_side = "right"
        full = self.processor(
            text=full_texts, audio=audio_arrays, return_tensors="pt", padding=True)
        prompt = self.processor(
            text=prompt_texts, audio=audio_arrays, return_tensors="pt", padding=True)

        batch = dict(full)
        batch["labels"] = self.mask_labels(
            full["input_ids"], full["attention_mask"], prompt["attention_mask"])
        return batch
