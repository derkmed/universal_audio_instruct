"""Qwen3-Omni training backend.

Mirrors `eval.backends.qwen.QwenBackend`'s batched path -- audio bytes written
to temp WAVs so `process_mm_info` can read them, conversations rendered with
`apply_chat_template(tokenize=False)`, then one batched
`processor(text=[...], audio=[...], padding=True)` call -- extended with the
assistant answer turn and prompt-masked labels for training.
"""
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

from .base import TrainBackend

USE_AUDIO_IN_VIDEO = False


class QwenTrainBackend(TrainBackend):

    def _load_processor(self):
        model_id = self.config.resolved_model_path
        print(f"Loading Qwen processor: {model_id}")
        return Qwen3OmniMoeProcessor.from_pretrained(model_id)

    def _load_model(self):
        model_id = self.config.resolved_model_path
        print(f"Loading Qwen model: {model_id}")
        kwargs = dict(
            dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
        )
        if self.config.load_in_4bit:
            kwargs["quantization_config"] = self._quantization_config()
        return Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, **kwargs)

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    @staticmethod
    def _write_temp_wav(audio_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            return f.name

    def _build_prompt_conversation(self, row: dict, audio_path: str) -> list:
        # Same conversation shape as eval's _build_conversation.
        conv = []
        sys_inst = (row.get("system_instruction") or "").strip()
        if sys_inst:
            conv.append({"role": "system", "content": sys_inst})
        conv.append({
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": (row.get("prompt") or "").strip()},
            ],
        })
        return conv

    def collate(self, rows: List[dict]) -> dict[str, torch.Tensor]:
        with ThreadPoolExecutor(max_workers=len(rows)) as ex:
            temp_files = list(ex.map(
                self._write_temp_wav, [r["audio"]["bytes"] for r in rows]))
        try:
            prompt_texts, full_texts, batch_audios = [], [], []
            for row, tmp_path in zip(rows, temp_files):
                prompt_conv = self._build_prompt_conversation(row, tmp_path)
                full_conv = prompt_conv + [{
                    "role": "assistant",
                    "content": [{"type": "text", "text": (row.get("output") or "").strip()}],
                }]
                prompt_texts.append(self.processor.apply_chat_template(
                    prompt_conv, add_generation_prompt=True, tokenize=False))
                full_texts.append(self.processor.apply_chat_template(
                    full_conv, add_generation_prompt=False, tokenize=False))
                audios, _, _ = process_mm_info(
                    prompt_conv, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                # Each conversation has exactly one audio file (same as eval).
                batch_audios.append(audios[0] if audios else None)

            # Right padding so prompts are prefixes (required by mask_labels).
            self.processor.tokenizer.padding_side = "right"
            full = self.processor(
                text=full_texts, audio=batch_audios, return_tensors="pt",
                padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            prompt = self.processor(
                text=prompt_texts, audio=batch_audios, return_tensors="pt",
                padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)

            batch = dict(full)
            batch["labels"] = self.mask_labels(
                full["input_ids"], full["attention_mask"], prompt["attention_mask"])
            return batch
        finally:
            for f in temp_files:
                try:
                    os.unlink(f)
                except OSError:
                    pass
