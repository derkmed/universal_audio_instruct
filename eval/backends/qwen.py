import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

from ..config import EvalConfig
from .base import InferenceRequest, ModelBackend


class QwenBackend(ModelBackend):
    """Qwen3-Omni backend with batched inference.

    Batching strategy:
      - Raw audio bytes are written unchanged to temp files (with a .wav suffix,
        whatever the source format) in parallel (ThreadPoolExecutor) so
        process_mm_info can read them. The preprocessed audio_array is unused.
      - process_mm_info is called per-conversation to extract audio features.
      - processor(text=[...], audio=[...], padding=True) batches all processed
        audio and text in one call.
      - model.generate runs once for the whole batch.
      - Temp files are cleaned up in a finally block.

    Falls back to sequential single-sample inference if batch call fails.
    """

    USE_AUDIO_IN_VIDEO = False

    def __init__(self, config: EvalConfig) -> None:
        model_id = config.resolved_model_path
        print(f"Loading Qwen processor: {model_id}")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
        print(f"Loading Qwen model: {model_id}")
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.max_new_tokens = config.max_new_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(self, requests: List[InferenceRequest]) -> List[str]:
        if len(requests) == 1:
            return [self._generate_single(requests[0])]
        try:
            return self._generate_batched(requests)
        except Exception as exc:
            print(f"[QwenBackend] Batch inference failed ({exc}); falling back to sequential.")
            return [self._generate_single(r) for r in requests]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_conversation(self, req: InferenceRequest, audio_path: str) -> list:
        conv = []
        if req.sys_inst:
            conv.append({"role": "system", "content": req.sys_inst})
        conv.append({
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": req.prompt_text},
            ],
        })
        return conv

    @staticmethod
    def _write_temp_wav(audio_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            return f.name

    def _generate_single(self, req: InferenceRequest) -> str:
        tmp_path = self._write_temp_wav(req.audio_bytes)
        try:
            conv = self._build_conversation(req, tmp_path)
            text = self.processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(
                conv, use_audio_in_video=self.USE_AUDIO_IN_VIDEO
            )
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=self.USE_AUDIO_IN_VIDEO,
            ).to(self.model.device).to(self.model.dtype)

            with torch.inference_mode():
                text_ids, _ = self.model.generate(
                    **inputs,
                    speaker="Ethan",
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.USE_AUDIO_IN_VIDEO,
                    max_new_tokens=self.max_new_tokens,
                )

            return self.processor.batch_decode(
                text_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _generate_batched(self, requests: List[InferenceRequest]) -> List[str]:
        # Write temp files in parallel — each write is independent I/O
        with ThreadPoolExecutor(max_workers=len(requests)) as ex:
            temp_files = list(ex.map(self._write_temp_wav, [r.audio_bytes for r in requests]))

        try:
            texts = []
            batch_audios = []
            for req, tmp_path in zip(requests, temp_files):
                conv = self._build_conversation(req, tmp_path)
                text = self.processor.apply_chat_template(
                    conv, add_generation_prompt=True, tokenize=False
                )
                audios, _, _ = process_mm_info(conv, use_audio_in_video=self.USE_AUDIO_IN_VIDEO)
                texts.append(text)
                # Each conversation has exactly one audio file
                batch_audios.append(audios[0] if audios else None)

            # Left-pad so prompt ends at the same position for all samples
            self.processor.tokenizer.padding_side = "left"
            inputs = self.processor(
                text=texts,
                audio=batch_audios,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=self.USE_AUDIO_IN_VIDEO,
            ).to(self.model.device).to(self.model.dtype)

            with torch.inference_mode():
                text_ids, _ = self.model.generate(
                    **inputs,
                    speaker="Ethan",
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.USE_AUDIO_IN_VIDEO,
                    max_new_tokens=self.max_new_tokens,
                )

            decoded = self.processor.batch_decode(
                text_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return [s.strip() for s in decoded]
        finally:
            for f in temp_files:
                try:
                    os.unlink(f)
                except OSError:
                    pass
