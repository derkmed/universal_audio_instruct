from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from ..config import EvalConfig
from .base import InferenceRequest, ModelBackend


class GemmaBackend(ModelBackend):
    """Gemma-4 backend with batched inference.

    Batching strategy:
      - apply_chat_template(tokenize=False) per row → formatted text string
        with audio placeholder tokens.
      - processor(text=[...], audio=[...], padding=True) batches all samples in
        one call; the processor handles audio feature extraction and padding.
      - model.generate runs once for the whole batch.

    Falls back to sequential single-row inference if the batch call raises
    (e.g. processor version doesn't support batch audio).
    """

    def __init__(self, config: EvalConfig) -> None:
        model_id = config.resolved_model_path
        print(f"Loading Gemma processor: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, token=config.hf_token)
        print(f"Loading Gemma model: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=config.hf_token,
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
            print(f"[GemmaBackend] Batch inference failed ({exc}); falling back to sequential.")
            return [self._generate_single(r) for r in requests]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(self, req: InferenceRequest) -> list:
        full_text = f"{req.sys_inst}\n\n{req.prompt_text}".strip()
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": req.audio_array},
                    {"type": "text", "text": full_text},
                ],
            }
        ]

    def _generate_single(self, req: InferenceRequest) -> str:
        messages = self._build_messages(req)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        return self.processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    def _generate_batched(self, requests: List[InferenceRequest]) -> List[str]:
        texts = []
        audio_arrays = []
        for req in requests:
            messages = self._build_messages(req)
            # tokenize=False so we can batch through processor.__call__ below
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            texts.append(text)
            audio_arrays.append(req.audio_array)

        # Left-pad so all prompts end at the same position → clean output slicing
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(
            text=texts,
            audio=audio_arrays,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device, dtype=self.model.dtype)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated = outputs[:, input_len:]
        return [
            self.processor.decode(g, skip_special_tokens=True).strip()
            for g in generated
        ]
