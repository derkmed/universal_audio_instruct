from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class InferenceRequest:
    audio_bytes: bytes       # raw bytes from dataset (used by Qwen → temp WAV)
    audio_array: np.ndarray  # preprocessed: float32 mono at target_sr, ≤ max_seconds
    sys_inst: str
    prompt_text: str
    ground_truth: str = ""
    task: str = ""


class ModelBackend(ABC):
    """Abstract base for audio+text model backends.

    Subclasses must implement generate_batch. The evaluator always calls
    generate_batch; generate is a convenience wrapper for single samples.
    """

    @abstractmethod
    def generate_batch(self, requests: List[InferenceRequest]) -> List[str]:
        """Run inference on a batch and return predictions in the same order."""
        ...

    def generate(self, request: InferenceRequest) -> str:
        return self.generate_batch([request])[0]
