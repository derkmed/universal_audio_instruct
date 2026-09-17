"""Audio preprocessing for the eval and train Gemma backends.

Turns the raw audio bytes carried on each dataset row into the normalized
float32 mono array the Gemma processor expects. Lives in `uad_data` (rather than
`eval/` or `train/`) because both harnesses need identical preprocessing so that
finetuned models are trained and evaluated on the same input distribution.

The evaluator runs it for every backend, but only the Gemma backends use the
array. The Qwen backends hand the raw bytes to `process_mm_info` instead, so the
16 kHz / 30 s settings don't apply to them.
"""
import io

import librosa
import numpy as np
import soundfile as sf


def preprocess_audio(
    audio_bytes: bytes,
    target_sr: int = 16_000,
    max_seconds: int = 30,
) -> np.ndarray:
    """Decode bytes → float32 mono array at target_sr, capped at max_seconds.

    The cap is ``max_seconds * target_sr`` audio samples, counted after
    resampling, so it holds whatever rate the clip was recorded at.
    """
    buf = io.BytesIO(audio_bytes)
    arr, orig_sr = sf.read(buf, dtype="float32")

    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    if orig_sr != target_sr:
        arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=target_sr, res_type="scipy")

    max_audio_samples = max_seconds * target_sr
    return arr[:max_audio_samples]
