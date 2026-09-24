"""Offline test for uad_data.audio_utils.preprocess_audio.

Encodes synthetic WAV bytes in memory and checks the decoded array: mono,
float32, resampled to target_sr, and capped at max_seconds * target_sr audio
samples counted after resampling.

Runnable directly (`python tests/test_audio_utils.py`) or under pytest. Requires
`numpy`, `soundfile` and `librosa`.
"""
import io

import numpy as np
import soundfile as sf

from uad_data.audio_utils import preprocess_audio


def _wav_bytes(seconds: float, sr: int, channels: int = 1) -> bytes:
    n = int(seconds * sr)
    data = np.zeros((n, channels), dtype="float32") if channels > 1 else np.zeros(n, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def test_long_clip_is_capped_after_resampling() -> None:
    """A 5 s stereo 8 kHz clip capped at 2 s yields 2 * 16000 mono samples."""
    arr = preprocess_audio(_wav_bytes(5, 8_000, channels=2), target_sr=16_000, max_seconds=2)
    assert arr.ndim == 1, arr.shape
    assert arr.dtype == np.float32, arr.dtype
    assert len(arr) == 2 * 16_000, len(arr)
    print("PASS: a long clip is downmixed, resampled and capped at max_seconds * target_sr.")


def test_short_clip_is_not_padded() -> None:
    """A clip shorter than the cap keeps its own length."""
    arr = preprocess_audio(_wav_bytes(1, 16_000), target_sr=16_000, max_seconds=30)
    assert len(arr) == 16_000, len(arr)
    print("PASS: a clip shorter than the cap is returned whole.")


if __name__ == "__main__":
    test_long_clip_is_capped_after_resampling()
    test_short_clip_is_not_padded()
