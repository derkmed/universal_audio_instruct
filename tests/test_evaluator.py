"""Offline tests for eval.evaluator.Evaluator (seam S5).

Runs `Evaluator(backend, config).evaluate(rows)` with a fake `ModelBackend` that
echoes each row's reference, tiny synthetic WAVs, and a temporary `output_dir`.

Runnable directly (`python tests/test_evaluator.py`) or under pytest. Needs all
of `requirements.txt` (CPU `torch` is enough), and the `wer` metric from the
`evaluate` package.
"""
import io
import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.backends.base import ModelBackend  # noqa: E402
from eval.config import EvalConfig  # noqa: E402
from eval.evaluator import Evaluator  # noqa: E402


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.float32), 16_000, format="WAV")
    return buffer.getvalue()


def _row(audio_path: str, split: str, caption: str) -> dict:
    return {
        "audio": {"path": audio_path, "bytes": _wav_bytes()},
        "audio_path": audio_path,
        "split": split,
        "task": "caption",
        "originating_dataset": "Clotho",
        "system_instruction": "You are an audio captioner.",
        "prompt": "Describe the audio.",
        "output": caption,
        "caption": caption,
    }


class EchoBackend(ModelBackend):
    """Predicts each row's reference, and records how many rows it saw."""

    def __init__(self):
        self.seen = 0

    def generate_batch(self, requests):
        self.seen += len(requests)
        return [r.ground_truth for r in requests]


ROWS = [
    _row("test/t2.wav", "test", "a cat meows"),
    _row("train/t0.wav", "train", "a dog barks"),
    _row("validation/t1.wav", "validation", "rain falls"),
]


def _evaluate(**config) -> tuple[list[dict], EchoBackend]:
    backend = EchoBackend()
    with tempfile.TemporaryDirectory() as output_dir:
        Evaluator(backend, EvalConfig(
            model_choice="GEMMA-4", batch_size=2, output_dir=output_dir, **config,
        )).evaluate(list(ROWS))
        with open(os.path.join(output_dir, "results.jsonl"), encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    return records, backend


def test_every_loaded_row_is_evaluated() -> None:
    # clips_per_split caps clips in the loader; the evaluator must not cut rows again.
    records, backend = _evaluate(clips_per_split=1)

    assert backend.seen == 3, backend.seen
    assert len(records) == 3, records

    print("PASS: the evaluator evaluates every row it is given.")


def test_each_record_keeps_its_own_row_split() -> None:
    """A smoke run asks for every split, so a run-level split would mislabel rows."""
    records, _ = _evaluate(clips_per_split=1)

    assert [r["split"] for r in records] == ["test", "train", "validation"], records

    print("PASS: results.jsonl records each row's own split.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
