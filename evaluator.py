import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import List

import evaluate as hf_evaluate

from audio_utils import preprocess_audio
from backends.base import InferenceRequest, ModelBackend
from config import EvalConfig


def _batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


class Evaluator:
    """Runs batched evaluation of a ModelBackend over an HF dataset.

    Performance characteristics:
      - Audio preprocessing is parallelised across `config.num_preprocessing_workers`
        threads within each batch. Librosa/scipy resampling releases the GIL, so
        true parallelism is achieved for CPU-bound preprocessing.
      - Model inference runs on the batch as a whole (one forward pass per batch)
        using the backend's generate_batch, which maximises GPU utilisation.
      - Preprocessing for batch N+1 overlaps with inference for batch N because the
        ThreadPoolExecutor persists across batches.
    """

    def __init__(self, backend: ModelBackend, config: EvalConfig) -> None:
        self.backend = backend
        self.config = config

    def evaluate(self, dataset) -> dict:
        samples = list(dataset)
        if self.config.max_samples is not None:
            samples = samples[: self.config.max_samples]

        total = len(samples)
        print(f"Evaluating {total} samples (batch_size={self.config.batch_size})")

        preprocess_fn = partial(
            preprocess_audio,
            target_sr=self.config.target_sr,
            max_seconds=self.config.max_audio_seconds,
        )

        all_predictions: List[str] = []
        all_references: List[str] = []

        # Opened once here; each batch flushes to it so results survive a mid-run crash.
        jsonl_file = self._open_jsonl(self.config.output_dir)
        try:
            # Single executor shared across batches so preprocessing for batch N+1
            # can overlap with GPU inference for batch N.
            with ThreadPoolExecutor(max_workers=self.config.num_preprocessing_workers) as executor:
                for batch in _batched(samples, self.config.batch_size):
                    # Parallel audio decode + resample, then one GPU forward pass.
                    requests = self._preprocess_batch(batch, preprocess_fn, executor)
                    predictions = self.backend.generate_batch(requests)

                    all_predictions.extend(predictions)
                    all_references.extend(r.ground_truth for r in requests)

                    start_idx = len(all_predictions) - len(predictions)
                    for i, (req, pred) in enumerate(zip(requests, predictions)):
                        n = start_idx + i + 1
                        print(f"[{n:>{len(str(total))}}/{total}] GT:   {req.ground_truth}")
                        print(f"{' ' * (len(str(total)) * 2 + 4)}Pred: {pred}")
                        if jsonl_file:
                            record = {
                                "index": n - 1,
                                "dataset": self.config.dataset_name,
                                "split": self.config.dataset_split,
                                "task": req.task,
                                "sys_inst": req.sys_inst,
                                "prompt": req.prompt_text,
                                "ground_truth": req.ground_truth,
                                "prediction": pred,
                            }
                            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            jsonl_file.flush()  # persist after every sample; safe to interrupt
        finally:
            if jsonl_file:
                jsonl_file.close()

        results = self._compute_metrics(all_predictions, all_references)
        print(f"\nFinal WER: {results['wer']:.4f}")

        if self.config.output_dir:
            summary_path = os.path.join(self.config.output_dir, "summary.json")
            summary = {"wer": results["wer"], "num_samples": results["num_samples"]}
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Results saved → {self.config.output_dir}/")

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open_jsonl(output_dir: str | None):
        """Create output_dir and open results.jsonl for appending, or return None."""
        if not output_dir:
            return None
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "results.jsonl")
        return open(path, "w", encoding="utf-8")

    @staticmethod
    def _preprocess_batch(
        batch: list,
        preprocess_fn,
        executor: ThreadPoolExecutor,
    ) -> List[InferenceRequest]:
        """Preprocess audio for all samples in the batch in parallel."""
        future_to_idx = {
            executor.submit(preprocess_fn, s["audio"]["bytes"]): idx
            for idx, s in enumerate(batch)
        }
        audio_arrays = [None] * len(batch)
        for future in as_completed(future_to_idx):
            audio_arrays[future_to_idx[future]] = future.result()

        return [
            InferenceRequest(
                audio_bytes=s["audio"]["bytes"],
                audio_array=audio_arrays[idx],
                sys_inst=(s.get("system_instruction") or "").strip(),
                prompt_text=(s.get("prompt") or "").strip(),
                ground_truth=(s.get("output") or "").strip(),
                task=(s.get("task") or "").strip(),
            )
            for idx, s in enumerate(batch)
        ]

    @staticmethod
    def _compute_metrics(predictions: List[str], references: List[str]) -> dict:
        wer_metric = hf_evaluate.load("wer")
        wer = wer_metric.compute(predictions=predictions, references=references)
        return {
            "wer": wer,
            "num_samples": len(predictions),
            "predictions": predictions,
            "references": references,
        }
