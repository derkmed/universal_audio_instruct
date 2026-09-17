"""Batched evaluation of a model backend over a run's rows.

`Evaluator.evaluate(rows)` gives every row exactly one status, groups the rows by
(internal dataset, split, task), and reports each group's pass/fail and its one
preliminary metric. A group passes when it has at least one row and every row is
`ok`; metrics never decide that.

A smoke run carries on past a row whose audio won't decode and past a batch whose
backend raised, recording `audio_error` and `model_error`. A regular run stops at
the first error, as it does today.

The run writes `results.jsonl` (one line per row, render failures included) and
`summary.json` (one entry per group) when `output_dir` is set, prints the group
table either way, and returns the summary plus the same records under `rows`, so
a caller with no `output_dir` -- the Colab notebook -- still sees everything.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Optional

from uad_data.audio_utils import preprocess_audio
from uad_data.load_report import LoadReport
from . import metrics
from .backends.base import InferenceRequest, ModelBackend
from .config import EvalConfig


# Every way a row can end. `ok` is the only one a passing group may contain.
STATUSES = ("ok", "empty_output", "render_error", "audio_error", "model_error")


def _batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _or_none(metric_fn, *args):
    """A preliminary metric's value, or None if computing it failed.

    Metrics are information only -- "a group passes or fails on its row statuses,
    never on these numbers" -- so one must never cost a run its `results.jsonl`
    and `summary.json`. `evaluate.load` reaches the Hub on a cold cache, and this
    runs per row inside the batch loop, so an offline runner or a Hub blip would
    otherwise abort a smoke run built to tolerate far worse.
    """
    try:
        return metric_fn(*args)
    except Exception as error:
        print(f"  metric unavailable: {_describe(error)}")
        return None


def print_group_table(summary: dict) -> None:
    """Print one line per group, then the overall result.

    Public so the Colab notebook can print a summary it holds in memory.
    """
    cap = summary.get("clips_per_split")
    header = "Group results" + (f" (clips per split: {cap})" if cap is not None else "")
    print(f"\n{header}")

    # `split clips`, not `clips`: the count belongs to the whole split, so every
    # task group of one split repeats it rather than each finding that many.
    columns = ["group", "split clips", "rows", *STATUSES, "metric", "result"]
    lines = [columns]
    for group in summary["groups"]:
        found, group_cap = group["clips_found"], group["clips_per_split"]
        clips = "-" if found is None else (
            str(found) if group_cap is None else f"{found}/{group_cap}")
        value = group["metric_value"]
        metric = "-" if group["metric"] is None or value is None else (
            f"{group['metric']} {value:.4f}")
        lines.append([
            f"{group['originating_dataset']}/{group['split']}/{group['task']}",
            clips,
            str(group["rows"]),
            *(str(group["statuses"][status]) for status in STATUSES),
            metric,
            "PASS" if group["passed"] else "FAIL",
        ])

    widths = [max(len(line[i]) for line in lines) for i in range(len(columns))]
    for line in lines:
        print("  " + "  ".join(cell.ljust(width) for cell, width in zip(line, widths)))

    for failure in summary["load_failures"]:
        print(f"  FAILED TO LOAD {failure['dataset']}: {failure['error']} "
              f"({failure['rows_kept']} rows kept)")

    print(f"\nOverall: {'PASS' if summary['passed'] else 'FAIL'}")


class Evaluator:
    """Runs batched evaluation of a ModelBackend over a list of uad_data rows.

    Performance characteristics:
      - Audio preprocessing is parallelised across `config.num_preprocessing_workers`
        threads within each batch. Librosa/scipy resampling releases the GIL, so
        true parallelism is achieved for CPU-bound preprocessing.
      - Model inference runs on the batch as a whole (one batched generate call
        per batch) using the backend's generate_batch, which maximises GPU
        utilisation.
      - Batches run one after another: batch N+1 is preprocessed only after
        inference for batch N returns. The ThreadPoolExecutor is reused across
        batches only to avoid recreating its threads.
    """

    def __init__(self, backend: ModelBackend, config: EvalConfig) -> None:
        self.backend = backend
        self.config = config

    def evaluate(self, dataset) -> dict:
        rows = list(dataset)
        # `load_uad_dataset` returns rows carrying their load report. A caller
        # that hands over a plain list (the tests, the notebook) gets an empty one.
        report: LoadReport = getattr(dataset, "report", None) or LoadReport()

        total = len(rows)
        print(f"Evaluating {total} rows (batch_size={self.config.batch_size})")

        preprocess_fn = partial(
            preprocess_audio,
            target_sr=self.config.target_sr,
            max_seconds=self.config.max_audio_seconds,
        )

        records: List[dict] = []
        # group key -> the (row, prediction) pairs its metric is computed over.
        predicted: dict[tuple[str, str, str], list[tuple[dict, str]]] = {}

        # Opened once here; each batch flushes to it so results survive a mid-run crash.
        jsonl_file = self._open_jsonl(self.config.output_dir)
        try:
            # One executor for the whole run, reused by every batch's preprocessing.
            with ThreadPoolExecutor(max_workers=self.config.num_preprocessing_workers) as executor:
                for batch in _batched(rows, self.config.batch_size):
                    # Parallel audio decode + resample, then one GPU forward pass.
                    prepared = self._preprocess_batch(batch, preprocess_fn, executor)
                    predictions, batch_error = self._predict(
                        [request for request in prepared if isinstance(request, InferenceRequest)])

                    for row, request in zip(batch, prepared):
                        if not isinstance(request, InferenceRequest):
                            # The row never reached the model, but its prompt and
                            # ground truth are its own, and triage wants them.
                            record = self._record(
                                len(records), row, status="audio_error",
                                error=_describe(request))
                        elif predictions is None:
                            record = self._record(
                                len(records), row, request=request, status="model_error",
                                error=batch_error)
                        else:
                            prediction = predictions.pop(0)
                            status = "ok" if prediction.strip() else "empty_output"
                            record = self._record(
                                len(records), row, request=request, status=status,
                                prediction=prediction)
                            predicted.setdefault(self._key(record), []).append((row, prediction))

                        records.append(record)
                        self._print_row(record, total)
                        self._write(jsonl_file, record)

            # Rows the loader could never render reach the report, not the model.
            for failure in report.render_failures:
                record = self._record(len(records), {
                    "originating_dataset": failure.dataset,
                    "split": failure.split,
                    "task": failure.task,
                    "audio_path": failure.audio_path,
                }, status="render_error", error=failure.error)
                records.append(record)
                self._write(jsonl_file, record)
        finally:
            if jsonl_file:
                jsonl_file.close()

        summary = self._summarise(records, predicted, report)
        summary["rows"] = records

        print_group_table(summary)

        if self.config.output_dir:
            summary_path = os.path.join(self.config.output_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in summary.items() if k != "rows"}, f, indent=2)
            print(f"Results saved → {self.config.output_dir}/")

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(record: dict) -> tuple[str, str, str]:
        """The group a record belongs to: its internal dataset, split and task."""
        return (record["originating_dataset"], record["split"], record["task"])

    def _record(
        self,
        index: int,
        row: dict,
        *,
        status: str,
        request: Optional[InferenceRequest] = None,
        prediction: Optional[str] = None,
        error: Optional[str] = None,
    ) -> dict:
        """One `results.jsonl` line. `prediction` is None when the row never ran."""
        task = row.get("task") or ""
        return {
            "index": index,
            "model_choice": self.config.model_choice,
            "model": self.config.resolved_model_path,
            "dataset": self.config.dataset_name,
            "originating_dataset": row.get("originating_dataset") or "",
            # The row's own split, not the run's: a smoke run asks for "all",
            # which would label every row alike.
            "split": row.get("split") or "",
            "task": task,
            "audio_path": row.get("audio_path") or "",
            # A row that never reached the model has no request, but it still
            # has its own rendered text, which is what triage reads.
            "sys_inst": request.sys_inst if request else (
                row.get("system_instruction") or "").strip(),
            "prompt": request.prompt_text if request else (row.get("prompt") or "").strip(),
            "ground_truth": request.ground_truth if request else (
                row.get("output") or "").strip(),
            "answer": metrics.answer_of(row),
            "prediction": prediction,
            "status": status,
            "error": error,
            "metric": metrics.metric_name(task),
            "metric_value": (
                None if prediction is None else _or_none(metrics.metric_value, row, prediction)),
        }

    def _predict(
        self, requests: List[InferenceRequest],
    ) -> tuple[Optional[List[str]], Optional[str]]:
        """One batch's predictions, or `(None, error)` when the backend let it down.

        A backend that raises, one that returns the wrong number of predictions
        and one that returns something other than text are all the same failure:
        this batch has no usable answers. The backend's whole return contract is
        checked here, in one place, so the rest of the loop can count on one
        string per request and a misbehaving backend costs its batch rather than
        the run. A regular run re-raises instead.
        """
        if not requests:
            return [], None
        try:
            predictions = list(self.backend.generate_batch(requests))
            if len(predictions) != len(requests):
                raise ValueError(
                    f"backend returned {len(predictions)} predictions "
                    f"for {len(requests)} requests")
            # `None` is itself one of the wrong values, so the search can't use
            # it as its "found nothing" marker.
            wrong = [p for p in predictions if not isinstance(p, str)][:1]
            if wrong:
                raise TypeError(
                    f"backend returned {type(wrong[0]).__name__}, "
                    f"not text: {wrong[0]!r:.60}")
        except Exception as error:
            if not self.config.is_smoke_run:
                raise  # a regular run stops at the first error
            described = _describe(error)
            print(f"  backend failed for this batch: {described}")
            return None, described
        return predictions, None

    @staticmethod
    def _print_row(record: dict, total: int) -> None:
        width = len(str(total))
        index = record["index"] + 1
        if record["prediction"] is None:
            print(f"[{index:>{width}}/{total}] {record['status'].upper()}: {record['error']}")
            return
        print(f"[{index:>{width}}/{total}] GT:   {record['ground_truth']}")
        print(f"{' ' * (width * 2 + 4)}Pred: {record['prediction']}")

    @staticmethod
    def _write(jsonl_file, record: dict) -> None:
        if not jsonl_file:
            return
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        jsonl_file.flush()  # persist after every row; safe to interrupt

    @staticmethod
    def _open_jsonl(output_dir: str | None):
        """Create output_dir and open results.jsonl (overwriting it), or return None."""
        if not output_dir:
            return None
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "results.jsonl")
        return open(path, "w", encoding="utf-8")

    def _preprocess_batch(
        self,
        batch: list,
        preprocess_fn,
        executor: ThreadPoolExecutor,
    ) -> List[InferenceRequest | BaseException]:
        """Preprocess a batch's audio in parallel, one entry per row, in row order.

        A row whose audio won't decode gets its exception in place of a request.
        A regular run re-raises it instead.
        """
        # The subscript is inside the submitted callable, so a row with no audio
        # field at all fails the same way a row with undecodable bytes does,
        # rather than raising in this thread and taking the run with it.
        futures = [executor.submit(lambda r=row: preprocess_fn(r["audio"]["bytes"]))
                   for row in batch]

        prepared: List[InferenceRequest | BaseException] = []
        for row, future in zip(batch, futures):
            try:
                audio_array = future.result()
            except Exception as error:
                if not self.config.is_smoke_run:
                    raise  # a regular run stops at the first error
                prepared.append(error)
                continue
            prepared.append(InferenceRequest(
                audio_bytes=row["audio"]["bytes"],  # read once the decode succeeded
                audio_array=audio_array,
                sys_inst=(row.get("system_instruction") or "").strip(),
                prompt_text=(row.get("prompt") or "").strip(),
                ground_truth=(row.get("output") or "").strip(),
                task=(row.get("task") or "").strip(),
            ))
        return prepared

    def _summarise(self, records: List[dict], predicted: dict, report: LoadReport) -> dict:
        """One entry per group, plus the failed loads and the run's overall result."""
        groups: dict[tuple[str, str, str], dict] = {}

        def entry(key: tuple[str, str, str], clips_found: Optional[int]) -> dict:
            dataset, split, task = key
            return groups.setdefault(key, {
                # The internal dataset, named as the row records name it. The
                # records' own `dataset` is the UAD repo, which is not this.
                "originating_dataset": dataset,
                "split": split,
                "task": task,
                "clips_found": clips_found,
                "clips_per_split": self.config.clips_per_split,
                "rows": 0,
                "statuses": {status: 0 for status in STATUSES},
                "metric": metrics.metric_name(task),
                "metric_value": None,
                "passed": False,
            })

        # The report names every selected split's groups, so a split that yielded
        # no clips still has groups -- which fail, having no rows.
        for split in report.splits:
            for task in split.tasks:
                entry((split.dataset, split.split, task), split.clips_found)

        for record in records:
            group = entry(self._key(record), None)
            group["rows"] += 1
            group["statuses"][record["status"]] += 1

        for key, group in groups.items():
            group["metric_value"] = _or_none(
                metrics.aggregate, group["task"], predicted.get(key, []))
            group["passed"] = group["rows"] > 0 and group["statuses"]["ok"] == group["rows"]

        load_failures = [
            {"dataset": f.dataset, "error": f.error, "rows_kept": f.rows_kept}
            for f in report.load_failures
        ]
        ordered = list(groups.values())
        return {
            "model_choice": self.config.model_choice,
            "model": self.config.resolved_model_path,
            "dataset": self.config.dataset_name,
            "clips_per_split": self.config.clips_per_split,
            "groups": ordered,
            "load_failures": load_failures,
            # A run passes when every group passed and nothing failed to load.
            "passed": bool(ordered) and all(g["passed"] for g in ordered) and not load_failures,
        }
