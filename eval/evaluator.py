"""Batched evaluation of a model backend over a run's rows.

`Evaluator.evaluate(dataset)` gives every row of it exactly one status, groups the rows by
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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Iterable, List, Optional, TextIO

from uad_data.audio_utils import preprocess_audio
from uad_data.load_report import LoadReport, RenderFailure
from . import metrics
from .backends.base import InferenceRequest, ModelBackend
from .config import EvalConfig


# Every way a row can end. `ok` is the only one a passing group may contain.
STATUSES = ("ok", "empty_output", "render_error", "audio_error", "model_error")

# A group: its internal dataset, split and task.
_GroupKey = tuple[str, str, str]
# One row that has a prediction, as its group's metric is computed over it.
_Predicted = tuple[dict, str]


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
    _print_aligned([columns, *(_table_line(group) for group in summary["groups"])])

    _print_short_split_warnings(summary["groups"])

    for failure in summary["load_failures"]:
        print(f"  FAILED TO LOAD {failure['dataset']}: {failure['error']} "
              f"({failure['rows_kept']} rows kept)")

    print(f"\nOverall: {'PASS' if summary['passed'] else 'FAIL'}")


def _table_line(group: dict) -> list[str]:
    """One group's cells in the group table, in `print_group_table`'s column order."""
    return [
        f"{group['originating_dataset']}/{group['split']}/{group['task']}",
        _clips_cell(group),
        str(group["rows"]),
        *(str(group["statuses"][status]) for status in STATUSES),
        _metric_cell(group),
        "PASS" if group["passed"] else "FAIL",
    ]


def _clips_cell(group: dict) -> str:
    found, cap = group["clips_found"], group["clips_per_split"]
    if found is None:
        return "-"
    return str(found) if cap is None else f"{found}/{cap}"


def _metric_cell(group: dict) -> str:
    value = group["metric_value"]
    if group["metric"] is None or value is None:
        return "-"
    return f"{group['metric']} {value:.4f}"


def _print_aligned(lines: list[list[str]]) -> None:
    """Print rows of cells with each column padded to its widest cell."""
    widths = [max(len(line[i]) for line in lines) for i in range(len(lines[0]))]
    for line in lines:
        print("  " + "  ".join(cell.ljust(width) for cell, width in zip(line, widths)))


def _print_short_split_warnings(groups: list[dict]) -> None:
    """Warn once per split that came up short of the cap.

    A split that came up short of the cap is a warning, not a failure: the
    groups can still pass and the run can still exit 0, so without saying so
    a truncated archive reads as an ordinary green run.
    """
    short = {
        (group["originating_dataset"], group["split"]):
            (group["clips_found"], group["clips_per_split"])
        for group in groups
        if group["clips_per_split"] is not None
        and group["clips_found"] is not None
        and group["clips_found"] < group["clips_per_split"]
    }
    for (dataset, split), (found, wanted) in short.items():
        print(f"  WARNING {dataset}/{split} found {found} clips, "
              f"fewer than the {wanted} asked for")


@dataclass
class _RunLog:
    """What a run has produced so far, and where its records are written."""
    # Opened once per run; each record flushes to it so results survive a mid-run crash.
    jsonl_file: Optional[TextIO]
    records: List[dict] = field(default_factory=list)
    # group key -> the (row, prediction) pairs its metric is computed over.
    predicted: dict[_GroupKey, list[_Predicted]] = field(default_factory=dict)


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
        # What this run has already said about unavailable metrics.
        self._metric_complaints: set[str] = set()

    def _metric_or_none(
        self, metric_fn: Callable[..., Optional[float]], *args: Any,
    ) -> Optional[float]:
        """A preliminary metric's value, or None if computing it failed.

        Metrics are information only -- "a group passes or fails on its row
        statuses, never on these numbers" -- so one must never cost a run its
        `results.jsonl` and `summary.json`. `evaluate.load` reaches the Hub on a
        cold cache, and this runs per row inside the batch loop, so an offline
        runner or a Hub blip would otherwise abort a smoke run built to tolerate
        far worse.

        Each distinct complaint is made once per run: repeated per row it would
        bury the GT/Pred output the run exists to produce, but latching on the
        first would hide a second, different fault behind it -- a rule raising
        `TypeError` on an odd answer field, say, after the Hub had already failed.
        """
        try:
            return metric_fn(*args)
        except Exception as error:
            described = _describe(error)
            if described not in self._metric_complaints:
                self._metric_complaints.add(described)
                print(f"  metric unavailable, so this run reports none: {described}")
            return None

    def evaluate(self, dataset: Iterable[dict]) -> dict:
        rows = list(dataset)
        # Each call is its own run: one notebook kernel evaluates many times, so
        # a metric that failed to load earlier gets another chance here, and this
        # run says so once if it fails again.
        metrics.forget_failed_load()
        self._metric_complaints.clear()
        # `load_uad_dataset` returns rows carrying their load report. A caller
        # that hands over a plain list (the tests, the notebook) gets an empty one.
        report: LoadReport = getattr(dataset, "report", None) or LoadReport()

        print(f"Evaluating {len(rows)} rows (batch_size={self.config.batch_size})")

        log = _RunLog(self._open_jsonl(self.config.output_dir))
        try:
            self._evaluate_rows(rows, log)
            # Rows the loader could never render reach the report, not the model.
            for failure in report.render_failures:
                self._log(log, self._render_failure_record(len(log.records), failure))
        finally:
            if log.jsonl_file:
                log.jsonl_file.close()

        summary = self._summarise(log.records, log.predicted, report)
        summary["rows"] = log.records

        print_group_table(summary)
        self._save_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_rows(self, rows: List[dict], log: _RunLog) -> None:
        """Run every row through the model, batch by batch, logging each record."""
        preprocess_fn = partial(
            preprocess_audio,
            target_sr=self.config.target_sr,
            max_seconds=self.config.max_audio_seconds,
        )
        # One executor for the whole run, reused by every batch's preprocessing.
        with ThreadPoolExecutor(max_workers=self.config.num_preprocessing_workers) as executor:
            for batch in _batched(rows, self.config.batch_size):
                self._evaluate_batch(batch, preprocess_fn, executor, log, total=len(rows))

    def _evaluate_batch(
        self,
        batch: List[dict],
        preprocess_fn: Callable[[bytes], Any],
        executor: ThreadPoolExecutor,
        log: _RunLog,
        *,
        total: int,
    ) -> None:
        # Parallel audio decode + resample, then one GPU forward pass.
        prepared = self._preprocess_batch(batch, preprocess_fn, executor)
        predictions, batch_error = self._predict(
            [request for request in prepared if isinstance(request, InferenceRequest)])

        for row, request in zip(batch, prepared):
            record = self._batch_record(
                len(log.records), row, request, predictions, batch_error)
            if record["prediction"] is not None:
                log.predicted.setdefault(self._key(record), []).append(
                    (row, record["prediction"]))
            self._print_row(record, total)
            self._log(log, record)

    def _batch_record(
        self,
        index: int,
        row: dict,
        request: InferenceRequest | BaseException,
        predictions: Optional[List[str]],
        batch_error: Optional[str],
    ) -> dict:
        """One batch row's record. Takes this row's prediction off `predictions`."""
        if not isinstance(request, InferenceRequest):
            # The row never reached the model, but its prompt and ground truth
            # are its own, and triage wants them.
            return self._record(index, row, status="audio_error", error=_describe(request))
        if predictions is None:
            return self._record(
                index, row, request=request, status="model_error", error=batch_error)
        prediction = predictions.pop(0)
        status = "ok" if prediction.strip() else "empty_output"
        return self._record(
            index, row, request=request, status=status, prediction=prediction)

    def _render_failure_record(self, index: int, failure: RenderFailure) -> dict:
        """The record of a row the loader could never render."""
        return self._record(index, {
            "originating_dataset": failure.dataset,
            "split": failure.split,
            "task": failure.task,
            "audio_path": failure.audio_path,
            "utterance_index": failure.utterance_index,
        }, status="render_error", error=failure.error)

    def _log(self, log: _RunLog, record: dict) -> None:
        log.records.append(record)
        self._write(log.jsonl_file, record)

    def _save_summary(self, summary: dict) -> None:
        """Write `summary.json`, everything but the row records, when there is an output_dir."""
        if not self.config.output_dir:
            return
        summary_path = os.path.join(self.config.output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in summary.items() if k != "rows"}, f, indent=2)
        print(f"Results saved → {self.config.output_dir}/")

    @staticmethod
    def _key(record: dict) -> _GroupKey:
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
            # asr_timestamp_search renders one row per utterance, so without this
            # two failures of one clip are the same line twice. None elsewhere.
            "utterance_index": row.get("utterance_index"),
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
                None if prediction is None else self._metric_or_none(metrics.metric_value, row, prediction)),
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
            _check_predictions(predictions, requests)
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
    def _write(jsonl_file: Optional[TextIO], record: dict) -> None:
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
        batch: List[dict],
        preprocess_fn: Callable[[bytes], Any],
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
        return [self._request(row, future) for row, future in zip(batch, futures)]

    def _request(self, row: dict, future: Future) -> InferenceRequest | BaseException:
        """One row's request once its audio is decoded, or the decode's exception."""
        try:
            audio_array = future.result()
        except Exception as error:
            if not self.config.is_smoke_run:
                raise  # a regular run stops at the first error
            return error
        return InferenceRequest(
            audio_bytes=row["audio"]["bytes"],  # read once the decode succeeded
            audio_array=audio_array,
            sys_inst=(row.get("system_instruction") or "").strip(),
            prompt_text=(row.get("prompt") or "").strip(),
            ground_truth=(row.get("output") or "").strip(),
            task=(row.get("task") or "").strip(),
        )

    def _summarise(
        self,
        records: List[dict],
        predicted: dict[_GroupKey, list[_Predicted]],
        report: LoadReport,
    ) -> dict:
        """One entry per group, plus the failed loads and the run's overall result."""
        ordered = self._groups(records, predicted, report)
        load_failures = [
            {"dataset": f.dataset, "error": f.error, "rows_kept": f.rows_kept}
            for f in report.load_failures
        ]
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

    def _groups(
        self,
        records: List[dict],
        predicted: dict[_GroupKey, list[_Predicted]],
        report: LoadReport,
    ) -> List[dict]:
        """Every group's `summary.json` entry, counted, measured and judged."""
        groups: dict[_GroupKey, dict] = {}

        # The report names every selected split's groups, so a split that yielded
        # no clips still has groups -- which fail, having no rows.
        for split in report.splits:
            for task in split.tasks:
                key = (split.dataset, split.split, task)
                groups.setdefault(key, self._new_group(key, split.clips_found))

        for record in records:
            key = self._key(record)
            group = groups.setdefault(key, self._new_group(key, None))
            group["rows"] += 1
            group["statuses"][record["status"]] += 1

        for key, group in groups.items():
            group["metric_value"] = self._metric_or_none(
                metrics.aggregate, group["task"], predicted.get(key, []))
            group["passed"] = group["rows"] > 0 and group["statuses"]["ok"] == group["rows"]

        return list(groups.values())

    def _new_group(self, key: _GroupKey, clips_found: Optional[int]) -> dict:
        """A group's entry before any of its rows are counted."""
        dataset, split, task = key
        return {
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
        }


def _check_predictions(predictions: List[Any], requests: List[InferenceRequest]) -> None:
    """Raise unless the backend gave back one string per request."""
    if len(predictions) != len(requests):
        raise ValueError(
            f"backend returned {len(predictions)} predictions "
            f"for {len(requests)} requests")
    # `None` is itself one of the wrong values, so the search can't use it as its
    # "found nothing" marker.
    wrong = [p for p in predictions if not isinstance(p, str)][:1]
    if wrong:
        raise TypeError(
            f"backend returned {type(wrong[0]).__name__}, "
            f"not text: {wrong[0]!r:.60}")
