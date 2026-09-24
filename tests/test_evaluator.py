"""Offline tests for eval.evaluator.Evaluator (seam S5).

Runs `Evaluator(backend, config).evaluate(rows)` with fake `ModelBackend`s, tiny
synthetic WAVs and a temporary `output_dir`. Covers every row status, groups and
pass/fail, the `results.jsonl` and `summary.json` fields, the return value, and
smoke-run tolerance against a regular run raising.

Runnable directly (`python tests/test_evaluator.py`) or under pytest. Needs all
of `requirements.txt` (CPU `torch` is enough), and the `wer` metric from the
`evaluate` package.
"""
import io
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable, NoReturn

import numpy as np
import soundfile as sf

from eval.backends.base import InferenceRequest, ModelBackend
from eval.config import EvalConfig
from eval.evaluator import STATUSES, Evaluator
from uad_data.load_report import (
    LoadedRows, LoadFailure, LoadReport, RenderFailure, SplitReport,
)


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.float32), 16_000, format="WAV")
    return buffer.getvalue()


def _row(audio_path: str, split: str, caption: str, *, audio: bytes | None = None) -> dict:
    return {
        "audio": {"path": audio_path, "bytes": _wav_bytes() if audio is None else audio},
        "audio_path": audio_path,
        "split": split,
        "task": "caption",
        "originating_dataset": "Clotho",
        "system_instruction": "You are an audio captioner.",
        "prompt": "Describe the audio.",
        "output": f"The caption is: {caption}",
        "caption": caption,
    }


class EchoBackend(ModelBackend):
    """Predicts each row's reference, and records how many rows it saw."""

    def __init__(self):
        self.seen = 0

    def generate_batch(self, requests):
        self.seen += len(requests)
        return [r.ground_truth for r in requests]


class ScriptedBackend(ModelBackend):
    """Returns canned predictions in order; a `BackendError` value is raised instead."""

    class BackendError(RuntimeError):
        pass

    def __init__(self, predictions: Iterable[str | BaseException]) -> None:
        self.predictions = list(predictions)
        self.seen = 0

    def generate_batch(self, requests: list[InferenceRequest]) -> list[str]:
        out = []
        for _ in requests:
            nxt = self.predictions.pop(0)
            if isinstance(nxt, BaseException):
                raise nxt
            out.append(nxt)
        self.seen += len(requests)
        return out


ROWS = [
    _row("test/t2.wav", "test", "a cat meows"),
    _row("train/t0.wav", "train", "a dog barks"),
    _row("validation/t1.wav", "validation", "rain falls"),
]


@dataclass
class RunResult:
    """What one `_run` left behind."""
    summary: dict  # summary.json, as written
    records: list[dict]  # results.jsonl, one record per line
    returned: dict  # what `evaluate` returned
    backend: ModelBackend  # the backend the run used


def _run(
    rows: list[dict], backend: ModelBackend | None = None, **config: Any,
) -> RunResult:
    """Evaluate `rows` into a temp dir; give back summary, jsonl records, return value."""
    backend = EchoBackend() if backend is None else backend
    with tempfile.TemporaryDirectory() as output_dir:
        returned = Evaluator(backend, EvalConfig(
            model_choice="GEMMA-4", batch_size=2, output_dir=output_dir, **config,
        )).evaluate(rows)
        with open(os.path.join(output_dir, "results.jsonl"), encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        with open(os.path.join(output_dir, "summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
    return RunResult(summary, records, returned, backend)


def _group(summary: dict, dataset: str, split: str, task: str) -> dict:
    for group in summary["groups"]:
        key = (group["originating_dataset"], group["split"], group["task"])
        if key == (dataset, split, task):
            return group
    raise AssertionError(f"no group {dataset}/{split}/{task} in {summary['groups']}")


# ----------------------------------------------------------------------
# Rows in, rows out
# ----------------------------------------------------------------------

def test_every_loaded_row_is_evaluated() -> None:
    # clips_per_split caps clips in the loader; the evaluator must not cut rows again.
    run = _run(list(ROWS), clips_per_split=1)
    records, backend = run.records, run.backend

    assert backend.seen == 3, backend.seen
    assert len(records) == 3, records

    print("PASS: the evaluator evaluates every row it is given.")


def test_each_record_keeps_its_own_row_split() -> None:
    """A smoke run asks for every split, so a run-level split would mislabel rows."""
    records = _run(list(ROWS), clips_per_split=1).records

    assert [r["split"] for r in records] == ["test", "train", "validation"], records

    print("PASS: results.jsonl records each row's own split.")


# ----------------------------------------------------------------------
# Row statuses
# ----------------------------------------------------------------------

def test_a_returned_prediction_is_ok() -> None:
    records = _run([ROWS[0]], clips_per_split=1).records

    assert records[0]["status"] == "ok", records[0]
    assert records[0]["error"] is None, records[0]

    print("PASS: a non-empty prediction is ok.")


def test_a_whitespace_only_prediction_is_empty_output() -> None:
    backend = ScriptedBackend(["", "   \n\t "])
    records = _run(ROWS[:2], backend, clips_per_split=1).records

    assert [r["status"] for r in records] == ["empty_output", "empty_output"], records

    print("PASS: a whitespace-only prediction is empty_output.")


def test_undecodable_audio_is_an_audio_error_and_skips_the_model() -> None:
    rows = [ROWS[0], _row("test/bad.wav", "test", "silence", audio=b"not audio at all")]
    backend = ScriptedBackend(["a cat meows"])

    records = _run(rows, backend, clips_per_split=1).records

    assert backend.seen == 1, backend.seen  # the bad row never reached the model
    assert [r["status"] for r in records] == ["ok", "audio_error"], records
    bad = records[1]
    assert bad["prediction"] is None and bad["error"], bad

    print("PASS: undecodable audio becomes audio_error and is left out of its batch.")


def test_a_raising_backend_fails_its_whole_batch_and_the_run_goes_on() -> None:
    # batch_size=2: the first batch raises, the second must still be evaluated.
    backend = ScriptedBackend([ScriptedBackend.BackendError("boom"), "rain falls"])

    records = _run(list(ROWS), backend, clips_per_split=1).records

    assert [r["status"] for r in records] == ["model_error", "model_error", "ok"], records
    assert all("boom" in r["error"] for r in records[:2]), records

    print("PASS: a raising backend fails its batch, and the next batch still runs.")


def test_a_backend_that_returns_too_few_predictions_fails_its_batch() -> None:
    """A short return is the backend misbehaving, and a smoke run survives it."""
    class ShortBackend(ModelBackend):
        def generate_batch(self, requests: list[InferenceRequest]) -> list[str]:
            return ["only one"]  # two rows go in

    records = _run(ROWS[:2], ShortBackend(), clips_per_split=1).records

    assert [r["status"] for r in records] == ["model_error", "model_error"], records
    assert all(r["error"] for r in records), records

    print("PASS: a backend returning too few predictions fails its batch, not the run.")


def test_a_row_that_never_reached_the_model_keeps_its_prompt_in_the_record() -> None:
    """Triaging a decode failure needs the row's own text, not just its path."""
    rows = [_row("test/bad.wav", "test", "silence", audio=b"not audio at all")]

    records = _run(rows, clips_per_split=1).records

    record = records[0]
    assert record["status"] == "audio_error", record
    assert record["sys_inst"] == "You are an audio captioner.", record
    assert record["prompt"] == "Describe the audio.", record
    assert record["ground_truth"] == "The caption is: silence", record

    print("PASS: a row that never ran still records its prompt and ground truth.")


def test_a_metric_that_raises_does_not_take_the_run_down() -> None:
    """Metrics are information only, so one must never cost the run its output.

    `evaluate.load` reaches the Hub on a cold cache, and it runs per row inside
    the batch loop; before this the first asr row could abort everything and
    leave no summary.json behind.
    """
    from eval import metrics as metrics_module

    def _boom(*args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError("metric unavailable")

    original = metrics_module.metric_value, metrics_module.aggregate
    metrics_module.metric_value, metrics_module.aggregate = _boom, _boom
    try:
        run = _run(list(ROWS), clips_per_split=1)
        summary, records = run.summary, run.records
    finally:
        metrics_module.metric_value, metrics_module.aggregate = original

    assert [r["status"] for r in records] == ["ok", "ok", "ok"], records
    assert all(r["metric_value"] is None for r in records), records
    assert all(g["metric_value"] is None for g in summary["groups"]), summary
    assert summary["passed"] is True, summary

    print("PASS: a metric that raises costs the run its numbers, not its output.")


def test_a_row_with_no_audio_at_all_is_an_audio_error() -> None:
    """Not just undecodable bytes: a row missing the field entirely."""
    rows = [ROWS[0], {**_row("test/x.wav", "test", "silence"), "audio": None}]

    records = _run(rows, ScriptedBackend(["a cat meows"]), clips_per_split=1).records

    assert [r["status"] for r in records] == ["ok", "audio_error"], records

    print("PASS: a row with no audio becomes audio_error, not a dead run.")


def test_a_backend_returning_anything_but_strings_fails_its_batch() -> None:
    """The backend's whole return contract is checked in one place.

    A short list, a None in place of text, a number: each is the backend letting
    its batch down, and each has to become model_error rather than an exception
    escaping the loop and costing the run its summary.
    """
    class NoneBackend(ModelBackend):
        def generate_batch(self, requests: list[InferenceRequest]) -> list[Any]:
            return [None] * len(requests)

    class NumberBackend(ModelBackend):
        def generate_batch(self, requests: list[InferenceRequest]) -> list[Any]:
            return [1.0] * len(requests)

    for backend in (NoneBackend(), NumberBackend()):
        run = _run(ROWS[:2], backend, clips_per_split=1)
        summary, records = run.summary, run.records
        assert [r["status"] for r in records] == ["model_error", "model_error"], records
        assert all(r["error"] for r in records), records
        assert summary["groups"], summary  # the run still produced its summary

    print("PASS: a backend returning anything but strings fails its batch.")


def test_an_unavailable_metric_is_complained_about_once_per_run() -> None:
    """Otherwise a down Hub buries the GT/Pred output the run exists to produce."""
    import contextlib
    import io as _io

    from eval import metrics as metrics_module

    def _boom(*args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError("metric unavailable")

    original = metrics_module.metric_value, metrics_module.aggregate
    metrics_module.metric_value, metrics_module.aggregate = _boom, _boom
    chatter = _io.StringIO()
    try:
        with contextlib.redirect_stdout(chatter):
            _run(list(ROWS), clips_per_split=1)
    finally:
        metrics_module.metric_value, metrics_module.aggregate = original

    complaints = [line for line in chatter.getvalue().splitlines()
                  if "metric unavailable" in line]
    assert len(complaints) == 1, complaints

    print("PASS: an unavailable metric is complained about once, not per row.")


def test_a_metric_that_will_not_load_is_complained_about_once() -> None:
    """The first failed load and every remembered one after it are the same news.

    The test above raises one constant error, so it can't see this: the first
    load fails with the Hub's own error, later rows get the remembered reason,
    and if those two read differently the complaint prints twice.
    """
    import contextlib
    import io as _io

    from eval import metrics as metrics_module

    def _failing_load(name: str) -> NoReturn:
        raise ConnectionError("hub down")

    original = (metrics_module.hf_evaluate.load, metrics_module._wer_metric,
                metrics_module._wer_load_error)
    (metrics_module.hf_evaluate.load, metrics_module._wer_metric,
     metrics_module._wer_load_error) = (_failing_load, None, None)
    chatter = _io.StringIO()
    try:
        with contextlib.redirect_stdout(chatter):
            _run(list(ROWS), clips_per_split=1)
    finally:
        (metrics_module.hf_evaluate.load, metrics_module._wer_metric,
         metrics_module._wer_load_error) = original

    complaints = [line for line in chatter.getvalue().splitlines()
                  if "metric unavailable" in line]
    assert len(complaints) == 1, complaints
    assert "hub down" in complaints[0], complaints

    print("PASS: a metric that will not load is complained about once.")


def test_a_render_failure_keeps_the_utterance_it_failed_on() -> None:
    """One clip renders one row per utterance, so the index is what tells them apart."""
    rows = LoadedRows([], LoadReport(
        clips_per_split=1,
        splits=[SplitReport("MELD", "test", ["asr_timestamp_search"], clips_found=1)],
        render_failures=[
            RenderFailure("MELD", "test", "asr_timestamp_search", "test/d0.wav",
                          "KeyError: 'transcription'", utterance_index=0),
            RenderFailure("MELD", "test", "asr_timestamp_search", "test/d0.wav",
                          "KeyError: 'transcription'", utterance_index=3),
        ],
    ))

    records = _run(rows, clips_per_split=1).records

    assert [r["utterance_index"] for r in records] == [0, 3], records

    print("PASS: a render failure records the utterance it failed on.")


def test_a_row_that_failed_to_render_is_reported_from_the_load_report() -> None:
    rows = LoadedRows([ROWS[0]], LoadReport(
        clips_per_split=1,
        splits=[SplitReport("Clotho", "test", ["caption"], clips_found=2)],
        render_failures=[RenderFailure(
            "Clotho", "test", "caption", "test/t9.wav", "KeyError: 'caption'")],
    ))

    run = _run(rows, clips_per_split=1)
    summary, records = run.summary, run.records

    unrendered = [r for r in records if r["status"] == "render_error"]
    assert len(unrendered) == 1, records
    assert unrendered[0]["audio_path"] == "test/t9.wav", unrendered
    assert unrendered[0]["error"] == "KeyError: 'caption'", unrendered
    # It belongs to its group, which therefore fails.
    assert _group(summary, "Clotho", "test", "caption")["rows"] == 2, summary
    assert _group(summary, "Clotho", "test", "caption")["passed"] is False, summary

    print("PASS: render failures reach results.jsonl and fail their group.")


# ----------------------------------------------------------------------
# Groups
# ----------------------------------------------------------------------

def test_a_group_passes_only_when_every_row_is_ok() -> None:
    summary = _run([ROWS[0]], clips_per_split=1).summary
    assert _group(summary, "Clotho", "test", "caption")["passed"] is True, summary
    assert summary["passed"] is True, summary

    summary = _run([ROWS[0]], ScriptedBackend([""]), clips_per_split=1).summary
    assert _group(summary, "Clotho", "test", "caption")["passed"] is False, summary
    assert summary["passed"] is False, summary

    print("PASS: a group passes only when every row is ok.")


def test_a_group_with_no_rows_fails() -> None:
    """A selected split that yielded nothing still has its groups, and they fail."""
    rows = LoadedRows([], LoadReport(
        clips_per_split=5,
        splits=[SplitReport("Clotho", "test", ["caption", "asr"], clips_found=0)],
    ))

    run = _run(rows, clips_per_split=5)
    summary, records = run.summary, run.records

    assert records == [], records
    for task in ("caption", "asr"):
        group = _group(summary, "Clotho", "test", task)
        assert group["rows"] == 0 and group["passed"] is False, group
        assert group["clips_found"] == 0 and group["clips_per_split"] == 5, group
    assert summary["passed"] is False, summary

    print("PASS: a group with no rows exists and fails.")


def test_internal_datasets_that_failed_to_load_are_listed_and_fail_the_run() -> None:
    rows = LoadedRows(list(ROWS), LoadReport(
        clips_per_split=1,
        splits=[SplitReport("Clotho", split, ["caption"], clips_found=1)
                for split in ("test", "train", "validation")],
        load_failures=[LoadFailure("MELD", "unexpected end of data", rows_kept=0)],
    ))

    summary = _run(rows, clips_per_split=1).summary

    assert summary["load_failures"] == [
        {"dataset": "MELD", "error": "unexpected end of data", "rows_kept": 0}], summary
    assert all(g["passed"] for g in summary["groups"]), summary
    assert summary["passed"] is False, summary

    print("PASS: a failed load is listed in the summary and fails the run.")


# ----------------------------------------------------------------------
# Output files and return value
# ----------------------------------------------------------------------

def test_results_jsonl_keeps_todays_fields_and_adds_the_new_ones() -> None:
    # The prediction matches the plain caption, not the rendered output, so a
    # metric of 0.0 is itself evidence of which field the row is read against.
    records = _run([ROWS[0]], ScriptedBackend(["a cat meows"]), clips_per_split=1).records

    record = records[0]
    for field in ("index", "model_choice", "model", "dataset", "sys_inst", "prompt",
                  "ground_truth"):
        assert field in record, (field, record)
    assert record["ground_truth"] == "The caption is: a cat meows", record
    for field in ("originating_dataset", "split", "task", "audio_path", "status",
                  "error", "answer", "prediction", "metric", "metric_value"):
        assert field in record, (field, record)
    # The answer is the plain field, not the rendered output.
    assert record["answer"] == "a cat meows", record
    assert record["metric"] == "wer" and record["metric_value"] == 0.0, record
    assert "audio" not in record, record

    print("PASS: results.jsonl keeps today's fields and adds the decided ones.")


def test_summary_json_has_one_entry_per_group_with_the_decided_fields() -> None:
    rows = LoadedRows([ROWS[0]], LoadReport(
        clips_per_split=1,
        splits=[SplitReport("Clotho", "test", ["caption"], clips_found=1)]))

    summary = _run(rows, ScriptedBackend(["a cat meows"]), clips_per_split=1).summary

    assert summary["model_choice"] == "GEMMA-4", summary
    assert summary["clips_per_split"] == 1, summary
    group = _group(summary, "Clotho", "test", "caption")
    assert group["clips_found"] == 1 and group["clips_per_split"] == 1, group
    assert group["rows"] == 1, group
    assert group["statuses"] == {
        "ok": 1, "empty_output": 0, "render_error": 0, "audio_error": 0, "model_error": 0,
    }, group
    assert group["metric"] == "wer" and group["metric_value"] == 0.0, group
    assert group["passed"] is True, group

    print("PASS: summary.json has one entry per group with the decided fields.")


def test_a_groups_metric_covers_its_rows_with_a_prediction() -> None:
    """An empty prediction is a real miss; a row that never ran is left out."""
    rows = [_row("test/a.wav", "test", "one two"), _row("test/b.wav", "test", "three four")]
    backend = ScriptedBackend(["one two", ""])

    summary = _run(rows, backend, clips_per_split=1).summary

    # Corpus WER over both rows: two of four reference words wrong.
    assert _group(summary, "Clotho", "test", "caption")["metric_value"] == 0.5, summary

    print("PASS: a group's metric covers every row that has a prediction.")


def test_evaluate_returns_the_summary_plus_the_row_records() -> None:
    run = _run(list(ROWS), clips_per_split=1)
    records, returned = run.records, run.returned

    assert returned["rows"] == records, (returned["rows"], records)
    assert returned["groups"] == _run(list(ROWS), clips_per_split=1).summary["groups"]
    assert "wer" not in returned and "num_samples" not in returned, returned
    assert "predictions" not in returned and "references" not in returned, returned

    print("PASS: evaluate returns the summary plus the row records.")


def test_a_split_that_came_up_short_of_the_cap_is_warned_about() -> None:
    """The spec asks for a warning, and a `3/5` cell in an all-PASS table isn't one.

    An operator whose archive was truncated otherwise sees Overall: PASS and
    exit 0 with nothing calling it out.
    """
    import contextlib
    import io as _io

    from eval.evaluator import print_group_table

    summary = {
        "clips_per_split": 5,
        "groups": [
            {"originating_dataset": "Clotho", "split": "test", "task": "caption",
             "clips_found": 3, "clips_per_split": 5, "rows": 3,
             "statuses": {status: (3 if status == "ok" else 0) for status in STATUSES},
             "metric": "wer", "metric_value": 0.0, "passed": True},
            {"originating_dataset": "Clotho", "split": "train", "task": "caption",
             "clips_found": 5, "clips_per_split": 5, "rows": 5,
             "statuses": {status: (5 if status == "ok" else 0) for status in STATUSES},
             "metric": "wer", "metric_value": 0.0, "passed": True},
        ],
        "load_failures": [],
        "passed": True,
    }

    printed = _io.StringIO()
    with contextlib.redirect_stdout(printed):
        print_group_table(summary)
    warnings = [line for line in printed.getvalue().splitlines() if "WARNING" in line]

    assert len(warnings) == 1, printed.getvalue()
    assert "Clotho/test" in warnings[0] and "3" in warnings[0], warnings

    print("PASS: a split that came up short of the cap is warned about.")


def test_the_summary_can_be_printed_without_an_output_dir() -> None:
    """The notebook has no output_dir, and still wants the per-group table."""
    from eval.evaluator import print_group_table

    returned = Evaluator(EchoBackend(), EvalConfig(
        model_choice="GEMMA-4", batch_size=2, clips_per_split=1,
    )).evaluate(list(ROWS))

    assert returned["rows"] and returned["groups"], returned
    print_group_table(returned)  # must not raise

    print("PASS: the per-group table is printable without an output_dir.")


# ----------------------------------------------------------------------
# Regular runs stop at the first error
# ----------------------------------------------------------------------

def test_a_regular_run_raises_on_undecodable_audio() -> None:
    rows = [_row("test/bad.wav", "test", "silence", audio=b"not audio at all")]

    try:
        _run(rows)
    except Exception as error:  # noqa: BLE001 - any decode error is the point
        assert not isinstance(error, AssertionError), error
    else:
        raise AssertionError("a regular run must raise on undecodable audio")

    print("PASS: a regular run raises on undecodable audio.")


def test_a_regular_run_raises_when_the_backend_raises() -> None:
    backend = ScriptedBackend([ScriptedBackend.BackendError("boom")])

    try:
        _run([ROWS[0]], backend)
    except ScriptedBackend.BackendError:
        pass
    else:
        raise AssertionError("a regular run must raise when the backend raises")

    print("PASS: a regular run raises when the backend raises.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
