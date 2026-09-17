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
import sys
import tempfile

import numpy as np
import soundfile as sf

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.backends.base import ModelBackend  # noqa: E402
from eval.config import EvalConfig  # noqa: E402
from eval.evaluator import Evaluator  # noqa: E402
from uad_data.load_report import (  # noqa: E402
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

    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.seen = 0

    def generate_batch(self, requests):
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


def _report(**kwargs) -> LoadReport:
    return LoadReport(**kwargs)


def _run(rows, backend=None, **config) -> tuple[dict, list[dict], dict, object]:
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
    return summary, records, returned, backend


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
    _, records, _, backend = _run(list(ROWS), clips_per_split=1)

    assert backend.seen == 3, backend.seen
    assert len(records) == 3, records

    print("PASS: the evaluator evaluates every row it is given.")


def test_each_record_keeps_its_own_row_split() -> None:
    """A smoke run asks for every split, so a run-level split would mislabel rows."""
    _, records, _, _ = _run(list(ROWS), clips_per_split=1)

    assert [r["split"] for r in records] == ["test", "train", "validation"], records

    print("PASS: results.jsonl records each row's own split.")


# ----------------------------------------------------------------------
# Row statuses
# ----------------------------------------------------------------------

def test_a_returned_prediction_is_ok() -> None:
    _, records, _, _ = _run([ROWS[0]], clips_per_split=1)

    assert records[0]["status"] == "ok", records[0]
    assert records[0]["error"] is None, records[0]

    print("PASS: a non-empty prediction is ok.")


def test_a_whitespace_only_prediction_is_empty_output() -> None:
    backend = ScriptedBackend(["", "   \n\t "])
    _, records, _, _ = _run(ROWS[:2], backend, clips_per_split=1)

    assert [r["status"] for r in records] == ["empty_output", "empty_output"], records

    print("PASS: a whitespace-only prediction is empty_output.")


def test_undecodable_audio_is_an_audio_error_and_skips_the_model() -> None:
    rows = [ROWS[0], _row("test/bad.wav", "test", "silence", audio=b"not audio at all")]
    backend = ScriptedBackend(["a cat meows"])

    _, records, _, _ = _run(rows, backend, clips_per_split=1)

    assert backend.seen == 1, backend.seen  # the bad row never reached the model
    assert [r["status"] for r in records] == ["ok", "audio_error"], records
    bad = records[1]
    assert bad["prediction"] is None and bad["error"], bad

    print("PASS: undecodable audio becomes audio_error and is left out of its batch.")


def test_a_raising_backend_fails_its_whole_batch_and_the_run_goes_on() -> None:
    # batch_size=2: the first batch raises, the second must still be evaluated.
    backend = ScriptedBackend([ScriptedBackend.BackendError("boom"), "rain falls"])

    _, records, _, _ = _run(list(ROWS), backend, clips_per_split=1)

    assert [r["status"] for r in records] == ["model_error", "model_error", "ok"], records
    assert all("boom" in r["error"] for r in records[:2]), records

    print("PASS: a raising backend fails its batch, and the next batch still runs.")


def test_a_backend_that_returns_too_few_predictions_fails_its_batch() -> None:
    """A short return is the backend misbehaving, and a smoke run survives it."""
    class ShortBackend(ModelBackend):
        def generate_batch(self, requests):
            return ["only one"]  # two rows go in

    _, records, _, _ = _run(ROWS[:2], ShortBackend(), clips_per_split=1)

    assert [r["status"] for r in records] == ["model_error", "model_error"], records
    assert all(r["error"] for r in records), records

    print("PASS: a backend returning too few predictions fails its batch, not the run.")


def test_a_row_that_never_reached_the_model_keeps_its_prompt_in_the_record() -> None:
    """Triaging a decode failure needs the row's own text, not just its path."""
    rows = [_row("test/bad.wav", "test", "silence", audio=b"not audio at all")]

    _, records, _, _ = _run(rows, clips_per_split=1)

    record = records[0]
    assert record["status"] == "audio_error", record
    assert record["sys_inst"] == "You are an audio captioner.", record
    assert record["prompt"] == "Describe the audio.", record
    assert record["ground_truth"] == "The caption is: silence", record

    print("PASS: a row that never ran still records its prompt and ground truth.")


def test_a_row_that_failed_to_render_is_reported_from_the_load_report() -> None:
    rows = LoadedRows([ROWS[0]], _report(
        clips_per_split=1,
        splits=[SplitReport("Clotho", "test", ["caption"], clips_found=2)],
        render_failures=[RenderFailure(
            "Clotho", "test", "caption", "test/t9.wav", "KeyError: 'caption'")],
    ))

    summary, records, _, _ = _run(rows, clips_per_split=1)

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
    summary, _, _, _ = _run([ROWS[0]], clips_per_split=1)
    assert _group(summary, "Clotho", "test", "caption")["passed"] is True, summary
    assert summary["passed"] is True, summary

    summary, _, _, _ = _run([ROWS[0]], ScriptedBackend([""]), clips_per_split=1)
    assert _group(summary, "Clotho", "test", "caption")["passed"] is False, summary
    assert summary["passed"] is False, summary

    print("PASS: a group passes only when every row is ok.")


def test_a_group_with_no_rows_fails() -> None:
    """A selected split that yielded nothing still has its groups, and they fail."""
    rows = LoadedRows([], _report(
        clips_per_split=5,
        splits=[SplitReport("Clotho", "test", ["caption", "asr"], clips_found=0)],
    ))

    summary, records, _, _ = _run(rows, clips_per_split=5)

    assert records == [], records
    for task in ("caption", "asr"):
        group = _group(summary, "Clotho", "test", task)
        assert group["rows"] == 0 and group["passed"] is False, group
        assert group["clips_found"] == 0 and group["clips_per_split"] == 5, group
    assert summary["passed"] is False, summary

    print("PASS: a group with no rows exists and fails.")


def test_internal_datasets_that_failed_to_load_are_listed_and_fail_the_run() -> None:
    rows = LoadedRows(list(ROWS), _report(
        clips_per_split=1,
        splits=[SplitReport("Clotho", split, ["caption"], clips_found=1)
                for split in ("test", "train", "validation")],
        load_failures=[LoadFailure("MELD", "unexpected end of data", rows_kept=0)],
    ))

    summary, _, _, _ = _run(rows, clips_per_split=1)

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
    _, records, _, _ = _run([ROWS[0]], ScriptedBackend(["a cat meows"]), clips_per_split=1)

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
    rows = LoadedRows([ROWS[0]], _report(
        clips_per_split=1,
        splits=[SplitReport("Clotho", "test", ["caption"], clips_found=1)]))

    summary, _, _, _ = _run(rows, ScriptedBackend(["a cat meows"]), clips_per_split=1)

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

    summary, _, _, _ = _run(rows, backend, clips_per_split=1)

    # Corpus WER over both rows: two of four reference words wrong.
    assert _group(summary, "Clotho", "test", "caption")["metric_value"] == 0.5, summary

    print("PASS: a group's metric covers every row that has a prediction.")


def test_evaluate_returns_the_summary_plus_the_row_records() -> None:
    _, records, returned, _ = _run(list(ROWS), clips_per_split=1)

    assert returned["rows"] == records, (returned["rows"], records)
    assert returned["groups"] == _run(list(ROWS), clips_per_split=1)[0]["groups"]
    assert "wer" not in returned and "num_samples" not in returned, returned
    assert "predictions" not in returned and "references" not in returned, returned

    print("PASS: evaluate returns the summary plus the row records.")


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
