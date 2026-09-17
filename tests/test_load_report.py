"""Offline tests for the loader's load report and smoke-run failure tolerance.

`load_uad_dataset` returns its rows as a list carrying a `.report`: clips found
per (internal dataset, selected split) with that split's tasks, internal
datasets that failed to load, and rows that failed to render. A smoke run
(`clips_per_split` set) records those failures and carries on; a regular run
raises. Splits that end short of the cap log a warning.

Reuses the synthetic archives and Hub fakes of `test_loader_splits.py`.
Runnable directly (`python tests/test_load_report.py`) or under pytest. Only
requires `datasets`, `jinja2`, `huggingface_hub`.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import test_loader_splits as fx  # noqa: E402


def test_rows_are_a_list_with_a_report_of_clips_per_split() -> None:
    config = {"name": "Clotho", "datasets": [
        {"name": "Clotho", "tasks": ["caption"], "splits": ["train", "test"]}]}
    rows, _ = fx._load(fx.CLOTHO, config, split="all", clips_per_split=2)

    assert isinstance(rows, list), type(rows)
    report = rows.report
    assert report.clips_per_split == 2, report
    assert [(s.dataset, s.split, s.tasks, s.clips_found) for s in report.splits] == [
        ("Clotho", "train", ["caption"], 2),
        ("Clotho", "test", ["caption"], 2),
    ], report.splits
    assert report.load_failures == [], report.load_failures
    assert report.render_failures == [], report.render_failures

    print("PASS: rows carry a report of clips found per selected split.")


EMNS_MEMBERS = [f"emns/{i}.wav" for i in range(4)]
EMNS_CONFIG = {"name": "EMNS", "datasets": [
    {"name": "EMNS", "tasks": ["classification", "asr"], "splits": ["train"]}]}


def _emns_without_category(*paths: str) -> dict:
    """EMNS-like data whose named clips have no `category`, so classification can't render."""
    records = []
    for p in EMNS_MEMBERS:
        record = fx._emns_record(p)
        if p in paths:
            del record["category"]
        records.append(record)
    return {"EMNS": {"members": EMNS_MEMBERS, "splits": {"train": records}}}


def test_smoke_run_records_render_failures_and_still_counts_the_clip() -> None:
    rows, _ = fx._load(
        _emns_without_category("emns/0.wav"), EMNS_CONFIG, split="train", clips_per_split=2)

    # emns/0 keeps its asr row and still counts, so the cap stops at emns/1.
    assert sorted((r["audio_path"], r["task"]) for r in rows) == [
        ("emns/0.wav", "asr"),
        ("emns/1.wav", "asr"), ("emns/1.wav", "classification"),
    ], rows
    [failure] = rows.report.render_failures
    assert (failure.dataset, failure.split, failure.task, failure.audio_path) == (
        "EMNS", "train", "classification", "emns/0.wav"), failure
    assert "category" in failure.error, failure.error
    assert rows.report.splits[0].clips_found == 2, rows.report.splits

    with _Warnings() as caught:
        fx._load(_emns_without_category("emns/0.wav"), EMNS_CONFIG,
                 split="train", clips_per_split=2)
    assert any("emns/0.wav" in m and "classification" in m and "category" in m
               for m in caught.messages), caught.messages

    print("PASS: a smoke run records and logs a render failure and still counts its clip.")


def test_smoke_run_records_a_task_with_no_prompt_templates_as_render_failures() -> None:
    # The fixture has no commonsense prompt templates.
    config = {"name": "Clotho", "datasets": [
        {"name": "Clotho", "tasks": ["caption", "commonsense"], "splits": ["test"]}]}
    rows, _ = fx._load(fx.CLOTHO, config, split="test", clips_per_split=2)

    assert fx._clips(rows) == [("test", "test/t0.wav"), ("test", "test/t1.wav")], rows
    assert {r["task"] for r in rows} == {"caption"}, rows
    assert [(f.task, f.audio_path) for f in rows.report.render_failures] == [
        ("commonsense", "test/t0.wav"), ("commonsense", "test/t1.wav")
    ], rows.report.render_failures

    # A clip whose only task can't render still counts: the cap stops the read.
    config["datasets"][0]["tasks"] = ["commonsense"]
    rows, _ = fx._load(fx.CLOTHO, config, split="test", clips_per_split=2)
    assert rows == [], rows
    assert rows.report.splits[0].clips_found == 2, rows.report.splits
    assert len(rows.report.render_failures) == 2, rows.report.render_failures

    print("PASS: a task with no prompt templates fails each clip's render, and the clip counts.")


def test_a_filtered_out_clip_does_not_count_when_its_task_has_no_prompt_templates() -> None:
    # _RejectClipsFilter drops t0 and t1; the fixture has no commonsense prompt templates.
    fx.filters.FILTER_REGISTRY["reject_clips"] = fx._RejectClipsFilter
    try:
        members = [f"test/t{i}.wav" for i in range(4)]
        datasets = {"Clotho": {
            "members": members,
            "splits": {"test": [fx._clotho_record(p) for p in members]},
        }}
        config = {"name": "Clotho", "row_filter": "reject_clips", "datasets": [
            {"name": "Clotho", "tasks": ["commonsense"], "splits": ["test"]}]}
        rows, _ = fx._load(datasets, config, split="test", clips_per_split=2)
    finally:
        del fx.filters.FILTER_REGISTRY["reject_clips"]

    assert [f.audio_path for f in rows.report.render_failures] == [
        "test/t2.wav", "test/t3.wav"], rows.report.render_failures
    assert rows.report.splits[0].clips_found == 2, rows.report.splits

    print("PASS: a clip the row filter drops neither counts nor fails to render.")


def test_regular_run_raises_on_a_task_with_no_prompt_templates_even_if_filtered() -> None:
    # The random filter rejects about half the rows, so a regular run must raise
    # before it asks the filter.
    config = {"name": "Clotho", "row_filter": "random", "datasets": [
        {"name": "Clotho", "tasks": ["commonsense"], "splits": ["test"]}]}
    try:
        rows, _ = fx._load(fx.CLOTHO, config, split="test")
    except RuntimeError as e:
        assert "commonsense" in str(e).lower(), e
    else:
        raise AssertionError(f"expected RuntimeError, got {len(rows)} rows")

    class _Raising(fx.filters.RowFilter):
        def include_row(self, row) -> bool:
            raise AssertionError("the filter ran before the template error")

    fx.filters.FILTER_REGISTRY["raising"] = _Raising
    try:
        fx._load(fx.CLOTHO, {**config, "row_filter": "raising"}, split="test")
    except RuntimeError as e:
        assert "commonsense" in str(e).lower(), e
    else:
        raise AssertionError("expected RuntimeError")
    finally:
        del fx.filters.FILTER_REGISTRY["raising"]

    print("PASS: a regular run raises on missing prompt templates before the filter runs.")


def test_regular_run_raises_on_a_render_failure() -> None:
    try:
        rows, _ = fx._load(_emns_without_category("emns/0.wav"), EMNS_CONFIG, split="train")
    except KeyError as e:
        assert "category" in str(e), e
    else:
        raise AssertionError(f"expected KeyError, got {len(rows)} rows")

    print("PASS: a regular run raises on a render failure.")


class _MissingFilesHub(fx._FakeHub):
    """A fake Hub on which the files in `missing` don't exist."""

    def __init__(self, fixture: dict, missing: set[str]):
        super().__init__(fixture)
        self.missing = missing

    def download_file(self, path_or_url, **kwargs):
        if os.path.basename(path_or_url) in self.missing:
            raise FileNotFoundError(f"{path_or_url} is not on the Hub")
        return super().download_file(path_or_url, **kwargs)


def _load_broken(datasets: dict, config: dict, *, truncate: dict[str, float] | None = None,
                 missing: set[str] | None = None, **kwargs):
    """Like fx._load, but cuts each archive in `truncate` to that fraction of its
    bytes, and serves no file named in `missing`."""
    with fx.tempfile.TemporaryDirectory() as root:
        fixture = fx._build_fixture(root, datasets, config)
        for base, fraction in (truncate or {}).items():
            path = fixture["files"][base]
            size = os.path.getsize(path)
            with open(path, "r+b") as f:
                f.truncate(int(size * fraction))
        fake = _MissingFilesHub(fixture, missing or set())
        with fx._patched_hub(fake):
            return fx.loader.load_uad_dataset(
                json_config_path=fixture["config_path"], token=None, **kwargs)


BROKEN_THEN_FINE = {
    **fx.CLOTHO,
    "EMNS": {"members": EMNS_MEMBERS,
             "splits": {"train": [fx._emns_record(p) for p in EMNS_MEMBERS]}},
}
BROKEN_THEN_FINE_CONFIG = {"name": "mixed", "datasets": [
    {"name": "EMNS", "tasks": ["asr"], "splits": ["train"]},
    {"name": "Clotho", "tasks": ["caption"], "splits": ["test"]},
]}


def test_smoke_run_keeps_the_rows_before_a_truncated_archive_and_moves_on() -> None:
    for stream in (None, False):
        # Four 64 KiB clips: the first two survive a cut at 60%.
        rows = _load_broken(
            BROKEN_THEN_FINE, BROKEN_THEN_FINE_CONFIG, truncate={"EMNS.tar.gz": 0.6},
            split="all", clips_per_split=4, stream=stream)

        emns = [r["audio_path"] for r in rows if r["originating_dataset"] == "EMNS"]
        assert emns == ["emns/0.wav", "emns/1.wav"], (stream, emns)
        assert fx._clips(r for r in rows if r["originating_dataset"] == "Clotho") == [
            ("test", f"test/t{i}.wav") for i in range(3)], (stream, rows)
        [failure] = rows.report.load_failures
        assert (failure.dataset, failure.rows_kept) == ("EMNS", 2), (stream, failure)
        assert failure.error, failure
        assert [(s.dataset, s.clips_found) for s in rows.report.splits] == [
            ("EMNS", 2), ("Clotho", 3)], (stream, rows.report.splits)

    print("PASS: a smoke run keeps a truncated archive's rows and loads the next dataset.")


def test_smoke_run_records_a_missing_metadata_file_and_moves_on() -> None:
    rows = _load_broken(
        BROKEN_THEN_FINE, BROKEN_THEN_FINE_CONFIG, missing={"EMNS_train.json"},
        split="all", clips_per_split=2)

    assert {r["originating_dataset"] for r in rows} == {"Clotho"}, rows
    [failure] = rows.report.load_failures
    assert (failure.dataset, failure.rows_kept) == ("EMNS", 0), failure
    assert "EMNS_train.json" in failure.error, failure
    # The failed dataset's split is still reported, with no clips.
    assert [(s.dataset, s.split, s.clips_found) for s in rows.report.splits] == [
        ("EMNS", "train", 0), ("Clotho", "test", 2)], rows.report.splits

    print("PASS: a smoke run records a missing metadata file and loads the next dataset.")


def test_regular_run_raises_on_a_truncated_archive_or_missing_metadata() -> None:
    for broken, expected in [
        ({"truncate": {"EMNS.tar.gz": 0.6}}, fx.tarfile.ReadError),
        ({"missing": {"EMNS_train.json"}}, FileNotFoundError),
    ]:
        try:
            rows = _load_broken(
                BROKEN_THEN_FINE, BROKEN_THEN_FINE_CONFIG, split="all", **broken)
        except expected:
            pass
        else:
            raise AssertionError(f"{broken}: expected a raise, got {len(rows)} rows")

    print("PASS: a regular run raises on a truncated archive or a missing metadata file.")


class _Warnings(logging.Handler):
    """Collects the loader's warnings while installed."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record) -> None:
        self.messages.append(record.getMessage())

    def __enter__(self):
        logging.getLogger(fx.loader.__name__).addHandler(self)
        return self

    def __exit__(self, *exc) -> None:
        logging.getLogger(fx.loader.__name__).removeHandler(self)


def test_a_split_short_of_the_cap_gets_a_warning() -> None:
    config = {"name": "Clotho", "datasets": [
        {"name": "Clotho", "tasks": ["caption"], "splits": ["train", "test"]}]}
    datasets = {"Clotho": {
        "members": fx.CLOTHO_MEMBERS,
        "splits": {
            "train": [fx._clotho_record(p) for p in fx.CLOTHO_MEMBERS if p.startswith("train/")],
            "test": [fx._clotho_record("test/t0.wav")],
        },
    }}
    with _Warnings() as caught:
        rows, _ = fx._load(datasets, config, split="all", clips_per_split=3)
    short = [m for m in caught.messages if "Clotho" in m]
    assert len(short) == 1, caught.messages
    assert "test" in short[0] and "1" in short[0] and "3" in short[0], short

    # Without a cap, nothing is short.
    with _Warnings() as caught:
        fx._load(datasets, config, split="all")
    assert caught.messages == [], caught.messages

    # A failed load's splits are short too.
    with _Warnings() as caught:
        _load_broken(BROKEN_THEN_FINE, BROKEN_THEN_FINE_CONFIG,
                     missing={"EMNS_train.json"}, split="all", clips_per_split=2)
    assert any("EMNS" in m and "train" in m and "0" in m for m in caught.messages
               if "Failed to load" not in m), caught.messages

    print("PASS: a split with fewer clips than the cap logs a warning.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
