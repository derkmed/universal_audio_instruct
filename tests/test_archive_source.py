"""Offline tests for where the loader reads each internal dataset's archive from.

A capped run whose row filter is `all_pass` reads a fresh smoke archive,
`smoke/<name>.tar.gz`, when one exists with at least n clips per split. Anything
else streams the full archive, with a log line; an explicit `stream` bypasses
smoke archives, and a regular run downloads the full archive. A read error the
manifest records is a failed load only when a selected split ends short.

Smoke archives here are built from the fixture's full archives by the builder
itself, and served with a manifest by the Hub fake of `test_loader_splits.py`.
Runnable directly (`python tests/test_archive_source.py`) or under pytest. Only
requires `datasets`, `jinja2`, `huggingface_hub`.
"""
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterator

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import test_loader_splits as fx  # noqa: E402

from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402

from uad_data import build_smoke_archives, smoke  # noqa: E402

EMNS_MEMBERS = [f"emns/{i}.wav" for i in range(3)]
CLOTHO_AND_EMNS = {
    **fx.CLOTHO,
    "EMNS": {"members": EMNS_MEMBERS,
             "splits": {"train": [fx._emns_record(p) for p in EMNS_MEMBERS]}},
}
CLOTHO_AND_EMNS_CONFIG = {"name": "two", "datasets": [
    {"name": "EMNS", "tasks": ["asr"], "splits": ["train"]},
    {"name": "Clotho", "tasks": ["caption"]},
]}


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _add_smoke(root: str, fixture: dict, datasets: dict, n: int, skip: tuple) -> None:
    """Build smoke archives of `n` clips per split, and their manifest, into the fixture."""
    smoke_dir = os.path.join(root, "smoke")
    os.makedirs(smoke_dir)
    served, entries = {}, {}
    for name, spec in datasets.items():
        if name in skip:
            continue
        full = fixture["files"][f"{name}.tar.gz"]
        split_paths = {split: [r["audio_path"] for r in records]
                       for split, records in spec["splits"].items()}
        path = os.path.join(smoke_dir, f"{name}.tar.gz")
        with open(full, "rb") as source, open(path, "wb") as destination:
            build = build_smoke_archives.write_smoke_archive(
                source, split_paths, n, destination)
        served[f"smoke/{name}.tar.gz"] = path
        entries[name] = smoke.SmokeEntry(
            clips_per_split=n, clips=build.clips, source_sha256=_sha256(full),
            metadata_versions={split: _sha256(fixture["files"][f"{name}_{split}.json"])
                               for split in spec["splits"]},
            revision="abc123", error=build.error)
    manifest = os.path.join(smoke_dir, "manifest.json")
    smoke.write_manifest(entries, manifest)
    served["smoke/manifest.json"] = manifest
    fixture["smoke"] = served
    fixture["manifest_entries"] = entries


class _Logs(logging.Handler):
    """Collects the loader's log lines at INFO and above."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def lines(self, level: int = logging.INFO) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= level]

    def __enter__(self) -> "_Logs":
        logger = logging.getLogger(fx.loader.__name__)
        self._level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(self)
        return self

    def __exit__(self, *exc: object) -> None:
        logger = logging.getLogger(fx.loader.__name__)
        logger.removeHandler(self)
        logger.setLevel(self._level)


def _load(datasets: dict, config: dict, *, built_n: int = 3, skip: tuple = (),
          manifest: bool = True, truncate: dict | None = None, versions: dict | None = None,
          version_error: Exception | None = None, prompts: tuple[dict, ...] = (),
          **kwargs: object) -> "Loaded":
    """Load with smoke archives on the fake Hub; return the rows, the fake and the log lines."""
    with fx.tempfile.TemporaryDirectory() as root:
        fixture = fx._build_fixture(root, datasets, config)
        for prompt in prompts:
            path = os.path.join(fixture["prompts_dir"], f"{prompt['task']}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(prompt, f)
        for base, fraction in (truncate or {}).items():
            with open(fixture["files"][base], "r+b") as f:
                f.truncate(int(os.path.getsize(fixture["files"][base]) * fraction))
        if manifest:
            _add_smoke(root, fixture, datasets, built_n, skip)
        fixture["versions"] = versions or {}
        fake = fx._FakeHub(fixture)
        if version_error is not None:
            fake.file_versions = _raising(version_error)
        with fx._patched_hub(fake), _Logs() as logs:
            rows = fx.loader.load_uad_dataset(
                json_config_path=fixture["config_path"], token=None, **kwargs)
    return Loaded(rows, fake, logs)


@dataclass
class Loaded:
    """What `_load` gives back. Unpacks as `rows, fake, logs`."""
    rows: list[dict]
    fake: fx._FakeHub
    logs: _Logs

    def __iter__(self) -> Iterator[object]:
        return iter((self.rows, self.fake, self.logs))


def _raising(error: Exception) -> Callable[..., dict]:
    def file_versions(*args: object, **kwargs: object) -> dict:
        raise error
    return file_versions


def test_a_capped_run_reads_a_fresh_smoke_archive() -> None:
    rows, fake, logs = _load(fx.CLOTHO, fx._clotho_config(), split="all", clips_per_split=2)

    assert fake.opens == ["smoke/Clotho.tar.gz"], fake.opens
    assert fake.reads == {}, "the full archive was streamed"
    assert fake.version_requests == 1, fake.version_requests
    assert not any("full archive" in line for line in logs.lines()), logs.lines()

    # Exactly the rows the full archive gives.
    full_rows, _ = fx._load(fx.CLOTHO, fx._clotho_config(), split="all",
                            clips_per_split=2, stream=True)
    assert list(rows) == list(full_rows), "smoke rows differ from full-archive rows"

    print("PASS: a capped run reads a fresh smoke archive and gets the full archive's rows.")


def _streamed_full(fake: fx._FakeHub, logs: _Logs, dataset: str,
                   level: int = logging.INFO) -> None:
    assert fake.opens == [f"{dataset}.tar.gz"], fake.opens
    assert f"{dataset}.tar.gz" in fake.reads, "the full archive was not streamed"
    assert any(dataset in line and "full archive" in line for line in logs.lines(level)), \
        logs.lines()


TIMESTAMP_PROMPT = {
    "task": "asr_timestamp_search",
    "prompts": ["What is said between {{start_time}} and {{end_time}}?"],
    "outputs": ["{{transcription}}"],
}
# The first clip of each dataset is an edge case: an EMNS clip with no category,
# so its classification row fails to render, and a libricss clip with no
# utterances. A full-archive run counts both toward the cap.
EDGE_EMNS = [f"emns/{i}.wav" for i in range(3)]
EDGE_LIBRICSS = [f"segments/segment_{i}.wav" for i in range(3)]
EDGE_CASES = {
    "EMNS": {"members": EDGE_EMNS, "splits": {"train": [
        {k: v for k, v in fx._emns_record(p).items() if not (p == EDGE_EMNS[0] and k == "category")}
        for p in EDGE_EMNS]}},
    "libricss": {"members": EDGE_LIBRICSS, "splits": {"test": [
        {"audio_path": p, "transcriptions": [] if p == EDGE_LIBRICSS[0] else [
            {"start_time": 0.0, "end_time": 1.0, "transcription": f"words of {p}"}]}
        for p in EDGE_LIBRICSS]}},
}
EDGE_CASES_CONFIG = {"name": "edges", "datasets": [
    {"name": "EMNS", "tasks": ["classification"], "splits": ["train"]},
    {"name": "libricss", "tasks": ["asr_timestamp_search"], "splits": ["test"]},
]}


def test_a_smoke_archive_gives_the_full_archives_clips_for_unrenderable_clips() -> None:
    """With n == N, clips that fail to render or have no utterances still count."""
    loads = [_load(EDGE_CASES, EDGE_CASES_CONFIG, built_n=2, prompts=(TIMESTAMP_PROMPT,),
                   split="all", clips_per_split=2, stream=stream)
             for stream in (None, True)]
    (smoke_rows, smoke_fake, _), (full_rows, _, _) = loads

    assert smoke_fake.opens == ["smoke/EMNS.tar.gz", "smoke/libricss.tar.gz"], smoke_fake.opens
    assert list(smoke_rows) == list(full_rows), "smoke rows differ from full-archive rows"
    assert smoke_rows.report == full_rows.report, (smoke_rows.report, full_rows.report)
    assert [s.clips_found for s in smoke_rows.report.splits] == [2, 2], smoke_rows.report
    assert len(smoke_rows.report.render_failures) == 2, smoke_rows.report.render_failures

    print("PASS: clips that fail to render count the same in smoke and full-archive runs.")


def test_the_staleness_check_is_one_request_for_every_internal_dataset() -> None:
    rows, fake, logs = _load(CLOTHO_AND_EMNS, CLOTHO_AND_EMNS_CONFIG,
                             split="all", clips_per_split=1)

    assert fake.version_requests == 1, fake.version_requests
    assert sorted(fake.version_checks) == [
        "Clotho.tar.gz", "Clotho_test.json", "Clotho_train.json", "Clotho_validation.json",
        "EMNS.tar.gz", "EMNS_train.json"], fake.version_checks
    assert fake.opens == ["smoke/EMNS.tar.gz", "smoke/Clotho.tar.gz"], fake.opens

    print("PASS: one metadata request checks every smoke archive for staleness.")


def test_a_cap_above_the_smoke_archives_n_streams_the_full_archive() -> None:
    rows, fake, logs = _load(fx.CLOTHO, fx._clotho_config(), built_n=2,
                             split="all", clips_per_split=3)

    _streamed_full(fake, logs, "Clotho")
    assert len(fx._clips(rows)) == 9, fx._clips(rows)

    print("PASS: a cap above the smoke archive's N streams the full archive.")


def test_a_dataset_without_a_smoke_archive_streams_its_full_archive() -> None:
    rows, fake, logs = _load(CLOTHO_AND_EMNS, CLOTHO_AND_EMNS_CONFIG, skip=("EMNS",),
                             split="all", clips_per_split=1)

    assert fake.opens == ["EMNS.tar.gz", "smoke/Clotho.tar.gz"], fake.opens
    assert list(fake.reads) == ["EMNS.tar.gz"], fake.reads
    assert any("EMNS" in line and "full archive" in line for line in logs.lines()), logs.lines()

    print("PASS: an internal dataset without a smoke archive streams its full archive.")


def test_no_manifest_streams_every_full_archive() -> None:
    rows, fake, logs = _load(fx.CLOTHO, fx._clotho_config(), manifest=False,
                             split="all", clips_per_split=1)

    _streamed_full(fake, logs, "Clotho")

    print("PASS: without a smoke manifest, the full archive is streamed.")


def test_a_stale_smoke_archive_is_skipped_with_a_warning() -> None:
    rows, fake, logs = _load(fx.CLOTHO, fx._clotho_config(),
                             versions={"Clotho.tar.gz": "0" * 64},
                             split="all", clips_per_split=1)

    _streamed_full(fake, logs, "Clotho", level=logging.WARNING)
    assert any("stale" in line for line in logs.lines(logging.WARNING)), logs.lines()

    print("PASS: a stale smoke archive is skipped with a warning.")


def test_changed_metadata_makes_the_smoke_archive_stale() -> None:
    """The smoke build picked its clips from the metadata; a re-split changes them."""
    rows, fake, logs = _load(fx.CLOTHO, fx._clotho_config(),
                             versions={"Clotho_test.json": "re-split"},
                             split="all", clips_per_split=1)

    _streamed_full(fake, logs, "Clotho", level=logging.WARNING)
    assert any("Clotho" in line and "stale" in line and "test" in line
               for line in logs.lines(logging.WARNING)), logs.lines()

    print("PASS: a changed metadata JSON makes the smoke archive stale.")


def test_metadata_of_a_split_the_run_config_leaves_out_does_not_matter() -> None:
    config = {"name": "Clotho test only",
              "datasets": [{"name": "Clotho", "tasks": ["caption"], "splits": ["test"]}]}
    rows, fake, logs = _load(fx.CLOTHO, config, versions={"Clotho_train.json": "re-split"},
                             split="all", clips_per_split=1)

    assert fake.opens == ["smoke/Clotho.tar.gz"], fake.opens
    assert "Clotho_train.json" not in fake.version_checks, fake.version_checks

    print("PASS: only the metadata of splits the run config lists is checked.")


def test_a_sha256_check_that_cannot_run_uses_the_smoke_archive_with_a_warning() -> None:
    rows, fake, logs = _load(fx.CLOTHO, fx._clotho_config(),
                             version_error=OSError("offline"),
                             split="all", clips_per_split=1)

    assert fake.opens == ["smoke/Clotho.tar.gz"], fake.opens
    assert any("Clotho" in line and "offline" in line
               for line in logs.lines(logging.WARNING)), logs.lines()
    assert len(fx._clips(rows)) == 3, fx._clips(rows)

    print("PASS: when the sha256 check can't run, the smoke archive is used with a warning.")


def test_a_network_failure_in_the_check_uses_the_smoke_archive() -> None:
    import httpx
    for error in (httpx.ConnectError("no route to host"), OSError("401 Unauthorized")):
        rows, fake, logs = _load(fx.CLOTHO, fx._clotho_config(), version_error=error,
                                 split="all", clips_per_split=1)

        assert fake.opens == ["smoke/Clotho.tar.gz"], (error, fake.opens)
        assert any(str(error) in line for line in logs.lines(logging.WARNING)), logs.lines()

    print("PASS: a network or HTTP failure in the staleness check uses the smoke archive.")


def test_a_bug_in_the_staleness_check_is_not_taken_for_being_offline() -> None:
    try:
        _load(fx.CLOTHO, fx._clotho_config(), version_error=TypeError("a bug"),
              split="all", clips_per_split=1)
    except TypeError as e:
        assert "a bug" in str(e), e
    else:
        raise AssertionError("a TypeError in the staleness check was swallowed")

    print("PASS: a bug in the staleness check raises instead of passing for offline.")


def test_a_full_archive_gone_from_the_hub_counts_as_stale() -> None:
    rows, fake, logs = _load(fx.CLOTHO, fx._clotho_config(), versions={"Clotho.tar.gz": None},
                             split="all", clips_per_split=1)

    _streamed_full(fake, logs, "Clotho", level=logging.WARNING)

    print("PASS: a full archive missing from the Hub makes its smoke archive stale.")


def _load_with_manifest_download(failure: Callable[[str], str]) -> Loaded:
    """Load where fetching `smoke/manifest.json` calls `failure(path)` instead."""
    with fx.tempfile.TemporaryDirectory() as root:
        fixture = fx._build_fixture(root, fx.CLOTHO, fx._clotho_config())
        fake = fx._FakeHub(fixture)
        serve = fake.download_file

        def download_file(path_or_url: str, **kwargs: object) -> str:
            if path_or_url == "smoke/manifest.json":
                return failure(os.path.join(root, "manifest.json"))
            return serve(path_or_url, **kwargs)

        fake.download_file = download_file
        with fx._patched_hub(fake), _Logs() as logs:
            rows = fx.loader.load_uad_dataset(json_config_path=fixture["config_path"],
                                              token=None, split="all", clips_per_split=1)
    return Loaded(rows, fake, logs)


def _raise(error: Exception) -> Callable[[str], str]:
    def failure(path: str) -> str:
        raise error
    return failure


def _serve(content: str) -> Callable[[str], str]:
    def failure(path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
    return failure


def test_a_manifest_that_cannot_be_fetched_warns() -> None:
    offline = LocalEntryNotFoundError("cannot reach the Hub and manifest.json is not cached")
    for error in (OSError("connection reset"), offline):
        rows, fake, logs = _load_with_manifest_download(_raise(error))

        assert any(str(error) in line for line in logs.lines(logging.WARNING)), logs.lines()
        assert fake.opens == ["Clotho.tar.gz"], fake.opens

    print("PASS: a manifest that can't be fetched logs a warning and streams full archives.")


def test_a_malformed_manifest_warns_and_streams_full_archives() -> None:
    entry = {"clips_per_split": 3, "clips": {"test": 3}, "source_sha256": "0" * 64,
             "metadata_versions": {"test": "0" * 64}, "revision": "abc123"}
    for content in ("{\"datasets\": {", json.dumps({"archives": {}}),
                    json.dumps({"datasets": {"Clotho": {**entry, "added_later": 1}}})):
        rows, fake, logs = _load_with_manifest_download(_serve(content))

        assert any("manifest" in line for line in logs.lines(logging.WARNING)), logs.lines()
        assert fake.opens == ["Clotho.tar.gz"], (content, fake.opens)

    print("PASS: a malformed manifest logs a warning and streams full archives.")


def test_a_row_filter_other_than_all_pass_streams_the_full_archive() -> None:
    rows, fake, logs = _load(fx.CLOTHO, fx._clotho_config(row_filter="random"),
                             split="all", clips_per_split=1)

    _streamed_full(fake, logs, "Clotho")
    assert "smoke/manifest.json" not in fake.downloads, fake.downloads

    print("PASS: a row filter other than all_pass streams the full archive.")


def test_an_explicit_stream_bypasses_smoke_archives() -> None:
    for stream in (True, False):
        rows, fake, logs = _load(fx.CLOTHO, fx._clotho_config(),
                                 split="all", clips_per_split=1, stream=stream)

        assert fake.opens == ["Clotho.tar.gz"], (stream, fake.opens)
        assert ("Clotho.tar.gz" in fake.reads) is stream, (stream, fake.reads)
        assert not any(p.startswith("smoke/") for p in fake.downloads), fake.downloads
        assert fake.version_checks == [], fake.version_checks

    print("PASS: an explicit stream bypasses smoke archives.")


def test_a_regular_run_downloads_the_full_archive() -> None:
    rows, fake, logs = _load(fx.CLOTHO, fx._clotho_config(), split="test")

    assert fake.opens == ["Clotho.tar.gz"] and fake.reads == {}, (fake.opens, fake.reads)
    assert not any(p.startswith("smoke/") for p in fake.downloads), fake.downloads

    print("PASS: a regular run downloads the full archive and never asks for smoke archives.")


# Clotho stores test/ first: a cut at 40% keeps the three test clips, and no train
# or validation clip, in the smoke archive.
TRUNCATED_CLOTHO = {"truncate": {"Clotho.tar.gz": 0.4}, "built_n": 3}


def test_a_recorded_build_error_is_a_failed_load_when_a_split_ends_short() -> None:
    rows, fake, logs = _load(CLOTHO_AND_EMNS, CLOTHO_AND_EMNS_CONFIG, **TRUNCATED_CLOTHO,
                             split="all", clips_per_split=2)

    assert "smoke/Clotho.tar.gz" in fake.opens, fake.opens
    clotho_rows = [r for r in rows if r["originating_dataset"] == "Clotho"]
    assert {r["split"] for r in clotho_rows} == {"test"}, clotho_rows
    [failure] = rows.report.load_failures
    recorded = fake.fx["manifest_entries"]["Clotho"].error
    assert recorded, "the build recorded no error"
    assert (failure.dataset, failure.error, failure.rows_kept) == (
        "Clotho", recorded, len(clotho_rows)), failure
    assert any(r["originating_dataset"] == "EMNS" for r in rows), "EMNS did not load"

    print("PASS: a recorded build error is a failed load when a selected split ends short.")


def test_a_recorded_build_error_is_not_reported_when_every_split_reaches_n() -> None:
    rows, fake, logs = _load(fx.CLOTHO, fx._clotho_config(), **TRUNCATED_CLOTHO,
                             split="test", clips_per_split=2)

    assert "smoke/Clotho.tar.gz" in fake.opens, fake.opens
    assert rows.report.load_failures == [], rows.report.load_failures
    assert len(fx._clips(rows)) == 2, fx._clips(rows)

    print("PASS: a recorded build error is not reported when every selected split reaches n.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
