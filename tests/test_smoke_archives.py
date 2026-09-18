"""Offline tests for the smoke-archive builder and the Hub's sha256 lookup.

`build_smoke_archives.write_smoke_archive` copies the first N clips of each
split, in archive order, from a full archive into a smoke archive, and records
the read error that stopped it, if any. `stale_sources` is the builder's safety
check for a local copy of the dataset. `hub.file_sha256` reads a file's LFS
sha256 from the Hub; here its `HfApi` is replaced with a fake.

Reuses the synthetic archives of `test_loader_splits.py`. Runnable directly
(`python tests/test_smoke_archives.py`) or under pytest. Only requires
`datasets`, `jinja2`, `huggingface_hub`.
"""
import hashlib
import io
import os
import sys
import tarfile
import tempfile
from typing import Callable, TypeVar

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import test_loader_splits as fx  # noqa: E402

from huggingface_hub.hf_api import RepoFile  # noqa: E402

from uad_data import build_smoke_archives, hub, internal_datasets, smoke  # noqa: E402

T = TypeVar("T")

# Clotho-like: every split registered, test/ stored first, four clips each.
MEMBERS = (
    [f"test/t{i}.wav" for i in range(4)]
    + [f"train/r{i}.wav" for i in range(4)]
    + [f"validation/v{i}.wav" for i in range(4)]
)
SPLIT_PATHS = {
    split: [p for p in MEMBERS if p.startswith(f"{split}/")]
    for split in ("train", "validation", "test")
}


def _archive(members: list[str]) -> bytes:
    """A gzipped tar of `members`, in order, with the fixture's payloads."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in members:
            payload = fx._payload(path)
            info = tarfile.TarInfo(name=path)
            info.size = len(payload)
            info.mtime = 1_700_000_000
            info.mode = 0o640
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _build(source: bytes, split_paths: dict, clips_per_split: int,
           ) -> tuple[build_smoke_archives.SmokeBuild, list[tuple[tarfile.TarInfo, bytes]]]:
    """Run the builder over `source`; return its result and the smoke archive's members."""
    destination = io.BytesIO()
    result = build_smoke_archives.write_smoke_archive(
        io.BytesIO(source), split_paths, clips_per_split, destination)
    destination.seek(0)
    with tarfile.open(fileobj=destination, mode="r:gz") as tar:
        members = [(info, tar.extractfile(info).read()) for info in tar.getmembers()]
    return result, members


def test_smoke_archive_holds_the_first_n_clips_of_each_split_in_archive_order() -> None:
    result, members = _build(_archive(MEMBERS), SPLIT_PATHS, 2)

    assert [info.name for info, _ in members] == [
        "test/t0.wav", "test/t1.wav", "train/r0.wav", "train/r1.wav",
        "validation/v0.wav", "validation/v1.wav"], members
    assert result.clips == {"train": 2, "validation": 2, "test": 2}, result
    assert result.error is None, result

    print("PASS: a smoke archive holds the first N clips of each split, in archive order.")


def test_smoke_archive_entries_and_bytes_are_copied_as_is() -> None:
    with tarfile.open(fileobj=io.BytesIO(_archive(MEMBERS)), mode="r:gz") as tar:
        source = {info.name: info for info in tar.getmembers()}

    _, members = _build(_archive(MEMBERS), SPLIT_PATHS, 1)

    for info, data in members:
        assert data == fx._payload(info.name), info.name
        original = source[info.name]
        assert (info.size, info.mtime, info.mode, info.type) == (
            original.size, original.mtime, original.mode, original.type), info.name

    print("PASS: smoke archive entries and bytes are copied as-is.")


def test_metadata_order_does_not_matter() -> None:
    reversed_paths = {split: list(reversed(paths)) for split, paths in SPLIT_PATHS.items()}

    _, members = _build(_archive(MEMBERS), reversed_paths, 1)

    assert [info.name for info, _ in members] == [
        "test/t0.wav", "train/r0.wav", "validation/v0.wav"], members

    print("PASS: the builder follows archive order, not metadata order.")


def test_a_clip_listed_in_two_splits_is_copied_once_and_counts_for_both() -> None:
    split_paths = {"train": ["shared.wav", "a.wav"], "test": ["shared.wav", "b.wav"]}

    result, members = _build(_archive(["shared.wav", "a.wav", "b.wav"]), split_paths, 1)

    assert [info.name for info, _ in members] == ["shared.wav"], members
    assert result.clips == {"train": 1, "test": 1}, result

    print("PASS: a clip in two splits is copied once and counts toward both.")


def test_members_no_split_lists_are_left_out() -> None:
    split_paths = {"test": ["test/t0.wav", "test/t1.wav"]}

    result, members = _build(_archive(["stray.wav", "test/t0.wav", "test/t1.wav"]),
                             split_paths, 5)

    assert [info.name for info, _ in members] == ["test/t0.wav", "test/t1.wav"], members
    assert result.clips == {"test": 2}, result
    assert result.error is None, result

    print("PASS: archive members that no split lists are left out.")


def test_reading_stops_once_every_split_has_n_clips() -> None:
    source = _archive(MEMBERS)
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "Clotho.tar.gz")
        with open(path, "wb") as f:
            f.write(source)
        log = {}
        reader = fx._CountingReader(path, log)
        build_smoke_archives.write_smoke_archive(reader, SPLIT_PATHS, 1, io.BytesIO())
        reader.close()

    # validation/v0 is the last clip wanted; v1..v3 are never read.
    assert log["read"] < log["size"] - 2 * fx.PAYLOAD_SIZE, log

    print("PASS: the builder stops reading once every split has N clips.")


def test_a_truncated_source_keeps_the_clips_before_the_cut_and_records_the_error() -> None:
    source = _archive(MEMBERS)
    # Cut inside train/: the four test clips survive, then the stream ends.
    truncated = source[:int(len(source) * 0.45)]

    result, members = _build(truncated, SPLIT_PATHS, 10)

    assert [info.name for info, _ in members][:4] == [f"test/t{i}.wav" for i in range(4)]
    assert all(data == fx._payload(info.name) for info, data in members), members
    assert result.clips["test"] == 4 and result.clips["validation"] == 0, result
    assert result.clips["train"] < 4, result
    assert result.error, result

    print("PASS: a truncated source gives the clips before the cut and a recorded error.")


def _repo_file(path: str, sha256: str | None) -> RepoFile:
    lfs = None if sha256 is None else {"size": 3, "oid": sha256, "pointerSize": 130}
    return RepoFile(path=path, size=3, oid="blob", lfs=lfs)


class _FakeApi:
    """Stands in for HfApi: answers get_paths_info from `files` and records calls."""
    calls: list[dict] = []

    def __init__(self, files: dict, token: str | None = None) -> None:
        self.files = files

    def get_paths_info(self, repo_id: str, paths: list[str], *, revision: str | None = None,
                       repo_type: str | None = None, **kwargs: object) -> list[RepoFile]:
        _FakeApi.calls.append({"repo_id": repo_id, "paths": paths,
                               "revision": revision, "repo_type": repo_type})
        return [self.files[p] for p in paths if p in self.files]


def _with_fake_api(files: dict, call: Callable[[], T]) -> T:
    original = hub.HfApi
    hub.HfApi = lambda token=None: _FakeApi(files, token)
    _FakeApi.calls = []
    try:
        return call()
    finally:
        hub.HfApi = original


def test_file_sha256_reads_the_lfs_sha256_at_a_revision() -> None:
    path = "data/Clotho/Clotho.tar.gz"
    files = {path: _repo_file(path, "ab" * 32)}

    sha256 = _with_fake_api(files, lambda: hub.file_sha256(
        internal_datasets.DATASETS_DIRECTORY["Clotho"].data_url, revision="abc123"))

    assert sha256 == "ab" * 32, sha256
    [call] = _FakeApi.calls
    assert call["paths"] == [path] and call["revision"] == "abc123", call
    assert call["repo_type"] == "dataset", call

    print("PASS: file_sha256 reads a file's LFS sha256 at a revision.")


def test_file_sha256_raises_for_a_missing_or_non_lfs_file() -> None:
    files = {"README.md": _repo_file("README.md", None)}

    for path, error in (("smoke/manifest.json", hub.EntryNotFoundError),
                        ("README.md", ValueError)):
        try:
            _with_fake_api(files, lambda: hub.file_sha256(path))
        except error as e:
            assert path in str(e), e
        else:
            raise AssertionError(f"{path}: expected {error.__name__}")

    print("PASS: file_sha256 raises for a missing file and for a non-LFS file.")


class _ShaHub:
    """Answers file_sha256 from a map of repo path to sha256."""

    def __init__(self, sha256: dict) -> None:
        self.sha256 = sha256

    def file_sha256(self, path_or_url: str, *, repo_id: str | None = None,
                    revision: str | None = None, token: str | None = None) -> str:
        return self.sha256[hub.to_repo_path(path_or_url)]


def _write(root: str, repo_path: str, data: bytes) -> None:
    path = os.path.join(root, *repo_path.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def test_stale_sources_names_local_archives_that_differ_from_the_hub() -> None:
    registry = internal_datasets.DATASETS_DIRECTORY
    clotho, emns, aesdd = registry["Clotho"], registry["EMNS"], registry["AESDD"]
    fresh, old = b"current archive", b"old archive"
    sha = {hub.to_repo_path(d.data_url): hashlib.sha256(fresh).hexdigest()
           for d in (clotho, emns, aesdd)}

    with tempfile.TemporaryDirectory() as root:
        _write(root, hub.to_repo_path(clotho.data_url), fresh)
        _write(root, hub.to_repo_path(emns.data_url), old)
        # AESDD has no local copy at all.
        original = hub.file_sha256
        hub.file_sha256 = _ShaHub(sha).file_sha256
        try:
            stale = build_smoke_archives.stale_sources(
                root, [clotho, emns, aesdd], revision="abc123",
                repo_id=hub.DEFAULT_REPO_ID, token=None)
        finally:
            hub.file_sha256 = original

    assert [problem.dataset for problem in stale] == ["EMNS", "AESDD"], stale
    assert "sha256" in stale[0].reason and "missing" in stale[1].reason, stale

    print("PASS: stale_sources names local archives that are stale or missing.")


def test_manifest_round_trips() -> None:
    entries = {
        "Clotho": smoke.SmokeEntry(
            clips_per_split=10, clips={"train": 10, "test": 10},
            source_sha256="ab" * 32, revision="abc123"),
        "MLEnd_Intonation": smoke.SmokeEntry(
            clips_per_split=10, clips={"train": 7, "validation": 0},
            source_sha256="cd" * 32, revision="abc123", error="EOFError: cut"),
    }
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "manifest.json")
        smoke.write_manifest(entries, path)
        assert smoke.read_manifest(path) == entries

    print("PASS: the manifest reads back what was written.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
