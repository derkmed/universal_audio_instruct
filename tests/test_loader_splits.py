"""Offline tests for how uad_data.loader reads splits and archives.

Covers the `split` string (`test`, `validation+test`, `all`), the per-split clip
cap taken in archive order, reading each internal dataset's archive once and
stopping early, and seeded prompt-template picks.

Builds synthetic archives, metadata JSONs and prompt files, and temporarily
swaps the Hub download functions for fakes that serve them (restoring them, and
prompts.PROMPTS_DIR, afterwards). Archive payloads are incompressible, so a
gzip stream is read a block at a time and the fakes can tell how far a read got.

Runnable directly (`python tests/test_loader_splits.py`) or under pytest. Only
requires `datasets`, `jinja2`, `huggingface_hub`.
"""
import contextlib
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import tempfile

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uad_data import filters, hub, loader, prompts  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Two templates per slot, so random picks vary between rows.
CAPTION_PROMPT = {
    "task": "caption",
    "system_instructions": ["Caption system A.", "Caption system B."],
    "prompts": ["Caption prompt A.", "Caption prompt B."],
    "outputs": ["{{caption}}", "Heard: {{caption}}"],
}
ASR_PROMPT = {
    "task": "asr",
    "prompts": ["Transcribe."],
    "outputs": ["{{transcription}}"],
}
CLASSIFICATION_PROMPT = {
    "task": "classification",
    "prompts": ["Pick one of {{categories}}."],
    "outputs": ["{{category}}"],
}

PAYLOAD_SIZE = 64 * 1024


def _payload(path: str) -> bytes:
    """Incompressible bytes, fixed per path."""
    return random.Random(path).randbytes(PAYLOAD_SIZE)


def _clotho_record(path: str) -> dict:
    return {"audio_path": path, "caption": f"caption of {path}"}


def _emns_record(path: str) -> dict:
    return {
        "audio_path": path, "transcription": f"words of {path}",
        "category": "happy", "categories": "happy, sad",
    }


def _build_fixture(root: str, datasets: dict, config: dict) -> dict:
    """Write prompts, archives, metadata and a run config under `root`.

    `datasets` maps an internal dataset name to `{"members": [paths in archive
    order], "splits": {split: [metadata records]}}`.
    """
    prompts_dir = os.path.join(root, "prompts")
    os.makedirs(prompts_dir)
    for prompt in (CAPTION_PROMPT, ASR_PROMPT, CLASSIFICATION_PROMPT):
        with open(os.path.join(prompts_dir, f"{prompt['task']}.json"), "w", encoding="utf-8") as f:
            json.dump(prompt, f)

    files = {}
    for name, spec in datasets.items():
        tar_path = os.path.join(root, f"{name}.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for path in spec["members"]:
                payload = _payload(path)
                info = tarfile.TarInfo(name=path)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        files[f"{name}.tar.gz"] = tar_path
        for split, records in spec["splits"].items():
            metadata_path = os.path.join(root, f"{name}_{split}.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(records, f)
            files[f"{name}_{split}.json"] = metadata_path

    config_path = os.path.join(root, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)
    return {"prompts_dir": prompts_dir, "files": files, "config_path": config_path}


class _CountingReader(io.RawIOBase):
    """A read-only file that records how many bytes were read from it."""

    def __init__(self, path: str, log: dict):
        self._f = open(path, "rb")
        self._log = log
        log["size"] = os.path.getsize(path)
        log["read"] = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        n = self._f.readinto(buffer)
        self._log["read"] += n
        return n

    def close(self) -> None:
        self._f.close()
        super().close()


class _FakeHub:
    """Serves fixture files by basename and records every archive open.

    Paths under `smoke/` are served from the fixture's optional `smoke` map,
    keyed by repo path; any other `smoke/` path is not on this Hub. A file's Hub
    version (an archive's LFS sha256, a metadata JSON's version) is the sha256 of
    its fixture file, unless the fixture's `versions` map, keyed by basename,
    overrides it; an override of None means the file is gone from the Hub.
    """

    def __init__(self, fx: dict):
        self.fx = fx
        self.opens: list[str] = []
        self.reads: dict[str, dict] = {}
        self.downloads: list[str] = []
        self.version_checks: list[str] = []
        self.version_requests = 0

    def download_file(self, path_or_url, *, repo_id=None, revision=None, token=None):
        path = hub.to_repo_path(path_or_url)
        self.downloads.append(path)
        if path.startswith("smoke/"):
            return self._smoke_file(path)
        base = os.path.basename(path)
        if base not in self.fx["files"]:
            raise AssertionError(f"unexpected download_file for {path_or_url!r}")
        if base.endswith(".tar.gz"):
            self.opens.append(base)
        return self.fx["files"][base]

    def _smoke_file(self, path: str) -> str:
        smoke = self.fx.get("smoke", {})
        if path not in smoke:
            raise hub.EntryNotFoundError(f"{path} is not on the Hub")
        if path.endswith(".tar.gz"):
            self.opens.append(path)
        return smoke[path]

    def file_versions(self, paths_or_urls, *, repo_id=None, revision=None, token=None):
        self.version_requests += 1
        versions = {}
        for path_or_url in paths_or_urls:
            path = hub.to_repo_path(path_or_url)
            base = os.path.basename(path)
            self.version_checks.append(base)
            version = self.fx.get("versions", {}).get(base, self._own_sha256(base))
            if version is not None:
                versions[path] = version
        return versions

    def _own_sha256(self, base: str) -> str:
        with open(self.fx["files"][base], "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def open_archive_stream(self, path_or_url, *, repo_id=None, revision=None, token=None):
        base = os.path.basename(hub.to_repo_path(path_or_url))
        self.opens.append(base)
        self.reads[base] = {}
        return _CountingReader(self.fx["files"][base], self.reads[base])

    def download_prompts_dir(self, *, repo_id=None, revision=None, token=None):
        return self.fx["prompts_dir"]


@contextlib.contextmanager
def _patched_hub(fake: _FakeHub):
    """Swap hub.* for the fake's methods, restoring them and PROMPTS_DIR on exit."""
    names = ("download_file", "open_archive_stream", "download_prompts_dir", "file_versions")
    originals = {name: getattr(hub, name) for name in names}
    original_prompts_dir = prompts.PROMPTS_DIR
    for name in names:
        setattr(hub, name, getattr(fake, name))
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(hub, name, original)
        prompts.PROMPTS_DIR = original_prompts_dir


def _load(datasets: dict, config: dict, **kwargs) -> tuple[list[dict], _FakeHub]:
    with tempfile.TemporaryDirectory() as root:
        fx = _build_fixture(root, datasets, config)
        fake = _FakeHub(fx)
        with _patched_hub(fake):
            rows = loader.load_uad_dataset(
                json_config_path=fx["config_path"], token=None, **kwargs)
    return rows, fake


def _clips(rows: list[dict]) -> list[tuple[str, str]]:
    """(split, audio_path) of each clip, in the order its first row appears."""
    seen = []
    for r in rows:
        key = (r["split"], r["audio_path"])
        if key not in seen:
            seen.append(key)
    return seen


# Clotho-like: every split registered, archive stores test/ first.
CLOTHO_MEMBERS = (
    [f"test/t{i}.wav" for i in range(3)]
    + [f"train/r{i}.wav" for i in range(3)]
    + [f"validation/v{i}.wav" for i in range(3)]
)
CLOTHO = {
    "Clotho": {
        "members": CLOTHO_MEMBERS,
        "splits": {
            split: [_clotho_record(p) for p in CLOTHO_MEMBERS if p.startswith(f"{split}/")]
            for split in ("train", "validation", "test")
        },
    },
}


def _clotho_config(**extra) -> dict:
    return {
        "name": "Clotho test",
        "datasets": [{"name": "Clotho", "tasks": ["caption"]}],
        **extra,
    }


def test_plus_joined_split_selects_each_named_split() -> None:
    rows, _ = _load(CLOTHO, _clotho_config(), split="validation+test")

    assert {r["split"] for r in rows} == {"validation", "test"}, rows
    assert len(_clips(rows)) == 6, _clips(rows)

    print("PASS: validation+test selects both splits.")


def test_split_can_be_a_datasets_split() -> None:
    import datasets

    rows, _ = _load(CLOTHO, _clotho_config(), split=datasets.Split.TEST)
    assert {r["split"] for r in rows} == {"test"}, rows

    print("PASS: a datasets.Split request loads like its name.")


def test_all_selects_every_split_the_entry_lists() -> None:
    rows, _ = _load(CLOTHO, _clotho_config(), split="all")
    assert {r["split"] for r in rows} == {"train", "validation", "test"}, rows

    config = {"name": "Clotho", "datasets": [
        {"name": "Clotho", "tasks": ["caption"], "splits": ["train", "test"]}]}
    rows, _ = _load(CLOTHO, config, split="all")
    assert {r["split"] for r in rows} == {"train", "test"}, rows

    print("PASS: all selects the splits the run config entry lists.")


def test_unknown_split_name_is_an_error() -> None:
    try:
        rows, _ = _load(CLOTHO, _clotho_config(), split="tset")
    except ValueError as e:
        assert "tset" in str(e), e
    else:
        raise AssertionError(f"expected ValueError, got {len(rows)} rows")

    print("PASS: an unknown split name raises.")


def test_split_no_entry_lists_is_skipped_or_an_error() -> None:
    members = ["emns/0.wav"]
    datasets = {
        **CLOTHO,
        "EMNS": {"members": members, "splits": {"train": [_emns_record(members[0])]}},
    }
    config = {"name": "mixed", "datasets": [
        {"name": "Clotho", "tasks": ["caption"], "splits": ["test"]},
        {"name": "EMNS", "tasks": ["asr"]},
    ]}

    # EMNS lists only train, so it is skipped for test.
    rows, fake = _load(datasets, config, split="test")
    assert {r["originating_dataset"] for r in rows} == {"Clotho"}, rows
    assert fake.opens == ["Clotho.tar.gz"], fake.opens

    # Nothing lists validation.
    try:
        rows, _ = _load(datasets, config, split="validation")
    except ValueError as e:
        assert "validation" in str(e), e
    else:
        raise AssertionError(f"expected ValueError, got {len(rows)} rows")

    print("PASS: unlisted splits are skipped, and a request nothing lists raises.")


def test_cap_counts_clips_per_split_even_when_test_is_stored_first() -> None:
    rows, fake = _load(CLOTHO, _clotho_config(), split="all", clips_per_split=2)

    assert _clips(rows) == [
        ("test", "test/t0.wav"), ("test", "test/t1.wav"),
        ("train", "train/r0.wav"), ("train", "train/r1.wav"),
        ("validation", "validation/v0.wav"), ("validation", "validation/v1.wav"),
    ], _clips(rows)
    assert fake.opens == ["Clotho.tar.gz"], fake.opens

    print("PASS: a per-split cap reaches train and validation behind test/.")


def test_cap_takes_clips_in_archive_order_not_metadata_order() -> None:
    members = [f"test/t{i}.wav" for i in range(4)]
    datasets = {"Clotho": {
        "members": members,
        # Metadata lists the clips back to front.
        "splits": {"test": [_clotho_record(p) for p in reversed(members)]},
    }}
    rows, _ = _load(datasets, _clotho_config(), split="test", clips_per_split=2)

    assert _clips(rows) == [("test", "test/t0.wav"), ("test", "test/t1.wav")], _clips(rows)

    print("PASS: the cap takes the first clips in archive order.")


class _RejectClipsFilter(filters.RowFilter):
    """Rejects every row of the clips named in REJECTED."""

    REJECTED = {"test/t0.wav", "test/t1.wav"}

    def include_row(self, row) -> bool:
        return row.audio_path not in self.REJECTED


def test_cap_skips_clips_whose_rows_are_all_filtered_out() -> None:
    filters.FILTER_REGISTRY["reject_clips"] = _RejectClipsFilter
    try:
        members = [f"test/t{i}.wav" for i in range(4)]
        datasets = {"Clotho": {
            "members": members,
            "splits": {"test": [_clotho_record(p) for p in members]},
        }}
        rows, _ = _load(datasets, _clotho_config(row_filter="reject_clips"),
                        split="test", clips_per_split=2)
    finally:
        del filters.FILTER_REGISTRY["reject_clips"]

    assert _clips(rows) == [("test", "test/t2.wav"), ("test", "test/t3.wav")], _clips(rows)

    print("PASS: a clip counts toward the cap only once one of its rows is kept.")


def test_clip_listed_in_two_splits_counts_toward_both() -> None:
    members = ["shared.wav", "val_only.wav", "test_only.wav"]
    datasets = {"Clotho": {
        "members": members,
        "splits": {
            "validation": [_clotho_record("shared.wav"), _clotho_record("val_only.wav")],
            "test": [_clotho_record("shared.wav"), _clotho_record("test_only.wav")],
        },
    }}
    rows, fake = _load(
        datasets, _clotho_config(), split="validation+test", clips_per_split=1)

    assert sorted(_clips(rows)) == [
        ("test", "shared.wav"), ("validation", "shared.wav")], _clips(rows)
    assert all(r["split"] in ("validation", "test") for r in rows)
    assert fake.opens == ["Clotho.tar.gz"], fake.opens

    print("PASS: a clip in two splits yields rows for, and counts toward, both.")


def test_cap_counts_clips_not_rows() -> None:
    members = [f"emns/{i}.wav" for i in range(8)]
    datasets = {"EMNS": {
        "members": members,
        "splits": {"train": [_emns_record(p) for p in members]},
    }}
    config = {
        "name": "EMNS", "randomize_prompt_format": True,
        "datasets": [{"name": "EMNS", "tasks": ["classification", "asr"]}],
    }
    rows, _ = _load(datasets, config, split="train", clips_per_split=5)

    assert len(rows) == 10, len(rows)
    assert _clips(rows) == [("train", p) for p in members[:5]], _clips(rows)
    assert sorted(r["task"] for r in rows) == ["asr"] * 5 + ["classification"] * 5

    print("PASS: EMNS with n = 5 gives 5 clips and 10 rows.")


def test_reading_stops_once_every_capped_split_is_full() -> None:
    rows, fake = _load(CLOTHO, _clotho_config(), split="test", clips_per_split=2)

    assert _clips(rows) == [("test", "test/t0.wav"), ("test", "test/t1.wav")]
    read = fake.reads["Clotho.tar.gz"]
    # 9 members of 64 KiB each; the two test clips sit in the first quarter.
    assert read["read"] < read["size"] / 2, read

    print("PASS: a capped read stops once its splits are full.")


def test_reading_stops_once_every_uncapped_clip_is_found() -> None:
    rows, fake = _load(CLOTHO, _clotho_config(), split="test", stream=True)

    assert len(_clips(rows)) == 3, _clips(rows)
    read = fake.reads["Clotho.tar.gz"]
    assert read["read"] < read["size"] / 2, read

    print("PASS: an uncapped read stops once every listed clip is found.")


def test_reading_continues_to_the_end_when_a_clip_is_missing() -> None:
    datasets = {"Clotho": {
        "members": CLOTHO_MEMBERS,
        "splits": {"test": [
            _clotho_record(p) for p in ["test/t0.wav", "test/missing.wav"]]},
    }}
    rows, fake = _load(datasets, _clotho_config(), split="test", clips_per_split=2)

    assert _clips(rows) == [("test", "test/t0.wav")], _clips(rows)
    read = fake.reads["Clotho.tar.gz"]
    assert read["read"] == read["size"], read

    print("PASS: a split short of clips reads the whole archive.")


def _template_picks(rows: list[dict]) -> dict[tuple, tuple]:
    """(split, audio_path, task) -> the (system_instruction, prompt, output) it rendered."""
    picks = {}
    for r in rows:
        key = (r["split"], r["audio_path"], r["task"])
        assert key not in picks, f"two rows for {key}"
        picks[key] = (r["system_instruction"], r["prompt"], r["output"])
    return picks


def _random_clotho_config(**extra) -> dict:
    return _clotho_config(randomize_prompt_format=True, **extra)


def test_template_picks_are_stable_across_n_and_split() -> None:
    picks = {}
    for split, n in [("test", 1), ("test", 2), ("test", None),
                     ("validation+test", 1), ("all", 2)]:
        rows, _ = _load(CLOTHO, _random_clotho_config(), split=split, clips_per_split=n)
        picks[(split, n)] = _template_picks(rows)

    everything = picks[("test", None)]
    for run, run_picks in picks.items():
        for key, pick in run_picks.items():
            if key in everything:
                assert pick == everything[key], f"{run}: {key} got {pick}, not {everything[key]}"
    assert ("test", "test/t0.wav", "caption") in picks[("test", 1)]

    print("PASS: a clip gets the same template whatever n or split is.")


def test_template_picks_do_not_depend_on_the_clips_read_before() -> None:
    # validation/ is stored last: a "validation" run reads no other clip first,
    # "validation+test" reads test/ first, and "all" reads test/ and train/ first.
    picks = {}
    for split, n in [("validation", None), ("validation", 1),
                     ("validation+test", None), ("all", None), ("all", 2)]:
        rows, _ = _load(CLOTHO, _random_clotho_config(), split=split, clips_per_split=n)
        picks[(split, n)] = {
            key: pick for key, pick in _template_picks(rows).items()
            if key[0] == "validation"}

    alone = picks[("validation", None)]
    assert len(alone) == 3, alone
    for run, run_picks in picks.items():
        assert run_picks, f"{run}: no validation rows"
        for key, pick in run_picks.items():
            assert pick == alone[key], f"{run}: {key} got {pick}, not {alone[key]}"

    print("PASS: a clip's template doesn't depend on which clips were read before it.")


def test_template_picks_vary_between_clips_of_one_split() -> None:
    members = [f"test/t{i}.wav" for i in range(12)]
    datasets = {"Clotho": {
        "members": members,
        "splits": {"test": [_clotho_record(p) for p in members]},
    }}
    rows, _ = _load(datasets, _random_clotho_config(), split="test")

    # 8 template combinations over 12 clips: one shared pick means the pick
    # ignores the clip.
    system_and_prompt = {(p[0], p[1]) for p in _template_picks(rows).values()}
    assert len(system_and_prompt) > 1, system_and_prompt

    print("PASS: clips in one split get different templates.")


def test_seed_changes_the_picks() -> None:
    by_seed = {}
    for seed in (42, 7):
        rows, _ = _load(CLOTHO, _random_clotho_config(), split="all", seed=seed)
        by_seed[seed] = _template_picks(rows)
    assert by_seed[42] != by_seed[7], "seeds 42 and 7 picked identical templates"

    rows, _ = _load(CLOTHO, _random_clotho_config(), split="all")
    assert _template_picks(rows) == by_seed[42], "the default seed is not 42"

    print("PASS: the seed decides the picks, and defaults to 42.")


def test_random_filter_does_not_change_the_picks() -> None:
    rows, _ = _load(CLOTHO, _random_clotho_config(), split="all")
    unfiltered = _template_picks(rows)

    # RandomFilter reseeds Python's shared `random`, and calls it once per row.
    rows, _ = _load(CLOTHO, _random_clotho_config(row_filter="random"), split="all")
    filtered = _template_picks(rows)
    assert filtered, "RandomFilter kept no rows"
    for key, pick in filtered.items():
        assert pick == unfiltered[key], f"{key}: {pick} != {unfiltered[key]}"

    print("PASS: RandomFilter does not change which templates get picked.")


def test_template_picks_do_not_depend_on_the_hash_seed() -> None:
    """Picks must not use the per-process salted hash()."""
    script = (
        "import json, sys; sys.path.insert(0, 'tests'); "
        "import test_loader_splits as t; "
        "rows, _ = t._load(t.CLOTHO, t._random_clotho_config(), split='all'); "
        "print(json.dumps(sorted(map(list, t._template_picks(rows).values()))))"
    )
    outputs = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=REPO_ROOT, env=env,
            capture_output=True, text=True, check=True)
        outputs.add(result.stdout.strip().splitlines()[-1])
    assert len(outputs) == 1, outputs

    print("PASS: template picks are the same under different hash seeds.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
