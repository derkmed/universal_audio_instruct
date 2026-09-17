"""Offline test: asr_timestamp_search renders one row per utterance of a clip.

The task's metadata gives each clip a `transcriptions` list of utterances
(`start_time`, `end_time`, `transcription`), while its prompt templates ask about a
single span. Builds a synthetic archive + metadata + prompt file (a copy of the
Hub's `prompts/asr_timestamp_search.json`), temporarily swaps the Hub download
functions for fakes so nothing touches the network (restoring them afterwards),
and asserts every utterance is rendered with every template -- or, with random
templates, with its own pick -- and that bad metadata fails only the rows that
render it.

Runnable directly (`python tests/test_asr_timestamp_search.py`) or under pytest.
Only requires `datasets`, `jinja2`, `huggingface_hub`.
"""
import contextlib
import io
import json
import os
import sys
import tarfile
import tempfile

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uad_data import filters, hub, loader, prompts  # noqa: E402

# Hub prompts/asr_timestamp_search.json at c4e1b16: 2 system instructions x 2 outputs.
TIMESTAMP_PROMPT = {
    "task": "asr_timestamp_search",
    "system_instructions": [
        "You are given an audio file. Can you please transcribe between {{start_time}} and {{end_time}}?",
        "Tell me what is said between {{start_time}} and {{end_time}} in the provided audio.",
    ],
    "outputs": [
        "Of course. Here is the transcription from {{start_time}} and {{end_time}}: {{transcription}}",
        "{{transcription}}",
    ],
}

# Shaped like the Hub's libricss_test.json. LibriCSS's own `segment` field numbers
# the clip within its recording session; it has nothing to do with utterances.
LIBRICSS_METADATA = [
    {
        "session": 7, "segment": 8, "overlap_ratio": 40.0,
        "audio_path": "segments/segment_8.wav",
        "transcriptions": [
            {"start_time": 0.0, "end_time": 6.76, "transcription": "THIS OUTWARD MUTABILITY"},
            {"start_time": 3.87, "end_time": 10.03, "transcription": "WERE I TO COMPLY"},
        ],
    },
    {
        "session": 7, "segment": 9, "overlap_ratio": 40.0,
        "audio_path": "segments/segment_9.wav",
        "transcriptions": [
            {"start_time": 18.12, "end_time": 21.26, "transcription": "THERE IS NO OPENING"},
        ],
    },
]

# Shaped like the Hub's libricss_subseg_test.json: the record's own start_time /
# end_time give the sub-clip's window within the libricss clip it was cut from
# (`original_audio_path`), while its utterances' times are relative to the sub-clip.
SUBSEG_METADATA = [
    {
        "start_time": 28.25, "end_time": 30.34,
        "transcriptions": [
            {"start_time": 0.06, "end_time": 2.09, "transcription": "HE ACTS AS THOUGH"},
        ],
        "session": 7, "segment": 8, "overlap_ratio": 40.0, "subseg_id": 1,
        "original_audio_path": "segments/segment_8.wav",
        "audio_path": "subsegs/subseg_1.wav",
    },
]

# One clip with many utterances, so that utterances sharing one random template
# would show up (see test_random_templates_are_picked_per_utterance).
MANY_UTTERANCES_METADATA = [
    {
        "audio_path": "sparse_2_0.6/wav/mix_clean/mix_0000001.wav",
        "transcriptions": [
            {"start_time": float(i), "end_time": i + 0.5, "transcription": f"WORD {i}"}
            for i in range(40)
        ],
    },
]

# The second utterance is shaped like SparseLibriMix's before Hub commit e80eab1,
# when its utterances said start / end rather than start_time / end_time.
BAD_UTTERANCE_METADATA = [
    {
        "audio_path": "sparse_2_0.6/wav/mix_clean/mix_0000002.wav",
        "transcriptions": [
            {"start_time": 0.281, "end_time": 3.421, "transcription": "I'LL JUST LOOK"},
            {"start": 1.537, "end": 4.317, "transcription": "MORNIN GIRLS"},
        ],
    },
]


class RejectAll(filters.RowFilter):
    def include_row(self, row) -> bool:
        return False


class FirstUtteranceOnly(filters.RowFilter):
    def include_row(self, row) -> bool:
        return row.utterance_index == 0


def _build_fixture(root: str, name: str, metadata: list[dict], **config) -> dict:
    prompts_dir = os.path.join(root, "prompts")
    os.makedirs(prompts_dir)
    with open(os.path.join(prompts_dir, "asr_timestamp_search.json"), "w", encoding="utf-8") as f:
        json.dump(TIMESTAMP_PROMPT, f)

    metadata_path = os.path.join(root, f"{name}_test.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    tar_path = os.path.join(root, f"{name}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for record in metadata:
            payload = f"FAKE-AUDIO {record['audio_path']}".encode()
            info = tarfile.TarInfo(name=record["audio_path"])
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    config_path = os.path.join(root, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": f"{name} timestamp search test",
            "datasets": [{"name": name, "splits": ["test"], "tasks": ["asr_timestamp_search"]}],
            **config,
        }, f)

    return {
        "name": name,
        "prompts_dir": prompts_dir,
        "metadata_path": metadata_path,
        "tar_path": tar_path,
        "config_path": config_path,
    }


def _fake_hub(fx: dict) -> dict:
    """Fake hub.* functions that serve local fixture files instead of the network."""
    def fake_download_file(path_or_url, *, repo_id=None, revision=None, token=None):
        base = os.path.basename(hub.to_repo_path(path_or_url))
        if base == f"{fx['name']}.tar.gz":
            return fx["tar_path"]
        if base == f"{fx['name']}_test.json":
            return fx["metadata_path"]
        raise AssertionError(f"unexpected download_file for {path_or_url!r}")

    def fake_download_prompts_dir(*, repo_id=None, revision=None, token=None):
        return fx["prompts_dir"]

    return {
        "download_file": fake_download_file,
        "download_prompts_dir": fake_download_prompts_dir,
    }


@contextlib.contextmanager
def _patched_hub(**fakes):
    """Swap hub.* attributes for fakes, restoring the real ones on exit.

    Same as in test_loader.py: restoring them keeps one test's fakes from leaking
    into later tests in the same pytest run. Also restores prompts.PROMPTS_DIR,
    which load_uad_dataset points at the faked, temporary prompts folder.
    """
    originals = {name: getattr(hub, name) for name in fakes}
    original_prompts_dir = prompts.PROMPTS_DIR
    for name, fake in fakes.items():
        setattr(hub, name, fake)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(hub, name, original)
        prompts.PROMPTS_DIR = original_prompts_dir


@contextlib.contextmanager
def _registered_filters(**classes):
    """Make extra filters selectable by name in a run config's row_filter."""
    filters.FILTER_REGISTRY.update(classes)
    try:
        yield
    finally:
        for name in classes:
            del filters.FILTER_REGISTRY[name]


def _load_rows(name: str, metadata: list[dict], **config) -> list[dict]:
    with tempfile.TemporaryDirectory() as root:
        fx = _build_fixture(root, name, metadata, **config)
        with _patched_hub(**_fake_hub(fx)), _registered_filters(
                reject_all=RejectAll, first_utterance_only=FirstUtteranceOnly):
            return loader.load_uad_dataset(
                json_config_path=fx["config_path"],
                split="test",
                token=None,
            )


def _load_error(name: str, metadata: list[dict], **config) -> Exception:
    try:
        rows = _load_rows(name, metadata, **config)
    except Exception as e:
        return e
    raise AssertionError(f"expected an error, got {len(rows)} rows")


def _expected_texts(utterance: dict) -> set[tuple[str, str]]:
    """Every (system_instruction, output) pair the prompt file gives one utterance."""
    start, end, text = utterance["start_time"], utterance["end_time"], utterance["transcription"]
    instructions = [
        f"You are given an audio file. Can you please transcribe between {start} and {end}?",
        f"Tell me what is said between {start} and {end} in the provided audio.",
    ]
    outputs = [f"Of course. Here is the transcription from {start} and {end}: {text}", text]
    return {(si, o) for si in instructions for o in outputs}


def _renders_own_utterance(row: dict) -> bool:
    utterance = row["transcriptions"][row["utterance_index"]]
    return (row["system_instruction"], row["output"]) in _expected_texts(utterance)


def test_one_row_per_utterance_and_template() -> None:
    rows = _load_rows("libricss", LIBRICSS_METADATA)

    # 3 utterances x (2 system instructions x 2 outputs) = 12 rows.
    assert len(rows) == 12, f"expected 12 rows, got {len(rows)}"

    for record in LIBRICSS_METADATA:
        clip_rows = [r for r in rows if r["audio_path"] == record["audio_path"]]
        got = sorted((r["system_instruction"], r["output"]) for r in clip_rows)
        want = sorted(set().union(*(_expected_texts(u) for u in record["transcriptions"])))
        assert got == want, f"{record['audio_path']}: got {got}"
        indices = sorted(r["utterance_index"] for r in clip_rows)
        assert indices == sorted(list(range(len(record["transcriptions"]))) * 4), indices
        for r in clip_rows:
            assert _renders_own_utterance(r), r
            assert r["task"] == "asr_timestamp_search", r["task"]
            assert r["originating_dataset"] == "libricss", r["originating_dataset"]
            assert r["prompt"] == "", r["prompt"]
            assert r["transcriptions"] == record["transcriptions"]

    print("PASS: libricss-shaped clips render one row per utterance and template.")


def test_utterance_times_win_over_record_times() -> None:
    rows = _load_rows("libricss_subseg", SUBSEG_METADATA)

    assert len(rows) == 4, f"expected 4 rows, got {len(rows)}"
    got = {(r["system_instruction"], r["output"]) for r in rows}
    assert got == _expected_texts(SUBSEG_METADATA[0]["transcriptions"][0]), got
    # The row still carries the record's own window, untouched.
    assert all((r["start_time"], r["end_time"]) == (28.25, 30.34) for r in rows)

    print("PASS: libricss_subseg-shaped clips render the utterance's times, not the window's.")


def test_random_templates_are_picked_per_utterance() -> None:
    rows = _load_rows(
        "SparseLibriMix", MANY_UTTERANCES_METADATA, randomize_prompt_format=True)

    # One row per utterance, each rendering its own utterance.
    assert sorted(r["utterance_index"] for r in rows) == list(range(40)), rows
    assert all(_renders_own_utterance(r) for r in rows), rows

    # Each utterance picks its own template. If the whole clip shared one pick,
    # all 40 rows would use the same (instruction, output) pair. Independent picks
    # from 4 pairs all match with probability 4 * (1/4)**40, about 3e-24.
    pairs = {
        (r["system_instruction"].startswith("You are given"), r["output"].startswith("Of course"))
        for r in rows
    }
    assert len(pairs) > 1, f"all 40 utterances got the same template: {pairs}"

    print("PASS: random templates are picked separately for each utterance.")


def test_bad_utterance_fails_only_its_own_rows() -> None:
    # Filtered out, the bad utterance never renders, so nothing raises.
    rows = _load_rows(
        "SparseLibriMix", BAD_UTTERANCE_METADATA, row_filter="first_utterance_only")
    assert len(rows) == 4, f"expected 4 rows, got {len(rows)}"
    assert all(r["utterance_index"] == 0 and _renders_own_utterance(r) for r in rows), rows

    # Rendered, it raises instead of rendering blanks, and says where it is.
    e = _load_error("SparseLibriMix", BAD_UTTERANCE_METADATA)
    assert isinstance(e, KeyError), repr(e)
    for part in ("start_time", "end_time", "mix_0000002.wav", "utterance 1"):
        assert part in str(e), f"{part!r} not in {e}"

    print("PASS: an utterance without start_time fails only when its rows render.")


def test_unusable_transcriptions_fail_only_when_rendered() -> None:
    cases = {
        "missing": {},
        "empty": {"transcriptions": []},
        # The shape Task.features used to declare.
        "dict": {"transcriptions": {"start_time": 0.0, "end_time": 1.0, "transcription": "HI"}},
    }
    for case, fields in cases.items():
        metadata = [{"audio_path": f"clips/{case}.wav", **fields}]

        rows = _load_rows("libricss", metadata, row_filter="reject_all")
        assert rows == [], f"{case}: {rows}"

        e = _load_error("libricss", metadata)
        assert isinstance(e, ValueError), f"{case}: {e!r}"
        for part in ("transcriptions", f"clips/{case}.wav"):
            assert part in str(e), f"{case}: {part!r} not in {e}"

    print("PASS: a missing, empty or non-list transcriptions fails only when its row renders.")


def test_non_object_utterance_fails_only_when_rendered() -> None:
    cases = {
        "string": ["HELLO"],
        "list": [[0.0, 1.0, "HI"]],
    }
    for case, transcriptions in cases.items():
        metadata = [{"audio_path": f"clips/{case}.wav", "transcriptions": transcriptions}]

        rows = _load_rows("libricss", metadata, row_filter="reject_all")
        assert rows == [], f"{case}: {rows}"

        # A ValueError, not the KeyError a missing-field check would give.
        e = _load_error("libricss", metadata)
        assert type(e) is ValueError, f"{case}: {e!r}"
        for part in ("expected an object", f"clips/{case}.wav", "utterance 0"):
            assert part in str(e), f"{case}: {part!r} not in {e}"

    print("PASS: a string or list utterance fails only when its rows render.")


def test_load_leaves_prompts_dir_unchanged() -> None:
    before = prompts.PROMPTS_DIR
    _load_rows("libricss", LIBRICSS_METADATA)
    assert prompts.PROMPTS_DIR == before, prompts.PROMPTS_DIR

    _load_error("SparseLibriMix", BAD_UTTERANCE_METADATA)
    assert prompts.PROMPTS_DIR == before, prompts.PROMPTS_DIR

    print("PASS: prompts.PROMPTS_DIR is restored after a load, even a failed one.")


if __name__ == "__main__":
    test_one_row_per_utterance_and_template()
    test_utterance_times_win_over_record_times()
    test_random_templates_are_picked_per_utterance()
    test_bad_utterance_fails_only_its_own_rows()
    test_unusable_transcriptions_fail_only_when_rendered()
    test_non_object_utterance_fails_only_when_rendered()
    test_load_leaves_prompts_dir_unchanged()
