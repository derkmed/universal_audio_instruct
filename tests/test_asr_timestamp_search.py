"""Offline test: asr_timestamp_search renders one row per segment of a clip.

The task's metadata gives each clip a `transcriptions` list of timed segments
(`start_time`, `end_time`, `transcription`), while its prompt templates ask about a
single span. Builds a synthetic archive + metadata + prompt file (a copy of the
Hub's `prompts/asr_timestamp_search.json`), monkeypatches the Hub download
functions so nothing touches the network, and asserts every segment is rendered
with every template.

Runnable directly (`python tests/test_asr_timestamp_search.py`) or under pytest.
Only requires `datasets`, `jinja2`, `huggingface_hub`.
"""
import io
import json
import os
import sys
import tarfile
import tempfile

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uad_data import hub, loader  # noqa: E402

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

# Shaped like the Hub's libricss_test.json.
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
# end_time are the sub-clip's window in the original segment, while the segment
# times are relative to the sub-clip's audio.
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

# Shaped like the Hub's SparseLibriMix_test.json at c4e1b16, whose segments say
# start / end rather than start_time / end_time.
START_END_METADATA = [
    {
        "transcriptions": [
            {"start": 0.246, "end": 3.826, "transcription": "THAT IS WHY WE CRY"},
        ],
        "audio_path": "sparse_2_0.6/wav/mix_clean/mix_0000001.wav",
    },
]


def _build_fixture(root: str, name: str, metadata: list[dict]) -> dict:
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
        }, f)

    return {
        "name": name,
        "prompts_dir": prompts_dir,
        "metadata_path": metadata_path,
        "tar_path": tar_path,
        "config_path": config_path,
    }


def _install_fakes(fx: dict):
    """Redirect hub.* to local fixture files instead of the network."""
    def fake_download_file(path_or_url, *, repo_id=None, revision=None, token=None):
        base = os.path.basename(hub.to_repo_path(path_or_url))
        if base == f"{fx['name']}.tar.gz":
            return fx["tar_path"]
        if base == f"{fx['name']}_test.json":
            return fx["metadata_path"]
        raise AssertionError(f"unexpected download_file for {path_or_url!r}")

    def fake_download_prompts_dir(*, repo_id=None, revision=None, token=None):
        return fx["prompts_dir"]

    hub.download_file = fake_download_file
    hub.download_prompts_dir = fake_download_prompts_dir


def _load_rows(name: str, metadata: list[dict]) -> list[dict]:
    with tempfile.TemporaryDirectory() as root:
        fx = _build_fixture(root, name, metadata)
        _install_fakes(fx)
        return loader.load_uad_dataset(
            json_config_path=fx["config_path"],
            split="test",
            token=None,
        )


def _expected_texts(segment: dict) -> set[tuple[str, str]]:
    """Every (system_instruction, output) pair the prompt file gives one segment."""
    start, end, text = segment["start_time"], segment["end_time"], segment["transcription"]
    instructions = [
        f"You are given an audio file. Can you please transcribe between {start} and {end}?",
        f"Tell me what is said between {start} and {end} in the provided audio.",
    ]
    outputs = [f"Of course. Here is the transcription from {start} and {end}: {text}", text]
    return {(si, o) for si in instructions for o in outputs}


def test_one_row_per_segment_and_template() -> None:
    rows = _load_rows("libricss", LIBRICSS_METADATA)

    # 3 segments x (2 system instructions x 2 outputs) = 12 rows.
    assert len(rows) == 12, f"expected 12 rows, got {len(rows)}"

    for record in LIBRICSS_METADATA:
        clip_rows = [r for r in rows if r["audio_path"] == record["audio_path"]]
        got = sorted((r["system_instruction"], r["output"]) for r in clip_rows)
        want = sorted(set().union(*(_expected_texts(s) for s in record["transcriptions"])))
        assert got == want, f"{record['audio_path']}: got {got}"
        for r in clip_rows:
            assert r["task"] == "asr_timestamp_search", r["task"]
            assert r["originating_dataset"] == "libricss", r["originating_dataset"]
            assert r["prompt"] == "", r["prompt"]
            assert r["transcriptions"] == record["transcriptions"]

    print("PASS: libricss-shaped clips render one row per segment and template.")


def test_segment_times_win_over_record_times() -> None:
    rows = _load_rows("libricss_subseg", SUBSEG_METADATA)

    assert len(rows) == 4, f"expected 4 rows, got {len(rows)}"
    got = {(r["system_instruction"], r["output"]) for r in rows}
    assert got == _expected_texts(SUBSEG_METADATA[0]["transcriptions"][0]), got
    # The row still carries the record's own window, untouched.
    assert all((r["start_time"], r["end_time"]) == (28.25, 30.34) for r in rows)

    print("PASS: libricss_subseg-shaped clips render the segment's times, not the window's.")


def test_segment_missing_start_time_raises() -> None:
    try:
        rows = _load_rows("SparseLibriMix", START_END_METADATA)
    except KeyError as e:
        assert "start_time" in str(e), e
    else:
        raise AssertionError(f"expected KeyError, got rows: {rows}")

    print("PASS: a segment without start_time fails instead of rendering blanks.")


if __name__ == "__main__":
    test_one_row_per_segment_and_template()
    test_segment_times_win_over_record_times()
    test_segment_missing_start_time_raises()
