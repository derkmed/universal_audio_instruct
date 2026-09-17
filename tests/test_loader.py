"""Offline end-to-end test for uad_data.loader.

Builds a synthetic audio archive + metadata + prompt file, monkeypatches the Hub
download functions so nothing touches the network, and asserts the loader emits
the same rows the old HF loading script would have -- including the
(audio x task x prompt-template) expansion and independent (non-aliased) rows.

Runnable directly (`python tests/test_loader.py`) or under pytest. Only requires
`datasets`, `jinja2`, `huggingface_hub` -- not the heavy eval deps.
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

# Real caption.json structure: 1 system instruction x 1 prompt x 2 outputs.
CAPTION_PROMPT = {
    "task": "caption",
    "system_instructions": ["You are an audio captioner."],
    "prompts": ["Describe the audio clip in a single sentence."],
    "outputs": [
        "This is what I hear in the attached audio clip: {{caption}}",
        "{{caption}}",
    ],
}

METADATA = [
    {"audio_path": "test/a.wav", "caption": "a cat meows"},
    {"audio_path": "test/b.wav", "caption": "a dog barks"},
]

AUDIO_BYTES = {
    "test/a.wav": b"FAKE-AUDIO-A",
    "test/b.wav": b"FAKE-AUDIO-B",
}

CONFIG = {
    "name": "Clotho Caption Test",
    "datasets": [{"name": "Clotho", "splits": ["test"], "tasks": ["caption"]}],
}


def _build_fixture(root: str) -> dict:
    prompts_dir = os.path.join(root, "prompts")
    os.makedirs(prompts_dir)
    with open(os.path.join(prompts_dir, "caption.json"), "w", encoding="utf-8") as f:
        json.dump(CAPTION_PROMPT, f)

    metadata_path = os.path.join(root, "Clotho_test.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(METADATA, f)

    tar_path = os.path.join(root, "Clotho.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, payload in AUDIO_BYTES.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    config_path = os.path.join(root, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f)

    return {
        "prompts_dir": prompts_dir,
        "metadata_path": metadata_path,
        "tar_path": tar_path,
        "config_path": config_path,
    }


def _install_fakes(monkeypatch_targets: dict):
    """Redirect hub.* to local fixture files instead of the network."""
    def fake_download_file(path_or_url, *, repo_id=None, revision=None, token=None):
        base = os.path.basename(hub.to_repo_path(path_or_url))
        if base.endswith(".tar.gz"):
            return monkeypatch_targets["tar_path"]
        if base == "Clotho_test.json":
            return monkeypatch_targets["metadata_path"]
        raise AssertionError(f"unexpected download_file for {path_or_url!r}")

    def fake_download_prompts_dir(*, repo_id=None, revision=None, token=None):
        return monkeypatch_targets["prompts_dir"]

    hub.download_file = fake_download_file
    hub.download_prompts_dir = fake_download_prompts_dir
    # loader imported these names into its own module namespace via `from . import hub`
    # so patching the hub module attributes is sufficient (loader calls hub.download_*).


def test_load_expands_rows() -> None:
    with tempfile.TemporaryDirectory() as root:
        fx = _build_fixture(root)
        _install_fakes(fx)

        rows = loader.load_uad_dataset(
            json_config_path=fx["config_path"],
            split="test",
            token=None,
        )

    # 2 audios x 1 task (caption) x (1 sysinst x 1 prompt x 2 outputs) = 4 rows.
    assert len(rows) == 4, f"expected 4 rows, got {len(rows)}"

    # Consumption contract: every key the Evaluator reads must be present.
    for r in rows:
        assert r["task"] == "caption", r["task"]
        assert r["split"] == "test", r["split"]
        assert r["originating_dataset"] == "Clotho", r["originating_dataset"]
        assert isinstance(r["audio"]["bytes"], bytes) and r["audio"]["bytes"], "missing audio bytes"
        assert r["system_instruction"] == "You are an audio captioner."
        assert r["prompt"] == "Describe the audio clip in a single sentence."
        assert "output" in r

    # Audio bytes routed to the correct row, and templating rendered per-caption.
    by_caption = {}
    for r in rows:
        by_caption.setdefault(r["caption"], []).append(r)

    assert set(by_caption) == {"a cat meows", "a dog barks"}, list(by_caption)

    a_rows = by_caption["a cat meows"]
    assert all(r["audio"]["bytes"] == AUDIO_BYTES["test/a.wav"] for r in a_rows), "audio bytes mismatch"
    a_outputs = sorted(r["output"] for r in a_rows)
    assert a_outputs == sorted([
        "This is what I hear in the attached audio clip: a cat meows",
        "a cat meows",
    ]), a_outputs

    # Aliasing guard: the two rows for the same audio must be independent objects
    # with distinct outputs (regression test for the shallow-copy fix in Sample).
    assert a_rows[0] is not a_rows[1]
    assert a_rows[0]["output"] != a_rows[1]["output"]

    print("PASS: load_uad_dataset produced 4 correctly-expanded, independent rows.")


def test_max_samples_streams_prefix() -> None:
    """max_samples auto-enables the streaming archive path and stops early."""
    with tempfile.TemporaryDirectory() as root:
        fx = _build_fixture(root)
        _install_fakes(fx)

        calls = {"stream": 0, "download_tar": 0}

        def fake_open_archive_stream(path_or_url, *, repo_id=None, revision=None, token=None):
            calls["stream"] += 1
            return open(fx["tar_path"], "rb")

        # Detect any full-download of the archive so we can prove it was NOT used.
        prev_download_file = hub.download_file

        def counting_download_file(path_or_url, *, repo_id=None, revision=None, token=None):
            if os.path.basename(hub.to_repo_path(path_or_url)).endswith(".tar.gz"):
                calls["download_tar"] += 1
            return prev_download_file(path_or_url, repo_id=repo_id, revision=revision, token=token)

        hub.open_archive_stream = fake_open_archive_stream
        hub.download_file = counting_download_file

        # max_samples set -> stream defaults to True.
        rows = loader.load_uad_dataset(
            json_config_path=fx["config_path"],
            split="test",
            token=None,
            max_samples=1,
        )

    assert len(rows) == 1, f"expected 1 row (capped), got {len(rows)}"
    assert calls["stream"] == 1, "streaming archive opener was not used"
    assert calls["download_tar"] == 0, "archive was fully downloaded despite streaming"
    r = rows[0]
    assert r["task"] == "caption" and r["caption"] in r["output"], r
    print("PASS: streaming path honored max_samples and avoided the full archive download.")


if __name__ == "__main__":
    test_load_expands_rows()
    test_max_samples_streams_prefix()
