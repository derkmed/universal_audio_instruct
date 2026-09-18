"""Offline end-to-end test for uad_data.loader.

Builds a synthetic audio archive + metadata + prompt file, temporarily swaps the
Hub download functions for fakes so nothing touches the network (restoring them,
and prompts.PROMPTS_DIR, afterwards), and asserts the loader emits
the same rows the old HF loading script would have -- including the
(audio x task x prompt-template) expansion and independent (non-aliased) rows.

Runnable directly (`python tests/test_loader.py`) or under pytest. Only requires
`datasets`, `jinja2`, `huggingface_hub` -- not the heavy eval deps.
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


def _build_fixture(root: str, metadata: list[dict] = METADATA, config: dict = CONFIG) -> dict:
    prompts_dir = os.path.join(root, "prompts")
    os.makedirs(prompts_dir)
    with open(os.path.join(prompts_dir, "caption.json"), "w", encoding="utf-8") as f:
        json.dump(CAPTION_PROMPT, f)

    metadata_path = os.path.join(root, "Clotho_test.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    tar_path = os.path.join(root, "Clotho.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, payload in AUDIO_BYTES.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    config_path = os.path.join(root, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f)

    return {
        "prompts_dir": prompts_dir,
        "metadata_path": metadata_path,
        "tar_path": tar_path,
        "config_path": config_path,
    }


def _fake_hub(fx: dict) -> dict:
    """Fake hub.* functions that serve local fixture files instead of the network."""
    def fake_download_file(path_or_url, *, repo_id=None, revision=None, token=None):
        if hub.to_repo_path(path_or_url).startswith("smoke/"):
            raise hub.EntryNotFoundError(f"{path_or_url} is not on the Hub")
        base = os.path.basename(hub.to_repo_path(path_or_url))
        if base.endswith(".tar.gz"):
            return fx["tar_path"]
        if base == "Clotho_test.json":
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

    loader calls the functions through the module (`hub.download_file(...)`), so
    patching the hub module's attributes is enough. Restoring them keeps one test's
    fakes from leaking into later tests in the same pytest run.

    Also restores prompts.PROMPTS_DIR, which load_uad_dataset points at the faked
    prompts folder -- a temporary directory that is gone once the test ends.
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


class RejectAll(filters.RowFilter):
    def include_row(self, row) -> bool:
        return False


@contextlib.contextmanager
def _registered_filters(**classes):
    """Make extra filters selectable by name in a run config's row_filter."""
    filters.FILTER_REGISTRY.update(classes)
    try:
        yield
    finally:
        for name in classes:
            del filters.FILTER_REGISTRY[name]


def test_load_expands_rows() -> None:
    with tempfile.TemporaryDirectory() as root:
        fx = _build_fixture(root)
        with _patched_hub(**_fake_hub(fx)):
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
    # with distinct outputs (regression test for the shallow-copy fix in Row).
    assert a_rows[0] is not a_rows[1]
    assert a_rows[0]["output"] != a_rows[1]["output"]

    print("PASS: load_uad_dataset produced 4 correctly-expanded, independent rows.")


def test_clips_per_split_streams_prefix() -> None:
    """clips_per_split auto-enables the streaming archive path and stops early."""
    with tempfile.TemporaryDirectory() as root:
        fx = _build_fixture(root)
        fakes = _fake_hub(fx)

        calls = {"stream": 0, "download_tar": 0}

        def fake_open_archive_stream(path_or_url, *, repo_id=None, revision=None, token=None):
            calls["stream"] += 1
            return open(fx["tar_path"], "rb")

        # Detect any full-download of the archive so we can prove it was NOT used.
        fixture_download_file = fakes["download_file"]

        def counting_download_file(path_or_url, *, repo_id=None, revision=None, token=None):
            if os.path.basename(hub.to_repo_path(path_or_url)).endswith(".tar.gz"):
                calls["download_tar"] += 1
            return fixture_download_file(
                path_or_url, repo_id=repo_id, revision=revision, token=token)

        fakes["download_file"] = counting_download_file
        fakes["open_archive_stream"] = fake_open_archive_stream

        # clips_per_split set -> stream defaults to True.
        with _patched_hub(**fakes):
            rows = loader.load_uad_dataset(
                json_config_path=fx["config_path"],
                split="test",
                token=None,
                clips_per_split=1,
            )

    # 1 clip x (1 sysinst x 1 prompt x 2 outputs) = 2 rows.
    assert len(rows) == 2, f"expected 2 rows (1 clip), got {len(rows)}"
    assert {r["audio_path"] for r in rows} == {"test/a.wav"}, rows
    assert calls["stream"] == 1, "streaming archive opener was not used"
    assert calls["download_tar"] == 0, "archive was fully downloaded despite streaming"
    r = rows[0]
    assert r["task"] == "caption" and r["caption"] in r["output"], r
    print("PASS: streaming path honored clips_per_split and avoided the full archive download.")


def test_clips_per_split_must_be_positive() -> None:
    with tempfile.TemporaryDirectory() as root:
        fx = _build_fixture(root)
        for bad in (0, -1):
            with _patched_hub(**_fake_hub(fx)):
                try:
                    rows = loader.load_uad_dataset(
                        json_config_path=fx["config_path"], split="test",
                        token=None, clips_per_split=bad)
                except ValueError as e:
                    assert "clips_per_split" in str(e), e
                else:
                    raise AssertionError(f"clips_per_split={bad}: got {len(rows)} rows")

    print("PASS: clips_per_split below 1 raises.")


def test_missing_field_fails_only_when_rendered() -> None:
    """A record without its task's field raises when its row renders, after the filter."""
    metadata = [{"audio_path": "test/a.wav"}, METADATA[1]]  # a.wav has no caption

    with tempfile.TemporaryDirectory() as root:
        fx = _build_fixture(root, metadata, {**CONFIG, "row_filter": "reject_all"})
        with _patched_hub(**_fake_hub(fx)), _registered_filters(reject_all=RejectAll):
            rows = loader.load_uad_dataset(
                json_config_path=fx["config_path"], split="test", token=None)
    assert rows == [], rows

    with tempfile.TemporaryDirectory() as root:
        fx = _build_fixture(root, metadata)
        with _patched_hub(**_fake_hub(fx)):
            try:
                rows = loader.load_uad_dataset(
                    json_config_path=fx["config_path"], split="test", token=None)
            except KeyError as e:
                assert "caption" in str(e) and "test/a.wav" in str(e), e
            else:
                raise AssertionError(f"expected KeyError, got {len(rows)} rows")

    print("PASS: a record missing its caption fails only when its row renders.")


def test_load_leaves_prompts_dir_unchanged() -> None:
    before = prompts.PROMPTS_DIR
    with tempfile.TemporaryDirectory() as root:
        fx = _build_fixture(root)
        with _patched_hub(**_fake_hub(fx)):
            loader.load_uad_dataset(
                json_config_path=fx["config_path"], split="test", token=None)
        assert prompts.PROMPTS_DIR == before, prompts.PROMPTS_DIR

        def failing_download_file(path_or_url, **kwargs):
            raise OSError("download failed")

        fakes = {**_fake_hub(fx), "download_file": failing_download_file}
        try:
            with _patched_hub(**fakes):
                loader.load_uad_dataset(
                    json_config_path=fx["config_path"], split="test", token=None)
        except OSError:
            pass
        else:
            raise AssertionError("expected the failing download to raise")
        assert prompts.PROMPTS_DIR == before, prompts.PROMPTS_DIR

    print("PASS: prompts.PROMPTS_DIR is restored after a load, even a failed one.")


if __name__ == "__main__":
    test_load_expands_rows()
    test_clips_per_split_streams_prefix()
    test_clips_per_split_must_be_positive()
    test_missing_field_fails_only_when_rendered()
    test_load_leaves_prompts_dir_unchanged()
