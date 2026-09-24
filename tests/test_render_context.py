"""Offline test: `Task.render_context` checks the `utterance_index` it is given.

The loader only passes indices from `Task.utterance_indices`, but `Row` and
`render_context` can be called directly. An index that doesn't name an utterance
of the clip must fail with a ValueError that says which row it is, and a task
without utterances must not accept one at all.

Runnable directly (`python tests/test_render_context.py`) or under pytest.
Only requires `datasets`.
"""
from uad_data.tasks import Task

THREE_UTTERANCE_RECORD = {
    "audio_path": "segments/segment_3.wav",
    "transcriptions": [
        {"start_time": float(i), "end_time": i + 0.5, "transcription": f"WORD {i}"}
        for i in range(3)
    ],
}

CAPTION_RECORD = {"audio_path": "clips/dog.wav", "caption": "A dog barks."}


def _render_error(task: Task, record: dict, utterance_index) -> Exception:
    try:
        context = task.render_context(record, utterance_index)
    except Exception as e:  # noqa: BLE001 - the test inspects whatever was raised
        return e
    raise AssertionError(
        f"{task.value} rendered utterance_index={utterance_index!r}: {context}")


def _assert_names_row(e: Exception, *parts: str) -> None:
    assert isinstance(e, ValueError), repr(e)
    for part in parts:
        assert part in str(e), f"{part!r} not in {e}"


def test_negative_index_is_rejected() -> None:
    e = _render_error(Task.ASR_TIMESTAMP_SEARCH, THREE_UTTERANCE_RECORD, -2)
    _assert_names_row(
        e, "asr_timestamp_search row for 'segments/segment_3.wav' (utterance -2)")
    print("PASS: a negative utterance_index raises instead of counting from the end.")


def test_index_past_the_end_is_rejected() -> None:
    e = _render_error(Task.ASR_TIMESTAMP_SEARCH, THREE_UTTERANCE_RECORD, 3)
    _assert_names_row(
        e, "asr_timestamp_search row for 'segments/segment_3.wav' (utterance 3)")
    print("PASS: an utterance_index past the end raises a ValueError naming the row.")


def test_non_int_index_is_rejected() -> None:
    # bool is an int subclass, so True would otherwise render utterance 1.
    for index in (True, 1.0, "1"):
        e = _render_error(Task.ASR_TIMESTAMP_SEARCH, THREE_UTTERANCE_RECORD, index)
        _assert_names_row(
            e, f"asr_timestamp_search row for 'segments/segment_3.wav' (utterance {index!r})")
    print("PASS: a bool, float or str utterance_index raises a ValueError naming the row.")


def test_task_without_utterances_rejects_an_index() -> None:
    e = _render_error(Task.CAPTION, CAPTION_RECORD, 0)
    _assert_names_row(e, "caption row for 'clips/dog.wav'", "utterance_index", "0")
    print("PASS: a task without utterances given an utterance_index raises a ValueError naming the row.")


def test_valid_renders_are_unchanged() -> None:
    assert Task.ASR_TIMESTAMP_SEARCH.render_context(THREE_UTTERANCE_RECORD, 2) == {
        "start_time": 2.0, "end_time": 2.5, "transcription": "WORD 2"}
    assert Task.CAPTION.render_context(CAPTION_RECORD) == {"caption": "A dog barks."}
    print("PASS: in-range and absent utterance_index values render as before.")


if __name__ == "__main__":
    test_negative_index_is_rejected()
    test_index_past_the_end_is_rejected()
    test_non_int_index_is_rejected()
    test_task_without_utterances_rejects_an_index()
    test_valid_renders_are_unchanged()
