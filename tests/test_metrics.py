"""Offline tests for eval.metrics (seam N1).

Each task has one preliminary metric, computed against the row's plain answer
field rather than the rendered `output`. These tests walk the six rules of the
spec's metric table with a matching and a non-matching prediction each, then the
group aggregates: corpus WER, mean hit rate, an empty prediction as a miss, and
`null` when a group has no row to score.

Runnable directly (`python tests/test_metrics.py`) or under pytest. Needs the
`wer` metric from the `evaluate` package.
"""
import os
import sys

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval import metrics  # noqa: E402


def _row(task: str, **answer) -> dict:
    return {"task": task, **answer}


# ----------------------------------------------------------------------
# Which metric each task reports, and which field it is scored against
# ----------------------------------------------------------------------

def test_each_task_names_its_metric_and_answer_field() -> None:
    assert metrics.metric_name("asr") == "wer"
    assert metrics.metric_name("english_translation") == "wer"
    assert metrics.metric_name("caption") == "wer"
    assert metrics.metric_name("classification") == "hit_rate"
    assert metrics.metric_name("commonsense") == "hit_rate"
    assert metrics.metric_name("qa") == "hit_rate"

    assert metrics.answer_field("asr") == "transcription"
    assert metrics.answer_field("english_translation") == "english_translation"
    assert metrics.answer_field("caption") == "caption"
    assert metrics.answer_field("classification") == "category"
    assert metrics.answer_field("commonsense") == "commonsense_answer"
    assert metrics.answer_field("qa") == "answer"

    print("PASS: every scored task names its metric and answer field.")


def test_an_unscored_task_has_no_metric() -> None:
    """Only the six tasks complete-1..5 use are scored; the rest report nothing."""
    assert metrics.metric_name("sentiment_analysis") is None
    assert metrics.answer_field("sentiment_analysis") is None
    assert metrics.score(_row("sentiment_analysis", Sentiment="happy"), "happy") is None
    assert metrics.aggregate("sentiment_analysis", [(_row("sentiment_analysis"), "x")]) is None

    print("PASS: a task outside the metric table has no metric.")


def test_the_answer_is_read_from_the_plain_field_not_the_rendered_output() -> None:
    row = _row("asr", transcription="hello world")
    row["output"] = "The transcription is: hello world."

    assert metrics.answer_of(row) == "hello world"
    assert metrics.score(row, "hello world") == 0.0

    print("PASS: metrics score against the plain answer field.")


# ----------------------------------------------------------------------
# Per-row rules: the three WER tasks
# ----------------------------------------------------------------------

def test_wer_tasks_score_word_error_rate() -> None:
    for task, field in (("asr", "transcription"),
                        ("english_translation", "english_translation"),
                        ("caption", "caption")):
        exact = _row(task, **{field: "a dog barks loudly"})
        assert metrics.score(exact, "a dog barks loudly") == 0.0, task
        # One substitution out of four words.
        assert metrics.score(exact, "a cat barks loudly") == 0.25, task

    print("PASS: asr, english_translation and caption score WER.")


def test_an_empty_prediction_is_a_whole_wer() -> None:
    """An empty prediction is a real miss: every reference word is a deletion."""
    row = _row("asr", transcription="a dog barks loudly")

    assert metrics.score(row, "") == 1.0
    assert metrics.score(row, "   ") == 1.0

    print("PASS: an empty prediction scores WER 1.0.")


def test_a_blank_reference_has_no_wer() -> None:
    """WER is undefined with nothing to compare against, so the row scores null."""
    assert metrics.score(_row("asr", transcription="   "), "anything") is None
    assert metrics.score(_row("asr"), "anything") is None

    print("PASS: a blank reference gives no WER.")


# ----------------------------------------------------------------------
# Per-row rules: the three hit-rate tasks
# ----------------------------------------------------------------------

def test_classification_hits_when_the_category_appears() -> None:
    row = _row("classification", category="car_horn")

    assert metrics.score(row, "I hear a car horn.") == 1.0
    # Case and punctuation are ignored, and `_` reads as a space.
    assert metrics.score(row, "CAR HORN!") == 1.0
    assert metrics.score(row, "car_horn") == 1.0
    assert metrics.score(row, "a dog barking") == 0.0
    assert metrics.score(row, "") == 0.0

    print("PASS: classification hits on the category appearing in the prediction.")


def test_commonsense_hits_on_the_answers_choice_letter() -> None:
    row = _row("commonsense", commonsense_answer="B. the kettle is boiling")

    assert metrics.score(row, "B. the kettle is boiling") == 1.0
    assert metrics.score(row, "b") == 1.0
    # Leading punctuation and whitespace don't hide the letter.
    assert metrics.score(row, "  (B) the kettle") == 1.0
    assert metrics.score(row, "A. the door closed") == 0.0
    # Naming the answer's words without its letter is not a hit.
    assert metrics.score(row, "the kettle is boiling") == 0.0
    assert metrics.score(row, "") == 0.0

    print("PASS: commonsense hits on the prediction starting with the choice letter.")


def test_commonsense_without_a_choice_letter_never_hits() -> None:
    """There is no letter to start with, so nothing can match it."""
    assert metrics.score(_row("commonsense", commonsense_answer="boiling"), "boiling") == 0.0

    print("PASS: a commonsense answer with no choice letter cannot be hit.")


def test_qa_hits_when_every_number_in_the_answer_appears() -> None:
    row = _row("qa", answer="The result is 102.")

    assert metrics.score(row, "I think the result is 102.") == 1.0
    assert metrics.score(row, "102") == 1.0
    assert metrics.score(row, "The result is 12.") == 0.0
    # A longer number that merely contains the digits is not the same number.
    assert metrics.score(row, "The result is 1020.") == 0.0
    assert metrics.score(row, "") == 0.0

    two = _row("qa", answer="Between 3 and 7 seconds.")
    assert metrics.score(two, "about 3 to 7 seconds") == 1.0
    assert metrics.score(two, "about 3 seconds") == 0.0

    print("PASS: qa hits when every number in the answer appears in the prediction.")


def test_a_qa_answer_with_no_numbers_hits_vacuously() -> None:
    """The decided rule is 'every number appears'; with no numbers, none are missing."""
    assert metrics.score(_row("qa", answer="Yes."), "No.") == 1.0

    print("PASS: a numberless qa answer hits vacuously, as the rule reads.")


# ----------------------------------------------------------------------
# Group aggregates
# ----------------------------------------------------------------------

def test_wer_groups_aggregate_at_corpus_level_not_as_a_mean() -> None:
    """One long row and one short row: corpus WER weights by reference length."""
    scored = [
        (_row("asr", transcription="one two three four"), "one two three four"),
        (_row("asr", transcription="five"), "six"),
    ]
    # Per-row WERs are 0.0 and 1.0, whose mean is 0.5. Corpus WER is 1 error
    # over 5 reference words.
    assert metrics.aggregate("asr", scored) == 0.2

    print("PASS: a WER group aggregates over the rows at corpus level.")


def test_an_empty_prediction_counts_against_a_wer_group() -> None:
    scored = [
        (_row("asr", transcription="one two"), "one two"),
        (_row("asr", transcription="three four"), ""),
    ]
    assert metrics.aggregate("asr", scored) == 0.5

    print("PASS: an empty prediction is a miss in a WER group.")


def test_hit_rate_groups_aggregate_as_a_mean() -> None:
    scored = [
        (_row("classification", category="dog"), "a dog"),
        (_row("classification", category="cat"), "a dog"),
        (_row("classification", category="cow"), ""),
    ]
    assert metrics.aggregate("classification", scored) == 1 / 3

    print("PASS: a hit-rate group is the mean of its rows' hits.")


def test_a_group_with_no_scorable_rows_has_no_value() -> None:
    assert metrics.aggregate("asr", []) is None
    assert metrics.aggregate("classification", []) is None
    # Rows whose reference is blank leave a WER group with nothing to compare.
    assert metrics.aggregate("asr", [(_row("asr", transcription=""), "hello")]) is None

    print("PASS: a group with no scorable rows reports null.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
