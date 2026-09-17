"""Offline tests for eval.metrics (seam N1).

Each task has one preliminary metric, computed against the row's plain answer
field rather than the rendered `output`. These tests walk the six rules of the
spec's metric table with a matching and a non-matching prediction each, then the
group aggregates: corpus WER, mean hit rate, an empty prediction as a miss, and
`null` when a group holds no row the rule can read.

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
# Which metric each task reports, and which field it reads
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

    print("PASS: every task with a metric names it and its answer field.")


def test_a_task_outside_the_table_has_no_metric() -> None:
    """Only the six tasks complete-1..5 use have a metric; the rest report nothing."""
    assert metrics.metric_name("sentiment_analysis") is None
    assert metrics.answer_field("sentiment_analysis") is None
    assert metrics.metric_value(_row("sentiment_analysis", Sentiment="happy"), "happy") is None
    assert metrics.aggregate("sentiment_analysis", [(_row("sentiment_analysis"), "x")]) is None

    print("PASS: a task outside the metric table has no metric.")


def test_an_answer_of_zero_is_not_read_as_a_missing_answer() -> None:
    """`0` is falsy but it is still the answer, and qa's rule needs the digit."""
    assert metrics.answer_of(_row("qa", answer=0)) == "0"
    assert metrics.metric_value(_row("qa", answer=0), "the answer is 5") == 0.0
    assert metrics.metric_value(_row("qa", answer=0), "the answer is 0") == 1.0

    print("PASS: an answer of 0 is kept, not collapsed to a missing answer.")


def test_the_answer_is_read_from_the_plain_field_not_the_rendered_output() -> None:
    row = _row("asr", transcription="hello world")
    row["output"] = "The transcription is: hello world."

    assert metrics.answer_of(row) == "hello world"
    assert metrics.metric_value(row, "hello world") == 0.0

    print("PASS: metrics read the plain answer field.")


# ----------------------------------------------------------------------
# Per-row rules: the three WER tasks
# ----------------------------------------------------------------------

def test_wer_tasks_report_word_error_rate() -> None:
    for task, field in (("asr", "transcription"),
                        ("english_translation", "english_translation"),
                        ("caption", "caption")):
        exact = _row(task, **{field: "a dog barks loudly"})
        assert metrics.metric_value(exact, "a dog barks loudly") == 0.0, task
        # One substitution out of four words.
        assert metrics.metric_value(exact, "a cat barks loudly") == 0.25, task

    print("PASS: asr, english_translation and caption report WER.")


def test_an_empty_prediction_is_a_whole_wer() -> None:
    """An empty prediction is a real miss: every reference word is a deletion."""
    row = _row("asr", transcription="a dog barks loudly")

    assert metrics.metric_value(row, "") == 1.0
    assert metrics.metric_value(row, "   ") == 1.0

    print("PASS: an empty prediction scores WER 1.0.")


def test_a_blank_reference_has_no_wer() -> None:
    """WER is undefined with nothing to compare against, so the row has no value."""
    assert metrics.metric_value(_row("asr", transcription="   "), "anything") is None
    assert metrics.metric_value(_row("asr"), "anything") is None

    print("PASS: a blank reference gives no WER.")


# ----------------------------------------------------------------------
# Per-row rules: the three hit-rate tasks
# ----------------------------------------------------------------------

def test_classification_hits_when_the_category_appears() -> None:
    row = _row("classification", category="car_horn")

    assert metrics.metric_value(row, "I hear a car horn.") == 1.0
    # Case and punctuation are ignored, and `_` reads as a space.
    assert metrics.metric_value(row, "CAR HORN!") == 1.0
    assert metrics.metric_value(row, "car_horn") == 1.0
    assert metrics.metric_value(row, "a dog barking") == 0.0
    assert metrics.metric_value(row, "") == 0.0

    print("PASS: classification hits on the category appearing in the prediction.")


def test_classification_matches_whole_words_not_letter_runs() -> None:
    """Otherwise short categories collect free hits off unrelated words."""
    assert metrics.metric_value(_row("classification", category="car"),
                                "a carpet on the floor") == 0.0
    assert metrics.metric_value(_row("classification", category="cat"),
                                "the cattle are lowing") == 0.0
    assert metrics.metric_value(_row("classification", category="car"),
                                "a car drives past") == 1.0

    print("PASS: classification matches whole words, not letter runs.")


def test_commonsense_hits_on_the_answers_choice_letter() -> None:
    row = _row("commonsense", commonsense_answer="B. the kettle is boiling")

    assert metrics.metric_value(row, "B. the kettle is boiling") == 1.0
    assert metrics.metric_value(row, "b") == 1.0
    # Leading punctuation and whitespace don't hide the letter.
    assert metrics.metric_value(row, "  (B) the kettle") == 1.0
    assert metrics.metric_value(row, "A. the door closed") == 0.0
    # Naming the answer's words without its letter is not a hit.
    assert metrics.metric_value(row, "the kettle is boiling") == 0.0
    assert metrics.metric_value(row, "") == 0.0

    print("PASS: commonsense hits on the prediction starting with the choice letter.")


def test_every_row_with_a_prediction_counts_in_its_hit_rate_group() -> None:
    """The spec's inclusion rule, taken as written.

    A hit-rate group is the mean of the included rows' hits, and the included
    rows are every row with a prediction. So an answer its rule reads as nothing
    still gets a hit of 1 or 0 by that rule, and still counts in the denominator.
    """
    # No choice letter to start with, so the prediction cannot start with it.
    assert metrics.metric_value(
        _row("commonsense", commonsense_answer="the kettle is boiling"),
        "the kettle is boiling") == 0.0
    # No category, so no category appears in the prediction.
    assert metrics.metric_value(_row("classification", category="  "), "a dog") == 0.0

    print("PASS: a row whose rule reads nothing is a miss, and still counts.")


def test_a_numberless_qa_answer_hits_on_any_real_prediction() -> None:
    """The consequence of the rule as written, recorded here rather than hidden.

    `qa`'s rule is "every number in the answer appears in the prediction". A
    free-text answer holds no numbers, so none of them is missing and the row
    hits -- which means a `qa` group of free-text answers reports a hit rate of
    1.0 whatever the model said. The metric is preliminary and never decides
    whether a group passes, so the rule stands as decided.
    """
    row = _row("qa", answer="Yes, a dog.")

    assert metrics.metric_value(row, "No, a cat.") == 1.0
    # An empty prediction is still a real miss, whatever the answer holds.
    assert metrics.metric_value(row, "") == 0.0

    assert metrics.aggregate("qa", [
        (row, "No, a cat."),
        (_row("qa", answer="Maybe."), "Certainly."),
    ]) == 1.0

    print("PASS: a numberless qa answer hits, as the rule reads.")


def test_qa_hits_when_every_number_in_the_answer_appears() -> None:
    row = _row("qa", answer="The result is 102.")

    assert metrics.metric_value(row, "I think the result is 102.") == 1.0
    assert metrics.metric_value(row, "102") == 1.0
    assert metrics.metric_value(row, "The result is 12.") == 0.0
    # A longer number that merely contains the digits is not the same number.
    assert metrics.metric_value(row, "The result is 1020.") == 0.0
    assert metrics.metric_value(row, "") == 0.0

    two = _row("qa", answer="Between 3 and 7 seconds.")
    assert metrics.metric_value(two, "about 3 to 7 seconds") == 1.0
    assert metrics.metric_value(two, "about 3 seconds") == 0.0

    print("PASS: qa hits when every number in the answer appears in the prediction.")


def test_an_empty_prediction_is_a_miss_on_every_hit_rate_task() -> None:
    for row in (_row("classification", category="dog"),
                _row("commonsense", commonsense_answer="B. the kettle"),
                _row("qa", answer="The result is 102.")):
        assert metrics.metric_value(row, "") == 0.0, row["task"]
        assert metrics.metric_value(row, "   ") == 0.0, row["task"]

    print("PASS: an empty prediction misses on every hit-rate task.")


# ----------------------------------------------------------------------
# Group aggregates
# ----------------------------------------------------------------------

def test_wer_groups_aggregate_at_corpus_level_not_as_a_mean() -> None:
    """One long row and one short row: corpus WER weights by reference length."""
    predicted = [
        (_row("asr", transcription="one two three four"), "one two three four"),
        (_row("asr", transcription="five"), "six"),
    ]
    # Per-row WERs are 0.0 and 1.0, whose mean is 0.5. Corpus WER is 1 error
    # over 5 reference words.
    assert metrics.aggregate("asr", predicted) == 0.2

    print("PASS: a WER group aggregates over the rows at corpus level.")


def test_an_empty_prediction_counts_against_a_wer_group() -> None:
    predicted = [
        (_row("asr", transcription="one two"), "one two"),
        (_row("asr", transcription="three four"), ""),
    ]
    assert metrics.aggregate("asr", predicted) == 0.5

    print("PASS: an empty prediction is a miss in a WER group.")


def test_hit_rate_groups_aggregate_as_a_mean() -> None:
    predicted = [
        (_row("classification", category="dog"), "a dog"),
        (_row("classification", category="cat"), "a dog"),
        (_row("classification", category="cow"), ""),
    ]
    assert metrics.aggregate("classification", predicted) == 1 / 3

    print("PASS: a hit-rate group is the mean of its rows' hits.")


def test_a_group_with_no_included_rows_has_no_value() -> None:
    assert metrics.aggregate("asr", []) is None
    assert metrics.aggregate("classification", []) is None
    # Every reference blank leaves corpus WER with no words to divide by.
    assert metrics.aggregate("asr", [(_row("asr", transcription=""), "hello")]) is None

    print("PASS: a group with no included rows reports null.")


def test_a_blank_reference_still_counts_its_row_in_a_wer_group() -> None:
    """The group is one corpus WER over every row with a prediction, not a subset.

    A row with a blank reference has no WER of its own, but the words the model
    invented for it are still insertions against the group's references.
    """
    predicted = [
        (_row("asr", transcription="one two"), "one two"),
        (_row("asr", transcription=""), "three four"),
    ]
    # Two insertions over two reference words.
    assert metrics.aggregate("asr", predicted) == 1.0

    print("PASS: a blank-reference row still counts in its WER group.")


def test_numbers_are_compared_as_numbers_not_as_digit_runs() -> None:
    """A thousands separator or a trailing zero must not invent a miss.

    Splitting "1,000" into "1" and "000" made a prediction saying 1000 miss on
    both, so qa under-reported on any corpus whose answers are formatted.
    """
    assert metrics.metric_value(_row("qa", answer="It cost 1,000 dollars."),
                                "about 1000 dollars") == 1.0
    assert metrics.metric_value(_row("qa", answer="It took 3.5 seconds."),
                                "3.50 seconds") == 1.0
    assert metrics.metric_value(_row("qa", answer="It took 3.5 seconds."),
                                "3.6 seconds") == 0.0
    # A different number is still a different number.
    assert metrics.metric_value(_row("qa", answer="It cost 1,000 dollars."),
                                "about 100 dollars") == 0.0

    print("PASS: numbers are compared as numbers.")


def test_a_failed_metric_load_can_be_retried_in_a_later_run() -> None:
    """One notebook kernel runs many evaluations; a Hub blip must not end them all."""
    attempts = []

    def _failing_load(name):
        attempts.append(name)
        raise OSError("no route to host")

    original_load, original_metric = metrics.hf_evaluate.load, metrics._wer_metric
    metrics.hf_evaluate.load, metrics._wer_metric = _failing_load, None
    try:
        row = _row("asr", transcription="a dog barks")
        for _ in range(3):
            try:
                metrics.metric_value(row, "a dog barks")
            except OSError:
                pass
        assert len(attempts) == 1, attempts

        metrics.forget_failed_load()
        try:
            metrics.metric_value(row, "a dog barks")
        except OSError:
            pass
        assert len(attempts) == 2, attempts
    finally:
        metrics.hf_evaluate.load, metrics._wer_metric = original_load, original_metric

    print("PASS: a failed metric load can be retried in a later run.")


def test_a_metric_that_will_not_load_is_only_attempted_once() -> None:
    """The load runs per row, so a Hub outage must not cost every row a timeout.

    Without remembering the failure, a 1000-row asr run makes 1000 load attempts,
    each with its own retry budget, and prints 1000 identical complaints.
    """
    attempts = []

    def _failing_load(name):
        attempts.append(name)
        raise OSError("no route to host")

    original_load, original_metric = metrics.hf_evaluate.load, metrics._wer_metric
    metrics.hf_evaluate.load, metrics._wer_metric = _failing_load, None
    try:
        row = _row("asr", transcription="a dog barks")
        for _ in range(5):
            try:
                metrics.metric_value(row, "a dog barks")
            except OSError:
                pass
    finally:
        metrics.hf_evaluate.load, metrics._wer_metric = original_load, original_metric

    assert len(attempts) == 1, attempts

    print("PASS: a metric that will not load is only attempted once.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
