"""The preliminary metric each task reports.

One rough metric per task, computed against the row's plain answer field (its
`transcription`, `category`, `answer`, ...) rather than the rendered `output`,
which wraps the answer in template prose. Preliminary metrics are for
information only: a group passes or fails on its row statuses, never on these
numbers.

| Task                  | Metric   | Answer field         |
|-----------------------|----------|----------------------|
| `asr`                 | WER      | `transcription`      |
| `english_translation` | WER      | `english_translation`|
| `caption`             | WER      | `caption`            |
| `classification`      | hit rate | `category`           |
| `commonsense`         | hit rate | `commonsense_answer` |
| `qa`                  | hit rate | `answer`             |

Those six are every task the `complete-1..5` run configs use. Any other task has
no preliminary metric, and reports `None` throughout.

Each rule reads its answer down to the one thing it compares: the reference
words, the category's words, the choice letter, the numbers.

Which rows a group covers is decided by status alone, as the spec decides it:
every row with a prediction, `ok` and `empty_output` alike. An answer that its
rule reads as nothing is not excluded, it is simply judged by that rule -- a
`commonsense_answer` with no choice letter has none to be started with, so the
row misses; a `qa` answer with no numbers has none missing from the prediction,
so the row hits.

Three consequences are worth knowing before reading one of these numbers, all of
them the rules as decided rather than oversights:

- a `qa` group of free-text answers hits on every row the model answered at all,
  whatever it said, because no number of the answer's is missing. Only its
  `empty_output` rows miss, so such a group reports 1.0 less the share of rows
  the model left blank;
- a `commonsense` group whose answers carry no choice letters reports 0.0,
  because there is no letter for a prediction to start with;
- `commonsense` asks that the prediction *start with* the choice letter, so a
  model answering "The answer is B." misses, and a group answering that way
  reports 0.0.

These metrics are preliminary and never decide whether a group passes.

`metric_value` gives one row's value; `aggregate` gives a group's. A WER group is
corpus-level -- one `wer.compute` over every row -- not the mean of the rows'
own WERs, so a long reference weighs more than a short one. A hit-rate group is
the mean of its rows' hits.
"""
import re
import string
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import evaluate as hf_evaluate


WER = "wer"
HIT_RATE = "hit_rate"

_PUNCTUATION = str.maketrans({c: " " for c in string.punctuation})
# A number, with optional thousands separators, so "1,000" is one number rather
# than "1" and "000". Compared as a value (see `_numbers_of`), so "3.50" and
# "3.5" are the same number and "1020" is still not "102".
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
# A leading multiple-choice letter: one letter, with nothing but punctuation or
# whitespace ahead of it, and not running into another word character. "B. ...",
# "(b)", a bare "B" and "B is correct" all match; "boiling" does not.
_CHOICE_LETTER = re.compile(r"^[^\w]*([A-Za-z])(?!\w)")
# "A" and "I" are also English words, so a prediction opening "A person is
# clapping" or "I think so" says nothing about which choice it means. Those two
# letters count only with a choice marker after them, or standing alone.
_WORD_LETTERS = frozenset("ai")
_MARKED_CHOICE_LETTER = re.compile(r"^[^\w]*([A-Za-z])\s*(?:[^\w\s]|$)")

# Loaded lazily and kept, because `evaluate.load` reads from disk on every call:
# the loaded metric, or the exception that loading it raised.
_wer_metric = None


def _normalise(text: str) -> str:
    """Lowercase, `_` and punctuation as spaces, runs of whitespace collapsed."""
    return " ".join(text.replace("_", " ").translate(_PUNCTUATION).lower().split())


def _choice_letter(text: str) -> str:
    """The multiple-choice letter `text` starts with, lowercased, or ""."""
    found = _CHOICE_LETTER.match(text)
    if not found:
        return ""
    letter = found.group(1).lower()
    if letter in _WORD_LETTERS and not _MARKED_CHOICE_LETTER.match(text):
        return ""  # "A person is clapping" is prose, not choice A
    return letter


def _category_appears(category: str, prediction: str) -> bool:
    """The category's words appear in the prediction, as words and in order.

    Both sides are normalised to space-separated words first, so "car_horn"
    matches "I hear a car horn." Matching on words rather than on a run of
    letters keeps a short category like "car" out of "a carpet on the floor".

    A row with no category has none to appear, so it misses.
    """
    return bool(category) and f" {category} " in f" {_normalise(prediction)} "


def _same_choice_letter(letter: str, prediction: str) -> bool:
    """A row whose answer has no choice letter has none to be started with."""
    return bool(letter) and _choice_letter(prediction) == letter


def _numbers_of(text: str) -> frozenset:
    """The numbers in `text`, as values, so "1,000" and "1000" are one number."""
    return frozenset(float(found.replace(",", "")) for found in _NUMBER.findall(text))


def _numbers_all_appear(numbers: frozenset, prediction: str) -> bool:
    """Every number in the answer is one of the prediction's own numbers.

    Compared as whole numbers, not as digits inside text, so "1020" does not
    stand in for "102".
    """
    return numbers <= _numbers_of(prediction)


@dataclass(frozen=True)
class _Metric:
    """One task's rule: where its answer lives, and what its answer reduces to.

    `read` turns the plain answer into the one thing the rule compares, and an
    empty result means the rule cannot read this answer. `hit` then decides one
    row, or is None for the WER tasks, which compare the read reference directly.
    """
    name: str
    field: str
    read: Callable[[str], Any]
    hit: Optional[Callable[[Any, str], bool]] = None


_METRICS: dict[str, _Metric] = {
    "asr": _Metric(WER, "transcription", read=str.strip),
    "english_translation": _Metric(WER, "english_translation", read=str.strip),
    "caption": _Metric(WER, "caption", read=str.strip),
    "classification": _Metric(
        HIT_RATE, "category", read=_normalise, hit=_category_appears),
    "commonsense": _Metric(
        HIT_RATE, "commonsense_answer", read=_choice_letter, hit=_same_choice_letter),
    "qa": _Metric(
        HIT_RATE, "answer",
        read=_numbers_of, hit=_numbers_all_appear),
}


def metric_name(task: str) -> Optional[str]:
    """The metric this task reports, or None when it has no preliminary metric."""
    metric = _METRICS.get(task)
    return metric.name if metric else None


def answer_field(task: str) -> Optional[str]:
    """The row field holding this task's plain answer, or None when it has no metric."""
    metric = _METRICS.get(task)
    return metric.field if metric else None


def answer_of(row: dict) -> str:
    """This row's plain answer, or "" when the task has no metric or the field is absent.

    A present answer is kept whatever it holds: an `answer` of `0` is falsy but
    it is still the answer, and `qa`'s rule needs that digit.
    """
    field = answer_field(row.get("task", ""))
    value = None if field is None else row.get(field)
    return "" if value is None else str(value)


def metric_value(row: dict, prediction: str) -> Optional[float]:
    """One row's metric value: its WER, or 1.0/0.0 for a hit-rate task.

    None only when the task has no metric, or when a WER row has no reference
    words of its own, which leaves that row's WER undefined. It still counts in
    its group -- see `aggregate`. Every hit-rate row has a value.

    A row is its own group of one, so the rule lives in `aggregate` alone and
    cannot drift between the two.
    """
    return aggregate(row.get("task", ""), [(row, prediction)])


def aggregate(task: str, predicted: Iterable[tuple[dict, str]]) -> Optional[float]:
    """One group's metric value over its `(row, prediction)` pairs.

    Pass every row that has a prediction -- `ok` and `empty_output` -- and no
    others: a row that never reached the model has nothing to compare and
    already fails its group. Every row passed in counts, none is filtered out
    here.

    A WER group is one corpus WER over all of them, so a row whose own reference
    is blank still contributes the words the model invented for it. A hit-rate
    group is the mean of their hits.

    None when the task has no metric, when there are no rows, and when a WER
    group's references are all blank.
    """
    metric = _METRICS.get(task)
    if metric is None:
        return None

    pairs = [(metric.read(answer_of(row)), prediction) for row, prediction in predicted]
    if not pairs:
        return None

    if metric.hit is None:
        references = [reading for reading, _ in pairs]
        if not any(references):
            return None  # no reference words to divide by
        return _wer(
            predictions=[prediction for _, prediction in pairs], references=references)

    hits = [_hit(metric, reading, prediction) for reading, prediction in pairs]
    return sum(hits) / len(hits)


def _hit(metric: "_Metric", reading, prediction: str) -> float:
    """One hit-rate row: 1.0 or 0.0, with the empty-prediction rule written once.

    An empty prediction is a real miss, whatever the answer holds. Without this,
    `qa`'s rule would hit on an answer whose numbers are all absent from a
    prediction that has no words at all.
    """
    if not prediction.strip():
        return 0.0
    return float(metric.hit(reading, prediction))


def forget_failed_load() -> None:
    """Let a failed metric load be attempted again.

    The failure is remembered for a whole run, but one notebook kernel runs many
    evaluations: a Hub blip, or a cell run before `HF_TOKEN` was set, must not
    cost every later run its numbers too. `Evaluator.evaluate` calls this as it
    starts.
    """
    global _wer_metric
    if isinstance(_wer_metric, BaseException):
        _wer_metric = None


def _wer(*, predictions: list[str], references: list[str]) -> float:
    """Corpus WER. An empty prediction is every reference word deleted: WER 1.0."""
    global _wer_metric
    if _wer_metric is None:
        try:
            _wer_metric = hf_evaluate.load(WER)
        except Exception as error:
            # Remember the failure. This runs once per row, so retrying a Hub
            # that is down would cost every row of the run its own timeout and
            # its own identical complaint.
            _wer_metric = error
            raise
    if isinstance(_wer_metric, BaseException):
        raise _wer_metric
    return float(_wer_metric.compute(predictions=predictions, references=references))
