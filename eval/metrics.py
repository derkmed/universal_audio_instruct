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
words, the category's words, the choice letter, the numbers. An answer that
leaves that empty -- a blank transcription, a `commonsense_answer` with no
leading choice letter, a `qa` answer with no numbers -- is one the rule cannot
read. Those rows report `None` and are left out of their group's hit rate,
because a rule that cannot tell right from wrong must not report either: a
group of them would otherwise read 1.0000 or 0.0000, indistinguishable from a
model that got everything right or nothing.

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
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# A leading multiple-choice letter: one letter, with nothing but punctuation or
# whitespace ahead of it, and a choice marker or the end of the text after it.
# "B. ...", "(b)" and a bare "B" match. "boiling" does not, because its "b" runs
# into another letter -- and neither does "A person is clapping", because a
# letter followed by prose is prose. Reading that as choice A would hand a free
# hit to every prediction opening with "A " or "I ".
_CHOICE_LETTER = re.compile(r"^[^\w]*([A-Za-z])\s*(?:[^\w\s]|$)")

# Loaded lazily and kept: `evaluate.load` reads from disk on every call.
_wer_metric = None


def _normalise(text: str) -> str:
    """Lowercase, `_` and punctuation as spaces, runs of whitespace collapsed."""
    return " ".join(text.replace("_", " ").translate(_PUNCTUATION).lower().split())


def _choice_letter(text: str) -> str:
    """The multiple-choice letter `text` starts with, lowercased, or ""."""
    found = _CHOICE_LETTER.match(text)
    return found.group(1).lower() if found else ""


def _category_appears(category: str, prediction: str) -> bool:
    """The category's words appear in the prediction, as words and in order.

    Both sides are normalised to space-separated words first, so "car_horn"
    matches "I hear a car horn." Matching on words rather than on a run of
    letters keeps a short category like "car" out of "a carpet on the floor".
    """
    return f" {category} " in f" {_normalise(prediction)} "


def _same_choice_letter(letter: str, prediction: str) -> bool:
    return _choice_letter(prediction) == letter


def _numbers_all_appear(numbers: frozenset, prediction: str) -> bool:
    """Every number in the answer is one of the prediction's own numbers.

    Compared as whole numbers, not as digits inside text, so "1020" does not
    stand in for "102".
    """
    return numbers <= set(_NUMBER.findall(prediction))


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
        read=lambda answer: frozenset(_NUMBER.findall(answer)), hit=_numbers_all_appear),
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

    None when the task has no metric, or when the rule cannot read this row's
    answer. A row with no WER of its own still counts in its group -- see
    `aggregate` -- while an unreadable hit-rate row does not.
    """
    metric = _METRICS.get(row.get("task", ""))
    if metric is None:
        return None

    reading = metric.read(answer_of(row))
    if not reading:
        return None

    if metric.hit is None:
        return _wer(predictions=[prediction], references=[reading])
    # An empty prediction is a real miss, whatever the answer holds. Without
    # this, `qa`'s rule would hit on an answer whose numbers are all absent
    # from a prediction that has no words at all.
    if not prediction.strip():
        return 0.0
    return float(metric.hit(reading, prediction))


def aggregate(task: str, predicted: Iterable[tuple[dict, str]]) -> Optional[float]:
    """One group's metric value over its `(row, prediction)` pairs.

    Pass every row that has a prediction -- `ok` and `empty_output` -- and no
    others: a row that never reached the model has nothing to compare and
    already fails its group.

    A WER group is one corpus WER over all of them, so a row whose own reference
    is blank still contributes the words the model invented for it. A hit-rate
    group averages only the rows its rule can read.

    None when the task has no metric, and when nothing readable is left.
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

    hits = [
        0.0 if not prediction.strip() else float(metric.hit(reading, prediction))
        for reading, prediction in pairs if reading
    ]
    if not hits:
        return None
    return sum(hits) / len(hits)


def _wer(*, predictions: list[str], references: list[str]) -> float:
    """Corpus WER. An empty prediction is every reference word deleted: WER 1.0."""
    global _wer_metric
    if _wer_metric is None:
        _wer_metric = hf_evaluate.load(WER)
    return float(_wer_metric.compute(predictions=predictions, references=references))
