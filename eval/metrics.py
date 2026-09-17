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
no metric, and reports `None` throughout.

`metric_value` gives one row's value; `aggregate` gives a group's. A WER group is
corpus-level -- one `wer.compute` over every row -- not the mean of the rows'
own WERs, so a long reference weighs more than a short one. A hit-rate group is
the mean of its rows' hits.
"""
import re
import string
from typing import Iterable, Optional

import evaluate as hf_evaluate


WER = "wer"
HIT_RATE = "hit_rate"

# task -> (metric name, the row field holding that task's plain answer).
_METRICS: dict[str, tuple[str, str]] = {
    "asr": (WER, "transcription"),
    "english_translation": (WER, "english_translation"),
    "caption": (WER, "caption"),
    "classification": (HIT_RATE, "category"),
    "commonsense": (HIT_RATE, "commonsense_answer"),
    "qa": (HIT_RATE, "answer"),
}

_PUNCTUATION = str.maketrans({c: " " for c in string.punctuation})
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# A leading multiple-choice letter: one letter, with nothing but punctuation or
# whitespace ahead of it, and a choice marker or the end of the text after it.
# "B. ...", "(b)" and a bare "B" match. "boiling" does not, because its "b" runs
# into another letter -- and neither does "A person is clapping", because a
# letter followed by a space is prose, not a choice. Reading that as choice A
# would hand a free hit to every prose prediction opening with "A " or "I ".
_CHOICE_LETTER = re.compile(r"^[^\w]*([A-Za-z])\s*(?:[^\w\s]|$)")

# Loaded lazily and kept: `evaluate.load` reads from disk on every call.
_wer_metric = None


def metric_name(task: str) -> Optional[str]:
    """The metric this task reports, or None when it has no preliminary metric."""
    entry = _METRICS.get(task)
    return entry[0] if entry else None


def answer_field(task: str) -> Optional[str]:
    """The row field holding this task's plain answer, or None when unscored."""
    entry = _METRICS.get(task)
    return entry[1] if entry else None


def answer_of(row: dict) -> str:
    """This row's plain answer, or "" when the task is unscored or the field is missing."""
    field = answer_field(row.get("task", ""))
    return "" if field is None else str(row.get(field) or "")


def metric_value(row: dict, prediction: str) -> Optional[float]:
    """One row's metric value: its WER, or 1.0/0.0 for a hit-rate task.

    None when the task has no metric, or when a WER row has no reference words
    of its own (an empty transcription), which leaves that row's WER undefined.
    The row still counts in its group -- see `aggregate`.
    """
    task = row.get("task", "")
    name = metric_name(task)
    if name is None:
        return None

    answer = answer_of(row)
    if name == WER:
        reference = answer.strip()
        if not reference:
            return None
        return _wer(predictions=[prediction], references=[reference])
    return float(_hits(task, answer, prediction))


def aggregate(task: str, predicted: Iterable[tuple[dict, str]]) -> Optional[float]:
    """One group's metric value over its `(row, prediction)` pairs.

    Pass every row that has a prediction -- `ok` and `empty_output` -- and no
    others: a row that never reached the model has nothing to compare and
    already fails its group. Which rows count is decided by status alone, so a
    row whose own reference is blank still contributes the words the model
    invented for it.

    None when the task has no metric, when there are no pairs, or when a WER
    group's references are all blank, leaving no words to divide by.
    """
    name = metric_name(task)
    if name is None:
        return None

    pairs = list(predicted)
    if not pairs:
        return None

    if name == WER:
        references = [answer_of(row).strip() for row, _ in pairs]
        if not any(references):
            return None
        return _wer(
            predictions=[prediction for _, prediction in pairs], references=references)

    hits = [_hits(task, answer_of(row), prediction) for row, prediction in pairs]
    return sum(hits) / len(hits)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _wer(*, predictions: list[str], references: list[str]) -> float:
    """Corpus WER. An empty prediction is every reference word deleted: WER 1.0."""
    global _wer_metric
    if _wer_metric is None:
        _wer_metric = hf_evaluate.load(WER)
    return float(_wer_metric.compute(predictions=predictions, references=references))


def _hits(task: str, answer: str, prediction: str) -> float:
    """Whether one hit-rate row's prediction counts as a hit (1.0) or a miss (0.0)."""
    # An empty prediction is a real miss, whatever the answer holds. Checked here
    # rather than per rule, because `qa`'s rule would otherwise hit vacuously on
    # an answer with no numbers in it.
    if not prediction.strip():
        return 0.0

    if task == "classification":
        # The category appears in the prediction, ignoring case and punctuation
        # and reading `_` as a space, so "car_horn" matches "I hear a car horn.".
        category = _normalise(answer)
        return float(bool(category) and category in _normalise(prediction))

    if task == "commonsense":
        # Answers look like "B. the kettle is boiling", and a prediction hits when
        # it starts with that choice letter. Leading punctuation and whitespace
        # ("  (B) ...") don't hide it.
        letter = _choice_letter(answer)
        return float(letter is not None and _choice_letter(prediction) == letter)

    if task == "qa":
        # Answers look like "The result is 102.", and a prediction hits when every
        # number in the answer is one of the prediction's own numbers -- so "1020"
        # does not stand in for "102". An answer with no numbers has none missing,
        # so it hits, which is how the rule reads.
        wanted = set(_NUMBER.findall(answer))
        return float(wanted <= set(_NUMBER.findall(prediction)))

    raise NotImplementedError(f"No hit rule for task {task!r}.")


def _normalise(text: str) -> str:
    """Lowercase, `_` and punctuation as spaces, runs of whitespace collapsed."""
    return " ".join(text.replace("_", " ").translate(_PUNCTUATION).lower().split())


def _choice_letter(text: str) -> Optional[str]:
    """The multiple-choice letter `text` starts with, lowercased, or None."""
    found = _CHOICE_LETTER.match(text)
    return found.group(1).lower() if found else None
