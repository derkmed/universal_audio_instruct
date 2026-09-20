# ADR-0003: A preliminary metric never decides whether a group passes

- **Status:** accepted
- **Date:** 2026-09-18

## Context

An evaluation run groups its rows by (internal dataset, split, task) and reports
a verdict per group. Each group also reports one rough metric: WER for `asr`,
`english_translation` and `caption`; a hit rate for `classification`,
`commonsense` and `qa`.

Those six rules are deliberately crude. `qa` checks only that the answer's
numbers appear in the prediction, so a group of free-text answers scores 1.0
minus the share of blank predictions. `commonsense` asks the prediction to
*start with* the choice letter, so "The answer is B." misses. A group whose
answers carry no choice letters scores 0.0.

A metric that decided a verdict would make these known crudities into run
failures, and would put the eval harness's exit status at the mercy of
`evaluate.load` reaching the Hub.

## Decision

A group passes when it has at least one row and every row's status is `ok`.
Metrics are reported and never consulted.

A metric that cannot be computed — the Hub is down, a rule raises on an odd
answer field — yields `None` and a single printed complaint per distinct cause.
It never costs a run its `results.jsonl` or `summary.json`.

## Consequences

- A run can be green with a WER of 0.9. That is intended: the smoke run asks
  "does every internal dataset load, render and answer?", not "is the model
  good?".
- `metrics.py`'s module docstring states the three surprising consequences
  explicitly, so nobody reads a 1.0 `qa` group as accuracy.
- Any task outside the six has no preliminary metric and reports `None`
  throughout, rather than being force-fitted to one of the rules.
- Real benchmark scoring is a separate, later concern with its own artefacts.
