# ADR-0008: One internal dataset gets one classification axis

- **Status:** accepted
- **Date:** 2026-09-24

## Context

`prompts/classification.json` is a single system instruction that names no
subject: *"Determine which of the following categories best describes or
categorizes the provided audio: {{categories}}."* All the meaning rides on the
dataset's own `categories` list, so esc50 (sound events), hi_kia (speaker),
MELD (emotion), MLEnd_Intonation (intonation) and slurp_real (intent) share one
prompt file.

Two Hub scripts produced that shape. `add_classification_category_to_slurp_real.py`
copies a dataset's label column into `category`;
`add_classification_categories.py` then collects `sorted(set(category))` over a
dataset's three split JSONs and writes the joined string into every record's
`categories`. All 12 classification datasets came out identical in structure:
one `categories` string, constant across every row and split, exactly equal to
the observed label set.

`Task.features` gives `classification` the pair `category` / `categories`, and
`Task.render_context` passes only those two. There is one `category` column, so
there is room for one axis.

Several datasets have a second label sitting beside it, left over from the
corpus: MELD's `Sentiment` (3 values) next to its 7 emotions, AESDD's
`language`, hi_kia's `gender`, esc50's `esc10`, OpenMic's `relevance`,
slurp_real's `action` (45 values). None of them has a candidate list, and none
is registered as a task.

`slurp_real` was the one dataset whose registration still named the old
per-axis tasks (`intent_detection`, `action_classification`) after the move to
`classification`, and the Hub has no prompt file for either. #57 is the result:
`_get_prompt_templates` raises `PromptTemplateError` on the first clip, and a
smoke run does not step over that one.

## Decision

An internal dataset registers `classification` for exactly one axis, the one its
`category` / `categories` columns carry. A second label stays an unused metadata
column.

`slurp_real` registers `classification`, `intent_detection_nl` and `asr`. Its
intent is the classification axis -- `category` is a copy of `intent`, verified
equal on all 50,569 records -- so `intent_detection` would have rendered the same
label twice. `action` keeps its column and loses its task, as MELD's `Sentiment`
did.

`Task.INTENT_DETECTION` and `Task.ACTION_CLASSIFICATION` are removed.
`Task.SENTIMENT_ANALYSIS` and `Task.INTONATION_DETECTION` are not, although no
dataset registers them: `prompts/` still holds a file naming each, and
`_get_prompt_templates` builds a `PromptFilepath` for every file it globs, so a
missing enum member would make `Task(...)` raise on that file and take every
other task's lookup down with it. They are removable once the files are.

## Consequences

`slurp_real`'s `action` and `intent` are unreachable as labels. Nothing selected
them: no run config in `universal_audio_dataset_configs/` names either task, and
`complete-1..5` name neither.

Giving a dataset a second axis is now a schema change, not a registration: a
second label column plus its candidate list, and a way for one dataset to
register `classification` twice. MELD would want the same thing. That is a
larger piece of work than #57 and is not done here.

`tests/test_prompt_contract.py` holds the invariant this ADR relies on -- every
registered task has a prompt file, and every prompt file renders from its task's
features -- against `tests/hub_prompts.json`, which
`python -m uad_data.check_hub_prompts` keeps honest against the Hub.

## Noticed but not decided here

`slurp_real`'s `categories` list has 90 entries, of which 30 are prefix-less
duplicates: SLURP labels intents `scenario_action`, but 1,308 records (2.6%)
carry only the action, so `joke` sits in the list beside `general_joke` and
`hue_lightoff` beside `iot_hue_lightoff`. The list is `sorted(set(...))` of the
column, so it inherits the defect. 27 of the 30 have exactly one compound twin
and are mechanically repairable (759 records); `query` (481), `remove` (47) and
`set` (21) do not and need the upstream scenario. Filed separately.
