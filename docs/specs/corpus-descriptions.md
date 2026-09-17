# Internal-dataset descriptions are corpus text

Issue: #31

## Problem

The `description` strings in `uad_data/internal_datasets.py` are the corpora's
own summaries, and several use words that `CONTEXT.md` avoids for a **Clip**
("audio samples", "examples", "utterances"). #27 reworded only MELD's and
URDU's "utterances" to "clips", which turned near-quotes into paraphrases and
left the rest untouched. Nothing said which rule applies.

## Decision

Descriptions are corpus text. They keep each corpus's own words, and the
glossary does not govern them. The glossary still governs UAD's code,
comments and docs.

## Seams

- `DATASETS_DIRECTORY[name].description`: the registry entry's text. It is
  tested directly, with no Hub access.

## Behaviour

- A comment on `DATASETS` states the rule. It also covers
  `LIBRICSS_DESCRIPTION`.
- MELD's and URDU's descriptions say "utterances" again, as they did before
  #27. The per-entry "clips replaces utterances" comments go away.
- URDU keeps the #30 fix: its four classes are Angry, Happy, Neutral and Sad.
  That fix corrected a wrong fact, not a word choice.
- AudioMNIST, Clotho and OpenMIC stay as they are.

## Acceptance

- `tests/test_corpus_descriptions.py` passes. It checks that MELD's and URDU's
  descriptions use "utterances" and not "clips".
- Every other test file still passes, including `tests/test_urdu_description.py`.

## Out of scope

- Checking the other descriptions word for word against the corpora's
  published text.
- `CONTEXT.md`. The rule is about the registry, so it sits next to the
  registry.
