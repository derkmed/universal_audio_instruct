# ADR-0002: A row is called a Row, not an Example

- **Status:** accepted
- **Date:** 2026-09-17

## Context

HuggingFace `datasets` and the `Trainer` both call one unit of data an
"example". The code inherited that word from the loading script: `Sample`,
`iter_samples`, `sample_filter`, `max_samples`.

UAD has two units that "example" would blur. A **clip** is one audio file in one
split of one internal dataset. A **row** is one clip rendered for one task with
one prompt template — and for `asr_timestamp_search`, one clip renders once per
utterance. One clip becomes many rows. The smoke-run cap counts clips, not rows,
so a name that blurs them makes the cap's meaning unreadable.

Renaming everything to `Example` (matching HF) was considered.

## Decision

The unit a model sees is a **Row**. `Sample` → `Row`, `uad_data/sample.py` →
`uad_data/row.py`, `iter_samples` → `_iter_rows`, the run config key
`sample_filter` → `row_filter`, and `max_samples` → `clips_per_split` (which also
changed what is counted).

Aligning on HF's "example" was rejected: it is the word that causes the
confusion, not the word that resolves it.

## Consequences

- The run config key `sample_filter` is *refused* with a `ValueError` naming
  `row_filter`, rather than ignored, so an old config cannot silently lose its
  filter.
- Corpus descriptions in `uad_data/internal_datasets.py` are quoted in each
  corpus's own words and may still say "sample" or "utterance". The glossary
  governs UAD's code and docs, not quoted source text.
- `docs/research/*.md` and the `.py` files beside them predate the rename and
  still use `Sample` / `iter_samples`. They are dated records of a past run and
  are deliberately not swept; each `.py` carries a note pinning it to its commit.
- `TODO(derkmed)` in `CONTEXT.md`: revisit Row vs Example one day.
