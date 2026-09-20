# ADR-0001: The Hub repo is data only; row expansion lives in `uad_data`

- **Status:** accepted
- **Date:** 2026-07-08 (recorded 2026-09-20)

## Context

UAD was published as a HuggingFace dataset with a `trust_remote_code` loading
script, `Universal-Audio-Understanding.py`. Loading it ran code downloaded from
the Hub. Anyone with read access to the dataset could therefore change what ran
inside every consumer's process, the script could not be reviewed in the code
repo's history, and the row-expansion logic could not be tested offline.

## Decision

The Hub repo holds data only. `uad_data` in this repo downloads plain files with
`huggingface_hub` and expands them into rows locally.

Authoritative on the Hub: `data/**` (per-internal-dataset archives and per-split
metadata), `prompts/*.json`, `universal_audio_dataset_configs/*.json`, and the
dataset card. Authoritative here: everything that turns those into rows.

All Hub access goes through `uad_data/hub.py`. That is the one seam, so the
offline tests fake five functions and the whole loader runs with no network.

## Consequences

- `load_dataset(..., trust_remote_code=True)` no longer works, by design. The
  loading script was deleted from the Hub, which disables that path rather than
  leaving it working-but-deprecated.
- Row expansion is testable: `tests/` runs the real loader against synthetic
  archives in ~18 seconds with no network.
- A dataset change and a code change are now two commits in two places. Onboarding
  an internal dataset touches both repos (see the README).
- Rows are built in memory rather than serialised to Arrow per yield. That
  surfaced a latent aliasing bug — the shared metadata record had to be copied
  per row (`Row.to_output`) — which the old script hid.

See `MIGRATION.md` for the full engineering rationale.
