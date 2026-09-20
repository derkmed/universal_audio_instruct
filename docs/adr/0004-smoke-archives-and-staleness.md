# ADR-0004: Smoke runs read published smoke archives, and judge their freshness

- **Status:** accepted (code shipped; the Hub side is not yet populated)
- **Date:** 2026-09-20

## Context

A smoke run loads the first N clips of each selected split of each internal
dataset, to check every one works end to end. The full archives are multi-GB;
`complete-1..5` covers 23 internal datasets across 41 (dataset, split) pairs.
Downloading them all to read 5 clips each is not viable in CI or on Colab.

Streaming a full archive over HTTP and stopping early transfers only the
compressed prefix, which is much better — but the prefix can still be large, and
nothing is cached, so a repeated run pays again every time.

## Decision

Publish one small **smoke archive** per internal dataset at `smoke/<name>.tar.gz`,
holding the first N clips of each of its registered splits *in archive order* —
exactly the clips a capped run would take from the full archive — plus
`smoke/manifest.json` describing how each was built. Build them with
`python -m uad_data.build_smoke_archives`.

The loader prefers a smoke archive only when all of these hold: the run is capped,
the row filter is `all_pass`, the manifest is readable, the entry's
`clips_per_split` is at least the N asked for, and the entry is not stale. It
otherwise streams the full archive, saying in a log line which condition failed.

**Staleness** is judged by content, not by time: the manifest records the full
archive's LFS sha256 and each split's metadata version at the commit it was built
from, and the loader compares them against the Hub's current versions in a single
`get_paths_info` request covering every internal dataset. Metadata versions are
included because a re-split changes which clips belong to which split without
touching the archive.

When that check cannot be answered — offline, or a Hub request error — the smoke
archives are used anyway, with a warning. Only request failures are tolerated;
anything else raises.

A read error that stopped a build is recorded in the manifest rather than raised,
so a truncated archive still yields the clips before the cut. It is re-raised as a
failed load only when a selected split ends short of what the run wanted.

## Consequences

- `all_pass` is the only filter smoke archives can serve, because they hold the
  *first* clips whatever a filter would have selected. A run with any other filter
  streams, and says so.
- The Hub must be kept in step: whenever an archive or a metadata JSON changes,
  rerun the builder and re-upload. Forgetting is safe but slow — the staleness
  check catches it and falls back to streaming.
- **Not yet done:** `smoke/` on the Hub is currently empty, and `complete.json`
  has not been uploaded. Every smoke run today takes the streaming fallback. The
  code path is built and tested; the data is not published.
