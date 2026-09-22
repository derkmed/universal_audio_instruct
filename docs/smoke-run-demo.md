# Demoing a smoke run

A **smoke run** loads only the first few clips, in archive order, of each selected
split of each internal dataset, to check that every one works end to end. This
walkthrough shows how to see a smoke slice of the Universal Audio Understanding
(UAD) dataset today, on a CPU-only machine, without downloading the ~48 GB of
audio.

For the full design — what a smoke run counts, where it reads each archive from,
and the load report it produces — see the spec,
[`docs/specs/smoke-runs.md`](specs/smoke-runs.md), and
[ADR-0004](adr/0004-smoke-archives-and-staleness.md). The terms used here (clip,
row, smoke run, smoke archive, load report, group) are the ones in
[`CONTEXT.md`](../CONTEXT.md).

## What exists today vs. the packaged smoke dataset

The loader has two ways to read a capped run's archives:

- **Fast path (not available yet):** small, published **smoke archives** at
  `smoke/<name>.tar.gz` plus `smoke/manifest.json`, so a smoke run downloads
  megabytes rather than gigabytes.
- **Streaming fallback (works today):** the full archive is streamed lazily over
  HTTP and the read stops once each selected split has enough clips. Nothing near
  48 GB lands on disk — a streamed source "transfers only the prefix up to the
  early-stop point" (loader docstring).

As of 2026-09-20 the smoke archives are **not on the Hub**: `smoke/` is empty and
`complete.json` has not been uploaded. Only `complete-1.json` … `complete-5.json`
exist there. So every smoke run today takes the streaming fallback, and there is
nothing to `hf download` yet. Publishing the packaged smoke set is the pending
build/upload step — [Track B](#track-b--build-the-real-downloadable-smoke-dataset-later)
below, and slice 8 of the spec.

Track A below therefore streams a real Hub run config and early-stops. It needs no
GPU and no smoke archives.

## Track A — preview a smoke slice now (CPU, no GPU, no upload)

Run everything with the parent venv's Python from the `audio_instruct` directory.
Do not create a new venv — the parent venv already has `datasets`, `jinja2`,
`huggingface_hub` and CPU `torch`:

```bash
cd C:/Users/derek/Desktop/UAD-DEV/audio_instruct
```

You need a Hugging Face login for the private dataset. If you have already run
`huggingface-cli login`, the token at `~/.cache/huggingface/token` is picked up
automatically (the loader's `token` argument defaults to that cached login), so
you do not need to pass a token below.

Preview two clips per split of `complete-1.json`, print the load report, and show
the first few rows. `clips_per_split` is what makes this a smoke run; with no
smoke archive present the loader streams the full archive and early-stops:

```bash
../.venv/Scripts/python.exe -c "from uad_data import load_uad_dataset; rows = load_uad_dataset(json_config_path='complete-1.json', split='all', clips_per_split=2); print(rows.report.describe()); [print(r['originating_dataset'], r['split'], r['task'], '|', r['audio_path'], '|', r['prompt'][:60].replace(chr(10),' '), '->', r.get('output','')[:60].replace(chr(10),' ')) for r in rows[:6]]"
```

Notes on the call:

- Every argument is **keyword-only** — pass `json_config_path=`, `split=`,
  `clips_per_split=` by name.
- `json_config_path='complete-1.json'` is resolved by name against the Hub's
  `universal_audio_dataset_configs/` (a local file of the same name would win if
  one existed). Swap in `complete-2.json` … `complete-5.json` to preview a
  different slice, or point at a local `configs/*.json`.
- `split='all'` loads every split each config entry lists. Use `'test'` or
  `'validation+test'` to narrow it.
- The return value is a plain `list` of row dicts that also carries `.report`.
  Existing `rows = load_uad_dataset(...)` callers are unaffected.

Each row is a dict with `originating_dataset`, `split`, `task`, `audio_path`, the
rendered `system_instruction` / `prompt` / `output`, an `audio` dict
(`{"path", "bytes"}`), and every field of the clip's metadata record (e.g. a qa
clip's `answer`, a caption clip's `caption`) plus a `tasks` list. No model is
involved — this is just what the dataset looks like.

### Reading the load report

`rows.report.describe()` prints a human-readable summary; the underlying
`LoadReport` also exposes the raw fields. It records:

- **`splits`** — clips found per `(internal dataset, selected split)`, each
  compared with the cap, with that group's tasks. A split that found fewer clips
  than the cap is flagged; one that found none leaves its groups empty (and would
  fail an eval).
- **`load_failures`** — internal datasets that failed to load, with the error and
  how many rows were kept before the failure.
- **`render_failures`** — rows that failed to render, with dataset, split, task,
  audio path and error.

In a smoke run these are recorded and the run carries on; a regular run (no
`clips_per_split`) raises on the first error instead.

## Memory & hardware

- **Local download:** small. The streaming fallback pulls only the compressed
  prefix of each archive up to the early-stop point, not the whole `.tar.gz`, so
  nothing close to 48 GB is fetched. Streamed reads are not cached, so a repeated
  preview streams again. (Once the smoke archives are published, a smoke run
  downloads the ~165 MiB packaged set once and caches it.)
- **RAM:** small. Rows are plain dicts and audio is decoded per batch, not all at
  once. A loader-only preview holds just the handful of rows you asked for.
- **GPU:** not needed to *preview* the dataset. Running an actual model — eval
  predictions or training — does need a GPU. This dev machine is CPU-only
  (`torch 2.14.0+cpu`, CUDA unavailable), so run model eval or finetuning on a
  GPU, e.g. Colab ([`eval/colab_eval.ipynb`](../eval/colab_eval.ipynb)). Seeing
  what the dataset looks like — rows plus the load report — runs fine on CPU.

## Track B — build the real downloadable smoke dataset (later)

This is the pending step that turns on the fast path for everyone. It is a
one-off operator job, documented in full under
[Smoke archives](specs/smoke-runs.md#smoke-archives-11) and slice 8 of the spec;
the outline is:

Build the combined config from `complete-1..5`:

```bash
../.venv/Scripts/python.exe -m uad_data.build_complete_config --output outputs/complete.json
```

Build the smoke archives (a one-off ~17 GiB Hub stream that reads each full
archive once and keeps the first 10 clips per split):

```bash
../.venv/Scripts/python.exe -m uad_data.build_smoke_archives --clips-per-split 10
```

Then upload `complete.json` to `universal_audio_dataset_configs/`, and the
`smoke/*.tar.gz` archives plus `smoke/manifest.json` to `smoke/` — each upload
confirmed with the user first, using the saved Hugging Face login. See the spec's
[Hub changes](specs/smoke-runs.md#8-hub-changes-confirm-with-the-user-before-each-upload)
for the exact steps and the Hub README edits that go with them.

After that, the whole smoke set (~165 MiB) is `hf download`-able, and a capped
`all_pass` run reads from `smoke/<name>.tar.gz` instead of streaming — the fast
path in [ADR-0004](adr/0004-smoke-archives-and-staleness.md).
