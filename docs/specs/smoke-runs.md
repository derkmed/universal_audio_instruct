# Smoke runs across every UAD dataset

Spec for the map [#1]. Its decisions come from three places:

- the resolution comments on the map's closed tickets ([#2]–[#8], [#11]);
- the amendment comment on [#7];
- a `/grill` session on 2026-09-16, which settled the points those tickets left
  open. Decisions from that session are marked **(grill)**.

Terms follow the glossary in [`CONTEXT.md`](../../CONTEXT.md): *clip*, *row*,
*run config*, *selected split*, *archive order*, *smoke archive*, *smoke run*,
*regular run*, *load report*, *row status*, *group*, *preliminary metric*.

In this spec, **n** is the `--clips-per-split` value a run asks for, and **N** is
the number of clips per split that the smoke archives hold (10).

## Goal

A single training or evaluation run should be able to check, cheaply and end to
end, every internal dataset, selected split and task listed in the Hub run
configs `complete-1..5`. Today that isn't possible. The only cap is
`max_samples`, which limits rows across the whole run, and a run loads just one
split. After this work, `--json-config complete.json --clips-per-split 5` loads
the first 5 clips, in archive order, of every selected split of every internal
dataset. It reads them from small smoke archives and carries on past failures.
It reports pass or fail for each (internal dataset, split, task) group, with one
preliminary metric per task.

## Seams

Tests stay offline, like the ones already in `tests/`. They use synthetic tar
archives, metadata JSONs and prompt files, and replace the `uad_data.hub`
functions with local fakes. The system Python on this machine has none of the
test dependencies, so the build needs a venv built from `requirements.txt`. CPU
`torch` is enough.

### Existing seams (preferred)

| # | Seam | What gets tested there | Imports needed |
|---|---|---|---|
| S1 | `uad_data.load_uad_dataset(...)`, with `hub.download_file`, `hub.open_archive_stream` and `hub.download_prompts_dir` faked, as `tests/test_loader.py` already does | Split syntax and matching; the per-split clip cap in archive order; one archive read per internal dataset; row order; early stop; seeded prompt templates; the load report; smoke-run failure tolerance vs regular-run raising; choosing between smoke archive, streamed full archive and downloaded full archive | `datasets`, `jinja2`, `huggingface_hub` |
| S2 | `UniversalJsonConfig(filepath=...).toCollection()`, as in `tests/test_json_config_loader.py` | The `row_filter` key; unknown top-level fields being ignored | same as S1 |
| S3 | Constructing `EvalConfig(...)` and `TrainConfig(...)` | `clips_per_split` validation; how the `dataset_split` default resolves; the `seed` default | same as S1 (both import only `eval.config`) |
| S4 | `eval.main.build_parser()` and `train.main.build_parser()` | `--clips-per-split`, `--seed`, the `--split` default, and `--max-samples` being gone | adds `torch` and `transformers`, which both modules import at load time (no GPU needed) |
| S5 | `Evaluator(backend, config).evaluate(rows)`, with a fake `ModelBackend` subclass and a temporary `output_dir` | Row statuses; groups and pass/fail; `results.jsonl`; `summary.json`; the console table; smoke-run tolerance vs regular-run raising; no second truncation | all of `requirements.txt` (no GPU). Tiny WAVs are written with `soundfile`; garbage bytes trigger `audio_error` |

The S1 hub fake gains one function (N4). It also has to serve two new paths,
`smoke/manifest.json` and `smoke/<name>.tar.gz`, which the loader fetches through
the existing `hub.download_file`.

### New seams (each justified)

| # | Seam | Why it's needed |
|---|---|---|
| N1 | A pure scoring function for each task in a new `eval/metrics.py` (e.g. `score(task, row, prediction)`), plus the group aggregate | The six preliminary metrics have fiddly normalisation rules: case, punctuation, `_` as a space, choice letters, "every number". A table of cases is much cheaper to test here than through S5. S5 still checks that metric values reach `results.jsonl` and `summary.json`. |
| N2 | The core of the smoke-archive builder: given a source tar stream, each split's set of `audio_path`s, and N, it writes the smoke tar and returns the clips copied per split plus any read error | A smoke run must see exactly the clips the full archive would give. That makes this new code worth testing against synthetic archives, including a truncated one. The Hub download and upload around it stay thin. |
| N3 | A pure function that combines the five `complete-*` config dicts into `complete.json` | New command with a tiny core. Order, fields and provenance can all be tested without the Hub. |
| N4 | One new `uad_data.hub` function that returns a repo file's current LFS sha256 | This is the only new network call. The loader's staleness check and the builder's safety check both use it, and the S1 and N2 fakes replace it. |

### Not unit-tested

- The exit codes of the two `main()` functions. Each is a line or two that turns
  the evaluator's overall result, or the load report, into an exit status. S5 and
  S1 cover those inputs, and the manual runs under [Acceptance](#acceptance)
  cover the wiring.
- Real Hub reads, the builder's pass over the Hub stream, and both uploads.

## Decisions

### Scope ([#1])

- The work covers exactly the internal datasets, splits and tasks that
  `complete-1..5` list.
- It lives in the shared `uad_data` loader and is wired into both eval and
  training.

### What n counts ([#4], [#8])

- n counts **clips** per internal dataset and selected split. Each clip still
  produces a row for every configured task, and for every prompt template when
  `randomize_prompt_format` is false. For example, EMNS train (tasks
  `classification` and `asr`, random templates) with n = 5 gives 5 clips and
  10 rows.
- The first n clips are taken in **archive order**, not metadata order. A given
  Hub revision always yields the same clips.
- A clip counts once at least one of its rows passes the run's row filter.
- A clip listed in two selected splits counts toward both.
- A clip whose row fails to render still counts toward n. It is not replaced by
  the next clip ([#8]).
- Reading an internal dataset's archive stops once every selected split is
  satisfied:
  - a capped split is satisfied at n clips;
  - an uncapped split is satisfied once every clip its metadata lists has been
    found.

  If some clips are missing, reading continues to the end of the archive, as it
  does today. Early stopping applies to regular runs too.

### Requesting splits ([#5], [#7])

- These all stay a single string: the loader's `split`, the `--split` flag on
  both command lines, and `dataset_split` in both config classes and the
  notebook. The string is written the Hugging Face way:
  - one name, e.g. `test`;
  - several names joined with `+`, e.g. `validation+test`;
  - `all`, meaning every split that the internal dataset's run config entry
    lists. An entry with no `splits` means every registered split.
- An internal dataset's selected splits are the requested splits that its run
  config entry lists.
- When the request doesn't match:
  - An unknown split name (e.g. `tset`) is an error.
  - A requested split that an entry doesn't list is skipped for that internal
    dataset, with a log line naming both.
  - If no internal dataset matches at all, the run fails with an error. This
    now applies to eval as well as training.
- `dataset_split` defaults to unset. It resolves to `all` when
  `clips_per_split` is set, and otherwise to `test` for eval or `train` for
  training.
  - An explicit value always wins.
  - Resolution happens in `EvalConfig` and `TrainConfig`, so the command line
    and the notebook behave the same way.
  - `load_uad_dataset` still requires an explicit `split`.
- Training prints a prominent warning when the selected splits include `test`,
  except in a smoke run.
- All of this applies to every run, not only smoke runs.

### Reading archives ([#5])

- Each internal dataset's archive is read once per run, however many splits are
  selected.
  - The metadata for all selected splits is combined before reading starts.
  - Each archive file produces rows for every selected split that lists it.
- `load_uad_dataset` still returns one flat list of rows.
  - Rows are grouped by internal dataset, in run config order.
  - Within an internal dataset they follow archive order, so different splits
    are mixed together.
  - Each row's own `split` field records where it came from.

### The cap ([#7], [#11])

- The command-line flag `--clips-per-split N` sets the cap, on both eval and
  training. The value is stored as `clips_per_split` on `EvalConfig` and
  `TrainConfig`, and passed on as `load_uad_dataset(clips_per_split=...)`.
  - It must be a positive integer.
  - Leaving the flag out means no cap: a regular run. There is no built-in
    value.
  - The cap does not live in the run config.
- `--max-samples` is removed:
  - from both command lines and both config classes;
  - from `load_uad_dataset`, which loses its `max_samples` argument;
  - from `Evaluator.evaluate`, which no longer truncates a second time.
- There is no smoke-only copy of `complete.json`. Smoke runs use the regular
  one, e.g. `--json-config complete.json --clips-per-split 2`.
- Docs use `--clips-per-split 5` in their smoke-run examples ([#11]).

### Where a run reads each archive from ([#7] amendment, [#11])

- **`stream` passed explicitly:** that choice wins. Smoke archives are bypassed,
  and the full archive is streamed or downloaded as asked. This is how to check
  a smoke run against the real archive. **(grill)**
- **`clips_per_split` unset:** the full archive is downloaded and cached, as
  today.
- **`clips_per_split` ≤ N, the run config's row filter is `all_pass`, and a
  fresh smoke archive exists for the internal dataset:** the run uses
  `smoke/<name>.tar.gz`. It's fetched with the normal cached Hub download and
  matched against the same metadata JSONs as before.
- **Anything else:** the full archive is streamed, with a log line. That covers:
  - n > N;
  - an internal dataset without a smoke archive;
  - a stale smoke archive;
  - a row filter other than `all_pass`. A smoke archive holds the first N clips
    whatever the filter, so with a rejecting filter it could give fewer clips,
    or different ones, than the full archive. **(grill)**
- **Staleness check:** on every run, the loader compares the full archive's
  sha256 recorded in the manifest with its current sha256 on the Hub, using one
  small metadata request.
  - If they differ, it logs a warning and falls back to the full archive.
  - If the check can't run (e.g. offline), it logs a warning and uses the smoke
    archive anyway.
- **A read error recorded at build time (grill):** a smoke archive should report
  what the full archive would. So when the manifest records a read error for an
  internal dataset (MLEnd_Intonation today), and a selected split of that
  dataset ends short of n clips, the loader reports a failed load. The report
  carries the recorded error and the number of rows kept. A full-archive run
  would hit the cut in exactly that situation. If every selected split reaches
  n (e.g. MLEnd train only), the recorded error isn't reported.

### Seed and prompt templates ([#4], [#7])

- With `randomize_prompt_format: true`, each row's template is picked by a
  random generator seeded from the seed plus the row's internal dataset, split,
  audio path and task.
  - The generator is separate from Python's shared `random`, so the reseeding
    that `RandomFilter` does can't disturb the picks.
  - A clip therefore gets the same template in every run that includes it,
    whatever n or `--split` is.
- `load_uad_dataset` takes `seed`, defaulting to 42.
  - Eval gains `--seed` and `EvalConfig.seed`, defaulting to 42.
  - Training gains `--seed`, defaulting to 42. It feeds the existing
    `TrainConfig.seed`, which already seeds the Trainer and now also seeds the
    loader. **(grill)** [#4] called this flag "existing", but `train/main.py`
    doesn't have it.

### Row statuses and passing ([#8])

- Every row ends with exactly one status:
  - `ok`: the model returned non-empty text;
  - `empty_output`: the prediction is empty once whitespace is stripped
    **(grill)**;
  - `render_error`;
  - `audio_error`: the audio couldn't be decoded;
  - `model_error`: the backend raised an exception.
- A **group** is one (internal dataset, split, task). A group passes when it has
  at least one row and every row is `ok`.
- A selected split that yields fewer than n clips gets a warning. A split that
  yields none leaves its groups with no rows, so those groups fail.
- Metrics are for information only and never decide pass or fail.

### Preliminary metrics ([#8])

Each task has one metric. It's computed against the row's plain answer field,
not against the rendered `output`, and uses only the `evaluate` package.

| Task | Metric | Compared against | Rule |
|---|---|---|---|
| `asr` | WER | `transcription` | |
| `english_translation` | WER | `english_translation` | |
| `caption` | WER | `caption` | only a crude similarity measure |
| `classification` | hit rate | `category` | `category` appears in the prediction, ignoring case and punctuation and treating `_` as a space |
| `commonsense` | hit rate | `commonsense_answer` | the prediction starts with the answer's choice letter (answers look like `B. …`) |
| `qa` | hit rate | `answer` | every number in `answer` appears in the prediction (answers look like `The result is 102.`) |

These six are all the tasks `complete-1..5` use ([#3]).

**How a group's value is combined (grill):**

- **Rows included:** rows with a prediction, i.e. `ok` and `empty_output`. An
  empty prediction is a real miss: WER 1.0 for that row, hit 0. Rows with
  `render_error`, `audio_error` or `model_error` have no prediction, are left
  out, and already fail their group.
- **WER tasks:** the group value is corpus-level, one `wer.compute` over the
  included rows, as today's single WER is. It is not the mean of per-row WERs.
- **Hit-rate tasks:** the group value is the mean of the included rows' hits.
- **No included rows:** the group has no metric value (`null`).
- **Per row:** each row's own value (its WER, or 0/1) goes into
  `results.jsonl`.

### Failures ([#8])

- **Smoke runs keep going; regular runs stop.** Regular runs raise on the first
  error, as they do today.
- **Load failures (smoke runs):** when an internal dataset fails to load (e.g.
  a truncated archive or a missing metadata file), the loader keeps the rows it
  read before the failure, records the error, and moves on to the next internal
  dataset.
- **Evaluator errors (smoke runs):**
  - a row whose audio fails to decode becomes `audio_error` and is left out of
    its batch;
  - when the backend raises, every row in that batch becomes `model_error`, and
    the run moves on to the next batch.

### Load report ([#8])

- `load_uad_dataset` returns the usable rows as a `list` subclass that carries a
  `.report`. Existing `rows = load_uad_dataset(...)` callers keep working, and
  training needs no filtering.
- The report lists:
  - clips found per (internal dataset, split), compared with n;
  - internal datasets that failed to load, with the error and the number of
    rows kept;
  - rows that failed to render, with internal dataset, split, task, audio path
    and error.

### Eval output ([#8])

Every eval run produces these, not only smoke runs.

- **`results.jsonl`:** one line per row, including rows that failed to render
  (taken from the load report).
  - Each line has the row's own `originating_dataset`, `split`, `task` and
    `audio_path`.
  - It also has `status`, `error`, the plain answer, `prediction` and the row's
    metric value.
  - It keeps today's other fields: `index`, `model_choice`, `model`, `dataset`,
    `sys_inst`, `prompt`, and `ground_truth` (the rendered `output`).
    **(grill)**
  - Its `split` is now the row's own. It no longer records the split given on
    the command line.
- **`summary.json` and the console table:** these replace today's single WER
  line.
  - They have one entry per group, with clips found and n, the row count, a
    count for each status, the metric name and value, and pass/fail.
  - They also give an overall result and list the internal datasets that failed
    to load.
- **Return value (grill):** `Evaluator.evaluate` returns the same content as
  `summary.json`, plus a `rows` list holding the same records as the
  `results.jsonl` lines (no audio). Callers such as the notebook can then show
  results even when no `output_dir` is set. The old `wer`, `num_samples`,
  `predictions` and `references` keys go.
- **Exit code:**
  - A smoke run exits non-zero if any group fails or any internal dataset
    failed to load.
  - A regular run keeps today's behaviour: it exits 0 unless something raised,
    even when a group fails (e.g. on `empty_output`). **(grill)**

### Training smoke runs ([#8])

- The load report is printed before training starts.
- Training uses the usable rows as normal.
- There's no per-group model check. Per-group reporting is eval-only, as
  decided while charting.
- The run exits non-zero at the end if the report lists any problem.
  **(grill)** "Problem" matches what fails eval:
  - an internal dataset that failed to load;
  - a row that failed to render;
  - a selected split with **zero** clips.

  A split with some clips but fewer than n only gets a warning, as in eval.

### Renames ([#7])

Wherever "sample" means a row, it becomes "row":

- `Sample` → `Row`, and `uad_data/sample.py` → `uad_data/row.py`;
- `iter_samples` → `iter_rows`;
- `SampleFilter` → `RowFilter`, and `include_sample` → `include_row`;
- the evaluator's `samples` variables → `rows`;
- the run config key `sample_filter` → `row_filter`. No config uses the old key,
  so it is no longer accepted. A run config that still has it raises a
  `ValueError` saying the key is now `row_filter`, so an old filter can't be
  dropped silently. **(grill)**

Not renamed here: `max_samples` in `uad_data/audio_utils.py`, which really
means audio samples. [#12] renamed it separately, to `max_audio_samples`.

### Colab notebook ([#7])

- The config form gets three fields, all passed to `load_uad_dataset`:
  - `clips_per_split`: blank means no cap;
  - `dataset_split`: blank uses the defaults above; it accepts `test`,
    `validation+test` or `all`;
  - `seed`: defaults to 42.
- The form's `max_samples` field is removed.
- **Results cells (grill):**
  - **Cell 14** prints the same per-group table and overall result as the
    console. It calls the evaluator's own table-printing function, so the
    command line and the notebook can't drift apart.
  - **Cell 15** previews the first `PREVIEW_N` entries of `results["rows"]`,
    with rows that aren't `ok` listed first. Each entry shows its group,
    status, plain answer and prediction.

### `complete.json` ([#6])

- A new Hub run config, `universal_audio_dataset_configs/complete.json`:
  - `"name": "Complete UAD"` and `randomize_prompt_format: true`;
  - its datasets are those of `complete-1` through `complete-5`, in that order
    (23 internal datasets, 41 (dataset, split) pairs, 50 groups).
- It's an ordinary run config. Runs refer to it by name, and there's no copy in
  this repo's `configs/`.
- `complete-1..5` stay the source of truth.
  - A small command in this repo builds `complete.json` from the five and
    records which configs it was built from.
  - Workflow: edit the five, rerun the command, upload the result.
- **The command (grill):** `python -m uad_data.build_complete_config`.
  - It reads the five configs from the Hub.
  - It writes to `--output`, which defaults to `outputs/complete.json`.
    `outputs/` is gitignored, and the file must not land in the working
    directory, where it would shadow the Hub copy.
  - It records its sources in a top-level `"built_from": ["complete-1.json", …,
    "complete-5.json"]`.
  - It fails if an internal dataset appears in more than one of the five.
- The build session uploads `complete.json` using the user's saved Hugging Face
  login, after checking with the user first.

### Smoke archives ([#11])

- Every internal dataset in `complete.json` gets a smoke archive at
  `smoke/<name>.tar.gz`.
  - It holds the first N = 10 clips of each registered split, in archive order,
    with tar entries and bytes copied as-is.
  - All smoke archives together come to about 165 MiB at most.
- `smoke/manifest.json` records, for each internal dataset:
  - N;
  - the number of clips each split received;
  - the sha256 of the full archive the smoke archive was built from;
  - the Hub revision it was built at;
  - the read error that stopped the build, if any. **(grill)**
- The archives are built with
  `python -m uad_data.build_smoke_archives --clips-per-split 10`.
  - It reads each full archive in one sequential pass, with the metadata JSONs
    of every registered split loaded.
  - It copies each file that belongs to a split still under N clips.
  - It stops once every split has N clips, or when the archive ends or fails.
    So MLEnd_Intonation gets its readable train clips and no validation or test
    clips.
- **Source:** by default the command streams from the Hub (a one-off ~17 GiB).
  `--source-dir` points it at a local copy of the dataset instead.
- **Safety check:** in either mode, the command first checks that each source
  archive's sha256 matches the Hub's current LFS sha256, and refuses to build
  from a stale source. How each mode does it **(grill)**:
  - **Hub mode:** resolve `main`'s current commit, read the archive's LFS
    sha256 at that commit, and stream the archive pinned to that commit. Both
    the commit and the sha256 go into the manifest.
  - **`--source-dir` mode:** hash the local file and compare it with the Hub's
    current LFS sha256.
- **Upload:** the build session runs the command. Then, after confirming with
  the user, it uploads the smoke archives, the manifest and the Hub README
  changes in one Hub commit. The README changes are:
  - "Repository layout" describes `smoke/` and `manifest.json`;
  - the onboarding guide gets a step to rebuild smoke archives after adding or
    changing an internal dataset;
  - "Intended usage" uses `clips_per_split`, `seed` and the `+`/`all` split
    syntax instead of `max_samples=None` and a single split name **(grill)**;
  - "Config format" names `row_filter` instead of `sample_filter` **(grill)**;
  - any other card text this work makes out of date (see
    [Keeping docs and code current](#keeping-docs-and-code-current-grill)).

### Keeping docs and code current (grill)

The build fixes **everything this work makes out of date**, not just the smoke
test commands that [#7] lists. It sweeps READMEs, the notebook, docstrings and
comments, runnable scripts, and the Hub card. Places known to need changes:

- **`README.md`:**
  - the flow-diagram labels ("lazy stream when max_samples", "WER + …");
  - the library snippet (`split`, `max_samples`, the `stream` comment);
  - the onboarding smoke test (`--max-samples 5`) and its explanation;
  - the note that the evaluator reports WER for every task.
- **`eval/README.md`:** the flow diagram (WER), and the module table, which
  needs the new `eval/metrics.py` and a new description for `evaluator.py`.
- **`train/README.md`:** the smoke test.
- **`FINETUNING.md`:**
  - the smoke test and its explanation;
  - the flag table: the `--split` default, removing `--max-samples`, adding
    `--clips-per-split` and `--seed`;
  - the troubleshooting tip that suggests `--max-samples 32`.
- **`eval/colab_eval.ipynb`:** the form, the config and load cells, the results
  cells, and any markdown that describes them.
- **Docstrings and comments** that mention `max_samples`, `sample_filter`,
  samples-as-rows, a single split, or a single WER. Known ones: the module
  docstring of `uad_data/loader.py`; `hub.open_archive_stream`;
  `uad_data/filters.py`; `uad_data/json_config_loader.py`;
  `uad_data/collection.py`; `eval/evaluator.py`.
- **The Hub card:** as listed under [Smoke archives](#smoke-archives-11).
- **Dated records (grill):**
  - **`docs/research/complete-configs-load-check.py`** is pinned rather than
    rewritten. It gets a header saying it runs against commit `8f0b6f9` (e.g.
    from a `git worktree` at that commit), and its code stays as is. It exists
    to reproduce its findings, and rewriting it for the new loader would change
    what it measures.
  - **`docs/research/*.md` findings** stay as they are. Each already names the
    commit it was checked against.
  - **`MIGRATION.md`** is a living doc. The build updates the parts people
    still use: the module list (`row.py`) and the run and test commands. The
    history of the July changes keeps its original names, with a "(now `Row`)"
    note.

Statements that are already wrong today, before this work, are being fixed
separately and ahead of the build.

## Approach

Build in this order. Each numbered slice is one coherent `/build` run, and each
slice assumes the ones before it.

### 1. Renames

- Apply the [renames](#renames-7) mechanically: files, classes, functions, the
  config key, `UadCollection.sample_filter` → `row_filter`, docstrings and
  error messages.
- The existing tests, with names updated, stay green.
- S2 checks that `row_filter` is read, and that a config with `sample_filter`
  raises a `ValueError` naming `row_filter`.
- Pin `docs/research/complete-configs-load-check.py` to commit `8f0b6f9` with a
  header note; don't change its code.
- In `MIGRATION.md`, rename `sample.py` to `row.py` in the module list, and add
  "(now `Row`)" where the history mentions `Sample`.

### 2. Loader: splits, clip cap, single read, seed (S1)

In `uad_data/loader.py`:

- **Parse `split`.** The loader parses the string itself: split on `+` and
  check each name is `train`, `validation` or `test`, or accept `all`.
- **Read each internal dataset**, in run config order:
  1. work out its selected splits, logging any it skips;
  2. merge those splits' metadata into a map from `audio_path` to its record in
     each split;
  3. open the archive once;
  4. for each member, emit rows for every selected split that lists it and is
     still under its cap;
  5. stop when every selected split is satisfied.
- **Replace `max_samples` with `clips_per_split`.** For now, setting it turns on
  streaming; slice 6 changes that to prefer smoke archives.
- **Add `seed=42`.** Pick each template with its own `random.Random`, seeded
  from a stable digest (`hashlib`) of the seed, internal dataset, split, audio
  path and task. Don't use `hash()`: it's salted per process, so the picks
  would change between runs.
- **Raise** when no internal dataset matches the request.

S1 tests to write at minimum:

- the split strings `validation+test`, `all` and `tset`;
- a Clotho-like archive that stores `test/` first, where a per-split cap still
  reaches train and validation;
- a clip listed in two splits;
- the EMNS-like case: n = 5 gives 5 clips and 10 rows;
- archive order differing from metadata order;
- two selected splits served by one archive open;
- early stop, for both capped and uncapped splits, with the fake stream
  recording how far it was read;
- one clip getting the same template at n = 1 and n = 2 and across different
  `split` values;
- `RandomFilter` not changing which templates get picked.

### 3. Loader: load report and smoke-run tolerance (S1)

- **Return type.** Return a `list` subclass with `.report`. For each
  (internal dataset, selected split), the report also records the configured
  tasks, so the evaluator can list every group, including empty ones.
- **Smoke runs** (`clips_per_split` set):
  - When an internal dataset fails to load (a truncated gzip, a missing
    metadata file), keep the rows read so far, record the failure and continue.
  - When a row fails to render, record it and still count the clip.
- **Regular runs** raise, as they do today.
- **Shortfalls.** Log a warning for every selected split with fewer than n
  clips.

### 4. Configs, command lines and training wiring (S3, S4)

- **`EvalConfig` and `TrainConfig`:**
  - add `clips_per_split: int | None`, which must be positive when set;
  - make `dataset_split: str | None = None`, resolved in `__post_init__`;
  - add `EvalConfig.seed = 42`;
  - remove `max_samples`.
- **`eval.main` and `train.main`:**
  - add `--clips-per-split` and `--seed` to both;
  - make `--split` default to `None`;
  - remove `--max-samples`;
  - pass `clips_per_split` and `seed` to the loader.
- **`train.main`:**
  - print the load report;
  - warn when `test` is among the selected splits (read from the report's
    (internal dataset, split) entries), unless this is a smoke run;
  - after saving, exit non-zero in a smoke run if the report lists a failed
    load, a render failure or a selected split with zero clips.
- **Load report:** give it one "has problems" check that implements that rule,
  so S1 can test the rule itself. `train.main` then only maps the result to an
  exit status.

### 5. Evaluator (N1, S5)

- **`eval/metrics.py` (N1):**
  - the per-row rules in the [table above](#preliminary-metrics-8);
  - the group aggregate: corpus-level WER, mean hit rate, taken over `ok` and
    `empty_output` rows, and `null` when a group has none of those rows.
  - Check whether `evaluate`'s WER accepts an empty prediction. If it doesn't,
    score an empty prediction as all deletions (WER 1.0), which is the decided
    meaning.
- **`Evaluator.evaluate(rows)`:**
  - no truncation;
  - a status for every row, with whitespace stripped before the empty check;
  - smoke-run tolerance for audio and backend errors;
  - render failures from `rows.report` go into `results.jsonl` and into their
    groups;
  - groups come from the report as well as from the rows, so empty groups
    exist and fail;
  - write `results.jsonl` (today's fields plus the new ones) and
    `summary.json`, print the console table, and give an overall result;
  - return the summary plus a `rows` list of the `results.jsonl` records;
  - expose the function that prints the per-group table, so the notebook can
    reuse it.
- **`eval.main`:**
  - print the summary;
  - in a smoke run, exit non-zero when the overall result fails or any
    internal dataset failed to load;
  - in a regular run, exit 0 unless something raised.

### 6. Smoke archives and `complete.json`, in code (N2, N3, N4, S1)

- **N4:** add a `hub` function that returns a file's current LFS sha256 (e.g.
  via `HfApi.get_paths_info`).
- **N3 and `python -m uad_data.build_complete_config`:** build `complete.json`
  from `complete-1..5`.
  - Test the pure core with the five configs as dicts. N3 checks order, the
    fixed `name` and `randomize_prompt_format`, `built_from`, and the error on
    a duplicate internal dataset.
  - `built_from` must be a **top-level** field. Config loading ignores unknown
    top-level fields, but `InternalDatasetJsonConfig(**d)` rejects unknown keys
    inside a dataset entry. S2 checks that the generated file loads.
  - `--output` defaults to `outputs/complete.json`. `_resolve_config_path`
    prefers a local `complete.json` in the working directory over the Hub copy,
    so the output must not land there.
- **N2 and `python -m uad_data.build_smoke_archives`:**
  - Take the list of internal datasets from `complete.json`, and each
    dataset's metadata for every registered split.
  - Run the sha256 check first:
    - **Hub mode:** resolve `main`'s commit and read the LFS sha256 there
      (N4), then stream the archive pinned to that commit.
    - **`--source-dir` mode:** hash the local file and compare it with N4's
      answer.
  - Write the `smoke/<name>.tar.gz` files and `smoke/manifest.json` to an
    output directory. The manifest includes the read error that stopped a
    build, if any.
  - Share the "which splits does this member count toward" logic with the
    loader, so a smoke archive holds exactly what the loader would take from
    the full archive.
- **Loader:** choose each archive's source as described under
  [Where a run reads each archive from](#where-a-run-reads-each-archive-from-7-amendment-11):
  - the explicit-`stream` bypass;
  - the `all_pass`-only rule;
  - the staleness check;
  - reporting a recorded build error as a failed load when a selected split
    ends short.

  Fetch the manifest with `hub.download_file`.

### 7. Docs and notebook

- **Notebook:** update the form, cells 7 and 10, and the results cells 14–15,
  as decided under [Colab notebook](#colab-notebook-7).
- **Docs and code comments:** do the sweep, including `MIGRATION.md`'s run and
  test commands, as described under
  [Keeping docs and code current](#keeping-docs-and-code-current-grill). Replace
  `--max-samples` with `--clips-per-split 5`, and cover `--seed`, the split
  syntax and the per-group output.
- **Final check:** `git grep` for `max_samples`, `max-samples`, `sample_filter`,
  `Sample`, `iter_samples`, `WER` and `split=` to catch anything the list missed.

### 8. Hub changes (confirm with the user before each upload)

1. Generate `complete.json` and upload it.
2. Run `build_smoke_archives --clips-per-split 10` against the Hub.
3. Upload `smoke/*.tar.gz`, `smoke/manifest.json` and the Hub README changes in
   one commit. The README changes are listed under
   [Smoke archives](#smoke-archives-11): layout, onboarding, intended usage,
   config format, and anything else this work makes out of date.

### 9. Manual acceptance runs

See [Acceptance](#acceptance).

## Out of scope

- **Unlisted datasets and tasks:** registered datasets and tasks that
  `complete-1..5` don't list, such as libricss, libricss_subseg, SparseLibriMix,
  MustardPP-SingleTurn, Clotho's commonsense tasks and slurp_real's intent
  tasks ([#1]).
- **Random or stratified selection:** smoke runs take the first n only ([#1]).
- **A full per-task metrics suite:** only the six preliminary metrics are in
  scope ([#1], [#8]).
- **Per-group model checks in training smoke runs** ([#8]).
- **A smoke-specific run config or a smoke copy of `complete.json`** ([#7]).
- **Putting the cap in the run config:** it stays on the command line ([#7]).
- **Per-split archives:** repacking every internal dataset into one archive per
  split is a ~47 GiB migration ([#11]).
- **Fixing the Hub data defects:** the truncated MLEnd_Intonation archive, the
  missing slurp_real train clips, the mislabelled hi_kia test records, and
  OpenMic's one record per instrument. The Hub README's "Known issues" section
  documents them (Hub commit `c4e1b16`). Smoke runs report these defects rather
  than fix them. OpenMic still collapses to the last record per clip.
- **Stale Hub configs:** cleaning up `complete-3-original.json`,
  `timestamp_search.json` and `demo.json` ([#6]).
- **Leftover helper modules in the Hub repo:** removing them is a separate,
  already pending effort ([#1]).
- **Renaming `audio_utils`' `max_samples`:** done separately in [#12]
  (now `max_audio_samples`).
- **Counting metadata records that never matched an archive entry:** [#3]
  suggested it, but the report in [#8] only compares clips found with n.

## Acceptance

### Offline test suite

- [ ] Every test in `tests/` passes with no network access.
- [ ] S1 shows that:
  - [ ] per-split caps are counted in clips and taken in archive order;
  - [ ] each archive is opened once per internal dataset;
  - [ ] reading stops early, for both capped and uncapped splits;
  - [ ] template picks are stable across n and `split`;
  - [ ] in smoke runs, a truncated archive and a missing metadata file are
    recorded while the other internal datasets still load;
  - [ ] in regular runs, both of those raise;
  - [ ] each archive source is chosen as decided: all four fallbacks, the
    explicit-`stream` bypass and the offline case;
  - [ ] a build error recorded in the manifest is reported as a failed load
    only when a selected split ends short;
  - [ ] a config with `sample_filter` raises (S2);
  - [ ] the report's "has problems" rule flags failed loads, render failures
    and zero-clip splits, but not a partial shortfall.
- [ ] N2 shows that a smoke archive holds exactly the first N clips of each
  split, in archive order and byte-identical, and that a truncated source gives
  the clips before the cut plus a recorded error.
- [ ] N3 shows the combined config's order, fields and `built_from`, and the
  error on a duplicate internal dataset.
- [ ] N1 covers each of the six metric rules with matching and non-matching
  predictions, plus the group aggregates (corpus WER, mean hit rate, empty
  prediction as a miss, `null` with no included rows).
- [ ] S5 shows that:
  - [ ] every status is produced, and a whitespace-only prediction counts as
    `empty_output`;
  - [ ] a group with no rows fails;
  - [ ] `results.jsonl` has today's fields plus the new ones, and
    `summary.json` has the decided fields;
  - [ ] `evaluate` returns the summary plus `rows`;
  - [ ] smoke runs continue past audio and backend errors, and regular runs
    raise.

### Code and docs

- [ ] Outside `docs/specs/` and `docs/research/`:
  - [ ] `git grep -n "max_samples\|max-samples"` finds nothing (the
    `uad_data/audio_utils.py` variable was renamed in [#12]);
  - [ ] `git grep -nw "Sample\|SampleFilter\|iter_samples\|include_sample\|sample_filter"`
    finds only `MIGRATION.md`'s history notes, each marked "(now `Row`)".
- [ ] `docs/research/complete-configs-load-check.py` starts with a note pinning
  it to commit `8f0b6f9`.
- [ ] Notebook cells 14–15 show the per-group table and a preview of
  `results["rows"]`, with rows that aren't `ok` listed first.
- [ ] Smoke-run examples in `README.md`, `FINETUNING.md`, `train/README.md` and
  the notebook use `--clips-per-split 5` (or the notebook field).
- [ ] The notebook form has `clips_per_split`, `dataset_split` and `seed`, and
  no `max_samples`.

### Hub

- [ ] `universal_audio_dataset_configs/complete.json` lists the 23 internal
  datasets of `complete-1..5` in order, and records its sources.
- [ ] `smoke/` holds one archive for each of those 23 internal datasets, plus
  `manifest.json`. The manifest has N, clip counts per split, the source
  sha256, the revision, and MLEnd_Intonation's read error.
- [ ] The Hub README describes `smoke/`, and its onboarding guide has the
  rebuild step.
- [ ] The Hub README's usage example and config format use the new names, and
  nothing on the card mentions `max_samples` or `sample_filter`.

### Manual runs

- [ ] **Loader only, no GPU:**
  `load_uad_dataset(json_config_path="complete.json", split="all", clips_per_split=5, token=...)`
  - [ ] reads every internal dataset from its smoke archive, with no fallback
    log lines;
  - [ ] finds 5 clips in each of the 41 (dataset, split) pairs, except
    MLEnd_Intonation validation and test, which find 0 and get a warning;
  - [ ] reports exactly one failed load: MLEnd_Intonation, with the manifest's
    recorded read error and its train rows kept.
- [ ] **Eval smoke run on a GPU (e.g. Colab):**
  `python -m eval.main --model GEMMA-4 --json-config complete.json --clips-per-split 5 --output-dir <dir>`
  - [ ] finishes and writes `results.jsonl`, plus `summary.json` with 50 group
    entries;
  - [ ] marks the four MLEnd_Intonation validation and test groups as failed,
    with 0 clips, and lists MLEnd_Intonation as failed to load;
  - [ ] exits non-zero;
  - [ ] has an explanation for any other failing group.
- [ ] **Training smoke run on a GPU:**
  `python -m train.main --model GEMMA-4 --json-config complete.json --clips-per-split 5 --epochs 1 --output-dir outputs/smoke`
  - [ ] prints the load report, trains and saves;
  - [ ] exits non-zero, because of MLEnd_Intonation's failed load and its
    zero-clip splits.
- [ ] **Regular eval run:** a run without `--clips-per-split` on
  `configs/clotho_config.json` still defaults to `test`, writes the per-group
  summary, and exits 0.

[#1]: https://github.com/derkmed/universal_audio_instruct/issues/1
[#2]: https://github.com/derkmed/universal_audio_instruct/issues/2
[#3]: https://github.com/derkmed/universal_audio_instruct/issues/3
[#4]: https://github.com/derkmed/universal_audio_instruct/issues/4
[#5]: https://github.com/derkmed/universal_audio_instruct/issues/5
[#6]: https://github.com/derkmed/universal_audio_instruct/issues/6
[#7]: https://github.com/derkmed/universal_audio_instruct/issues/7
[#8]: https://github.com/derkmed/universal_audio_instruct/issues/8
[#11]: https://github.com/derkmed/universal_audio_instruct/issues/11
[#12]: https://github.com/derkmed/universal_audio_instruct/issues/12
