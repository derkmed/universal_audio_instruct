# Do complete-1..5 load cleanly today?

Research for [#3](https://github.com/derkmed/universal_audio_instruct/issues/3), part of the map [#1 Smoke runs across every UAD dataset](https://github.com/derkmed/universal_audio_instruct/issues/1). Checked on 2026-09-16 against `main` at `8f0b6f9`.

> **Naming note.** This document predates the `Sample` → `Row` rename
> ([ADR-0002](../adr/0002-row-not-example.md)) and uses the names the code had at
> the commit it was run against: `Sample`, `Sample.to_output`, `iter_samples`,
> `sample_filter`, `sample.py`. Today those are `Row`, `Row.to_output`,
> `_iter_rows`, `row_filter` and `row.py`. The text is kept as written so it
> still describes the run it reports.

> **Caveat: local copy.** Every data finding here comes from a **local copy** of the private Hub repo (`C:\Users\derek\Desktop\UAD-DEV\Universal-Audio-Understanding\`), not from the Hub itself. `huggingface_hub` downloaded that copy between 11:53 and 12:22 UTC on 2026-07-08, from Hub commit **`09f8a0f303ede57828b07886993c5490f29cbf09`**. Every in-scope file still matches that snapshot (see [Local copy vs the Hub](#local-copy-vs-the-hub)). So the findings hold for that commit. Whether the Hub has changed since was **not** checked, because this research didn't touch the network. Before acting on the fixes below, check the four affected files against the Hub.

## Answer

**All five configs pass validation, but they don't all load cleanly. complete-3 crashes on load, and complete-2 and complete-4 silently drop a lot of data.** complete-1 and complete-5 load cleanly.

- **Checks that pass everywhere.** `uad_data` accepts all five configs. Every split's metadata file exists and parses. Every configured task has exactly one prompt file. Every record has every key in `Task.features`. Every record renders through the real `Sample.to_output` without error.
- **complete-3: fails.** `data/MLEnd_Intonation/MLEnd_Intonation.tar.gz` is **truncated**.
  - Its gzip stream ends without an end-of-stream marker, and the cut falls inside entry 11,670. `tarfile` raises `ReadError: unexpected end of data` there. The first 11,669 entries are readable, and all of them are train clips.
  - Of the 25,734 clips in MLEnd's metadata, 14,065 are unreachable: 8,923 of 20,592 train clips, all 2,557 validation clips and all 2,585 test clips.
  - A complete-3 run on **validation or test** reads the 414 MB archive, gets zero MLEnd rows, and then **raises**. `load_uad_dataset` builds the full row list before returning (`loader.py:183`), so the whole load fails, including the MELD rows already read.
  - A **train** run survives a small row cap, because MLEnd's first train clip is entry 0. An uncapped train run yields 11,669 clips (23,338 rows) and then raises.
  - The file's sha256 matches the Hub snapshot's `lfs_sha256`, so the Hub served this truncated file at `09f8a0f`. This isn't a broken local download.
- **complete-4: loads, but silently drops data.**
  - **slurp_real/train:** the archive holds only 12,423 of the 35,199 train clips in the metadata. The other 22,776 (65%) are skipped. Every metadata record from index 12,424 onward is missing, which suggests the archive was built from a partial list.
  - **OpenMic:** the metadata has one record per (clip, instrument). `_load_split_metadata` keys records by `audio_path`, so only the last record per clip survives. That drops 16,041 of 30,956 train records and 5,493 of 10,578 test records.
    - The surviving labels are often negative: 7,880 of the 14,915 train rows that survive have `relevance` < 0.5, and 5,332 of those have `relevance` 0.0.
    - The classification output renders `category` whatever its relevance is.
- **complete-2: loads, but silently drops data.** The hi_kia test metadata (`hi_kia_test.json`) lists 488 clips. 429 of them are hi_kia's own 381 train and 48 validation clips, relabelled `test/<file>`, and those paths don't exist in the archive. Only the 59 real test clips load. If someone "fixed" the paths, train and validation clips would leak into test.
- **Minor issues.**
  - MELD is missing one train clip and one validation clip.
  - Single duplicate `audio_path` records get collapsed in Clotho (each split), MusicCapsCommonSense/test and slurp_real/train.
  - complete-1 and complete-5 have nothing that loses more than one clip.

| Config | Validates | Loads on train | Loads on validation | Loads on test | Silent loss |
|---|---|---|---|---|---|
| complete-1 | yes | yes | yes | yes | 1 duplicate record per Clotho split |
| complete-2 | yes | yes | yes | yes | **hi_kia/test: 429 of 488 clips** |
| complete-3 | yes | capped: yes. Uncapped: **no** (ReadError at MLEnd) | **no** (ReadError at MLEnd) | **no** (ReadError at MLEnd) | MLEnd/train 8,923 clips (then the crash); MELD 1+1 clips; MusicCaps 1 duplicate |
| complete-4 | yes | yes | yes | yes | **slurp_real/train: 22,776 clips; OpenMic: 16,041 + 5,493 records** |
| complete-5 | yes | yes | yes | yes | none |

**What this means for smoke runs.** Today's problems come in two kinds:
- **Hard failures:** exceptions in the middle of the archive stream. Only MLEnd has one.
- **Silent losses:** metadata records whose `audio_path` has no archive entry, and duplicates that get collapsed.

The loader handles the two kinds very differently. A hard failure kills the whole `load_uad_dataset` call. A silent loss leaves no trace. A smoke run should therefore catch errors **per internal dataset**, so one truncated archive doesn't hide the other 22. It should also **count metadata records that never matched an archive entry**. With a first-n cap, most silent losses stay invisible unless something counts them, because the first n clips usually do match.

## Results by (dataset, split, task)

Key: PASS = no problem. WARN = the split loads but a few records are lost. **FAIL** = the split crashes, or more than 5% of its clips are silently skipped. The numbered columns match the checks under [Method](#method). Column 3 gives the number of prompt-template combinations in brackets. Column 6 compares unique metadata `audio_path`s with regular-file archive entries of exactly the same name. Column 7 runs the real `loader.iter_samples` capped at 3 rows, in stream mode, which is how a capped run reads (`loader.py:171-172`).

| Config | Dataset | Split | Task | 1 Config | 2 Metadata | 3 Prompt | 4 Fields | 5 Render | 6 Audio paths (matched / unique) | 7 Real loader, 3 rows | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| complete-1 | AESDD | train | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 605 / 605 | PASS 3 rows |  |
| complete-1 | AudioMNIST | train | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 30,000 / 30,000 | PASS 3 rows |  |
| complete-1 | AudioMNISTCommonSense | test | qa | PASS | PASS | PASS (1) | PASS | PASS | PASS 30,000 / 30,000 | PASS 3 rows |  |
| complete-1 | Clotho | train | caption | PASS | WARN | PASS (2) | PASS | PASS | PASS 3,838 / 3,838 | PASS 3 rows | 1 duplicate record |
| complete-1 | Clotho | validation | caption | PASS | WARN | PASS (2) | PASS | PASS | PASS 1,044 / 1,044 | PASS 3 rows | 1 duplicate record |
| complete-1 | Clotho | test | caption | PASS | WARN | PASS (2) | PASS | PASS | PASS 1,044 / 1,044 | PASS 3 rows | 1 duplicate record |
| complete-1 | colombian_spanish | train | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 4,903 / 4,903 | PASS 3 rows |  |
| complete-1 | EMNS | train | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 1,181 / 1,181 | PASS 3 rows |  |
| complete-1 | EMNS | train | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 1,181 / 1,181 | PASS 3 rows |  |
| complete-2 | Ewe_BibleTTS | train | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 22,192 / 22,192 | PASS 3 rows |  |
| complete-2 | esc50 | train | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 1,200 / 1,200 | PASS 3 rows |  |
| complete-2 | esc50 | test | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 400 / 400 | PASS 3 rows |  |
| complete-2 | esc50 | validation | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 400 / 400 | PASS 3 rows |  |
| complete-2 | hi_kia | train | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 381 / 381 | PASS 3 rows |  |
| complete-2 | hi_kia | validation | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 48 / 48 | PASS 3 rows |  |
| complete-2 | hi_kia | test | classification | PASS | PASS | PASS (1) | PASS | PASS | **FAIL** 59 / 488 | PASS 3 rows | 429 test records point at train/validation clips |
| complete-3 | Lingala_BibleTTS | train | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 11,093 / 11,093 | PASS 3 rows |  |
| complete-3 | MELD | train | classification | PASS | PASS | PASS (1) | PASS | PASS | WARN 9,988 / 9,989 | PASS 3 rows | `dia125_utt3.wav` not in archive |
| complete-3 | MELD | test | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 2,610 / 2,610 | PASS 3 rows |  |
| complete-3 | MELD | validation | classification | PASS | PASS | PASS (1) | PASS | PASS | WARN 1,108 / 1,109 | PASS 3 rows | `dia110_utt7.wav` not in archive |
| complete-3 | MESD | train | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 862 / 862 | PASS 3 rows |  |
| complete-3 | MLEnd_Intonation | train | classification | PASS | PASS | PASS (1) | PASS | PASS | **FAIL** 11,669 / 20,592 (then ReadError) | PASS 3 rows; uncapped: 11,669 clips then `ReadError` | archive truncated inside entry 11,670 |
| complete-3 | MLEnd_Intonation | train | asr | PASS | PASS | PASS (1) | PASS | PASS | **FAIL** 11,669 / 20,592 (then ReadError) | PASS 3 rows; uncapped: 11,669 clips then `ReadError` | archive truncated inside entry 11,670 |
| complete-3 | MLEnd_Intonation | validation | classification | PASS | PASS | PASS (1) | PASS | PASS | **FAIL** 0 / 2,557 (then ReadError) | **FAIL** 0 rows, `ReadError` | archive truncated; no validation clips before the cut |
| complete-3 | MLEnd_Intonation | validation | asr | PASS | PASS | PASS (1) | PASS | PASS | **FAIL** 0 / 2,557 (then ReadError) | **FAIL** 0 rows, `ReadError` | archive truncated; no validation clips before the cut |
| complete-3 | MLEnd_Intonation | test | classification | PASS | PASS | PASS (1) | PASS | PASS | **FAIL** 0 / 2,585 (then ReadError) | **FAIL** 0 rows, `ReadError` | archive truncated; no test clips before the cut |
| complete-3 | MLEnd_Intonation | test | asr | PASS | PASS | PASS (1) | PASS | PASS | **FAIL** 0 / 2,585 (then ReadError) | **FAIL** 0 rows, `ReadError` | archive truncated; no test clips before the cut |
| complete-3 | MusicCapsCommonSense | test | commonsense | PASS | WARN | PASS (1) | PASS | PASS | PASS 149 / 149 | PASS 3 rows | 1 duplicate record |
| complete-4 | nigerian_english | train | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 3,359 / 3,359 | PASS 3 rows |  |
| complete-4 | OpenMic | train | classification | PASS | WARN | PASS (1) | PASS | PASS | PASS 14,915 / 14,915 | PASS 3 rows | one record per (clip, instrument); 16,041 records collapsed away; 7,880 of the 14,915 kept labels have relevance < 0.5 |
| complete-4 | OpenMic | test | classification | PASS | WARN | PASS (1) | PASS | PASS | PASS 5,085 / 5,085 | PASS 3 rows | one record per (clip, instrument); 5,493 records collapsed away; 2,653 of the 5,085 kept labels have relevance < 0.5 |
| complete-4 | peruvian_spanish | train | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 5,447 / 5,447 | PASS 3 rows |  |
| complete-4 | ravnursson_faroese | train | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 65,616 / 65,616 | PASS 3 rows |  |
| complete-4 | ravnursson_faroese | validation | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 3,331 / 3,331 | PASS 3 rows |  |
| complete-4 | ravnursson_faroese | test | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 3,002 / 3,002 | PASS 3 rows |  |
| complete-4 | slurp_real | train | classification | PASS | WARN | PASS (1) | PASS | PASS | **FAIL** 12,423 / 35,199 | PASS 3 rows | archive holds only 12,423 train clips; 1 duplicate record |
| complete-4 | slurp_real | train | asr | PASS | WARN | PASS (1) | PASS | PASS | **FAIL** 12,423 / 35,199 | PASS 3 rows | archive holds only 12,423 train clips; 1 duplicate record |
| complete-4 | slurp_real | validation | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 6,116 / 6,116 | PASS 3 rows |  |
| complete-4 | slurp_real | validation | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 6,116 / 6,116 | PASS 3 rows |  |
| complete-4 | slurp_real | test | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 9,253 / 9,253 | PASS 3 rows |  |
| complete-4 | slurp_real | test | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 9,253 / 9,253 | PASS 3 rows |  |
| complete-5 | URDU | train | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 400 / 400 | PASS 3 rows |  |
| complete-5 | VIVOS | train | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 11,660 / 11,660 | PASS 3 rows |  |
| complete-5 | VIVOS | train | english_translation | PASS | PASS | PASS (6) | PASS | PASS | PASS 11,660 / 11,660 | PASS 3 rows |  |
| complete-5 | VIVOS | test | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 760 / 760 | PASS 3 rows |  |
| complete-5 | VIVOS | test | english_translation | PASS | PASS | PASS (6) | PASS | PASS | PASS 760 / 760 | PASS 3 rows |  |
| complete-5 | VocalSound | train | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 15,531 / 15,531 | PASS 3 rows |  |
| complete-5 | VocalSound | test | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 3,591 / 3,591 | PASS 3 rows |  |
| complete-5 | VocalSound | validation | classification | PASS | PASS | PASS (1) | PASS | PASS | PASS 1,855 / 1,855 | PASS 3 rows |  |
| complete-5 | Yoruba_BibleTTS | train | asr | PASS | PASS | PASS (1) | PASS | PASS | PASS 7,491 / 7,491 | PASS 3 rows |  |

## Details and evidence

### 1. Config validation: passes for all five configs

`UniversalJsonConfig(filepath=...).toCollection()` succeeds for complete-1..5, with a fresh registry for each config. It also succeeds when all five load one after another in a single process. The checks it runs are:

- The dataset name must be in the registry: `json_config_loader.py:53-57`.
- Each task string goes through `Task(t)`: `json_config_loader.py:61`.
- Each split string goes through `datasets.Split(s)`: `json_config_loader.py:65`.
- The configured tasks and splits must be subsets of the registered ones: `json_config_loader.py:78-88`.

Every name in the five configs is registered in `uad_data/internal_datasets.py` with the configured tasks and splits. All five configs set `"randomize_prompt_format": true` and no `sample_filter`, so each run uses `AllPassFilter` and renders one random template per (clip, task) (`loader.py:58-59`).

Each run loads **one** split (`loader.py:99`, `collection.py:44-45`), so covering every split takes up to three runs per config. Here is what each split reaches:

| Config | train | validation | test |
|---|---|---|---|
| complete-1 | AESDD, AudioMNIST, Clotho, colombian_spanish, EMNS | Clotho | AudioMNISTCommonSense, Clotho |
| complete-2 | Ewe_BibleTTS, esc50, hi_kia | esc50, hi_kia | esc50, hi_kia |
| complete-3 | Lingala_BibleTTS, MELD, MESD, MLEnd_Intonation | MELD, MLEnd_Intonation | MELD, MLEnd_Intonation, MusicCapsCommonSense |
| complete-4 | nigerian_english, OpenMic, peruvian_spanish, ravnursson_faroese, slurp_real | ravnursson_faroese, slurp_real | OpenMic, ravnursson_faroese, slurp_real |
| complete-5 | URDU, VIVOS, VocalSound, Yoruba_BibleTTS | VocalSound | VIVOS, VocalSound |

### 2. Metadata files: all exist and parse

The loader looks for `data/<name>/<name>_<split>.json` (`internal_dataset.py:45-53`). The file exists and parses for all 41 (dataset, split) pairs; see `data/<name>/` in the local copy. Every record has an `audio_path` string.

`_load_split_metadata` builds a dict keyed by `audio_path` (`loader.py:40`), so when a path appears more than once, only the last record survives:

| Split | Records | Unique paths | Lost | Do the lost copies differ? |
|---|---|---|---|---|
| OpenMic/train | 30,956 | 14,915 | 16,041 | yes: `category` differs for 8,564 paths |
| OpenMic/test | 10,578 | 5,085 | 5,493 | yes: `category` differs for 2,960 paths |
| Clotho/train, validation, test | 3,839 / 1,045 / 1,045 | 3,838 / 1,044 / 1,044 | 1 each | train and validation: no. Test: only in `hard_commonsense_*`, which complete-1 doesn't configure |
| MusicCapsCommonSense/test | 150 | 149 | 1 | no |
| slurp_real/train | 35,200 | 35,199 | 1 | `transcription` differs for `train/audio-1502309360-headset.wav` |

The OpenMic metadata holds one record per (clip, instrument), with a `relevance` field. In `data/OpenMic/OpenMic_train.json`:
- 12,749 of 30,956 records have `relevance` = 0.0, and 5,009 more are below 0.5.
- After the collapse, 7,880 of the 14,915 surviving train labels are below 0.5, and 5,332 of those are exactly 0.0.
- 3,278 train clips have a record at 0.5 or above, but the loader keeps a record below 0.5 instead.

The classification template renders `{{category}}.` as the expected output (`prompts/classification.json`) whatever the relevance is. I'm inferring from the field name that low-relevance records are negative labels in OpenMIC-2018; I didn't confirm that against the upstream dataset.

### 3. Prompt files: exactly one per configured task, and all valid

`_get_prompt_templates` globs `PROMPTS_DIR/*.json` and builds a `PromptFilepath` for **every** file (`loader.py:46-47`). Any malformed prompt file would therefore break every task. All 11 files in `prompts/` parse, carry a valid `task`, and are pure ASCII. That matters because `prompts.py:58` and `json_config_loader.py:114` open files without an explicit encoding.

The configured tasks, with the files that match them and their (system instruction × prompt × output) combinations:

| Task | File | Combinations |
|---|---|---|
| asr | `asr.json` | 1 |
| classification | `classification.json` | 1 |
| caption | `caption.json` | 2 |
| qa | `qa.json` | 1 |
| commonsense | `commonsense.json` | 1 |
| english_translation | `english_translation.json` | 6 |

Matching compares the file's `task` field with `task.value` (`prompts.py:112`).

For every configured task, the `{{ }}` variables in the templates equal the `Task.features` keys (`tasks.py:37-79`).

### 4. Fields: nothing missing

`Sample.build_*` indexes `metadata[k]` for every `k` in `task.features` (`sample.py:59, 65, 71`). No record in any of the 41 splits lacks a key its configured tasks need, and none has an empty, whitespace-only or non-string value for such a key.

### 5. Rendering: no exceptions

The script rendered **every** surviving record with **every** template combination through the real `Sample.to_output` (`sample.py:30-53`). That's 50 (dataset, split, task) groups and 499,810 rows, with zero exceptions and zero undefined template variables. Example rows:
- AESDD classification gives the system instruction `Determine which of the following categories best describes or categorizes the provided audio: anger, disgust, fear, happiness, sadness.` and the output `disgust.`
- nigerian_english asr gives the output `Alexandra shared a photo with you`.

### 6. Audio paths: three real gaps and two single-clip gaps

Every in-scope archive was read end to end with `tarfile.open(path, mode="r|gz")`, the same mode the loader uses (`loader.py:75, 82`). The loader keeps a record only when a regular-file entry's `member.name` exactly equals its `audio_path` (`loader.py:111-116`). Across all 23 archives there were no near-misses: no leading `./`, backslash, case, Unicode-normalisation or whitespace differences. None of the matched paths appears twice in its archive, and none has a symlink or hardlink entry.

- **MLEnd_Intonation.** `data/MLEnd_Intonation/MLEnd_Intonation.tar.gz` is 434,159,616 bytes, an exact multiple of 4,096. `gzip` raises `EOFError: Compressed file ended before the end-of-stream marker was reached` after 822,083,584 decompressed bytes. `tarfile` lists 11,670 entries, all under `train/`, and the loader can extract 11,669 of them. Every `validation/` and `test/` path, and 8,923 `train/` paths, have no entry at all.
- **slurp_real/train.** The archive has 27,792 files: 12,423 under `train/`, 6,116 under `val/` and 9,253 under `test/`. The validation and test metadata match their archive entries exactly. In `data/slurp_real/slurp_real_train.json`, every record from index 12,424 onward is missing, 22,776 unique paths in all, with headset and far-field recordings missing in similar numbers. No archive entry has the same file name. The archive reads cleanly, so this is a build-time gap, not truncation.
- **hi_kia/test.** The archive has 381 `train/`, 48 `validation/` and 59 `test/` files. `data/hi_kia/hi_kia_test.json` has 488 records. For 429 of them, the file name exists only under `train/` (381) or `validation/` (48), and the category is the same as in the train or validation record. The test metadata seems to list the whole dataset with a `test/` prefix. The loader silently keeps the 59 real test clips.
- **MELD.** `MELD/train_wavs/dia125_utt3.wav` and `MELD/validation_wavs/dia110_utt7.wav` are absent. A file with the same name exists under `MELD/test_wavs/`, but MELD numbers dialogues separately in each split (every split starts at `dia0_utt0`), so that's a different clip, not a near-miss. The archive also holds two `.DS_Store` files, which do no harm.

### 7. Anything else

- **End-to-end runs.** The real `loader.iter_samples` ran against local files for all 41 (dataset, split) pairs, with a 3-row cap in stream mode. The `hub.download_file` and `hub.open_archive_stream` stand-ins return local paths or handles, as in `tests/test_loader.py`. Every run except MLEnd validation and test returned 3 well-formed rows:
  - the task is one of the configured tasks;
  - `split` and `originating_dataset` are correct;
  - the audio bytes and output are non-empty.

  MLEnd validation and test raised `ReadError` with 0 rows, whether capped or not. Uncapped MLEnd train yielded 23,338 rows and then raised.
- **No split has zero matching clips**, except MLEnd validation and test.
- **The registry is mutated as configs load (latent).** `toInternalDataset` calls `set_tasks`/`set_splits` on the module-level objects in `DATASETS_DIRECTORY` (`json_config_loader.py:75-88`). A later config in the same process is then checked against the narrowed lists. After complete-1 loads, a config asking for Clotho `commonsense` fails with `ValueError: Task: [<Task.COMMONSENSE: 'commonsense'>] requested of Clotho, whichonly contains tasks: [<Task.CAPTION: 'caption'>]`. That message is quoted verbatim, including the missing space. complete-1..5 name disjoint datasets, so loading all five in one process works today. A smoke-run harness that writes its own configs, or loads overlapping ones in one process, would hit this.
- **Rows carry extra keys.** A row is a copy of the whole metadata record (`sample.py:41`), including the `tasks` list of `Task` enums that `loader.py:40` adds. The evaluator only serialises named string fields (`eval/evaluator.py:79-91`), so this is harmless today. Serialising whole rows, for example to JSON, would fail on the enums.
- **Prompt files are re-read for every clip.** `_get_prompt_templates` re-globs and re-parses every prompt file for each (clip, task) (`loader.py:123`). That costs time on uncapped runs but doesn't cause errors.
- **Some splits start deep in their archive.** This affects how long a capped run takes; [#2](https://github.com/derkmed/universal_audio_instruct/issues/2) covers it. The first matching entry is at index 68,618 for ravnursson_faroese/validation, 18,539 for slurp_real/test, 12,737 for MELD/validation, 12,423 for slurp_real/validation, 11,662 for VIVOS/test, and 2,090 / 1,045 for Clotho train / validation. The capped end-to-end runs on local disk took 18.6 s, 10.1 s, 40.6 s, 6.9 s, 10.5 s and 21.1 s / 10.4 s respectively.
- **ravnursson_faroese** has both `_dev.json` and `_validation.json`. They're byte-identical (same sha256), and the loader uses `_validation`.
- **Out of scope but noticed.** `intonation_detection.json` uses `{{intonation}}`, but `Task.INTONATION_DETECTION.features` gives `category`. `sentiment_analysis.json` uses `{{sentiment}}`, but the feature key is `Sentiment`. Real jinja2 would render both as empty strings without any error. `action_classification` and `intent_detection`, which slurp_real registers, have no prompt file, so `_get_prompt_templates` would raise `RuntimeError` (`loader.py:48-50`). complete-1..5 use none of these tasks.

## Method

`docs/research/complete-configs-load-check.py` (standard library only) does the following:

1. **Stands in for missing packages.** It registers minimal fakes for `datasets`, `jinja2` and `huggingface_hub` in `sys.modules`, then imports the repo's real `uad_data` package, `__init__` included. It points `uad_data.prompts.PROMPTS_DIR` at the local `prompts/` folder.
2. **Checks config validation.** It runs the real `UniversalJsonConfig(...).toCollection()` for each config, reloading `uad_data.internal_datasets` before each one. It then loads all five in one process without reloading, and runs a probe that shows the registry being mutated.
3. **Checks metadata, fields and rendering.** For each (dataset, split), it resolves the path with the real `split_metadata_path` and `hub.to_repo_path`, and loads it with the real `_load_split_metadata`. For each configured task, it matches prompts with the real `_get_prompt_templates`, counts missing, empty and non-string feature values in the raw records, and renders every kept record with every template through the real `Sample.to_output`.
4. **Lists the archives.** It streams each archive with `tarfile` mode `r|gz` (4 worker processes), records every entry name and type, and caches the listing. It then compares entries with the metadata paths and classifies near-misses.
5. **Compares against the Hub snapshot.** It checks the local copy against the snapshot `huggingface_hub` recorded at download time (below).
6. **Runs the loader end to end.** With `--e2e-rows 3`, it runs the real `loader.iter_samples` for each (dataset, split), capped at 3 rows. It also runs uncapped passes for any dataset whose archive failed to list.

Run used for this report: `python docs/research/complete-configs-load-check.py --cache-dir <scratch> --workers 4 --e2e-rows 3 --json-out <scratch>/full_e2e.json`. Listing all 23 archives (about 45 GB) took about 2 minutes on this machine.

### What the fakes assume

- **`datasets.Split`** is a `str` subclass. `Split("x")` returns an object that compares and hashes equal to `"x"`, gives `"x"` from `str()` and f-strings, and must match `^\w+(\.\w+)*$`. I reproduced this from memory of `datasets/splits.py` (`NamedSplit.__eq__` / `__hash__` / `__str__`), and **didn't verify it against an installed `datasets`**. The results depend on it in three places:
  - `Split("validation") in [Split.VALIDATION]` (`json_config_loader.py:84`, `collection.py:45`);
  - the metadata file name `..._validation.json` (`internal_dataset.py:52`);
  - the `split` field in each row (`loader.py:37`).

  `Split("all")`, which returns `NamedSplitAll` in real `datasets`, isn't modelled. No config uses it.
- **`datasets.Value` / `Version` / `Audio`** are inert holders. Only `Task.features.keys()` matters to the loader.
- **`jinja2.Template`** supports only `{{ name }}` and raises on any other jinja syntax (none was hit). A missing name renders as `""`, like jinja2's default `Undefined`, and any other value renders with `str()`. Every template in `prompts/` uses only this syntax.
- **`huggingface_hub`** raises on every call. `uad_data.hub.download_file` and `open_archive_stream` are replaced by local-file stand-ins only in the end-to-end step.
- **Not exercised:** audio decoding (`uad_data/audio_utils.py`, `soundfile`/`librosa`), for example whether OpenMic's `.ogg` and ravnursson's `.flac` clips decode.

## Local copy vs the Hub

The local copy holds `huggingface_hub`'s `--local-dir` metadata:
- `.cache/huggingface/trees/09f8a0f303ede57828b07886993c5490f29cbf09.json` lists every file at that commit, with its size and, for LFS files, its `lfs_sha256`.
- `.cache/huggingface/download/<path>.metadata` records the commit and etag for each download.

The script checked the 80 in-scope files: 23 archives, 41 split metadata files, 11 prompt files and 5 configs.
- **All 80** are in the snapshot, with the local size equal to the snapshot size.
- **For the 64 LFS files**, the recorded download etag equals the snapshot's `lfs_sha256`.
- **For MLEnd_Intonation.tar.gz**, the sha256 was also recomputed locally: `acfa27611011a11061af86adb5483c9d687f5b0f7dc617890c9c5e5f32c9fd0d`. That equals the snapshot's `lfs_sha256` (`lfs_size` 434159616).

So the truncation isn't a partial local download: it's what the Hub served at `09f8a0f`.

**Still to check against the Hub** (needs a token; not done here):
- whether `main` has moved past `09f8a0f` (`HfApi().list_repo_commits`);
- the current `lfs.size` / `lfs.sha256` of the four affected files (`HfApi().get_paths_info(repo_id, [...], repo_type="dataset")`):
  - `data/MLEnd_Intonation/MLEnd_Intonation.tar.gz`
  - `data/slurp_real/slurp_real.tar.gz`
  - `data/slurp_real/slurp_real_train.json`
  - `data/hi_kia/hi_kia_test.json`

## Reproduce

```
python docs/research/complete-configs-load-check.py \
    --uad-root C:/Users/derek/Desktop/UAD-DEV/Universal-Audio-Understanding \
    --cache-dir <scratch>/listings --workers 4 --e2e-rows 3 --json-out <scratch>/results.json
```

Add `--skip-archives` for the fast checks only (1–5, about 10 s). Use `--list-only` to fill the archive-listing cache on its own.
