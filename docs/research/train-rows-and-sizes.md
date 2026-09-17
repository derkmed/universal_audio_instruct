# How many rows does each internal dataset contribute, and how big are the downloads?

Research for [#44](https://github.com/derkmed/universal_audio_instruct/issues/44), part of the map [#42 Finetune Gemma and Qwen on GCP](https://github.com/derkmed/universal_audio_instruct/issues/42). Checked on 2026-09-17 against the Hub's `main` at **`e80eab1`** (`e80eab1b158283a4e451f8f87ce971e70e3ae2e6`) and this repo's `main`. Script: [`train-rows-and-sizes.py`](train-rows-and-sizes.py). Vocabulary (internal dataset, split, clip, row, run config) as in [`CONTEXT.md`](../../CONTEXT.md).

## Answer

- **Train split, complete-1..5 together:** 21 internal datasets have a train split, giving **234,754 clips and 271,687 rows**. MLEnd_Intonation counts here with its 11,669 reachable clips, but an uncapped load of it raises (see below).
- **Excluding the three datasets whose train data is broken on the Hub** (MLEnd_Intonation, slurp_real, OpenMic): **195,747 clips and 208,588 rows**.
- **Validation split:** 8 of the 21 datasets have one in their run config. They give 13,902 clips and 20,018 rows, but MLEnd_Intonation's validation split yields 0 clips. The other 13 have no configured validation split, so the map's 80/20 clip-level holdout applies to them.
- **Downloads:** all 23 archives in complete-1..5 total **50.76 GB (47.3 GiB)**. A train-only run needs the 21 archives of datasets with a train split: **49.55 GB (46.1 GiB)**, or **44.56 GB** without the three broken datasets.
- **Disk on the VM:** the loader streams each cached `.tar.gz` and never extracts it, so about **50 GB** of HF cache covers the data. Extracting everything would add **68.2 GB** of audio, but nothing requires it.
- **RAM is the bigger constraint.** `load_uad_dataset` returns a Python list, and every row holds its clip's audio bytes. A full train load therefore keeps about **55.8 GB (52.0 GiB)** of raw audio in host memory.
- **Imbalance:** ravnursson_faroese alone supplies **24%** of train rows. The top six datasets (ravnursson_faroese, AudioMNIST, slurp_real, MLEnd_Intonation, VIVOS, Ewe_BibleTTS) supply **70%**. The five smallest (esc50, MESD, AESDD, URDU, hi_kia) supply 1.3% together.
- **By task:** ASR accounts for about **69%** of rows (187,034 of 271,687).

## How rows are counted

- In every complete-N config, `randomize_prompt_format` is `true` (checked by the script). The loader then picks **one** template per (clip, task) (`uad_data/loader.py:63-64`).
- None of these configs uses asr_timestamp_search, so each task renders a clip once (`uad_data/tasks.py:85`, `Task.utterance_indices`).
- So **rows = clips with an archive entry × configured tasks**.
- The script checks this by rendering every such record through the real `Sample.to_output` with empty audio bytes. The render needs no audio. There were zero render errors.
- **Clips** are unique metadata `audio_path`s that have a regular-file entry of exactly that name in the archive. These are the only records the loader keeps (`uad_data/loader.py:40`, `:111-125`).
- **Archive listings.** No archive was downloaded. The script listed the local copy (`C:\Users\derek\Desktop\UAD-DEV\Universal-Audio-Understanding`), and only because each local archive's recorded `lfs_sha256` and size **equal the Hub's current LFS sha256 and size** for all 23 archives.
- **Current Hub files.** Metadata JSONs, configs and prompts were downloaded fresh from the Hub at `e80eab1`. File sizes come from `HfApi.list_repo_tree` at that commit.

## Per internal dataset

Sizes are in decimal units: MB = 10^6 bytes and GB = 10^9 bytes. "Archive" is the `.tar.gz` that has to be downloaded. "Train audio" is the uncompressed size of the train clips' files inside the archive, which is roughly the RAM a full load takes. "Val" is the validation split **as configured in the run config**.

| Config | Internal dataset | Tasks | Train clips | Train rows per task | Val (clips / rows) | Archive | Train audio | Notes |
|---|---|---|---:|---|---|---:|---:|---|
| complete-1 | AESDD | classification | 605 | classification 605 | none | 352 MB | 409 MB | Hub file is `aesdd.tar.gz` (lower case); the registry's `data_url` matches it |
| complete-1 | AudioMNIST | asr | 30,000 | asr 30,000 | none | 994 MB | 1.85 GB | |
| complete-1 | AudioMNISTCommonSense | qa | — (test only) | — | none | 994 MB | — | test: 30,000 clips / rows; not needed for training |
| complete-1 | Clotho | caption | 3,838 | caption 3,838 | 1,044 / 1,044 | 9.39 GB | 7.61 GB | largest archive; about 2 MB of audio per row; 1 duplicate record per split |
| complete-1 | colombian_spanish | asr | 4,903 | asr 4,903 | none | 1.61 GB | 2.62 GB | |
| complete-1 | EMNS | classification, asr | 1,181 | classification 1,181; asr 1,181 | none | 525 MB | 661 MB | |
| complete-2 | Ewe_BibleTTS | asr | 22,192 | asr 22,192 | none (Hub has an unregistered one: 186) | 5.91 GB | 8.83 GB | |
| complete-2 | esc50 | classification | 1,200 | classification 1,200 | 400 / 400 | 645 MB | 529 MB | |
| complete-2 | hi_kia | classification | 381 | classification 381 | 48 / 48 | 41 MB | 48 MB | **Hub defect in test only:** 429 of 488 test records point at train or validation clips, so only 59 load. Train and validation are fine |
| complete-3 | Lingala_BibleTTS | asr | 11,093 | asr 11,093 | none (Hub has an unregistered one: 202) | 4.26 GB | 6.22 GB | |
| complete-3 | MELD | classification | 9,988 | classification 9,988 | 1,108 / 1,108 | 6.05 GB | 5.54 GB | 1 train and 1 validation clip missing from the archive |
| complete-3 | MESD | classification | 862 | classification 862 | none | 86 MB | 90 MB | |
| complete-3 | MLEnd_Intonation | classification, asr | **11,669 of 20,592**, then `ReadError` | classification 11,669; asr 11,669 (only if the error is caught) | **0 of 2,557** (`ReadError`) | 434 MB | ≈ 817 MB (reachable part) | **Hub defect:** the archive is truncated. An uncapped train load raises and fails the whole `load_uad_dataset` call |
| complete-3 | MusicCapsCommonSense | commonsense | — (test only) | — | none | 218 MB | — | test: 149 clips / rows; not needed for training |
| complete-4 | nigerian_english | asr | 3,359 | asr 3,359 | none | 1.21 GB | 1.99 GB | |
| complete-4 | OpenMic | classification | 14,915 | classification 14,915 | none | 2.54 GB | 1.93 GB | **Hub defect:** 30,956 records, one per (clip, instrument). The loader keeps only the last record per clip, and 7,880 of the kept labels have relevance < 0.5 (likely negative labels) |
| complete-4 | peruvian_spanish | asr | 5,447 | asr 5,447 | none | 1.94 GB | 3.19 GB | |
| complete-4 | ravnursson_faroese | asr | **65,616** | asr 65,616 | 3,331 / 3,331 | 6.16 GB | 5.65 GB | largest by rows |
| complete-4 | slurp_real | classification, asr | **12,423 of 35,199** | classification 12,423; asr 12,423 | 6,116 / 12,232 | 2.01 GB | 1.07 GB | **Hub defect:** the archive holds only 12,423 train clips; the other 22,776 are skipped silently |
| complete-5 | URDU | classification | 400 | classification 400 | none | 76 MB | 88 MB | |
| complete-5 | VIVOS | asr, english_translation | 11,660 | asr 11,660; english_translation 11,660 | none | 1.47 GB | 1.72 GB | english_translation has 6 templates; one is picked at random per row |
| complete-5 | VocalSound | classification | 15,531 | classification 15,531 | 1,855 / 1,855 | 1.78 GB | 2.08 GB | |
| complete-5 | Yoruba_BibleTTS | asr | 7,491 | asr 7,491 | none (Hub has an unregistered one: 63) | 2.05 GB | 2.85 GB | |

**Validation splits.**
- **Configured in a run config (8 datasets):** Clotho, esc50, hi_kia, MELD, MLEnd_Intonation (0 clips), ravnursson_faroese, slurp_real and VocalSound. Each run config configures every split the registry lists for the dataset. The script compared the run configs with `uad_data/internal_datasets.py`.
- **Not in a run config (3 datasets):** Ewe_, Lingala_ and Yoruba_BibleTTS have `_validation.json` and `_test.json` on the Hub, with 186/66, 202/63 and 63/40 clips, and all of those clips are in the archives. But the registry lists only `train` for them (`uad_data/internal_datasets.py:97, 127, 276`), so a run config that asked for their validation split would fail validation.
  - **Map question:** should these use the Hub's validation split instead of an 80/20 holdout? That needs a registry change.
- **Size after the per-dataset cap.** With the map's cap of about 200 validation rows per dataset, the 7 configured validation splits that load give at most about 1,400 rows. The other 13 datasets, plus MLEnd_Intonation if it stays, take theirs from the holdout.

## Totals

| Scope | Datasets | Train clips | Train rows | Archives to download | Train audio (raw) |
|---|---:|---:|---:|---:|---:|
| Every dataset with a train split in complete-1..5 | 21 | 234,754 | 271,687 | 49.55 GB | 55.81 GB |
| The same, without MLEnd_Intonation, slurp_real and OpenMic | 18 | 195,747 | 208,588 | 44.56 GB | 51.99 GB |
| All 23 archives in complete-1..5 (including the two test-only datasets) | 23 | — | — | 50.76 GB | 68.22 GB (all splits) |

Train rows by task (21 datasets): asr 187,034, classification 69,155, english_translation 11,660, caption 3,838. The slurp_real, MLEnd_Intonation and EMNS rows are split between asr and classification.

## Imbalance in a plain concatenation (train rows, 21 datasets)

| Internal dataset | Rows | Share | Cumulative |
|---|---:|---:|---:|
| ravnursson_faroese | 65,616 | 24.2% | 24.2% |
| AudioMNIST | 30,000 | 11.0% | 35.2% |
| slurp_real | 24,846 | 9.1% | 44.3% |
| MLEnd_Intonation | 23,338 | 8.6% | 52.9% |
| VIVOS | 23,320 | 8.6% | 61.5% |
| Ewe_BibleTTS | 22,192 | 8.2% | 69.7% |
| VocalSound | 15,531 | 5.7% | 75.4% |
| OpenMic | 14,915 | 5.5% | 80.9% |
| Lingala_BibleTTS | 11,093 | 4.1% | 85.0% |
| MELD | 9,988 | 3.7% | 88.6% |
| Yoruba_BibleTTS | 7,491 | 2.8% | 91.4% |
| peruvian_spanish | 5,447 | 2.0% | 93.4% |
| colombian_spanish | 4,903 | 1.8% | 95.2% |
| Clotho | 3,838 | 1.4% | 96.6% |
| nigerian_english | 3,359 | 1.2% | 97.9% |
| EMNS | 2,362 | 0.9% | 98.7% |
| esc50 | 1,200 | 0.4% | 99.2% |
| MESD | 862 | 0.3% | 99.5% |
| AESDD | 605 | 0.2% | 99.7% |
| URDU | 400 | 0.1% | 99.9% |
| hi_kia | 381 | 0.1% | 100.0% |

- **Without the three broken datasets**, ravnursson_faroese is 31.5% of 208,588 rows, and AudioMNIST, spoken digits only, is 14.4%.
- **Two narrow ASR sources dominate.** A plain shuffle means roughly one row in three is Faroese ASR and one in seven is a spoken digit. The smallest emotion and keyword datasets (hi_kia, URDU, AESDD, MESD) each contribute under 0.35%.
- **Row counts understate audio cost.** Clotho has 1.4% of rows but 13.6% of the train audio (7.6 GB), about 2 MB per row. Its clips are long, so it may dominate the step time and memory per batch.

## Disk and memory on the VM

- **Download path.** `train/main.py:99` calls `load_uad_dataset` without `max_samples`, so `stream` defaults to `False` (`uad_data/loader.py:182`). Each archive is then fetched whole into the HF cache with `hf_hub_download` (`uad_data/hub.py`, `download_file`) and read as a sequential `r|gz` stream. Nothing is extracted to disk.
  - **Disk for data:** about 49.6 GB of archives, plus a few MB of metadata and prompts.
  - **Budget more for the rest of the VM:** the venv, the base-model weights, checkpoints and a margin. Those are outside this ticket.
  - **Extraction is optional.** Extracting every archive, which nothing requires, would add 68.2 GB.
- **Host RAM.** `load_uad_dataset` builds the whole row list before returning (`uad_data/loader.py:193`, `return list(iter_samples(...))`).
  - Every row carries `audio.bytes` (`uad_data/sample.py`, `to_output`).
  - The rows of one clip share one `bytes` object (`file_bytes`, `uad_data/loader.py:125`), so the audio counts once per clip, not once per row.
  - A full train load therefore holds **about 55.8 GB** of raw audio, plus Python overhead for 271,687 dicts, before the Trainer starts.
  - **For the map:** either the VM needs well over 64 GB of RAM, or `train/` should stream or lazily decode rows.
- **Download time** wasn't measured; it's still an open question in #42.

## Known Hub defects, still present at `e80eab1`

All 23 archives have the same LFS sha256 as at `09f8a0f`, where [complete-configs-load-check.md](complete-configs-load-check.md) found these defects. The metadata counts here match that research too.
- **MLEnd_Intonation:** the archive is truncated inside entry 11,670, and `tarfile` raises `ReadError: unexpected end of data`. An uncapped train load yields 11,669 clips (23,338 rows) and then **fails the whole call**, rows from other datasets included. Validation and test yield nothing.
  - **Consequence:** a single `load_uad_dataset` over complete-3's train split cannot finish today.
- **slurp_real:** only 12,423 of 35,199 train clips are in the archive. This is a silent loss of 22,776 clips (65%). Validation (6,116) and test (9,253) are complete.
- **OpenMic:** there is one record per (clip, instrument), and the loader keeps only the last one. That keeps 14,915 of 30,956 train records, and many of the kept labels are likely negatives (relevance < 0.5).
- **hi_kia:** only the test split is affected (429 of 488 records are really train or validation clips). Train and validation are clean, so this doesn't affect finetuning.

## Method

1. `HfApi().dataset_info(...).sha` gives the current Hub commit. `HfApi().list_repo_tree(recursive=True)` at that commit gives each file's size and LFS sha256.
2. The script downloads `complete-1..5.json`, every `prompts/*.json` and every `data/<name>/<name>_{train,validation,test}.json` of the 23 datasets at that commit (`hf_hub_download` into a scratch directory).
3. For each config, it reloads `uad_data.internal_datasets`, because `toCollection` mutates the registry, and runs the real `UniversalJsonConfig(...).toCollection()`. Each archive path comes from the dataset's own `data_url`.
4. For each of the 23 archives, it checks that the local copy's recorded `lfs_sha256` and size equal the Hub's (all 23 do), then streams the local archive with `tarfile` `r|gz` to record every regular file's name and size.
5. For each (dataset, split), it loads metadata with the real `_load_split_metadata`, intersects it with the archive listing, picks templates with the real `_get_prompt_templates(task, randomize=True)`, and renders every matched clip for every task with `Sample.to_output` (empty audio), counting rows and errors.
6. **MLEnd_Intonation:** `tarfile` lists the header of the cut 11,670th entry, so the script's raw count is 11,670. This report uses 11,669, the number the real loader yields before raising (measured in [complete-configs-load-check.md](complete-configs-load-check.md)). Its audio byte total includes the cut entry, so it's approximate.

Run: `python docs/research/train-rows-and-sizes.py --cache-dir <scratch> --json-out <scratch>/rows.json`, using the repo venv with an HF token. With cached listings, it takes a few minutes.
