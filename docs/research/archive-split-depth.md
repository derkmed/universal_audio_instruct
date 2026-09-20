# How deep into each archive are the first clips of each split?

Research for [#2](https://github.com/derkmed/universal_audio_instruct/issues/2), part of [#1 Smoke runs across every UAD dataset](https://github.com/derkmed/universal_audio_instruct/issues/1). Written 2026-09-16.

Scope: every (internal dataset, split) listed in the run configs `complete-1.json` to `complete-5.json`. That is 23 internal datasets and 41 splits.

> **Naming note.** This document predates the `Sample` → `Row` rename
> ([ADR-0002](../adr/0002-row-not-example.md)) and uses the names the code had
> when it was written, such as `iter_samples` (now `_iter_rows`). The text is
> kept as written so it still describes the run it reports.

## Answer

- **Cheap (30 of 41 splits).** In archive order, the 20th clip appears within the first 37 MiB of compressed data. For 22 of these splits, it appears within the first 11 MiB. Reaching 20 clips in all 30 of these splits takes about 336 MiB of compressed data in total.
- **Expensive (8 splits).** These splits are stored behind other splits in the archive they share, so a lot has to be read before their 1st clip:

  | Split | Read before the 1st clip |
  |---|---|
  | ravnursson_faroese validation | 5.5 GiB |
  | MELD validation | 5.2 GiB |
  | Clotho train | 3.1 GiB |
  | Clotho validation | 1.5 GiB |
  | VIVOS test | 1.3 GiB |
  | slurp_real test | 1.3 GiB |
  | MELD train | 1.2 GiB |
  | slurp_real validation | 0.84 GiB |

  ravnursson_faroese train falls in between (235 MiB). Reaching the 20th clip of every split, one split at a time, takes 20.5 GiB. These 9 splits account for 20.2 GiB of it.
- **Impossible (2 splits).** The archive holds no MLEnd_Intonation validation or test clips at all. The archive is also truncated, and the loader raises `tarfile.ReadError` where it is cut. Any MLEnd_Intonation run that reads to that point fails, including train. Only a capped train run that stops early can succeed.
- **What "first" can mean.** Clips mostly do **not** appear in metadata order:
  - **Same order:** only 8 datasets store their configured splits in exactly metadata-JSON order (AESDD, AudioMNIST, AudioMNISTCommonSense, colombian_spanish, MESD, nigerian_english, peruvian_spanish, ravnursson_faroese).
  - **Nearly the same:** Clotho and MusicCapsCommonSense match apart from one duplicated record each.
  - **Blocks reordered:** URDU keeps metadata order within each class block, but stores the blocks in reverse order.
  - **Unrelated:** in all the others, the orders are unrelated (|Spearman rho| < 0.3).

  So "the first n records of `<name>_<split>.json`" can be anywhere in the stretch of the archive that holds the split. A single sequential read can cheaply serve only "the first n clips of the split in archive order". Even that is cheap only for the 30 splits above.
- **New Hub files.** The loader alone can't make the 8 expensive splits cheap. That needs a different data layout, such as per-split archives or a small smoke-test archive. A rough upper bound for an archive holding the first 20 clips of every split is ~330 MiB compressed. All 23 archives together are 47.3 GiB. Separately, MLEnd_Intonation's archive needs to be rebuilt and re-uploaded, whatever is decided for smoke runs.

## Caveat: this was measured on a local copy

All data facts here come from a local copy of the private Hub repo, `C:\Users\derek\Desktop\UAD-DEV\Universal-Audio-Understanding` (written `<local>` below), not from the Hub (`AudioInstruct/Universal-Audio-Understanding`). The Hub is canonical.

The local copy matches the Hub **as of commit `09f8a0f303ede57828b07886993c5490f29cbf09`** (fetched 2026-07-08):
- **Download records:** `huggingface_hub` recorded that commit for every file used here, in `<local>/.cache/huggingface/download/**/*.metadata`.
- **Archives:** the sha256 of all 23 archives equals the LFS etag recorded for that commit.
- **Metadata and configs:** all 57 metadata JSONs and the `complete-*.json` configs match their recorded git-blob etags.

If the Hub has moved past `09f8a0f` since then, re-run the script against a fresh copy. It needs no network access.

## Table: every (dataset, split)

Rows are in config order (complete-1 to complete-5). MiB = 2^20 bytes of **compressed** archive, counted from the start of the file.

**How to read the columns:**
- **"entries":** the 1-based number of tar members read up to and including that clip. This counts directories and macOS `._*` files too.
- **"Split spans":** the compressed position of the split's first and last clip.
- **"Metadata order kept?":** "yes, all" means the matched clips appear in exactly the metadata order. Otherwise the cell gives the Spearman rho between archive rank and metadata index.
- **"Clips (matched / records)":** unique clips found / records in the JSON.

| Dataset | Split | Archive MiB | Clips (matched / records) | 1st clip: entries, MiB (%) | 5th clip: entries, MiB | 20th clip: entries, MiB (%) | Split spans (% of archive) | Metadata order kept? | Unmatched paths | Duplicate records |
|---|---|---:|---:|---|---|---|---|---|---:|---:|
| AESDD | train | 335.5 | 605 / 605 | 2, 0.8 (0.2%) | 6, 2.6 | 21, 10.7 (3.2%) | 0.0% to 99.8% | yes, all | 0 | 0 |
| AudioMNIST | train | 948.2 | 30,000 / 30,000 | 2, 0.0 (0.0%) | 6, 0.2 | 21, 0.6 (0.1%) | 0.0% to 100.0% | yes, all | 0 | 0 |
| AudioMNISTCommonSense | test | 948.2 | 30,000 / 30,000 | 2, 0.0 (0.0%) | 6, 0.2 | 21, 0.6 (0.1%) | 0.0% to 100.0% | yes, all | 0 | 0 |
| Clotho | train | 8,951.7 | 3,838 / 3,839 | 2,094, 3,179.3 (35.5%) | 2,099, 3,186.5 | 2,114, 3,210.7 (35.9%) | 35.5% to 100.0% | no; rho 0.9984 | 0 | 1 |
| Clotho | validation | 8,951.7 | 1,044 / 1,045 | 1,048, 1,577.8 (17.6%) | 1,053, 1,583.7 | 1,068, 1,604.8 (17.9%) | 17.6% to 35.5% | no; rho 0.9943 | 0 | 1 |
| Clotho | test | 8,951.7 | 1,044 / 1,045 | 2, 1.3 (0.0%) | 7, 8.0 | 22, 29.4 (0.3%) | 0.0% to 17.6% | no; rho 0.9943 | 0 | 1 |
| colombian_spanish | train | 1,535.9 | 4,903 / 4,903 | 2, 0.5 (0.0%) | 6, 1.7 | 21, 6.4 (0.4%) | 0.0% to 100.0% | yes, all | 0 | 0 |
| EMNS | train | 500.6 | 1,181 / 1,181 | 2, 0.3 (0.1%) | 6, 2.2 | 21, 8.1 (1.6%) | 0.0% to 100.0% | no; rho -0.0357 | 0 | 0 |
| Ewe_BibleTTS | train | 5,636.0 | 22,192 / 22,192 | 69, 20.0 (0.4%) | 73, 21.1 | 88, 25.3 (0.4%) | 0.4% to 99.1% | no; rho 0.0022 | 0 | 0 |
| esc50 | train | 615.4 | 1,200 / 1,200 | 4, 0.4 (0.1%) | 14, 1.9 | 52, 7.8 (1.3%) | 0.0% to 99.8% | no; rho 0.0144 | 0 | 0 |
| esc50 | test | 615.4 | 400 / 400 | 12, 1.8 (0.3%) | 62, 9.6 | 220, 33.4 (5.4%) | 0.2% to 100.0% | no; rho -0.063 | 0 | 0 |
| esc50 | validation | 615.4 | 400 / 400 | 26, 4.0 (0.6%) | 70, 10.8 | 198, 29.6 (4.8%) | 0.6% to 99.9% | no; rho -0.1679 | 0 | 0 |
| hi_kia | train | 38.8 | 381 / 381 | 2, 0.1 (0.2%) | 6, 0.5 | 21, 1.6 (4.2%) | 0.0% to 80.5% | no; rho -0.0161 | 0 | 0 |
| hi_kia | validation | 38.8 | 48 / 48 | 444, 35.1 (90.6%) | 448, 35.4 | 463, 36.5 (94.1%) | 90.4% to 99.8% | no; rho 0.1546 | 0 | 0 |
| hi_kia | test | 38.8 | 59 / 488 | 384, 31.4 (80.9%) | 388, 31.6 | 403, 32.6 (84.0%) | 80.8% to 90.2% | no; rho 0.1447 | 429 | 0 |
| Lingala_BibleTTS | train | 4,063.3 | 11,093 / 11,093 | 2, 0.3 (0.0%) | 6, 1.9 | 21, 7.5 (0.2%) | 0.0% to 97.7% | no; rho 0.004 | 0 | 0 |
| MELD | train | 5,773.0 | 9,988 / 9,989 | 2,754, 1,189.4 (20.6%) | 2,758, 1,190.4 | 2,773, 1,198.0 (20.8%) | 20.6% to 92.0% | no; rho -0.0099 | 1 | 0 |
| MELD | test | 5,773.0 | 2,610 / 2,610 | 7, 0.1 (0.0%) | 11, 1.1 | 28, 9.2 (0.2%) | 0.0% to 20.6% | no; rho -0.0003 | 0 | 0 |
| MELD | validation | 5,773.0 | 1,108 / 1,109 | 12,742, 5,312.0 (92.0%) | 12,746, 5,314.6 | 12,761, 5,320.4 (92.2%) | 92.0% to 100.0% | no; rho 0.0745 | 1 | 0 |
| MESD | train | 81.9 | 862 / 862 | 4, 0.1 (0.1%) | 12, 0.4 | 42, 2.1 (2.6%) | 0.0% to 99.9% | yes, all | 0 | 0 |
| MLEnd_Intonation | train | 414.0 | 11,670 / 20,592 | 2, 0.0 (0.0%) | 6, 0.2 | 21, 0.6 (0.2%) | 0.0% to 100.0% | no; rho -0.0058 | 8,922 | 0 |
| MLEnd_Intonation | validation | 414.0 | 0 / 2,557 | not reached | not reached | not reached | - | n/a (no clips) | 2,557 | 0 |
| MLEnd_Intonation | test | 414.0 | 0 / 2,585 | not reached | not reached | not reached | - | n/a (no clips) | 2,585 | 0 |
| MusicCapsCommonSense | test | 207.7 | 149 / 150 | 2, 1.6 (0.8%) | 7, 8.9 | 22, 31.6 (15.2%) | 0.0% to 99.3% | no; rho 0.9611 | 0 | 1 |
| nigerian_english | train | 1,158.2 | 3,359 / 3,359 | 2, 0.2 (0.0%) | 6, 1.5 | 21, 6.2 (0.5%) | 0.0% to 100.0% | yes, all | 0 | 0 |
| OpenMic | train | 2,427.1 | 14,915 / 30,956 | 316, 0.2 (0.0%) | 328, 0.8 | 362, 2.9 (0.1%) | 0.0% to 100.0% | no; rho -0.0868 | 0 | 16,041 |
| OpenMic | test | 2,427.1 | 5,085 / 10,578 | 318, 0.3 (0.0%) | 364, 3.1 | 478, 10.0 (0.4%) | 0.0% to 100.0% | no; rho -0.1133 | 0 | 5,493 |
| peruvian_spanish | train | 1,848.2 | 5,447 / 5,447 | 2, 0.4 (0.0%) | 6, 1.8 | 21, 7.6 (0.4%) | 0.0% to 100.0% | yes, all | 0 | 0 |
| ravnursson_faroese | train | 5,875.6 | 65,616 / 65,616 | 3,006, 234.8 (4.0%) | 3,010, 235.1 | 3,025, 236.5 (4.0%) | 4.0% to 95.9% | yes, all | 0 | 0 |
| ravnursson_faroese | validation | 5,875.6 | 3,331 / 3,331 | 68,623, 5,634.1 (95.9%) | 68,627, 5,634.4 | 68,642, 5,635.3 (95.9%) | 95.9% to 100.0% | yes, all | 0 | 0 |
| ravnursson_faroese | test | 5,875.6 | 3,002 / 3,002 | 3, 0.1 (0.0%) | 7, 0.3 | 22, 1.4 (0.0%) | 0.0% to 4.0% | yes, all | 0 | 0 |
| slurp_real | train | 1,913.3 | 12,423 / 35,200 | 2, 0.1 (0.0%) | 6, 0.4 | 21, 1.5 (0.1%) | 0.0% to 45.1% | no; rho -0.0094 | 22,776 | 1 |
| slurp_real | validation | 1,913.3 | 6,116 / 6,116 | 12,426, 864.0 (45.2%) | 12,430, 864.3 | 12,445, 865.4 (45.2%) | 45.2% to 67.2% | no; rho -0.0222 | 0 | 0 |
| slurp_real | test | 1,913.3 | 9,253 / 9,253 | 18,543, 1,285.1 (67.2%) | 18,547, 1,285.4 | 18,562, 1,286.3 (67.2%) | 67.2% to 100.0% | no; rho -0.0022 | 0 | 0 |
| URDU | train | 72.1 | 400 / 400 | 6, 0.2 (0.2%) | 10, 0.8 | 25, 3.8 (5.3%) | 0.0% to 99.8% | no; rho -0.875 | 0 | 0 |
| VIVOS | train | 1,406.0 | 11,660 / 11,660 | 4, 0.1 (0.0%) | 8, 0.3 | 23, 1.6 (0.1%) | 0.0% to 95.2% | no; rho -0.1515 | 0 | 0 |
| VIVOS | test | 1,406.0 | 760 / 760 | 11,714, 1,339.1 (95.2%) | 11,718, 1,339.4 | 11,733, 1,340.6 (95.3%) | 95.2% to 100.0% | no; rho -0.2781 | 0 | 0 |
| VocalSound | train | 1,699.0 | 15,531 / 15,531 | 4, 0.0 (0.0%) | 14, 0.3 | 60, 2.1 (0.1%) | 0.0% to 100.0% | no; rho 0.0123 | 0 | 0 |
| VocalSound | test | 1,699.0 | 3,591 / 3,591 | 38, 1.2 (0.1%) | 66, 2.2 | 198, 7.9 (0.5%) | 0.1% to 99.9% | no; rho -0.0118 | 0 | 0 |
| VocalSound | validation | 1,699.0 | 1,855 / 1,855 | 10, 0.2 (0.0%) | 62, 2.1 | 310, 12.1 (0.7%) | 0.0% to 100.0% | no; rho -0.012 | 0 | 0 |
| Yoruba_BibleTTS | train | 1,958.0 | 7,491 / 7,491 | 2, 0.1 (0.0%) | 6, 0.9 | 21, 5.2 (0.3%) | 0.0% to 98.6% | no; rho 0.0138 | 0 | 0 |

## Table: every archive

- **"Files no JSON lists":** regular files that none of the dataset folder's metadata JSONs mention, configured or not.
- **"Pass seconds":** one full sequential read on local disk, including sha256. It says nothing about HTTP speed.

| Dataset | Archive bytes | Tar bytes (uncompressed) | Tar members (files / dirs) | Files no JSON lists | Pass seconds | sha256 = Hub etag? | Read error |
|---|---:|---:|---|---:|---:|---|---|
| AESDD | 351,846,149 | 409,310,208 | 605 / 1 | 0 | 2.3 | yes |  |
| AudioMNIST | 994,256,042 | 1,874,991,616 | 30,000 / 1 | 0 | 11.3 | yes |  |
| AudioMNISTCommonSense | 994,274,061 | 1,874,991,616 | 30,000 / 1 | 0 | 11.1 | yes |  |
| Clotho | 9,386,499,184 | 11,771,539,968 | 5,929 / 3 | 3 | 77.3 | yes |  |
| colombian_spanish | 1,610,537,073 | 2,624,486,912 | 4,903 / 1 | 0 | 12.0 | yes |  |
| EMNS | 524,965,152 | 661,840,384 | 1,181 / 1 | 0 | 3.5 | yes |  |
| Ewe_BibleTTS | 5,909,765,926 | 8,961,233,408 | 22,444 / 3 | 0 | 35.3 | yes |  |
| esc50 | 645,271,272 | 887,810,560 | 4,001 / 1 | 2,001 | 4.9 | yes |  |
| hi_kia | 40,663,206 | 60,701,696 | 488 / 3 | 0 | 0.3 | yes |  |
| Lingala_BibleTTS | 4,260,690,800 | 6,375,368,704 | 11,358 / 3 | 0 | 26.1 | yes |  |
| MELD | 6,053,474,834 | 7,759,617,024 | 13,849 / 4 | 143 | 43.5 | yes |  |
| MESD | 85,847,991 | 92,175,360 | 1,725 / 1 | 863 | 0.5 | yes |  |
| MLEnd_Intonation | 434,159,616 | - | 11,670 / 1 | 0 | 3.9 | yes | ReadError: unexpected end of data |
| MusicCapsCommonSense | 217,832,894 | 248,245,248 | 150 / 1 | 1 | 1.3 | yes |  |
| nigerian_english | 1,214,471,685 | 1,996,611,072 | 3,359 / 1 | 0 | 8.9 | yes |  |
| OpenMic | 2,544,957,878 | 2,652,476,416 | 40,157 / 157 | 20,157 | 14.6 | yes |  |
| peruvian_spanish | 1,937,998,305 | 3,191,799,296 | 5,447 / 1 | 0 | 14.7 | yes |  |
| ravnursson_faroese | 6,161,003,328 | 6,205,158,912 | 71,949 / 4 | 0 | 20.3 | yes |  |
| slurp_real | 2,006,244,675 | 2,401,207,296 | 27,792 / 3 | 0 | 13.0 | yes |  |
| URDU | 75,609,756 | 88,400,896 | 400 / 5 | 0 | 0.5 | yes |  |
| VIVOS | 1,474,345,794 | 1,816,300,032 | 12,424 / 69 | 4 | 9.7 | yes |  |
| VocalSound | 1,781,569,563 | 2,868,897,792 | 42,049 / 1 | 21,072 | 18.9 | yes |  |
| Yoruba_BibleTTS | 2,053,086,519 | 2,900,769,792 | 7,594 / 3 | 0 | 14.2 | yes |  |

In total, the 23 archives are 50,759,371,703 bytes (47.3 GiB). The whole pass took 356 s in one process, about 136 MiB/s.

## Findings in detail

### 1. How the splits are laid out inside shared archives

Most splits are expensive because of the order of the folders inside the archive. (Positions are from the "Split spans" column. The data files are `<local>/data/<name>/<name>.tar.gz`, apart from AESDD's `aesdd.tar.gz`.)

- **Clotho:** `test/` (0–17.6%), then `dev/` (validation, 17.6–35.5%), then `train/` (35.5–100%).
- **MELD:** `MELD/test_wavs/` (0–20.6%), then `train_wavs/` (20.6–92.0%), then `validation_wavs/` (92.0–100%).
- **ravnursson_faroese:** `speech/test/` (0–4.0%), then `speech/train/` (4.0–95.9%), then `speech/dev/` (validation, 95.9–100%).
- **slurp_real:** `train/`, then `val/` (validation, from 45.2%), then `test/` (from 67.2%).
- **VIVOS:** `train/`, then `test/` (from 95.2%).
- **hi_kia:** `train/`, then `test/`, then `validation/`. Validation and test sit deep in percentage terms, but the whole archive is only 38.8 MiB.
- **Ewe_BibleTTS:** the `test/` folder and its 66 unconfigured test clips come first, then `train/`. So the 1st train clip is entry 69 (20 MiB in).
- **Lingala_BibleTTS and Yoruba_BibleTTS:** train comes first.
- **esc50, VocalSound and OpenMic:** each is one flat folder with the splits mixed together, so every split starts near the front. Their entry counts are high because of macOS `._*` companion files. For example, VocalSound validation needs 310 entries for 20 clips.

### 2. Archive order compared with metadata order

- **Same order for every matched clip (rho 1.0):** AESDD, AudioMNIST, AudioMNISTCommonSense, colombian_spanish, MESD, nigerian_english, peruvian_spanish, and all three ravnursson_faroese splits.
- **Same except for one record:** Clotho (each split) and MusicCapsCommonSense. Record #2 of each JSON is a duplicate of a later record, and the archive holds that clip once, further down. So the first 20 archive clips are metadata records 0, 1, 3, 4, …, 20.
- **URDU:** the archive holds the metadata's four 100-record class blocks, each in metadata order, but with the blocks in reverse order: Angry (records 300–399), then Neutral, Sad and Happy (records 0–99). So the first archive clip is metadata record 300 (rho −0.875).
- **Everything else:** the orders are unrelated, with |rho| < 0.3.
  - EMNS: the first archive clip is metadata record 1,010.
  - MLEnd_Intonation train: record 18,386.
  - Yoruba_BibleTTS: record 6,555.
  - esc50, OpenMic and VIVOS train: the metadata JSON is sorted by path, but the archive is not.
- **No archive is sorted by path.**

### 3. Metadata records with no exactly matching archive entry

I checked every unmatched path for these near-misses:
- a leading `./` or `/`;
- backslashes;
- surrounding whitespace;
- Unicode normalisation;
- letter case;
- a different extension;
- the same file name in a different folder.

**None of the first five kinds occurs anywhere.** The only near-miss found is the same file name in a different folder.

- **MLEnd_Intonation: truncated archive.** 8,922 of 20,592 train records, all 2,557 validation records and all 2,585 test records are missing, with no near-misses.
  - **The archive is cut short.** `<local>/data/MLEnd_Intonation/MLEnd_Intonation.tar.gz`:
    - is 434,159,616 bytes, an exact multiple of 4,096;
    - has a gzip stream with no end-of-stream marker;
    - decompresses to 827,722,537 bytes;
    - has a last tar header (`train/44461.wav`) that promises data up to byte 827,730,476.
  - **Only train clips.** It holds only `train/` files (11,670 of them).
  - **The Hub copy is the same file.** Its sha256 equals the Hub LFS etag at `09f8a0f`, so the Hub file is equally truncated, not a bad local download.
  - **Every full read fails.** `tarfile` raises `ReadError: unexpected end of data` after the last complete member.
- **slurp_real train: most of the audio is missing.** 22,776 of 35,199 unique paths are missing, with no near-misses. Examples: `train/audio-1501751894.wav`, `train/audio-1434529472-headset.wav`. The archive has only 12,423 `train/` files, so train loads 12,423 clips, not 35,199.
- **hi_kia test: the JSON lists every clip under `test/`.** 429 of 488 records name a file under `test/` that exists only under another folder:
  - 381 exist only as `train/<file>`, e.g. `test/M4_S03_6_a.wav` → `train/M4_S03_6_a.wav`;
  - 48 exist only as `validation/<file>`.

  `<local>/data/hi_kia/hi_kia_test.json` lists all 488 clips in the dataset under a `test/` prefix. Only the 59 real test clips load.
- **MELD: one record in each of two splits points at a test clip.** Both target files are listed by `MELD_test.json`.
  - Train: `MELD/train_wavs/dia125_utt3.wav` exists only as `MELD/test_wavs/dia125_utt3.wav`.
  - Validation: `MELD/validation_wavs/dia110_utt7.wav` exists only as `MELD/test_wavs/dia110_utt7.wav`.

### 4. Duplicate metadata records

- **OpenMic** has several records per clip:
  - train: 30,956 records for 14,915 clips, up to 8 per clip;
  - test: 10,578 records for 5,085 clips, up to 7 per clip.

  The loader keeps only the last record for each `audio_path` (`uad_data/loader.py:40`), so the other records for a clip are dropped silently.
- **One duplicated path each:**
  - Clotho train (`train/broad neighborhood background.wav`);
  - Clotho validation (`dev/Nighttime in rural Jenks, Oklahooma.wav`);
  - Clotho test (`test/md1trk22.wav`);
  - MusicCapsCommonSense test (`wav/[-mA_bqD1tgU]-[30-40].wav`);
  - slurp_real train (`train/audio-1502309360-headset.wav`).

### 5. Other surprises

- **`ravnursson_faroese_dev.json` is byte-for-byte identical to `ravnursson_faroese_validation.json`** (same folder). The loader never reads the `_dev` file, because it only opens `<name>_<split>.json` (`uad_data/internal_dataset.py:45-53`).
- **AudioMNIST and AudioMNISTCommonSense are two copies of the same set of clips.**
  - The two archives are each about 948 MiB. They have identical member name lists and identical uncompressed tar sizes (1,874,991,616 bytes), but differ in compressed bytes and sha256.
  - `AudioMNISTCommonSense_test.json` uses exactly the same 30,000 `audio_path`s as `AudioMNIST_train.json`. So the same clips are AudioMNIST *train* and AudioMNISTCommonSense *test*.
- **BibleTTS archives hold splits that no config can request.** The Ewe, Lingala and Yoruba archives contain validation and test clips (186/66, 202/63 and 63/40), and each has matching `_validation.json` and `_test.json` files. But the registry declares only `train` for them (`uad_data/internal_datasets.py:82`, `:123`, `:272`). No config can request the other splits, because the loader rejects unregistered splits (`uad_data/json_config_loader.py:84-87`).
- **Archive files that no JSON lists:**
  - **macOS junk:** `._*` / `.DS_Store` files in esc50 (2,001), MESD (863), OpenMic (20,157) and VocalSound (21,072), plus 2 in MELD.
  - **MELD:** 132 `MELD/test_wavs/final_videos_test*.wav`, 5 other test clips and 4 validation clips.
  - **Clotho:** 3 files: `dev/City Apartment .wav`, `test/Small Junk Dropped.wav` and `train/jetgrunge.wav`.
  - **VIVOS:** `genders.txt` and `prompts.txt` in both `train/` and `test/`.
  - **MusicCapsCommonSense:** 1 file, `wav/[-FlvaZQOr2I]-[90-100].wav`.

  The loader skips all of these, but they still cost reads and add to the entry counts.
- **Stray files in dataset folders:**
  - `<local>/data/MusicCapsCommonSense/wav/` holds 10 loose `.m4a` files. They came from the Hub (download records exist) and the loader doesn't use them.
  - `<local>/data/slurp_real/.gitattributes` is an empty file.
  - `<local>/data/LibriSpeech/` has metadata JSONs but no archive. It isn't registered, so it's out of scope.
- **Full runs read more than they need.** The loader keeps reading an archive after the split's last clip, because nothing stops the loop except the end of the archive or `max_samples` (`uad_data/loader.py:111-139`). For example:
  - ravnursson_faroese test ends at 4.0% of a 5.9 GiB archive;
  - Clotho test ends at 17.6%;
  - MELD test ends at 20.6%;
  - slurp_real train ends at 45.1%.

  A full run of any of these still reads the whole archive.

## Loader behaviour this relies on

- **One split per call.** `iter_samples` handles one split. For each internal dataset that has that split, it downloads `<name>_<split>.json` (`uad_data/loader.py:99-103`, path built in `uad_data/internal_dataset.py:45-53`). It then builds a map from `audio_path` to record, where a later duplicate overwrites an earlier one (`uad_data/loader.py:35-41`).
- **Sequential read of the whole archive.** The archive is opened as a sequential `tarfile` stream in mode `r|gz`, whether it is streamed or downloaded (`uad_data/loader.py:63-83`, lines 75 and 82). Every member is visited in order.
- **Exact-name matching.** Non-regular files are skipped (`:112-113`). So is any member whose `member.name` isn't exactly a metadata key (`:114-116`). A matched clip is read in full (`:117-120`).
- **Where the archive path comes from.** The archive path is the registry's `data_url`, e.g. AESDD's is `data/AESDD/aesdd.tar.gz` (`uad_data/internal_datasets.py:27`).
- **All splits share one archive.** Every split of a dataset reads the same archive from its first byte.
- **The only early stop is `max_samples`.** It counts rows across the whole run (`uad_data/loader.py:97`, `:137-139`). It defaults `stream=True` when it is set (`:171-172`).
- **Streaming.** `stream=True` uses `HfFileSystem(...).open(path, "rb")` (`uad_data/hub.py:52-75`). This transfers only the ranges that are read, plus whatever read-ahead `HfFileSystem` does. I did not check its block size, because `huggingface_hub` is not installed here. So the bytes transferred over HTTP are at least the byte counts above, and may be rounded up to that block size.

## Method

Everything was produced by [`archive-split-depth.py`](archive-split-depth.py), using only the Python 3.10.11 standard library. It made no network access and used no HF token.

1. **Configs and registry.** It reads the configured splits from `<local>/universal_audio_dataset_configs/complete-{1..5}.json`. It reads each dataset's `data_url` from `uad_data/internal_datasets.py` by parsing the file with `ast`, without importing it.
2. **One full pass per archive.** For each dataset it loads every `*.json` in `<local>/data/<name>/`, then makes one full pass over the archive with `tarfile.open(fileobj=CountingReader(file), mode="r|gz")`.
   - **Counting bytes.** `CountingReader` counts and hashes every compressed byte that `tarfile` requests.
   - **Matching.** Matching mirrors the loader: regular files only, and exact `member.name` equality.
   - **Measuring a clip.** For each split's 1st, 5th and 20th unique matched clip, the script reads the clip's data (as the loader does) and then records the member count and the compressed byte count.
   - **Split spans.** These are the compressed byte counts when the header of the split's first and last clip was read.
3. **Order.** The split's unique metadata paths, in first-occurrence order and restricted to matched paths, are compared with the archive order of matched clips. The comparison checks both the first 20 and the full list, and computes Spearman rho between metadata index and archive rank.
4. **Unmatched paths.** For every unmatched metadata path, the script tries the near-miss keys listed in finding 3. It also records which JSON, if any, lists the near-miss archive file.
5. **Checksums.** After `tarfile` stops, the rest of the file is read so the sha256 covers the whole archive. The sha256 is compared with `<local>/.cache/huggingface/download/<path>.metadata`, whose lines are commit, etag and timestamp.
6. **Anomaly checks.** A few one-off checks were made with small scratch scripts that aren't committed:
   - that the ravnursson `_dev` and `_validation` files are byte-identical;
   - the AudioMNIST name and path comparison;
   - the MLEnd gzip end-of-stream check;
   - the MELD breakdown of files no JSON lists;
   - the git-blob etag check of the 57 JSONs and the configs.

**The byte counts are approximate.** `tarfile` reads its file object in `bufsize` chunks, and the default is `RECORDSIZE`, 10,240 bytes. See CPython 3.10's `Lib/tarfile.py`: line 79 defines `RECORDSIZE`, line 1583 is `open(..., bufsize=RECORDSIZE)`, and lines 540 and 560 are `_Stream` reads of `self.bufsize`. Each count can therefore overshoot the true compressed end of a clip by up to about 10 KiB, which matters only for the smallest numbers in the table.

**The "~330 MiB smoke archive" figure is rough.** It is the sum over splits of (compressed position of the 20th clip − that of the 1st) × 20/19. It overstates splits whose clips are mixed with other splits' clips (esc50, OpenMic, VocalSound), and it ignores MLEnd validation and test.

**To reproduce:**

```
python docs/research/archive-split-depth.py --data-root <local copy of the Hub repo> --out results.json
python docs/research/archive-split-depth.py --table results.json   # re-print both tables
```

`--only NAME ...` restricts the run to some datasets. `--workers N` reads archives in parallel, but that distorts the per-archive timings.
