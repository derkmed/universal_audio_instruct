# Universal Audio Instruct

Training and evaluation of audio-instruction models on the Universal Audio Understanding (UAD) dataset, a private collection of audio corpora recast as instruction-following tasks.

## Dataset

**UAD**:
The Universal Audio Understanding dataset: every internal dataset, its audio, and the prompt templates that turn it into instructions.
_Avoid_: the HF repo, the loading script

**Internal dataset**:
One source corpus inside UAD (e.g. Clotho, MELD), with its own registered tasks and splits.
_Avoid_: subset, source dataset

**Split**:
A named partition of an internal dataset: train, validation, or test.

**Task**:
A kind of question UAD asks of a clip (e.g. asr, caption, classification).

**Prompt template**:
A system-instruction, prompt, and expected-output pattern that renders a clip's metadata into text for one task.

**Clip**:
One audio file in one split of one internal dataset.
_Avoid_: example, element, sample, utterance (an Utterance is part of a clip; MELD's `Utterance` field holds a clip's text, and its `Utterance_ID` field and `dia<N>_utt<M>.wav` file names number clips within a dialogue)

**Utterance**:
One entry in a clip's `transcriptions` list: a start time, an end time and what is said between them, so always part of a clip and never a whole clip. Only asr_timestamp_search uses utterances.
_Avoid_: segment (LibriCSS's `segment` field numbers a libricss clip within its recording session)

**Archive order**:
The order in which an internal dataset's clips are stored in its audio archive. It usually differs from the order of records in a split's metadata file.

**Row**:
One clip rendered for one task with one prompt template; the unit a model sees. For asr_timestamp_search, each utterance of the clip is rendered as its own rows.
_Avoid_: example, sample. HF `datasets` and the Trainer call a row an "example", but here that word blurs clips and rows, and the smoke-run cap counts clips, not rows. Renaming everything to Example was considered and not taken (2026-09-17).
TODO(derkmed): revisit Row vs Example one day.

## Runs

**Run config**:
A named selection of internal datasets, and for each the splits and tasks to include, that a training or evaluation run draws its rows from.
_Avoid_: UAD config, collection

**Run options**:
The handful of answers, shared by both harnesses and the loader, to "how many clips per split?" and "which split, when none was asked for?". A run config says *what* rows exist; run options narrow *how many* of them one run builds.
_Avoid_: run settings, run parameters

**Selected split**:
A split that a run actually loads for an internal dataset: one that the run config lists for that dataset and that the run also asked for.
_Avoid_: configured split, active split

**Smoke archive**:
A small archive for one internal dataset that holds the first few clips of each of its splits, in archive order. Smoke archives are published on the Hub for smoke runs to use.
_Avoid_: cache, sample archive

**Smoke run**:
A training or evaluation run that loads only the first few clips, in archive order, of each selected split of each internal dataset, to check that each one works end to end.
_Avoid_: debug run, sampled run

**Regular run**:
A training or evaluation run with no cap on clips per split. It stops at the first error, where a smoke run records the error and carries on.
_Avoid_: full run (a regular run can still use a small run config)

**Valid internal dataset**:
An internal dataset whose train split loads and renders every row without error in a smoke run. Only valid internal datasets go into a finetune's run config.
_Avoid_: clean dataset, well-formed dataset

**Load report**:
The record that comes back with a run's rows: clips found for each internal dataset and selected split, internal datasets that failed to load, and rows that failed to render.

## Results

**Row status**:
How one row of an evaluation run ended: `ok`, `empty_output`, `render_error`, `audio_error` or `model_error`. Only `ok` means the model returned non-empty text.

**Group**:
The rows of one evaluation run that share an internal dataset, split and task. A group passes when it has at least one row and every row's status is `ok`.
_Avoid_: bucket, slice

**Preliminary metric**:
The one rough metric an evaluation run reports for each task: WER for asr, english_translation and caption, and a hit rate for classification, commonsense and qa. Those six are the tasks the complete-1..5 run configs use; any other task has no preliminary metric and reports nothing. It is for information only and never decides whether a group passes.
_Avoid_: score, accuracy

**Answer field**:
The plain metadata field a task's preliminary metric reads a row against — a caption's `caption`, a qa row's `answer` — as opposed to the rendered `output`, which wraps that answer in template prose. Which rows a group's metric covers is decided by row status alone: every row with a prediction. An answer field its rule reads as nothing is judged by that rule like any other, not excluded — a `commonsense_answer` with no choice letter has none to be started with, so the row misses.
_Avoid_: reference (a WER reference is one answer field, not all of them), label

## Finetune runs on GCP

**Run prefix**:
The single GCS location that holds everything for one finetune run: `gs://<bucket>/finetunes/<model>/<run-id>/`, containing `checkpoints/`, `final/`, `run_config.json` and `train_log.jsonl`.
_Avoid_: run directory, output dir (the output dir is the VM's local disk, not GCS)

**Run-id**:
The identifier for one finetune run: a UTC timestamp plus a short git SHA. Minted on the first launch when `--run-id` is omitted, and passed back with `--run-id` to resume the same run on a fresh VM. It is the operator's contract — a fresh VM resumes a run only when handed that run-id; there is no auto-discovery, and runs for the same model may be in flight concurrently.
_Avoid_: job id, session id

**Checkpoint marker**:
An empty `_COMPLETE` object written into a remote `checkpoint-<step>/` only after every file in that checkpoint has finished uploading. A remote checkpoint counts as resumable only if its marker is present; a partially-uploaded checkpoint has no marker and is ignored on resume. Remote checkpoints are pruned to the newest 2 marked ones, marker deleted first.
_Avoid_: done flag, sentinel

**Final artifacts**:
What a completed run leaves at `<run prefix>/final/` for `eval.main` to load: `final/adapter/` (always, the trained LoRA adapter) and `final/merged/` (only for Gemma, and only when `--merge` was passed). Their presence is not a resume signal — resume is driven by `--run-id` alone.
_Avoid_: output, results
