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
_Avoid_: example, sample

## Runs

**Run config**:
A named selection of internal datasets, and for each the splits and tasks to include, that a training or evaluation run draws its rows from.
_Avoid_: UAD config, collection

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

**Load report**:
The record that comes back with a run's rows: clips found for each internal dataset and selected split, internal datasets that failed to load, and rows that failed to render.

## Results

**Row status**:
How one row of an evaluation run ended: `ok`, `empty_output`, `render_error`, `audio_error` or `model_error`. Only `ok` means the model returned non-empty text.

**Group**:
The rows of one evaluation run that share an internal dataset, split and task. A group passes when it has at least one row and every row's status is `ok`.
_Avoid_: bucket, slice

**Preliminary metric**:
The one rough metric an evaluation run reports for each task: WER for asr, english_translation and caption, and a hit rate for the rest. It is for information only and never decides whether a group passes.
_Avoid_: score, accuracy
