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
_Avoid_: example, element, sample

**Archive order**:
The order in which an internal dataset's clips are stored in its audio archive. It usually differs from the order of records in a split's metadata file.

**Row**:
One clip rendered for one task with one prompt template; the unit a model sees.
_Avoid_: example, sample

## Runs

**Run config**:
A named selection of internal datasets, and for each the splits and tasks to include, that a training or evaluation run draws its rows from.
_Avoid_: UAD config, collection

**Selected split**:
A split that a run actually loads for an internal dataset: one that the run config lists for that dataset and that the run also asked for.
_Avoid_: configured split, active split

**Smoke run**:
A training or evaluation run that loads only the first few clips, in archive order, of each selected split of each internal dataset, to check that each one works end to end.
_Avoid_: debug run, sampled run
