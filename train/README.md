# train — finetuning / training

Scripts for finetuning audio-instruction models on the Universal Audio Understanding
dataset. Sibling to [`eval/`](../eval); both reuse the shared
[`uad_data`](../uad_data) loader so training and evaluation see identical rows.

> **Status: scaffold.** [`finetune_qlora.py`](./finetune_qlora.py) sketches the
> intended structure (data loading wired up; model/QLoRA specifics are `TODO`).

## Approach

QLoRA finetuning, following the Gemma reference:
https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora

The data side is already solved: `uad_data.load_uad_dataset(...)` yields rows with
`system_instruction` / `prompt` / `output` (+ `audio` bytes), which map directly to
chat-format SFT examples. The remaining work is model-specific (chat templating,
audio feature handling for audio-capable models, LoRA target modules, trainer args).

## Extra dependencies

Beyond the repo's `requirements.txt`, training needs (see `train/requirements.txt`):

```
peft
trl
```

`transformers`, `datasets`, `accelerate`, and `bitsandbytes` are already pinned in the
top-level `requirements.txt`.

## Usage (planned)

```bash
pip install -r requirements.txt -r train/requirements.txt
python -m train.finetune_qlora --json-config configs/clotho_config.json --split train
```
