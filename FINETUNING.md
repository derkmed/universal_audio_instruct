# Finetuning guide

How to finetune audio-instruction models (Gemma, Qwen3-Omni) on the Universal
Audio Understanding dataset with this repo's `train/` harness, built on the
HuggingFace `Trainer` API.

- [Architecture](#architecture)
- [Supported finetuning modes](#supported-finetuning-modes)
- [Usage](#usage)
- [CLI reference](#cli-reference)
- [How batches are built (label masking)](#how-batches-are-built-label-masking)
- [Evaluating a finetuned model](#evaluating-a-finetuned-model)
- [Adding a new model](#adding-a-new-model)
- [Caveats & troubleshooting](#caveats--troubleshooting)

## Architecture

`train/` is the training-side twin of `eval/`. Both consume the shared
[`uad_data`](./uad_data) loader, so training and evaluation see **identical
rows** (same prompt templates; for Gemma, also the same audio preprocessing via
`uad_data.audio_utils`, while the Qwen backends pass the raw audio bytes to
`process_mm_info`), and each supported model family has one backend in each
harness:

| model (`--model`) | HF id (default) | eval backend | train backend |
| --- | --- | --- | --- |
| `GEMMA-4` | `google/gemma-4-e2b-it` | `eval.backends.GemmaBackend` | `train.backends.GemmaTrainBackend` |
| `QWEN3-Omni` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `eval.backends.QwenBackend` | `train.backends.QwenTrainBackend` |

Flow: `train.main` parses flags into a `TrainConfig` → loads rows with
`uad_data.load_uad_dataset` → builds the chosen backend (model + processor,
quantized/LoRA-wrapped per the mode) → runs `transformers.Trainer` with the
backend's `collate` as `data_collator` → saves the adapter (or full weights)
plus the processor to `--output-dir`.

## Supported finetuning modes

Two independent knobs — `load_in_4bit` (quantize the frozen base to NF4) and
`use_lora` (train low-rank adapters instead of all weights) — give three modes:

| mode | flags | trainable params | base weights | optimizer |
| --- | --- | --- | --- | --- |
| **QLoRA** (default) | *(none)* | LoRA adapters (~0.1–1%) | 4-bit NF4 | `paged_adamw_8bit` |
| **LoRA** | `--no-4bit` | LoRA adapters | bf16 | `adamw_torch` |
| **Full finetune** | `--no-4bit --no-lora` | all | bf16 | `adamw_torch` |

`--no-lora` *without* `--no-4bit` is rejected: a 4-bit quantized base cannot be
trained directly.

### Which mode for which model?

- **Qwen3-Omni-30B → QLoRA.** ~60 GB of bf16 weights alone; 4-bit brings the
  base to ~15–18 GB, which is what makes it fit a single 80 GB A100 alongside
  activations. There is no realistic single-GPU alternative.
- **Gemma (~2B) → LoRA (`--no-4bit`) as the default.** The bf16 base fits
  easily; skipping quantization avoids NF4 noise in the frozen weights and
  makes steps faster (no dequantization overhead). Use the full finetune if you
  want the highest quality ceiling and have the VRAM (optimizer states +
  gradients cost roughly 4x parameter memory beyond the weights).
- **QLoRA still helps Gemma** on small GPUs (e.g. 16 GB cards).

QLoRA recipe reference:
https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora

## Usage

```bash
pip install -r requirements.txt -r train/requirements.txt
export HF_TOKEN=...   # the dataset is private; some models are gated
```

Smoke test (tiny slice; `max_samples` streams the archive and stops early
instead of downloading the whole tar. Clotho's archive stores its test clips
first, so reaching the first train clips still reads about 3 GiB):

```bash
python -m train.main --model GEMMA-4 --max-samples 32 --epochs 1 --output-dir outputs/smoke
```

Gemma, LoRA on a bf16 base (recommended Gemma default):

```bash
python -m train.main --model GEMMA-4 --no-4bit \
    --json-config configs/clotho_config.json --split train \
    --batch-size 2 --grad-accum 8 --lr 2e-4 \
    --output-dir outputs/gemma_clotho_lora
```

Qwen3-Omni, QLoRA (default mode):

```bash
python -m train.main --model QWEN3-Omni \
    --json-config configs/clotho_config.json --split train \
    --batch-size 1 --grad-accum 16 \
    --output-dir outputs/qwen_clotho_qlora
```

Full finetune (small models only):

```bash
python -m train.main --model GEMMA-4 --no-4bit --no-lora --lr 1e-5 \
    --output-dir outputs/gemma_clotho_full
```

Note the lower learning rate: ~2e-4 is a LoRA-adapter rate; full finetunes
typically want 1e-5..5e-5.

## CLI reference

| flag | default | meaning |
| --- | --- | --- |
| `--model` | *(required)* | `GEMMA-4` or `QWEN3-Omni` |
| `--model-path` | registry default | override the HF model id/path |
| `--dataset` | `AudioInstruct/Universal-Audio-Understanding` | HF Hub dataset repo_id |
| `--split` | `train` | dataset split |
| `--json-config` | `configs/clotho_config.json` | UAD config (local path, or name under the dataset repo's `universal_audio_dataset_configs/`) |
| `--max-samples` | full split | cap on training rows; also enables archive streaming |
| `--output-dir` | `outputs/finetune` | where adapter/weights + processor are saved |
| `--epochs` | `1.0` | training epochs |
| `--batch-size` | `2` | per-device batch size |
| `--grad-accum` | `8` | gradient accumulation steps (effective batch = batch-size x grad-accum) |
| `--lr` | `2e-4` | learning rate (LoRA-scale; lower it for full finetunes) |
| `--lora-r` / `--lora-alpha` | `16` / `32` | LoRA rank / scaling |
| `--no-4bit` | 4-bit on | keep the base model in bf16 |
| `--no-lora` | LoRA on | train all weights (requires `--no-4bit`) |
| `--hf-token` | `HF_TOKEN` env | HuggingFace token |

Not exposed on the CLI (set on `TrainConfig` directly): `lora_dropout` (0.05),
`lora_target_modules` (attention projections `q/k/v/o_proj`; add
`gate_proj`/`up_proj`/`down_proj` to also adapt MLPs), `warmup_ratio`,
`logging_steps`, `save_steps`, `gradient_checkpointing` (on),
`target_sr`/`max_audio_seconds` (16 kHz / 30 s; Gemma only, since the Qwen
backend reads raw audio bytes — keep matched with eval).

## How batches are built (label masking)

The `Trainer` receives **raw uad_data row dicts** (`remove_unused_columns=False`);
all processing happens in the backend's `collate`:

1. Render the same chat conversation the eval backend uses — audio content part
   plus system-instruction/prompt text — and append the row's ground-truth
   `output` as the assistant turn.
2. Process the whole batch through the model's processor
   (`processor(text=[...], audio=[...], padding=True)`), which handles audio
   feature extraction and padding. It's the same call shape as eval's
   batched-inference path, but training pads on the right, while eval pads on
   the left for generation.
3. Build labels with the **prompt/full two-pass recipe**: the batch is processed
   a second time with only the prompt (including the generation header); each
   row's prompt token count (right padding ⇒ `attention_mask.sum()`) is
   masked to `-100` in the labels, as is padding. Only answer tokens contribute
   to the loss. The prompt pass reuses the *same audio* because processors
   expand the audio placeholder into a variable number of tokens based on the
   audio features — a text-only tokenize would undercount the prefix.

## Evaluating a finetuned model

LoRA/QLoRA runs save an **adapter**, not full weights. To evaluate with
`eval/`, merge the adapter into the base model first:

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-e2b-it", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "outputs/gemma_clotho_lora")
model.merge_and_unload().save_pretrained("outputs/gemma_clotho_merged")
```

then point the eval harness at the merged directory:

```bash
python -m eval.main --model GEMMA-4 --model-path outputs/gemma_clotho_merged \
    --json-config configs/clotho_config.json --split test
```

Full finetunes save complete weights, so their `--output-dir` can be passed to
`--model-path` directly.

## Adding a new model

Mirror the existing pairs: implement the eval backend first
(`eval/backends/<model>.py`), then subclass `train.backends.TrainBackend` with
`_load_processor`, `_load_model` (respecting `config.load_in_4bit`), and
`collate` reusing the eval backend's conversation shape plus the assistant
turn. Export the classes from `eval/backends/__init__.py` and
`train/backends/__init__.py`, register them in the dispatch dicts in both
`main.py` files and in the Colab notebook's backend cell, and add the model to
`eval.config.DEFAULT_MODEL_PATHS`.

## Caveats & troubleshooting

- **Verified by construction, not by GPU run.** The harness compiles, but
  `train/` has no tests, and no end-to-end training step has been run in this
  environment. Do a `--max-samples 32` smoke run first.
- **Label-mask boundary.** The two-pass recipe assumes the rendered full text
  extends the rendered prompt text and right padding. Both hold for the current
  Gemma/Qwen chat templates; if a template changes, decode a few `labels` rows
  (ignoring `-100`) and confirm they contain only the answers.
- **OOM:** lower `--batch-size` (audio features are large), raise
  `--grad-accum` to keep the effective batch, and keep gradient checkpointing
  on. For Qwen, `--batch-size 1` is a reasonable floor.
- **Loss not falling:** with LoRA, check `print_trainable_parameters()` output
  is non-trivial; consider adding MLP projections to `lora_target_modules` or
  raising `--lora-r`.
- **`use_cache` warnings** during training are expected (disabled for gradient
  checkpointing).
