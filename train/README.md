# train — finetuning

QLoRA finetuning of audio-instruction models on the Universal Audio Understanding
dataset, using the HuggingFace `Trainer` API. Sibling to [`eval/`](../eval); both
reuse the shared [`uad_data`](../uad_data) loader, so training and evaluation see
identical rows, and each supported model has one backend in each harness:

| model | eval backend | train backend |
| --- | --- | --- |
| Gemma (`GEMMA-4`) | `eval.backends.GemmaBackend` | `train.backends.GemmaTrainBackend` |
| Qwen3-Omni (`QWEN3-Omni`) | `eval.backends.QwenBackend` | `train.backends.QwenTrainBackend` |

Three finetuning modes via two independent flags (`load_in_4bit`, `use_lora`):

| mode | flags | when |
| --- | --- | --- |
| **QLoRA** (default) | — | Qwen3-Omni-30B, or any GPU where bf16 weights don't fit |
| **LoRA** (bf16 base) | `--no-4bit` | Gemma on an A100 — faster steps, no quantization noise |
| **Full finetune** | `--no-4bit --no-lora` | small models, max fidelity, most VRAM |

(4-bit without LoRA is rejected — a quantized base can't be trained directly.)
The QLoRA recipe follows the Gemma guide:
https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
(4-bit NF4 base via `bitsandbytes`, LoRA adapters via `peft`, paged 8-bit AdamW).

## How it works

- `train.main` loads rows with `uad_data.load_uad_dataset` and hands **raw row
  dicts** to the HF `Trainer` (`remove_unused_columns=False`).
- Each backend's `collate(rows)` is the `data_collator`: it renders the same chat
  conversation the eval backend uses (audio part + system/prompt text), appends the
  ground-truth `output` as the assistant turn, and processes the whole batch through
  the model's processor (audio features + padding included).
- **Labels** use the prompt/full two-pass recipe: the batch is processed once with
  just the prompt (incl. generation header) and once with the full conversation;
  each sample's prompt-token count is masked to -100 so only answer tokens carry
  loss. The prompt pass uses the *same audio*, since processors expand the audio
  placeholder into a variable number of tokens.

## Usage

```bash
pip install -r requirements.txt -r train/requirements.txt
export HF_TOKEN=...   # private dataset (+ gated models)

# smoke test: tiny slice, streams only an archive prefix
python -m train.main --model GEMMA-4 --max-samples 32 --epochs 1

# real run
python -m train.main --model GEMMA-4 \
    --json-config configs/clotho_config.json --split train \
    --batch-size 2 --grad-accum 8 --lr 2e-4 --output-dir outputs/gemma_clotho
```

Key flags: `--model {GEMMA-4,QWEN3-Omni}`, `--lora-r/--lora-alpha`, `--no-4bit`
(bf16 LoRA), `--no-4bit --no-lora` (full finetune), `--max-samples` (smoke tests). The LoRA adapter + processor are
saved to `--output-dir`; evaluate the result by passing that directory to
`python -m eval.main --model ... --model-path <output-dir>` (after merging the
adapter, or loading it with `peft`).

## Caveats

- Batched audio training is memory-hungry; start with `--batch-size 1-2` and lean
  on `--grad-accum`.
- Label masking assumes right padding and that the rendered full text extends the
  rendered prompt text; both hold for the current Gemma/Qwen chat templates. If a
  template changes, re-verify with a few decoded examples.
- Default LoRA targets are the attention projections (`q/k/v/o_proj`); pass
  different `lora_target_modules` in `TrainConfig` to also adapt MLPs.
