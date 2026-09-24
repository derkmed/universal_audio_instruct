# train — finetuning

Finetuning of audio-instruction models (Gemma, Qwen3-Omni) on the Universal
Audio Understanding dataset via the HuggingFace `Trainer` API. Sibling to
[`eval/`](../eval); both reuse the shared [`uad_data`](../uad_data) loader, so
training and evaluation see identical rows.

Three modes — **QLoRA** (default), **LoRA** on a bf16 base (`--no-4bit`), and
**full finetune** (`--no-4bit --no-lora`).

```bash
pip install -e . -r requirements.txt -r train/requirements.txt
export HF_TOKEN=...

# smoke test
python -m train.main --model GEMMA-4 --clips-per-split 5 --epochs 1

# real run (Gemma: LoRA on bf16 is the recommended default)
python -m train.main --model GEMMA-4 --no-4bit \
    --json-config configs/clotho_config.json --split train \
    --output-dir outputs/gemma_clotho_lora
```

## Flow

```mermaid
flowchart TD
    CLI["train.main<br/>(CLI flags → TrainConfig)"] --> LOADER["uad_data.load_uad_dataset<br/>(same rows eval sees)"]
    LOADER -- "raw row dicts" --> DS["RowDataset"]
    CLI --> BE["TrainBackend<br/>GemmaTrainBackend / QwenTrainBackend"]
    BE -- "load_in_4bit? → NF4 quant<br/>use_lora? → LoRA adapters" --> MODEL["model + processor"]

    subgraph COLLATE["backend.collate (data_collator)"]
        C1["render eval-identical chat:<br/>audio part + system/prompt"] --> C2["append assistant turn<br/>(ground-truth output)"]
        C2 --> C3["processor twice:<br/>full conv + prompt-only (same audio)"]
        C3 --> C4["labels: prompt prefix<br/>and padding → -100"]
    end

    DS --> TR["transformers.Trainer<br/>(remove_unused_columns=False)"]
    MODEL --> TR
    TR -- "each batch" --> COLLATE
    COLLATE -- "batch + labels" --> TR
    TR --> SAVE["adapter/weights + processor<br/>→ output_dir"]
    SAVE -. "merge_and_unload →<br/>eval.main --model-path" .-> EVAL["eval/"]
```

**Full guide — modes, CLI reference, label masking, evaluating a finetuned
model, troubleshooting: [`FINETUNING.md`](../FINETUNING.md).**
