# eval — evaluation harness

Batched evaluation of audio-instruction models (Gemma, Qwen3-Omni) on the
Universal Audio Understanding dataset. Rows come from the shared
[`uad_data`](../uad_data) loader — the same rows [`train/`](../train) finetunes on.

```bash
pip install -r requirements.txt
export HF_TOKEN=...   # private dataset; some models are gated
python -m eval.main --model GEMMA-4 --json-config configs/clotho_config.json --split test
```

Or run on an A100 in Colab: [`colab_eval.ipynb`](./colab_eval.ipynb).

## Flow

```mermaid
flowchart TD
    CLI["eval.main<br/>(CLI flags → EvalConfig)"] --> LOADER["uad_data.load_uad_dataset<br/>(audio + metadata + prompts from HF Hub)"]
    CLI --> BE["ModelBackend<br/>GemmaBackend / QwenBackend"]
    LOADER -- "row dicts" --> EV["Evaluator.evaluate"]

    subgraph BATCH["per batch (batches run one after another)"]
        PRE["preprocess_audio (uad_data.audio_utils)<br/>decode → 16 kHz mono float32, thread pool<br/>(Gemma uses it; Qwen reads the raw bytes)"]
        REQ["InferenceRequest batch<br/>(audio + sys_inst + prompt + ground truth)"]
        GEN["backend.generate_batch<br/>(one batched generate call; falls back to sequential)"]
        PRE --> REQ --> GEN
    end

    EV --> BATCH
    GEN -- "predictions" --> JSONL["results.jsonl<br/>(flushed per row — crash-safe)"]
    GEN -- "predictions" --> SCORE["eval/metrics.py<br/>one preliminary metric per task"]
    SCORE --> SUM["groups (dataset, split, task):<br/>statuses, pass/fail, metric<br/>summary.json + console table"]
```

## Pieces

| file | role |
| --- | --- |
| `main.py` | CLI entry point; wires config → loader → backend → evaluator |
| `config.py` | `EvalConfig` + `DEFAULT_MODEL_PATHS` (registry shared with `train/`) |
| `evaluator.py` | batch loop: threaded audio decoding, then one batched generate call per batch; incremental `results.jsonl`; a status for every row, then per-group pass/fail and metrics |
| `metrics.py` | the one preliminary metric each task reports: WER for `asr`, `english_translation` and `caption`, a hit rate for `classification`, `commonsense` and `qa` |
| `backends/base.py` | `ModelBackend` ABC + `InferenceRequest` |
| `backends/gemma.py` | Gemma: audio arrays in chat messages, batched `processor(text, audio)` |
| `backends/qwen.py` | Qwen3-Omni: raw audio bytes in temp files + `process_mm_info`, batched processing |

To evaluate a finetuned checkpoint, merge the LoRA adapter and pass it via
`--model-path` — see [`FINETUNING.md`](../FINETUNING.md#evaluating-a-finetuned-model).
