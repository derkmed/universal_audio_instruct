"""CLI entry point for finetuning on the Universal Audio Understanding dataset.

Run from the repo root as a module:

    python -m train.main --model GEMMA-4 --json-config configs/clotho_config.json

Flow: parse flags into a `TrainConfig`, load the requested dataset slice via
`uad_data.load_uad_dataset` (the exact rows eval sees), build the chosen train
backend (QLoRA, LoRA or full finetune, per the flags), then run the HuggingFace
`Trainer` with the backend's `collate` as the data collator. Saves the LoRA
adapter (or the full weights, for a full finetune) + processor to
`--output-dir`.

QLoRA recipe: https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
"""
import argparse
import os

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from uad_data import load_uad_dataset
from .backends import GemmaTrainBackend, QwenTrainBackend
from .config import DEFAULT_MODEL_PATHS, DEFAULT_SEED, TrainConfig


class RowDataset(Dataset):
    """Wraps raw uad_data row dicts; all processing happens in backend.collate."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audio Instruct Finetuning (QLoRA + HF Trainer)")

    p.add_argument("--model", required=True, choices=list(DEFAULT_MODEL_PATHS),
                   dest="model_choice", help="Which model backend to finetune")
    p.add_argument("--model-path", default=None,
                   help="Override the default HuggingFace model path/id")

    p.add_argument("--dataset", default="AudioInstruct/Universal-Audio-Understanding")
    p.add_argument("--split", default=None, dest="dataset_split",
                   help="Splits to load: one name, several joined with '+', or 'all' "
                        "(default: all for a smoke run, train otherwise)")
    p.add_argument("--json-config", default="configs/clotho_config.json",
                   dest="json_config_path",
                   help="UAD dataset JSON config (default: configs/clotho_config.json)")
    p.add_argument("--clips-per-split", type=int, default=None, dest="clips_per_split",
                   help="Train on only the first N clips of each selected split (for smoke runs)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, dest="seed",
                   help="Seeds the Trainer and the prompt-template picks")

    p.add_argument("--output-dir", default="outputs/finetune", dest="output_dir")
    p.add_argument("--epochs", type=float, default=1.0, dest="num_train_epochs")
    p.add_argument("--batch-size", type=int, default=2, dest="per_device_train_batch_size")
    p.add_argument("--grad-accum", type=int, default=8, dest="gradient_accumulation_steps")
    p.add_argument("--lr", type=float, default=2e-4, dest="learning_rate")
    p.add_argument("--lora-r", type=int, default=16, dest="lora_r")
    p.add_argument("--lora-alpha", type=int, default=32, dest="lora_alpha")
    p.add_argument("--no-4bit", action="store_false", dest="load_in_4bit",
                   help="Keep the base model in bf16 instead of 4-bit NF4 "
                        "(LoRA without quant noise; needs more VRAM)")
    p.add_argument("--no-lora", action="store_false", dest="use_lora",
                   help="Train all weights instead of LoRA adapters (full finetune; "
                        "requires --no-4bit and much more VRAM)")

    p.add_argument("--hf-token", default=None, dest="hf_token",
                   help="HuggingFace token (falls back to HF_TOKEN env var)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    config = TrainConfig(
        model_choice=args.model_choice,
        model_path=args.model_path,
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        json_config_path=args.json_config_path,
        clips_per_split=args.clips_per_split,
        seed=args.seed,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.load_in_4bit,
        use_lora=args.use_lora,
        hf_token=hf_token,
    )

    print(f"Loading dataset: {config.dataset_name} (split={config.dataset_split})")
    rows = load_uad_dataset(
        json_config_path=config.json_config_path,
        split=config.dataset_split,
        repo_id=config.dataset_name,
        token=hf_token,
        clips_per_split=config.clips_per_split,
        seed=config.seed,
    )
    print(f"Dataset loaded: {len(rows)} rows")
    report = rows.report
    print(report.describe())
    if not rows:
        # With no rows there is nothing to save, so this stands in for the
        # end-of-run exit below. The report says which of the two it is.
        raise SystemExit(
            "No rows produced. See the load report above."
            if report.has_problems
            else "No rows produced — check the config/split.")

    if not config.is_smoke_run and any(split.split == "test" for split in report.splits):
        print("\n*** WARNING: the selected splits include test. Training on the "
              "test split invalidates every evaluation on it. ***\n")

    backend_cls = {"GEMMA-4": GemmaTrainBackend, "QWEN3-Omni": QwenTrainBackend}[
        config.model_choice]
    backend = backend_cls(config)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        # transformers 5.15 removed `warmup_ratio`; since 5.5, a float in
        # [0, 1) passed as `warmup_steps` is read as a ratio of total steps.
        warmup_steps=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        seed=config.seed,
        bf16=True,
        gradient_checkpointing=config.gradient_checkpointing,
        optim="paged_adamw_8bit" if config.load_in_4bit else "adamw_torch",
        report_to="none",
        # Our dataset yields raw dicts consumed by backend.collate; without this
        # the Trainer would strip every column it doesn't recognize.
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=backend.model,
        args=training_args,
        train_dataset=RowDataset(rows),
        data_collator=backend.collate,
    )
    trainer.train()

    trainer.save_model(config.output_dir)          # LoRA adapter (or full weights)
    backend.processor.save_pretrained(config.output_dir)
    print(f"Saved model + processor to {config.output_dir}/")

    # A smoke run is a check, so its exit status reports what the load found:
    # a failed internal dataset, a row that wouldn't render, or an empty split.
    if config.is_smoke_run and report.has_problems:
        raise SystemExit(
            "Smoke run finished with problems in the load report:\n"
            + report.describe())


if __name__ == "__main__":
    main()
