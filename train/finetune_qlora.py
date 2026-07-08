"""QLoRA finetuning scaffold for audio-instruction models.

Reference: https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora

STATUS: scaffold. The data path is wired up and correct — rows come from the shared
`uad_data.load_uad_dataset` and are mapped to chat-format SFT examples below. The
model-loading, QLoRA config, and trainer wiring are marked `TODO`: they depend on the
target model (and, for audio-capable models, on processing the `audio` bytes the way
`eval/backends` does). Heavy deps (`transformers`, `peft`, `trl`) are imported inside
`main()` so this module can be imported without them installed.

Planned usage:
    python -m train.finetune_qlora --json-config configs/clotho_config.json --split train
"""
import argparse
import os
from typing import Any

from uad_data import load_uad_dataset


def to_chat_example(row: dict[str, Any]) -> dict[str, Any]:
    """Map a uad_data row to a chat-format SFT example.

    Uses the same instruction triple the evaluator consumes, so training and eval
    stay consistent. For text-only finetuning this is sufficient; for audio-capable
    models the audio bytes (`row["audio"]["bytes"]`) must additionally be fed through
    the model's processor (TODO — mirror the handling in eval/backends).
    """
    messages = []
    if row.get("system_instruction"):
        messages.append({"role": "system", "content": row["system_instruction"]})
    messages.append({"role": "user", "content": row.get("prompt", "")})
    messages.append({"role": "assistant", "content": row.get("output", "")})
    return {"messages": messages}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QLoRA finetuning (scaffold)")
    p.add_argument("--model", default="google/gemma-4-e2b-it", help="Base model id to finetune")
    p.add_argument("--json-config", default="configs/clotho_config.json", dest="json_config_path")
    p.add_argument("--split", default="train", dest="split")
    p.add_argument("--dataset", default="AudioInstruct/Universal-Audio-Understanding", dest="repo_id")
    p.add_argument("--output-dir", default="outputs/qlora", dest="output_dir")
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    p.add_argument("--hf-token", default=None, dest="hf_token")
    return p


def main() -> None:
    args = build_parser().parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # --- Data (implemented, shared with eval) --------------------------------
    rows = load_uad_dataset(
        json_config_path=args.json_config_path,
        split=args.split,
        repo_id=args.repo_id,
        token=hf_token,
        max_samples=args.max_samples,
    )
    examples = [to_chat_example(r) for r in rows]
    print(f"Prepared {len(examples)} SFT examples from {args.repo_id} (split={args.split}).")

    # --- Model + QLoRA + trainer (TODO) --------------------------------------
    # from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    # from peft import LoraConfig
    # from trl import SFTTrainer, SFTConfig
    #
    # TODO: load `args.model` with a 4-bit BitsAndBytesConfig (nf4, bf16 compute).
    # TODO: LoraConfig(target_modules=..., r=..., lora_alpha=..., task_type="CAUSAL_LM").
    # TODO: build an HF Dataset from `examples`, apply the tokenizer chat template.
    # TODO: for audio models, process `row["audio"]["bytes"]` via the processor
    #       (see eval/backends/gemma.py / qwen.py) instead of text-only SFT.
    # TODO: SFTTrainer(model, args=SFTConfig(output_dir=args.output_dir, ...)).train().
    raise NotImplementedError(
        "Training loop not implemented yet — this is a scaffold. Data loading works; "
        "wire up model/QLoRA/trainer per the reference guide."
    )


if __name__ == "__main__":
    main()
