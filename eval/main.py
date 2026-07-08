"""CLI entry point for audio instruction evaluation.

Run from the repo root as a module:

    python -m eval.main --model GEMMA-4 --json-config clotho_config.json --split test

Flow: parse flags into an `EvalConfig`, build the chosen model backend, load the
requested dataset slice via `uad_data.load_uad_dataset` (which fetches audio +
metadata + prompts from the private HF Hub repo -- no loading script), then run
batched inference through the `Evaluator` and print/save metrics.
"""

import argparse
import os

from .backends import GemmaBackend, QwenBackend
from .config import DEFAULT_MODEL_PATHS, EvalConfig
from .evaluator import Evaluator
from uad_data import load_uad_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audio Instruct Evaluation")

    p.add_argument(
        "--model",
        required=True,
        choices=list(DEFAULT_MODEL_PATHS),
        dest="model_choice",
        help="Which model backend to use",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Override the default HuggingFace model path/id",
    )
    p.add_argument(
        "--json-config",
        default="configs/clotho_config.json",
        dest="json_config_path",
        help="UAD dataset JSON config (default: configs/clotho_config.json)",
    )
    p.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        dest="num_preprocessing_workers",
        help="Threads for parallel audio preprocessing",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        dest="max_samples",
        help="Evaluate only the first N samples (useful for debugging)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        dest="max_new_tokens",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Directory to write results.jsonl and summary.json (created if absent)",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        dest="hf_token",
        help="HuggingFace token (falls back to HF_TOKEN env var)",
    )
    p.add_argument("--dataset", default="AudioInstruct/Universal-Audio-Understanding")
    p.add_argument("--split", default="test", dest="dataset_split")
    return p


def main() -> None:
    args = build_parser().parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    config = EvalConfig(
        model_choice=args.model_choice,
        json_config_path=args.json_config_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_preprocessing_workers=args.num_preprocessing_workers,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        hf_token=hf_token,
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
    )

    backend_cls = {"GEMMA-4": GemmaBackend, "QWEN3-Omni": QwenBackend}[config.model_choice]
    backend = backend_cls(config)

    print(f"Loading dataset: {config.dataset_name} (split={config.dataset_split})")
    dataset = load_uad_dataset(
        json_config_path=config.json_config_path,
        split=config.dataset_split,
        repo_id=config.dataset_name,
        token=hf_token,
        max_samples=config.max_samples,
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    evaluator = Evaluator(backend, config)
    results = evaluator.evaluate(dataset)

    print(f"\n=== Final Results ===")
    # Add other audio metrics here.
    print(f"  WER:     {results['wer']:.4f}")
    print(f"  Samples: {results['num_samples']}")


if __name__ == "__main__":
    main()
