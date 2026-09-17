"""CLI entry point for audio instruction evaluation.

Run from the repo root as a module:

    python -m eval.main --model GEMMA-4 --json-config configs/clotho_config.json

Flow: parse flags into an `EvalConfig`, build the chosen model backend, load the
requested dataset slice via `uad_data.load_uad_dataset` (which fetches audio +
metadata + prompts from the private HF Hub repo -- no loading script), then run
batched inference through the `Evaluator` and print/save metrics.
"""

import argparse
import os

from .backends import GemmaBackend, QwenBackend
from .config import DEFAULT_MODEL_PATHS, DEFAULT_SEED, EvalConfig
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
        "--clips-per-split",
        type=int,
        default=None,
        dest="clips_per_split",
        help="Evaluate only the first N clips of each selected split (for smoke runs)",
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
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        dest="seed",
        help="Seeds the prompt-template picks, so a clip keeps its template across runs",
    )
    p.add_argument("--dataset", default="AudioInstruct/Universal-Audio-Understanding")
    p.add_argument(
        "--split",
        default=None,
        dest="dataset_split",
        help="Splits to load: one name, several joined with '+', or 'all' "
             "(default: all for a smoke run, test otherwise)",
    )
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
        clips_per_split=args.clips_per_split,
        seed=args.seed,
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
        clips_per_split=config.clips_per_split,
        seed=config.seed,
    )
    print(f"Dataset loaded: {len(dataset)} rows")

    # The group table carries what the load report would say here -- clips found
    # against the cap, and the internal datasets that failed to load -- so the
    # summary is printed once, by the evaluator, rather than twice in two shapes.
    evaluator = Evaluator(backend, config)
    summary = evaluator.evaluate(dataset)

    # A smoke run is a check, so its exit status reports what it found. A regular
    # run exits 0 unless something raised, even when a group fails on
    # empty_output. Name the cause: an operator whose archive was truncated
    # shouldn't be sent looking for a failing group in an all-PASS table.
    if config.is_smoke_run and not summary["passed"]:
        causes = []
        if any(not group["passed"] for group in summary["groups"]):
            causes.append("failing groups")
        if summary["load_failures"]:
            causes.append("internal datasets that failed to load")
        if not summary["groups"]:
            causes.append("no groups at all — nothing was evaluated")
        raise SystemExit(
            f"Smoke run finished with {' and '.join(causes)}. See the table above.")


if __name__ == "__main__":
    main()
