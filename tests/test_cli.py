"""Offline tests for the eval and train command lines and config classes (S3, S4).

Checks that both command lines take `--clips-per-split` and no longer take
`--max-samples`, and that both config classes store `clips_per_split`.

Runnable directly (`python tests/test_cli.py`) or under pytest. Needs all of
`requirements.txt` (CPU `torch` is enough): both `main` modules import it.
"""
import os
import sys

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval import main as eval_main  # noqa: E402
from eval.config import EvalConfig  # noqa: E402
from train import main as train_main  # noqa: E402
from train.config import TrainConfig  # noqa: E402

PARSERS = {"eval": eval_main.build_parser, "train": train_main.build_parser}


def test_clips_per_split_flag() -> None:
    for name, build_parser in PARSERS.items():
        args = build_parser().parse_args(["--model", "GEMMA-4", "--clips-per-split", "5"])
        assert args.clips_per_split == 5, f"{name}: {args}"
        args = build_parser().parse_args(["--model", "GEMMA-4"])
        assert args.clips_per_split is None, f"{name}: {args}"

    print("PASS: both command lines take --clips-per-split, unset by default.")


def test_max_samples_flag_is_gone() -> None:
    for name, build_parser in PARSERS.items():
        try:
            build_parser().parse_args(["--model", "GEMMA-4", "--max-samples", "5"])
        except SystemExit:
            continue
        raise AssertionError(f"{name} still accepts --max-samples")

    print("PASS: neither command line accepts --max-samples.")


def test_configs_store_clips_per_split() -> None:
    assert EvalConfig(model_choice="GEMMA-4", clips_per_split=5).clips_per_split == 5
    assert TrainConfig(model_choice="GEMMA-4", clips_per_split=5).clips_per_split == 5
    for config_cls in (EvalConfig, TrainConfig):
        assert not hasattr(config_cls(model_choice="GEMMA-4"), "max_samples"), config_cls

    print("PASS: both configs store clips_per_split and drop max_samples.")


if __name__ == "__main__":
    test_clips_per_split_flag()
    test_max_samples_flag_is_gone()
    test_configs_store_clips_per_split()
