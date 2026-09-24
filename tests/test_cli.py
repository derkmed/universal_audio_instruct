"""Offline tests for the eval and train command lines and config classes (S3, S4).

Checks that both command lines take `--clips-per-split` and `--seed`, leave
`--split` unset by default, and no longer take `--max-samples`; and that both
config classes store `clips_per_split`, reject a cap below 1, carry a `seed`,
and resolve an unset `dataset_split`.

Runnable directly (`python tests/test_cli.py`) or under pytest. Needs all of
`requirements.txt` (CPU `torch` is enough): both `main` modules import it.
"""
from eval import main as eval_main
from eval.config import EvalConfig
from train import main as train_main
from train.config import TrainConfig

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


def test_configs_reject_a_non_positive_clips_per_split() -> None:
    for config_cls in (EvalConfig, TrainConfig):
        for bad in (0, -1):
            try:
                config_cls(model_choice="GEMMA-4", clips_per_split=bad)
            except ValueError:
                continue
            raise AssertionError(f"{config_cls.__name__} accepted clips_per_split={bad}")

    print("PASS: both configs reject a clips_per_split below 1.")


def test_dataset_split_resolves_from_the_cap() -> None:
    # Unset: "all" for a smoke run, the harness's own default otherwise.
    assert EvalConfig(model_choice="GEMMA-4").dataset_split == "test"
    assert TrainConfig(model_choice="GEMMA-4").dataset_split == "train"
    assert EvalConfig(model_choice="GEMMA-4", clips_per_split=5).dataset_split == "all"
    assert TrainConfig(model_choice="GEMMA-4", clips_per_split=5).dataset_split == "all"

    # An explicit value always wins.
    for config_cls in (EvalConfig, TrainConfig):
        config = config_cls(
            model_choice="GEMMA-4", clips_per_split=5, dataset_split="validation")
        assert config.dataset_split == "validation", config_cls.__name__

    # A blank notebook field is no split at all, not the split "".
    for config_cls in (EvalConfig, TrainConfig):
        config = config_cls(model_choice="GEMMA-4", clips_per_split=5, dataset_split="")
        assert config.dataset_split == "all", config_cls.__name__

    print("PASS: dataset_split defaults to all for a smoke run, and explicit wins.")


def test_configs_seed_the_loader_at_42() -> None:
    for config_cls in (EvalConfig, TrainConfig):
        assert config_cls(model_choice="GEMMA-4").seed == 42, config_cls.__name__
        assert config_cls(model_choice="GEMMA-4", seed=7).seed == 7, config_cls.__name__

    print("PASS: both configs carry a seed defaulting to 42.")


def test_seed_flag() -> None:
    for name, build_parser in PARSERS.items():
        args = build_parser().parse_args(["--model", "GEMMA-4", "--seed", "7"])
        assert args.seed == 7, f"{name}: {args}"
        args = build_parser().parse_args(["--model", "GEMMA-4"])
        assert args.seed == 42, f"{name}: {args}"

    print("PASS: both command lines take --seed, defaulting to 42.")


def test_split_flag_defaults_to_unset() -> None:
    """The config, not the parser, decides what an unasked-for split means."""
    for name, build_parser in PARSERS.items():
        args = build_parser().parse_args(["--model", "GEMMA-4"])
        assert args.dataset_split is None, f"{name}: {args}"
        args = build_parser().parse_args(["--model", "GEMMA-4", "--split", "validation+test"])
        assert args.dataset_split == "validation+test", f"{name}: {args}"

    print("PASS: --split is unset by default on both command lines.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
