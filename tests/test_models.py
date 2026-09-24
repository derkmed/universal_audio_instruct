"""Offline tests for the shared model registry (`models.py`).

`resolve_model_path` is the whole of what both `EvalConfig.resolved_model_path`
and `TrainConfig.resolved_model_path` used to duplicate. Testing it here means
the rule -- override wins, else the registry, else a ValueError naming the
choices -- is checked once, with no config object in the way.

Runnable directly (`python tests/test_models.py`) or under pytest.
"""
import models


def test_an_override_wins_over_the_registry() -> None:
    assert models.resolve_model_path("GEMMA-4", "org/my-finetune") == "org/my-finetune"
    # Even a choice the registry has never heard of: the override is the answer.
    assert models.resolve_model_path("NOT-A-MODEL", "org/my-finetune") == "org/my-finetune"

    print("PASS: an override wins over the registry.")


def test_a_known_choice_resolves_to_its_default_path() -> None:
    for choice, path in models.DEFAULT_MODEL_PATHS.items():
        assert models.resolve_model_path(choice, None) == path, choice
    assert models.resolve_model_path("GEMMA-4", None) == "google/gemma-4-e2b-it"

    print("PASS: a known choice resolves to its registry path.")


def test_an_unknown_choice_with_no_override_raises_naming_the_choices() -> None:
    try:
        models.resolve_model_path("NOT-A-MODEL", None)
    except ValueError as error:
        message = str(error)
        assert "NOT-A-MODEL" in message, message
        for choice in models.DEFAULT_MODEL_PATHS:
            assert choice in message, (choice, message)
    else:
        raise AssertionError("an unknown model_choice did not raise")

    print("PASS: an unknown choice raises, naming the valid choices.")


def test_a_blank_override_falls_back_to_the_registry() -> None:
    """`model_path` is an empty string when a form field is left blank."""
    assert models.resolve_model_path("GEMMA-4", "") == "google/gemma-4-e2b-it"

    print("PASS: a blank override falls back to the registry.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
