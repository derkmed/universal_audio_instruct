# Decouple train/ and eval/, and gate the suite in CI

Issue: #61

## Problem

`train/config.py` imports four names from `eval/config.py`
(`DEFAULT_MODEL_PATHS`, `DEFAULT_SEED`, `resolve_dataset_split`,
`validate_clips_per_split`). Training therefore cannot run unless the eval
package imports cleanly, though the two harnesses are meant to be siblings that
share only the loader. The same four names, plus the `resolved_model_path`
property, are the shared surface; `resolved_model_path` is duplicated verbatim in
both config classes, and `validate_clips_per_split`'s rule is duplicated a third
time inline in `uad_data/loader.py`.

Separately, the 162-test offline suite (~18 s) runs nothing automatically. Each
of the 17 test files hand-writes a `sys.path.insert(.., "..")` so it can find the
packages, and three of them also insert `tests/` to import `test_loader_splits`
as a shared fixture module.

## Decision

The shared surface moves to two homes that neither `train/` nor `eval/` owns, so
neither harness imports the other:

- **`uad_data/run_options.py`** (new) holds the dataset-slice helpers:
  `DEFAULT_SEED`, `resolve_dataset_split`, `validate_clips_per_split`. These are
  concerns of what the loader builds, so they belong beside it. `uad_data.loader`
  imports `validate_clips_per_split` from here instead of duplicating it.
- **`models.py`** (new, repo top level) holds the model registry
  `DEFAULT_MODEL_PATHS` and a `resolve_model_path(choice, override)` function.
  Model ids and checkpoint-path resolution are not a dataset concern, so they
  stay out of `uad_data`; a top-level module keeps the loader importable with no
  model config present.

Both `EvalConfig` and `TrainConfig` import from `uad_data.run_options` and
`models`, and their `resolved_model_path` property becomes a one-line call to
`models.resolve_model_path`.

The repo becomes an installable package via `pyproject.toml`, so `import
uad_data`, `import eval`, `import train` and `import models` resolve from an
editable install with no path manipulation. Every per-file `sys.path.insert` for
the repo root is removed. CI runs the suite on every push to `main` and on every
pull request.

## Seams

- **`models.resolve_model_path(choice: str, override: str | None) -> str`**:
  returns `override` if set, else the registry entry for `choice`, else raises
  `ValueError` naming the valid choices. This is the whole of the old
  `resolved_model_path` property body, tested directly with no config object.
- **`uad_data.run_options`**: `resolve_dataset_split` and
  `validate_clips_per_split` keep their current signatures and behaviour; the
  tests that exercise them through the configs still pass unchanged.
- **`EvalConfig` / `TrainConfig`**: same public fields and `__post_init__`
  behaviour as today. Only their imports and the two-line `resolved_model_path`
  body change.
- **The test suite under `python -m pytest tests`**: the runner and the
  collected set do not change; only the removed path inserts do.

## Behaviour

- `train/config.py` has no `from eval...` import. `eval/config.py` has no
  `from train...` import. A grep for cross-harness imports finds nothing.
- `uad_data/loader.py` calls `run_options.validate_clips_per_split`; the inline
  `clips_per_split < 1` check is gone. The error text is unchanged.
- `eval/main.py` and `train/main.py` take `DEFAULT_MODEL_PATHS` from `models` and
  `DEFAULT_SEED` from `uad_data.run_options`; the argparse `choices` are the same
  list as today.
- The notebook's cell 7 import (`from eval.config import EvalConfig,
  DEFAULT_MODEL_PATHS`) is updated in the same commit: it imports `EvalConfig`
  from `eval.config` and, if still referenced, `DEFAULT_MODEL_PATHS` from
  `models`. Cell 2 already `git pull`s the repo, so no other notebook change is
  needed.
- `pyproject.toml` declares the `uad_data`, `eval`, `train` packages (and their
  `backends` subpackages) and the top-level `models` module. The package name
  `eval` is kept as-is (it is how `python -m eval.main` already runs); renaming
  it is out of scope.
- The 17 test files no longer insert the repo root on `sys.path`. Pytest's
  default prepend-import mode already places `tests/` on the path, so
  `import test_loader_splits as fx` in the three fixture-consumers keeps working
  with no manual insert. The subprocess in
  `test_loader_splits.test_template_picks_do_not_depend_on_the_hash_seed` keeps
  its own `sys.path.insert(0, "tests")`, because it is a fresh interpreter; with
  an editable install its child also finds `uad_data`.
- **CI**: `.github/workflows/tests.yml` runs on `push` to `main` and on
  `pull_request`. It sets up Python 3.10, installs CPU-only torch from the
  PyTorch CPU wheel index, then `pip install -e . -r requirements.txt -r
  train/requirements.txt`, with pip caching, and runs `python -m pytest tests`.
  `flash-attn` stays optional and is not installed.

## Acceptance

- `python -m pytest tests` passes from a clean editable install with no
  `sys.path` inserts in any test file except the one subprocess child.
- `git grep -n "from eval" train/ && git grep -n "from train" eval/` finds
  nothing.
- `git grep -n "resolved_model_path" | wc -l` shows the body exists once, in
  `models.py`; both configs only delegate.
- `git grep -n "must be a positive integer"` finds the message once, in
  `uad_data/run_options.py`.
- New `tests/test_models.py` covers `resolve_model_path`: an override wins, a
  known choice resolves, an unknown choice with no override raises naming the
  choices.
- The existing config tests (`tests/test_cli.py` and any that build an
  `EvalConfig`/`TrainConfig`) pass unchanged.
- The CI workflow runs green on a push to `main`, installing in under a few
  minutes with the pip cache warm.

## Out of scope

- Renaming the `eval` package (it shadows the builtin `eval`, but that is a
  separate, larger change).
- The `test_loader_splits`-as-fixture-module smell. It keeps working; extracting
  a real fixtures module is its own cleanup.
- Making the backend imports lazy. CI installs the full stack on purpose, so it
  tests what a real run imports.
- Any change to what the suite asserts. This item is packaging and wiring only.
