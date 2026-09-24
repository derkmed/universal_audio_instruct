"""Offline tests that the editable install the suite depends on is real.

No test file inserts the repo root into `sys.path` any more, so `uad_data`,
`models`, `eval` and `train` resolve only through `pip install -e .`. Three
things have to hold for that to be true rather than lucky: the project has to
declare pytest (nothing in `requirements.txt` pulls it in), CI has to install it
and run the suite from outside the repo root -- from the root, `python -m pytest`
puts the source tree on `sys.path` and a broken `pyproject.toml` still looks
green -- and every setup block a human follows has to say to install the repo.

Runnable directly (`python tests/test_packaging.py`) or under pytest. Only
requires `pyyaml`, which `datasets` and `transformers` both already need.
"""
import os
import re

import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "tests.yml")
PYPROJECT = os.path.join(ROOT, "pyproject.toml")
# Every doc that hands a reader a setup command: the two harness READMEs and
# FINETUNING.md each repeat the root README's block.
SETUP_DOCS = [
    "README.md",
    os.path.join("eval", "README.md"),
    os.path.join("train", "README.md"),
    "FINETUNING.md",
]


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _install_runs_and_suite_step() -> tuple[list[str], dict]:
    """Split the job's steps into the installs and the one step that runs pytest."""
    steps = yaml.safe_load(_read(WORKFLOW))["jobs"]["pytest"]["steps"]
    suite = [step for step in steps if "pytest" in step.get("run", "")]
    assert len(suite) == 1, suite
    return [step.get("run", "") for step in steps if step is not suite[0]], suite[0]


def _declared_extras() -> str:
    """The `[project.optional-dependencies]` table, as text.

    Read as text rather than parsed: `tomllib` arrived in 3.11 and this project
    supports 3.10, which is also the version CI installs.
    """
    text = _read(PYPROJECT)
    assert "[project.optional-dependencies]" in text, "pyproject declares no extras"
    return text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]


def test_the_project_declares_pytest() -> None:
    assert "pytest" in _declared_extras(), _declared_extras()

    print("PASS: pyproject declares pytest as an extra.")


def test_ci_installs_pytest_before_running_the_suite() -> None:
    installs, _ = _install_runs_and_suite_step()
    extras = re.findall(r"pip install[^\n]*?\.\[([\w,-]+)\]", "\n".join(installs))
    named = [extra for extra in extras if extra in _declared_extras()]
    assert "pytest" in "\n".join(installs) or named, installs

    print("PASS: CI installs pytest before running the suite.")


def test_ci_runs_the_suite_from_outside_the_repo_root() -> None:
    _, suite = _install_runs_and_suite_step()
    assert suite.get("working-directory"), suite

    print("PASS: CI runs the suite from outside the repo root.")


def test_every_setup_block_documents_the_editable_install() -> None:
    for doc in SETUP_DOCS:
        # Any language tag, so fences stay correctly paired in docs that also
        # hold ```python and ```text blocks.
        blocks = re.findall(r"```\w*\n(.*?)```", _read(os.path.join(ROOT, doc)), re.S)
        installs = [b for b in blocks if "pip install" in b and "requirements.txt" in b]
        assert installs, f"{doc}: no setup block found"
        for block in installs:
            assert "pip install -e ." in block, (doc, block)

    print("PASS: every setup block installs the repo itself.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
