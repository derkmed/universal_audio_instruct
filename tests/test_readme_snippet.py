"""Offline test: README.md's library snippet calls load_uad_dataset correctly.

Parses the Python code block under "Loading the dataset directly" and checks
that every keyword it passes to `load_uad_dataset` is one the function accepts,
so a renamed argument can't leave the README raising TypeError.

Runnable directly (`python tests/test_readme_snippet.py`) or under pytest. Only
requires `datasets`, `jinja2`, `huggingface_hub`.
"""
import ast
import inspect
import os
import re
import sys

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uad_data import load_uad_dataset  # noqa: E402

README = os.path.join(os.path.dirname(__file__), "..", "README.md")


def _snippet() -> str:
    with open(README, encoding="utf-8") as f:
        text = f.read()
    section = text.split("## Loading the dataset directly", 1)[1]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


def test_readme_snippet_passes_only_accepted_arguments() -> None:
    calls = [
        node for node in ast.walk(ast.parse(_snippet()))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_uad_dataset"
    ]
    assert len(calls) == 1, calls

    accepted = set(inspect.signature(load_uad_dataset).parameters)
    passed = {kw.arg for kw in calls[0].keywords}
    assert passed <= accepted, f"README passes unknown arguments: {passed - accepted}"

    print("PASS: the README snippet passes only arguments load_uad_dataset accepts.")


if __name__ == "__main__":
    test_readme_snippet_passes_only_accepted_arguments()
