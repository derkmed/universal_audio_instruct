"""The Colab notebook's code cells parse as Python.

Nothing else runs the notebook offline -- it wants a GPU and the Hub -- so a
mangled string or a stale name only shows up when someone opens it in Colab.
Parsing every cell is cheap and catches exactly that.

Cells that use Colab's `!shell` or `%magic` lines aren't Python and are skipped.

Runnable directly (`python tests/test_notebook.py`) or under pytest.
"""
import ast
import json
import os
import sys

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

NOTEBOOK = os.path.join(
    os.path.dirname(__file__), "..", "eval", "colab_eval.ipynb")


def _code_cells() -> list[tuple[int, str, str]]:
    with open(NOTEBOOK, encoding="utf-8") as f:
        notebook = json.load(f)
    cells = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
            continue  # Colab magics, not Python
        cells.append((index, cell.get("id", str(index)), source))
    return cells


def test_every_python_cell_parses() -> None:
    for index, cell_id, source in _code_cells():
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(f"cell {index} ({cell_id}) is not valid Python: {error}")

    print("PASS: every Python cell of the notebook parses.")


def test_the_results_cells_read_the_summary_the_evaluator_returns() -> None:
    """`evaluate` returns the summary; the old wer/predictions keys are gone."""
    sources = {cell_id: source for _, cell_id, source in _code_cells()}

    assert "summary = evaluator.evaluate(dataset)" in sources["run-eval"], sources["run-eval"]
    for cell_id in ("show-results", "show-samples"):
        source = sources[cell_id]
        for gone in ("results[", "['wer']", "num_samples"):
            assert gone not in source, (cell_id, gone, source)

    print("PASS: the results cells read the returned summary.")


def test_the_preview_lists_rows_that_are_not_ok_first() -> None:
    """On a long run the interesting rows are the failures, not the first five."""
    source = dict((cell_id, src) for _, cell_id, src in _code_cells())["show-samples"]

    def _record(index: int, status: str) -> dict:
        return {
            "index": index, "status": status, "originating_dataset": "Clotho",
            "split": "test", "task": "caption", "answer": "a", "prediction": "a",
            "error": None,
        }

    shown = []
    namespace = {
        "summary": {"rows": [
            _record(0, "ok"), _record(1, "ok"), _record(2, "audio_error"),
            _record(3, "ok"), _record(4, "ok"), _record(5, "empty_output"),
        ]},
        "print": lambda *args: shown.append(" ".join(str(a) for a in args)),
    }
    exec(compile(source, "show-samples", "exec"), namespace)

    previewed = [line for line in shown if line.startswith("[")]
    assert previewed[0].startswith("[2]") and previewed[1].startswith("[5]"), previewed

    print("PASS: the preview cell lists non-ok rows first.")


if __name__ == "__main__":
    for _name, _test in list(globals().items()):
        if _name.startswith("test_"):
            _test()
