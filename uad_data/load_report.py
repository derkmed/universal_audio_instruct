"""The load report that comes back with a run's rows.

`load_uad_dataset` returns `LoadedRows`: a plain list of row dicts that also
carries a `LoadReport`. The report lists clips found for each internal dataset
and selected split (with that split's tasks, so every group can be listed, even
an empty one), internal datasets that failed to load, and rows that failed to
render.
"""
from dataclasses import dataclass, field


@dataclass
class SplitReport:
    """Clips found for one (internal dataset, selected split)."""
    dataset: str
    split: str
    tasks: list[str]
    clips_found: int = 0


@dataclass
class LoadFailure:
    """An internal dataset that stopped loading, and the rows it kept before that."""
    dataset: str
    error: str
    rows_kept: int


@dataclass
class RenderFailure:
    """A row that failed to render. Its clip still counts toward the cap."""
    dataset: str
    split: str
    task: str
    audio_path: str
    error: str
    utterance_index: int | None = None


@dataclass
class LoadReport:
    """What a load found. `clips_per_split` is the run's cap (None when uncapped)."""
    clips_per_split: int | None = None
    splits: list[SplitReport] = field(default_factory=list)
    load_failures: list[LoadFailure] = field(default_factory=list)
    render_failures: list[RenderFailure] = field(default_factory=list)


class LoadedRows(list):
    """The usable rows of a load, as a list, with the load's `report`."""

    def __init__(self, rows=(), report: LoadReport | None = None):
        super().__init__(rows)
        self.report = report if report is not None else LoadReport()
