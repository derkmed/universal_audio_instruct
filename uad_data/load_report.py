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

    def describe(self) -> str:
        """The report as text, for a run to print before it starts work."""
        cap = self.clips_per_split
        header = "Load report" + (f" (clips per split: {cap})" if cap is not None else "")
        lines = [header]

        for split in self.splits:
            found = split.clips_found
            clips = f"{found} clip" + ("" if found == 1 else "s")
            short = f" of {cap}" if cap is not None and found < cap else ""
            lines.append(
                f"  {split.dataset}/{split.split}: {clips}{short} "
                f"({', '.join(split.tasks)})")

        for failure in self.load_failures:
            lines.append(
                f"  FAILED {failure.dataset}: {failure.error} "
                f"({failure.rows_kept} rows kept)")

        for failure in self.render_failures:
            utterance = (
                "" if failure.utterance_index is None
                else f" utterance {failure.utterance_index}")
            lines.append(
                f"  UNRENDERED {failure.dataset}/{failure.split} {failure.task} "
                f"{failure.audio_path}{utterance}: {failure.error}")

        return "\n".join(lines)

    @property
    def has_problems(self) -> bool:
        """Whether this load hit anything a smoke run should fail on.

        A selected split that found no clips at all is a problem; one that found
        fewer clips than the cap is only a warning.
        """
        return bool(self.load_failures) or bool(self.render_failures) or any(
            split.clips_found == 0 for split in self.splits)


class LoadedRows(list):
    """The usable rows of a load, as a list, with the load's `report`."""

    def __init__(self, rows=(), report: LoadReport | None = None):
        super().__init__(rows)
        self.report = report if report is not None else LoadReport()
