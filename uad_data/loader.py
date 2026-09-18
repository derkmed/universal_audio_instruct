"""Expand the UAD dataset into evaluation rows without a HF loading script.

Public entry point: `load_uad_dataset(...)`. It produces the same core fields as
the old `load_dataset("AudioInstruct/Universal-Audio-Understanding", ...,
trust_remote_code=True)` -- audio, system_instruction, prompt, output, task,
split, originating_dataset, plus the per-task metadata fields -- so the
downstream Evaluator is unchanged. Rows also carry every other field of the
clip's metadata record, and a `tasks` list of `Task` values. asr_timestamp_search
rows also carry `utterance_index`, the position in `transcriptions` of the
utterance they render.

The `split` request is `test`, `validation+test` or `all` (see `parse_split`).
Pipeline per internal dataset, in run config order:
  1. work out its selected splits: the requested ones its run config entry lists,
  2. fetch those splits' metadata JSONs and merge them by audio_path,
  3. read the tar archive once, looking up each member's records by audio_path,
  4. for each selected split that lists the member and is still under its clip
     cap, and every (task, utterance, prompt-template) combination, build a Row
     and emit its row. Only asr_timestamp_search renders a clip once per
     utterance (see `Task.utterance_indices`); other tasks render it once,
  5. stop once every selected split is full or has found all its clips.

The rows come back as a list with a `.report` (see `uad_data.load_report`): clips
found per selected split, internal datasets that failed to load, and rows that
failed to render. A smoke run (`clips_per_split` set) records those failures and
carries on; a regular run raises on the first one. Splits that end with fewer
clips than the cap log a warning.

Each internal dataset's archive comes from one of three places (see
`_archive_sources`): its full archive, downloaded and cached (a regular run); its
smoke archive `smoke/<name>.tar.gz`, downloaded and cached (a smoke run, when a
fresh one holds enough clips); or its full archive streamed lazily over HTTP, so
that stopping early transfers only the compressed prefix (a smoke run that can't
use a smoke archive). An explicit `stream` overrides the choice.
"""
import contextlib
import glob
import hashlib
import json
import logging
import random
import tarfile
import zlib
from dataclasses import dataclass
from typing import Any, Iterator

import datasets

from . import filters as filters_lib
from . import hub
from . import prompts as prompts_lib
from . import smoke
from .clip_quota import ClipQuota, WantedClip, wanted_clips
from .collection import UadCollection
from .internal_dataset import InternalDataset
from .json_config_loader import UniversalJsonConfig
from .load_report import LoadedRows, LoadFailure, LoadReport, RenderFailure, SplitReport
from .row import Row
from .tasks import Task

logger = logging.getLogger(__name__)


SPLIT_NAMES = ("train", "validation", "test")
ALL_SPLITS = "all"


def parse_split(split: str | datasets.Split) -> list[str] | None:
    """Parse a split request: `test`, `validation+test`, or `all` (None).

    A `datasets.Split` is read as its name. Raises ValueError for any name
    other than train, validation or test.
    """
    split = str(split)
    if split == ALL_SPLITS:
        return None
    names = split.split("+")
    unknown = [name for name in names if name not in SPLIT_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown split {unknown} in {split!r}: use {', '.join(SPLIT_NAMES)}, "
            f"several joined with '+', or {ALL_SPLITS!r}.")
    return list(dict.fromkeys(names))


def _selected_splits(internal_dataset: InternalDataset, requested: list[str] | None) -> list[str]:
    """The requested splits that the run config entry lists, logging the rest."""
    listed = [str(s) for s in internal_dataset.get_splits()]
    if requested is None:
        return listed
    for name in requested:
        if name not in listed:
            logger.info(
                "Skipping split %s for %s: its run config entry lists only %s.",
                name, internal_dataset.name, listed)
    return [name for name in requested if name in listed]


def _load_split_metadata(metadata_path: str, tasks: list[Task]) -> dict[str, Any]:
    """audio_path -> record (with tasks attached) for one split's metadata JSON."""
    with open(metadata_path, encoding="utf-8") as f:
        return {record["audio_path"]: {**record, "tasks": tasks} for record in json.load(f)}


def _template_rng(
    seed: int,
    dataset_name: str,
    split: str,
    audio_path: str,
    task: Task,
    utterance_index: int | None,
) -> random.Random:
    """A generator for one row's random template pick, fixed by the row's identity.

    Seeded from a sha256 digest rather than hash(), which is salted per process,
    and separate from the shared `random`, which RandomFilter reseeds. So a row
    gets the same pick in every run that includes it, whatever the cap or split
    request. Rows that render one utterance add its index, so each utterance of a
    clip gets its own pick.
    """
    key = [str(seed), dataset_name, split, audio_path, task.value]
    if utterance_index is not None:
        key.append(str(utterance_index))
    digest = hashlib.sha256("\0".join(key).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


class PromptTemplateError(ValueError):
    """A task has no usable prompt templates: a config or Hub problem, not a data defect.

    Every clip of the task would fail the same way, so this stops the run even in
    a smoke run, where a failure to load or render is recorded and stepped over.
    """


def _get_prompt_templates(task: Task, rng: random.Random | None):
    """Select prompt/instruction/output template tuples for a task from PROMPTS_DIR.

    With `rng`, pick one tuple at random with it; without, return them all.
    """
    files = glob.glob(f"{prompts_lib.PROMPTS_DIR}/*.json")
    prompt_files = [prompts_lib.PromptFilepath(filepath=f) for f in files]
    if task not in {pf.task for pf in prompt_files}:
        raise PromptTemplateError(
            f"No prompt file exists for {task} in {prompts_lib.PROMPTS_DIR}/.")
    task_prompt_files = [pf for pf in prompt_files if pf.accepts_task(task)]
    if not task_prompt_files:
        raise PromptTemplateError(f"No prompt files found for task: {task}")
    if len(task_prompt_files) > 1:
        raise PromptTemplateError(
            f"Multiple prompt files correspond to task: {task_prompt_files}. Should only be 1.")
    task_prompt_file = task_prompt_files[0]
    if not task_prompt_file.all_templates:
        raise PromptTemplateError(
            f"The prompt file {task_prompt_file.filepath} for {task} gives no templates.")
    if rng is not None:
        return [task_prompt_file.random_template_selection(rng)]
    return task_prompt_file.all_templates


@dataclass
class ArchiveSource:
    """Where to read one internal dataset's clips from.

    `path` is a repo path or URL: the full archive, or a smoke archive. `stream`
    reads it lazily over HTTP instead of downloading and caching it.
    `recorded_error` is the read error that stopped a smoke archive's build, if
    one did.
    """
    path: str
    stream: bool
    recorded_error: str | None = None


class RecordedLoadError(OSError):
    """A smoke archive ended where its build hit a read error, short of what a run wants.

    A full-archive run would hit that error there, so the smoke run reports it
    as a failed load, with the error the build recorded.
    """


@contextlib.contextmanager
def _open_archive(
    source: ArchiveSource, *, repo_id: str, revision: str | None, token: str | None,
) -> Iterator[tarfile.TarFile]:
    """Yield a streaming tar handle for an internal dataset's audio archive.

    Without `source.stream` the whole `.tar.gz` is downloaded (and cached) first;
    with it, it's read lazily over HTTP so an early break transfers only the
    prefix consumed. Either way the caller gets a sequential `r|gz` tar object.
    """
    if source.stream:
        fileobj = hub.open_archive_stream(
            source.path, repo_id=repo_id, revision=revision, token=token)
        try:
            with tarfile.open(fileobj=fileobj, mode="r|gz") as archive:
                yield archive
        finally:
            fileobj.close()
    else:
        tar_path = hub.download_file(
            source.path, repo_id=repo_id, revision=revision, token=token)
        with tarfile.open(tar_path, "r|gz") as archive:
            yield archive


def _archive_sources(
    collection: UadCollection,
    internal_datasets: list[InternalDataset],
    *,
    clips_per_split: int | None,
    stream: bool | None,
    repo_id: str,
    revision: str | None,
    token: str | None,
) -> dict[str, ArchiveSource]:
    """Choose each internal dataset's archive source, keyed by its name.

    An explicit `stream` reads the full archive as asked. A regular run
    downloads it. A smoke run uses a fresh smoke archive holding at least
    `clips_per_split` clips per split when the row filter is `all_pass`, and
    otherwise streams the full archive, with a log line.
    """
    if stream is not None or clips_per_split is None:
        return {d.name: ArchiveSource(d.data_url, stream=bool(stream))
                for d in internal_datasets}
    if not isinstance(collection.row_filter, filters_lib.AllPassFilter):
        for d in internal_datasets:
            logger.info(
                "Streaming the full archive of %s: smoke archives hold the first clips "
                "whatever the row filter, so they only serve the all_pass filter.", d.name)
        return {d.name: ArchiveSource(d.data_url, stream=True) for d in internal_datasets}
    manifest = _smoke_manifest(repo_id=repo_id, revision=revision, token=token)
    return {
        d.name: _smoke_or_stream(
            d, manifest.get(d.name), clips_per_split,
            repo_id=repo_id, revision=revision, token=token)
        for d in internal_datasets
    }


def _smoke_manifest(*, repo_id: str, revision: str | None, token: str | None,
                    ) -> dict[str, smoke.SmokeEntry]:
    """The Hub's smoke manifest, or no entries when it can't be fetched."""
    try:
        path = hub.download_file(
            smoke.MANIFEST_PATH, repo_id=repo_id, revision=revision, token=token)
    except hub.EntryNotFoundError as error:
        logger.info("No smoke manifest (%s); smoke runs read full archives.", error)
        return {}
    except OSError as error:
        logger.warning(
            "Could not fetch the smoke manifest, so smoke runs read full archives: %s",
            describe_error(error))
        return {}
    return smoke.read_manifest(path)


def _smoke_or_stream(
    internal_dataset: InternalDataset,
    entry: smoke.SmokeEntry | None,
    clips_per_split: int,
    *,
    repo_id: str,
    revision: str | None,
    token: str | None,
) -> ArchiveSource:
    """An internal dataset's smoke archive when it's usable, else its streamed full archive."""
    name = internal_dataset.name
    full = ArchiveSource(internal_dataset.data_url, stream=True)
    if entry is None:
        logger.info("Streaming the full archive of %s: it has no smoke archive.", name)
        return full
    if clips_per_split > entry.clips_per_split:
        logger.info(
            "Streaming the full archive of %s: its smoke archive holds %d clips per "
            "split, fewer than the %d asked for.", name, entry.clips_per_split, clips_per_split)
        return full
    if _smoke_is_stale(internal_dataset, entry, repo_id=repo_id, revision=revision, token=token):
        logger.warning(
            "Streaming the full archive of %s: its smoke archive is stale, built "
            "from a full archive that has since changed.", name)
        return full
    return ArchiveSource(
        smoke.smoke_archive_path(name), stream=False, recorded_error=entry.error)


def _smoke_is_stale(
    internal_dataset: InternalDataset,
    entry: smoke.SmokeEntry,
    *,
    repo_id: str,
    revision: str | None,
    token: str | None,
) -> bool:
    """Whether the full archive's sha256 on the Hub differs from the one the smoke build used.

    A full archive that's gone from the Hub counts as changed. When the check
    can't run (offline, say), logs a warning and says not stale.
    """
    try:
        current = hub.file_sha256(
            internal_dataset.data_url, repo_id=repo_id, revision=revision, token=token)
    except hub.EntryNotFoundError:
        return True
    except Exception as error:  # noqa: BLE001 -- any failure means the check can't run.
        logger.warning(
            "Could not check whether the smoke archive of %s is stale, so using it "
            "anyway: %s", internal_dataset.name, describe_error(error))
        return False
    return current != entry.source_sha256


def _iter_rows(
    collection: UadCollection,
    split: str | datasets.Split,
    *,
    repo_id: str,
    revision: str | None,
    token: str | None,
    clips_per_split: int | None = None,
    seed: int = 42,
    stream: bool | None = None,
    report: LoadReport | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield expanded row dicts for the requested splits, one archive read per internal dataset.

    Rows are grouped by internal dataset in run config order and follow archive
    order within one, so rows of different splits are mixed. With
    `clips_per_split`, each selected split takes only its first that many clips
    in archive order; a clip counts once one of its rows passes the row filter.
    Reading an archive stops once every selected split is satisfied: full, or
    with every clip its metadata lists found. Raises ValueError when no internal
    dataset lists any requested split.

    With `report`, records each selected split's clips found in it. A smoke
    run (`clips_per_split` set) with a `report` also carries on past failures,
    recording them there: an internal dataset that fails to load keeps the rows
    read before the failure, and a row that fails to render is left out while
    its clip still counts. Otherwise both errors propagate.
    """
    requested = parse_split(split)
    tolerant = report is not None and clips_per_split is not None
    selections = [
        (internal_dataset, selected)
        for internal_dataset in collection.internal_datasets
        if (selected := _selected_splits(internal_dataset, requested))
    ]
    if not selections:
        raise ValueError(
            f"No internal dataset in run config {collection.name!r} lists split {split!r}.")
    sources = _archive_sources(
        collection, [internal_dataset for internal_dataset, _ in selections],
        clips_per_split=clips_per_split, stream=stream,
        repo_id=repo_id, revision=revision, token=token)

    for internal_dataset, selected in selections:
        found = {
            split_name: SplitReport(
                dataset=internal_dataset.name, split=split_name,
                tasks=[task.value for task in internal_dataset.tasks])
            for split_name in selected
        }
        if report is not None:
            report.splits.extend(found.values())
        dataset_rows = _dataset_rows(
            collection, internal_dataset, found, sources[internal_dataset.name],
            repo_id=repo_id, revision=revision, token=token,
            clips_per_split=clips_per_split, seed=seed,
            render_failures=report.render_failures if tolerant else None)
        if tolerant:
            yield from _rows_until_failure(dataset_rows, internal_dataset.name, report)
        else:
            yield from dataset_rows
        if clips_per_split is not None:
            for entry in found.values():
                if entry.clips_found < clips_per_split:
                    logger.warning(
                        "%s %s has %d clip(s), fewer than the %d asked for.",
                        entry.dataset, entry.split, entry.clips_found, clips_per_split)


# What "this internal dataset failed to load" is made of: a missing or unreadable
# metadata JSON, and an archive that can't be fetched, decompressed or read.
# `PromptTemplateError` and anything else is a config or code problem that every
# internal dataset would hit, so it stops the run even in a smoke run.
LOAD_ERRORS = (OSError, tarfile.TarError, EOFError, zlib.error, json.JSONDecodeError)


def _rows_until_failure(
    dataset_rows: Iterator[dict[str, Any]], dataset: str, report: LoadReport,
) -> Iterator[dict[str, Any]]:
    """Yield an internal dataset's rows until they end or fail; record a failure in `report`."""
    rows_kept = 0
    while True:
        try:
            row = next(dataset_rows)
        except StopIteration:
            break
        except LOAD_ERRORS as error:
            logger.warning(
                "Failed to load %s after %d rows; moving on: %s",
                dataset, rows_kept, error)
            report.load_failures.append(LoadFailure(
                dataset=dataset,
                error=describe_error(error),
                rows_kept=rows_kept))
            break
        yield row
        rows_kept += 1


def describe_error(error: Exception) -> str:
    """An error as the load report and the smoke manifest record it.

    A recorded build error is already described, so it's given as recorded.
    """
    if isinstance(error, RecordedLoadError):
        return str(error)
    return f"{type(error).__name__}: {error}"


def _dataset_rows(
    collection: UadCollection,
    internal_dataset: InternalDataset,
    found: dict[str, SplitReport],
    source: ArchiveSource,
    *,
    repo_id: str,
    revision: str | None,
    token: str | None,
    clips_per_split: int | None,
    seed: int,
    render_failures: list[RenderFailure] | None,
) -> Iterator[dict[str, Any]]:
    """Yield one internal dataset's rows from one archive read, counting clips in `found`.

    `found` maps each selected split to its report entry. When `source` is a
    smoke archive whose build hit a read error, and a selected split still wants
    clips at its end, raises `RecordedLoadError`: a full-archive read would have
    hit that error there.
    """
    randomize = collection.is_random_prompt_format_selection()
    clips = _merged_metadata(
        internal_dataset, list(found), repo_id=repo_id, revision=revision, token=token)
    quota = ClipQuota(
        {split_name: [path for path, records in clips.items() if split_name in records]
         for split_name in found},
        clips_per_split)

    for clip in _read_wanted_clips(
            source, quota, repo_id=repo_id, revision=revision, token=token):
        for split_name in clip.splits:
            rows, counts = _clip_rows(
                collection, internal_dataset, split_name, clip.audio_path,
                clip.data, clips[clip.audio_path][split_name], randomize, seed,
                render_failures)
            yield from rows
            if counts:
                quota.count(split_name)
                found[split_name].clips_found = quota.found(split_name)

    if source.recorded_error and not quota.all_satisfied():
        raise RecordedLoadError(source.recorded_error)


def _merged_metadata(
    internal_dataset: InternalDataset,
    splits: list[str],
    *,
    repo_id: str,
    revision: str | None,
    token: str | None,
) -> dict[str, dict[str, Any]]:
    """audio_path -> {split: record}, over the metadata of every split in `splits`."""
    clips: dict[str, dict[str, Any]] = {}
    for split_name in splits:
        metadata_path = hub.download_file(
            internal_dataset.split_metadata_path(datasets.Split(split_name)),
            repo_id=repo_id, revision=revision, token=token)
        for audio_path, record in _load_split_metadata(
                metadata_path, internal_dataset.tasks).items():
            clips.setdefault(audio_path, {})[split_name] = record
    return clips


def _read_wanted_clips(
    source: ArchiveSource,
    quota: ClipQuota,
    *,
    repo_id: str,
    revision: str | None,
    token: str | None,
) -> Iterator[WantedClip]:
    """Open `source` and yield the clips `quota` still wants, stopping once it wants none.

    The archive is read as a sequential stream (archives are multi-GB); a
    streamed source transfers only the prefix up to the early-stop point.
    """
    with _open_archive(source, repo_id=repo_id, revision=revision, token=token) as archive:
        yield from wanted_clips(archive, quota)


def _clip_rows(
    collection: UadCollection,
    internal_dataset: InternalDataset,
    split: str,
    audio_path: str,
    file_bytes: bytes,
    record: dict[str, Any],
    randomize: bool,
    seed: int,
    render_failures: list[RenderFailure] | None,
) -> tuple[list[dict[str, Any]], bool]:
    """One clip's rendered rows for one split, and whether the clip counts.

    The clip counts once one of its rows passes the run's row filter, whether or
    not that row renders. With `render_failures`, a row that fails to render is
    recorded there and left out; without it, the error propagates.
    """
    rows = []
    counts = False

    def failed(task: Task, utterance_index: int | None, error: Exception) -> None:
        if render_failures is None:
            raise error
        failure = RenderFailure(
            dataset=internal_dataset.name, split=split, task=task.value,
            audio_path=audio_path, error=describe_error(error),
            utterance_index=utterance_index)
        logger.warning(
            "Failed to render a %s %s %s row for %s (utterance %s); moving on: %s",
            failure.dataset, failure.split, failure.task, failure.audio_path,
            failure.utterance_index, failure.error)
        render_failures.append(failure)

    for task in record["tasks"]:
        # One pass per utterance for asr_timestamp_search, one pass otherwise.
        for utterance_index in task.utterance_indices(record):
            rng = _template_rng(
                seed, internal_dataset.name, split, audio_path, task, utterance_index,
            ) if randomize else None
            # A task with no templates raises (PromptTemplateError), in smoke runs too.
            for si_t, p_t, o_t in _get_prompt_templates(task, rng):
                row = Row(
                    audio_path=audio_path,
                    dataset_name=internal_dataset.name,
                    split=split,
                    task=task,
                    audio_data=file_bytes,
                    metadata=record,
                    system_instruction_template=si_t,
                    prompt_template=p_t,
                    output_template=o_t,
                    utterance_index=utterance_index,
                )
                if not collection.row_filter.include_row(row):
                    continue
                counts = True
                try:
                    rows.append(row.to_output())
                except Exception as error:
                    # Every template of this task renders the same fields, so it
                    # fails the same way: report the pass once.
                    failed(task, utterance_index, error)
                    break
    return rows, counts


def load_uad_dataset(
    *,
    json_config_path: str,
    split: str | datasets.Split,
    repo_id: str = hub.DEFAULT_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
    clips_per_split: int | None = None,
    seed: int = 42,
    stream: bool | None = None,
) -> LoadedRows:
    """Load and expand the UAD dataset for a given config + split.

    Args:
        json_config_path: Either a local path to a UAD JSON config, or the name of
            a config hosted in the repo's `universal_audio_dataset_configs/` folder.
        split: The splits to load: one name (`test`), several joined with `+`
            (`validation+test`), or `all` for every split each run config entry
            lists. Names are `train`, `validation` and `test`.
        repo_id: HF Hub dataset repo id (private).
        revision: Optional Hub revision/commit to pin for reproducibility.
        token: HF access token (required for the private repo).
        clips_per_split: Optional cap: take only the first this many clips, in
            archive order, of each selected split of each internal dataset.
            None (the default) takes every clip.
        seed: Seeds each row's template pick when the run config sets
            `randomize_prompt_format`. A row gets the same pick in every run
            with the same seed.
        stream: Leave unset (None) to let the loader choose where each archive
            is read from: a regular run downloads and caches the full archive; a
            smoke run (`clips_per_split` set) downloads a fresh smoke archive
            (`smoke/<name>.tar.gz`) that holds enough clips when the run config's
            row filter is `all_pass`, and otherwise streams the full archive, so
            an early stop transfers only the prefix consumed. True or False
            bypasses smoke archives and streams, or downloads, the full archive:
            the way to check a smoke run against the real archive. Streamed reads
            are not cached, so avoid stream=True for large or repeated runs.

    Returns:
        The row dicts, consumable directly by the Evaluator, as a list whose
        `report` is the load report (see `uad_data.load_report`).
    """
    if clips_per_split is not None and clips_per_split < 1:
        raise ValueError(f"clips_per_split must be a positive integer, got {clips_per_split}.")

    config_path = _resolve_config_path(
        json_config_path, repo_id=repo_id, revision=revision, token=token)

    # Make the prompt templates available locally and point the library at them.
    prompts_lib.PROMPTS_DIR = hub.download_prompts_dir(
        repo_id=repo_id, revision=revision, token=token)

    collection = UniversalJsonConfig(filepath=config_path).toCollection()
    report = LoadReport(clips_per_split=clips_per_split)
    rows = list(_iter_rows(
        collection, split,
        repo_id=repo_id, revision=revision, token=token,
        clips_per_split=clips_per_split, seed=seed, stream=stream, report=report))
    return LoadedRows(rows, report)


def _resolve_config_path(
    json_config_path: str,
    *,
    repo_id: str,
    revision: str | None,
    token: str | None,
) -> str:
    """Use a local config file if it exists, else fetch it from the repo by name."""
    import os
    if os.path.exists(json_config_path):
        return json_config_path
    if json_config_path.startswith("universal_audio_dataset_configs/"):
        repo_relative = json_config_path
    else:
        repo_relative = f"universal_audio_dataset_configs/{os.path.basename(json_config_path)}"
    return hub.download_file(
        repo_relative, repo_id=repo_id, revision=revision, token=token)
