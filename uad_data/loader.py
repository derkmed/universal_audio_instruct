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

The archive is read one of two ways (see `_open_archive`): fully downloaded and
cached via `hub.download_file` (default), or lazily streamed via
`hub.open_archive_stream` so that stopping early (with `clips_per_split`) only
transfers the compressed prefix of the archive.
"""
import contextlib
import glob
import hashlib
import json
import logging
import random
import tarfile
from typing import Any, Iterator

import datasets

from . import hub
from . import prompts as prompts_lib
from .collection import UadCollection
from .internal_dataset import InternalDataset
from .json_config_loader import UniversalJsonConfig
from .row import Row
from .tasks import Task

logger = logging.getLogger(__name__)


SPLIT_NAMES = ("train", "validation", "test")
ALL_SPLITS = "all"


def parse_split(split: str) -> list[str] | None:
    """Parse a split request: `test`, `validation+test`, or `all` (None).

    Raises ValueError for any name other than train, validation or test.
    """
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


def _get_prompt_templates(task: Task, rng: random.Random | None):
    """Select prompt/instruction/output template tuples for a task from PROMPTS_DIR.

    With `rng`, pick one tuple at random with it; without, return them all.
    """
    files = glob.glob(f"{prompts_lib.PROMPTS_DIR}/*.json")
    prompt_files = [prompts_lib.PromptFilepath(filepath=f) for f in files]
    if task not in {pf.task for pf in prompt_files}:
        raise RuntimeError(
            f"No prompt file exists for {task} in {prompts_lib.PROMPTS_DIR}/.")
    task_prompt_files = [pf for pf in prompt_files if pf.accepts_task(task)]
    if not task_prompt_files:
        raise ValueError(f"No prompt files found for task: {task}")
    if len(task_prompt_files) > 1:
        raise ValueError(
            f"Multiple prompt files correspond to task: {task_prompt_files}. Should only be 1.")
    task_prompt_file = task_prompt_files[0]
    if rng is not None:
        return [task_prompt_file.random_template_selection(rng)]
    return task_prompt_file.all_templates


@contextlib.contextmanager
def _open_archive(data_url: str, *, stream: bool, repo_id: str, revision, token):
    """Yield a streaming tar handle for an internal dataset's audio archive.

    stream=False downloads (and caches) the whole `.tar.gz` first; stream=True
    reads it lazily over HTTP so an early break transfers only the prefix consumed.
    Either way the caller gets a sequential `r|gz` tar object.
    """
    if stream:
        fileobj = hub.open_archive_stream(
            data_url, repo_id=repo_id, revision=revision, token=token)
        try:
            with tarfile.open(fileobj=fileobj, mode="r|gz") as archive:
                yield archive
        finally:
            fileobj.close()
    else:
        tar_path = hub.download_file(
            data_url, repo_id=repo_id, revision=revision, token=token)
        with tarfile.open(tar_path, "r|gz") as archive:
            yield archive


def iter_rows(
    collection: UadCollection,
    split: str,
    *,
    repo_id: str,
    revision: str | None,
    token: str | None,
    clips_per_split: int | None = None,
    seed: int = 42,
    stream: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield expanded row dicts for the requested splits, one archive read per internal dataset.

    Rows are grouped by internal dataset in run config order and follow archive
    order within one, so rows of different splits are mixed. With
    `clips_per_split`, each selected split takes only its first that many clips
    in archive order; a clip counts once one of its rows passes the row filter.
    Reading an archive stops once every selected split is satisfied: full, or
    with every clip its metadata lists found. Raises ValueError when no internal
    dataset lists any requested split.
    """
    requested = parse_split(split)
    randomize = collection.is_random_prompt_format_selection()
    selections = [
        (internal_dataset, selected)
        for internal_dataset in collection.internal_datasets
        if (selected := _selected_splits(internal_dataset, requested))
    ]
    if not selections:
        raise ValueError(
            f"No internal dataset in run config {collection.name!r} lists split {split!r}.")

    for internal_dataset, selected in selections:
        # audio_path -> {split: record}, over every selected split.
        clips: dict[str, dict[str, Any]] = {}
        for split_name in selected:
            metadata_path = hub.download_file(
                internal_dataset.split_metadata_path(datasets.Split(split_name)),
                repo_id=repo_id, revision=revision, token=token)
            for audio_path, record in _load_split_metadata(
                    metadata_path, internal_dataset.tasks).items():
                clips.setdefault(audio_path, {})[split_name] = record

        found = {split_name: 0 for split_name in selected}
        # Clips each split lists that the archive hasn't reached yet.
        unread = {split_name: set() for split_name in selected}
        for audio_path, records in clips.items():
            for split_name in records:
                unread[split_name].add(audio_path)

        def satisfied(split_name: str) -> bool:
            return not unread[split_name] or (
                clips_per_split is not None and found[split_name] >= clips_per_split)

        # Read the archive as a sequential stream (archives are multi-GB); with
        # stream=True only the prefix up to the early-stop point is downloaded.
        with _open_archive(
            internal_dataset.data_url, stream=stream,
            repo_id=repo_id, revision=revision, token=token,
        ) as archive:
            for member in archive:
                if all(satisfied(split_name) for split_name in selected):
                    break
                if not member.isfile() or member.name not in clips:
                    continue
                wanted = [
                    (split_name, record)
                    for split_name, record in clips[member.name].items()
                    if not satisfied(split_name)
                ]
                for split_name in clips[member.name]:
                    unread[split_name].discard(member.name)
                if not wanted:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                file_bytes = extracted.read()
                for split_name, record in wanted:
                    kept = False
                    for row in _clip_rows(
                            collection, internal_dataset, split_name, member.name,
                            file_bytes, record, randomize, seed):
                        kept = True
                        yield row
                    if kept:
                        found[split_name] += 1


def _clip_rows(
    collection: UadCollection,
    internal_dataset: InternalDataset,
    split: str,
    audio_path: str,
    file_bytes: bytes,
    record: dict[str, Any],
    randomize: bool,
    seed: int,
) -> Iterator[dict[str, Any]]:
    """Yield one clip's rows for one split that pass the run's row filter."""
    for task in record["tasks"]:
        # One pass per utterance for asr_timestamp_search, one pass otherwise.
        for utterance_index in task.utterance_indices(record):
            rng = _template_rng(
                seed, internal_dataset.name, split, audio_path, task, utterance_index,
            ) if randomize else None
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
                if collection.row_filter.include_row(row):
                    yield row.to_output()


def load_uad_dataset(
    *,
    json_config_path: str,
    split: str,
    repo_id: str = hub.DEFAULT_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
    clips_per_split: int | None = None,
    seed: int = 42,
    stream: bool | None = None,
) -> list[dict[str, Any]]:
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
        stream: Read audio archives lazily over HTTP instead of downloading them
            in full, so an early stop transfers only the prefix consumed. Defaults
            to True when `clips_per_split` is set and False otherwise -- full runs
            prefer the cached download. Streamed reads are not cached, so avoid
            stream=True for large or repeated runs.

    Returns:
        A list of row dicts consumable directly by the Evaluator.
    """
    if clips_per_split is not None and clips_per_split < 1:
        raise ValueError(f"clips_per_split must be a positive integer, got {clips_per_split}.")
    if stream is None:
        stream = clips_per_split is not None

    config_path = _resolve_config_path(
        json_config_path, repo_id=repo_id, revision=revision, token=token)

    # Make the prompt templates available locally and point the library at them.
    prompts_lib.PROMPTS_DIR = hub.download_prompts_dir(
        repo_id=repo_id, revision=revision, token=token)

    collection = UniversalJsonConfig(filepath=config_path).toCollection()
    return list(iter_rows(
        collection, split,
        repo_id=repo_id, revision=revision, token=token,
        clips_per_split=clips_per_split, seed=seed, stream=stream))


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
