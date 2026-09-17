"""Expand the UAD dataset into evaluation rows without a HF loading script.

Public entry point: `load_uad_dataset(...)`. It produces the same core fields as
the old `load_dataset("AudioInstruct/Universal-Audio-Understanding", ...,
trust_remote_code=True)` -- audio, system_instruction, prompt, output, task,
split, originating_dataset, plus the per-task metadata fields -- so the
downstream Evaluator is unchanged. Rows also carry every other field of the
clip's metadata record, and a `tasks` list of `Task` values. asr_timestamp_search
rows also carry `utterance_index`, the position in `transcriptions` of the
utterance they render.

Pipeline per selected internal dataset + split:
  1. obtain the audio archive and the split metadata JSON from the Hub,
  2. stream the tar archive, looking up each member's metadata by audio_path,
  3. for every (task, utterance, prompt-template) combination, build a Sample and
     emit its row. Only asr_timestamp_search renders a clip once per utterance
     (see `Task.utterance_indices`); other tasks render it once.

The archive is read one of two ways (see `_open_archive`): fully downloaded and
cached via `hub.download_file` (default), or lazily streamed via
`hub.open_archive_stream` so that stopping early (with `max_samples`) only
transfers the compressed prefix of the archive.
"""
import contextlib
import glob
import json
import tarfile
from typing import Any, Iterator

import datasets

from . import hub
from . import prompts as prompts_lib
from .collection import UadCollection
from .json_config_loader import UniversalJsonConfig
from .sample import Sample
from .tasks import Task


def _load_split_metadata(metadata_path: str, tasks: list[Task], split) -> dict[str, Any]:
    """audio_path -> record (with tasks attached); mirrors the old _get_split_audio_metadata."""
    merged: dict[str, Any] = {"split": str(split)}
    with open(metadata_path, encoding="utf-8") as f:
        for record in json.load(f):
            merged[record["audio_path"]] = {**record, "tasks": tasks}
    return merged


def _get_prompt_templates(task: Task, randomize: bool):
    """Select prompt/instruction/output template tuples for a task from PROMPTS_DIR."""
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
    if randomize:
        return [task_prompt_file.random_template_selection()]
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


def iter_samples(
    collection: UadCollection,
    split,
    *,
    repo_id: str,
    revision: str | None,
    token: str | None,
    max_samples: int | None = None,
    stream: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield expanded row dicts for one split across the collection's datasets."""
    count = 0
    randomize = collection.is_random_prompt_format_selection()
    for internal_dataset in collection.get_datasets_with_splits(split):
        metadata_path = hub.download_file(
            internal_dataset.split_metadata_path(split),
            repo_id=repo_id, revision=revision, token=token)
        metadata = _load_split_metadata(metadata_path, internal_dataset.tasks, split)

        # Read the archive as a sequential stream (archives are multi-GB); with
        # stream=True only the prefix up to the early-stop point is downloaded.
        with _open_archive(
            internal_dataset.data_url, stream=stream,
            repo_id=repo_id, revision=revision, token=token,
        ) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                sample_path = member.name
                if sample_path not in metadata:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                file_bytes = extracted.read()
                record = metadata[sample_path]
                for task in record["tasks"]:
                    # One pass per utterance for asr_timestamp_search, one pass otherwise.
                    # With randomize on, each utterance gets its own template pick, so
                    # a seeded pick must include the utterance index in its seed.
                    for utterance_index in task.utterance_indices(record):
                        for si_t, p_t, o_t in _get_prompt_templates(task, randomize):
                            sample = Sample(
                                audio_path=sample_path,
                                dataset_name=internal_dataset.name,
                                split=metadata["split"],
                                task=task,
                                audio_data=file_bytes,
                                metadata=record,
                                system_instruction_template=si_t,
                                prompt_template=p_t,
                                output_template=o_t,
                                utterance_index=utterance_index,
                            )
                            if collection.sample_filter.include_sample(sample):
                                yield sample.to_output()
                                count += 1
                                if max_samples is not None and count >= max_samples:
                                    return


def load_uad_dataset(
    *,
    json_config_path: str,
    split: str,
    repo_id: str = hub.DEFAULT_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
    max_samples: int | None = None,
    stream: bool | None = None,
) -> list[dict[str, Any]]:
    """Load and expand the UAD dataset for a given config + split.

    Args:
        json_config_path: Either a local path to a UAD JSON config, or the name of
            a config hosted in the repo's `universal_audio_dataset_configs/` folder.
        split: Split name ("train" | "test" | "validation").
        repo_id: HF Hub dataset repo id (private).
        revision: Optional Hub revision/commit to pin for reproducibility.
        token: HF access token (required for the private repo).
        max_samples: Optional cap on the number of rows (stops iterating early).
        stream: Read audio archives lazily over HTTP instead of downloading them
            in full, so an early stop transfers only the prefix consumed. Defaults
            to True when `max_samples` is set (the debugging/smoke-test case) and
            False otherwise -- full runs prefer the cached download. Streamed reads
            are not cached, so avoid stream=True for large or repeated runs.

    Returns:
        A list of row dicts consumable directly by the Evaluator.
    """
    if stream is None:
        stream = max_samples is not None

    config_path = _resolve_config_path(
        json_config_path, repo_id=repo_id, revision=revision, token=token)

    # Make the prompt templates available locally and point the library at them.
    prompts_lib.PROMPTS_DIR = hub.download_prompts_dir(
        repo_id=repo_id, revision=revision, token=token)

    collection = UniversalJsonConfig(filepath=config_path).toCollection()
    split_key = datasets.Split(split) if isinstance(split, str) else split
    return list(iter_samples(
        collection, split_key,
        repo_id=repo_id, revision=revision, token=token,
        max_samples=max_samples, stream=stream))


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
