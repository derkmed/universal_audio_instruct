"""Build a smoke archive for every internal dataset in a run config.

Usage:
```
python -m uad_data.build_smoke_archives --clips-per-split 10
python -m uad_data.build_smoke_archives --clips-per-split 10 --source-dir <local copy of UAD>
```

Each smoke archive holds the first N (`--clips-per-split`) clips of each
registered split of one internal dataset, in archive order, with tar entries and
bytes copied as-is from its full archive. The full archive is read once, front to
back, and reading stops once every split has N clips, or when the archive ends
or fails. A read error is recorded in the manifest rather than raised, so a
truncated archive still gives the clips before the cut.

The command writes `<output-dir>/<name>.tar.gz` for each internal dataset, plus
`<output-dir>/manifest.json` (see `uad_data.smoke`). Uploading them to the Hub's
`smoke/` folder is a separate step.

Source archives come from the Hub by default, streamed at the current commit of
`main`. With `--source-dir`, they're read from a local copy laid out like the
Hub repo, and the command refuses to build unless each local archive's sha256
matches the Hub's current one. Metadata always comes from the Hub at that commit,
and the manifest records each split's metadata version, so the loader can tell
when a re-split makes a smoke archive stale.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import tarfile
import zlib
from dataclasses import dataclass
from typing import BinaryIO, Iterator

from . import hub, smoke
from .clip_quota import ClipQuota, wanted_clips
from .internal_dataset import InternalDataset
from .internal_datasets import DATASETS_DIRECTORY
from .json_config_loader import UniversalJsonConfig
from .loader import _resolve_config_path, describe_error

# What stops a build and gets recorded: an archive that can't be decompressed or
# read further. A network error is not recorded -- it would outlive a transient
# failure in the manifest -- so it stops the command instead.
READ_ERRORS = (tarfile.TarError, EOFError, zlib.error)


@dataclass
class SmokeBuild:
    """The clips one smoke archive got per split, and the read error that stopped it, if any."""
    clips: dict[str, int]
    error: str | None


@dataclass
class StaleSource:
    """A local source archive the command won't build from, and why."""
    dataset: str
    reason: str


def write_smoke_archive(
    source: BinaryIO,
    split_paths: dict[str, list[str]],
    clips_per_split: int,
    destination: BinaryIO,
) -> SmokeBuild:
    """Copy the first `clips_per_split` clips of each split from `source` to `destination`.

    `source` is a full `.tar.gz` archive and `split_paths` maps each split to
    the audio paths its metadata lists. Clips are taken in archive order, as the
    loader takes them, and a clip listed in two splits is copied once and counts
    toward both. A read error stops the copy and is returned, with the clips
    copied before it.
    """
    quota = ClipQuota(split_paths, clips_per_split)
    error = None
    with tarfile.open(fileobj=destination, mode="w:gz") as smoke_archive:
        try:
            _copy_wanted_clips(source, quota, smoke_archive)
        except READ_ERRORS as read_error:
            error = describe_error(read_error)
    return SmokeBuild(clips={split: quota.found(split) for split in split_paths}, error=error)


def _copy_wanted_clips(source: BinaryIO, quota: ClipQuota, smoke_archive: tarfile.TarFile) -> None:
    """Copy each clip `quota` wants, entry and bytes as-is, counting it toward its splits."""
    with tarfile.open(fileobj=source, mode="r|gz") as archive:
        for clip in wanted_clips(archive, quota):
            smoke_archive.addfile(clip.member, io.BytesIO(clip.data))
            for split in clip.splits:
                quota.count(split)


def stale_sources(
    source_dir: str,
    internal_datasets: list[InternalDataset],
    *,
    revision: str,
    repo_id: str,
    token: str | None,
) -> list[StaleSource]:
    """The internal datasets whose local archive is missing or differs from the Hub's at `revision`."""
    stale = []
    for internal_dataset in internal_datasets:
        reason = _source_problem(
            source_dir, internal_dataset, revision=revision, repo_id=repo_id, token=token)
        if reason is not None:
            stale.append(StaleSource(internal_dataset.name, reason))
    return stale


def _source_problem(
    source_dir: str,
    internal_dataset: InternalDataset,
    *,
    revision: str,
    repo_id: str,
    token: str | None,
) -> str | None:
    """Why one local archive can't be built from, or None when it matches the Hub."""
    path = _local_archive(source_dir, internal_dataset)
    if not os.path.isfile(path):
        return f"{path} is missing"
    local = _sha256_of(path)
    expected = hub.file_sha256(
        internal_dataset.data_url, repo_id=repo_id, revision=revision, token=token)
    if local != expected:
        return f"{path} has sha256 {local}, but the Hub's archive at {revision} has {expected}"
    return None


def _local_archive(source_dir: str, internal_dataset: InternalDataset) -> str:
    """Where a local copy laid out like the Hub repo keeps an internal dataset's archive."""
    return os.path.join(source_dir, *hub.to_repo_path(internal_dataset.data_url).split("/"))


def _sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class BuildOptions:
    """What every internal dataset's build shares."""
    clips_per_split: int
    output_dir: str
    source_dir: str | None
    repo_id: str
    revision: str
    token: str | None


def build_one(internal_dataset: InternalDataset, options: BuildOptions) -> smoke.SmokeEntry:
    """Build one internal dataset's smoke archive into the output directory; return its entry."""
    split_paths = _split_paths(internal_dataset, options)
    source_sha256 = hub.file_sha256(
        internal_dataset.data_url, repo_id=options.repo_id,
        revision=options.revision, token=options.token)
    metadata_versions = _metadata_versions(internal_dataset, options)
    path = os.path.join(options.output_dir, f"{internal_dataset.name}.tar.gz")
    with _open_source(internal_dataset, options) as source, open(path, "wb") as destination:
        build = write_smoke_archive(source, split_paths, options.clips_per_split, destination)
    return smoke.SmokeEntry(
        clips_per_split=options.clips_per_split, clips=build.clips,
        source_sha256=source_sha256, metadata_versions=metadata_versions,
        revision=options.revision, error=build.error)


def _split_paths(internal_dataset: InternalDataset, options: BuildOptions) -> dict[str, list[str]]:
    """Each registered split's audio paths, from its metadata on the Hub at the build's commit."""
    paths = {}
    for split in internal_dataset.get_splits():
        metadata_path = hub.download_file(
            internal_dataset.split_metadata_path(split), repo_id=options.repo_id,
            revision=options.revision, token=options.token)
        with open(metadata_path, encoding="utf-8") as f:
            paths[str(split)] = [record["audio_path"] for record in json.load(f)]
    return paths


def _metadata_versions(internal_dataset: InternalDataset, options: BuildOptions) -> dict[str, str]:
    """Each registered split's metadata JSON version on the Hub at the build's commit.

    Raises `EntryNotFoundError` when a split's metadata has no version there.
    """
    paths = {str(split): hub.to_repo_path(internal_dataset.split_metadata_path(split))
             for split in internal_dataset.get_splits()}
    versions = hub.file_versions(
        list(paths.values()), repo_id=options.repo_id,
        revision=options.revision, token=options.token)
    missing = [path for path in paths.values() if path not in versions]
    if missing:
        raise hub.EntryNotFoundError(f"No version on the Hub at {options.revision} for {missing}.")
    return {split: versions[path] for split, path in paths.items()}


@contextlib.contextmanager
def _open_source(internal_dataset: InternalDataset, options: BuildOptions) -> Iterator[BinaryIO]:
    """Open the full archive: the local copy, or the Hub's, streamed at the build's commit."""
    if options.source_dir is not None:
        source = open(_local_archive(options.source_dir, internal_dataset), "rb")
    else:
        source = hub.open_archive_stream(
            internal_dataset.data_url, repo_id=options.repo_id,
            revision=options.revision, token=options.token)
    try:
        yield source
    finally:
        source.close()


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uad_data.build_smoke_archives",
        description="Build a smoke archive, and a manifest, for every internal dataset "
                    "in a run config.")
    parser.add_argument("--clips-per-split", type=_positive_int, required=True,
                        help="N: clips per registered split each smoke archive holds.")
    parser.add_argument("--json-config", default="complete.json",
                        help="Run config naming the internal datasets (default: complete.json).")
    parser.add_argument("--source-dir", default=None,
                        help="Read full archives from this local copy of the Hub repo "
                             "instead of streaming them from the Hub.")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "smoke"),
                        help="Where to write the smoke archives and manifest.json "
                             "(default: outputs/smoke).")
    parser.add_argument("--repo-id", default=hub.DEFAULT_REPO_ID)
    parser.add_argument("--token", default=None,
                        help="HF token (default: the saved Hugging Face login).")
    return parser


def _internal_datasets(json_config: str, *, repo_id: str, revision: str,
                       token: str | None) -> list[InternalDataset]:
    """The registry entries of the run config's internal datasets, in its order."""
    config_path = _resolve_config_path(json_config, repo_id=repo_id, revision=revision, token=token)
    config = UniversalJsonConfig(filepath=config_path)
    return [DATASETS_DIRECTORY[d.name] for d in config.internal_datasets]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    revision = hub.current_commit(repo_id=args.repo_id, token=args.token)
    internal_datasets = _internal_datasets(
        args.json_config, repo_id=args.repo_id, revision=revision, token=args.token)
    if args.source_dir is not None:
        stale = stale_sources(args.source_dir, internal_datasets,
                              revision=revision, repo_id=args.repo_id, token=args.token)
        if stale:
            raise SystemExit("Refusing to build from stale source archives:\n" + "\n".join(
                f"  {source.dataset}: {source.reason}" for source in stale))

    options = BuildOptions(
        clips_per_split=args.clips_per_split, output_dir=args.output_dir,
        source_dir=args.source_dir, repo_id=args.repo_id, revision=revision, token=args.token)
    os.makedirs(options.output_dir, exist_ok=True)
    entries = {}
    for internal_dataset in internal_datasets:
        entries[internal_dataset.name] = entry = build_one(internal_dataset, options)
        error = f"; stopped by {entry.error}" if entry.error else ""
        print(f"{internal_dataset.name}: {entry.clips}{error}")
    smoke.write_manifest(entries, os.path.join(options.output_dir, "manifest.json"))
    print(f"Wrote {len(entries)} smoke archives and manifest.json to {options.output_dir} "
          f"(Hub commit {revision}).")


if __name__ == "__main__":
    main()
