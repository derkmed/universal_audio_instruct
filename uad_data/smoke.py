"""Where smoke archives live on the Hub, and the manifest that describes them.

Each internal dataset's smoke archive is `smoke/<name>.tar.gz`: the first few
clips of each of its registered splits, in archive order, copied from its full
archive by `python -m uad_data.build_smoke_archives`. `smoke/manifest.json`
records, per internal dataset, how each smoke archive was built, so the loader
can tell whether one is usable for a run, and still fresh: built from the full
archive and metadata JSONs now on the Hub (see `loader._archive_sources`).
"""
import dataclasses
import json
from dataclasses import dataclass

SMOKE_DIR = "smoke"
MANIFEST_PATH = f"{SMOKE_DIR}/manifest.json"


def smoke_archive_path(dataset: str) -> str:
    """The repo path of an internal dataset's smoke archive."""
    return f"{SMOKE_DIR}/{dataset}.tar.gz"


@dataclass
class SmokeEntry:
    """How one internal dataset's smoke archive was built.

    `clips_per_split` is N, the clips per split the build asked for, and `clips`
    the clips each registered split got. `source_sha256` is the LFS sha256 of the
    full archive it was built from, and `metadata_versions` maps each registered
    split to the Hub version of the metadata JSON its clips were picked from (see
    `hub.file_versions`), both at Hub commit `revision`. `error` is the read error
    that stopped the build, if one did.
    """
    clips_per_split: int
    clips: dict[str, int]
    source_sha256: str
    metadata_versions: dict[str, str]
    revision: str
    error: str | None = None


def read_manifest(path: str) -> dict[str, SmokeEntry]:
    """Internal dataset name -> its smoke archive's entry, from a local manifest file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {name: SmokeEntry(**entry) for name, entry in data["datasets"].items()}


def write_manifest(entries: dict[str, SmokeEntry], path: str) -> None:
    """Write the manifest for `entries`, keyed by internal dataset name."""
    data = {"datasets": {name: dataclasses.asdict(entry) for name, entry in entries.items()}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
