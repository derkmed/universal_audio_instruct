"""Fetch UAD dataset assets from the (private) HuggingFace Hub repo.

This replaces the loading script's `dl_manager` downloads and the
`_ensure_hub_resources()` shim. Everything here uses `huggingface_hub` to pull
plain files -- there is no `trust_remote_code` and no executable dataset script
on the Hub. Assets fetched: per-dataset audio archives (`data/<name>/<name>.tar.gz`),
per-split metadata JSONs, prompt templates (`prompts/*.json`), named configs
(`universal_audio_dataset_configs/*.json`), and smoke archives with their manifest
(`smoke/`). It also reads a file's LFS sha256 and the current commit of a branch.

A file that isn't on the Hub raises `EntryNotFoundError`, which is re-exported
here so callers needn't import from `huggingface_hub`.
"""
import os

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

DEFAULT_REPO_ID = "AudioInstruct/Universal-Audio-Understanding"
_RESOLVE_MARKER = "/resolve/"


def to_repo_path(path_or_url: str) -> str:
    """Normalise a data_url to a repo-relative path.

    `InternalDataset` builds LFS-style absolute URLs of the form
    `https://huggingface.co/datasets/<repo>/resolve/<rev>/<path>`. `hf_hub_download`
    wants just `<path>`, so strip everything up to and including the revision.
    Plain relative paths are returned unchanged (minus any leading slash).
    """
    # Hub paths always use forward slashes; guard against any OS-native separators
    # that may have crept in via os.path.join upstream.
    path_or_url = path_or_url.replace("\\", "/")
    if _RESOLVE_MARKER in path_or_url:
        after_resolve = path_or_url.split(_RESOLVE_MARKER, 1)[1]  # "<rev>/<path>"
        return after_resolve.split("/", 1)[1]                     # "<path>"
    return path_or_url.lstrip("/")


def download_file(
    path_or_url: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
) -> str:
    """Download a single file from the dataset repo, returning its local path."""
    return hf_hub_download(
        repo_id=repo_id,
        filename=to_repo_path(path_or_url),
        repo_type="dataset",
        revision=revision,
        token=token,
    )


def open_archive_stream(
    path_or_url: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
):
    """Open a repo file as a lazily-read, range-request-backed binary stream.

    Unlike `download_file` (which fetches the whole file into the local HF cache),
    this returns an `HfFileSystem` file object that only transfers the bytes that
    are actually read. Feeding it to `tarfile.open(fileobj=..., mode="r|gz")` and
    stopping early therefore downloads just the compressed prefix of the archive --
    useful with `clips_per_split`. Trade-off: nothing is cached, so full/repeat reads
    are better served by `download_file`.
    """
    from huggingface_hub import HfFileSystem

    rel = to_repo_path(path_or_url)
    hf_path = (
        f"datasets/{repo_id}@{revision}/{rel}" if revision
        else f"datasets/{repo_id}/{rel}"
    )
    return HfFileSystem(token=token).open(hf_path, "rb")


def download_prompts_dir(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
) -> str:
    """Snapshot the repo's `prompts/` folder and return the local prompts directory."""
    local_repo = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
        allow_patterns="prompts/*",
    )
    return os.path.join(local_repo, "prompts")


def file_sha256(
    path_or_url: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
) -> str:
    """Return a repo file's LFS sha256 at `revision`, with one small metadata request.

    Raises `EntryNotFoundError` when the file isn't there, and ValueError when it
    isn't stored with LFS (only LFS files have a sha256 on the Hub).
    """
    rel = to_repo_path(path_or_url)
    infos = HfApi(token=token).get_paths_info(
        repo_id, [rel], revision=revision, repo_type="dataset")
    if not infos:
        raise EntryNotFoundError(f"{rel} is not in {repo_id} at {revision or 'main'}.")
    lfs = getattr(infos[0], "lfs", None)
    if lfs is None:
        raise ValueError(f"{rel} is not an LFS file, so the Hub records no sha256 for it.")
    return lfs.sha256


def current_commit(
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
    token: str | None = None,
) -> str:
    """Return the commit sha that `revision` (a branch, by default `main`) points at now."""
    return HfApi(token=token).dataset_info(repo_id, revision=revision).sha
