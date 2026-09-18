"""Fetch UAD dataset assets from the (private) HuggingFace Hub repo.

This replaces the loading script's `dl_manager` downloads and the
`_ensure_hub_resources()` shim. Everything here uses `huggingface_hub` to pull
plain files -- there is no `trust_remote_code` and no executable dataset script
on the Hub. Assets fetched: per-dataset audio archives (`data/<name>/<name>.tar.gz`),
per-split metadata JSONs, prompt templates (`prompts/*.json`), named configs
(`universal_audio_dataset_configs/*.json`), and smoke archives with their manifest
(`smoke/`). It also reads files' versions (an LFS file's sha256) and the current
commit of a branch.

A file that isn't on the Hub raises `EntryNotFoundError`. So does one that can't
be fetched because the Hub is unreachable and it isn't cached: that raises the
subclass `LocalEntryNotFoundError`. Both are re-exported here so callers needn't
import from `huggingface_hub`.
"""
import os

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

# What a Hub request raises when it can't be answered: an HTTP error (401, 404,
# 5xx) or an offline mode is an OSError; a connection that fails below HTTP is an
# httpx error in huggingface_hub 1.x.
try:
    from httpx import HTTPError as _TransportError
except ImportError:  # huggingface_hub 0.x uses requests, whose errors are OSErrors.
    _TransportError = OSError
REQUEST_ERRORS: tuple[type[Exception], ...] = (OSError, _TransportError)

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


def _paths_info(rels: list[str], *, repo_id: str, revision: str | None,
                token: str | None) -> list:
    """The Hub's file info for each of `rels` that's there, from one metadata request."""
    return HfApi(token=token).get_paths_info(
        repo_id, rels, revision=revision, repo_type="dataset")


def file_versions(
    paths_or_urls: list[str],
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, str]:
    """Return repo path -> version at `revision` for each path, in one metadata request.

    An LFS file's version is its sha256; any other file's is its git blob id.
    Either changes whenever the file's content does. Paths that aren't there are
    left out.
    """
    infos = _paths_info([to_repo_path(p) for p in paths_or_urls],
                        repo_id=repo_id, revision=revision, token=token)
    return {info.path: info.lfs.sha256 if info.lfs is not None else info.blob_id
            for info in infos}


def file_sha256(
    path_or_url: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
) -> str:
    """Return one repo file's LFS sha256 at `revision`.

    Raises `EntryNotFoundError` when the file isn't there, and ValueError when it
    isn't stored with LFS (only LFS files have a sha256 on the Hub).
    """
    rel = to_repo_path(path_or_url)
    infos = _paths_info([rel], repo_id=repo_id, revision=revision, token=token)
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
