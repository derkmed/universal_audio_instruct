"""Which splits still want a clip, as an internal dataset's archive is read in order.

The loader and the smoke-archive builder both read an archive front to back and
share this bookkeeping, so a smoke archive holds exactly the clips a run would
take from the full archive: for each split, its first clips in archive order.
"""
import tarfile
from dataclasses import dataclass
from typing import Iterable, Iterator


class ClipQuota:
    """Per-split clip counts against a cap, over one sequential archive read.

    `split_paths` maps each split to the audio paths its metadata lists. A split
    is satisfied once it has `clips_per_split` clips, or once every clip it
    lists has been read (with no cap, only the latter).
    """

    def __init__(self, split_paths: dict[str, Iterable[str]], clips_per_split: int | None) -> None:
        self._cap = clips_per_split
        self._unread = {split: set(paths) for split, paths in split_paths.items()}
        self._found = dict.fromkeys(split_paths, 0)
        # audio path -> the splits that list it, in `split_paths` order.
        self._splits_of: dict[str, list[str]] = {}
        for split, paths in self._unread.items():
            for path in paths:
                self._splits_of.setdefault(path, []).append(split)

    def claim(self, audio_path: str) -> list[str]:
        """Mark an archive member read; return the splits that list it and still want clips.

        The caller counts the clip toward each of them with `count`.
        """
        splits = self._splits_of.get(audio_path, [])
        wanting = [split for split in splits if not self.satisfied(split)]
        for split in splits:
            self._unread[split].discard(audio_path)
        return wanting

    def count(self, split: str) -> None:
        """Count one more clip toward `split`."""
        self._found[split] += 1

    def found(self, split: str) -> int:
        """Clips counted toward `split` so far."""
        return self._found[split]

    def satisfied(self, split: str) -> bool:
        """Whether `split` wants no more clips: full, or with every listed clip read."""
        full = self._cap is not None and self._found[split] >= self._cap
        return full or not self._unread[split]

    def all_satisfied(self) -> bool:
        """Whether no split wants any more clips, so reading can stop."""
        return all(self.satisfied(split) for split in self._found)


@dataclass
class WantedClip:
    """An archive member that some splits still want, with its bytes read."""
    member: tarfile.TarInfo
    data: bytes
    splits: list[str]

    @property
    def audio_path(self) -> str:
        return self.member.name


def wanted_clips(archive: tarfile.TarFile, quota: ClipQuota) -> Iterator[WantedClip]:
    """Yield, in archive order, each member that `quota` still wants, until it wants none.

    Reading stops as soon as the quota is satisfied, before the next member's
    header, so an early stop reads as little of the archive as it can. The
    caller counts each clip toward its splits with `quota.count`.
    """
    if quota.all_satisfied():
        return
    for member in archive:
        if not member.isfile():
            continue
        splits = quota.claim(member.name)
        if not splits:
            continue
        yield WantedClip(member, archive.extractfile(member).read(), splits)
        if quota.all_satisfied():
            return
