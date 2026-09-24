"""Options that pick out which slice of the dataset a run builds.

Both harnesses' config classes and `loader.load_uad_dataset` need the same
answers to "how many clips per split?" and "which split, when none was asked
for?". These are concerns of what the loader builds, so they live beside it
rather than in `eval/` or `train/` -- neither of which the other may import.
"""
from typing import Optional


# Both command lines and the notebook default the seed here, so there is one 42.
DEFAULT_SEED = 42


def validate_clips_per_split(clips_per_split: Optional[int]) -> None:
    """Reject a cap that asks for no clips at all.

    Args:
        clips_per_split: The per-split cap, or None for every clip.

    Raises:
        ValueError: The cap is set and below 1.
    """
    if clips_per_split is not None and clips_per_split < 1:
        raise ValueError(
            f"clips_per_split must be a positive integer, got {clips_per_split}.")


def resolve_dataset_split(
    dataset_split: Optional[str], *, clips_per_split: Optional[int], uncapped_default: str,
) -> str:
    """Fill in an unset or blank `dataset_split`.

    Both config classes resolve here, so a command line and the notebook read an
    unset split the same way: a smoke run wants every registered split, and a
    regular run wants the one split its harness is for.

    Args:
        dataset_split: The split asked for, or None/"" for none asked.
        clips_per_split: The per-split cap; set means this is a smoke run.
        uncapped_default: The split an uncapped run of this harness wants.

    Returns:
        The split expression to hand the loader.
    """
    # A blank string is the notebook's empty form field: no split asked for.
    if dataset_split:
        return dataset_split
    return "all" if clips_per_split is not None else uncapped_default
