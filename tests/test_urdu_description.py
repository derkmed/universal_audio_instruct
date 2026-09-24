"""Offline test: the URDU registry description names URDU's real emotion classes.

The class list is copied from the Hub's `data/URDU/URDU_train.json`. Its 400
rows have 100 each of `category` Angry, Happy, Neutral and Sad, and every row's
`categories` is "Angry, Happy, Neutral, Sad".

Runnable directly (`python tests/test_urdu_description.py`) or under pytest.
Only requires `datasets`.
"""
from uad_data.internal_datasets import DATASETS_DIRECTORY

URDU_CLASSES = "Angry, Happy, Neutral, and Sad"


def test_description_lists_the_metadata_classes() -> None:
    description = DATASETS_DIRECTORY["URDU"].description
    assert f"four basic emotions: {URDU_CLASSES}." in description, description
    assert "and Emotion" not in description, description
    print("PASS: the URDU description lists Angry, Happy, Neutral and Sad.")


if __name__ == "__main__":
    test_description_lists_the_metadata_classes()
