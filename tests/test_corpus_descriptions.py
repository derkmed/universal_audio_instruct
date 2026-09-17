"""Offline test: internal-dataset descriptions keep the corpora's own words.

Descriptions are corpus text, so CONTEXT.md's glossary does not govern them
(see the comment on `DATASETS`). MELD and URDU call a whole clip an
"utterance", and their descriptions say so.

Runnable directly (`python tests/test_corpus_descriptions.py`) or under pytest.
Only requires `datasets`.
"""
import os
import sys

# Make the package importable when run directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uad_data.internal_datasets import DATASETS_DIRECTORY  # noqa: E402


def test_meld_description_uses_the_corpus_wording() -> None:
    description = DATASETS_DIRECTORY["MELD"].description
    assert "13000 utterances from Friends TV series." in description, description
    assert "clips" not in description, description
    print("PASS: the MELD description says utterances.")


def test_urdu_description_uses_the_corpus_wording() -> None:
    description = DATASETS_DIRECTORY["URDU"].description
    assert "contains emotional utterances of Urdu speech" in description, description
    assert "It contains 400 utterances of four basic emotions" in description, description
    assert "clips" not in description, description
    print("PASS: the URDU description says utterances.")


if __name__ == "__main__":
    test_meld_description_uses_the_corpus_wording()
    test_urdu_description_uses_the_corpus_wording()
