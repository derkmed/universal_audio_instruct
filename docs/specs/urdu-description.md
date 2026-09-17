# URDU description names its real emotion classes

Issue: #30

## Problem

The `URDU` entry in `uad_data/internal_datasets.py` says its 400 clips cover
"four basic emotions: Angry, Happy, Neutral, and Emotion". "Emotion" is not a
class. The Hub's `data/URDU/URDU_train.json` has 400 rows, 100 per `category`, and
the `category` values are `Angry`, `Happy`, `Neutral`, `Sad`. Every row's
`categories` field is `"Angry, Happy, Neutral, Sad"`.

## Seams

- `DATASETS_DIRECTORY['URDU'].description`: the registry entry's text. It is
  tested directly, with no Hub access.

## Behaviour

- The description lists the four classes as `Angry, Happy, Neutral, and Sad`, in
  the order the metadata's `categories` field uses.
- The rest of the text stays as it is: 400 clips, 38 speakers, and the
  clips-vs-utterances comment.

## Acceptance

- `tests/test_urdu_description.py` passes. It checks that the description lists
  exactly the Hub's four classes and no longer lists "Emotion".
- Every other test file still passes.

## Out of scope

- Other dataset descriptions. The test does not fetch from the Hub, so the class
  list it checks against is copied from the Hub file.
