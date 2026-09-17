"""Audio-understanding task definitions.

`Task` enumerates every task the dataset supports and, via `Task.features`, the
metadata column(s) each task expects. `Task.render_context` reads those fields
from a record for `sample.Sample` to render the prompt / instruction / output
templates with, so the keys here must match the fields present in the per-split
metadata JSONs and the placeholders used in `prompts/*.json`.
"""
import datasets
import enum
from typing import Any


class Task(enum.Enum):
    """Enum class for Audio Understanding tasks."""
    ASR = "asr"
    ASR_TIMESTAMP_SEARCH = "asr_timestamp_search"
    CLASSIFICATION = "classification"
    # Classification tasks should be suffixed with 'classification'.
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CAPTION = "caption"
    COMMONSENSE = "commonsense"
    COMMONSENSE_HARD = "commonsense_hard"
    QA = "qa"
    ENGLISH_TRANSLATION = "english_translation"
    INTONATION_DETECTION = "intonation_detection"
    INTENT_DETECTION = "intent_detection"
    INTENT_DETECTION_NL = "intent_detection_nl"
    ACTION_CLASSIFICATION = "action_classification"

    @property
    def features(self) -> dict[str, Any]:
        """Maps each task to the corresponding feature column names.

        Each task has an expected feature output. This is linked here.
        """
        if self == Task.CLASSIFICATION:
            return {
                # Ground truth category for this sample.
                'category': datasets.Value('string'),
                # Comma-delimited string of all available categories.
                'categories': datasets.Value('string'),
            }
        elif self == Task.ASR_TIMESTAMP_SEARCH:
            return {
                # List of utterances; each one renders as its own rows.
                'transcriptions': [{
                    "start_time": datasets.Value("float"),
                    "end_time": datasets.Value("float"),
                    "transcription": datasets.Value("string")
                }]
            }
        elif self == Task.ASR:
            return {'transcription': datasets.Value('string')}
        elif self == Task.SENTIMENT_ANALYSIS:
            return {'Sentiment': datasets.Value('string')}
        elif self == Task.CAPTION:
            return {'caption': datasets.Value('string')}
        elif self == Task.COMMONSENSE:
            return {
                'commonsense_choices': datasets.Value('string'),
                'commonsense_answer': datasets.Value('string')
            }
        elif self == Task.COMMONSENSE_HARD:
            return {
                'hard_commonsense_choices': datasets.Value('string'),
                'hard_commonsense_answer': datasets.Value('string')
            }
        elif self == Task.QA:
            return {'question': datasets.Value('string'), 'answer': datasets.Value('string')}
        elif self == Task.ENGLISH_TRANSLATION:
            return {"english_translation": datasets.Value("string")}
        elif self == Task.INTENT_DETECTION_NL:
            return {"intent_nl": datasets.Value("string")}
        elif self == Task.INTENT_DETECTION:
            return {"intent": datasets.Value("string")}
        elif self == Task.ACTION_CLASSIFICATION:
            return {"action": datasets.Value("string")}
        elif self == Task.INTONATION_DETECTION:
            return {"category": datasets.Value("string")}
        else:
            raise NotImplementedError(
                f'{self.value} prompt handling not yet implemented.')

    def utterance_indices(self, record: dict[str, Any]) -> list[int | None]:
        """Which utterances of a clip's metadata record render as separate rows.

        asr_timestamp_search renders each utterance in the record's
        `transcriptions` list as its own rows, so this returns their indices.
        Every other task renders the whole record once: `[None]`. So does a
        `transcriptions` that is missing, empty or not a list. That gives the clip
        one pass, and a pass is one row per prompt template (a single row when
        templates are picked at random). Each of those rows fails in
        `render_context`.

        Never raises: bad metadata is reported when a row renders, after the
        sample filter has decided whether that row is wanted at all.
        """
        if self == Task.ASR_TIMESTAMP_SEARCH:
            utterances = record.get('transcriptions')
            if isinstance(utterances, list) and utterances:
                return list(range(len(utterances)))
        return [None]

    def render_context(
            self, record: dict[str, Any], utterance_index: int | None = None) -> dict[str, Any]:
        """The values one row's templates render with.

        Most tasks read the record's `features` fields. asr_timestamp_search reads
        the fields of the utterance at `utterance_index` instead, so a record-level
        key of the same name (libricss_subseg's `start_time`) never leaks into the
        template. A missing field raises KeyError naming the clip, rather than
        rendering as a blank.

        `utterance_index` must be an int (not a bool) in
        `0..len(transcriptions) - 1` for asr_timestamp_search, and `None` for every
        other task, which has no utterances to pick from. Anything else raises
        ValueError naming the row, rather than rendering another utterance
        (a negative index) or a row whose `utterance_index` means nothing.
        """
        where = f'{self.value} row for {record.get("audio_path")!r}'
        source, keys = record, self.features.keys()
        if self == Task.ASR_TIMESTAMP_SEARCH:
            utterances = record.get('transcriptions')
            if not isinstance(utterances, list) or not utterances:
                raise ValueError(
                    f'{where}: `transcriptions` must be a non-empty list of utterances, '
                    f'got {utterances!r:.80}')
            if utterance_index is None:
                raise ValueError(f'{where}: pass the index of the utterance to render.')
            where += f' (utterance {utterance_index!r})'
            if (not isinstance(utterance_index, int) or isinstance(utterance_index, bool)
                    or not 0 <= utterance_index < len(utterances)):
                raise ValueError(
                    f'{where}: the clip has {len(utterances)} utterance(s), '
                    f'so the index must be an int in 0..{len(utterances) - 1}.')
            source = utterances[utterance_index]
            keys = self.features['transcriptions'][0].keys()
            if not isinstance(source, dict):
                raise ValueError(f'{where}: expected an object, got {source!r:.80}')
        elif utterance_index is not None:
            raise ValueError(
                f'{where}: only asr_timestamp_search rows pick an utterance, '
                f'got utterance_index={utterance_index!r}')
        missing = [k for k in keys if k not in source]
        if missing:
            raise KeyError(f'{where} has no {", ".join(map(repr, missing))}')
        return {k: source[k] for k in keys}

    def __lt__(self, other):
        return self.value < other.value
