"""Registry of every internal dataset that makes up Universal Audio Understanding.

`DATASETS_DIRECTORY` maps dataset name -> `InternalDataset`, recording each one's
canonical tasks, splits and archive path under `data/<name>/`. A JSON run config
(see `json_config_loader`) may only reference names in this registry, and may
only request a subset of the tasks/splits declared here.
"""
import datasets

from .internal_dataset import InternalDataset
from .tasks import Task

# All datasets, in case-insensitive alphabetical order.
LIBRICSS_DESCRIPTION = '''Continuous speech separation (CSS) is an approach tohandling overlapped
speech in conversational audio signals. A real recorded dataset, called LibriCSS, is derived from
LibriSpeech by concatenating the corpus utterances to simulate a conversation and capturing the
audio replays with far-field microphones.
'''
DATASETS = [
 InternalDataset(
        name='AESDD',
        description=(
            'Aced Emotional Speech Dynamic Database is a Greek Speech Emotion Recognition Dataset publically available for research purposes.\nhttps://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/'
        ),
        tasks=[Task.CLASSIFICATION],
        splits=[datasets.Split.TRAIN],
        data_url='data/AESDD/aesdd.tar.gz'
    ),
    InternalDataset(
        name='AudioMNIST',
        description=(
            'AudioMNIST consists of 30000 audio samples of spoken digits (0-9) of 60 different speakers.'
        ),
        tasks=[Task.ASR],
        splits=[datasets.Split.TRAIN],
        data_url='data/AudioMNIST/AudioMNIST.tar.gz'
    ),
    InternalDataset(
        name='AudioMNISTCommonSense',
        description=(
            'AudioMNISTCommonSense is a multimodal numerical commonsense reasoning task based on AudioMNIST.'
        ),
        tasks=[Task.QA],
        splits=[datasets.Split.TEST],
        data_url='data/AudioMNISTCommonSense/AudioMNISTCommonSense.tar.gz'
    ),
    InternalDataset(
        name='Clotho',
        description=(
            'Clotho is a novel audio captioning dataset, consisting of 4981 audio samples, and each audio sample '
            'has five captions (a total of 24 905 captions). Audio samples are of 15 to 30 s duration and captions '
            'are eight to 20 words long. '
        ),
        tasks=[Task.CAPTION, Task.COMMONSENSE, Task.COMMONSENSE_HARD],
        splits=[datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST],
        data_url='data/Clotho/Clotho.tar.gz'
    ),
    InternalDataset(
        name='colombian_spanish',
        description=(
            'A data set which contains recordings of Colombian Spanish.'
        ),
        tasks=[Task.ASR],
        splits=[datasets.Split.TRAIN],
        data_url='data/colombian_spanish/colombian_spanish.tar.gz'
    ),
    InternalDataset(
        name='EMNS',
        description=(
            """We have reformatted the data from .webm format into .wav format.\nhttps://www.openslr.org/136/"""
        ),
        tasks=[Task.CLASSIFICATION, Task.ASR],
        splits=[datasets.Split.TRAIN],
        data_url='data/EMNS/EMNS.tar.gz'
    ),
    InternalDataset(
        name='esc50',
        description=(
            'The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings '
            'suitable for benchmarking methods of environmental sound classification.'
        ),
        tasks=Task.CLASSIFICATION,
        splits=[datasets.Split.TRAIN, datasets.Split.TEST,
                datasets.Split.VALIDATION],
        data_url='data/esc50/esc50.tar.gz'
    ),
    InternalDataset(
        name='Ewe_BibleTTS',
        description=(
            'High fidelity speech corpus in Ewe from BibleTTS.\nhttps://www.openslr.org/129/'
        ),
        tasks=[Task.ASR],
        splits=[datasets.Split.TRAIN],
        data_url='data/Ewe_BibleTTS/Ewe_BibleTTS.tar.gz'
    ),
    InternalDataset(
        name='hi_kia',
        description=(
            """Wake-up word emotion recognition is a task to capture the speakers’ emotional state using short lexically-matched speech such as Ok Google or Hey Siri."""
        ),
        tasks=[Task.CLASSIFICATION],
        splits=[datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST],
        data_url='data/hi_kia/hi_kia.tar.gz'
    ),
    InternalDataset(
        name='libricss', description=LIBRICSS_DESCRIPTION,
        tasks=Task.ASR_TIMESTAMP_SEARCH,
        splits=datasets.Split.TEST,
        data_url='data/libricss/libricss.tar.gz'
    ),
    InternalDataset(
        name='libricss_subseg', description=LIBRICSS_DESCRIPTION,
        tasks=Task.ASR_TIMESTAMP_SEARCH,
        splits=datasets.Split.TEST,
        data_url='data/libricss_subseg/libricss_subseg.tar.gz'
    ),
    InternalDataset(
        name='Lingala_BibleTTS',
        description=(
            'High fidelity speech corpus in Lingala from BibleTTS.\nhttps://www.openslr.org/129/'
        ),
        tasks=[Task.ASR],
        splits=[datasets.Split.TRAIN],
        data_url='data/Lingala_BibleTTS/Lingala_BibleTTS.tar.gz'
    ),
    InternalDataset(
        name='MELD',
        description=(
            'MELD contains the same dialogue instances available in EmotionLines, but it also '
            'encompasses audio and visual modality along with text. MELD has more than 1400 '
            'dialogues and 13000 utterances from Friends TV series.'
        ),
        tasks=[Task.CLASSIFICATION],
        splits=[datasets.Split.TRAIN, datasets.Split.TEST,
                datasets.Split.VALIDATION],
        data_url='data/MELD/MELD.tar.gz'
    ),
    InternalDataset(
        name='MESD',
        description=(
            'A talking-face video corpus featuring 60 actors and actresses talking with '
            'eight different emotions at three different intensity levels.'
        ),
        tasks=Task.CLASSIFICATION,
        splits=datasets.Split.TRAIN,
        data_url='data/MESD/MESD.tar.gz'
    ),
    InternalDataset(
        name='MLEnd_Intonation',
        description=(
            """https://www.kaggle.com/datasets/jesusrequena/mlend-spoken-numerals"""
        ),
        tasks=[Task.CLASSIFICATION, Task.ASR],
        splits=[datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST],
        data_url='data/MLEnd_Intonation/MLEnd_Intonation.tar.gz'
    ),
    InternalDataset(
        name='MusicCapsCommonSense',
        description=(
            'MusicCapsCommonSense is a multimodal multiple choice commonsense reasoning task based on MusicCaps.'
        ),
        tasks=[Task.COMMONSENSE],
        splits=[datasets.Split.TEST],
        data_url='data/MusicCapsCommonSense/MusicCapsCommonSense.tar.gz'
    ),
    InternalDataset(
        name='MustardPP-SingleTurn',
        description=(
            'MUStARD++ is a multimodal sarcasm detection dataset (MUStARD) pre-annotated with 9 emotions. '
            'It can be used for the task of detecting the emotion in a sarcastic statement.'
        ),
        tasks=[Task.CLASSIFICATION, Task.ASR],
        splits=[datasets.Split.TRAIN],
        data_url='data/MustardPP-SingleTurn/MustardPP-SingleTurn.tar.gz'
    ),
    InternalDataset(
        name='nigerian_english',
        description=(
            'A data set which contains recordings of Nigerian English.'
        ),
        tasks=[Task.ASR],
        splits=[datasets.Split.TRAIN],
        data_url='data/nigerian_english/nigerian_english.tar.gz'
    ),
    InternalDataset(
        name='OpenMic',
        description=(
            'OpenMIC-2018 is an instrument recognition dataset containing 20,000 examples of '
            'Creative Commons-licensed music available on the Free Music Archive.'
        ),
        tasks=Task.CLASSIFICATION,
        splits=[datasets.Split.TRAIN, datasets.Split.TEST],
        data_url='data/OpenMic/OpenMic.tar.gz'
    ),
    InternalDataset(
        name='peruvian_spanish',
        description=(
            'A data set which contains recordings of Peruvian Spanish.'
        ),
        tasks=[Task.ASR],
        splits=[datasets.Split.TRAIN],
        data_url='data/peruvian_spanish/peruvian_spanish.tar.gz'
    ),
    InternalDataset(
        name='ravnursson_faroese',
        description=(
            'The corpus Ravnursson Faroese Speech and Transcrips (or Ravnursson Corpus for short) is a collection of speech recordings with transcriptions intended for Automatic Speech Recognition (ASR) applications in Faroese.\nhttps://mtd.setur.fo/en/resource/ravnursson-faroese-speech-and-transcripts/'
        ),
        tasks=[Task.ASR],
        splits=[datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST],
        data_url='data/ravnursson_faroese/ravnursson_faroese.tar.gz'
    ),
    InternalDataset(
        name='slurp_real',
        description=(
            'Speech Dataset from "A Spoken Language Understanding Resource Package"\nLicense: CC BY-NC 4.0\nhttps://github.com/pswietojanski/slurp/tree/master'
        ),
        tasks=[Task.CLASSIFICATION, Task.INTENT_DETECTION, Task.INTENT_DETECTION_NL, Task.ACTION_CLASSIFICATION, Task.ASR],
        splits=[datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST],
        data_url='data/slurp_real/slurp_real.tar.gz'
    ),
    InternalDataset(
        name='SparseLibriMix',
        description=(
            'An open source dataset for source separation in noisy environments and with '
            'variable overlap-ratio. Both are derived from LibriSpeech (clean subset) and WHAM '
            'noise.'
        ),
        tasks=Task.ASR_TIMESTAMP_SEARCH,
        splits=datasets.Split.TEST,
        data_url='data/SparseLibriMix/SparseLibriMix.tar.gz'
    ),
    InternalDataset(
        name='URDU',
        description=(
            'URDU dataset contains emotional utterances of Urdu speech gathered from Urdu '
            'talk shows. It contains 400 utterances of four basic emotions: Angry, Happy, '
            'Neutral, and Emotion. There are 38 speakers (27 male and 11 female).'
        ),
        tasks=Task.CLASSIFICATION,
        splits=datasets.Split.TRAIN,
        data_url='data/URDU/URDU.tar.gz'
    ),
    InternalDataset(
        name='VIVOS',
        description=(
            'VIVOS is a free Vietnamese speech corpus consisting of 15 hours of recording speech prepared for Automatic Speech Recognition task. The corpus was published by AILAB, a computer science lab of VNUHCM - University of Science.'
        ),
        tasks=[Task.ASR, Task.ENGLISH_TRANSLATION],
        splits=[datasets.Split.TRAIN, datasets.Split.TEST],
        data_url='data/VIVOS/VIVOS.tar.gz'
    ),
    InternalDataset(
        name='VocalSound',
        description=(
            'VocalSound is a free dataset consisting of 21,024 crowdsourced recordings of laughter, '
            'sighs, coughs, throat clearing, sneezes, and sniffs from 3,365 unique subjects. '
            'The VocalSound dataset also contains meta information such as speaker age, gender, '
            'native language, country, and health condition.'
        ),
        tasks=Task.CLASSIFICATION,
        splits=[datasets.Split.TRAIN, datasets.Split.TEST,
                datasets.Split.VALIDATION],
        data_url='data/VocalSound/VocalSound.tar.gz'
    ),
    InternalDataset(
        name='Yoruba_BibleTTS',
        description=(
            'High fidelity speech corpus in Yoruba from BibleTTS.\nhttps://www.openslr.org/129/'
        ),
        tasks=[Task.ASR],
        splits=[datasets.Split.TRAIN],
        data_url='data/Yoruba_BibleTTS/Yoruba_BibleTTS.tar.gz'
    ),
]

DATASETS_DIRECTORY = {dataset.name: dataset for dataset in DATASETS}
