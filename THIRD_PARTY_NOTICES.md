# Third-party notices

## FastSpeech 2 reference implementation

- Source: https://github.com/ming024/FastSpeech2
- Commit used: `d4e79eb52e8b01d24703b2dfc0385544092958f3` (2021-07-08)
- Licence: MIT, Copyright (c) 2020 Chung-Ming Chien
- Files copied into `English/` and `Telugu/`: `audio/`, `model/`,
  `preprocessor/`, `text/`, `transformer/`, `utils/`, `hifigan/`, `dataset.py`,
  `evaluate.py`, `train.py`, `preprocess.py`, plus `LICENSE`.
- The MIT licence text is kept unchanged in `English/LICENSE` and
  `Telugu/LICENSE`.

## HiFi-GAN vocoder

- Generator and `hifigan/config.json` are redistributed inside the reference
  repository above (MIT licence, see `English/hifigan/LICENSE`).
- Pretrained generator: `generator_universal.pth.tar`
- Audio settings (from `hifigan/config.json`): sampling rate 22050 Hz, 80 mel
  bins, hop size 256, win size 1024, n_fft 1024, fmin 0 Hz, fmax 8000 Hz.
- Used only as a vocoder. No pretrained TTS acoustic model is used.

## Datasets

The datasets are **not** redistributed in this repository.

- Svarah: https://huggingface.co/datasets/ai4bharat/Svarah
  Reported licence: CC BY 4.0. Gated on Hugging Face. Both the licence and the
  gating terms are re-checked by the Phase 1 inspection, which records the
  dataset card metadata verbatim.
- IndicTTS Telugu (Hugging Face copy): https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu
  Licence and gating terms are recorded from the dataset card by the Phase 1
  inspection script. The data originates from the IIT Madras IndicTTS project
  and must be cited as such.

If a dataset card states terms that conflict with this repository's use, the
dataset card wins and the human must be consulted before training.
