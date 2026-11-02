# FastSpeech 2 for English and Telugu

Take-home assignment for the Speech Processing Lab (SLFI), LTRC, IIIT Hyderabad.
Two **separate** FastSpeech 2 models trained **from scratch** on two datasets:

| Language | Dataset | Source |
|---|---|---|
| English | Svarah | https://huggingface.co/datasets/ai4bharat/Svarah |
| Telugu | IndicTTS Telugu | https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu |

Status: **Phase 0 (skeleton), Phase 1 (dataset inspection scripts), and the
pre-training infrastructure fixes (environment, training control, preprocessing
drop accounting).** No dataset preparation has been run, no model has been
trained, no audio has been generated, and no result is reported.
The assignment PDF is now at `docs/FastSpeech2_Assignment (1).pdf`; its
comparison with the audit is in `docs/REQUIREMENTS_AUDIT.md` section 9. See
`docs/PLAN_PHASES.md` for the plan and `CHANGES.md` for every change.

## Repository layout

```
English/                 self-contained English model
  config.yaml            preprocess / model / train / inference sections
  preprocess.py          feature extraction (reference pipeline)
  train.py               training (reference pipeline, one config file)
  inference.py           test-sentence synthesis (reference frontend)
  dataset.py evaluate.py
  audio/ model/ preprocessor/ text/ transformer/ utils/ hifigan/
  checkpoints/           trained weights (git-ignored; see "Checkpoints")
  manifests/             train/val clip lists (Phase 2)
  lexicon/               LibriSpeech lexicon used by the reference English frontend
  tests/                 smoke test, config/frontend/manifest/audio unit tests
Telugu/                  same layout, Telugu-specific config and frontend
scripts/                 split_dataset.py, check_no_overlap.py
Results/English|Telugu/  loss plots, generated audio, variance-adaptor plots (Phase 5-6)
Report/                  Technical_Report.pdf (Phase 7)
docs/                    assignment notes and dataset report
kaggle/                  dataset inspection scripts and Kaggle run guide
```

## Setup

`requirements.txt` targets **Kaggle Linux, Python 3.12** (the current Kaggle
image). It is not installable on the author's Windows Python 3.14 because the
pinned NumPy has no cp314 wheel; the local smoke tests use the existing local
environment, and the audio tests were run in an isolated Python 3.11
environment.

On Kaggle, install the build tools first so that `pyworld` compiles against the
same NumPy it will run with:

```bash
python -m pip install numpy==2.0.2 Cython==3.0.10 setuptools==69.5.1 wheel==0.43.0
python -m pip install -r requirements.txt --no-build-isolation
```

`pyworld==0.3.5` is source-only on Linux. The first Kaggle run must confirm the
build and record the result. PyTorch is supplied by Kaggle; do not replace it
with a CPU wheel. The reference repo ships the pretrained HiFi-GAN generator
zipped:

```bash
cd English && unzip hifigan/generator_universal.pth.tar.zip -d hifigan && cd ..
cd Telugu  && unzip hifigan/generator_universal.pth.tar.zip -d hifigan && cd ..
```

## CPU smoke test (no training)

PowerShell, from the repository root:

```powershell
cd English; python tests/smoke_test.py
cd ../Telugu; python tests/smoke_test.py
```

Each run builds the model from `config.yaml` with random weights and performs one
forward pass on a tiny random batch. It does not read data and does not load any
pretrained acoustic model.

## Manifests and preprocessing

The preprocessor no longer splits randomly and no longer uses `val_size`. It
reads `manifests/train.csv` and `manifests/val.csv` with the columns
`clip_id,speaker,duration,transcript`, and requires both splits to be non-empty.
A missing WAV, transcript or TextGrid, an unreadable alignment, a frontend
length mismatch and similar problems each get a category in
`data/preprocessed/preprocessing_log.csv` and a count in
`data/preprocessed/preprocessing_summary.json`. By default any dropped clip
stops the run **before** normalisation, so a rerun cannot normalise features
twice. `--allow-dropped` is the explicit opt-in to continue with the remaining
clips; it prints the per-category counts and never drops silently.

Build the manifests from measured per-clip durations:

```bash
python scripts/split_dataset.py \
  --manifest manifests/all.csv \
  --train-hours <measured-hours> --val-hours <measured-hours> \
  --seed 1234 \
  --train-out English/manifests/train.csv \
  --val-out English/manifests/val.csv
python scripts/check_no_overlap.py \
  --train English/manifests/train.csv \
  --val English/manifests/val.csv \
  --expect-train-hours <measured-hours> --expect-val-hours <measured-hours>
```

Repeat with the `Telugu/...` output paths. The splitter is deterministic for a
given `--seed` (default 1234), groups identical normalized transcripts, and
keeps at least one training clip for every validation speaker. Dataset export
(WAV, `.lab`, TextGrid) is still not implemented; preprocessing needs those
inputs.

```bash
# from English/ (same for Telugu/)
python preprocess.py config.yaml                  # stops on any dropped clip
python preprocess.py config.yaml --allow-dropped  # continue and report counts
python train.py --config config.yaml
python train.py --config config.yaml --resume     # newest checkpoint in checkpoints/
python train.py --config config.yaml --restore_step <STEP>
python inference.py --config config.yaml --restore_step <STEP>
```

## Training controls

- `train.optimizer.batch_size` and `train.optimizer.group_size` are both read
  from `config.yaml`. For the 50-utterance overfit test use `batch_size: 8` and
  `group_size: 1`; the model can then be trained on 50 clips.
- Losses are appended to `logs/train/train_loss.csv` and
  `logs/val/val_loss.csv` (step plus total, mel, post-net, pitch, energy and
  duration) in addition to TensorBoard and `log.txt`.
- Checkpoints are written every `train.step.save_step` steps and once more at
  `total_step` even when it is not a multiple of `save_step`.
- `--resume` picks the highest numbered `checkpoints/<step>.pth.tar`;
  `--restore_step` picks an explicit one. Both restore the model, optimizer,
  step and Python/NumPy/PyTorch RNG state. The data loader restarts the current
  epoch, so batches already seen in that epoch are not replayed in the same
  order. `train.seed` controls the initial seed.

## Unit tests

```bash
# from each language folder
python -m unittest discover -s tests -p "test_*.py" -v
# from the repository root
python -m unittest discover -s scripts/tests -v
python -m pytest kaggle/tests -q
```

The audio/STFT tests are skipped when librosa is not installed. With the pinned
`librosa==0.10.2.post1` they cover the librosa filter-bank match, inverse
reconstruction, the NumPy waveform helper, the Griffin-Lim writer and the
absence of hard-coded `.cuda()` calls. `test_train_utils.py` covers the CSV loss
log, atomic checkpoint save/load, latest-checkpoint discovery and RNG restore;
`test_overfit_batches.py` batches 50 synthetic clips with `batch_size: 8` and
`group_size: 1`; `test_preprocessor_report.py` covers the drop categories and
counts.

## Checkpoints

Trained checkpoints are not committed. Files over 100 MB are published on the
Hugging Face Hub and linked here (link added after training). Git LFS or an
external download link is also allowed by the assignment; the link must remain
accessible during evaluation.
`English/checkpoints/README.md` and `Telugu/checkpoints/README.md` record the
source and step of each checkpoint.

## Vocoder

Reference implementation's pretrained **HiFi-GAN universal** generator
(MIT, from https://github.com/ming024/FastSpeech2), 22.05 kHz, 80 mel bins,
hop length 256, win length 1024, fmax 8000 Hz (`English/hifigan/config.json`).
It is used for waveform generation only; no pretrained TTS acoustic weights are
loaded. Both acoustic models are trained from scratch.

## Credits and licence

Model and pipeline code adapted from https://github.com/ming024/FastSpeech2
(commit `d4e79eb52e8b01d24703b2dfc0385544092958f3`), MIT licence, Copyright (c)
2020 Chung-Ming Chien. The MIT licence is kept in `English/LICENSE` and
`Telugu/LICENSE`. See `THIRD_PARTY_NOTICES.md`.

## AI assistance

An AI coding assistant was used for implementation help (code copying, refactoring
and documentation); the human author reviewed all code and results and is
responsible for every decision and claim in the submission.
