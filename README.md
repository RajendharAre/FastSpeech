# FastSpeech 2 for English and Telugu

Take-home assignment for the Speech Processing Lab (SLFI), LTRC, IIIT Hyderabad.
Two **separate** FastSpeech 2 models trained **from scratch** on two datasets:

| Language | Dataset | Source |
|---|---|---|
| English | Svarah | https://huggingface.co/datasets/ai4bharat/Svarah |
| Telugu | IndicTTS Telugu | https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu |

Status: **Phase 0 (skeleton), Phase 1 (dataset inspection), the pre-training
infrastructure fixes (environment, training control, preprocessing drop
accounting), alignment QC, English rule-based normalisation with rule provenance
in the manifests, a G2P supplementary-dictionary builder, the variance-adaptor
plot/array export in `inference.py` and the loss-CSV plotting script.**
Telugu data selection is measured and split on Kaggle (male-only, capped at 980
mel frames: train 2,835 clips / 5.486 h, val 528 clips / 1.002 h, 0 clip and 0
transcript overlap, `--seed 1234`); the audio export to `Telugu/data/raw` and the
16 kHz MFA corpus were still running at the time of writing, and no alignment,
preprocessing or training run has completed for either language. No model has
been trained, no audio has been generated and no result is reported. The
alignment parser's expected tier names (`words`, `phones`) are confirmed by the
real Kaggle MFA output; the QC script itself has still only been exercised on
synthetic fixtures, and `g2p_en` is not installed locally, so real G2P generation
is unverified. The English (Svarah) side has had no data work at all.
The assignment PDF is now at `docs/FastSpeech2_Assignment (1).pdf`; its
comparison with the audit is in `docs/REQUIREMENTS_AUDIT.md` sections 9 and 13.
See `docs/PLAN_PHASES.md` for the plan and `CHANGES.md` for every change.

## Repository layout

```
English/                 self-contained English model
  config.yaml            preprocess / model / train / inference sections
  preprocess.py          feature extraction (reference pipeline)
  train.py               training (reference pipeline, one config file)
  inference.py           test-sentence synthesis, variance-adaptor plots + arrays
  dataset.py evaluate.py
  audio/ model/ preprocessor/ text/ transformer/ utils/ hifigan/
  checkpoints/           trained weights (git-ignored; see "Checkpoints")
  manifests/             train/val clip lists (Phase 2)
  lexicon/               LibriSpeech lexicon used by the reference English frontend
  text/normalise.py      rule-based English normalisation (raw text is kept)
  text/acronyms.csv      acronym -> spoken form table (letters or word mode)
  tests/                 smoke test, config/frontend/manifest/audio/variance-export unit tests
Telugu/                  same layout, Telugu-specific config and frontend
tests/                   root pytest tests (alignment QC) and conftest.py
scripts/                 split_dataset.py, check_no_overlap.py,
                         validate_alignments.py, build_g2p_names.py,
                         plot_training_losses.py
Results/English|Telugu/  loss plots, generated audio, variance-adaptor plots (Phase 5-6)
Report/                  Technical_Report.pdf (Phase 7)
docs/                    assignment notes and dataset report
kaggle/                  dataset inspection scripts, Kaggle run guide and bootstrap.sh
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

For the forced-alignment environment, `kaggle/bootstrap.sh` installs Miniforge,
creates an `mfa` conda environment (Montreal Forced Aligner, Python 3.13),
installs the pip pins confirmed by the first real Kaggle run (NumPy 2.1.3,
SciPy 1.16.3, librosa 0.10.2.post1, and pyworld built with
`--no-build-isolation`), unzips the HiFi-GAN generators and downloads the MFA
`english_us_arpa` acoustic model and dictionary. Every step is idempotent and
prints what it skips. The script has not been run yet; the first real Kaggle
run must record its output in `CHANGES.md`.

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
`data/preprocessed/preprocessing_summary.json`. A clip whose saved mel exceeds
`model.max_seq_len` minus `preprocess.preprocessing.max_seq_len_margin` frames
(default 20, i.e. 980 frames here) is dropped as `too_long`; the length comes
from the real STFT output, not from an estimate based on duration. By default any dropped clip
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

### English normalisation and provenance

`transcript` always stays the raw text. Adding `--normalise-english` runs
`English/text/normalise.py` over each clip and appends two optional columns:

| Column | Content |
|---|---|
| `transcript_normalised` | the text the model will actually receive |
| `rules_fired` | semicolon-separated rule names that changed the text |

```bash
python scripts/split_dataset.py \
  --manifest manifests/all.csv \
  --train-hours <measured-hours> --val-hours <measured-hours> \
  --normalise-english --english-root English \
  --train-out English/manifests/train.csv \
  --val-out English/manifests/val.csv
```

The rules, applied in order, are `currency`, `percent`, `time`, `date`,
`abbreviation`, `acronym`, `symbol`, `decimal`, `number`, `whitespace`. Only a
rule that actually changed the text is recorded. Acronyms are read from
`English/text/acronyms.csv` (`acronym,spoken,mode`, mode is `letters` or
`word`); the file is validated on load and an invalid row aborts the run.
Unnormalised manifests keep exactly the four required columns, and both
`split_dataset.py` and `validate_alignments.py` accept manifests with or
without the provenance columns. Telugu is not normalised by this rule set.

### Alignment QC

`scripts/validate_alignments.py` is read-only: it parses MFA TextGrids, compares
`words`/`phones` intervals against configured thresholds, cross-checks clip IDs
against the manifest, and writes `alignment_review.csv`. It never edits an
alignment.

```bash
python scripts/validate_alignments.py \
  --manifest English/manifests/train.csv \
  --textgrid-dir <path-to-TextGrids> \
  --output alignment_review.csv
```

Flags are `short_word_interval`, `long_phone_interval`,
`very_short_phone_interval`, `large_gap_between_non_silence_intervals`,
`textgrid_clip_id_not_in_manifest` and `manifest_clip_missing_textgrid`, each
with the clip, interval and timings. The thresholds are CLI options. The parser
expects the standard Praat long format with `IntervalTier` items named `words`
and `phones`; both tier names are confirmed by the real Kaggle MFA output. The
mapping from manifest `clip_id` to the exported TextGrid file names is still
unverified.

### G2P dictionary for names

`scripts/build_g2p_names.py` collects capitalised out-of-vocabulary tokens from
the manifest transcripts, converts them with `g2p_en`, and writes a
`NAME<TAB>PHONES` dictionary plus a per-name status report. Words already in the
LibriSpeech lexicon and words that are common in lowercase form are skipped.

```bash
python scripts/build_g2p_names.py \
  --manifest English/manifests/train.csv \
  --lexicon English/lexicon/librispeech-lexicon.txt \
  --output English/lexicon/g2p_names.txt \
  --report English/lexicon/g2p_names_report.csv
```

Set `preprocess.path.supplementary_lexicon_path` in `English/config.yaml` to
the generated file; `inference.py` loads it after the main lexicon, and the main
lexicon wins on conflicts. This step needs `g2p_en`, which is pinned in
`requirements.txt` but is not installed in the local Windows environment, so no
real pronunciations have been generated yet.

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

## Deliverable plots

`scripts/plot_training_losses.py` turns the loss CSVs written by `train.py` into
the two PNGs the submission needs (read-only; it neither trains nor edits logs):

```bash
# from the repository root, after a run
python scripts/plot_training_losses.py \
  --train-csv English/logs/train/train_loss.csv \
  --val-csv English/logs/val/val_loss.csv \
  --out-dir Results/English --title English
```

It checks the CSV header against `utils/train_utils.LOSS_COLUMNS` and refuses a
missing column, a non-numeric value or non-monotonic steps rather than drawing a
wrong curve. `training_loss.png` plots the total loss plus a log-scale panel of
the five components; `validation_loss.png` does the same for the validation log.

The variance-adaptor plots (duration, pitch/F0 and energy per test sentence) are
produced by `inference.py` itself: for every sentence it writes
`Results/<Language>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png` and
a `variance_adaptor_plots/<ID>.json` holding the predicted arrays, so the report
can quote numbers without re-running synthesis. Pitch and energy are denormalised
with `data/preprocessed/stats.json`; the run aborts if the config does not use
`phoneme_level` features, because the export indexes predictions per symbol. If
no installed font covers the Telugu code points the x labels fall back to
`U+XXXX` and a warning is printed — install a Telugu font (e.g. Noto Sans Telugu)
before generating the submitted figures.

## Unit tests

```bash
# from each language folder
python -m unittest discover -s tests -p "test_*.py" -v
# from the repository root
python -m unittest discover -s scripts/tests -v
python -m pytest tests/test_validate_alignments.py -q
python -m pytest kaggle/tests -q
```

`tests/conftest.py` puts the repository root on `sys.path`, so the root pytest
tests import `scripts.*` whether pytest is run as a module or as a console
script.

The audio/STFT tests are skipped when librosa is not installed. With the pinned
`librosa==0.10.2.post1` they cover the librosa filter-bank match, inverse
reconstruction, the NumPy waveform helper, the Griffin-Lim writer and the
absence of hard-coded `.cuda()` calls. `test_train_utils.py` covers the CSV loss
log, atomic checkpoint save/load, latest-checkpoint discovery and RNG restore;
`test_overfit_batches.py` batches 50 synthetic clips with `batch_size: 8` and
`group_size: 1`; `test_preprocessor_report.py` covers the drop categories and
counts. `English/tests/test_normalise.py` covers each normalisation rule and the
acronym table; `test_lexicon_merge.py` covers the supplementary dictionary.
`scripts/tests/test_manifest_normalisation.py` and `test_build_g2p_names.py`
cover provenance columns and the G2P builder, and
`tests/test_validate_alignments.py` runs the alignment QC on synthetic TextGrids
with CRLF line endings. `test_max_seq_len.py` (each language) pins what happens
when a mel batch exceeds `model.max_seq_len`: training silently truncates to
1000 frames, eval and inference do not truncate. `test_too_long.py` (each
language) checks the preprocessing guard against the real STFT path: the frame
count is `1 + floor(samples / hop)` and matches `librosa.stft(center=True)`,
and a mel one frame over `max_seq_len - margin` is dropped as `too_long`.
`scripts/tests/test_plot_training_losses.py` checks the loss-CSV validation and
both PNG outputs, and `tests/test_variance_export.py` (each language) performs a
real variance export into a temp directory and asserts the three plots, the JSON
arrays, the `phoneme_level` abort and the missing-`stats.json` abort.

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
