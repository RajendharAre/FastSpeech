# FastSpeech 2 for English and Telugu

Take-home assignment for the Speech Processing Lab (SLFI), LTRC, IIIT Hyderabad.
Two **separate** FastSpeech 2 models trained **from scratch** on two datasets:

| Language | Dataset | Source |
|---|---|---|
| English | Svarah | https://huggingface.co/datasets/ai4bharat/Svarah |
| Telugu | IndicTTS Telugu | https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu |

## Status (2026-11-07)

Both languages are aligned, preprocessed and trained to 60,000 steps. The
selected Telugu inference outputs are in the required
`Results/Telugu/test_generated_audios/` and
`Results/Telugu/variance_adaptor_plots/` folders. The 5,000-step comparison
outputs are stored separately under `../FastSpeech_Data/Telugu/`.

| | English (Svarah) | Telugu (IndicTTS) |
|---|---|---|
| Manifest clips | 5,582 | 3,363 |
| Usable after alignment + features | 5,322 (4,500 train / 822 val) | 3,269 (2,751 train / 518 val) |
| Usable audio | 4.842 h | 4.700 h |
| Speakers | 2 (one female, one male) | 1 (male) |
| Dropped, by category | 260 (alignment 164, missing TextGrid 71, pitch 20, feature stats 4, transcript 1) | 94 (alignment 83, missing TextGrid 3, missing audio 8) |
| Aligner | MFA `english_us_arpa` + G2P for 4.47 % OOV | MFA grapheme dictionary trained on the corpus |
| Training | 60,000 steps; checkpoint 60,000 is the one shipped, picked by listening rather than by loss (7/10 against 5/10 for step 5,000); five test WAVs generated | 60,000 steps completed; checkpoint 5,000 had the lowest validation loss among saved checkpoints (3.73799); user slightly prefers the 60,000-step audio; sentence-level listening observations are in the report |
| Audio provenance | exported at 22.05 kHz | **16 kHz source upsampled to 22.05 kHz** |

**English output quality.** All five English test sentences come out
unintelligible. The pipeline is not the cause: the vocoder reproduces the mel it
is handed, text conditioning and durations both check out, and the same
speech-recognition stack scores 0.000 WER on real LibriSpeech through the same
decode path. The cause is that 4.84 h of audio was memorised over 213 epochs
instead of generalised, and the variance adaptor collapses towards the mean.
Report sections 7 and 8 give the measurements.

Still to do before submission: Telugu ASR evaluation, checkpoint hosting, and
re-running `scripts/check_no_overlap.py` against the final Kaggle manifests
(disjointness was reported by the splitter when the manifests were built, but
the check script has not been run on those files).

Completed work: Phase 0 (skeleton), Phase 1 (dataset inspection), the
pre-training infrastructure fixes (environment, training control, preprocessing
drop accounting), alignment QC, English rule-based normalisation with rule
provenance in the manifests, the G2P supplementary-dictionary builder, the
variance-adaptor plot/array export in `inference.py` and the loss-CSV plotting
script. The resumed English training and validation loss plots are in
`Results/English/`; English inference at checkpoint step 35,000 generated five
WAVs plus duration, pitch, and energy plots/JSONs. The selected Telugu inference
outputs are in the required result folders under `Results/Telugu/`.
The 5,000-step comparison is kept outside the repository in
`../FastSpeech_Data/Telugu/checkpoint_5000/`.
Kaggle lacked a Telugu font, so those plots use code-point fallback labels.
Section 7 of `Report/Technical_Report.md` discusses the English plots' scope and
the earlier overlapping-run log limitation. `CHANGES.md` records every change.

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
scripts/                 split_dataset.py, check_no_overlap.py,
                         validate_alignments.py, build_g2p_names.py,
                         plot_training_losses.py, export_svarah.py,
                         export_male.py
conftest.py              puts the repository root on sys.path for pytest
Results/English|Telugu/  loss plots, generated audio, variance-adaptor plots (Phase 5-6)
Report/                  Technical_Report.md and the built Technical_Report.pdf
docs/                    assignment notes, dataset report, test sentences
kaggle/                  dataset inspection scripts, Kaggle run guide and bootstrap.sh
```

## Setup

`requirements.txt` is a **proposed** pin set, not a record of the environment
that actually ran. It targets Kaggle Linux, and it has never been installed as a
set on Kaggle. What the two training/preprocessing sessions really used:

| | Measured on the Kaggle sessions |
|---|---|
| Python | 3.13 (`/usr/local/lib/python3.13/dist-packages`) |
| PyTorch | 2.11.0+cu130, supplied by the image, `torch.cuda.is_available()` True |
| Accelerator | Tesla T4 x2 (`nn.DataParallel` splits the batch of 16 across both) |
| Installed on top of the image | `pyworld`, `tgt`, `unidecode`, `inflect` |
| Not installed | `g2p_en` (see "G2P dictionary for names") |

The pins are also not installable on the author's Windows machine under Python
3.14 (the pinned NumPy has no cp314 wheel); the local tests and smoke tests run
against the local environment instead. Treat `requirements.txt` as a starting
point to reproduce on a fresh image, and the table above as what is known to
work.

To build `pyworld` against a matching NumPy on a fresh Linux image:

```bash
python -m pip install numpy==2.0.2 Cython==3.0.10 setuptools==69.5.1 wheel==0.43.0
python -m pip install -r requirements.txt --no-build-isolation
```

`pyworld==0.3.5` is source-only on Linux. Do not replace Kaggle's PyTorch with a
CPU wheel. The reference repo ships the pretrained HiFi-GAN generator zipped:

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
prints what it skips. **The script itself has never been executed end to end**:
the two Kaggle sessions built their MFA 3.4.3 environment by hand (`micromamba`
plus conda-forge, because `kalpy` is only on conda-forge), and after a session
restore needed repair work (`chmod +x` on the environment's `bin`, and recreating
`lib*.so.N` soname symlinks that the restore had flattened). Treat the script as
unverified and the manual sequence recorded in `CHANGES.md` as the working one.

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
keeps at least one training clip for every validation speaker.

Dataset export is done by `scripts/export_svarah.py` (English) and
`scripts/export_male.py` (Telugu), both run on Kaggle: they write `<clip>.wav` at
22,050 Hz plus the `<clip>.lab` transcript, and a separate 16 kHz corpus for the
aligner, with a per-clip status CSV (`preprocessing_log.csv` style) so a rerun
can be reconciled clip by clip. The Telugu `.lab` must contain the **alignment**
text, not the raw dataset transcript, because
`Telugu/preprocessor/preprocessor.py:352` requires
`len(text_to_sequence(text)) == len(duration)`.

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
and `phones`; both tier names are confirmed by the real Kaggle MFA output, and the
`<clip_id>` to TextGrid file-name mapping is now exercised end to end by the
English preprocessing run (5,322 clips consumed). The QC script itself has still
only been executed against synthetic fixtures, so its flags have never been read
on real alignments.

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
lexicon wins on conflicts. **This step was not used for the submitted run.**
`g2p_en` is pinned in `requirements.txt` but was never installed on either Kaggle
session, so `English/lexicon/g2p_names.txt` does not exist and
`supplementary_lexicon_path` is left empty. Out-of-vocabulary names were handled
one level earlier instead: MFA's own G2P model (`--g2p_model_path`) resolved the
4.47 % of English tokens that are not in `english_us_arpa`, so those words still
got phone alignments and durations. At inference, any word missing from the
LibriSpeech lexicon falls back to `g2p_en` at runtime.

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
`U+XXXX` and a warning is printed. The author's Windows machine does have a
Telugu font (`C:\Windows\Fonts\Nirmala.ttc`), so the figures render there; the
Kaggle image has none, so matplotlib figures with Telugu labels must be made
locally or with a font installed first.

## Report PDF

`Report/Technical_Report.pdf` is built from the Markdown by
`scripts/render_report_pdf.py`, which needs a Chromium-family browser and no
other tool (the author's machine has no pandoc):

```bash
python scripts/render_report_pdf.py --check
```

It converts with `markdown-it-py`, styles the page for A4 and prints through
headless Chrome or Edge (`REPORT_BROWSER=<path>` overrides the search). `--check`
re-reads the PDF and fails if an English sentence is absent or if fewer than 90 %
of the Telugu code points in the Markdown reached the page, which is how a missing
Telugu font is caught. Telugu is **not** matched as a whole string: a PDF content
stream stores glyph runs in visual order, so extraction reorders the vowel signs
and can repeat an anusvara even when the page renders perfectly. The last build
measured 220 of 220 Telugu code points present, 0 replacement characters and 11
pages. `--telugu-font <ttf>` embeds a font as a data URI for machines that have
no Indic font installed.

## Unit tests

```bash
# from each language folder
python -m unittest discover -s tests -p "test_*.py" -v
# from the repository root
python -m unittest discover -s scripts/tests -v
python -m pytest kaggle/tests -q
```

Run the suites **one folder at a time**. A single root-level `pytest English/tests Telugu/tests`
cannot work: each language folder is self-contained by requirement, so both ship
modules with identical names (`text`, `utils`, `model`, `preprocessor`), and only one
of them can be importable at a time. With `--import-mode=importlib` the collision
surfaces as `cannot import name 'telugu' from 'text' (English/text/__init__.py)`.

Latest local run (2026-11-05, Windows, Python 3.14):

| Invocation | Result |
|---|---|
| `pytest English/tests` | 92 passed, 1 skipped |
| `pytest Telugu/tests` | 96 passed, 2 skipped |
| `pytest kaggle/tests scripts/tests` | 73 passed |

The skips are `test_preprocessing_drops.py:47` in both languages (the preprocessing
stack — `pyworld`, `tgt` — is not installed in the local Windows environment, so that
path is only exercised on Kaggle) and `Telugu/tests/test_config_validation.py:205`
(an English-only rule).

`conftest.py` at the repository root puts the root on `sys.path`, so the
`scripts/tests` suite imports `scripts.*` whether pytest is run as a module or
as a console script.

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
`scripts/tests/test_validate_alignments.py` runs the alignment QC on synthetic TextGrids
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
`scripts/tests/test_render_report_pdf.py` covers the Telugu code-point counter,
the Markdown-to-HTML table and `@font-face` embedding, and skips the HTML tests
when `markdown-it-py` is absent.

## Checkpoints

Trained checkpoints are not committed: each `60000.pth.tar` is about 400 MiB,
above GitHub's 100 MB per-file limit, and `*.pth.tar` is git-ignored. The
assignment allows either Git LFS or an accessible external download link, so the
shipped weights are hosted externally and linked here.

| Language | Shipped step | Parameters | Download |
|---|---|---|---|
| English | 60,000 | 35,159,617 | pending |
| Telugu | 60,000 | 35,099,201 | pending |

`English/checkpoints/README.md` and `Telugu/checkpoints/README.md` record the
selection rationale, the payload format, the resume command and the config
checksum for each. The link must remain accessible during evaluation, so open it
in a signed-out browser window and confirm it downloads the file before
submitting.

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
