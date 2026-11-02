# Requirements audit

- Date: 2026-11-02
- Repository: `D:\NAAC\Data Science Projects\FastSpeech 2`
- Scope: audit only. No code was changed for this audit, no training was started, no audio or plots were generated, and nothing was committed or pushed (the folder is not a git repository yet).
- Method: read every file in `docs/`, inspected every project-owned script under `English/` and `Telugu/`, compared them with the binding docs, and ran only cheap CPU checks (smoke test, unit tests, compile check, text-frontend probes). Every claim below cites a file and line or a command output in section 1.

## 0. Authority limitation: the assignment PDF is not on this machine

> **Update (2026-11-02):** the PDF was supplied after this audit at
> `docs/FastSpeech2_Assignment (1).pdf`, not in `docs/source/`. The limitation
> below applied to the original audit only. The PDF comparison and the list of
> differences are in section 9.

The binding docs say the original assignment PDF (`FastSpeech2_Assignment.pdf`, named in `docs/PROBLEM_STATEMENT.md:3`) is the source of truth. That PDF was not found:

- the whole project folder (`find . -iname "*.pdf"` returned nothing);
- a Qoder session workspace under the user's `Documents/Qoder/<session>` folder;
- `Desktop`, `Downloads` and `Documents` (searched for `*FastSpeech*` and `*IIIT*`);
- `D:\NAAC` and `D:\` (searched for PDFs in the earlier part of this audit).

**Consequence:** this audit uses `docs/PROBLEM_STATEMENT.md` (which quotes the PDF), `docs/DELIVERABLES_CHECKLIST.md`, `docs/DATA_STRATEGY.md`, `docs/KAGGLE_WORKFLOW.md`, `docs/PLAN_PHASES.md`, `docs/REPORT_OUTLINE.md` and `docs/RULES_FOR_AI.md` as the available proxy. No PDF-only requirement was invented. A requirement marked "PDF proxy" below is one the docs attribute to the PDF; it must still be checked against the PDF by the human. Until the PDF is compared, the audit is complete against the docs, not against the PDF.

## 1. Commands run during this audit (real output)

All commands were run locally on Windows, CPU only. No checkpoint, dataset or audio file exists, so no training, preprocessing or synthesis command was run.

| Command | Result |
|---|---|
| `cd English && python tests/smoke_test.py` | `forward pass: OK in 0.200 s`; `parameters: 35,159,617`; mel `(2, 36, 80)`, pitch/energy/log-duration `(2, 12)`, `mel_lens_out [36, 27]` |
| `cd Telugu && python tests/smoke_test.py` | `forward pass: OK in 0.094 s`; same parameter count and shapes |
| `python -m pytest kaggle/tests -q` | `10 passed, 1 warning in 0.20s` |
| `python -m py_compile` on both `train.py`, `preprocess.py`, `inference.py` | `compile OK` |
| Call `load_test_sentences('../docs/test_sentences.json', 'English')` | `SystemExit: Refusing to run: ... still has null 'final' text for E1, E2a, E2b, E3, E4.` |
| Call `load_test_sentences('../docs/test_sentences.json', 'Telugu')` | `SystemExit: Refusing to run: ... still has null 'final' text for T1, T2a, T2b, T3, T4.` |
| `text_to_sequence("తాడుపై అతని పట్టుచాలా బలంగా ఉంది.", ["basic_cleaners"])` in `Telugu/` | `Telugu symbols kept: 5 [11, 11, 11, 11, 7]` for a full sentence |
| `english_cleaners("The meeting will start at 10:45 AM on 12 November 2026, ...")` | `the meeting will start at ten:forty-five am on twelve november twenty twenty-six, ...` (colon retained) |
| `from g2p_en import G2p` | `ModuleNotFoundError: No module named 'g2p_en'` locally; the English frontend could not be exercised |
| `sha256sum docs/test_sentences.json` | `420187d15f4f9c4a61202ce8d6c3b45f72fbb4535c8f3be0689c01cf60847a6f` (unchanged) |
| `git rev-parse --is-inside-work-tree` | `fatal: not a git repository` |
| Largest project file | `51,942,859` bytes (`hifigan/generator_universal.pth.tar.zip`), below the 100 MB rule |
| `find Results -type f ! -name .gitkeep` | no output: zero audio files and zero plots |
| `find Report -type f` | only `.gitkeep`: no `Technical_Report.pdf` |

Full inference was **not** run: all `final` sentences are null, no checkpoint exists, and `g2p_en` is not installed locally. The null-refusal result above comes from calling the loader directly; it does not synthesise audio.

Status legend:

- **Met**: present and exercised by a command.
- **Met (code only)**: implemented, but not executed end to end (no data, checkpoint or GPU).
- **Partial**: some implementation exists, but a required part is missing or unverified.
- **Missing**: no implementation.
- **Blocked**: cannot be completed until a human decision or an external input exists.
- **Not due yet**: scheduled for a later phase; not a defect by itself.
- **Unverified**: a claim or dependency that has not been checked against the real data or environment.

## 2. Requirement-by-requirement status

### A. Submission, folder structure, files and licensing

| ID | Requirement | Source | Status | Evidence / gap |
|---|---|---|---|---|
| A1 | Exact required folder tree | DELIVERABLES_CHECKLIST | Partial | Core folders and five required files per language exist. `Results/<Lang>/training_loss.png`, `validation_loss.png`, generated audio and plots are absent (Phases 5-6). `Report/Technical_Report.pdf` is absent. |
| A2 | Each language folder is self-contained | RULES_FOR_AI §4 | Met | Both contain `audio/`, `model/`, `preprocessor/`, `text/`, `transformer/`, `utils/`, `hifigan/`, `dataset.py`, `evaluate.py`. The English lexicon is only in `English/`; the Telugu config does not use it. |
| A3 | Separate config file per language with `preprocess`, `model`, `train`, `inference` | RULES_FOR_AI §5 | Met | `English/config.yaml`, `Telugu/config.yaml`; `utils/config.py` enforces the four sections. Smoke tests load both. |
| A4 | Two separate FastSpeech 2 models, trained from scratch, no pretrained acoustic weights | RULES_FOR_AI (not PDF; see §9.3 #3) | Partial | Separate folders, configs and model instances. No training yet. Code inspection found no acoustic checkpoint load except the project's own `checkpoints/<step>.pth.tar`; the only pretrained file is the HiFi-GAN vocoder. |
| A5 | Public GitHub repository and assignment-form submission | DELIVERABLES_CHECKLIST / PROBLEM_STATEMENT (PDF does not name the form; §9.3 #4) | Missing | `git rev-parse` reports no repository; no remote, no form submission. |
| A6 | `README.md` | DELIVERABLES_CHECKLIST | Partial | Exists and is honest about Phase 0/1 only (`README.md:11-13`). It does not yet contain data preparation, training results, checkpoint links or final commands. |
| A7 | `requirements.txt` | DELIVERABLES_CHECKLIST | Partial | Now pins a Kaggle 3.12 set including `datasets`, `huggingface_hub`, `nltk`, `pytest`, `tgt==1.5` (the old `1.4.4` had no wheel) and the audio stack. `torch` is deliberately left to Kaggle. The full set has not been installed on Kaggle; `pyworld==0.3.5` is source-only on Linux and needs the documented `--no-build-isolation` build. |
| A8 | Trained checkpoints for both languages | DELIVERABLES_CHECKLIST | Missing | `checkpoints/` contains only `README.md`. No weights. |
| A9 | `Report/Technical_Report.pdf` | DELIVERABLES_CHECKLIST | Missing | `Report/` is empty except `.gitkeep`; only an outline exists (`docs/REPORT_OUTLINE.md`). |
| A10 | Training and validation loss plots for both models | DELIVERABLES_CHECKLIST | Missing | No PNGs. No CSV and no plotting script; only TensorBoard event logging in code (`English/train.py:100-112`), and TensorBoard logs are git-ignored. |
| A11 | Five generated audio files per language, `<ID>.wav` | DELIVERABLES_CHECKLIST | Missing | `Results/*/test_generated_audios/` contains only `.gitkeep`. Inference is blocked (see F1/F2). |
| A12 | Fifteen variance-adaptor plots per language (duration, pitch, energy for five IDs) | DELIVERABLES_CHECKLIST | Missing | `Results/*/variance_adaptor_plots/*` contains only `.gitkeep`; no plotting code exists. **0 of 30 plots can currently be produced.** |
| A13 | Keep the reference MIT licence and credit it | RULES_FOR_AI §4 | Met | `English/LICENSE`, `Telugu/LICENSE`, `THIRD_PARTY_NOTICES.md`, `README.md` credits commit `d4e79eb...`. |
| A14 | Respect dataset licences and cite both datasets | RULES_FOR_AI §6 | Unverified | Svarah's CC BY 4.0 is repeated from the card, not checked by a run; the IndicTTS-HF licence has not been read. `docs/dataset_report.md` is still a stub. |
| A15 | Allowed extras only | DELIVERABLES_CHECKLIST / Gate 0 approval | Met | Gate 0 (2026-11-02) approved keeping `tests/`, `lexicon/`, `kaggle/` and both HiFi-GAN copies so each language folder stays self-contained. `docs/`, `CHANGES.md`, `THIRD_PARTY_NOTICES.md`, `.gitignore` and `manifests/` remain allowed. See §9.3 #10. |
| A16 | Never commit tokens, datasets, raw audio or files over 100 MB | RULES_FOR_AI §6 | Met (local state) | `.gitignore` excludes data, `.npy`, TextGrids, checkpoints*; largest file is 51,942,859 B; no token-like string found. Not verifiable in git history because there is no repository yet. |

\* `.gitignore` ignores `*.pth.tar`; the 52 MB zipped vocoder is intentionally kept.

### B. Dataset selection, split and manifests

| ID | Requirement | Source | Status | Evidence / gap |
|---|---|---|---|---|
| B1 | Use Svarah (English) and IndicTTS Telugu (Telugu) | PDF proxy | Met (configuration only) | `English/config.yaml:6`, `Telugu/config.yaml:5`; no data is present. |
| B2 | About 5 h training and 1 h validation per language | PDF proxy | Missing | No selection script and no measured hours. `val_size: 512` (`English/config.yaml:15`) is a clip count, not an hour target, and the achieved hours are unknown. |
| B3 | Train and validation must not overlap (clip ID and identical transcript) | PDF proxy / DATA_STRATEGY | Missing | No `check_no_overlap.py`; no manifests. The reference preprocessor performs one unseeded random shuffle (`English/preprocessor/preprocessor.py:142-151`). |
| B4 | Validation speakers must also appear in training | DATA_STRATEGY | Missing | Nothing enforces a speaker-aware split. The random shuffle can put all clips of a rare speaker in one side. |
| B5 | Document selection and subsampling strategy | PDF proxy | Missing | Only comments in the configs; no script, no report section. |
| B6 | Commit `manifests/train.csv` and `val.csv` plus a reproducing script | RULES_FOR_AI §6 / DATA_STRATEGY | Missing | `English/manifests/` and `Telugu/manifests/` contain only `.gitkeep`. `English/config.yaml:14` and `Telugu/config.yaml:16` refer to `scripts/select_data.py`, which does not exist. |
| B7 | Fixed random seed | RULES_FOR_AI §5 | Missing | No `random.seed`, `np.random.seed` or `torch.manual_seed` in `train.py` or `preprocessor.py`; only the smoke test seeds itself. |
| B8 | Dataset facts verified from the real data | RULES_FOR_AI §2 / DATA_STRATEGY | Missing | `docs/dataset_report.md:1-6` explicitly says the report is not generated. The Phase 1 inspector exists and its helpers pass 10 tests, but it was never run on the datasets. |
| B9 | Kaggle Stage A export of selected clips | KAGGLE_WORKFLOW | Missing | `kaggle/inspect_datasets.py` only measures; there is no download/export script that writes `data/raw/<speaker>/<id>.wav` and `.lab`. |

### C. Preprocessing and FastSpeech 2 supervision

| ID | Requirement | Source | Status | Evidence / gap |
|---|---|---|---|---|
| C1 | 80-bin mel, 22.05 kHz, hop 256, win 1024, fmin 0, fmax 8000 | Vocoder config / DATA_STRATEGY | Met (configuration) | `English/config.yaml:24-31` and `Telugu/config.yaml:28-35` match `hifigan/config.json` (`sampling_rate` 22050, `hop_size` 256, `win_size` 1024, `fmax` 8000). |
| C2 | Duration per phoneme from a forced alignment | FastSpeech 2 | Partial | `preprocessor.py:155-291` reads a TextGrid `phones` tier and saves per-symbol frame durations. Nothing creates the TextGrids or the `.lab` files. |
| C3 | Pitch supervision at phoneme level, with normalisation statistics | FastSpeech 2 | Met (code only) | `preprocessor.py:181-217`: `pyworld.dio`, `stonemask`, interpolation, phoneme averaging; `stats.json` written at 119-134. Not executed. |
| C4 | Energy supervision at phoneme level, with normalisation statistics | FastSpeech 2 | Met (code only) | `preprocessor.py:193-228`; energy from STFT magnitude norm; same stats file. Not executed. |
| C5 | Speaker map for the multi-speaker model | FastSpeech 2 / config `multi_speaker: true` | Met (code only) | `speakers.json` written at `preprocessor.py:116-117`; `FastSpeech2` reads it at `model/fastspeech2.py:30-40`. Not executed. |
| C6 | `train.txt` / `val.txt` metadata | dataset.py | Partial | Written at `preprocessor.py:146-151`, but from an unseeded shuffle and not from manifests, so the required 5 h/1 h split and no-overlap rule are not enforced. |
| C7 | Resample all source audio to 22.05 kHz and convert to mono | DATA_STRATEGY | Partial | `librosa.load(wav_path)` at `preprocessor.py:172` relies on librosa's default `sr=22050` and mono mix-down. There is no explicit `sr=` argument, assertion or log, and the 48 kHz Telugu source is not handled by a dedicated export step. |
| C8 | Transcript cleaning and text normalisation | PDF proxy / DATA_STRATEGY | Missing | No script normalises the `.lab` transcripts before alignment. The English inference frontend bypasses the configured `english_cleaners` (see D8). Telugu has no normaliser (see D2). |
| C9 | Record alignment failures and dropped clips with counts | DATA_STRATEGY | Missing | A clip without a TextGrid is silently skipped; `process_utterance` returning `None` is `continue`d (`preprocessor.py:77-83`). No counts are kept. |
| C10 | Language-appropriate phone inventory and TextGrid tier | DATA_STRATEGY | Unverified | The tier name is hardcoded to `phones` and silences to `sil`, `sp`, `spn` (`preprocessor.py:254`). This is plausible for an English MFA recipe; no Telugu aligner or phone set has been chosen. |
| C11 | Filter clips that are too long for memory | Stability | Missing | No duration or symbol-length filter in selection or preprocessing. `max_seq_len` (`English/config.yaml:63`) only truncates at inference in `transformer/Models.py:154`; training has no guard. |
| C12 | Deterministic speaker IDs | Reproducibility | Missing | Speakers are numbered by `enumerate(os.listdir(...))` (`preprocessor.py:67-68`), so IDs depend on filesystem order. `inference.speaker_id: 0` is a placeholder that may not mean the intended speaker. |
| C13 | All supervision is consumed by the model and loss | FastSpeech 2 | Met (code only) | `dataset.py:32-74` loads mel/pitch/energy/duration/speaker; `model/loss.py:19-92` uses all five losses; smoke test proves tensor shapes only. |
| C14 | Preprocessing command documented | PDF proxy | Partial | `README.md:60-65` gives `python preprocess.py config.yaml`, but there are no commands for export, `.lab` creation or alignment, which are prerequisites. |

### D. Text frontends, English vs Telugu

| ID | Requirement | Source | Status | Evidence / gap |
|---|---|---|---|---|
| D1 | English lexicon + `g2p_en` frontend | Reference implementation | Met (code) / Unverified (run) | `English/inference.py:26-64` reads the LibriSpeech lexicon and falls back to `G2p`. `g2p_en` is not installed locally, so the path is unexercised. |
| D2 | English number, date, time and abbreviation normalisation | Test sentence E4 / RULES_FOR_AI §8 | Partial | `english_cleaners` expands numbers (`text/cleaners.py:82-89`), but `preprocess_english` never calls it before lexicon/g2p (`inference.py:38-64`). Local output for E4-style text keeps the colon: `ten:forty-five`. `andaroo` is OOV. `g2p_en` behaviour on `10:45` was not verified. |
| D3 | Telugu-specific symbol set | DATA_STRATEGY gate | Missing | `Telugu/text/symbols.py` is byte-identical to the English file: ASCII, ARPAbet and pinyin only. A full Telugu sentence survives as 5 symbols (section 1). |
| D4 | Telugu Unicode handling, cleaning and number-to-words | DATA_STRATEGY gate / T4 | Missing | `Telugu/config.yaml:21` uses `basic_cleaners`, which only lowercases and collapses whitespace (`text/cleaners.py:67-71`). There is no Telugu number/date/time normaliser. |
| D5 | Telugu G2P / lexicon / character-to-symbol mapping | DATA_STRATEGY gate | Missing | `lexicon_path` is empty (`Telugu/config.yaml:11`); no mapping, no alignment dictionary, no decision recorded. |
| D6 | Telugu inference for T1-T4 | PDF proxy | Missing | `Telugu/inference.py:67-77` raises `NotImplementedError` for `language: te`, intentionally. |
| D7 | Load test sentences only from `final`; refuse null | RULES_FOR_AI §8 | Met | `English/inference.py:80-94`; verified for both language sections (section 1). |
| D8 | Never edit the test sentences | RULES_FOR_AI §8 | Met | SHA-256 still matches the Phase 0 baseline; no sentence text was changed. |
| D9 | Report both original and normalised text per sentence | RULES_FOR_AI §8 / REPORT_OUTLINE | Missing | English prints raw text and phonemes but does not save the normalised form; Telugu produces none. No per-sentence record exists for the report table. |

### E. Training, validation, checkpointing and resuming

| ID | Requirement | Source | Status | Evidence / gap |
|---|---|---|---|---|
| E1 | Separate `train.py` per language | DELIVERABLES_CHECKLIST | Met | Both exist and compile; only the config loader differs from the reference. |
| E2 | FastSpeech 2 training loop with the five losses | FastSpeech 2 | Met (code only) | `train.py:76-170`, `model/loss.py`; no real data, no run. |
| E3 | Validation loss every `val_step` on a held-out set | PDF proxy | Partial | `train.py:143-150` calls `evaluate`; `evaluate.py:18-31` loads `val.txt`. The split itself is the problem (B2-B4), and no validation loss has ever been logged. |
| E4 | TensorBoard logging | RULES_FOR_AI §5 | Met (code only) | `train.py:54-59, 112`; not run. |
| E5 | CSV loss logging in addition to TensorBoard | RULES_FOR_AI §5 / KAGGLE_WORKFLOW | Missing | Only `log.txt` and TensorBoard events (`train.py:100-112`); no CSV anywhere in `English/` or `Telugu/`. |
| E6 | Save checkpoints during training | KAGGLE_WORKFLOW / DELIVERABLES | Met (code only) | `train.py:152-162` saves model and optimizer every `save_step`; not exercised. |
| E7 | Resumable training via `--restore_step` | KAGGLE_WORKFLOW | Partial | `utils/model.py:15-30` restores model and optimizer; `train.py:62-74` resumes the step counter. Not tested. It does not restore RNG state or data order, and `torch.load` has no `map_location`. |
| E8 | Save a usable final checkpoint | Deliverables | Partial | The loop exits with `quit()` at `total_step` (`train.py:164-165`); a final save happens only if `total_step` is a multiple of `save_step`. The two current values are compatible, but this is fragile for a Gate-4 step cap. |
| E9 | Overfit test on about 50 utterances | KAGGLE_WORKFLOW / Gate 4 | Missing | Not run. With `batch_size: 16` and `group_size = 4`, `assert batch_size * group_size < len(dataset)` (`train.py:30-33`) requires more than 64 utterances, so a 50-utterance test fails before training unless the config or the test changes. |
| E10 | 200-step timing run to measure seconds per step | KAGGLE_WORKFLOW / Gate 4 | Missing | Not run. |
| E11 | Seed everything for reproducibility | RULES_FOR_AI §5 | Missing | No seeds in `train.py`; DataLoader shuffling and DataParallel are unseeded. |
| E12 | Training command and hyperparameters documented | PDF proxy | Partial | `README.md:60-65` and the configs; the step cap, batch size and hardware are not yet decided. `total_step: 900000` (`English/config.yaml:86`) is the reference default, not a measured choice. |
| E13 | Checkpoint files over 100 MB go to the Hugging Face Hub, or use Git LFS/external links | RULES_FOR_AI §6 / PDF | Not due yet | No checkpoint exists. From the measured 35,159,617 parameters, float32 weights alone are about 134 MiB, so each checkpoint will exceed 100 MB before optimizer state is added. The PDF allows Git LFS or an external link in the README; the link must remain accessible during evaluation. |

### F. Inference

| ID | Requirement | Source | Status | Evidence / gap |
|---|---|---|---|---|
| F1 | Generate all five English test sentences | PDF proxy | Blocked | Every `final` is null, the loader refuses, no step checkpoint exists, and `g2p_en` is absent locally. |
| F2 | Generate all five Telugu test sentences | PDF proxy | Blocked | Null finals and no checkpoint, plus the missing Telugu frontend (D3-D6). |
| F3 | Save audio as `<ID>.wav` | DELIVERABLES_CHECKLIST | Met (code only) | `inference.py:130-131`; no file exists. |
| F4 | Save into `Results/<Language>/test_generated_audios/` | DELIVERABLES_CHECKLIST | Met (code only) | `config.yaml` inference `result_path`; `os.makedirs` at `inference.py:104`. |
| F5 | Choose which speaker embedding to use | DATA_STRATEGY gate | Blocked | `speaker_id: 0` (`English/config.yaml:96`, `Telugu/config.yaml:98`) is labelled a placeholder; the speaker strategy is not decided. |
| F6 | Pitch/energy/duration controls | FastSpeech 2 | Partial | Controls are passed through (`inference.py:118-124`), but at phoneme level the energy embedding uses `p_control` instead of `e_control` (`model/modules.py:124,146`). This is inherited from the reference at the same commit, confirmed from the upstream file; it means `energy_control` has no effect in the current configuration. |
| F7 | Use the allowed pretrained HiFi-GAN vocoder | RULES_FOR_AI §7 | Met (code only) | `utils/model.py:42-71` loads `hifigan/generator_universal.pth.tar`; the zip must be extracted first, and the code is not exercised. |
| F8 | Save predicted duration, pitch and energy arrays and normalised text | PLAN_PHASES Phase 6 / REPORT_OUTLINE | Missing | `inference.py:97-132` writes only the WAV; it does not retain any predictor outputs. |
| F9 | Clear failure when prerequisites are missing | Robustness | Met | `inference.py:141-165` refuses null finals, a missing step, a missing checkpoint and a missing unzipped vocoder with specific messages. |
| F10 | End-to-end inference verified | Deliverables | Missing | Never run; zero WAV files. |

### G. Variance-adaptor analysis (duration, pitch, energy)

| ID | Requirement | Source | Status | Evidence / gap |
|---|---|---|---|---|
| G1 | Duration per symbol as a labelled bar plot | PLAN_PHASES Phase 6 | Missing | No code. `utils/tools.py:109-161` draws one combined mel/pitch/energy figure for training samples, not a duration bar chart. |
| G2 | Predicted pitch (F0 in Hz), denormalised, at the configured level | PLAN_PHASES Phase 6 | Missing | No code. `plot_mel` overlays pitch on a spectrogram for a training sample (`utils/tools.py:213-262`), using targets in `synth_one_sample`. |
| G3 | Predicted energy, denormalised | PLAN_PHASES Phase 6 | Missing | No code. |
| G4 | Save the arrays as JSON next to each plot | PLAN_PHASES Phase 6 | Missing | None. |
| G5 | Exactly 15 plots per language with the agreed names | DELIVERABLES_CHECKLIST | Missing | 0 of 30 exist. |
| G6 | Plot functions tested on dummy data and file names checked | RULES_FOR_AI §11 | Missing | No plotting tests. |
| G7 | Plots must show predictions, not ground truth | PLAN_PHASES Phase 6 | Missing | The model outputs `p_predictions`, `e_predictions`, `d_rounded` and `mel_lens` (`model/fastspeech2.py:99-110`), but inference discards them. |

### H. README, technical report and reproducibility

| ID | Requirement | Source | Status | Evidence / gap |
|---|---|---|---|---|
| H1 | Report covers implementation, data, losses, listening, variance, comparison, reproducibility | DELIVERABLES_CHECKLIST | Missing | PDF does not exist; only the outline `docs/REPORT_OUTLINE.md`. |
| H2 | Loss-curve analysis (convergence, instability, overfitting) | DELIVERABLES_CHECKLIST | Blocked | Requires real training and saved curves; none exist. |
| H3 | Listening observations per test sentence | DELIVERABLES_CHECKLIST / RULES_FOR_AI §2 | Blocked | Human-only task; no audio exists. The assistant must not invent quality claims. |
| H4 | Variance-plot interpretation per sentence | DELIVERABLES_CHECKLIST | Blocked | Requires G1-G5. |
| H5 | English vs Telugu comparison tied to the sentences, including limitations, failures and possible improvements | DELIVERABLES_CHECKLIST / PDF (§9.3 #6) | Blocked | Requires results for both models. |
| H6 | Reproducibility details: dependencies, configs, commands | PDF proxy | Partial | README has setup/smoke/training/inference commands; missing export, `.lab`, alignment and plot commands, and the dependency set is incomplete (A7). |
| H7 | `CHANGES.md` lists every modification | RULES_FOR_AI §12 | Met (for Phases 0-1) | 19 rows, each with a verification column that honestly says "not run" where applicable. Future changes are not yet logged. |
| H8 | AI-assistance disclosure in README | RULES_FOR_AI §12 | Met | `README.md:89-93`. |
| H9 | README accurately reflects the implementation | Audit task | Met (as a status statement) | `README.md:11-13` says no model trained, no audio, no results; that matches the repository. It is incomplete for submission but not misleading. |

### I. Environment, secrets and limits

| ID | Requirement | Source | Status | Evidence / gap |
|---|---|---|---|---|
| I1 | No long training locally | RULES_FOR_AI §3 | Met | Only forward-pass smoke tests and unit tests were run. |
| I2 | Heavy work on Kaggle | RULES_FOR_AI §3 | Partial | Guide exists only for Phase 1 (`kaggle/INSPECT_GUIDE.md`); no Stage A/B/C notebook or exact commands for export/alignment/training. |
| I3 | Tokens from environment or Kaggle Secrets, never committed | RULES_FOR_AI §6 | Met (local state) | Inspector reads `HF_TOKEN`; `.gitignore` excludes `.env`/token files; no token-like string found. |
| I4 | No datasets, raw audio or preprocessed features in the repo | RULES_FOR_AI §6 | Met (local state) | No `data/` directories; `.gitignore` covers them and `*.npy`/`*.TextGrid`. |
| I5 | No file over 100 MB | RULES_FOR_AI §6 | Met (local state) | Maximum 51,942,859 B. Future checkpoints will exceed this (E13). |
| I6 | Pin or record versions of Python, PyTorch and key libraries | RULES_FOR_AI §3 | Partial | `requirements.txt` now pins the non-torch stack and targets Kaggle Python 3.12. PyTorch is supplied by Kaggle and must be recorded from the run. The pinned audio stack was executed in an isolated Python 3.11 environment; the full Kaggle install is still untested. |

## 3. Implementation defects and missing features

Severity: **Blocker** means the next phase cannot succeed without it; **High** means a required deliverable or a correctness property fails; **Medium** is a robustness/reproducibility problem; **Low** is cleanup.

### D1. Blocker: the assignment PDF is unavailable

**Resolved 2026-11-02.** The PDF was supplied at
`docs/FastSpeech2_Assignment (1).pdf`. Section 9 records the comparison and
the differences from this audit. No PDF-only deliverable was found missing
from section 2; the differences are listed there.

### D2. Blocker: the Telugu frontend is the English frontend

- `Telugu/text/symbols.py` is byte-identical to `English/text/symbols.py` (ASCII, ARPAbet, pinyin).
- `Telugu/config.yaml:21` uses `basic_cleaners`; Telugu texts are not transliterated and are not mapped to phones.
- `text_to_sequence` silently drops every symbol not in the table (`text/__init__.py:66-75`): a full Telugu sentence becomes 5 tokens (spaces/punctuation). Training would receive near-empty text sequences, and inference raises `NotImplementedError` (`Telugu/inference.py:67-77`).
- There is no Telugu number-to-words normaliser for T4, no Unicode normalisation policy, and no symbol-set decision.

### D3. Blocker: no corpus export or forced-alignment pipeline

`preprocess.py` only calls `Preprocessor.build_from_path()` (`preprocess.py:21-23`), which expects:

- `data/raw/<speaker>/<id>.wav`;
- `data/raw/<speaker>/<id>.lab`;
- `data/preprocessed/TextGrid/<speaker>/<id>.TextGrid` with a `phones` tier.

Nothing in the repository downloads the Hugging Face datasets, selects clips, writes WAV/lab pairs, resamples to 22.05 kHz, runs MFA, or creates TextGrids. `grep` found no MFA or download code. Therefore `preprocess.py` cannot run on either dataset today, regardless of the frontend.

### D4. Blocker: preprocessing is incompatible with the pinned librosa version

`requirements.txt:25` pins `librosa==0.10.2.post1`. In that release:

- `librosa.util.pad_center` is `pad_center(data, *, size, ...)` (keyword-only `size`);
- `librosa.filters.mel` is `mel(*, sr, n_fft, ...)` (all keyword-only).

The copied code calls them positionally:

- `audio/stft.py:42`: `pad_center(fft_window, filter_length)`;
- `audio/stft.py:145-152`: `librosa_mel_fn(sampling_rate, filter_length, ...)`.

`TacotronSTFT.__init__` therefore raises `TypeError` under the pinned version, before any audio is processed. This affects both languages.

### D5. Blocker: the STFT hardcodes CUDA, so Kaggle CPU preprocessing fails

`audio/stft.py:68-69` calls `.cuda()` on the input and on `forward_basis` inside `STFT.transform`. On a CPU-only Kaggle session (the workflow that `docs/KAGGLE_WORKFLOW.md` recommends for data preparation), this raises a CUDA error. The code must either be made device-agnostic or preprocessing must be documented and run on a GPU session. This is inherited from the reference, but it conflicts with the planned workflow.

### D6. Blocker: selection, split, manifests and overlap checking do not exist

- `preprocessor.py:142` shuffles with no seed and splits by a clip count `val_size` (`:146-151`).
- There are no manifests, no `check_no_overlap.py`, and no `scripts/select_data.py` even though both configs mention it.
- The required ~5 h/1 h proportions cannot be controlled, and identical transcripts across splits are not checked.
- Speaker coverage of validation speakers cannot be enforced.

### D7. High: the preprocessor can crash or silently mis-count clips

At `preprocessor.py:77-90`:

- If the first clip has no TextGrid, `pitch` is unbound at `if len(pitch) > 0` (line 85) and `build_from_path` raises `NameError`.
- If a later clip has no TextGrid, the previous clip's `pitch`/`energy` are fitted into the scaler again and `n_frames += n` repeats the previous clip's frame count.
- Clips without a TextGrid are dropped with no count; clips returning `None` are skipped with `continue` (`:79-80`).
- `random.shuffle(out)` at line 142 is unseeded.
- `val_size` larger than the number of surviving clips would produce an empty training file, with no guard.

### D8. High: English inference bypasses the configured text normalisation

`inference.py:38-64` sends the raw sentence straight to the lexicon and `g2p_en`; it never applies `english_cleaners`. The local probe shows `english_cleaners` turns `10:45` into `ten:forty-five` while retaining the colon. The token `ten:forty-five` is not in the lexicon. `g2p_en` is not installed locally, so whether it pronounces or drops that token is unverified. E4's digits, time and date are therefore a known risk. The normalised text is also not saved for the report, contrary to `RULES_FOR_AI.md` §8.

### D9. High: no CSV loss logs and no loss-plot generation

`train.py:100-112` appends a human-readable `log.txt` and writes TensorBoard scalars. There is no CSV, and no script produces `Results/<Language>/training_loss.png` or `validation_loss.png`. TensorBoard logs are git-ignored, so the evidence for the required plots is not reproducible from the repository as it stands.

### D10. High: no variance-adaptor export or plotting pipeline

`inference.py:97-132` writes only the waveform. The model returns predicted pitch, energy, rounded duration and mel lengths (`model/fastspeech2.py:99-110`), but none are retained. `utils/tools.py:109-161` produces a single combined mel+pitch+energy figure for a *training* batch, not three per-sentence plots, and it uses targets in `synth_one_sample`. There is no duration plot and no JSON export. **0 of the 30 required plots can currently be generated.**

### D11. High: the 50-utterance overfit test cannot start with the current config

`train.py:30-33` asserts `batch_size * group_size < len(dataset)`. With `batch_size: 16` and `group_size = 4`, the dataset must exceed 64 utterances. The planned overfit test uses about 50, so it fails at the assert before training. Gate 4 must adjust the batch size, group size or the assert, and record the change.

### D12. Medium: reproducibility is incomplete

- No seeds in `train.py` or `preprocessor.py` (B7).
- Speaker IDs follow `os.listdir` order (`preprocessor.py:67`).
- `utils/model.py:20` loads the checkpoint without `map_location`.
- Resume restores model and optimizer but not RNG state or loader order (`train.py:62-74`, `utils/model.py:15-30`).
- `quit()` at `train.py:165` may skip the final save if the Gate-4 step cap is not a multiple of `save_step`.
- There is no record of the exact Kaggle torch version.

### D13. Medium: `energy_control` is ignored at phoneme level

`model/modules.py:124` and `:146` pass `p_control` to `get_energy_embedding`; `e_control` is never used in the phoneme-level path. This is identical to the reference commit (confirmed from the upstream file), so it is not a regression, but the assignment's inference controls should be correct or the limitation documented. It only matters if the report claims energy control was varied.

### D14. Medium: no length or memory guard, and reference-scale step/checkpoint settings

- No maximum duration or symbol count is filtered before training (C11). Svarah's mixed spontaneous speech may contain long clips.
- `total_step: 900000` and `save_step: 10000` are reference-scale defaults. From 35,159,617 parameters, weights alone are about 134 MiB per checkpoint; with Adam state each file is larger. Keeping many checkpoints on a Kaggle disk is a storage risk. Gate 4 must set both values from the measured step time and disk budget.

### D15. Medium: `requirements.txt` is incomplete and unvalidated

Missing from the pin set:

- `datasets` and `huggingface_hub` (imported inside `kaggle/inspect_datasets.py:75` and `:152-157`);
- `nltk` (a runtime dependency of `g2p_en`);
- `pytest` (used for the tests);
- an explicit torch record for the Kaggle environment;
- any mention of the Montreal Forced Aligner as an external tool and its version.

The header itself says the set has not been installed end to end. The librosa incompatibility in D4 is exactly the kind of failure that install would have caught.

### D16. Medium: required tests are missing

`RULES_FOR_AI.md` §11 asks for tests of:

- train/validation overlap (clip ID and transcript);
- config loading for both languages;
- text-normaliser examples including E4/T4;
- plot functions on dummy data and their file names.

Only the smoke tests (`tests/smoke_test.py`) and the inspector helper tests (`kaggle/tests/test_inspect_utils.py`, 10 passing) exist. None of the four required test groups is present. The smoke test checks shapes, not training correctness.

### D17. Medium: configuration contains unresolved placeholders that look final

- `scripts/select_data.py` is referenced in both configs but does not exist.
- `val_size: 512` is a clip count presented as the validation split; it is not tied to one hour.
- `speaker_id: 0` is a placeholder in both inference sections.
- `restore_step: null` must be filled before inference.
- `multi_speaker: true` is set for both languages before the speaker strategy is decided.

These are documented as provisional in comments, but any reader can mistake them for final choices; Gate 1 must replace them with measured values.

### D18. Low: stale `__main__` blocks in copied files

`dataset.py:201-257` imports `utils.utils` (a module that does not exist), old `config/LJSpeech/*.yaml` paths and `yaml`. `evaluate.py:89-120` parses the old three-YAML CLI. Neither block is used by `train.py` or `inference.py`, but both are misleading and will fail if someone runs the files directly. They were inherited from the reference and should be removed or clearly marked.

### D19. Low: `THIRD_PARTY_NOTICES` and licence details are partly unverified

- The HiFi-GAN generator is redistributed from the reference repository; the licence of the *pretrained weights* (as opposed to the code) is not independently verified.
- The LibriSpeech lexicon file is redistributed; its provenance/licence is not stated.
- IndicTTS-HF licence terms are not recorded yet; Svarah's CC BY 4.0 is from the card only.

The assignment allows a pretrained vocoder, but the report must name it and its settings; the licence note should be completed before publication.

### D20. Low: README and report are honest but incomplete

`README.md` correctly says Phase 0/1 only, and `docs/dataset_report.md` correctly refuses to invent numbers. However, `README.md` does not yet contain the data-preparation, alignment, plotting or checkpoint commands, and `Report/Technical_Report.pdf` does not exist. The report outline is a plan, not a result; it cannot be checked against an implementation until the report is written.

## 4. Prioritized fixes

### P0 - before any real preprocessing or training

1. **Obtain and check the assignment PDF.** Compare section 2 with it; correct any requirement this audit could not see. This is a human action and gates the rest.
2. **Run Phase 1 on Kaggle and record real numbers** (`docs/dataset_report.md`), then make the Gate 1 decisions: speaker strategy, Telugu aligner, Telugu symbol set, vocoder confirmation, subsampling criteria. Do not choose these from the card facts in `docs/DATA_STRATEGY.md`.
3. **Implement the Telugu frontend** after Gate 1: Unicode policy, symbol set, number/date/time normaliser, text-to-symbol mapping; add a test for T4-style input. Without this, Telugu training is meaningless (5 tokens per sentence today).
4. **Build the corpus and alignment pipeline**: export selected clips as 22.05 kHz mono WAV plus `.lab` under `data/raw/<speaker>/`, run the chosen aligner, and place TextGrids where `preprocessor.py:74-76` expects them. Count clips dropped at every step.
5. **Implement Phase 2 selection/split**: `manifests/train.csv`, `val.csv`, a reproducing script, `check_no_overlap.py`, fixed seed, and speaker coverage of validation. Enforce ~5 h/~1 h and report achieved hours.
6. **Fix preprocessing before using it**: correct the librosa calls (D4), make the STFT device-agnostic or document a GPU-only path (D5), handle missing TextGrids safely (D7), use deterministic speaker order, drive the split from manifests instead of an unseeded shuffle, validate that every text sequence is non-empty, and log all dropped clips.
7. **Fill every `final` test sentence from the rendered PDF** (human). Keep `docs/test_sentences.json` otherwise unchanged; rerun the loader check. This does not block training but blocks inference and the report.

### P1 - required for the deliverables

8. **Add CSV loss logging and a loss-plot script** for `training_loss.png` and `validation_loss.png`; keep TensorBoard as a secondary view.
9. **Add the variance export and plot pipeline**: predicted duration per symbol, predicted F0 in Hz and energy (denormalised with `stats.json`), save arrays as JSON, write 15 named plots per language, and test the plot functions on dummy data.
10. **Complete inference**: Telugu frontend, English normalisation path, save the normalised text and predicted arrays, and produce five WAVs per language. Verify the null refusal with real finals and a real checkpoint.
11. **Harden training**: seeds, `map_location`, final checkpoint, a tested resume, overfit-friendly batch settings (D11), a length filter, and a Gate-4 step/save plan that fits the disk and time budget.
12. **Complete the dependency and Kaggle documentation**: add `datasets`, `huggingface_hub`, `nltk`, `pytest`, record the Kaggle torch version, and write the exact export/alignment/preprocess/train/inference commands.

### P2 - correctness/cleanup before submission

13. Fix or document the inherited `e_control` behaviour (D13).
14. Remove or mark the stale `__main__` blocks (D18).
15. Verify and record licences for IndicTTS, the HiFi-GAN weights and the LibriSpeech lexicon (D19).
16. Update README with real results, checkpoint links and the AI-assistance note; write `Technical_Report.pdf` from saved evidence only.
17. Initialise git only when the human approves; then check file sizes and secrets before the first commit.

## 5. Exact next steps required before training on Kaggle

The order matters. Steps 1-2 are human inputs; steps 3-8 are code and documentation changes; step 9 measures before any long run.

1. **Human: supply or confirm the PDF.** If the PDF differs from the docs, update the affected requirements before proceeding. This audit is not a substitute for that comparison.
2. **Human: Phase 1 inspection and Gate 1.** Follow `kaggle/INSPECT_GUIDE.md`: accept the Svarah conditions with the same account as the token, add `HF_TOKEN` as a Kaggle secret, enable Internet, use a CPU session, run `python kaggle/inspect_datasets.py --dataset svarah` and `python kaggle/inspect_datasets.py --dataset indictts_telugu`, then commit the generated `docs/dataset_report*.json` and `docs/dataset_report.md`. Decide speaker strategy, Telugu aligner, Telugu symbols, vocoder settings and subsampling from the measured report.
3. **Human: fill the ten `final` sentences** from the rendered PDF and confirm the hash change is intentional. Do not edit any other field.
4. **Implement Phase 2** (selection, manifests, `check_no_overlap.py`, fixed seed, hours report). Gate 2 requires the achieved hours and the overlap check output.
5. **Implement Phase 3** (Telugu frontend, export, `.lab`, alignment, preprocessing fixes D4-D7). Gate 3 requires a few human-checked alignments and a report of dropped clips.
6. **Implement the logging/plot/inference gaps** (P1 items 8-10) enough to run the overfit and timing tests; the final plots can wait for the final checkpoint.
7. **Update `requirements.txt` and the Kaggle guide**, install the set on Kaggle, and record the exact versions. The librosa and CUDA defects must be fixed before this install test is meaningful.
8. **Run the CPU smoke tests again** and the new unit tests (overlap, configs, normalisers, plots) locally. Do not run preprocessing locally.
9. **Gate 4: overfit 50 utterances and a 200-step timing run on Kaggle.** Record loss behaviour, whether audio is produced, seconds per step, peak memory and disk use. Only then set `total_step`, `save_step` and the checkpoint-retention plan, and start full training.

Do not start full training before steps 1-9 are complete. Steps 1-3 can proceed in parallel with coding, but the Telugu frontend and the corpus/alignment pipeline depend on Gate 1.

## 6. What could not be verified

- The original assignment PDF: not present on the machine.
- Real dataset columns, sampling rates, hours, speakers, durations, transcripts, licence and gating: no Kaggle run and no HF token; `docs/dataset_report.md` is a stub.
- End-to-end preprocessing: local Python lacks librosa; the pinned librosa API is incompatible with the copied STFT code (D4); the STFT also requires CUDA (D5).
- English inference: `g2p_en` is not installed locally, so the OOV/number path is unverified; no checkpoint exists.
- Telugu inference: frontend not implemented; no checkpoint; null finals.
- Training, validation, resume and checkpointing: no data, no GPU, no run. Only code inspection and the random-tensor smoke test.
- Loss curves, convergence, overfitting and audio quality: no training and no audio. These must not be claimed in the report until measured or heard by the human.
- The dependency pin set end to end: never installed as a set.
- Checkpoint size: estimated from the measured parameter count, not from a saved checkpoint.
- Dataset and vocoder licences: not independently checked.

## 7. Stop point

This audit makes no code changes beyond this document and the `CHANGES.md` entry. Nothing was trained, generated, committed or pushed. The next action is human review of this document, the PDF comparison, and the Gate 1 decisions; implementation resumes only after approval.

## 8. Implementation update (2026-11-02, after audit approval)

Only the requested pre-training infrastructure fixes were implemented. No
dataset preparation, preprocessing run or training was performed.

Resolved or partially resolved:

- **D2 (Telugu frontend):** replaced the English ARPAbet frontend with a
  character-level, NFC-normalized Telugu frontend (`Telugu/text/telugu.py`,
  `Telugu/text/symbols.py`, `telugu_cleaners`). Unsupported characters raise
  with their U+ code; alignment tokens must map one character per phone.
  Covered by `Telugu/tests/test_telugu_frontend.py` (22 tests).
- **D4 (librosa API):** `pad_center` and `librosa.filters.mel` now use keyword
  arguments in both copies of `audio/stft.py`; the third positional
  `pad_center` call in `audio/audio_processing.py:57` was fixed too.
- **D5 (CUDA):** `STFT.transform`/`inverse` and the mel basis follow the input
  device, so preprocessing runs on CPU. `tests/test_audio_cpu.py` exercises
  this path, but it was **skipped locally** because librosa is not installed.
- **D6 (split/overlap):** added `scripts/split_dataset.py` (deterministic,
  duration-targeted, transcript-atomic, speaker-coverage preserving) and
  `scripts/check_no_overlap.py`. The corpus export/alignment pipeline (D3)
  still does not exist, so the manifests cannot be built from the real
  datasets yet.
- **D7 (preprocessor):** replaced the random shuffle and `val_size` split with
  manifest order. Missing files, unreadable TextGrids, empty labels, invalid
  spans, unreadable audio, too few voiced frames, frontend length mismatches
  and metadata separators are logged to `preprocessing_log.csv` with reasons.
  The run aborts before normalisation if any selected clip fails, so reruns
  cannot double-normalise features. Speaker IDs are sorted deterministically.
  `preprocessor/manifest.py` is unit-tested without the audio stack.
- **D11 (overfit assertion):** `train.py` now requires `batch_size <= len(dataset)`,
  so a 50-utterance overfit run is no longer blocked by the `16 * 4` assertion.
  The overfit run itself has not been executed.
- **D12 (partial):** manifest splitting is seeded via SHA-256 and speaker IDs
  are deterministic. `train.py` still does not seed Python/NumPy/PyTorch, and
  checkpoint resume still does not restore RNG/loader state.
- **D17 (partial):** `config.yaml` now has real manifest paths and validated
  fields; `val_size` was removed. `restore_step: null`, `speaker_id: 0` and
  null test sentences remain deliberate placeholders.
- **Configuration validation:** `utils/config.py` now rejects missing sections
  and fields, unknown cleaners, language/cleaner mismatches, absolute paths,
  invalid audio/STFT/mel ranges, non-divisible attention dimensions, invalid
  optimizer/step values and invalid inference controls before preprocessing or
  training. Covered by 23 tests per language.

Still open (unchanged by this update): D1 (PDF unavailable), D3 (dataset
export/alignment), D8-D10 (inference normalisation, CSV/plots, variance
figures), D13-D16, D18-D20, actual preprocessing, training and generated
audio.

## 9. PDF verification (2026-11-02)

### 9.1 Source and method

- File found: `docs/FastSpeech2_Assignment (1).pdf` (3 pages, 272,293 bytes).
  The requested `docs/source/` directory does not exist. `docs/PROBLEM_STATEMENT.md:3`
  names a different filename (`FastSpeech2_Assignment.pdf`) and does not record
  this path.
- Extraction: `pdftotext -layout` (English) and `pypdf 6.17.0` (English and
  Telugu). No PDF page renderer (`pdftoppm` or Ghostscript) is installed on
  this machine, so the Telugu strings below are **text-layer evidence only**,
  not a visual confirmation.
- The PDF was not modified. `docs/test_sentences.json` still has the same
  SHA-256 as the Phase 0 baseline:
  `420187d15f4f9c4a61202ce8d6c3b45f72fbb4535c8f3be0689c01cf60847a6f`.
  Every `final` remains `null`.

### 9.2 Requirements confirmed by the PDF

The PDF confirms the following rows that this audit previously marked "PDF
proxy": the Svarah/IndicTTS datasets (B1), approximately 5 h training and 1 h
validation per language (B2), no train/validation overlap (B3), a documented
selection/subsampling strategy (B5), separate models and configuration files
(A2-A4), the exact required folder tree (A1), five test sentences per language
(A11/A12), preservation of the sentence text (D8), the five-sentence audio
requirement (A11), the 15 variance plots (A12), all seven submission items, and
the seven technical-report contents (H1-H6). No PDF-only deliverable was found
that is absent from section 2.

The PDF lists four IDs per language, with E2 and T2 split into `a)` and `b)`.
That is five sentences per language, matching the five audio files and fifteen
plots already required by `DELIVERABLES_CHECKLIST.md`. The `E2a/E2b/T2a/T2b`
file naming is our own convention, not a PDF requirement.

### 9.3 Differences and corrections

| # | Audit statement | What the PDF actually says | Effect |
|---|---|---|---|
| 1 | Section 0 and D1: "the assignment PDF is unavailable". | The PDF was supplied at `docs/FastSpeech2_Assignment (1).pdf`, not `docs/source/`. | D1 is **resolved**. Section 0 is historical; this section supersedes it. |
| 2 | No deadline row. | "Deadline: 07-11-2026"; the docs add that late submissions are not accepted. | Add a schedule constraint: five calendar days from 2026-11-02. Not a deliverable defect. |
| 3 | A4 source: "PDF proxy (PROBLEM_STATEMENT)" for "trained from scratch, no pretrained acoustic weights". | The PDF requires two separate models trained on the specified datasets and use of the GitHub implementation as a starting point. It does **not** say "from scratch" or "no pretrained acoustic weights". | Correct the source to `RULES_FOR_AI.md` (and `PROBLEM_STATEMENT.md` hard constraints, which misattribute it to the PDF). The rule itself still applies. |
| 4 | A5: "Google Form submission". | The PDF says "Submit the GitHub repository URL through the assignment submission form". The Google Form URL is in `PROBLEM_STATEMENT.md`, not the PDF. | Correct the attribution; the action is unchanged. |
| 5 | H1/H6 do not require citing the paper. | The PDF says "referring to the original research paper" and lists the FastSpeech 2 paper as a reference. | Report must cite arXiv:2006.04558, not only the GitHub implementation. |
| 6 | H5: "English vs Telugu comparison tied to the sentences". | The PDF requires comparative observations "including limitations, failures and possible improvements based on the test sentences". | Expand H5 to name limitations, failures and improvements explicitly. |
| 7 | Stored E3 is `The rhyth m ...` (one space); the JSON `raw_extracted` field is unchanged. | The PDF text layer is `The rhyth  m ...` (two spaces between "rhyth" and "m"). | Visual check still required; do not edit the sentence or the JSON from this extraction. |
| 8 | D4/D7: Telugu test strings are unavailable/corrupted; all finals null. | `pypdf` extracts readable Telugu, but with spacing defects: T1 has doubled word spaces; T2b is `వచ్చ ింది`; T3 is `అడ  విలో నిశ్శ బ్దంగా`; T4 is complete with `2026`, `12వ`, `10:45`. The stored `raw_extracted` strings are more corrupted (for example T1 splits the first word and T3 has `నిశ్శబ్ంద`). | Do not copy either extraction into `final`. A Telugu reader must confirm from the rendered page. The null-refusal behaviour (D7) remains correct. T4 confirms the need for a Telugu number/date/time/ordinal normaliser. |
| 9 | E13: checkpoints over 100 MB go to the Hugging Face Hub. | The PDF says to use Git LFS **or** an accessible external download link in the README if checkpoints exceed GitHub's file-size limits, and the link must stay accessible during evaluation. | The Hub is one allowed option, not a PDF requirement. Add Git LFS and link-accessibility checks to the submission checklist. |
| 10 | A15: `tests/`, `lexicon/`, `kaggle/` are not in the explicit extras list. | The PDF specifies the required tree; it does not enumerate allowed extras. | Gate 0 approval (2026-11-02) keeps the extra subfolders and both HiFi-GAN copies. A15 is now **Met by human decision**, not by the PDF. |
| 11 | Several rows cite the PDF for vocoder, alignment, Telugu-symbol or 50-utterance requirements. | The PDF does not mention HiFi-GAN, 22.05 kHz, MFA, Telugu symbols, or an overfit test. | Those sources remain `RULES_FOR_AI.md`, `DATA_STRATEGY.md`, `KAGGLE_WORKFLOW.md` and `PLAN_PHASES.md`; they are internal project decisions, not PDF requirements. Do not describe them as PDF-mandated in the report. |

### 9.4 Status changes after the comparison

- **D1: resolved** (PDF available).
- **A4: source corrected**; status remains Partial because no model is trained.
- **A5: wording corrected**; status remains Missing (no public repository/form
  submission yet).
- **A15: Met** by Gate 0 human approval (extras retained).
- **H5: expanded** to require limitations, failures and improvements; status
  remains Blocked until results exist.
- **H1/H6: expanded** to require the FastSpeech 2 paper citation; status
  remains Missing/Partial.
- **E13: expanded** with the Git LFS/external-link alternative and the
  requirement that the link remains accessible during evaluation.
- **A7/I6: still Partial.** `requirements.txt` now pins a coherent Kaggle
  3.12 set (NumPy 2.0.2, SciPy 1.13.1, librosa 0.10.2.post1, numba 0.60.0,
  pyworld 0.3.5, tgt 1.5) and adds the missing `datasets`, `huggingface_hub`,
  `nltk` and `pytest` pins. The pyworld source build and the full set have not
  yet been installed on Kaggle.

### 9.5 Still a human action

1. Open the rendered PDF and fill every `final` in `docs/test_sentences.json`.
   Check E3's gap and have a Telugu reader check T1-T4 character by character.
   Do not edit any other field; rerun the null-refusal check afterwards.
2. Confirm whether the assignment form is the Google Form in
   `docs/PROBLEM_STATEMENT.md` or a different form; the PDF does not name it.
3. Confirm the deadline timezone; the PDF only gives the date.

## 10. Items 2-3 status update (2026-11-02)

This section updates only the rows affected by the training-control and
preprocessing work. No dataset, training run or checkpoint exists.

| Row | Previous | Now | Evidence |
|---|---|---|---|
| E5 (CSV loss logs) | Missing | **Met (code)** | `utils/train_utils.append_loss_csv`; `logs/train/train_loss.csv`, `logs/val/val_loss.csv`; 6 tests |
| E7 (resume) | Partial | **Met (code)** | `--resume` / `--restore_step`, `map_location`, Python/NumPy/PyTorch RNG restore; load/save tests. Loader restarts the epoch, so intra-epoch batch order is not replayed |
| E8 (final checkpoint) | Partial | **Met (code)** | Final save at `total_step` even when it is not a multiple of `save_step` |
| E9 (50-utterance overfit) | Missing | **Met (code only)** | `batch_size` and `group_size` come from `train.optimizer`; `test_overfit_batches.py` batches 50 clips at 8x1. The overfit *run* is still not executed |
| D11 (overfit assertion) | Resolved | **Met (code)** | Assertion now only requires `batch_size <= len(dataset)`; configurable settings tested |
| Rule: no silent skips | Partial | **Met (code)** | Every skip has a category; strict abort by default; `--allow-dropped` is explicit; `preprocessing_summary.json` counts drops |
| B6 (manifests) | Missing | **Missing** | No corpus; item 4 will export data and build measured manifests |

Verification: English 56 tests and Telugu 78 tests pass in the pinned Python
3.11 environment (Telugu skips the English-only lexicon test); 23 script tests
pass; `compileall` exits 0; both smoke tests pass (35,159,617 and 35,099,201
parameters). Preprocessing, training and resume against real data remain
unverified.
