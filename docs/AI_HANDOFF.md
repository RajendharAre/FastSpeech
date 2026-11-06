# Agent handoff — FastSpeech 2 (English + Telugu)

**Read this first if you are an AI agent (Qoder, GitHub Copilot in VS Code, or
anything else) continuing this project.** It exists because the work spans
several tools and sessions and the chat history is not portable. Everything
below is measured, not remembered: re-verify anything marked `UNVERIFIED`
before acting on it.

- Last updated: 2026-11-06 (local time), by Qoder, immediately before the Telugu
  training run was launched.
- Assignment deadline: **2026-11-07 midnight**.
- Local repository: `D:\NAAC\Data Science Projects\FastSpeech 2`
- Public repository: https://github.com/RajendharAre/FastSpeech, branch `main`.
- Reference implementation: https://github.com/ming024/FastSpeech2 (MIT), commit
  `d4e79eb52e8b01d24703b2dfc0385544092958f3`. Keep the attribution.

## 1. Update this file before you hand off

When your session ends, is interrupted, or you run out of context:

1. Add a dated row to `CHANGES.md` (what changed, why, how it was verified).
2. Update section 3 ("state") and section 5 ("next actions") below.
3. Commit and push to `origin main`. The human has authorised commit **and**
   push for this repository, provided the exclusions in section 2 are respected.

If you are starting work and this file is older than the newest `CHANGES.md`
row, trust `CHANGES.md` and re-derive the state from `git log` + the files.

## 2. Standing rules (do not break them)

- Never commit: datasets, raw audio, preprocessed features, checkpoints over the
  GitHub 100 MB limit, `.env`, tokens or personal data. `Qoder_FastSpeech2_History.md`
  (a 9.9 MB chat export at the repo root) and the duplicate `docs/training_loss.png` /
  `docs/validation_loss.png` must stay out of git.
- `docs/RULES_FOR_AI.md` is the assignment's rule file. Section 8 forbids editing
  `docs/test_sentences.json` (overridden once, by the human, on 2026-11-05, for the
  `final` fields only). Section 9 reserves speaker strategy, aligner, symbol set,
  vocoder and subsampling criteria as human decisions — propose, never decide.
- Never start training or change the training design without the human's explicit
  direction. Never stop a running Kaggle GPU session to free the slot.
- One Kaggle account only. No multi-accounting: the human decided to stay serial.
- Never run destructive git commands (`reset --hard`, `clean -f`, `checkout --`)
  inside a Kaggle clone: those trees hold uncommitted, sometimes one-of-a-kind
  edits and logs. Prefer reversible moves (`pruned/`, copies) over deletion.
- Claims relayed from another AI (a code review from ChatGPT/Claude, a suggested
  patch) must be verified against the real code and data before being implemented.
- Work on one fix at a time, verify it, then confirm before moving on.
- Report unverified or failed things honestly. Do not describe the current English
  output as successful speech.

## 3. State as of this handoff

### Code

- Everything is committed and pushed except the changes listed in section 6.
- Both language folders are self-contained copies of the reference with local
  modifications; `CHANGES.md` is the authoritative diff log.

### English — trained, deliverables exist, quality is the open problem

- 60,000 steps completed on Kaggle (T4 x2). Final validation total loss **4.0910**;
  the validation minimum was **3.8246 near step 7,000**.
- Checkpoints held locally: `FastSpeech_Data/english_60000.pth.tar` (418,127,227 bytes)
  and `Downloads/english_35000.pth.tar`. Intermediate checkpoints (5k–30k) exist only
  in the English Kaggle working directory and are **UNVERIFIED / possibly gone**.
- Deliverables present under `Results/English/`: 5 WAVs, 15 variance PNGs,
  5 variance JSONs, `training_loss.png`, `validation_loss.png`. The loss PNGs cover
  only the resumed segment (steps 30,000–60,000) because two overlapping processes
  wrote the earlier logs.
- Inference used checkpoint **35,000**, `speaker_id: 0`, `language: en`, English
  ARPAbet symbols — all verified from the saved JSONs and the console output.
- **The human's listening verdict: the audio sounds like speech but the sentences are
  not accurate.** Improvement is an explicit goal, alongside Telugu.

### Telugu — preprocessed, training starts now

- Preprocessing already complete on Kaggle: 3,269 of 3,363 clips ok (94 dropped:
  alignment 83, missing TextGrid 3, missing audio 8), **2,751 train / 518 val,
  1 speaker, 4.700 h**. Provenance: source is 16 kHz, upsampled to 22,050 Hz.
- `Telugu/config.yaml` now matches English: `total_step: 60000`,
  `anneal_steps: [20000, 26000, 33000]`, `save_step: 2500`, `warm_up_step: 4000`.
  Verified by loading the config: `{'total_step': 60000, 'log_step': 100,
  'synth_step': 1000, 'val_step': 1000, 'save_step': 2500}`.
- Telugu has one speaker, so `speaker_id: 0` is the only valid ID — no GATE decision
  pending there.
- No Telugu results exist yet: `Results/Telugu/` holds only `.gitkeep` files, and
  report sections 9 and 10 are PENDING.

## 4. What the English failure is (measured — do not re-derive)

- Frontend and length regulator are correct: predicted symbol durations sum exactly
  to the reported mel-frame count for all five sentences; mean 5.9–6.7 frames per
  symbol (69–79 ms), a plausible English rate.
- The waveform is not noise: spectral flatness 0.007–0.013 (strongly tonal), no
  clipping, correct mono 22,050 Hz and exact expected length. So `inference.py` →
  `vocoder_infer` shape handling is right (`English/inference.py:320`, output index
  1 = postnet mel, index 5 = rounded durations, 8/9 = lengths).
- The variance adaptor collapses to the mean. Pitch and energy targets are
  std-normalised, so the training distribution has std 1.0; predicted stds are
  pitch 0.24–0.40 and energy 0.32–0.70, and predicted pitch spans −0.9…+1.4 against
  a dataset range of −2.0…+9.1. Predicted energy means are negative (−0.40…+0.13) and
  the WAV RMS is correspondingly low (0.033–0.077).
- The loss curves say the same thing. Pitch/energy/duration losses are **MSE**
  (`English/model/loss.py:71`); mel loss is **L1**. Train pitch/energy ≈ 0.02–0.08,
  **validation pitch 0.68 and energy 1.15** — against a unit-variance target 1.0 is
  the "predict the mean" baseline, so the energy predictor is worse than a constant
  on unseen clips. Validation mel L1 ≈ 1.05 nat vs train ≈ 0.6.
- Root cause: 60,000 steps × batch 16 over 4,500 clips = **213 epochs** of 4.84 h of
  audio with 35,159,617 parameters. The model memorised and did not generalise;
  validation loss bottomed near step 7,000 and rose after that.

## 5. Next actions, in priority order

1. **Telugu training (GPU, in progress at handoff).** Launch detached, monitor RAM,
   and never kill the session to free the slot. Budget ≈ 8.5 h at the English
   measured 0.511 s/step (Telugu step time is uncalibrated — measure it in the first
   300 steps and record it).
2. **Checkpoint selection for Telugu by validation minimum**, not last step. Read
   `logs/val/val_loss.csv` and pick the lowest total; inference from that checkpoint.
3. **Telugu deliverables**: `Results/Telugu/{training_loss.png,validation_loss.png}`,
   5 WAVs and 15 variance PNGs via `python inference.py -c config.yaml --restore_step <best>`.
   Telugu figures need a Telugu font — Kaggle's Linux image has none, so generate the
   variance PNGs there only if a font is installed, otherwise run inference on CPU
   locally where `C:\Windows\Fonts\Nirmala.ttc` exists.
4. **English improvement, CPU only** (safe to run while Telugu trains on GPU):
   - Decisive vocoder test, needs no dataset: run inference locally on CPU with
     `english_35000.pth.tar`, save the predicted mel, then decode that same mel two
     ways — through HiFi-GAN (`utils.model.vocoder_infer`) and through the repo's
     Griffin-Lim (`audio/tools.py:inv_mel_spec`). Intelligible from Griffin-Lim but
     not HiFi-GAN ⇒ mel-domain/scaling mismatch with the universal vocoder, fixable
     in minutes without retraining. Garbage from both ⇒ the model, as section 4 says.
   - Controls sweep on CPU with the same checkpoint: `pitch_control`,
     `energy_control`, `duration_control` multiply the *normalised* predictions, so
     values above 1.0 counteract the collapse measured in section 4. Record what each
     setting sounds like; a documented mitigation is a legitimate result.
   - Compare 35,000 against 60,000 by ear. Both are on disk.
5. **Checkpoint hosting decision (human).** 12+ checkpoints × 418 MB cannot go on
   GitHub. Options: upload the selected checkpoint(s) to Hugging Face and link them,
   or ship only the best English + best Telugu checkpoint. `English/checkpoints/README.md`
   still says `Hugging Face Hub URL: pending`.
6. **Report**: fill section 7 (checkpoint comparison), 8 (listening), 9 (variance
   numbers from the saved JSONs) and 10 (English vs Telugu), then rebuild
   `Report/Technical_Report.pdf` with `python scripts/render_report_pdf.py --check`.
7. **Re-run `scripts/check_no_overlap.py`** against the final Kaggle manifests
   (train/val/test must not share a speaker or a clip).
8. **Telugu duration calibration** for report section 6: replace the PENDING cells
   with the measured s/step and wall clock.

## 6. Uncommitted at handoff (commit these)

- `English/utils/train_utils.py`, `Telugu/utils/train_utils.py`: `_cpu_state`, so a
  GPU-saved checkpoint's RNG ByteTensors are moved back to CPU before
  `torch.set_rng_state`. This was the Kaggle-only patch that blocked the English
  restart at step 30,000; it is now in the repo and covered by two new tests per
  language.
- `English/tests/test_train_utils.py`, `Telugu/tests/test_train_utils.py`: the two
  new RNG tests (9 tests per file, both pass locally).
- `Telugu/config.yaml`: step block and anneal points (section 3).
- `English/utils/model.py`, `Telugu/utils/model.py`, `English/inference.py`,
  `Telugu/inference.py`, `English/checkpoints/README.md`, `README.md`, `CHANGES.md`,
  `Report/Technical_Report.md`, `Report/Technical_Report.pdf`,
  `scripts/render_report_pdf.py`, `scripts/tests/test_render_report_pdf.py`: from the
  2026-11-05 session.
- Untracked deliverables that *should* be committed: everything under
  `Results/English/` except `.gitkeep` siblings (2.8 MB total).
- Untracked files that must **not** be committed: `Qoder_FastSpeech2_History.md`,
  `docs/training_loss.png`, `docs/validation_loss.png`.

## 7. Kaggle layout (so you know where things live)

- Notebook `FastSpeech 2` → Telugu, repository clone at `/kaggle/working/repo`.
- Notebook `FastSpeech 2-English` → English, repository clone at
  `/kaggle/working/FastSpeech`.
- One interactive GPU session per account. Jupyter runs one cell at a time and
  `!cmd &` raises `OSError: Background processes not supported`; launch long jobs
  with `subprocess.Popen([...], start_new_session=True)` or `setsid`.
- An OOM kill takes only the offending process; a detached child survives and the
  container does not necessarily restart (proved from PID 1 and kernel elapsed
  times).
- Training runs with `nn.DataParallel` applied unconditionally, so batch 16 is 8+8
  and both GPUs read 100 % utilisation for a single process.
- Measured English step time: **0.511 s/step** mean over 32,000 steps. The 300-step
  calibration said 0.304 s/step and was optimistic; do not budget from a short
  calibration.
- The English host-RAM ceiling is 30 GiB; the run was OOM-killed at step 32,000. The
  figure leak in `utils/tools.py log()` is fixed (both languages) and `save_step` is
  2,500, which caps a restart at about 21 minutes of lost work.
