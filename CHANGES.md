# CHANGES

Every modification to the reference implementation
(https://github.com/ming024/FastSpeech2, MIT license) is recorded below, newest
first. `Verified` means the thing was checked in the way stated, not that it was
assumed.

Reference commit: `d4e79eb52e8b01d24703b2dfc0385544092958f3` (2021-07-08).

## Current status

English training finished at 60,000 steps and all five test sentences are
synthesised. My detailed listening notes in report section 8.3 describe mostly
unclear clips, with E3 the clearest. I ruled the pipeline out by measurement,
and the cause is memorisation over 213 epochs of 4.84 h of audio (report
sections 7 and 8). Checkpoint 60,000 is the one shipped; I preferred it in an
earlier listening comparison, at 7/10 against 5/10 for step 5,000. Those
aggregate ratings are subjective and the new clip-by-clip notes are also rough
estimates, not a formal listening evaluation.

Telugu training completed at 60,000 steps. Validation loss among saved
checkpoints was lowest at step 5,000 (3.73799); step 60,000 was 3.791061.
Inference at steps 5,000 and 60,000 generated five WAVs and all 15 variance
plots plus five prediction JSONs per checkpoint. The final inference set is in
the required `Results/Telugu/` folders; the 5,000-step comparison is outside
the repository under `../FastSpeech_Data/Telugu/`. The 35,000-step comparison
is intentionally excluded. The console warned that no Telugu font was installed,
so plot labels use code-point fallbacks. My sentence-level listening
observations are recorded in Section 8.2 of the report; I slightly prefer 60,000
over 5,000 but report problems in T1, T2b and T4. This is a qualitative
single-listener review, not an ASR result or formal score.

All eleven report sections are written. Section 9 analyses the Telugu
variance-adaptor predictions, and section 10 compares the two languages and
separates the causes into data, frontend and training budget. The 60,000-step
Telugu inference outputs are present locally and are the only Telugu inference
result presented in the report; the 5,000-step validation loss and prediction
figures are reported as comparisons.

I submitted the repository through the assignment Google Form on 2026-11-10.
The checkpoint download links are in the root README and both checkpoint
READMEs, but the two Drive links currently open folders of extracted archive
fragments rather than loadable `.pth.tar` files; replacing them with the genuine
single files is the one deliverable still open (see the 2026-11-10 log entry).
I did not run Telugu ASR because the English probe cannot transcribe Telugu. The
final train/validation manifests remain outside this repository, so I cannot
rerun `scripts/check_no_overlap.py` against those exact files here.

## Change log

### 2026-11-10

- `docs/DELIVERABLES_CHECKLIST.md` — Submitted the repository through the assignment Google Form. I ticked "Repository is public", "README links work without logging in" and "Repository URL submitted through the Google Form", and left "All code, results, checkpoints (or accessible links) and the report are reachable" unticked because the two hosted checkpoint links do not yet serve loadable weights (next entry).
  Why: The form is in, but the checklist has to reflect what a grader can actually download today, not what the READMEs intend to ship.
  Verified: I confirmed the submission on 2026-11-10. HEAD `9c4cac4` is pushed and the working tree is otherwise clean; the repository and both Drive folder pages resolve signed-out (HTTP 200). The links opening without a login is why that box is ticked; the files behind them being unusable is why the reachability box is not.

- Google Drive checkpoint links (root `README.md` lines 430-431, `English/checkpoints/README.md` line 20, `Telugu/checkpoints/README.md` line 20) — Recorded that both links currently point to Drive *folders* named `english_60000.pth.tar` and `telugu_60000.pth.tar`, each holding the seven files a GUI unzip of a torch archive produces (`.data`, `data`, `data.pkl`, `version`, `byteorder`, `.format_version`, `.storage_alignment`) instead of the single `.pth.tar`. `torch.load` cannot open that layout and a Drive folder-zip does not restore the `archive/` prefix the loader expects, so the hosted checkpoints are not usable as published. Replacing them with the genuine single files is still outstanding.
  Why: A grader who follows the README download link gets fragments, not weights, while the report and both checkpoint READMEs promise a loadable step-60,000 checkpoint. The form is already submitted, but the repository is live, so correcting the links still reaches the evaluators.
  Verified: Fetched both folder pages signed-out on 2026-11-10 (HTTP 200); each lists exactly those seven child names and no `.pth.tar`. The genuine English checkpoint is local at `../FastSpeech_Data/English_Data/english_60000.pth.tar` (418,127,227 bytes, sha256 `d116f2cb74a9e909841c215438a7f70ed119380d2511dce7d6b7396a7265dea3`; the git-ignored copy at `English/checkpoints/60000.pth.tar` matches byte-for-byte and loads as step 60,000 / 235 tensors). The genuine Telugu checkpoint was copied to Kaggle Output `/kaggle/working/telugu_60000.pth.tar` (417,399,099 bytes, sha256 `e37bc8caab40430af080c1da80e6718c73ead1eecd5d3ebcc2f279f74016cac9`, step 60,000 / 235 tensors, measured on Kaggle); it still needs to be downloaded to the PC and uploaded to Drive as a single file.

### 2026-11-07

- `Report/Technical_Report.md`, `Report/Technical_Report.pdf` — Updated the report's opening section map and Section 8 heading to make clear that the section includes English listening observations as well as diagnostics.
  Why: The detailed English listening table had been added, but the opening summary still described Section 8 as English diagnostics only.
  Verified: `python scripts/render_report_pdf.py --check` passes after rebuilding the PDF.

- `Results/English/training_loss.png`, `Results/English/validation_loss.png`, `Report/Technical_Report.md`, `Report/Technical_Report.pdf`, `README.md`, `scripts/plot_training_losses.py`, `scripts/tests/test_plot_training_losses.py` — Added the early loss segment recovered from PID 28091 TensorBoard event files alongside the resumed-run segment. The plots leave the unrecorded intervals blank; the report explains the separate-process provenance and does not present them as one continuous run. The plotter also breaks lines over missing cadence, shades gaps, keeps unique legend labels and rejects duplicate steps.
  Why: The raw CSVs were interleaved by two processes, but the per-process event file preserved monotonically increasing scalar values for the early process. The evaluators should see the available evidence and the missing intervals without mistaking distinct runs for one complete 0–60,000 trajectory.
  Verified: Imported train CSV: 256 early rows (100–25,600, cadence 100) and 300 resumed rows (30,100–60,000); validation CSV: 25 early rows (1,000–25,000, cadence 1,000) and 30 resumed rows (31,000–60,000). Early-event validation minimum is 3.8246 at step 7,000; the resumed-run value at step 60,000 is 4.0910. `python scripts/plot_training_losses.py` regenerated both English plots from the imported CSVs. `python -m pytest scripts/tests/test_plot_training_losses.py -q`: 12 passed. `python scripts/render_report_pdf.py --check`: no problems, 237 Telugu codepoints drawn, all Latin sentences present; PDF 1,199,044 bytes.

- `Report/Technical_Report.md`, `Report/Technical_Report.pdf` (root) — Corrected the reproducibility paragraph and Table 10.1 to match the shipped English step-60,000 payloads: the paragraph no longer says the English payloads are from step 35,000, and the table's English spread rows now read 0.33 pitch and 0.44 energy, the mean per-sentence standard deviations of the shipped payloads, with a caption note that the Section 7.4 Kaggle A/B decode agrees to within 0.01. The prose comparison in Section 10.4 now says 0.599 against 0.33.
  Why: After the English results were regenerated at 60,000, the old paragraph claimed the payloads were the 35,000 export and the table quoted the Kaggle A/B spreads rather than the shipped ones, so the report contradicted the files in `Results/English/`.
  Verified: Recomputed both aggregations from the shipped and the preserved payloads: mean per-sentence std 0.3326/0.4410 at 60,000 and 0.3246/0.4445 at 35,000, pooled std 0.3482/0.4567 and 0.3417/0.4590. The Telugu column's 0.599/0.630 is the mean-per-sentence method, the same one `scripts/summarise_variance_export.py` prints, so the English column now uses that method too. Section 7.4 and the Section 8.3 range keep the Kaggle A/B values on purpose, as the multi-checkpoint comparison record. `python scripts/render_report_pdf.py --check`: no problems, 237 Telugu code points drawn, all Latin sentences present, PDF 1,237,051 bytes. `python -m unittest discover -s scripts/tests`: 56 tests pass.

- `Results/English/`, `../FastSpeech_Data/English_Data/checkpoint_35000/`, `README.md`, `English/checkpoints/README.md`, `docs/AI_HANDOFF.md` — Re-ran English inference at checkpoint 60,000 on CPU and replaced the 35,000-step export in the required result folders, so the published WAVs, plots and prediction JSONs come from the same checkpoint the READMEs ship. Preserved the old 35,000-step set outside the repository before overwriting it.
  Why: The report and both READMEs state that the shipped English audio is checkpoint 60,000, but every JSON in `Results/English/variance_adaptor_plots/` reported `checkpoint_step: 35000`, so a reader who downloaded the shipped weights would regenerate different audio from the one the report describes.
  Verified: The 60,000 checkpoint was already backed up locally at `FastSpeech_Data/English_Data/english_60000.pth.tar`; loading it reports `step: 60000` and 235 model tensors, and `git check-ignore` confirms the copy placed at `English/checkpoints/60000.pth.tar` stays out of Git. The local box lacked `g2p_en`, so I installed it together with its NLTK tagger and cmudict data before the run; no repository code changed. `python inference.py --config config.yaml --restore_step 60000` wrote 5 WAVs, 5 JSONs and 15 PNGs; all five JSONs now report `checkpoint_step: 60000`, the WAVs read back at 22,050 Hz with non-zero peaks, and the mean mel length is 392.8 frames, matching the 393 recorded for step 60,000 in report section 7.4. Mean predicted pitch and energy standard deviations are 0.33 and 0.44 against the report's 0.34 and 0.45, a 0.01 rounding-level difference. The 35,000-step comparison set is intact at `../FastSpeech_Data/English_Data/checkpoint_35000/`.

- `English/checkpoints/README.md`, `Telugu/checkpoints/README.md`, `README.md` (root) — Both checkpoint READMEs rewritten to one shared structure: why the weights are absent, a shipped-checkpoint table, the other checkpoints decoded for the report, the payload and resume command, the config checksum, and what to fill in once the upload is done. The root README's Checkpoints section now carries the same two-row download table instead of claiming the weights are already on the Hugging Face Hub.
  Why: The English README still said "Hugging Face Hub URL: pending" and the Telugu one said "No checkpoint yet: the Telugu model has not been trained (Phase 5)", which is false now that both runs finished at 60,000. The assignment requires the trained checkpoints and permits an external download link, so the READMEs have to say where the link goes and what it is for.
  Verified: `save_step` 2,500 English and 5,000 Telugu, `total_step` 60,000 in both, read from the two `config.yaml` files. `utils/train_utils.py:95-108` builds `{step, model, optimizer, rng_state}` and writes it through a `.tmp` plus `os.replace`; `checkpoint_path` at line 134 gives `<step>.pth.tar`. Both `train.py` files accept `-c` and `--config` (line 243) and both `inference.py` files accept `--config` and `--restore_step`, so every command printed in the READMEs is valid. Config sha256 prefixes computed locally: English `edb1e4dcecba631c` over 4,022 bytes, Telugu `c85ff84ca14777ee` over 4,463 bytes. The ~400 MiB size is an estimate, not a measurement: 35.1 M parameters x 4 bytes x 3 tensors for the weights plus Adam's two moment buffers gives 402.4 MiB English and 401.7 MiB Telugu, and the READMEs show that arithmetic rather than asserting a measured file size. The Telugu selection wording matches the sentence-level review now in report section 8.2, including the reported problems in T1 and T4, rather than the earlier "sounds good".

- `Results/Telugu/`, `../FastSpeech_Data/Telugu/checkpoint_5000/`, `Report/Technical_Report.md`, `README.md`, `docs/AI_HANDOFF.md`, and `scripts/summarise_variance_export.py` - Moved the final Telugu WAVs and variance JSON/plots into the required root result folders with standard filenames; moved the 5,000-step comparison set outside the Git repository. Updated paths and the comparison script.
  Why: Match the assignment's Results folder structure and keep checkpoint-specific filenames out of the final result folders.
  Verified: Final folders contain five WAVs, five JSONs and 15 PNGs; the comparison folder outside the repo contains the 5,000-step 5 WAVs, 5 JSONs and 15 PNGs. The variance summary script reads both locations and completes.

- `Report/Technical_Report.md`, `Report/Technical_Report.pdf` (root) - Recorded the user's sentence-level listening review of the Telugu 60,000-step samples, their slight preference for 60,000 over 5,000, and the limitation that this is a single listener's subjective assessment. Changed the report status to complete for the available project evidence. Corrected the input-length label in Section 8.1, the Telugu pitch-baseline interpretation in Section 7.5, and the statement about split manifests not being included in the repository.
  Why: The assignment's generated-speech observations require per-sentence notes; the report also needed to distinguish completed observations from unavailable Telugu ASR and avoid factual inconsistencies.
  Verified: User-provided notes captured for T1, T2a, T2b, T3, T4 and the 5,000-versus-60,000 audio comparison. Manifests checked with `git ls-files`; only `.gitkeep` placeholders are tracked. The measured Telugu pitch baseline is 0.885, so no listed validation pitch loss is below it.

- `Report/Technical_Report.md`, `Report/Technical_Report.pdf` (root) — Section 10, "English vs Telugu comparison", written in place of the pending block: Table 10.1 side by side, a scale note on which losses are comparable, the parameter difference attributed to the text embedding, and subsections on what the data, the frontend and the training budget each explain, what could not be measured, and a cause-sorted limitations table. The status block and the section map now say all eleven sections are written.
  Why: Section 10 was the last pending section, and the assignment asks it to separate failures attributable to data, frontend and training budget rather than list symptoms.
  Verified: Symbol tables counted by importing both `text/symbols.py` modules, 360 English and 124 Telugu. `transformer/Models.py:40,56` builds `nn.Embedding(len(symbols) + 1, encoder_hidden)` and both configs set `encoder_hidden: 256`, so 361 x 256 = 92,416 against 125 x 256 = 32,000, a difference of 60,416, exactly 35,159,617 minus 35,099,201. `Telugu/dataset.py:43` loads the mel `.npy` with no normalisation and `Telugu/utils/tools.py:38` only casts it to float, so mel L1 is on the same log-mel scale in both languages. Baselines computed as 0.941² = 0.885 and 0.948² = 0.899; every Telugu pitch value in Section 7.5 is above 0.885 and every energy value below 0.899. English phoneme counts 85, 50, 56, 37, 81 read from `Results/English/variance_adaptor_plots/*.json`, whose `text` field is the raw string and whose `checkpoint_step` is 35000 in all five. Telugu spread 0.599 pitch and 0.630 energy from `python scripts/summarise_variance_export.py`. Final validation 3.791061 against the 3.6590 minimum is 3.6 percent. `python scripts/render_report_pdf.py --check`: no problems, 224 Telugu code points drawn, all Latin sentences present, PDF 1,226,188 bytes, 18 pages.

- `scripts/render_report_pdf.py`, `scripts/tests/test_render_report_pdf.py` (root) — Tables of twelve rows or fewer now carry `class="keep-together"` with `page-break-inside: avoid`, so a small table moves to the next page whole instead of leaving orphan rows behind. Longer tables still paginate per row.
  Why: Section 10 pushed Appendix A over a page boundary and the build left a final page holding 97 characters, two table rows and nothing else.
  Verified: `python -m unittest discover -s scripts/tests` passes 56 tests, including a new one asserting a ten-row table gets the class and a twenty-row table does not. The rebuilt PDF's last page holds 781 characters, the end of section 11 plus the whole Appendix A table, instead of 97.

- The Section 8.1 input-length header, Section 7.5 measured pitch baseline interpretation, and Section 5 manifest availability statement were corrected after review.
  Why: These statements previously confused character counts with English phoneme counts, used 1.0 instead of the measured Telugu pitch baseline for interpretation, and incorrectly said the run manifests were committed.
  Verified: Section 8.1 now labels the counts as characters and lists actual English phoneme counts. Section 7.5 states the measured 0.885 baseline and that no listed pitch loss is below it. `git ls-files English/manifests Telugu/manifests` lists only `.gitkeep` placeholders.

- `Results/Telugu/training_loss.png`, `Results/Telugu/validation_loss.png`, and `Report/Technical_Report.md` (root) - Added the Telugu training and validation curves for steps 0 to 60,000, with captions and interpretation in Sections 7.6 and 7.7. Renamed the supplied plot files to match the required per-language deliverable filenames.
  Why: Section 7 previously showed English curves only, despite already discussing Telugu validation values and checkpoint loss.
  Verified: Both PNGs are valid 1,500 x 1,050 images. Their visible step range is 0 to 60,000. The validation plot agrees with the recorded minimum at step 2,000 (3.6590), saved-checkpoint minimum at 5,000 (3.73799), and final value at 60,000 (3.791061).

### 2026-11-06

- `Report/Technical_Report.md`, `Report/Technical_Report.pdf` (root) — Section 9, "Variance-adaptor analysis", written in place of the pending block: Table 9.1 with frames, seconds, mean F0, F0 range, energy range and pitch σ for the five Telugu sentences at checkpoint 60,000; Table 9.2 with content and pause durations split by symbol class; and four paragraphs on duration, pitch, energy and the change since checkpoint 5,000.
  Why: Section 9 was the only place the report promised to look inside the adaptors' outputs rather than at the losses, and the saved `<ID>.json` payloads beside each plot had never been turned into numbers in the text.
  Verified: Every figure recomputed from `Results/Telugu/checkpoint_60000/variance_adaptor_plots/*.json`. `sum(duration)` equals `mel_frames` on all ten clips across both checkpoints, so the frame counts are internally consistent. Pauses 1.12 to 1.57 times content, punctuation 6 to 9 frames, sentence-initial 1.83 / 2.00 / 3.20 times median in three of five, last content symbol 0.57 to 1.00 of median, last pair 7 to 10 frames against a median pair of about 12, first-quarter mean between −0.09 and +0.63 and last-quarter between −0.27 and −0.68, pitch σ 0.47 to 0.73, and 5,000-to-60,000 totals growth 5.3 to 9.3 % with pitch spread 0.696 to 0.599. Five numbers from my first draft were wrong and were corrected before the entry was written: punctuation 7 to 9 became 6 to 9 frames, an energy ratio to the minimum became the absolute ranges, T2a's first quarter +0.33 became −0.09, the last-pair sum 7 became 7 to 10, and the totals growth 5.1 % became 5.3 %. `python scripts/render_report_pdf.py --check`: no problems, 224 Telugu code points drawn, all Latin sentences present, PDF 858,317 bytes.

- `Report/Technical_Report.md`, `Report/Technical_Report.pdf`, and `scripts/render_report_pdf.py` - The report now presents the Telugu 60,000-step inference result only, while noting that the saved 5,000-step checkpoint had validation loss 3.73799 versus 3.791061 at step 60,000. It clarifies that the lower loss is not evidence that the audio sounds better. Simplified typographic marks and tightened PDF spacing and table pagination.
  Why: Keep the report aligned with the requested final Telugu checkpoint and improve readability without hiding the loss comparison.
  Verified: `python scripts/render_report_pdf.py --check` passes with all Latin sentences present and 219 Telugu code points drawn. The PDF has 12 pages. All 54 renderer tests pass; the page overview shows no large blank areas from table pagination.

- Telugu training completed at step 60,000. Inference ran from checkpoints 5,000, 35,000 and 60,000 in Kaggle for listening comparison. The user requested that only 5,000 and 60,000 be kept in the project, in separate folders; 35,000 was omitted. The user reports that 60,000-step audio sounds good, while step 5,000 had the lowest validation loss (3.73799).
  Why: Keep the lower-validation-loss and user-preferred-audio candidates side by side without mixing or overwriting their outputs.
  Verified: Copied `step_5000` to `Results/Telugu/checkpoint_5000/` and `step_60000` to `Results/Telugu/checkpoint_60000/`. Each contains exactly 5 WAVs, 5 JSONs and 15 PNGs; sizes are approximately 2.02 MB and 2.12 MB. The source ZIP passed integrity checking and contained all three comparison sets. Kaggle lacked a Telugu font, so plot labels use code-point fallbacks.

- `scripts/summarise_variance_export.py` (both) — NEW: reads the `<ID>.json` payloads that `inference.py` writes and prints frames per symbol, normalised pitch and energy spread, the duration-versus-mel-frame check, and per-sentence duration, for every `Results/Telugu/checkpoint_*` directory present.
  Why: Those numbers were being recomputed by hand in notebook cells for each checkpoint, so the report's §9 figures were reproducible only from a transcript.
  Verified: Run against `checkpoint_5000` and `checkpoint_60000`: `sum(duration)` equals `mel_frames` on all ten clips, and the 5,000 figures match the values measured earlier on Kaggle (5.98 frames/symbol, pitch std 0.696, T1 4.78 s, T4 6.91 s).

- `Report/Technical_Report.md` (root) — Section 6's Telugu step-time and wall-clock rows now carry the post-restart measurement: 0.390 s/step over 3,674 steps against 0.362 before the crash, and the wall clock split into 3 h 42 m to step 36,695 plus about 2 h 43 m to 60,000.
  Why: The row still described the pre-crash rate and a 6.1 h projection that the restart invalidated.
  Verified: 1,434 s of tqdm clock from the outer bar in `telugu_resume.log` over steps 35,000 to 38,674; PID 343 at 24 m 27 s in the same reading, and `train_loss.csv` at 402 rows with the newest row 28 s old when it was taken.

- `Report/Technical_Report.md`, `Report/Technical_Report.pdf` (root) — New subsection 7.5, the Telugu loss analysis the report did not have: nine validation points, the validation minimum and where it is lost, and the training-to-validation ratio at step 36,000 for all six components.
  Why: Section 7 described English only, so the smaller corpus's run was documented in sections 4 to 6 and then never shown to the reader as a learning curve.
  Verified: Read from `logs/train/train_loss.csv` (381 rows) and `logs/val/val_loss.csv` (37 rows) in the Kaggle working copy. The six components at step 36,000 sum to 3.7865 against a printed total of 3.7866, and the argmins quoted agree with the ones the plotting script printed independently. The dropout caveat is stated because the training figure comes from a forward pass with dropout active. `python scripts/render_report_pdf.py --check`: no problems, 220 Telugu code points drawn, PDF 828,995 bytes.

- `Report/Technical_Report.md`, `Report/Technical_Report.pdf` (root) — New paragraph at the end of section 6, "Run interruption at step 36,695", recording the container restart, the evidence that separates it from a training failure, and the resume from `35000.pth.tar`.
  Why: Section 6 documented the English OOM interruption in detail and said nothing about the Telugu one, so the report read as if that run had gone through uninterrupted. The restart also explains a dependency failure the reader would otherwise have no context for.
  Verified: `/proc/uptime` 776 s, the notebook's first process at 05:01, all seven checkpoints written within 20 s of each other, zero `dmesg` OOM lines, no `Traceback`, and the log ending mid-redraw at step 36,695. The `unidecode` failure on the following decode was reproduced and fixed by reinstalling. The resumed segment's first 900 training rows were compared against the dead run's own rows at the same steps: deltas between −0.13 and +0.04, inside the run's own spread of about 0.10 between adjacent hundred-step windows. `python scripts/render_report_pdf.py --check`: no problems, 220 Telugu code points drawn, PDF 791,494 bytes.

- `CHANGES.md` (root) — The status block said Telugu pitch "never dropped below the 1.0 predict-the-mean baseline in sixteen validations". It did: 0.9141 at step 1,000 and 0.9861 at step 2,000. The sentence now gives those two steps, the worst value 1.1005 at 3,000 and the best 1.0341 at 25,000.
  Why: I copied the claim forward from an earlier summary without re-reading the validation CSV, and the cleaned CSV rebuilt today contradicts it in its own second row.
  Verified: `logs/val/val_loss.csv`, 36 rows, pitch column read directly: 0.9141 at step 1,000, 0.9861 at 2,000, 1.1005 at 3,000, 1.0341 at 25,000.

- `Report/Technical_Report.md` (root) — Three claims about the feature files corrected in sections 5 and 7. The feature row of the preprocessing table now says the mel is stored time-major, `T × 80`, not `80 × T`. The `stats.json` paragraph now says which of the four values are raw and which are measured after normalisation. The two "ground truth of 1.00" statements in §7.4 and §7.5 now give 0.941 for pitch and 0.948 for energy as the measured value, and say the English figure was not measured.
  Why: All three were true of the reference code rather than of this repository. Both preprocessors call `np.save(..., mel_spectrogram.T)` on a `(n_mels, T)` array, so the file on disk is `(T, 80)`. The `stats.json` tuple is `[min, max, mean, std]` where `normalize()` computes `min` and `max` after dividing by the mean and std, which is why the Telugu pitch entry reads `[-2.7697, 9.7678, 190.710, 45.158]`: two z-scores next to two raw quantities. And the ground-truth standard deviation is not 1.0, because the corpus is standardised with a single global constant while the per-clip spread differs.
  Verified: `Telugu/preprocessor/preprocessor.py:534-537` and `English/preprocessor/preprocessor.py:534-537` both save `mel_spectrogram.T`. Measured on the 518 Telugu validation clips: normalised pitch std 0.941, energy std 0.948, and `sum(duration) - mel frames` is `+0.0` on every clip. The English ground truth was not measured because the preprocessed features are not in this repository, so the report says so instead of quoting the Telugu number for both.
- `Report/Technical_Report.md` (root) — The parameter count is no longer one number for both languages. The architecture table reads 35,159,617 in English and 35,099,201 in Telugu, and the comparison table's `Params` row carries the second number instead of repeating the first.
  Why: The two models are not the same size, and the resumed Telugu run printed 35,099,201 again with the trained weights loaded, so the single figure was wrong for half the table.
  Verified: The Phase-3 CPU smoke-test entry in this file already recorded both numbers, English 35,159,617 and Telugu 35,099,201; the Telugu trainer printed 35,099,201 on resume from step 35,000, and the English trainer printed 35,159,617. I did not confirm which layer accounts for the 60,416-parameter difference, so the report states the two sizes and no mechanism.

- `Report/Technical_Report.md`, `Report/Technical_Report.pdf` (root) — Sections 1 to 3 rewritten in the author's voice: first person for decisions, passive for measurements, the bold-lead bullet lists in section 3 turned into paragraphs, and "this assignment" constructions removed. No number, path, table cell or claim was changed.
  Why: The prose read as machine-written, and the request was that every sentence read as the author's own writing.
  Verified: Every figure and file path cross-checked against the previous version; `python scripts/render_report_pdf.py --check` reports no problems, 220 Telugu code points drawn and all Latin sentences present.

- `Report/Technical_Report.md`, `Report/Technical_Report.pdf` (root) — Sections 4 to 6 rewritten in the same voice as sections 1 to 3: the dataset and configuration choices are stated as my decisions, the counts and rates stay impersonal, and the `Drops:` shorthand lines became sentences that add up. Two cells corrected while I was in there: the `Batch size x grad-accum 16 x 1` row now names all three settings, because `grad_acc_step` really is 1 in both configs and the 4 that bundles the loader is `group_size`, not accumulation; and the Telugu step-time and wall-clock rows now carry the measured 0.362 s/step at step 15,039 instead of the 220-step launch estimate. The section 6 ceiling paragraph gains the Telugu 5.8 GiB reading, with its confound stated rather than presented as proof.
  Why: The same request as sections 1 to 3, and the batch row was actively misleading: `16 x 1` reads as no gradient accumulation, while the loader in fact pulls 64 clips per iteration and the epoch bar therefore counts a quarter of the trainer's steps.
  Verified: Both `config.yaml` files re-read for `batch_size 16`, `group_size 4`, `grad_acc_step 1`, Telugu `save_step 5000`; `English/train.py` lines 82-87 and 137-157 confirm the loader batches `batch_size * group_size` and steps per micro-batch. Every figure from the old sections 4 to 6 checked present in the new text by script (49 tokens, none missing). `python scripts/render_report_pdf.py --check` reports no problems, 220 Telugu code points drawn, PDF 789,236 bytes.

- `Report/Technical_Report.md`, `Report/Technical_Report.pdf` (root) — Sections 7 and 8 rewritten in the same voice: the bold-lead bullets of §7.3 and §7.4 became paragraphs, and the conclusions are claimed as my reading ("I do not trust the step", "I read WER 1.0 as not recoverable"). Three content corrections came with the re-read. The §8 heading said the listening side was PENDING while §7.4 and §8.2 already gave English listening verdicts, so it now says English is measured and heard and Telugu is waiting on training. §8.2 opened with "all five sentences" when it meant the English five. The font note said the checker "counted 223 of the 220 Telugu codepoints", a number its own success path cannot print: `scripts/render_report_pdf.py:161-175` compares code-point sets, prints only the wanted count on success, and fails below 90 % coverage, so the sentence now describes the check that exists.
  Why: Sections 1 to 6 were done and these two still read as a report about somebody else's work. The stale PENDING marker was worse than a style problem because it contradicted two other sections of the same document.
  Verified: All 100 distinct numeric tokens in the old sections 7 and 8 compared against the new text by script. Two differ: `223` is gone because it was never a real checker output, and `60,000` survives with different punctuation. Nothing else changed. `python -m unittest discover -s scripts/tests -p test_render_report_pdf.py`: 9 passed. `python scripts/render_report_pdf.py --check`: no problems, 220 Telugu code points drawn, PDF 781,248 bytes.

- `CHANGES.md` (root) — The six-column table became dated sections with one bullet per change, and the status list became a current summary.
  Why: 139 rows of six columns with very long cells rendered as a wall of text that nobody could read through. The status list had also drifted: one bullet said checkpoint 35,000 was selected, a later one said 60,000 shipped, and one pointed at `docs/AI_HANDOFF.md`, which is no longer tracked.
  Verified: Converted from the table mechanically, so every cell survives word for word; 139 rows in and 139 bullets out across the dates 2026-11-02 to 2026-11-06; the file now renders as headings and lists with no table and no code block.

- `English/utils/train_utils.py`, `Telugu/utils/train_utils.py`, `English/tests/test_train_utils.py`, `Telugu/tests/test_train_utils.py` (both) — NEW `_cpu_state(tensor)`; `restore_rng_state` now passes the torch and cuda RNG entries through it before `torch.set_rng_state`/`set_rng_state_all`. Two new tests per language.
  Why: The English restart at step 30,000 aborted because `load_checkpoint(..., map_location=device)` moves the checkpoint's CPU ByteTensor RNG state onto CUDA, and PyTorch only accepts a CPU ByteTensor. The repair existed only in the Kaggle working copy, so a Telugu OOM restart would have hit the same wall.
  Verified: `python -m unittest tests.test_train_utils` from `Telugu/` and from `English/`: 9 passed, 0 failed each. CUDA itself cannot be exercised on this CPU-only box, so the contract is pinned directly rather than by a GPU round trip.

- `Telugu/config.yaml` (Telugu) — `step.total_step` 900000 -> 60000, `optimizer.anneal_steps` [300000,400000,500000] -> [20000,26000,33000], `step.save_step` 10000 -> 2500, with the reasoning recorded in comments.
  Why: The reference defaults were never replaced on the Telugu side: 900,000 steps is days of GPU time and would leave the learning rate flat, and a 10,000-step save interval would throw away most of the validation-minimum search that section 5 of `docs/AI_HANDOFF.md` depends on. Values now match `English/config.yaml` so the two runs are comparable in the report.
  Verified: Loaded through `utils.config.split_config`: `{'total_step': 60000, 'log_step': 100, 'synth_step': 1000, 'val_step': 1000, 'save_step': 2500}`, anneal `[20000, 26000, 33000]`, warm-up 4000.

- `docs/AI_HANDOFF.md`, `README.md` (root) — NEW tool-agnostic handoff document: update protocol, standing rules, measured state of both languages, the English failure analysis with its numbers, prioritised next actions, the uncommitted list and the Kaggle layout. README docs line mentions it.
  Why: The work is being continued by a second AI tool (VS Code GitHub Copilot) when the Qoder daily limit is hit, and by future sessions with no chat history; the human asked for a handoff that keeps the flow going. Every number in it was re-read from the files or from `CHANGES.md` rather than recalled.
  Verified: Written against `git status`, `Results/English/variance_adaptor_plots/*.json`, the five WAVs, `Results/English/*.png`, `Telugu/config.yaml` and `English/model/loss.py`. No training, inference or data run.

- none (CPU diagnostics only; scripts kept outside the repo) (English) — Decoded the saved predicted mels three ways on CPU: shipping HiFi-GAN, HiFi-GAN after mel/ln(10), and the repo's Griffin-Lim; then tested text conditioning by shuffling and swapping symbol sequences.
  Why: Decide whether the unintelligible English audio is a vocoder/pipeline fault, which would be fixable in minutes, or a model fault, which is not.
  Verified: HiFi-GAN reproduces the input mel at L1 0.24-0.28 nat (in-domain); mel/ln(10) degrades it to 2.9-3.2 with flatness 0.13 vs 0.01; Griffin-Lim gives 0.70-0.74; shuffled E1 symbols move the mel by L1 1.098 and swapped E3 symbols by 1.167 against 1.058 for two genuinely different sentences. Predicted mel median -5.5, p99 -0.5, max 0.7. Conclusion: pipeline correct, acoustic detail wrong. No training or inference run against the dataset.

- `docs/`, `README.md`, `English/inference.py`, `Telugu/inference.py`, `English/tests/test_max_seq_len.py`, `Telugu/tests/test_max_seq_len.py`, `docs/export_svarah.py` (root) — Six working notes are no longer tracked in the public repository: `AI_HANDOFF.md`, `REQUIREMENTS_AUDIT.md`, `RULES_FOR_AI.md`, `PLAN_PHASES.md`, `REPORT_OUTLINE.md`, `QODER_PROMPT.md`. The files are kept on disk and hidden from the working tree with a local-only `.git/info/exclude`, so they remain usable and are not published. Every reference to them in code, tests and the README was rewritten, and the README's pointer to the git-ignored assignment PDF was removed.
  Why: These documents record how the work was coordinated rather than what the models do, and they do not belong in a submission repository.
  Verified: `git grep` finds no reference outside the historical rows above; `English/tests` 92 passed, 1 skipped and `Telugu/tests` 100 passed, 2 skipped, both unchanged from the pre-edit baseline.

- `scripts/export_svarah.py`, `scripts/export_male.py`, `scripts/tests/test_validate_alignments.py`, `conftest.py`, `README.md` (root) — The two Kaggle export scripts moved from `docs/` to `scripts/`, and the lone root test moved into `scripts/tests/` with `conftest.py` promoted to the repository root, so `tests/` no longer exists.
  Why: Python sat in a documentation folder, and the alignment-QC test was split away from the seven other script tests, which forced a `pytest Telugu/tests tests` invocation that bundled a root suite with a language suite.
  Verified: Both export scripts use absolute `/kaggle/working` paths, so the move cannot change their behaviour; `py_compile` clean; suites now run as three separate invocations, English 92 passed/1 skipped, Telugu 96 passed/2 skipped, kaggle+scripts 73 passed, 261 total and unchanged.

- `README.md` (root) — Status section corrected: date moved to 2026-11-06, the "English training is in progress" claim removed, the training row rewritten (60,000 ships and was chosen by listening at 7/10 against 5/10 for step 5,000; Telugu is running rather than not started), the outstanding-work list shortened, and a paragraph added stating that all five English outputs are unintelligible with the measured cause.
  Why: The README still said English was in progress and that step 35,000 had the lowest validation loss, which report section 7 now contradicts, and it promised that nothing in the file was a generated-speech result.
  Verified: Numbers taken from report sections 7 and 8 and from `logs/val/val_loss.csv`; documentation only, no code changed.

### 2026-11-05

- `English/config.yaml` (English) — Training step budget set from measurement: `total_step: 60000`, `anneal_steps: [20000, 26000, 33000]`, `save_step: 5000`.
  Why: A 300-step calibration on a Kaggle T4 measured 0.304 s/step, so 60000 steps is a ~5.3 h run. The reference 900000 steps with anneal at 300000/400000/500000 is unattainable for 5,322 clips (~281 steps/epoch) and would leave the learning rate flat for the whole run; the new anneal points keep the reference fractions (33/43/55 %).
  Verified: Calibration run `config_calib.yaml`: exit 0, 300/300 steps in 130 s wall, training loss 5.0787 and validation 5.0589 at step 300; checkpoint `checkpoints/300.pth.tar` written.

- Kaggle `English/data/preprocessed/` (generated, not tracked) (English) — Full preprocessing of the 5,582 manifest clips with `--allow-dropped`.
  Why: Phase 2 completion for English; every dropped clip must be accounted for before training.
  Verified: `Clips: 5582 total, 5322 ok, 260 dropped`: alignment 164 (non-positive phone duration), missing TextGrid 71, pitch extraction 20 (0 voiced frames), feature statistics 4, transcript 1. `Processed 5322 clips (4500 train, 822 val), 2 speakers`. 4.842 h of aligned audio; note the script's `Total time: ... hours` line is audio duration, not wall time (the run took 6.5 min). stats: pitch `[-2.0096, 9.0707, 177.64, 56.06]`, energy `[-1.0123, 8.1121, 27.01, 26.67]`.

- Kaggle `Telugu/data/raw/male/`, `Telugu/manifests/`, preprocessor run (Telugu) — Rebuilt the missing raw audio by upsampling `/kaggle/working/mfa_corpus/male` from 16,000 to 22,050 Hz (`resample_poly` 441/320, PCM_16), restored `.lab` from the same aligner corpus, and ran preprocessing on the 3,363 manifest clips.
  Why: A Kaggle restore left the repo tree and the HuggingFace cache empty, so `export_male.py` could not be re-exported; `mel_fmax: 8000` means the 16 kHz source already carries the whole mel band. The `.lab` must hold the alignment text, not the raw dataset text, because `Telugu/preprocessor/preprocessor.py:352` requires `len(text_to_sequence(text)) == len(duration)`.
  Verified: `corpus labs: 3355`; 3,355 wavs + 3,355 `.lab` written in 64 s; smoke `30 total, 30 ok, 0 dropped`; full run `Clips: 3363 total, 3269 ok, 94 dropped` (alignment 83, missing TextGrid 3, missing audio 8); `Processed 3269 clips (2751 train, 518 val), 1 speakers`; 4.700 h. **Provenance: Telugu audio is 16 kHz upsampled to 22,050 Hz, not exported from the source at 22,050 Hz**.

- `Report/Technical_Report.md` (both) — NEW draft: sections 1–5 and 11 written from measured values (architecture, modifications, datasets, preprocessing, training config, reproducibility); sections 6–9 are explicit PENDING placeholders for the loss curves, generated-audio observations, variance-adaptor analysis and the English/Telugu comparison.
  Why: The report is a graded deliverable and the training wait is otherwise idle time; writing it now avoids inventing numbers later.
  Verified: Cross-checked every figure against `CHANGES.md` rows 126–128 and `English/config.yaml`; no training or inference result is claimed.

- `docs/test_sentences.json` (both) — All ten `final` fields filled from the rendered PDF (page 1 English, page 2 Telugu); each `final` set equal to the previously verified `raw_extracted` string and the note updated.
  Why: The assignment requires the sentence text preserved as provided, and inference refuses to run while any `final` is null. PyMuPDF rendering of pages 1–2 showed the folder's extraction matched the render; the human authorised the edit explicitly ("Update the final then").
  Verified: Validator over the edited file: `PROBLEMS: none` (non-null, stripped, NFC, no Latin letters in Telugu, no placeholders). `text_to_sequence` encodes all ten: Telugu 68/34/40/33/91 and English 108/72/75/49/115 symbols. E4 normalises to `ten:forty-five am on twelve november twenty twenty-six`; T4 keeps digits `2026`, `12వ`, `10:45` unchanged — recorded as a known frontend gap, not a crash.

- `Report/Technical_Report.md` (both) — §8 rewritten: text-processing side measured (symbol counts and normalised strings per sentence), listening side left PENDING; E4 normalisation and the T4 digit gap documented.
  Why: The frontend behaviour is measurable now and belongs in the report as a limitation the assignment explicitly asks for.
  Verified: Commands in the two `text_to_sequence` runs recorded above; no audio claim made.

- `Report/Technical_Report.md` §7 (English) — Interim loss analysis written from the Kaggle logs at step 25,000 of 60,000: full train/validation tables, convergence, the single warm-up spike, and the overfitting finding (validation minimum 3.8246 at step 7,000, 4.0588 at 25,000; duration loss +21 % while mel stays flat).
  Why: The assignment asks for convergence, instability and overfitting analysis, and the logs are already available; the section is marked interim until the full run finishes.
  Verified: Parsed both logs with the regex table format (242 train points, 25 validation points, steps 100–25,000). Missing lines noted: 12,100–12,800 (8 points, buffering artefact).

- `.gitignore` (root) — Added `docs/log*.txt`.
  Why: Training logs pasted into `docs/` for analysis must never be committed; the analysed copies were moved out of the repo to the Qoder session folder.
  Verified: `git status` clean of log files after the move.

- Kaggle `mfa_conda` environment (not tracked) (both) — Record the repairs that made MFA 3.4.3 usable after a session restore: `chmod -R +x` on the environment `bin` directories, recreation of ~760 `lib*.so.N` soname symlinks the restore had flattened to real files, `/usr/bin/sqlite3` required by `collect_alignments`, and invocation as `PATH=/kaggle/working/mfa_conda/bin:$PATH /kaggle/working/mfa_conda/bin/mfa ...` because `micromamba run` broke once `bin/python` was gone.
  Why: The README claimed `kaggle/bootstrap.sh` was the environment path; it was never run end to end, and these manual steps are what actually worked.
  Verified: `mfa version` printed 3.4.3 with 956 entries in `bin` and 0 zero-byte files; `mfa align` then produced 3,352 Telugu TextGrids in 378.511 s and the English alignment completed for 5,322 clips.

- `README.md` (root) — Rewrote the stale parts: status block replaced by a measured per-language table (English 5,582 manifest / 5,322 usable / 4.842 h / 2 speakers / 260 drops, training at 60,000 steps; Telugu 3,363 / 3,269 / 4.700 h / 1 speaker / 94 drops, untrained) with the Telugu 16 kHz provenance and the remaining work listed; Setup replaced by the measured environment (Python 3.13, torch 2.11.0+cu130, T4x2, targeted pip installs); dataset export now points at `docs/export_svarah.py` and `docs/export_male.py`; alignment-QC clip_id mapping recorded as exercised; G2P supplementary dictionary documented as **not used** (`English/lexicon/g2p_names.txt` absent, MFA's own G2P handled the 4.47 % OOV); `bootstrap.sh` marked never run end to end; unit-test counts added.
  Why: The old text asserted "no model has been trained" and "the English (Svarah) side has had no data work at all", both now false, and the Telugu manifest numbers quoted were pre-MFA.
  Verified: `pytest English/tests` 90 passed / 1 skipped; `pytest Telugu/tests tests` 98 passed / 2 skipped; `pytest kaggle/tests scripts/tests` 60 passed; every number cross-checked against `CHANGES.md` rows 126-128 and the live session panel.

- `README.md` unit-test section (root) — Documented that the suites must be run one folder at a time.
  Why: A combined root run fails by design: both language folders ship identically named modules (`text`, `utils`, `model`, `preprocessor`), so `pytest English/tests Telugu/tests` raises 12 import-mismatch collection errors and `--import-mode=importlib` fails with `cannot import name 'telugu' from 'text' (English/text/__init__.py)`.
  Verified: Both failures reproduced locally; per-folder runs pass.

- `requirements.txt`, `Report/Technical_Report.md` §6 and §11 (root) — Header rewritten to state the file is a proposed pin set never installed as a whole on Kaggle; report hardware/reproducibility rows corrected to GPU T4x2 with `nn.DataParallel` splitting batch 16 as 8+8, and to the real pip state.
  Why: The pins implied a verified environment that does not exist; `train.py:91` applies `nn.DataParallel` unconditionally, which is why both GPUs show 100 % for one process.
  Verified: Session panel (GPU T4x2, both 100 %, 4.7 + 3.9 GiB, Python 3.13 site-packages path) plus `grep` of `train.py` and `requirements.txt`.

- `scripts/render_report_pdf.py`, `scripts/tests/test_render_report_pdf.py`, `Report/Technical_Report.md` 8.0 and status block, `Report/Technical_Report.pdf`, `README.md` (root) — NEW report-to-PDF build (Markdown -> HTML via `markdown-it-py` -> headless Chrome/Edge, no pandoc) with a `--check` that verifies the PDF; section 8.0 added holding all ten `final` sentences copied programmatically from `docs/test_sentences.json`; the section map in the opening block was renumbered to the real headings (loss = 7, test sentences = 8, variance = 9, comparison = 10, reproducibility = 11) and the status line corrected to "Sections 1-6 and 11"; README gained a Report PDF section and the Telugu-font claim was replaced with the measured one.
  Why: The submission needs a PDF and the report must show the exact Telugu input, so the render path had to be reproducible and the glyph coverage proven rather than assumed.
  Verified: First `--check` reported T1/T2b/T4 missing; investigation showed the page renders correctly and PDF extraction reorders Telugu vowel signs (visual-order glyph runs), so exact string matching is the wrong test for Indic text and was replaced by a code-point coverage check. Measured: 220 of 220 Telugu code points in the PDF, 0 U+FFFD, 11 pages, `PROBLEMS: none`. Telugu coverage comes from `C:\Windows\Fonts\Nirmala.ttc` - the machine has no Noto or Gautami Telugu font and the Kaggle image has none at all, so the PDF must be built locally or with `--telugu-font`. `pytest scripts/tests` 52 passed (7 new), `pytest kaggle/tests scripts/tests` 67 passed. **Process note:** the first 8.0 insert was typed by hand and corrupted four Telugu words; it was discarded and regenerated from the JSON with an assertion that each `final` string appears verbatim.

- `English/utils/tools.py`, `Telugu/utils/tools.py`, `English/config.yaml`, `Report/Technical_Report.md` (English (code also in Telugu)) — **English run 1 was OOM-killed at step 32,000.** Two changes before resuming from `30000.pth.tar`: `log()` now calls `plt.close(fig)` after `add_figure` in both language folders, and `train.step.save_step` 5000 -> 2500. Report section 6 rows corrected to the measured 0.511 s/step mean and ~8.5 h full-run wall clock, with the interruption recorded.
  Why: The session panel reached 30/30 GiB RAM and the shell reported `Killed` for the `setsid python -u train.py` process; the log has no Traceback and no CUDA out-of-memory error, so it was host RAM, and the tqdm rate degraded from ~0.30 to ~0.65 s/step over the run, which is the same growth showing up as time. `synth_one_sample` creates a figure on every `synth_step` and every validation and nothing in the reference released it. `ScheduledOptim` is constructed with `args.restore_step`, so the resumed LR is the one the continuous run would have had.
  Verified: Log analysed locally: last step line 32,000/60,000 at 4:32:45 elapsed, 6 `Saved checkpoint:` lines (5k-30k), 31 validation rows 1,000-31,000, minimum total 3.8246, 4.0806 at 31,000. Kaggle inventory after the restart: all six checkpoints present (`30000.pth.tar` 418,124,155 bytes) and 5,322 files each in mel/pitch/energy/duration, so nothing but steps 30,001-32,000 was lost. `pytest English/tests` 90 passed / 1 skipped, `Telugu/tests tests` 98 / 2, `kaggle/tests scripts/tests` 67.

### 2026-11-04

- `preprocessor/report.py` (both) — NEW `mel_frame_limit(max_seq_len, margin)` and `check_mel_length(mel_frames, max_seq_len, margin)`: raises `ClipSkipped('too_long')` once a clip's mel exceeds `max_seq_len - margin`; validates both values and rejects a non-positive limit. `DEFAULT_MAX_SEQ_LEN_MARGIN = 20`.
  Why: The decoder truncates training batches at `max_seq_len` without warning; preprocessing must not hand it such clips.
  Verified: Covered by new `tests/test_too_long.py` (7 tests per language).

- `preprocessor/preprocessor.py`, `preprocess.py`, `config.yaml` (both) — `Preprocessor` now takes a required `max_seq_len` (wired from `model.max_seq_len` in `preprocess.py`) and reads `preprocessing.max_seq_len_margin` (default 20, set to 20 in both configs). After the mel is computed by the existing STFT path and trimmed to the alignment, its frame count is checked against the limit; violators are dropped and counted under `too_long`. Selection logic and feature outputs unchanged.
  Why: Hard guard requested so a long clip is reported, not silently shortened.
  Verified: `check_mel_length` call site in `process_utterance`; config loader still validates; end-to-end run not possible locally (pyworld/tgt missing).

- `tests/test_too_long.py` (both) — NEW. Asserts against the real STFT path (`TacotronSTFT` + `get_mel_from_wav`): frame count is `1 + floor(samples / hop)` for 1024/22050/30011 samples; equals `librosa.stft(center=True)` and differs from `center=False`; guard allows the limit, rejects one frame over with category `too_long`, honours a custom margin, rejects invalid limits.
  Why: Test request; also records the center-padding answer in executable form.
  Verified: Ran: 7 tests OK in English (7.1 s) and Telugu (2.9 s); full discovery English 84 tests OK (1 skip), Telugu 89 tests OK (2 skips). The end-to-end `process_utterance` path could not be executed locally because pyworld and tgt are not installed.

- `README.md`, `docs/REQUIREMENTS_AUDIT.md` (root) — README documents the `too_long` drop and its limit; audit section 12 preprocessing row and consequence updated from "no cap exists" to the new guard, with the frame formula and test references.
  Why: Keep documentation consistent with the code.
  Verified: Docs only.

- `inference.py` (both) — NEW variance-adaptor export: after each synthesis, `export_variance_outputs` writes `Results/<Language>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png` plus a combined `variance_adaptor_plots/<ID>.json` holding the raw predicted arrays (frames, seconds, Hz, normalised values), symbol labels, checkpoint step, speaker, controls and stats. Predictions are sliced with `output[8]`/`output[9]` lengths (no padding), pitch/energy are denormalised from `stats.json` (`v*std + mean`), and the sequence is verified against `src_len` before plotting. Symbol labels fall back to `U+XXXX` code points when no installed font has Telugu glyphs (detected with `matplotlib.ft2font.FT2Font.get_char_index`); `check_phoneme_level()` refuses to run unless pitch and energy are `phoneme_level`; an array whose length differs from the symbol count raises instead of producing a misleading plot.
  Why: Deliverable 6 requires duration, F0 and energy plots per test sentence for both languages; previously no code produced them (`synth_samples` is dead code) and the figures were not recoverable after the run.
  Verified: `Telugu/tests/test_variance_export.py` and `English/tests/test_variance_export.py`: 7 tests each, all PASSED (not skipped); the exported helper block is byte-identical between the two files (verified function by function); no real inference run (no checkpoint).

- `inference.py` (both) — Generated WAVs are now saved to `Results/<Language>/test_generated_audios/<ID>.wav` (`AUDIO_DIR_NAME`) instead of the `Results/<Language>` root.
  Why: Deliverable 5 and the required folder structure name that subdirectory; the audit had wrongly recorded this as met.
  Verified: Null-refusal run still exits 1 as before; the path itself is covered by the new export tests; full synthesis not run (no checkpoint).

- `tests/test_variance_export.py` (both) — NEW (one per language). Directory-name constants, denormalisation math, symbol-ID round trip (Telugu `encode_alignment_tokens` / English table index), ASCII label pass-through, a real export producing 3 PNGs + JSON, `phoneme_level` rejection, and the missing-`stats.json` abort.
  Why: Rule 11.
  Verified: Ran: 7 passed in 5.30 s (Telugu) and 7 passed in 5.11 s (English).

- `scripts/plot_training_losses.py` (root) — NEW. Read-only deliverable plotter: `read_loss_csv` validates the header against `utils/train_utils.LOSS_COLUMNS` and rejects missing columns, header-only files, non-numeric values and non-monotonic steps; `plot_losses` writes `training_loss.png` (total + per-component, log scale) and `validation_loss.png` for a language from the train/val loss CSVs `train.py` already appends.
  Why: Deliverable 4 requires training and validation loss plots; nothing in the repo produced them, and the reference only logs to TensorBoard.
  Verified: `scripts/tests/test_plot_training_losses.py`: 9 tests, all pass; not run against a real CSV yet (no training run).

- `scripts/tests/test_plot_training_losses.py` (root) — NEW. Header/round-trip, missing-column, header-only, non-numeric, non-monotonic, empty-file, PNG output and two-panel layout cases.
  Why: Rule 11.
  Verified: Ran: `python -m pytest scripts/tests/test_plot_training_losses.py -q` -> 9 passed; `python -m compileall -q` on the three new/changed files -> exit 0.

- `docs/REQUIREMENTS_AUDIT.md` (root) — Corrected F4 (audio path was recorded as Met but the WAVs went to the `Results/<Language>` root), F8 and A12 from Missing to "Partial (code only, no run)", A10 likewise, refreshed A6; appended section 13 covering deliverables 4, 5 and 6 with file:line evidence and the font caveat.
  Why: The audit must not claim a deliverable the code did not implement, and must distinguish "code exists" from "run produced the file".
  Verified: Corrections verified against `inference.py:32,210,278-279,306-307` and the test runs below.

- `README.md` (root) — New "Deliverable plots" section with the `plot_training_losses.py` invocation and the variance-export behaviour (including the Telugu-font warning); layout and unit-test paragraphs updated; status paragraph now records the measured Telugu split and the still-running export.
  Why: The README is a deliverable and was silent on how the two plot sets are produced.
  Verified: The documented command was executed against synthetic train/val CSVs in a temp directory: exit 0, `training_loss.png` + `validation_loss.png` written; a header-truncated CSV exited 1 naming the missing columns.

- tests (verification run) (both) — Re-ran every local suite after the changes.
  Why: Acceptance for the export/plotter work.
  Verified: `python -m pytest scripts/tests -q` -> 45 passed; Telugu `tests` -> 94 passed, 2 skipped; English `tests` -> 90 passed, 1 skipped; `compileall` exit 0. No preprocessing, training or inference run.

- `docs/test_sentences.json` (root) — **Edited by the human, not by the AI**, with their approval to record it: 10 `raw_extracted` strings corrected because the PDF text layer splits Telugu grapheme clusters and one English word (`rhyth m`, `న్యూ రోఫార్మ కోలాజికల్`, `వచ్చ ింది`, `అడ  విలో`, `నిశ్శ బ్దంగా`, plus stray spaces/newlines). SHA-256 `420187d15f4f9c4a61202ce8d6c3b45f72fbb4535c8f3be0689c01cf60847a6f` -> `944621434fc8d0cec6f8163354e1267bedba65c7be0138fb6e0a98460f2a012b`.
  Why: Rule 8 forbids the AI editing these; the human did it deliberately and asked for the record.
  Verified: Re-extracted the PDF text layer with `pypdf 6.17.0` and compared every sentence: 9 of 10 now equal the PDF with its extraction spaces joined; **T2b additionally merges `పట్టు చీరను` -> `పట్టుచీరను`, a space the PDF really prints** (audit section 14). No page renderer installed, so this is text-layer evidence, not visual. `final` is still null for all ten, so `load_test_sentences` still refuses to run.

- `docs/REQUIREMENTS_AUDIT.md` (root) — D8 changed from "Met" to "Not met (human decision)"; the section 1 SHA line and section 9.2 marked as superseded; new section 14 records the old/new SHA-256, the per-sentence PDF comparison and the two open questions (T2b reading; text belongs in `final`, not `raw_extracted`).
  Why: The audit claimed an unchanged hash that is no longer true; the claim had to be corrected in place, not deleted.
  Verified: Verified against the file's own diff (`git diff docs/test_sentences.json`, 6 lines) and the re-extraction above.

- `.gitignore` (root) — Added `docs/*.csv` (the per-clip duration table `docs/telugu_clip_table.csv` is derived from the dataset and re-creatable, so it stays out of the submission repo) and a `*.log` rule for Kaggle-side `nohup` redirects.
  Why: Keeps 3,355-row data tables and stale crash logs out of a public repo without touching `Telugu/manifests/*.csv`, which Rule 6 requires committing.
  Verified: `git check-ignore -v` confirms `docs/telugu_clip_table.csv`, `docs/export_male.log` and `__pycache__` are ignored; `Telugu/manifests/train.csv` is not.

- `docs/export_male.py` (root) — NEW (record, not pipeline code). The exact Kaggle-side script that streamed the 3,363 male IndicTTS clips out of the HF Arrow cache into `data/raw/male` (22050 Hz) and `mfa_corpus/male` (16 kHz), resampling with `resample_poly`, writing `.lab` files, and routing any clip whose MFA-side text normalises to empty to `empty_mfa_text`. Its absolute `/kaggle/...` paths are deliberate: they identify the run being recorded.
  Why: Phase 2 needs the export reproducible and reviewable; the script existed only on the ephemeral Kaggle disk.
  Verified: Run on Kaggle, not here: `DONE exported 3363 bad 0 of 3363` (log `export_male_log.csv`, per-clip statuses). The committed `docs/export_male.log` is the *first*, crashed attempt (missing `unidecode`) and is now ignored by the `*.log` rule.

- git (root) — First commit and push of the 2026-11-03/04 working tree to `origin/main` (`https://github.com/RajendharAre/FastSpeech`, repo confirmed public: `GET /repos/RajendharAre/FastSpeech` -> 200) so both Kaggle notebooks can `git pull` the `too_long` guard, the variance exports and `scripts/plot_training_losses.py`.
  Why: The Kaggle notebooks clone from GitHub; unpushed local work cannot run there.
  Verified: `git fetch --dry-run` showed no divergence before the push; no candidate file exceeds 5 MB; suites re-run green immediately before the commit (see next row).

- tests (verification run, pre-push) (all) — Re-ran every suite from its own rootdir after the doc edits.
  Why: Pushing to the submission repo needs a clean suite record.
  Verified: `scripts/tests` 45 passed (1 warning); `tests` 4 passed; Telugu 94 passed / 2 skipped; English 90 passed / 1 skipped; `kaggle/tests` 15 passed. Note: running all five from the repo root in one pytest invocation gives 12 collection errors (each language suite needs its own rootdir on `sys.path`) - use per-directory commands. No preprocessing, training or inference run.

- `docs/export_svarah.py` (English) — NEW (Kaggle-side run record, like `docs/export_male.py`). Streams Svarah out of the HF cache and writes four things: `English/data/raw/<speaker>/<clip>.wav` upsampled 16 kHz -> 22,050 Hz (`resample_poly`, PCM_16) with the **raw** transcript in `.lab`; `/kaggle/working/mfa_corpus/english/<speaker>/` at the native 16 kHz with the **aligner** text (`text.normalise.normalise_english`, lowercased, non-alphanumerics to spaces, apostrophes kept); `/kaggle/working/svarah_manifest/all.csv` with the four required manifest columns and durations measured from the decoded audio, not the metadata; and a per-clip status CSV plus `too_long.csv`. The `speaker` directory/column comes from Svarah's `gender` column because the release has **no speaker labels** (3,232 filename tokens, max 0.105 h each, 53.8% agreement with `gender`), which the human chose as the English speaker strategy. Clips whose decoded length exceeds 250,879 samples (= exactly 980 mel frames at 22,050/hop 256) are not exported and are listed instead. A normaliser exception falls back to the raw text rather than dropping the clip.
  Why: Phase 2 needs the same export/align/preprocess route for English that Telugu uses, and `audio` is null for all 6,656 Svarah rows so the bytes must be read from `audio_filepath`.
  Verified: Helper functions checked locally with a stubbed `datasets` import: `MAX_SAMPLES` -> 980 frames and +1 sample -> 981; resample 2.0000 s -> 2.0000 s with peak 1.0 and float32 out; `mfa_text` output on a real sentence and on a raising normaliser. `python -m compileall` exit 0. **NOT RUN on the real data yet** - the laptop has no `datasets` package and Rule 3 keeps downloads on Kaggle.

- `docs/export_svarah.py` (English) — **Bug fix in my own guard.** The pre-export exclusion compared the **16 kHz source** sample count against `MAX_SAMPLES = 250,879`, which is a **22,050 Hz** budget, so it cut at 15.68 s instead of 11.37773 s. The predicate is now `seconds > MAX_SEC`, and the resampled array that is actually written is re-checked against `MAX_SAMPLES` with an assertion.
  Why: The Kaggle run exposed it: `DONE exported 6419 hours 8.2654`, `too_long 237`, and the script's own summary line printed `longest exported clip 15.626 s, guard 11.378 s` - impossible for the intended limit.
  Verified: Predicate re-checked locally through the real resampler: 11.37770 s -> 250,879 samples -> 980 frames PASS; 11.37775 s -> 250,880 -> 981 REJECT; 15.626 s -> 1,346 frames REJECT; 29.871 s -> 2,573 frames REJECT. The 316 over-guard clips already written are removed from `all.csv` and moved to `/kaggle/working/pruned/english/` by a Kaggle re-filter cell rather than re-exporting.

- `docs/export_svarah.py` (English) — **Second export bug: raw WAV/`.lab` were written flat in `English/data/raw/` instead of `English/data/raw/<speaker>/`.** The speaker subdirectory was added to the `mfa_corpus` writes and forgotten on the `data/raw` writes, so the two halves of the export disagreed on layout. Both `sf.write`/`open` targets now include `speaker`, matching the module docstring.
  Why: Found from the Kaggle re-filter cell's own output: `files moved: 632 expected 1264` and `wav counts -> raw: 6419 | mfa: 6103 | pruned: 0` - the MFA side moved, the raw side had nothing where the mover looked. `English/preprocessor/preprocessor.py:289` reads `data/raw/<speaker>/<clip>.wav`, so **every** English clip would have failed as a missing file.
  Verified: Fixed by inspection against the docstring and the preprocessor's path contract; `python -m compileall -q docs/export_svarah.py` exit 0. The 6,419 already-written flat files are relocated on Kaggle from `svarah_manifest/all.csv` (clip_id -> speaker) with no re-export; the 316 over-guard pairs go to `/kaggle/working/pruned/english/raw/`.

### 2026-11-03

- `scripts/validate_alignments.py` (root) — NEW. Read-only MFA TextGrid QC: parses `words`/`phones` tiers, flags short words, long/very short phones, gaps and manifest/TextGrid mismatches, writes `alignment_review.csv`; requires only the four manifest columns.
  Why: Item 4 prerequisite: find unusable alignments before preprocessing.
  Verified: Synthetic fixtures only: no real TextGrid exists in the repo; tier-name match with the corpus is UNVERIFIED.

- `scripts/validate_alignments.py` (root) — Fixed the interval regex so the first interval is not skipped on CRLF files, and deduplicated the case-insensitive TextGrid glob; removed an unused import.
  Why: The supplied regex anchored past the first interval on Windows line endings and `rglob` returned each file twice on Windows.
  Verified: Targeted regex check returned `('0','0.02','a')` and `('0.02','1','hello')`; validator reported 2 clips, 2 TextGrids, 5 findings on the fixtures.

- `tests/test_validate_alignments.py` (root) — NEW. Three supplied tests on synthetic TextGrids plus one compatibility test proving provenance columns are accepted.
  Why: Item 4 / rule 11.
  Verified: `python -m pytest tests/test_validate_alignments.py -q`: 4 passed, 1 warning.

- `tests/conftest.py` (root) — NEW. Adds the repository root to `sys.path` for root-level pytest tests.
  Why: `scripts` was not importable when pytest ran as a console script rather than `python -m pytest`.
  Verified: Simulated console-script invocation now imports `scripts.*`; targeted tests pass in both invocation styles.

- `English/text/normalise.py` (root) — NEW. Rule-based English normalisation returning `(text, rules)`: currency, percent, time, ISO date, abbreviation, acronym, symbol, decimal, number, whitespace; raw text is never replaced in place.
  Why: Requirement D2: normalise for the frontend while keeping the source sentence and the rules that fired.
  Verified: Probed outputs, e.g. `10:45 AM` -> `ten forty-five a m` (rules `time;number`), `2026-11-12` -> `the twelfth of november twenty twenty-six`, `$5.50, or 45%` -> `five point five zero dollars, or forty-five percent`.

- `English/text/acronyms.csv` (root) — NEW. 25 acronyms with `spoken` form and `mode` (`letters`/`word`).
  Why: Acronym pronunciation must be data, not guesses in code.
  Verified: Table validates on load; invalid mode or missing column aborts.

- `English/tests/test_normalise.py` (root) — NEW. 13 tests: time, ISO date, currency+percent, CSV acronym, dotted acronym, abbreviation, decimal/ordinal, unchanged text, rule formatting, acronym-table validation.
  Why: Rule 11.
  Verified: Ran: 13 tests, OK.

- `scripts/split_dataset.py` (root) — Manifest rows gained optional `transcript_normalised`/`rules_fired`; `--normalise-english` and `--english-root`; writer emits the extra columns only when present (`extrasaction="ignore"`); empty normalised text aborts.
  Why: Keep raw and normalised text plus provenance in the manifests without breaking the four required columns.
  Verified: End-to-end CLI run produced the 6-column header; `Svarah met Dr. Rao at 10:45 AM.` -> `Svarah met doctor Rao at ten forty-five a m.` with `time;abbreviation`.

- `scripts/tests/test_manifest_normalisation.py` (root) — NEW. 4 tests: raw text unchanged, rules attached, empty normalisation raises, provenance round-trip, four-column output when normalisation is off, normaliser import.
  Why: Rule 11 / manifest compatibility.
  Verified: Ran: 4 tests, OK.

- `scripts/build_g2p_names.py` (root) — NEW. Collects capitalised out-of-vocabulary name candidates from manifests, converts them with `g2p_en`, writes a `NAME<TAB>PHONES` dictionary and a status report; skips lexicon hits and lowercase-common words; fails with an install message when `g2p_en` is missing.
  Why: Supplementary dictionary for names not covered by LibriSpeech.
  Verified: Unit-tested with a fake G2P (8 tests, OK); the real CLI exited 1 with the missing-`g2p_en` message and wrote nothing, so real generation is UNVERIFIED.

- `scripts/tests/test_build_g2p_names.py` (root) — NEW. 8 tests: candidate filtering, min-count, statuses, duplicate counting, dictionary format, comments/blanks, manifest reading, lexicon parsing.
  Why: Rule 11.
  Verified: Ran: 8 tests, OK.

- `inference.py` (English) — Normalises the input first and prints raw, normalised and fired rules; `read_lexicon` accepts an optional supplementary dictionary, with the main lexicon winning; blank/`;;;` lines skipped.
  Why: Requirement D2 at inference; names missing from the main lexicon.
  Verified: `test_lexicon_merge.py`: 3 tests, OK; no real inference run (no checkpoint).

- `config.yaml` (English) — Added `preprocess.path.supplementary_lexicon_path: ""` with a note that it stays empty until the G2P dictionary exists.
  Why: Wire the optional dictionary without changing default behaviour.
  Verified: Config validation still passes; unknown keys are tolerated.

- `tests/test_lexicon_merge.py` (English) — NEW. 3 tests: main wins/supplement fills gaps, blank and comment lines, missing file raises.
  Why: Rule 11.
  Verified: Ran: 3 tests, OK.

- `preprocessor/manifest.py` (both) — Docstring: provenance columns are produced by `--normalise-english` and ignored by preprocessing.
  Why: Avoid implying the preprocessor consumes the normalised column.
  Verified: Docstring only; tests unchanged.

- `README.md` (root) — Documented status, new layout entries, provenance columns and `--normalise-english`, alignment QC, G2P build/config key, and the root pytest command.
  Why: Keep the operational docs current.
  Verified: Commands match the implemented CLIs; explicit UNVERIFIED notes for real TextGrids and `g2p_en`.

- `docs/REQUIREMENTS_AUDIT.md` (root) — Appended section 11.
  Why: Keep the audit current without rewriting earlier sections.
  Verified: See section 11.

- tests / CLIs (verification) (both) — Re-ran targeted pytest, both language suites, script suite, compileall and smoke tests; ran the alignment QC and the normalising splitter on scratch fixtures in the system temp directory.
  Why: Acceptance for the QC, normalisation and G2P work.
  Verified: Targeted pytest 4 passed (1 warning); English 72 tests OK (1 skip); Telugu 78 OK (2 skips); scripts 36 OK; compileall exit 0; smoke forward pass OK in 0.095 s (English) and 0.093 s (Telugu). Alignment QC: 2 clips, 2 TextGrids, 5 findings. Splitter wrote the 6-column manifests. G2P exited 1 (`No module named 'g2p_en'`) and wrote nothing. No training, preprocessing or data generation run.

- `kaggle/bootstrap.sh` (root) — NEW. Idempotent Kaggle bootstrap: Miniforge install, `mfa` conda env (Montreal Forced Aligner, Python 3.13), pip pins from the first real Kaggle run (NumPy 2.1.3, SciPy 1.16.3, librosa 0.10.2.post1, pyworld 0.3.5 with `--no-build-isolation`), HiFi-GAN unzip for both languages, `english_us_arpa` acoustic + dictionary download; every step prints DONE or SKIP with its reason.
  Why: Set up the MFA alignment environment the same way the successful Kaggle run did, without re-doing completed steps.
  Verified: `bash -n` passed; file is ASCII with LF endings and the executable bit set. NOT RUN yet; the first Kaggle run must record its output. Does not install the full `requirements.txt` by design.

- `README.md`, `docs/REQUIREMENTS_AUDIT.md` (root) — Marked the `words`/`phones` tier names as verified against the real Kaggle MFA output (audit section 11 unknown 1 resolved); documented `kaggle/bootstrap.sh` in Setup and the layout.
  Why: The user confirmed the real MFA output tiers; the previous "not checked against the real corpus" note is obsolete.
  Verified: No code change needed in `scripts/validate_alignments.py`; clip-id-to-TextGrid-name mapping remains unverified.

- `text/acronyms.csv` (English) — ISRO changed from `I S R O,letters` to `ISRO,word`, matching the existing word-mode pattern (spoken form equals the token; pronunciation is handled downstream by the lexicon/G2P).
  Why: ISRO is normally spoken as a word, not letter by letter; no other token's policy changed without evidence.
  Verified: `test_word_mode_acronym_leaves_text_unchanged` added; English suite re-run (see next row).

- `tests/test_normalise.py` (English) — Letters/word acronym test now uses ASR (letters) next to NASA (word); new test asserts word-mode ISRO leaves text and rules unchanged.
  Why: The old test asserted ISRO in letters mode.
  Verified: Ran: English 73 tests, OK (1 skip).

- `tests/test_max_seq_len.py` (both) — NEW. Pins the observed `max_seq_len` behaviour without changing it: training forward+loss with a 1005-frame mel batch succeeds with 1000-frame outputs (decoder truncates at `transformer/Models.py:154`, loss slices targets at `model/loss.py:43`); decoder truncation asserted in train mode; full length kept in eval mode; encoder raises RuntimeError beyond 1001 phonemes.
  Why: Audit request: state the exact behaviour for mel sequences exceeding `model.max_seq_len`.
  Verified: Ran: 4 tests per language, OK (English 4.9 s, Telugu 4.6 s).

- `docs/REQUIREMENTS_AUDIT.md`, `README.md` (root) — Appended audit section 12: mel > `max_seq_len` is silent truncation in training (~11.6 s at 22050 Hz / hop 256), no truncation at eval/inference (O(L²) memory), encoder text path errors beyond 1001 phonemes, no preprocessing duration cap; README test paragraph updated.
  Why: The audit must record the finding with file/line citations.
  Verified: Behaviour asserted by the new tests; no training-design change made.

- `kaggle/inspect_utils.py` (root) — `duration_stats` now also emits p50 and p90 (p99 already existed); the duration table renders p50/p90/p99; a per-column sampling-rate histogram table (rate, clips, share) was added to the Audio section; old JSON reports without the new keys rebuild with "not measured" instead of failing.
  Why: Report request: duration percentiles p50/p90/p99 and a sampling-rate histogram.
  Verified: Hours per group already existed (group tables + CSV); covered by a new test instead of duplicating.

- `kaggle/tests/test_inspect_utils.py` (root) — NEW tests: percentile values on 1..100 (p50=50.5, p90=90.1, p99=99.01), duration table header/row, single- and mixed-rate histogram rows, group table hours cells, old-JSON rebuild tolerance.
  Why: Rule 11 / report request.
  Verified: Ran: `python -m pytest kaggle/tests -q` -> 15 passed (1 warning); `python kaggle/tests/test_inspect_utils.py` -> 15 tests OK.

- `kaggle/INSPECT_GUIDE.md` (root) — Documented the new report fields and that the committed `docs/dataset_report.md` predates them.
  Why: The committed report keeps its measured numbers until the next Kaggle rerun; no local data exists to regenerate it.
  Verified: Guide text only.

- verification run (no code change) (root) — Rebuilt the report from the existing Telugu JSON (`dataset_report_indictts_telugu.json`, generated 2026-11-03 10:47 UTC) via `--rebuild-only` in a temp directory, writing `FastSpeech_Data/dataset_report_telugu_rebuilt.md` outside the repository; surveyed the preprocessing/alignment pipeline for long-clip decision inputs.
  Why: Regenerate without re-downloading; information gathering for the filter-vs-chunk decision (no policy implemented).
  Verified: Rebuild rendered: p99 = 60.398 s (present in the old JSON), p50/p90 = "not measured" (keys absent from the pre-update JSON; a Kaggle rerun fills them), histogram 16000 Hz x 4043 (47.1%) / 48000 Hz x 4533 (52.9%), gender hours female 4,043 clips / 23.9210 h, male 4,533 / 12.7913 h. Suites re-run green (77/82/36/15/4 tests).

- `docs/dataset_report_indictts_telugu.json` (added by the human, not written here), `docs/dataset_report.md` (root) — The measured Telugu JSON was copied into `docs/`; `docs/dataset_report.md` was regenerated in place from it with `python kaggle/inspect_datasets.py --rebuild-only`, replacing the "Not generated yet" placeholder. English (Svarah) still has no JSON, so the report covers Telugu only.
  Why: Phase 1 done-criterion requires real numbers in `docs/dataset_report.md`.
  Verified: Rebuild printed `Wrote docs/dataset_report.md from 1 report(s)`. Cross-checked every rendered figure against the JSON: 8,576 rows, 36.7123 h, 0 audio failures, 16000 Hz x 4,043 / 48000 Hz x 4,533, gender female 23.9210 h / male 12.7913 h, p99 60.398 s, p95 39.987 s. p50 and p90 render "not measured": the JSON predates the percentile change and has no `p50`/`p90` keys (it does contain `median` = 10.967 s). Licence field is empty (`license: null` on the Hub card).

### 2026-11-02

- `audio/`, `model/`, `preprocessor/`, `text/`, `transformer/`, `utils/`, `dataset.py`, `evaluate.py`, `hifigan/`, `LICENSE` (both) — Copied unmodified from the reference repository into both language folders.
  Why: Each language folder must be self-contained; MIT licence kept.
  Verified: `tests/smoke_test.py` (forward pass).

- `hifigan/generator_universal.pth.tar.zip` (both) — Copied the reference pretrained HiFi-GAN universal generator (52 MB).
  Why: Waveform generation; pretrained vocoder is allowed by the rules.
  Verified: Not loaded yet (no checkpoint).

- `lexicon/librispeech-lexicon.txt` (English) — Copied from the reference repo.
  Why: Required by the reference English g2p/lexicon frontend.
  Verified: Not exercised yet (frontend runs at inference).

- `utils/config.py` (both) — NEW. `load_config()` + `split_config()` for one YAML file.
  Why: Assignment requires one `config.yaml` per language with `preprocess`, `model`, `train`, `inference` sections.
  Verified: Loaded by both smoke tests.

- `config.yaml` (both) — NEW. Reference LJSpeech values reorganised into the four required sections; relative paths; `multi_speaker: true`; `vocoder.speaker: universal`; `save_step: 10000`.
  Why: Single per-language config; 22.05 kHz must match the HiFi-GAN config.
  Verified: Both smoke tests load it.

- `train.py` (both) — Replaced `-p/-m/-t` YAML arguments with `-c/--config`; config read through `utils.config`; removed unused `yaml` import.
  Why: One config file per language.
  Verified: `py_compile`; not run (needs preprocessed data).

- `preprocess.py` (both) — Reads the `preprocess` section of `config.yaml` (argument optional, defaults to `config.yaml`).
  Why: One config file per language.
  Verified: `py_compile`; not run (needs alignment data).

- `inference.py` (both) — NEW, adapted from the reference `synthesize.py` single-sentence mode. Reads `inference` config + `docs/test_sentences.json`, uses only `final`, stops if any is null, saves `<ID>.wav` to `Results/<Language>/test_generated_audios/`.
  Why: Required by the folder structure; rule 8 (never edit/assume test sentences).
  Verified: Null-refusal run for both languages (exit code 1, lists missing IDs); full synthesis not run (no checkpoint).

- `inference.py` (Telugu) — `LANGUAGE_SECTION = "Telugu"`; text frontend raises `NotImplementedError` for `language: te`.
  Why: Telugu symbol set/frontend is a GATE 1 / Phase 3 decision; must fail loudly, not guess.
  Verified: `py_compile`; null-refusal run.

- `tests/smoke_test.py` (both) — NEW. Builds the model from `config.yaml` and runs one forward pass on a tiny random batch (B=2, T=12, 36/27 frames), CPU, no checkpoint.
  Why: Phase 0 acceptance criterion.
  Verified: Ran for both languages: forward pass OK, 35,159,617 parameters.

- `checkpoints/README.md` (both) — NEW. Records that no checkpoint exists yet and where future ones go.
  Why: Checkpoint provenance / 100 MB rule.
  Verified: n/a.

- `Results/English/**`, `Results/Telugu/**`, `Report/`, `English/manifests/`, `Telugu/manifests/` (both) — NEW empty `.gitkeep` placeholders in the required deliverable folders.
  Why: Keep the exact folder tree in git before any output exists.
  Verified: Folders exist; no audio, plot, report or manifest content.

- `requirements.txt`, `.gitignore`, `README.md`, `THIRD_PARTY_NOTICES.md` (root) — NEW.
  Why: Required deliverables / licence credit / secret and data exclusions.
  Verified: `requirements.txt` pin set not yet installed on Kaggle (reported as unverified).

- `kaggle/inspect_utils.py` (root) — NEW. Pure helpers: audio-header probing, duration buckets, text/character stats, group stats, markdown renderer.
  Why: Phase 1 needs measured numbers and a report written only from measured values.
  Verified: 10 unit tests, all pass.

- `kaggle/inspect_datasets.py` (root) — NEW. Schema-agnostic inspector: enumerates configs/splits, finds the audio column by feature type, counts hours per column and per low-cardinality group, dumps JSON and rebuilds the markdown report.
  Why: Phase 1; do not assume column names.
  Verified: Helper tests pass; full dataset run NOT executed (no Kaggle session / HF token).

- `kaggle/tests/test_inspect_utils.py` (root) — NEW. Synthetic WAV/text/group tests, including a "not measured" render path.
  Why: Rule 11 (tests).
  Verified: Ran: 10 tests, OK.

- `kaggle/INSPECT_GUIDE.md` (root) — NEW. Exact Kaggle steps, secret handling, disk notes, failure handling.
  Why: The human runs Phase 1 on Kaggle.
  Verified: Not run (no Kaggle account).

- `docs/dataset_report.md` (root) — NEW stub stating it is not generated and will be overwritten by the inspector.
  Why: Honesty: no invented dataset numbers.
  Verified: Inspector unit tests only; report not generated.

- `docs/REQUIREMENTS_AUDIT.md` (root) — NEW requirements audit: requirement-by-requirement status, defects, prioritized fixes and pre-Kaggle steps. No code changed.
  Why: Human requested a complete audit before implementing the next phase.
  Verified: Commands and outputs recorded in the audit (smoke tests, 10 helper tests, compile check, null-refusal loader, Telugu symbol probe).

- `text/telugu.py` (Telugu) — NEW. Character-level frontend: NFC normalization, whitespace/format-control handling, punctuation folding, `encode_text`, `decode_text`, `encode_alignment_tokens`, unknown-character/symbol/token errors.
  Why: D2: the Telugu config was using the English ARPAbet frontend.
  Verified: `tests/test_telugu_frontend.py`: 22 tests, OK.

- `text/symbols.py` (Telugu) — Replaced the English/ARPAbet/pinyin symbol set with a frozen Telugu block set (explicit code-point ranges), ASCII punctuation/digits and `@sp/@spn/@sil`.
  Why: D2; IDs must not change with the local `unicodedata` version.
  Verified: Same 22 tests; symbol uniqueness and required-symbol assertions pass.

- `text/cleaners.py` (Telugu) — Added `telugu_cleaners` delegating to `text.telugu.normalize_text`; reference cleaners retained.
  Why: D2.
  Verified: Frontend tests cover the cleaner through `text_to_sequence`.

- `text/__init__.py` (Telugu) — Routes plain text to `encode_text` and `{...}` input to `encode_alignment_tokens`; `sequence_to_text` uses `decode_text`; unknown cleaner raises `ValueError`.
  Why: D2.
  Verified: Frontend tests: plain/alignment encode and decode.

- `inference.py` (Telugu) — Added the `language: te` branch; unsupported languages still raise `NotImplementedError`.
  Why: D2.
  Verified: Frontend probe `తెలుగు భాష` -> 10 symbols and exact decode; full synthesis not run (no checkpoint).

- `config.yaml` (Telugu) — `text_cleaners: ["telugu_cleaners"]`; added `manifest_train`/`manifest_val`; removed `val_size`; updated comments.
  Why: D2/D6/D17.
  Verified: `utils.config.load_config`, smoke test and config tests pass.

- `config.yaml` (English) — Added `manifest_train`/`manifest_val`; removed `val_size`; updated comments.
  Why: D6/D17.
  Verified: `utils.config.load_config`, smoke test and config tests pass.

- `audio/stft.py` (both) — `pad_center(size=...)` and `librosa_mel_fn(sr=..., n_fft=...)` keyword arguments; `.cuda()` removed, bases/window sums/mel basis follow the input device.
  Why: D4/D5.
  Verified: `py_compile`; CPU STFT tests written but skipped locally (librosa missing).

- `audio/audio_processing.py` (both) — `pad_center(win_sq, size=n_fft)` keyword argument (third incompatible call found by grep).
  Why: D4.
  Verified: `py_compile`; exercised by the skipped CPU STFT inverse test.

- `tests/test_audio_cpu.py` (both) — NEW. CPU mel/energy, NumPy helper and inverse-STFT checks; skipped when librosa/scipy/torch are unavailable.
  Why: D5 / rule 11.
  Verified: Ran: 3 tests, all skipped locally (librosa missing).

- `train.py` (both) — Replaced `batch_size * group_size < len(dataset)` with `batch_size <= len(dataset)` and a message.
  Why: D11: a 50-utterance overfit run was impossible.
  Verified: `py_compile`; both smoke tests pass; overfit run not executed.

- `utils/config.py` (both) — Added `validate_config`/`ConfigError`: sections, required fields, relative paths, language/cleaner match, audio/STFT/mel ranges, model dimensions, optimizer/step values, inference controls; `load_config` and `split_config` validate.
  Why: Item 10: fail before preprocessing/training.
  Verified: `tests/test_config_validation.py`: 23 tests per language, OK; both real configs validate.

- `preprocessor/manifest.py` (both) — NEW. Manifest CSV loader/validator (columns, durations, duplicate clips, train/val collision, path traversal); no audio imports, so it is unit-testable.
  Why: D6/D7.
  Verified: `tests/test_preprocessor_manifest.py`: 9 tests per language, OK.

- `preprocessor/preprocessor.py` (both) — Manifest-driven processing (no shuffle/`val_size`); deterministic speaker IDs; explicit `ClipSkipped` reasons for missing/unreadable/invalid audio, TextGrid, transcript and frontend length mismatches; `preprocessing_log.csv`; abort before normalization on any failure; explicit resampling/mono; normalizes only files from this run.
  Why: D6/D7, items 5 and 9 (CPU execution, failure logging).
  Verified: Manifest loader tests pass; full preprocessing NOT run (no corpus/alignment, librosa/pyworld/tgt missing locally).

- `scripts/split_dataset.py` (root) — NEW. Deterministic SHA-256-seeded, duration-targeted split; transcript groups kept atomic; validation speaker coverage; unused CSV option; validates manifests and refuses accidental overwrite.
  Why: D6.
  Verified: `scripts/tests/test_split_dataset.py`: 16 tests, all pass.

- `scripts/check_no_overlap.py` (root) — NEW. Explicit duplicate-clip, cross-split clip, shared-transcript, speaker-coverage and optional duration-target checks; non-zero exit on violations.
  Why: D6.
  Verified: `scripts/tests/test_check_no_overlap.py`: 7 tests, all pass.

- `scripts/tests/test_split_dataset.py` (root) — NEW. Determinism/order independence, duration targets, atomic transcript groups, speaker coverage, invalid manifests, path traversal.
  Why: Rule 11.
  Verified: Ran: included in 23 script tests, OK.

- `scripts/tests/test_check_no_overlap.py` (root) — NEW. Clean/duplicate/shared-transcript/missing-speaker/duration/empty-split cases.
  Why: Rule 11.
  Verified: Ran: included in 23 script tests, OK.

- `tests/test_telugu_frontend.py` (Telugu) — NEW. NFC composition, whitespace/control/punctuation normalization, symbol set, encode/decode roundtrip, unknown characters/symbols, alignment tokens and silence mapping.
  Why: Rule 11 / item 2.
  Verified: Ran: 22 tests, OK.

- `tests/test_config_validation.py` (both) — NEW. Valid real config plus 22 invalid mutations covering sections, fields, cleaners, language, audio/STFT/mel, model, optimizer/steps, paths and inference.
  Why: Rule 11 / item 10.
  Verified: Ran: 23 tests per language, OK (Telugu: 22 pass + 1 English-only skip).

- `tests/test_preprocessor_manifest.py` (both) — NEW. Manifest loading/validation without the audio stack.
  Why: Rule 11 / item 9.
  Verified: Ran: 9 tests per language, OK.

- `README.md` (root) — Updated status, layout, manifest/preprocessing commands, unit-test commands; documented the failure log and that dataset export is still missing.
  Why: Keep README accurate after the infrastructure changes.
  Verified: Commands match the implemented CLIs; no training claimed.

- `docs/REQUIREMENTS_AUDIT.md` (root) — Appended section 8: infrastructure fixes, resolved/partial defects, and what remains open.
  Why: Keep the audit current without rewriting the historical audit.
  Verified: Results above; preprocessing/training still unverified.

- `docs/REQUIREMENTS_AUDIT.md` (root) — Appended section 9: PDF found at `docs/FastSpeech2_Assignment (1).pdf`; confirmed requirements and listed 11 differences/corrections (deadline, from-scratch source, form name, paper citation, report comparison wording, sentence spacing, checkpoint link rule, extras approval); marked D1 resolved and A15 met; corrected A4/A5/H5/E13/A7/I6.
  Why: Human supplied the PDF and asked for re-verification against it.
  Verified: `pdftotext` + `pypdf 6.17.0` on all 3 pages; `docs/test_sentences.json` SHA-256 still `420187d1...`.

- `requirements.txt` (root) — Re-pinned for Kaggle Linux Python 3.12: NumPy 2.0.2, SciPy 1.13.1, librosa 0.10.2.post1, numba 0.60.0, pyworld 0.3.5, scikit-learn 1.5.2, tgt 1.5, matplotlib 3.9.2; added `datasets`, `huggingface_hub`, `nltk`, `pytest`, Cython/setuptools/wheel; documented the two-step pyworld source build.
  Why: `tgt==1.4.4` had no wheel, pyworld is source-only on Linux, and the old NumPy 1.26 pins were not validated on Kaggle 3.12.
  Verified: Candidate set resolved for Linux cp312 (excluding pyworld/g2p_en); pyworld 0.3.5 API exercised on a Python 3.11 Windows wheel; Linux pyworld build and full install NOT yet run.

- `audio/tools.py` (both) — Fixed `_stft._stft_fn` -> `_stft.stft_fn` in `inv_mel_spec`; moved the mel basis to the feature device.
  Why: The old attribute does not exist, so the Griffin-Lim helper would raise `AttributeError`; device handling keeps the path CPU/GPU safe.
  Verified: `test_inv_mel_spec_writes_audio` passes.

- `tests/test_audio_cpu.py` (both) — Expanded 3 -> 7 tests: exact librosa mel-filter match, sine-wave inverse reconstruction error < 1e-4, Griffin-Lim WAV output, and a static no-`.cuda()` check on `stft.py`, `audio_processing.py`, `tools.py`.
  Why: Item 1: prove the pinned librosa API and CPU path rather than only checking shapes.
  Verified: Ran 7 tests per language: OK locally and in a Python 3.11 venv with numpy 2.0.2, scipy 1.13.1, librosa 0.10.2.post1, numba 0.60.0, torch 2.15.0+cpu.

- `README.md` (root) — Added the Kaggle Python 3.12 install recipe and pyworld build warning; noted the PDF location; documented the expanded audio tests and Git LFS/external checkpoint option.
  Why: Keep setup instructions consistent with the new pins and the PDF.
  Verified: Commands match `requirements.txt`; full Kaggle install still unverified.

- `tests/smoke_test.py` (verification only) (both) — Re-ran both smoke tests inside the pinned Python 3.11 environment.
  Why: The previous pinned-venv run failed only because that incomplete venv lacked `unidecode`; confirm the pin set can build and run the model.
  Verified: English: 35,159,617 params, forward pass OK in 0.133 s; Telugu: 35,099,201 params, forward pass OK in 0.098 s. Venv: numpy 2.0.2, scipy 1.13.1, librosa 0.10.2.post1, numba 0.60.0, torch 2.15.0+cpu, pyworld 0.3.5, tgt 1.5.

- `.gitignore` (root) — Added `docs/*.pdf` and `docs/source/`.
  Why: The supplied assignment PDF must not be redistributed; it lives under `docs/`.
  Verified: `git check-ignore` not run (not a git repository); pattern inspected.

- `utils/train_utils.py` (both) — NEW. `set_seed`, RNG capture/restore, `append_loss_csv`, atomic `save_checkpoint`, `load_checkpoint` with `map_location`, `find_latest_checkpoint`, `checkpoint_path`.
  Why: Item 2: CSV logs, resume, safe save interval, final checkpoint.
  Verified: `tests/test_train_utils.py`: 6 tests, OK in both languages.

- `train.py` (both) — `group_size` now read from `train.optimizer.group_size`; `train.seed` applied; `--resume` picks the newest checkpoint; checkpoints save every `save_step` and again at `total_step`; train/val losses append to `logs/train/train_loss.csv` and `logs/val/val_loss.csv`; RNG state restored on resume; `quit()` replaced by an orderly return.
  Why: Item 2 / audit D11, E5, E7, E8, E9.
  Verified: `tests/test_overfit_batches.py` proves 50 clips batch with `batch_size: 8`, `group_size: 1`; `import train` OK in the pinned venv; training not run.

- `utils/model.py` (both) — Checkpoint load uses `load_checkpoint` (`map_location` + clear missing-file error).
  Why: Resume/inference must work when a GPU checkpoint is loaded on CPU.
  Verified: `tests/test_train_utils.py` exercises load/save; full model resume not run (no data).

- `evaluate.py` (both) — Returns `(message, loss_means)`; standalone CLI updated.
  Why: Item 2: train.py appends validation losses to CSV without parsing text.
  Verified: Imported successfully in the pinned venv; evaluation not run (no data/checkpoint).

- `config.yaml` (both) — Added `train.seed: 1234`, `train.optimizer.group_size: 4`; documented the overfit settings and the split seed.
  Why: Items 2-3.
  Verified: Both real configs validate; 56 English / 78 Telugu tests OK.

- `utils/config.py` (both) — Validate `train.seed >= 0` and `train.optimizer.group_size >= 1`.
  Why: Item 2: bad batch/group settings fail before training.
  Verified: `test_config_validation.py`: added 2 tests per language, OK.

- `preprocessor/report.py` (both) — NEW. `ClipSkipped(category, detail)`, `preprocessing_log.csv` columns with `category`, per-split/per-category summary, JSON writer, text formatter.
  Why: Item 3: report counts of dropped clips, including missing TextGrids.
  Verified: `tests/test_preprocessor_report.py`: 5 tests per language, OK.

- `preprocessor/preprocessor.py` (both) — Every skip now carries a category; missing WAV/transcript/TextGrid, unreadable TextGrid, alignment, frontend, audio, pitch and feature failures are counted; writes `preprocessing_summary.json`; strict by default (aborts before normalisation); `allow_dropped=True` continues and omits dropped clips from `train.txt`/`val.txt`; speaker IDs use successful clips only.
  Why: Item 3 / rule: never silently skip invalid files.
  Verified: Missing-TextGrid test OK in the pinned venv; full preprocessing not run (no corpus/TextGrids).

- `preprocess.py` (both) — Added `--allow-dropped` (default off).
  Why: Item 3: explicit opt-in to continue after counted drops.
  Verified: `py_compile`; CLI not run (no corpus).

- `tests/test_train_utils.py` (both) — NEW. CSV header/append/format, numeric latest-checkpoint lookup, atomic save/load, RNG round-trip, missing-file error.
  Why: Item 2 / rule 11.
  Verified: Ran: 6 tests per language, OK.

- `tests/test_overfit_batches.py` (both) — NEW. Builds 50 synthetic clips, batches them as `batch_size: 8`, `group_size: 1`, checks 6 micro-batches of 8.
  Why: Item 2: prove the 50-utterance overfit configuration can start.
  Verified: Ran: OK in both languages.

- `tests/test_preprocessor_report.py` (both) — NEW. Category/detail propagation, per-split and per-category counts, formatting, JSON output.
  Why: Item 3 / rule 11.
  Verified: Ran: 5 tests per language, OK.

- `tests/test_preprocessing_drops.py` (both) — NEW. Missing TextGrid is classified as `missing TextGrid`; skips when librosa/pyworld/tgt/sklearn are absent.
  Why: Item 3 / rule 11.
  Verified: Ran locally (skipped: stack missing) and in the pinned venv: OK.

- `tests/test_config_validation.py` (both) — Added zero `group_size` and negative `seed` cases.
  Why: Item 2.
  Verified: Ran: 56 English tests (1 skip) / 78 Telugu tests (2 skips), OK.

- `README.md` (root) — Documented drop categories/summary, `--allow-dropped`, seeded split command, CSV logs, resume, final checkpoint, overfit batch settings and the new tests.
  Why: Keep the operational docs current.
  Verified: Commands match the implemented CLIs; no run claimed.

- tests / smoke (verification) (both) — Re-ran English 56 tests (1 skip), Telugu 78 tests (2 skips), 23 script tests, compileall exit 0; both smoke tests OK (35,159,617 and 35,099,201 params); `import train` OK in the pinned Python 3.11 venv.
  Why: Item 2-3 acceptance.
  Verified: Outputs above; no training, preprocessing or checkpoint creation run.

- `docs/REQUIREMENTS_AUDIT.md` (root) — Appended section 10 updating E5, E7, E8, E9, D11 and the no-silent-skip rule to Met (code); B6 remains Missing.
  Why: Keep the audit aligned with items 2-3 without rewriting earlier sections.
  Verified: Verification numbers from the preceding row; no dataset or training run.

- `.gitignore`, `.gitattributes` (root) — Extended ignores for tool/local clutter (`.vscode/`, `.qoder/`, `.idea/`, `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `logs/`, `output/`, `data/`, `raw_data/`, `preprocessed_data/`); added `* text=auto` and `*.sh text eol=lf`.
  Why: First-push preparation: keep tool state, logs and data out of the repository; normalise line endings.
  Verified: `git check-ignore -v` confirmed the assignment PDF, caches, `English/data/` and `English/logs/` are ignored.

- `docs/REQUIREMENTS_AUDIT.md` (root) — Replaced the local absolute search path in section 0 with a `<user>`/`<session>` placeholder.
  Why: Pre-commit audit found a username-bearing path; the fact of the search is kept, the personal path is not published.
  Verified: Re-scan of staged content for the username, local user paths and e-mail addresses returned no matches.
