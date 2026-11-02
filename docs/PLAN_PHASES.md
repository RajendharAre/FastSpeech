# Plan and Phases

Today is Monday 02-11-2026. The deadline is 07-11-2026 midnight (ambiguous), so the working target is to submit on the
evening of 06-11-2026. That leaves about 4 days. Dates below are targets. Stop at every GATE for human review.

Time budget: this is tight. Prefer a shorter, honest training run with a good analysis over a long run that does not finish.
Overlap work where possible: while one language trains on the GPU, prepare the other language's data on a CPU session.
Decide the training step cap at Gate 4 from the measured seconds per step.

## Phase 0: Repository skeleton (Nov 2, today)
- Create the exact folder tree. Copy the needed code from https://github.com/ming024/FastSpeech2 into `English/` and `Telugu/`.
- Add `requirements.txt`, `.gitignore`, `CHANGES.md`, `THIRD_PARTY_NOTICES.md`.
- CPU smoke test: import the model and run one forward pass on a tiny random batch.
- Done when: tree matches the checklist and the smoke test output is pasted.
- GATE 0.

## Phase 1: Inspect the datasets (Nov 2, today)
- Write inspection scripts (see `DATA_STRATEGY.md`) and a Kaggle run guide. The human runs them.
- Done when: `docs/dataset_report.md` contains real numbers.
- GATE 1: the human makes the decisions listed in `DATA_STRATEGY.md`.

## Phase 2: Selection and split (Nov 3)
- Selection scripts, manifests, `check_no_overlap.py`, a data-preparation README section.
- Done when: about 5 h and about 1 h per language, no overlap, and real hours are reported.
- GATE 2.

## Phase 3: Frontend, alignment, features (Nov 3)
- English: reuse the reference frontend; adapt for Svarah transcripts.
- Telugu: symbol set, number normaliser, alignment approach (as decided), lexicon or mapping.
- `preprocess.py` for each language. Report clips dropped and why.
- Done when: processed datasets exist and a few alignments are spot-checked by the human.
- GATE 3.

## Phase 4: Overfit test and timing (Nov 3, evening)
- 50 utterances, short run, and a 200-step timing run. Done when: loss falls, a sample is synthesised, and seconds per step is known.
- GATE 4: the human sets the step cap for each language.

## Phase 5: Full training (Nov 3 night to Nov 5)
- Train English and Telugu as separate runs. Prefer one GPU notebook at a time; because of the deadline, two at once is acceptable if the quota meter allows it (the quota drains twice as fast).
- Outputs: loss CSVs, `training_loss.png`, `validation_loss.png`, checkpoints with a hub backup.
- GATE 5: the human reviews the curves and decides when to stop.

## Phase 6: Inference and variance-adaptor plots (Nov 5)
- `inference.py`: synthesise the five sentences per language with the final checkpoint.
- Export the model's predicted values, not ground truth:
  - duration per symbol (frames), bar plot labelled with symbols;
  - pitch (F0) in Hz and energy, denormalised with the saved statistics, at the level the config uses (phoneme or frame);
  - arrays saved as JSON next to the plots.
- Done when: 5 audio files and 15 plots per language exist with the agreed names.

## Phase 7: README and report (Nov 5 to 6)
- Follow `REPORT_OUTLINE.md`. The assistant drafts structure and fills numbers from files. The human writes listening and interpretation.
- README: setup, commands, checkpoint links, AI-assistance note.

## Phase 8: Final audit and submit (Nov 6, evening; Nov 7 is buffer only)
- Walk through `DELIVERABLES_CHECKLIST.md` item by item in a fresh clone. Check all links in a private browser window.
- Submit the Google Form.
