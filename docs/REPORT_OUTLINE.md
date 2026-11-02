# Technical Report Outline

The report must cover the seven items in `DELIVERABLES_CHECKLIST.md`. Suggested sections:

1. Introduction and approach: what FastSpeech 2 is (in the author's own words), which parts of the reference implementation were used.
2. Modifications: a table per language taken from `CHANGES.md` (file, change, reason). Telugu support in detail.
3. Data: per language, selection and subsampling strategy, hours achieved, speakers, preprocessing, split, no-overlap check. Use the real report numbers.
4. Training configuration: key hyperparameters, steps, batch size, hardware, wall-clock time, vocoder (name, source, audio settings).
5. Loss analysis: curves, convergence, instability, overfitting. Describe only what the plots show.
6. Test-sentence analysis: one row or subsection per sentence (see table below).
7. English vs Telugu comparison: limitations, failures, possible improvements, tied to the sentences.
8. Reproducibility: dependencies, config files, exact commands.

## Per-sentence table

| ID | Original text | Normalised text given to model | Symbols | Pronunciation (human) | Quality (human) | Duration pattern | F0 pattern | Energy pattern | Failures |
|---|---|---|---|---|---|---|---|---|---|

Who fills what:
- Assistant may fill: original and normalised text, symbols, and numeric facts taken from saved arrays (total predicted duration, mean and range of F0, energy range).
- Human writes: everything based on listening (pronunciation, quality, naturalness) and the interpretation of why the plots look as they do.

## Rules
- No claim without evidence from a saved file or a human listening note.
- Include failures and limits honestly. The assignment asks for them.
- Mention the data limits: Svarah is ASR-oriented and mixed-condition, and Telugu data per speaker is small.
