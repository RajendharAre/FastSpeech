You are my coding assistant for a take-home assignment: FastSpeech 2 text-to-speech for English and Telugu.

Before doing anything, read these files in this order and treat them as binding:
docs/RULES_FOR_AI.md, docs/PROBLEM_STATEMENT.md, docs/DELIVERABLES_CHECKLIST.md, docs/TEST_SENTENCES.md,
docs/DATA_STRATEGY.md, docs/KAGGLE_WORKFLOW.md, docs/PLAN_PHASES.md, docs/REPORT_OUTLINE.md.
The original assignment PDF is the source of truth.

Your job right now is Phase 0 and Phase 1 from docs/PLAN_PHASES.md only.

Phase 0: create the exact folder structure from the assignment. Copy the needed code from
https://github.com/ming024/FastSpeech2 (MIT license, keep the license and credit it) into English/ and Telugu/ so each folder
is self-contained. Create requirements.txt, .gitignore and keep CHANGES.md up to date. Run a CPU smoke test (one forward pass
with a tiny random batch) to prove the copied code imports and runs.

Phase 1: write scripts that inspect the two Hugging Face datasets (ai4bharat/Svarah and SPRINGLab/IndicTTS_Telugu) and
produce docs/dataset_report.md with REAL numbers (columns, sampling rate, total hours, speakers, duration distribution,
license and gating). Do not assume column names. I will run the inspection on Kaggle, so give me exact steps.

Rules that matter most: do not invent results, numbers or audio-quality claims. Do not train long jobs locally (I am on
Windows with CPU only; smoke tests only). Log every change in CHANGES.md. Never commit tokens, datasets or files over 100 MB.
Never edit the test sentences. Train from scratch (no pretrained TTS acoustic model). If anything is unclear or conflicts
with the PDF, stop and ask me.

When finished, report: files created or changed, commands run with real output, what you could not verify, and the decisions
I must make next. Then stop and wait for my review.
