# Kaggle Workflow

Check the current Kaggle interface. Details below may have changed.

## Division of labour
- Local laptop (Windows, CPU): write and edit code, unit tests, CPU smoke tests, git. No long jobs.
- Kaggle (Linux): data export, alignment, feature extraction, training, inference.
- Code reaches Kaggle by `git clone` of the public repository (or by attaching a Kaggle Dataset of the code).

## Notebook settings
- Settings, then Accelerator: GPU (P100 or T4) for training only. CPU for everything else, to save GPU quota.
- Settings, then Internet: On (Kaggle usually needs phone verification for GPU and internet).
- Add-ons, then Secrets: `HF_TOKEN` (read token). Add a write token only if uploading checkpoints to the Hugging Face Hub.
- GPU quota is roughly 30 hours a week and shared across notebooks. Check the quota meter. Prefer one GPU notebook at a time; with the deadline, two at once is acceptable if enough quota remains (it drains twice as fast).
- Interactive-session files disappear when the session ends. Save outputs to `/kaggle/working` and use "Save Version" so they persist, or upload them to the Hugging Face Hub.

## Stages (one set of notebooks per language: do not mix English and Telugu)

Stage A: data prep (CPU)
1. Log in to Hugging Face with the secret. Download the dataset (Svarah is about 1 GB). For the Telugu 48 kHz data, stream and keep only the selected clips if disk space is tight.
2. Run the selection script to produce manifests. Export only the selected clips (WAV and transcript) at the required sampling rate.
3. Run alignment (current Montreal Forced Aligner docs, or the aligner chosen at the Telugu gate).
4. Run `preprocess.py` to compute durations, pitch and energy, plus statistics.
5. Save the processed subset as a PRIVATE Kaggle Dataset (for example `fs2-english-preprocessed`). Do not publish the raw gated data.

Stage B: training (GPU, one language at a time)
1. Clone the repo, attach the processed Kaggle Dataset, install requirements.
2. Run `train.py` with the language's `config.yaml`.
3. Save a checkpoint every N steps (N set in config), keep the last few, and copy them to `/kaggle/working`.
4. Back up checkpoints to the Hugging Face Hub every few thousand steps, so an interrupted session loses little.
5. The script must be resumable (a `--restore_step` style option) and write losses to CSV.
6. For long runs, use "Save Version" then "Save and Run All" (commit), which should keep running in the background after you close the tab. Verify this on a short run first.

Stage C: inference and plots (CPU is fine)
1. Load the final checkpoint, read the finalised test sentences, and synthesise the five sentences.
2. Save audio and the predicted duration, pitch and energy arrays and plots into `Results/<Language>/`.
3. Download the results and commit them from the laptop.

## Before any long run
- Overfit test: train on about 50 utterances and confirm the loss drops and audio is generated.
- Short dry run of 200 steps to measure seconds per step and estimate the total time.
