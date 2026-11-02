# Deliverables Checklist

Tick items only after verifying them (run the command, open the file). Do not tick from memory.

## Required folder structure (exact, from the PDF)

```
project/                      (this is the repository root)
├── English/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Telugu/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/
│   │       ├── duration/
│   │       ├── pitch/
│   │       └── energy/
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/
│       └── variance_adaptor_plots/
│           ├── duration/
│           ├── pitch/
│           └── energy/
├── Report/
│   └── Technical_Report.pdf
├── README.md
└── requirements.txt
```

Allowed extras (keep them minimal and mention them in the README): `docs/`, `kaggle/`, `CHANGES.md`, `THIRD_PARTY_NOTICES.md`, `.gitignore`, and per-language `manifests/`.
Supporting code copied from the reference implementation (model, transformer, text, hifigan, utils, preprocessor) goes inside each language folder, so each language folder is self-contained.

## Naming convention (our choice, not from the PDF)

- Audio: `Results/<Language>/test_generated_audios/<ID>.wav` with IDs E1, E2a, E2b, E3, E4 and T1, T2a, T2b, T3, T4.
- Plots: `Results/<Language>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png`.
- That is 5 audio files and 15 plots per language.

## Submission requirements (from the PDF)

1. Complete preprocessing, training and inference code for both languages.
2. Modified files from the selected GitHub implementation, and separate configuration files for English and Telugu.
3. Trained model checkpoints, separately for English and Telugu.
4. Training-loss and validation-loss plots for both models.
5. Generated audio samples for all five test sentences in each language.
6. Variance-adaptor plots for predicted duration, pitch (F0) and energy, for each test sentence, for both languages.
7. The specified folder structure in the GitHub repository.

## Technical report (`Report/Technical_Report.pdf`) must include

- Implementation approach, architecture components used from the reference implementation, and modifications made for English and Telugu.
- Dataset selection, subsampling, preprocessing, train/validation split, and training configuration for each language.
- Analysis of the training and validation loss curves: convergence, instability, overfitting.
- Observations on the generated speech for each test sentence: pronunciation and overall quality.
- Analysis of the variance-adaptor plots (duration, F0, energy) with explanations of the patterns, for each test sentence.
- Comparative observations between the English and Telugu models: limitations, failures, and possible improvements, based on the given test sentences.
- Reproducibility details: dependencies, config files, and commands to preprocess, train and run inference.

## GitHub

- [ ] Repository is public.
- [ ] All code, results, checkpoints (or accessible links) and the report are reachable.
- [ ] README links work without logging in.
- [ ] Repository URL submitted through the Google Form.
