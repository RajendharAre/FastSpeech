# Problem Statement

Source of truth: the assignment PDF (`FastSpeech2_Assignment.pdf`) and the email that came with it.
If anything in these docs conflicts with the PDF, the PDF wins. Tell the human immediately.

- Issued by: Speech Processing Lab (SLFI), LTRC, IIIT Hyderabad. Position: Intern / Research Assistant, Voice AI models.
- Deadline: 07-11-2026 (Saturday), "midnight 12 am". The wording is ambiguous, so the working target is to submit by the evening of 06-11-2026.
- Late submissions are not considered.
- Submission: a public GitHub repository, plus the Google Form https://forms.gle/7tgHgKJLjpBXwcBj6

## Objective (from the PDF)

Implement and evaluate the FastSpeech 2 text-to-speech model separately for English and Telugu, referring to the original paper (https://arxiv.org/abs/2006.04558) and an available GitHub implementation (https://github.com/ming024/FastSpeech2).

The applicant must:
1. Understand the model architecture.
2. Adapt and run the existing codebase independently for each language.
3. Train two separate models on the specified datasets.
4. Generate speech samples for the given test sentences.
5. Analyse the duration, pitch and energy predictions of the variance adaptor for the given test sentences.

Use the GitHub implementation as a starting point and document the changes made for each language, particularly for Telugu support.

## Datasets (from the PDF)

| Language | Dataset | Hugging Face page |
|---|---|---|
| English | Svarah | https://huggingface.co/datasets/ai4bharat/Svarah |
| Telugu | IndicTTS Telugu | https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu (a Hugging Face copy of the IIT Madras IndicTTS data) |

Requirements:
- Approximately 5 hours of audio for training and 1 hour for validation, per language.
- Train the English and Telugu models separately, each on its own dataset.
- Specify the data selection and subsampling strategy for each dataset.
- Training and validation data must be separate, with no overlap.
- Document the dataset preparation and preprocessing steps for both languages.

## Test sentences

Five sentences per language: E1, E2a, E2b, E3, E4 and T1, T2a, T2b, T3, T4. See `TEST_SENTENCES.md`.
The sentence text must be preserved as provided.

## Deliverables

See `DELIVERABLES_CHECKLIST.md` for the exact folder structure, file list and technical-report contents.

## Hard constraints

- Two separate models, trained from scratch on the specified datasets. No pretrained TTS acoustic-model weights.
- Exact folder structure from the PDF.
- Public GitHub repository. Checkpoints over GitHub's 100 MB file limit must use Git LFS or an external download link in the README that stays accessible during evaluation.
- The report must contain reproducibility details: software dependencies, config files, and the commands for preprocessing, training and inference.
