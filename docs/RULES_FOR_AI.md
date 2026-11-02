# Rules for the AI Coding Assistant

These rules are binding. If a rule blocks you, stop and ask the human. Do not work around it.

## 1. Authority and scope
- Order of authority: the assignment PDF, then `PROBLEM_STATEMENT.md`, then the other docs.
- Do only the phase the human assigned (see `PLAN_PHASES.md`). Stop at each human gate and wait.
- The human is responsible for this submission and will be asked to explain every choice. Keep code simple and commented.

## 2. Honesty
- Never invent results, loss values, durations, F0 numbers, audio-quality claims, or dataset facts.
- Every number in a doc or report must come from a file produced by a real run. Name that file next to the number.
- If you did not run a command, say "not run". Never say tests pass unless you ran them and can paste the output.
- Facts in `DATA_STRATEGY.md` came from web pages and are unverified. Verify them by inspecting the data before relying on them.
- Never claim speech "sounds natural" or "is intelligible". You cannot hear audio. Leave listening-based observations to the human.

## 3. Environment
- The human edits and runs smoke tests on a Windows laptop with no GPU. Give PowerShell commands for local steps.
- Heavy work (data download, alignment, training, inference) runs on Kaggle (Linux). Give bash commands for those.
- Do not start long training locally. Do not install the forced aligner on Windows.
- Pin or record versions of Python, PyTorch and key libraries in `requirements.txt` and the report.

## 4. Repository structure
- The folder tree in `DELIVERABLES_CHECKLIST.md` is mandatory and exact.
- Allowed extras: `docs/`, `kaggle/`, `CHANGES.md`, `THIRD_PARTY_NOTICES.md`, `.gitignore`, per-language `manifests/`.
- Each language folder is self-contained (it includes the supporting code it needs).
- Keep the reference implementation's license (MIT) and credit it in the README and `THIRD_PARTY_NOTICES.md`.

## 5. Code
- Make minimal, well-explained changes to the reference implementation.
- Mark every modified or new block with a comment: `# MODIFIED(English|Telugu|both): reason`.
- One `config.yaml` per language with sections `preprocess`, `model`, `train`, `inference`. Update the code to read it.
- No absolute paths. Fixed random seeds. Log training and validation losses to CSV as well as TensorBoard.
- Write small unit tests (see section 11) and run them.

## 6. Data and secrets
- Never commit tokens, raw audio, preprocessed data, or datasets. Tokens come from environment variables or Kaggle Secrets.
- Do commit the manifests: CSV lists of the selected train and validation clip IDs, with a script that reproduces them.
- Checkpoints: save weights only for the submission. If a file exceeds 100 MB, upload it to the Hugging Face Hub, put the link in the README, and add a note in `checkpoints/`.
- Respect dataset licenses. Cite Svarah (CC BY 4.0) and IndicTTS properly.

## 7. Models
- Train both models from scratch. Do not load pretrained TTS acoustic-model weights.
- A pretrained vocoder (HiFi-GAN) is allowed, but name it, give its source, and state its audio settings in the report.
- English and Telugu models must be separate, with no shared weights.

## 8. Test sentences
- Read them only from `docs/test_sentences.json`, field `final`. If any is `null`, stop with a clear error.
- Never edit them. Normalisation happens inside the language frontend. Report both original and normalised text.

## 9. Decisions that need the human (stop and ask)
Speaker strategy, Telugu aligner, Telugu symbol set, vocoder choice, subsampling criteria, any change to the folder structure.
When you ask, give 2 or 3 options with the evidence for each and your recommendation.

## 10. End-of-phase report (mandatory)
Files created or changed; commands run with their real output; what you could not verify; questions for the human.

## 11. Tests to write
- No overlap between train and validation (by clip ID, and by identical transcript text).
- The test-sentence loader refuses `null` values.
- Config loading for both languages.
- Text-normaliser examples, including E4 and T4 style inputs.
- Plot functions run on dummy data and write the expected file names.

## 12. Disclosure
Add one line to the README that an AI coding assistant was used for implementation help, and that the human reviewed the work.
`CHANGES.md` must list every modification.
