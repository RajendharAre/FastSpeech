# Data Strategy

## Facts gathered from web pages (UNVERIFIED: inspect the data before relying on any of them)

Svarah (English):
- AI4Bharat Indian-accented English speech benchmark, about 9.6 hours from 117 speakers, so roughly 5 minutes per speaker on average.
- Mix of read and spontaneous speech, designed for ASR evaluation, not studio TTS.
- License CC BY 4.0. Gated on Hugging Face: log in and accept the conditions with the same account as the token.
- Reported as a single split, so we make our own train/validation split.
- About 6 hours are needed (5 train + 1 validation), which is most of the corpus. A single-speaker model is not possible.

IndicTTS Telugu (SPRINGLab copy on Hugging Face):
- About 8.7 hours total, two speakers (male about 4.5 h, female about 4.3 h), audio at 48 kHz.
- No single speaker appears to have 6 hours in this copy. The original IIT Madras database may hold much more per speaker.
- Check the page for gating and license terms.

## First task: inspect (Phase 1), then decide

Write an inspection script that prints, from the real data:
- Column names and types, sampling rate, number of clips, total hours.
- Speaker identifiers and hours per speaker.
- Clip-duration distribution (min, median, max).
- Transcript samples, and any tags or symbols in transcripts (for example disfluency markers).
- Whether sentences repeat across speakers.
Save the results to `docs/dataset_report.md`. Do not assume any column name.

## Decisions for the human (after the report)

1. Speaker strategy per language: single speaker, or multi-speaker with the reference implementation's speaker embedding.
   Svarah must be multi-speaker. For Telugu choose between both speakers of the Hugging Face copy, or a single speaker from the original database.
2. Subsampling criteria (propose, then the human decides): duration range, clean transcripts, read speech only for Svarah if the metadata allows it, exclude noisy clips if a simple measure is available, balance across speakers.
3. Telugu forced aligner: check whether a pretrained Telugu acoustic model and dictionary exist for the Montreal Forced Aligner (current docs). If not, options include training an aligner on our own data with a character-level dictionary, or another multilingual forced aligner. Give evidence before choosing.
4. Telugu text symbols: characters (after Unicode normalisation) or a phoneme mapping. Include a Telugu number-to-words normaliser (needed for T4).
5. Vocoder compatibility: read the audio settings in the repository's HiFi-GAN config (sampling rate, mel bins, hop length). Do not guess. Resample all audio to match.

## Selection and split rules

- Target about 5 h train and about 1 h validation per language (about 10 percent tolerance). Record the exact hours achieved.
- No overlap: no shared clip ID, and no identical transcript text across the two splits (verify and document).
- For multi-speaker models, every validation speaker must also appear in training (different clips).
- Fix a random seed. Save manifests in `English/manifests/` and `Telugu/manifests/` (`train.csv`, `val.csv`) with clip ID, speaker, duration and transcript.
- Write `check_no_overlap.py` and run it in the tests.

## Preprocessing to document (the PDF asks for it)

Resampling, mono conversion, optional loudness handling, transcript cleaning, text normalisation rules, alignment method and its failures (clips dropped, with counts), pitch and energy extraction settings, statistics used for normalisation. Report real counts at every step.
