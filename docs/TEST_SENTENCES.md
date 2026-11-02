# Test Sentences

The PDF says: "Preserve the sentence text as provided."

## Problem: the text extracted from the PDF is unreliable

The sentences below were extracted from the PDF text layer. Telugu conjuncts and spacing are visibly corrupted
(for example broken letters and stray spaces), and English E3 shows "rhyth m" with a gap.
Do not trust these strings, and do not "fix" them yourself. The human will read the rendered PDF pages and fill in
the `final` field of each sentence in `docs/test_sentences.json`.

Rules for the assistant:
- Use only the `final` field. If any `final` is `null`, inference scripts must stop with a clear error message.
- Never edit, correct, normalise or retype the sentences in code or docs.
- Text normalisation (numbers, dates, abbreviations) belongs in the language frontend, not in the stored sentences.
  The report must show both the original sentence and the normalised form the model actually received.

## What each sentence probably tests (our interpretation, not stated in the PDF)

| ID | What to watch |
|---|---|
| E1 | Very long, rare word. Letter-to-sound conversion for out-of-vocabulary words. |
| E2a / E2b | "conduct" as noun vs verb (stress and pronunciation differ). Does the frontend choose by context? |
| E3 | Irregular spelling "rhythm" (and the odd gap, if the PDF really shows it). |
| E4 | Time, AM, date and year in digits, plus the word "andaroo" (a Telugu word written in English). |
| T1 | Long loanword (neuropharmacological) with conjunct consonants. |
| T2a / T2b | Same spelling, different meaning (grip vs silk). Does prosody differ? Probably not. |
| T3 | Geminate / conjunct consonants (duration of long consonants). |
| T4 | Digits, ordinal suffix, time and date in Telugu. Number-to-words handling. |

## Raw extraction (reference only, likely corrupted)

English:
- E1: The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.
- E2a: The orchestra's conduct during the ceremony was praised by the audience.
- E2b: The researchers will conduct an experiment to evaluate the proposed method.
- E3: The rhyth m of the orchestra changed unexpectedly.
- E4: The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.

Telugu (extraction is corrupted):
- T1: న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లషే ించారు.
- T2a: తాడుపై అతని పట్టుచాలా బలంగా ఉంది.
- T2b: ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది.
- T3: పులి అడ విలో నిశ్శబ్ంద గా నడిచింది.
- T4: ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ంధ గా ఉంచండి.

## Human checklist (before inference)

- [ ] Open the PDF and read each sentence as rendered. Fill every `final` in `docs/test_sentences.json`.
- [ ] Check E3 visually: is the gap really there? Keep it if yes, remove it if no.
- [ ] Ask a Telugu reader to confirm the five Telugu sentences character by character.
