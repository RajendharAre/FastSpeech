# Dataset report

> **Not generated yet.** This file will be overwritten by
> `kaggle/inspect_datasets.py` with numbers measured from the real datasets.
> No dataset fact has been filled in by hand, and no number is copied from a
> dataset card.

## How this file is produced

The Phase 1 inspection runs on Kaggle (Linux) because Svarah is gated and the
Telugu download is large. See `kaggle/INSPECT_GUIDE.md` for the exact steps.

```bash
python kaggle/inspect_datasets.py --dataset svarah
python kaggle/inspect_datasets.py --dataset indictts_telugu
```

The script writes `docs/dataset_report_svarah.json` and
`docs/dataset_report_indictts_telugu.json`, then rebuilds this file from those
JSON reports.

## What will appear here, per dataset

- The Hugging Face revision sha, inspection timestamp and library versions.
- Licence and `gated` fields, quoted from the Hugging Face API.
- Every config and split, with row counts and the declared feature types.
- The audio column (found by feature type, not by name), its sampling rate(s),
  measured total hours, and the clip-duration distribution.
- Clip counts and hours for every low-cardinality column, including the column
  that identifies speakers, so the speaker strategy is decided from evidence.
- Transcript columns: character inventory, tag-like patterns, duplicate
  transcripts and repeats across groups.
- Warnings listing every clip whose audio header could not be read, so no
  failure is silently dropped.

## Why it is empty in this commit

The assistant has no Kaggle session and no Hugging Face token, and the rules
forbid inventing numbers. The inspection scripts and their helper functions are
unit-tested on synthetic data (`kaggle/tests/test_inspect_utils.py`); the
dataset numbers must come from the human's Kaggle run.

Until this file is regenerated, treat `docs/DATA_STRATEGY.md` facts as
**unverified**.
