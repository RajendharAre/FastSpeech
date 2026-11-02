# Kaggle inspection guide (Phase 1)

Goal: run `kaggle/inspect_datasets.py` on Kaggle and produce `docs/dataset_report.md`
with numbers measured from the real datasets. The script does not decode waveforms:
it reads audio headers, so the scan is fast. No GPU is needed.

**Nothing in this guide has been executed by the assistant: it has no Kaggle account
and no Hugging Face token.** The command names and the script were syntax-checked and
unit-tested locally (see `kaggle/tests/test_inspect_utils.py`), but the Kaggle steps
themselves are unverified.

## Before you start

1. **Hugging Face access to Svarah.** Open
   https://huggingface.co/datasets/ai4bharat/Svarah while logged in and accept the
   conditions. Svarah is gated; the account that accepts must own the token.
2. **Read token.** https://huggingface.co/settings/tokens -> "New token" -> role
   "Read". Do not paste it into a notebook cell.
3. **Kaggle secret.** In the notebook: *Add-ons -> Secrets -> Add secret*, name it
   `HF_TOKEN`, paste the token, and enable it for this notebook.
4. **Notebook settings.** *Settings -> Accelerator: None (CPU)*, *Internet: On*.
   Internet and GPU may require phone verification.
5. **Get the code into the notebook.** Either:
   - push this repository to GitHub and `git clone` it in the notebook, or
   - upload the repository folder as a **private** Kaggle Dataset and copy it into
     `/kaggle/working`.

## Exact steps

### Cell 1 - check the libraries (do not force upgrades blindly)

```python
import datasets, huggingface_hub, soundfile
print("datasets", datasets.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("soundfile", soundfile.__version__)
```

If `datasets` or `soundfile` is missing:

```python
!pip install -q "datasets>=3.0" "huggingface_hub>=0.25" soundfile
```

Record the versions in the technical report. If `pip` proposes to change `numpy`,
stop and check with the human: downgrading numpy can break the preinstalled stack.

### Cell 2 - get the code and set the token

```python
!git clone https://github.com/<your-account>/<your-repo>.git fs2
%cd fs2

import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
print("token loaded:", bool(os.environ["HF_TOKEN"]))  # never print the token itself
```

If you uploaded a Kaggle Dataset instead, replace the clone with a copy, for example:

```python
!cp -r /kaggle/input/<your-code-dataset> /kaggle/working/fs2
%cd /kaggle/working/fs2
```

### Cell 3 - inspect Svarah (English)

```python
!python kaggle/inspect_datasets.py --dataset svarah
```

Expected output: the gated flag, the licence field, the config and split names, the
row count, progress every 1000 rows, then a summary with rows, clips measured, total
hours and warnings. It writes:

- `docs/dataset_report_svarah.json`
- `docs/dataset_report.md` (rebuilt from every JSON report present)

If you get a `GatedRepoError`, the token's account has not accepted the Svarah
conditions, or the secret is not enabled for the notebook. Fix that and rerun.

### Cell 4 - inspect IndicTTS Telugu

```python
!python kaggle/inspect_datasets.py --dataset indictts_telugu
```

The download is large (the dataset card reports a 6.44 GB parquet repository), so
check free disk first:

```python
!df -h /kaggle/working /root/.cache
```

The script streams rows from the cached parquet files and reads headers; it does not
decode audio. It rebuilds `docs/dataset_report.md` so that both datasets appear.

### Cell 5 - read and download the report

```python
from IPython.display import Markdown, display
display(Markdown(open("docs/dataset_report.md", encoding="utf-8").read()))
!ls -lh docs/dataset_report*
```

Download the `.md`, the `.json` files and any `dataset_report_*_groups_*.csv` files
from the notebook's output panel. Commit them from the laptop:

```bash
git add docs/dataset_report.md docs/dataset_report_*.json docs/dataset_report_*_groups_*.csv
git commit -m "Phase 1: measured dataset report"
```

Never commit the Hugging Face cache, the datasets or the raw audio.

## If you run the two datasets in separate sessions

Each run rebuilds `docs/dataset_report.md` from the JSON files that are present, so
download `docs/dataset_report_svarah.json` after the first run, put it in `docs/` in
the repo, and rerun the second dataset (or run
`python kaggle/inspect_datasets.py --rebuild-only` afterwards).

## What the report answers (and what it does not)

Measured by the script:

- every config, split and column with its declared feature type;
- the audio column found by feature type, the sampling rate(s) and the total hours;
- min / percentiles / max of clip duration, plus duration buckets;
- clip count and hours for every low-cardinality column, so the speaker column can be
  chosen from evidence (Svarah has no speaker-name assumption in the script);
- transcript character inventory, tag-like patterns, duplicate transcripts, and how
  many transcripts repeat across groups;
- licence and `gated` fields, quoted from the Hugging Face API, with the revision sha.

Not decided by the script (human decisions at GATE 1, see `docs/DATA_STRATEGY.md`):
speaker strategy, subsampling, Telugu aligner, Telugu symbol set, vocoder use beyond
the settings already read from `hifigan/config.json`.

Known limits:

- Total hours are the sum of the audio-header durations for the rows the script could
  read. Clips whose header failed are counted in the warnings, not silently dropped.
- If the dataset card and the measured total disagree, the report keeps the measured
  value and the disagreement must be discussed in the technical report.
