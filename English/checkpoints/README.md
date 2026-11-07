# English checkpoints

Trained weights for the English model (Svarah, two speakers). The `.pth.tar`
files are **not in this repository**: each is about 400 MiB, above GitHub's
100 MB per-file limit, and `*.pth.tar` is git-ignored. That size is the weights
plus Adam's two moment buffers, roughly 35.1 M parameters x 4 bytes x 3 tensors;
report section 6 uses 418 MB for the same file. The assignment permits an
accessible external download link in the README in place of Git LFS, so the
shipped checkpoint is hosted externally and linked below.

## Shipped checkpoint

| Field | Value |
|---|---|
| Step | 60,000 |
| Filename | `60000.pth.tar` |
| Parameters | 35,159,617 |
| Validation total loss | 4.0910 |
| Selected by | listening: 7/10 against 5/10 for step 5,000 |
| Download link | pending |

I chose 60,000 by ear rather than by loss. Validation loss would have picked
35,000 (4.0838) and ASR WER would have picked 5,000 (1.04), but at WER near 1.0
that metric is saturated, and 5,000 sounds materially worse. Report section 7.4
gives all three rankings.

## Other checkpoints decoded for the report

Steps 5,000, 10,000, 35,000 and 60,000 were each decoded on the same five test
sentences through the shipping HiFi-GAN vocoder. Only 60,000 is shipped. The
prediction payloads kept in `Results/English/variance_adaptor_plots/` are the
35,000-step export; the per-checkpoint figures at 60,000 are the Kaggle
measurements recorded in report section 7.4.

## Payload and resume

`utils/train_utils.py:95` writes `{step, model, optimizer, rng_state}` to a
temporary file and moves it into place with `os.replace`, so an interrupted save
leaves a `.tmp` rather than a corrupt checkpoint. Because the optimizer and RNG
state are included and `ScheduledOptim` is constructed with `restore_step`, a
resume continues the learning-rate schedule instead of restarting it.

```bash
cd English
python train.py -c config.yaml --restore_step 60000
python inference.py --config config.yaml --restore_step 60000
```

`save_step` is 2,500 and `total_step` is 60,000 in `config.yaml`.

## Config used

`English/config.yaml`, sha256 prefix `edb1e4dcecba631c` (4,022 bytes). Verify with
`sha256sum English/config.yaml`. This is the file as it stands in the repository;
the run itself was launched from the Kaggle working copy.

## When the upload is done

Replace `pending` in the table above with the public URL, add the same URL to the
root `README.md`, and open it in a signed-out browser window to confirm it
downloads without a login. Google Drive interposes a virus-scan page on files
this size, so check that the link actually yields the file rather than only a
preview.