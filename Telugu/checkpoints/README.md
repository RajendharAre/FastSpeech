# Telugu checkpoints

Trained weights for the Telugu model (IndicTTS Telugu, one male speaker). The
`.pth.tar` files are **not in this repository**: each is about 400 MiB, above
GitHub's 100 MB per-file limit, and `*.pth.tar` is git-ignored. That size is the
weights plus Adam's two moment buffers, roughly 35.1 M parameters x 4 bytes x 3
tensors. The assignment permits an accessible external download link in the
README in place of Git LFS, so the shipped checkpoint is hosted externally and
linked below.

## Shipped checkpoint

| Field | Value |
|---|---|
| Step | 60,000 |
| Filename | `60000.pth.tar` |
| Parameters | 35,099,201 |
| Validation total loss | 3.791061 |
| Selected by | listening: a slight preference for 60,000 over 5,000 |
| Download link | [Google Drive](https://drive.google.com/file/d/1oWwu6tj8OJB1LzTbzqOeZ2X-xfOOQmmK/view?usp=drive_link) |

I chose 60,000 by ear rather than by loss. Among saved checkpoints, step 5,000
had the lowest validation loss, 3.73799, and the overall validation minimum was
3.6590 at step 2,000, which was not a saved checkpoint. The two versions sounded
broadly similar and the preference for 60,000 is slight and subjective, not a
measured quality score. The listening review is not uniformly positive: T3 is
the clearest, while T1 degrades after its opening and T4 is mostly noisy.
Report section 8.2 records the sentence-level observations, and sections 7.5 and
7.7 give the curve and the loss ranking.

## Other checkpoints decoded for the report

Inference ran at steps 5,000, 35,000 and 60,000 on the same five test sentences
through the shipping HiFi-GAN vocoder. Only 60,000 is shipped. The 5,000-step
export is kept outside the repository and serves as the early reference in
report section 9; the 35,000-step export is not included.

## Payload and resume

`utils/train_utils.py:95` writes `{step, model, optimizer, rng_state}` to a
temporary file and moves it into place with `os.replace`, so an interrupted save
leaves a `.tmp` rather than a corrupt checkpoint. Because the optimizer and RNG
state are included and `ScheduledOptim` is constructed with `restore_step`, a
resume continues the learning-rate schedule instead of restarting it.

This run needed that path: a container restart cut it at step 36,695, and it was
resumed from `35000.pth.tar` with

```bash
cd Telugu
python train.py --config config.yaml --restore_step 35000
```

The resumed segment matched the dead run's own losses at the same steps inside
that run's step-to-step noise (report section 6). To decode the shipped weights:

```bash
cd Telugu
python inference.py --config config.yaml --restore_step 60000
```

`save_step` is 5,000 and `total_step` is 60,000 in `config.yaml`.

## Config used

`Telugu/config.yaml`, sha256 prefix `c85ff84ca14777ee` (4,463 bytes). Verify with
`sha256sum Telugu/config.yaml`. This is the file as it stands in the repository;
the run itself was launched from the Kaggle working copy.

## When the upload is done

The link opens the shared Google Drive file preview. Use its Download button to
save `telugu_60000.pth.tar`; the file is shared as Anyone with the link /
Viewer.