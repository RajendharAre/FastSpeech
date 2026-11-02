"""Prove the 50-utterance overfit configuration can be batched.

Uses synthetic arrays only (no audio, no model). The batch counts mirror
train.py: DataLoader(batch_size * group_size) with collate_fn(batch_size).

Run from English/:
    python -m unittest tests.test_overfit_batches
"""
import os
import sys
import tempfile
import unittest

import numpy as np
from torch.utils.data import DataLoader

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

from dataset import Dataset  # noqa: E402
from text import text_to_sequence  # noqa: E402

ALIGNMENT_TEXT = "{HH AH0 L OW1}"
RAW_TEXT = "hello"
N_CLIPS = 50
OVERSAMPLE_BATCH = 8
GROUP_SIZE = 1


class OverfitBatchingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.preprocessed = os.path.join(self.tempdir.name, "preprocessed")
        for kind in ("mel", "pitch", "energy", "duration"):
            os.makedirs(os.path.join(self.preprocessed, kind))
        with open(os.path.join(self.preprocessed, "speakers.json"), "w") as handle:
            handle.write('{"spk0": 0}')

        phone_len = len(text_to_sequence(ALIGNMENT_TEXT, ["english_cleaners"]))
        lines = []
        for index in range(N_CLIPS):
            basename = "clip{}".format(index)
            np.save(
                os.path.join(
                    self.preprocessed, "mel", "spk0-mel-{}.npy".format(basename)
                ),
                np.zeros((30, 80), dtype=np.float32),
            )
            for kind in ("pitch", "energy"):
                np.save(
                    os.path.join(
                        self.preprocessed,
                        kind,
                        "spk0-{}-{}.npy".format(kind, basename),
                    ),
                    np.ones(phone_len, dtype=np.float32),
                )
            np.save(
                os.path.join(
                    self.preprocessed, "duration", "spk0-duration-{}.npy".format(basename)
                ),
                np.ones(phone_len, dtype=np.int64),
            )
            lines.append(
                "{}|spk0|{}|{}".format(basename, ALIGNMENT_TEXT, RAW_TEXT)
            )
        with open(
            os.path.join(self.preprocessed, "train.txt"), "w", encoding="utf-8"
        ) as handle:
            handle.write("\n".join(lines) + "\n")

        self.preprocess_config = {
            "dataset": "synthetic",
            "path": {"preprocessed_path": self.preprocessed},
            "preprocessing": {
                "text": {"text_cleaners": ["english_cleaners"]}
            },
        }
        self.train_config = {
            "optimizer": {
                "batch_size": OVERSAMPLE_BATCH,
                "group_size": GROUP_SIZE,
            }
        }

    def tearDown(self):
        self.tempdir.cleanup()

    def test_fifty_clips_can_be_batched_for_overfit(self):
        dataset = Dataset(
            "train.txt", self.preprocess_config, self.train_config,
            sort=True, drop_last=True,
        )
        self.assertEqual(len(dataset), N_CLIPS)
        self.assertLessEqual(OVERSAMPLE_BATCH, len(dataset))

        loader = DataLoader(
            dataset,
            batch_size=OVERSAMPLE_BATCH * GROUP_SIZE,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        microbatches = [
            microbatch for batchs in loader for microbatch in batchs
        ]
        # drop_last=True removes the final partial microbatch (48 used).
        self.assertEqual(len(microbatches), (N_CLIPS // OVERSAMPLE_BATCH))
        self.assertEqual(len(microbatches[0][0]), OVERSAMPLE_BATCH)


if __name__ == "__main__":
    unittest.main()
