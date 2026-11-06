"""Unit tests for utils/train_utils.py (no dataset or audio needed).

Run from English/ or Telugu/:
    python -m unittest tests.test_train_utils
"""
import csv
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

from utils.train_utils import (  # noqa: E402
    LOSS_COLUMNS,
    _cpu_state,
    append_loss_csv,
    capture_rng_state,
    checkpoint_path,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    set_seed,
)


class LossCsvTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "logs", "train_loss.csv")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_header_is_written_once_and_rows_are_appended(self):
        append_loss_csv(self.path, 100, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        append_loss_csv(self.path, 200, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        with open(self.path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], list(LOSS_COLUMNS))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][0], "100")
        self.assertEqual(rows[1][1], "1.000000")
        self.assertEqual(rows[2][0], "200")

    def test_wrong_loss_count_is_rejected(self):
        with self.assertRaises(ValueError):
            append_loss_csv(self.path, 1, [1.0, 2.0])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self.tempdir.name, "checkpoints")
        os.makedirs(self.ckpt_dir)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_find_latest_checkpoint_uses_numeric_step(self):
        self.assertIsNone(find_latest_checkpoint(self.ckpt_dir))
        for step in (100, 900, 1000):
            path = checkpoint_path(self.ckpt_dir, step)
            with open(path, "wb") as handle:
                handle.write(b"placeholder")
        with open(os.path.join(self.ckpt_dir, "latest.pth.tar"), "wb") as handle:
            handle.write(b"ignored")
        self.assertEqual(find_latest_checkpoint(self.ckpt_dir), 1000)

    def test_save_checkpoint_is_atomic_and_resumable(self):
        path = checkpoint_path(self.ckpt_dir, 300)
        save_checkpoint(
            path,
            {"weight": torch.zeros(2)},
            {"state": 1},
            300,
        )
        self.assertTrue(os.path.isfile(path))
        self.assertFalse(os.path.exists(path + ".tmp"))
        payload = load_checkpoint(path, torch.device("cpu"))
        self.assertEqual(payload["step"], 300)
        self.assertEqual(payload["optimizer"], {"state": 1})
        self.assertEqual(list(payload["model"]), ["weight"])
        self.assertIn("rng_state", payload)

    def test_missing_checkpoint_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(os.path.join(self.ckpt_dir, "1.pth.tar"), "cpu")


class RngTests(unittest.TestCase):
    def test_rng_state_round_trip(self):
        set_seed(1234)
        state = capture_rng_state()
        first = np.random.rand()
        restore_rng_state(state)
        second = np.random.rand()
        self.assertAlmostEqual(first, second)

    def test_empty_state_is_ignored(self):
        restore_rng_state(None)

    def test_rng_tensors_are_coerced_to_cpu(self):
        # torch.set_rng_state only accepts a CPU ByteTensor, but a checkpoint
        # loaded with map_location="cuda" hands back CUDA tensors. _cpu_state is
        # the guard that made the English resume possible. CUDA cannot be
        # exercised on this box, so the contract is pinned directly.
        moved = _cpu_state(torch.get_rng_state())
        self.assertEqual(moved.device.type, "cpu")
        self.assertEqual(moved.dtype, torch.uint8)
        self.assertEqual(_cpu_state(7), 7)

    def test_restore_accepts_a_rebuilt_tensor(self):
        set_seed(1234)
        state = capture_rng_state()
        expected = torch.rand(1).item()
        state["torch"] = state["torch"].clone()
        restore_rng_state(state)
        self.assertEqual(torch.rand(1).item(), expected)


if __name__ == "__main__":
    unittest.main()
