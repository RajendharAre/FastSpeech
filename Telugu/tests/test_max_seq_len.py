"""Pins the observed behaviour when a sequence exceeds model.max_seq_len.

Audit result (design unchanged; see docs/REQUIREMENTS_AUDIT.md section 12):

- Training: the decoder silently truncates its input to ``max_seq_len``
  frames (``transformer/Models.py``, Decoder branch
  ``max_len = min(max_len, self.max_seq_len)``) and ``model/loss.py`` slices
  the mel targets to the truncated mask, so a batch whose mel is longer than
  ``max_seq_len`` neither errors nor runs out of memory: the frames beyond
  ``max_seq_len`` never contribute to the mel/postnet loss.
- Validation runs with ``model.eval()`` (train.py calls eval() before
  evaluate) and inference.py likewise, so those paths compute the full
  sinusoid table on the fly and nothing is truncated.
- The encoder's phoneme path behaves differently: longer than
  ``max_seq_len + 1`` phonemes in training mode raises RuntimeError because
  the fixed positional table is only sliced, not grown.

``max_seq_len`` is 1000 frames; at 22050 Hz / hop 256 that is ~11.6 s.
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest

import torch

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

from model import FastSpeech2, FastSpeech2Loss  # noqa: E402
from utils.config import load_config, split_config  # noqa: E402
from utils.tools import get_mask_from_lengths  # noqa: E402


class MaxSeqLenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1234)
        config = load_config(os.path.join(LANG_DIR, "config.yaml"))
        cls.preprocess_config, cls.model_config, _ = split_config(config)
        cls.n_mels = cls.preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        cls.max_seq_len = cls.model_config["max_seq_len"]
        cls.tmp = tempfile.mkdtemp(prefix="fs2_max_seq_len_")
        with open(os.path.join(cls.tmp, "stats.json"), "w") as f:
            json.dump(
                {"pitch": [0.0, 1.0, 0.0, 1.0], "energy": [0.0, 1.0, 0.0, 1.0]}, f
            )
        if cls.model_config["multi_speaker"]:
            with open(os.path.join(cls.tmp, "speakers.json"), "w") as f:
                json.dump({"spk": 0}, f)
        preprocess_config = copy.deepcopy(cls.preprocess_config)
        preprocess_config["path"]["preprocessed_path"] = cls.tmp
        cls.model = FastSpeech2(preprocess_config, cls.model_config)
        cls.loss_fn = FastSpeech2Loss(cls.preprocess_config, cls.model_config)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def long_training_batch(self, target_mel_len):
        batch_size, max_src_len = 2, 20
        speakers = torch.zeros(batch_size, dtype=torch.long)
        texts = torch.randint(1, 20, (batch_size, max_src_len), dtype=torch.long)
        src_lens = torch.tensor([max_src_len, max_src_len - 5])
        durations = torch.zeros(batch_size, max_src_len, dtype=torch.long)
        durations[0] = target_mel_len // max_src_len
        durations[0, 0] += target_mel_len % max_src_len
        durations[1] = target_mel_len // max_src_len
        mel_lens = durations.sum(dim=1)
        max_mel_len = int(mel_lens.max())
        mels = torch.randn(batch_size, max_mel_len, self.n_mels)
        pitches = torch.randn(batch_size, max_src_len)
        energies = torch.randn(batch_size, max_src_len)
        return (
            ["clip_0", "clip_1"],
            ["raw 0", "raw 1"],
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    def test_training_batch_beyond_max_seq_len_is_silently_truncated(self):
        batch = self.long_training_batch(self.max_seq_len + 5)
        self.assertEqual(int(batch[8]), self.max_seq_len + 5)
        self.model.train()
        with torch.no_grad():
            output = self.model(*batch[2:])
            losses = self.loss_fn(batch, output)

        mel_out, postnet_out, mel_masks_out = output[0], output[1], output[7]
        self.assertEqual(
            mel_out.shape, (2, self.max_seq_len, self.n_mels)
        )
        self.assertEqual(
            postnet_out.shape, (2, self.max_seq_len, self.n_mels)
        )
        self.assertEqual(mel_masks_out.shape[1], self.max_seq_len)
        self.assertTrue(torch.isfinite(losses[0]).item())

    def test_decoder_truncates_training_input_to_max_seq_len(self):
        decoder = self.model.decoder
        decoder.train()
        mel_len = self.max_seq_len + 5
        seq = torch.randn(2, mel_len, decoder.d_model)
        mask = get_mask_from_lengths(
            torch.tensor([mel_len, mel_len - 5]), mel_len
        )
        out, out_mask = decoder(seq, mask)
        self.assertEqual(out.shape[1], self.max_seq_len)
        self.assertEqual(out_mask.shape[1], self.max_seq_len)

    def test_eval_decoder_keeps_full_length(self):
        decoder = self.model.decoder
        decoder.eval()
        mel_len = self.max_seq_len + 2
        seq = torch.randn(1, mel_len, decoder.d_model)
        mask = get_mask_from_lengths(torch.tensor([mel_len]), mel_len)
        with torch.no_grad():
            out, out_mask = decoder(seq, mask)
        self.assertEqual(out.shape[1], mel_len)
        self.assertEqual(out_mask.shape[1], mel_len)

    def test_encoder_training_beyond_max_seq_len_raises(self):
        encoder = self.model.encoder
        encoder.train()
        src_len = self.max_seq_len + 2
        texts = torch.randint(1, 20, (1, src_len), dtype=torch.long)
        src_masks = get_mask_from_lengths(torch.tensor([src_len]), src_len)
        with torch.no_grad(), self.assertRaises(RuntimeError):
            encoder(texts, src_masks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
