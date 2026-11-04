"""The 'too_long' mel guard used by preprocessing.

Frame counts are taken from the real STFT path (``audio/stft.py``), not from a
duration estimate. ``STFT.transform`` reflect-pads ``filter_length // 2``
samples on both sides and then runs conv1d with stride ``hop_length``, so a
clip of N samples yields ``1 + floor(N / hop_length)`` frames -- the same count
as ``librosa.stft(..., center=True)``.

Run from English/ or Telugu/:
    python -m unittest tests.test_too_long
"""
import os
import sys
import unittest

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

from preprocessor.report import (  # noqa: E402
    DEFAULT_MAX_SEQ_LEN_MARGIN,
    ClipSkipped,
    check_mel_length,
    mel_frame_limit,
)

AUDIO_IMPORT_ERROR = None
try:
    import numpy as np

    from audio.stft import TacotronSTFT
    from audio.tools import get_mel_from_wav
except ImportError as exc:  # pragma: no cover - depends on the local install
    AUDIO_IMPORT_ERROR = str(exc)


@unittest.skipIf(AUDIO_IMPORT_ERROR is not None, "audio dependencies missing")
class StftFrameCountTests(unittest.TestCase):
    def setUp(self):
        self.stft = TacotronSTFT(
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=80,
            sampling_rate=22050,
            mel_fmin=0,
            mel_fmax=8000,
        )

    def test_frame_count_is_one_plus_floor_samples_over_hop(self):
        for samples in (1024, 22050, 30011):
            mel, energy = get_mel_from_wav(
                np.zeros(samples, dtype=np.float32), self.stft
            )
            self.assertEqual(mel.shape[1], 1 + samples // 256, samples)
            self.assertEqual(energy.shape[0], mel.shape[1], samples)

    def test_matches_librosa_center_true_not_center_false(self):
        import librosa

        wav = np.zeros(22050, dtype=np.float32)
        mel, _ = get_mel_from_wav(wav, self.stft)
        centered = librosa.stft(wav, n_fft=1024, hop_length=256, center=True)
        uncentered = librosa.stft(wav, n_fft=1024, hop_length=256, center=False)

        self.assertEqual(centered.shape[1], 1 + wav.size // 256)
        self.assertEqual(uncentered.shape[1], 1 + (wav.size - 1024) // 256)
        self.assertEqual(mel.shape[1], centered.shape[1])
        self.assertNotEqual(mel.shape[1], uncentered.shape[1])


class MelLengthGuardTests(unittest.TestCase):
    def test_clip_at_the_limit_is_allowed(self):
        check_mel_length(980, 1000, 20)

    def test_clip_above_the_limit_is_too_long(self):
        with self.assertRaises(ClipSkipped) as caught:
            check_mel_length(981, 1000, 20)
        self.assertEqual(caught.exception.category, "too_long")
        self.assertIn("981", caught.exception.detail)
        self.assertIn("980", caught.exception.detail)

    def test_default_margin_is_20_frames(self):
        self.assertEqual(DEFAULT_MAX_SEQ_LEN_MARGIN, 20)
        check_mel_length(980, 1000)
        with self.assertRaises(ClipSkipped):
            check_mel_length(981, 1000)

    def test_margin_is_configurable(self):
        check_mel_length(995, 1000, 5)
        with self.assertRaises(ClipSkipped):
            check_mel_length(996, 1000, 5)

    def test_invalid_limits_are_rejected(self):
        for max_seq_len, margin in (
            (1000, -1),
            (1000, 1000),
            (1000, True),
            (1000, 1.5),
            (True, 20),
            (0, 0),
        ):
            with self.assertRaises(ValueError, msg=repr((max_seq_len, margin))):
                mel_frame_limit(max_seq_len, margin)


if __name__ == "__main__":
    unittest.main()
