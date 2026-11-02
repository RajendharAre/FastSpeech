"""CPU-only checks for the STFT used during preprocessing.

These run without CUDA. If the audio dependencies (librosa/scipy) are not
installed the tests are skipped instead of failing.

Run from English/ or Telugu/:
    python -m unittest tests.test_audio_cpu
"""
import os
import sys
import tempfile
import unittest

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

AUDIO_IMPORT_ERROR = None
try:
    import numpy as np
    import torch

    from audio.stft import TacotronSTFT
    from audio.tools import get_mel_from_wav, inv_mel_spec
except ImportError as exc:  # pragma: no cover - depends on the environment
    AUDIO_IMPORT_ERROR = str(exc)


@unittest.skipIf(AUDIO_IMPORT_ERROR is not None, "audio dependencies missing")
class StftCpuTests(unittest.TestCase):
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

    def test_mel_spectrogram_runs_on_cpu(self):
        torch.manual_seed(1234)
        wav = torch.rand(1, 22050) * 2 - 1

        mel, energy = self.stft.mel_spectrogram(wav)

        self.assertEqual(mel.device.type, "cpu")
        self.assertEqual(mel.shape[0], 1)
        self.assertEqual(mel.shape[1], 80)
        self.assertGreater(mel.shape[2], 0)
        self.assertEqual(tuple(energy.shape), (1, mel.shape[2]))
        self.assertTrue(torch.isfinite(mel).all())
        self.assertTrue(torch.isfinite(energy).all())

    def test_numpy_waveform_helper_runs_on_cpu(self):
        rng = np.random.RandomState(1234)
        wav = (rng.rand(22050) * 2 - 1).astype(np.float32)

        mel, energy = get_mel_from_wav(wav, self.stft)

        self.assertEqual(mel.ndim, 2)
        self.assertEqual(mel.shape[0], 80)
        self.assertEqual(energy.ndim, 1)
        self.assertEqual(energy.shape[0], mel.shape[1])
        self.assertTrue(np.isfinite(mel).all())
        self.assertTrue(np.isfinite(energy).all())

    def test_inverse_runs_on_cpu(self):
        torch.manual_seed(1234)
        wav = torch.rand(1, 22050) * 2 - 1

        magnitude, phase = self.stft.stft_fn.transform(wav)
        reconstruction = self.stft.stft_fn.inverse(magnitude, phase)

        self.assertEqual(reconstruction.device.type, "cpu")
        self.assertEqual(reconstruction.shape[0], 1)
        self.assertGreater(reconstruction.shape[2], 0)
        self.assertTrue(torch.isfinite(reconstruction).all())

    def test_mel_filterbank_matches_librosa(self):
        import librosa

        expected = librosa.filters.mel(
            sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000
        )
        np.testing.assert_allclose(
            self.stft.mel_basis.numpy(), expected, rtol=0.0, atol=1e-7
        )

    def test_inverse_reconstructs_a_sine_wave(self):
        sample_rate = 22050
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        wav = torch.from_numpy(
            (0.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        ).unsqueeze(0)

        magnitude, phase = self.stft.stft_fn.transform(wav)
        reconstruction = self.stft.stft_fn.inverse(magnitude, phase)

        # The transform trims half a window from both ends of the signal.
        length = min(reconstruction.shape[2], wav.shape[1])
        error = torch.sqrt(
            torch.mean(
                (reconstruction[0, 0, :length] - wav[0, :length]) ** 2
            )
        ).item()
        self.assertLess(error, 1e-4)

    def test_inv_mel_spec_writes_audio(self):
        torch.manual_seed(1234)
        mel, _ = self.stft.mel_spectrogram(torch.rand(1, 4096) * 2 - 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "reconstructed.wav")
            inv_mel_spec(mel[0], path, self.stft, griffin_iters=1)

            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_audio_sources_have_no_cuda_calls(self):
        for filename in ("stft.py", "audio_processing.py", "tools.py"):
            path = os.path.join(LANG_DIR, "audio", filename)
            with open(path, "r", encoding="utf-8") as handle:
                source = handle.read()
            self.assertNotIn(".cuda(", source, filename)


if __name__ == "__main__":
    unittest.main()
