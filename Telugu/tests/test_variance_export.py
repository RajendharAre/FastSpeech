"""Unit tests for the variance-adaptor export added to inference.py.

Run from Telugu/:
    python -m unittest tests.test_variance_export
"""
import json
import os
import sys
import tempfile
import unittest

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

IMPORT_ERROR = None
try:
    import numpy as np

    import inference
    from text import text_to_sequence
except Exception as exc:  # torch / matplotlib missing
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR, "inference cannot be imported: {}".format(IMPORT_ERROR))
class ExportTests(unittest.TestCase):
    def _arrays(self, n=5):
        return {
            "duration_frames": np.arange(1, n + 1, dtype=float),
            "duration_seconds": np.arange(1, n + 1, dtype=float) * 256 / 22050,
            "pitch_hz": np.linspace(120.0, 180.0, n),
            "energy": np.linspace(0.5, 1.5, n),
            "pitch_normalized": np.linspace(-1.0, 1.0, n),
            "energy_normalized": np.linspace(-1.0, 1.0, n),
        }

    def test_deliverable_directory_names(self):
        self.assertEqual(inference.AUDIO_DIR_NAME, "test_generated_audios")
        self.assertEqual(inference.VARIANCE_DIR_NAME, "variance_adaptor_plots")

    def test_denormalize_uses_mean_and_std(self):
        stats_block = [0.0, 10.0, 150.0, 30.0]
        values = inference.denormalize([-1.0, 0.0, 1.0], stats_block)
        self.assertEqual(list(values), [120.0, 150.0, 180.0])

    def test_symbols_from_ids_round_trips(self):
        sequence = text_to_sequence("తెలుగు", ["telugu_cleaners"])
        self.assertEqual("".join(inference.symbols_from_ids(sequence)), "తెలుగు")

    def test_ascii_labels_pass_through(self):
        labels, fontproperties = inference.symbol_labels(["AA", "B", "sp"])
        self.assertEqual(labels, ["AA", "B", "sp"])
        self.assertIsNone(fontproperties)

    def test_export_writes_three_plots_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            symbols = list("తెలుగు")
            paths = inference.export_variance_outputs(
                tmp,
                {"id": "T1", "final": "తెలుగు"},
                symbols,
                self._arrays(len(symbols)),
                {"checkpoint_step": 12000, "mel_frames": 42},
            )
            root = os.path.join(tmp, "variance_adaptor_plots")
            for kind in ("duration", "pitch", "energy"):
                plot = os.path.join(root, kind, "T1.png")
                self.assertTrue(os.path.isfile(plot), plot)
                self.assertGreater(os.path.getsize(plot), 1000)
            json_path = os.path.join(root, "T1.json")
            self.assertTrue(os.path.isfile(json_path))
            self.assertEqual(len(paths), 4)

            with open(json_path, encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["id"], "T1")
            self.assertEqual(payload["symbols"], symbols)
            self.assertEqual(payload["checkpoint_step"], 12000)
            self.assertEqual(payload["duration_frames"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            self.assertEqual(payload["pitch_hz"][0], 120.0)
            self.assertIn("pitch_normalized", payload)

    def test_check_phoneme_level_rejects_frame_level(self):
        config = {
            "preprocessing": {
                "pitch": {"feature": "phoneme_level"},
                "energy": {"feature": "frame_level"},
            }
        }
        with self.assertRaises(SystemExit):
            inference.check_phoneme_level(config)

    def test_load_feature_stats_requires_stats_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"path": {"preprocessed_path": tmp}}
            with self.assertRaises(SystemExit):
                inference.load_feature_stats(config)

            with open(os.path.join(tmp, "stats.json"), "w", encoding="utf-8") as f:
                json.dump({"pitch": [0, 10, 150, 30], "energy": [0, 1, 0.5, 0.1]}, f)
            self.assertEqual(inference.load_feature_stats(config)["pitch"][2], 150)


if __name__ == "__main__":
    unittest.main()
