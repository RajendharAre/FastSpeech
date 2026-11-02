"""Missing-input handling in Preprocessor.process_utterance.

Skips when the preprocessing stack (librosa, pyworld, tgt, scikit-learn) is not
installed locally. The missing TextGrid is detected before any audio is read.

Run from English/ or Telugu/:
    python -m unittest tests.test_preprocessing_drops
"""
import os
import sys
import tempfile
import unittest

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

try:
    from preprocessor.preprocessor import Preprocessor
    from preprocessor.report import ClipSkipped
except Exception as exc:  # pragma: no cover - depends on the local install
    Preprocessor = None
    ClipSkipped = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(Preprocessor is None, "preprocessing stack not installed")
class MissingInputTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.raw = os.path.join(self.tempdir.name, "raw")
        self.out = os.path.join(self.tempdir.name, "out")
        os.makedirs(os.path.join(self.raw, "spk"))
        os.makedirs(self.out)
        for name in ("clip.wav", "clip.lab"):
            with open(os.path.join(self.raw, "spk", name), "wb") as handle:
                handle.write(b"placeholder")

        self.preprocessor = object.__new__(Preprocessor)
        self.preprocessor.in_dir = self.raw
        self.preprocessor.out_dir = self.out

    def tearDown(self):
        self.tempdir.cleanup()

    def test_missing_textgrid_has_its_own_category(self):
        with self.assertRaises(ClipSkipped) as caught:
            self.preprocessor.process_utterance("spk", "clip")
        self.assertEqual(caught.exception.category, "missing TextGrid")
        self.assertIn("missing TextGrid file", caught.exception.detail)


if __name__ == "__main__":
    unittest.main()
