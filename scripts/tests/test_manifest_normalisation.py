"""Tests for transcript provenance in split manifests."""
import os
import sys
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

from split_dataset import (  # noqa: E402
    ManifestRow,
    SplitError,
    load_english_normaliser,
    load_manifest,
    normalise_rows,
    write_manifest,
)


def row(clip_id="0001", speaker="spk0", transcript="Meet at 10:45 AM."):
    return ManifestRow(
        clip_id=clip_id,
        speaker=speaker,
        duration=1.0,
        duration_text="1.0",
        transcript=transcript,
        line_number=2,
    )


def fake_normaliser(text):
    if not text.strip():
        return "   ", ()
    return text.replace("10:45 AM", "ten forty-five a m"), ("time",)


class NormaliseRowsTests(unittest.TestCase):
    def test_attaches_normalised_text_and_rules(self):
        rows = normalise_rows([row()], fake_normaliser)

        self.assertEqual(rows[0].transcript, "Meet at 10:45 AM.")
        self.assertEqual(
            rows[0].transcript_normalised, "Meet at ten forty-five a m."
        )
        self.assertEqual(rows[0].rules_fired, "time")

    def test_empty_normalisation_raises(self):
        with self.assertRaises(SplitError):
            normalise_rows([row(transcript="   ")], fake_normaliser)


class ManifestColumnTests(unittest.TestCase):
    def test_provenance_columns_roundtrip(self):
        rows = normalise_rows([row()], fake_normaliser)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train.csv")
            write_manifest(path, rows)
            with open(path, encoding="utf-8") as handle:
                header = handle.readline().strip()
            loaded = load_manifest(path)

        self.assertEqual(
            header,
            "clip_id,speaker,duration,transcript,"
            "transcript_normalised,rules_fired",
        )
        self.assertEqual(loaded[0].transcript, "Meet at 10:45 AM.")
        self.assertEqual(
            loaded[0].transcript_normalised, "Meet at ten forty-five a m."
        )
        self.assertEqual(loaded[0].rules_fired, "time")

    def test_unnormalised_manifest_keeps_four_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "train.csv")
            write_manifest(path, [row()])
            with open(path, encoding="utf-8") as handle:
                header = handle.readline().strip()

        self.assertEqual(header, "clip_id,speaker,duration,transcript")


class EnglishNormaliserImportTests(unittest.TestCase):
    def test_loads_from_english_folder(self):
        normaliser = load_english_normaliser(os.path.join(REPO_ROOT, "English"))
        text, rules = normaliser("Start at 10:45 AM.")
        self.assertEqual(text, "Start at ten forty-five a m.")
        self.assertIn("time", rules)


if __name__ == "__main__":
    unittest.main()
