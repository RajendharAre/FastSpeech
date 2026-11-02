"""Unit tests for scripts/check_no_overlap.py.

Run from the repository root:
    python -m unittest discover -s scripts/tests -v
"""
import os
import sys
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

from check_no_overlap import check_manifests  # noqa: E402
from split_dataset import ManifestRow  # noqa: E402


def row(clip_id, speaker, transcript, seconds=3600.0, line=2):
    return ManifestRow(
        clip_id=clip_id,
        speaker=speaker,
        duration=seconds,
        duration_text=str(seconds),
        transcript=transcript,
        line_number=line,
    )


class OverlapTests(unittest.TestCase):
    def test_clean_manifests_have_no_violations(self):
        train = [row("a", "spk0", "train one"), row("b", "spk1", "train two")]
        val = [row("c", "spk0", "val one")]

        report = check_manifests(train, val)

        self.assertEqual(report["violations"], [])
        self.assertEqual(report["clips_in_both"], [])
        self.assertEqual(report["shared_transcripts"], [])
        self.assertEqual(report["val_speakers_missing_from_train"], [])

    def test_clip_in_both_splits_is_reported(self):
        train = [row("a", "spk0", "train one")]
        val = [row("a", "spk0", "val one")]

        report = check_manifests(train, val, require_speaker_coverage=False)

        self.assertEqual(report["clips_in_both"], [("spk0", "a")])
        self.assertTrue(any("both train and validation" in v for v in report["violations"]))

    def test_shared_transcript_is_reported(self):
        train = [row("a", "spk0", "Hello   World")]
        val = [row("b", "spk0", "hello world")]

        report = check_manifests(train, val)

        self.assertEqual(report["shared_transcripts"], ["hello world"])
        self.assertTrue(any("transcripts present" in v for v in report["violations"]))

    def test_missing_validation_speaker_is_reported(self):
        train = [row("a", "spk0", "train one")]
        val = [row("b", "spk1", "val one")]

        report = check_manifests(train, val, require_speaker_coverage=True)
        self.assertEqual(report["val_speakers_missing_from_train"], ["spk1"])
        self.assertTrue(any("no training clip" in v for v in report["violations"]))

        relaxed = check_manifests(train, val, require_speaker_coverage=False)
        self.assertEqual(relaxed["violations"], [])

    def test_duplicate_clips_inside_split_are_reported(self):
        train = [row("a", "spk0", "one"), row("a", "spk0", "two")]
        val = [row("b", "spk0", "val one")]

        report = check_manifests(train, val)

        self.assertEqual(report["duplicate_train_clips"], [("spk0", "a")])
        self.assertTrue(any("duplicate clips inside train" in v for v in report["violations"]))

    def test_duration_expectation_is_checked(self):
        train = [row("a", "spk0", "one", seconds=3600.0)]
        val = [row("b", "spk0", "val one", seconds=1800.0)]

        report = check_manifests(
            train,
            val,
            expected_train_hours=2.0,
            expected_val_hours=0.5,
            tolerance=0.1,
        )

        self.assertAlmostEqual(report["train_hours"], 1.0)
        self.assertAlmostEqual(report["val_hours"], 0.5)
        self.assertEqual(len(report["violations"]), 1)
        self.assertIn("train duration", report["violations"][0])

    def test_empty_split_is_reported(self):
        report = check_manifests([], [], require_speaker_coverage=False)
        self.assertTrue(any("training split is empty" in v for v in report["violations"]))
        self.assertTrue(any("validation split is empty" in v for v in report["violations"]))


if __name__ == "__main__":
    unittest.main()
