"""Unit tests for scripts/split_dataset.py. No audio or datasets needed.

Run from the repository root:
    python -m unittest discover -s scripts/tests -v
"""
import os
import sys
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

from split_dataset import (  # noqa: E402
    ManifestError,
    ManifestRow,
    SplitError,
    load_manifest,
    normalize_transcript,
    split_rows,
    write_manifest,
)


def make_rows(n_speakers=3, n_clips=10, seconds=60.0):
    rows = []
    line = 2
    for speaker_index in range(n_speakers):
        speaker = "spk{}".format(speaker_index)
        for clip_index in range(n_clips):
            rows.append(
                ManifestRow(
                    clip_id="{:04d}".format(clip_index),
                    speaker=speaker,
                    duration=seconds,
                    duration_text=str(seconds),
                    transcript="sentence {} {}".format(speaker, clip_index),
                    line_number=line,
                )
            )
            line += 1
    return rows


def write_csv(path, rows, header=("clip_id", "speaker", "duration", "transcript")):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class ManifestTests(unittest.TestCase):
    def test_load_and_write_roundtrip(self):
        rows = make_rows(n_speakers=2, n_clips=3)
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "first.csv")
            second = os.path.join(tmp, "second.csv")
            write_manifest(first, rows)
            loaded = load_manifest(first)
            write_manifest(second, loaded)
            with open(first, encoding="utf-8") as handle:
                first_text = handle.read()
            with open(second, encoding="utf-8") as handle:
                second_text = handle.read()
        self.assertEqual(first_text, second_text)
        self.assertEqual(len(loaded), 6)

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            write_csv(path, [("a", "spk", "1.0")], header=("clip_id", "speaker", "duration"))
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_invalid_durations_raise(self):
        for duration in ("0", "-1", "nan", "inf", "abc"):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "bad.csv")
                write_csv(path, [("a", "spk", duration, "hello")])
                with self.assertRaises(ManifestError):
                    load_manifest(path)

    def test_empty_transcript_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            write_csv(path, [("a", "spk", "1.0", "   ")])
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_path_like_ids_raise(self):
        for clip_id, speaker in (("../a", "spk"), ("a", "spk/sub"), ("a", "..")):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "bad.csv")
                write_csv(path, [(clip_id, speaker, "1.0", "hello")])
                with self.assertRaises(ManifestError):
                    load_manifest(path)

    def test_duplicate_clip_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            write_csv(
                path,
                [
                    ("a", "spk", "1.0", "one"),
                    ("a", "spk", "2.0", "two"),
                ],
            )
            with self.assertRaises(ManifestError):
                load_manifest(path)

    def test_normalize_transcript(self):
        self.assertEqual(
            normalize_transcript("  Hello\u00A0 \t World  "),
            normalize_transcript("hello world"),
        )


class SplitTests(unittest.TestCase):
    def test_deterministic_and_order_independent(self):
        rows = make_rows()
        first = split_rows(rows, 0.3, 0.05, seed=17, tolerance=0.1)
        second = split_rows(list(reversed(rows)), 0.3, 0.05, seed=17, tolerance=0.1)

        for index in (0, 1, 2):
            self.assertEqual(
                [row.key for row in first[index]],
                [row.key for row in second[index]],
            )
        self.assertEqual(first[3], second[3])

    def test_duration_targets_and_coverage(self):
        rows = make_rows()
        train_rows, val_rows, unused_rows, summary = split_rows(
            rows, 0.3, 0.05, seed=17, tolerance=0.1
        )

        self.assertAlmostEqual(summary["val_hours"], 0.05, places=6)
        self.assertGreaterEqual(summary["train_hours"], 0.3 * 0.9)
        self.assertLessEqual(summary["train_hours"], 0.3 * 1.1 + 1e-9)
        self.assertLessEqual(
            summary["val_hours"], 0.05 * 1.1 + 1e-9
        )
        self.assertEqual(
            len(train_rows) + len(val_rows) + len(unused_rows), len(rows)
        )

        val_speakers = {row.speaker for row in val_rows}
        train_speakers = {row.speaker for row in train_rows}
        self.assertTrue(val_speakers)
        self.assertFalse(val_speakers - train_speakers)

    def test_no_transcript_overlap(self):
        train_rows, val_rows, _unused, _summary = split_rows(
            make_rows(), 0.3, 0.05, seed=17, tolerance=0.1
        )
        train_transcripts = {normalize_transcript(row.transcript) for row in train_rows}
        val_transcripts = {normalize_transcript(row.transcript) for row in val_rows}
        self.assertFalse(train_transcripts & val_transcripts)

    def test_same_transcript_group_is_not_split(self):
        rows = make_rows()
        # Replace the first clip of two different speakers with the same
        # sentence, ignoring case and repeated whitespace.
        rows[0] = ManifestRow(
            "0000", "spk0", 60.0, "60.0", "Shared  Sentence", 2
        )
        rows[10] = ManifestRow(
            "0000", "spk1", 60.0, "60.0", "shared sentence", 12
        )

        train_rows, val_rows, _unused, _summary = split_rows(
            rows, 0.3, 0.05, seed=17, tolerance=0.1
        )
        shared = {row.key for row in rows if normalize_transcript(row.transcript) == "shared sentence"}
        train_keys = {row.key for row in train_rows}
        val_keys = {row.key for row in val_rows}
        unused_keys = {row.key for row in _unused}

        self.assertEqual(len(shared), 2)
        labels = set()
        for key in shared:
            if key in train_keys:
                labels.add("train")
            elif key in val_keys:
                labels.add("val")
            elif key in unused_keys:
                labels.add("unused")
            else:
                self.fail("shared clip {} is missing from every split".format(key))
        self.assertEqual(len(labels), 1, "transcript group was split")

    def test_single_group_speaker_stays_in_train(self):
        rows = [
            ManifestRow("a", "solo", 360.0, "360", "only solo clip", 2),
        ]
        for index in range(5):
            rows.append(
                ManifestRow(
                    "b{}".format(index),
                    "busy",
                    360.0,
                    "360",
                    "busy clip {}".format(index),
                    3 + index,
                )
            )

        train_rows, val_rows, _unused, _summary = split_rows(
            rows, 0.5, 0.1, seed=5, tolerance=0.0
        )
        self.assertIn(("solo", "a"), {row.key for row in train_rows})
        self.assertNotIn(("solo", "a"), {row.key for row in val_rows})
        self.assertTrue(val_rows)

    def test_targets_above_total_raise(self):
        with self.assertRaises(SplitError):
            split_rows(make_rows(n_speakers=1, n_clips=2), 10.0, 1.0)

    def test_validation_target_smaller_than_smallest_group_raises(self):
        with self.assertRaises(SplitError):
            split_rows(
                make_rows(n_speakers=2, n_clips=3, seconds=60.0),
                0.05,
                0.001,
                tolerance=0.0,
            )

    def test_non_positive_targets_raise(self):
        with self.assertRaises(SplitError):
            split_rows(make_rows(), 0.0, 0.1)

    def test_negative_tolerance_raises(self):
        with self.assertRaises(SplitError):
            split_rows(make_rows(), 0.1, 0.1, tolerance=-0.1)


if __name__ == "__main__":
    unittest.main()
