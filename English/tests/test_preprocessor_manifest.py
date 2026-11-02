"""Unit tests for preprocessor/manifest.py (no audio stack required).

Run from English/ or Telugu/:
    python -m unittest tests.test_preprocessor_manifest
"""
import csv
import os
import sys
import tempfile
import unittest

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

from preprocessor.manifest import ManifestError, load_manifests  # noqa: E402

HEADER = ("clip_id", "speaker", "duration", "transcript")


def write_rows(path, rows, header=HEADER):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class ManifestLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.train = os.path.join(self.tempdir.name, "train.csv")
        self.val = os.path.join(self.tempdir.name, "val.csv")
        write_rows(self.train, [("a", "spk0", "1.5", "hello")])
        write_rows(self.val, [("b", "spk0", "2.5", "world")])

    def tearDown(self):
        self.tempdir.cleanup()

    def test_valid_manifests_are_loaded_in_order(self):
        entries = load_manifests({"train": self.train, "val": self.val})
        self.assertEqual(entries, [("train", "spk0", "a"), ("val", "spk0", "b")])

    def test_missing_manifest_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_manifests(
                {"train": self.train, "val": os.path.join(self.tempdir.name, "nope.csv")}
            )

    def test_missing_column_raises(self):
        write_rows(self.train, [("a", "spk0", "1.5")], header=("clip_id", "speaker", "duration"))
        with self.assertRaises(ManifestError):
            load_manifests({"train": self.train, "val": self.val})

    def test_empty_manifest_raises(self):
        write_rows(self.train, [])
        with self.assertRaises(ManifestError):
            load_manifests({"train": self.train, "val": self.val})

    def test_duplicate_inside_manifest_raises(self):
        write_rows(
            self.train,
            [("a", "spk0", "1.5", "hello"), ("a", "spk0", "2.5", "again")],
        )
        with self.assertRaises(ManifestError):
            load_manifests({"train": self.train, "val": self.val})

    def test_clip_in_both_manifests_raises(self):
        write_rows(self.val, [("a", "spk0", "2.5", "world")])
        with self.assertRaises(ManifestError):
            load_manifests({"train": self.train, "val": self.val})

    def test_path_traversal_is_rejected(self):
        write_rows(self.train, [("../a", "spk0", "1.5", "hello")])
        with self.assertRaises(ManifestError):
            load_manifests({"train": self.train, "val": self.val})

        write_rows(self.train, [("a", "../spk0", "1.5", "hello")])
        with self.assertRaises(ManifestError):
            load_manifests({"train": self.train, "val": self.val})

    def test_invalid_duration_raises(self):
        for duration in ("0", "-1", "nan", "inf", "abc"):
            write_rows(self.train, [("a", "spk0", duration, "hello")])
            with self.assertRaises(ManifestError):
                load_manifests({"train": self.train, "val": self.val})

    def test_empty_transcript_raises(self):
        write_rows(self.train, [("a", "spk0", "1.5", "   ")])
        with self.assertRaises(ManifestError):
            load_manifests({"train": self.train, "val": self.val})


if __name__ == "__main__":
    unittest.main()
