"""Unit tests for preprocessor/report.py (no audio stack required).

Run from English/ or Telugu/:
    python -m unittest tests.test_preprocessor_report
"""
import json
import os
import sys
import tempfile
import unittest

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

from preprocessor.report import (  # noqa: E402
    LOG_COLUMNS,
    ClipSkipped,
    format_summary,
    summarize,
    write_summary,
)


def row(split, status, category="", reason=""):
    return {
        "speaker": "spk",
        "clip": "clip",
        "split": split,
        "status": status,
        "category": category,
        "reason": reason,
    }


class ClipSkippedTests(unittest.TestCase):
    def test_carries_category_and_detail(self):
        exc = ClipSkipped("missing TextGrid", "missing TextGrid file: x")
        self.assertEqual(exc.category, "missing TextGrid")
        self.assertEqual(exc.detail, "missing TextGrid file: x")
        self.assertIn("missing TextGrid", str(exc))

    def test_log_has_a_category_column(self):
        self.assertEqual(
            LOG_COLUMNS, ("speaker", "clip", "split", "status", "category", "reason")
        )


class SummaryTests(unittest.TestCase):
    def test_counts_overall_split_and_category(self):
        rows = [
            row("train", "ok"),
            row("train", "failed", "missing TextGrid"),
            row("train", "failed", "missing TextGrid"),
            row("val", "ok"),
            row("val", "failed", "audio read"),
        ]
        summary = summarize(rows)
        self.assertEqual(summary["total"], 5)
        self.assertEqual(summary["ok"], 2)
        self.assertEqual(summary["dropped"], 3)
        self.assertEqual(summary["by_split"]["train"]["dropped"], 2)
        self.assertEqual(summary["by_split"]["val"]["dropped"], 1)
        self.assertEqual(
            summary["by_category"],
            {"audio read": 1, "missing TextGrid": 2},
        )

    def test_format_names_the_categories(self):
        summary = summarize([row("train", "failed", "missing TextGrid")])
        text = format_summary(summary)
        self.assertIn("0 ok, 1 dropped", text)
        self.assertIn("missing TextGrid: 1", text)

    def test_no_drops_hides_category_block(self):
        text = format_summary(summarize([row("train", "ok")]))
        self.assertNotIn("Dropped by category", text)

    def test_write_summary_is_json(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "preprocessing_summary.json")
            write_summary(path, summarize([row("train", "ok")]))
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertEqual(payload["ok"], 1)


if __name__ == "__main__":
    unittest.main()
