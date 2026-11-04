"""Unit tests for kaggle/inspect_utils.py using synthetic data only.

Run from the repository root:
    python kaggle/tests/test_inspect_utils.py
"""
import io
import sys
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_utils import (  # noqa: E402
    duplicate_across_groups,
    duration_stats,
    group_stats,
    md_escape,
    probe_audio_bytes,
    probe_audio_value,
    render_markdown,
    text_stats,
)


def make_wav(frames=24000, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


class ProbeTests(unittest.TestCase):
    def test_wav_header(self):
        result = probe_audio_bytes(make_wav())
        self.assertEqual(result["frames"], 24000)
        self.assertEqual(result["samplerate"], 16000)

    def test_bytes_and_decoded_value(self):
        value = {"bytes": make_wav(frames=8000, rate=8000), "path": None}
        self.assertEqual(probe_audio_value(value)["samplerate"], 8000)

        decoded = {"array": np.zeros(4000), "sampling_rate": 4000}
        result = probe_audio_value(decoded)
        self.assertEqual(result["frames"], 4000)
        self.assertEqual(result["samplerate"], 4000)

    def test_garbage_reports_error(self):
        self.assertIn("error", probe_audio_bytes(b"not audio"))


class StatsTests(unittest.TestCase):
    def test_duration_buckets(self):
        stats = duration_stats([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(stats["total_seconds"], 10.0)
        self.assertAlmostEqual(stats["median"], 2.5)
        counts = {b["label"]: b["count"] for b in stats["buckets"]}
        self.assertEqual(counts["1-3 s"], 2)
        self.assertEqual(counts["3-10 s"], 2)
        self.assertEqual(counts["< 1 s"], 0)

    def test_duration_stats_p50_p90_p99(self):
        stats = duration_stats([float(i) for i in range(1, 101)])
        self.assertAlmostEqual(stats["p50"], 50.5)
        self.assertAlmostEqual(stats["p90"], 90.1)
        self.assertAlmostEqual(stats["p99"], 99.01)
        self.assertAlmostEqual(stats["median"], stats["p50"])

    def test_text_stats_duplicates(self):
        stats = text_stats(["a", "a", "b"])
        self.assertEqual(stats["distinct_values"], 2)
        self.assertEqual(stats["duplicate_values"], 1)
        self.assertEqual(stats["duplicate_rows"], 2)
        self.assertEqual(stats["distinct_chars"], 2)

    def test_group_stats_counts_missing_durations(self):
        table = group_stats(["s1", "s1", "s2"], [10.0, None, 20.0])
        by_value = {row["value"]: row for row in table}
        self.assertEqual(by_value["s1"]["clips"], 2)
        self.assertEqual(by_value["s1"]["missing"], 1)
        self.assertAlmostEqual(by_value["s1"]["hours"], 10.0 / 3600.0)
        self.assertAlmostEqual(by_value["s2"]["hours"], 20.0 / 3600.0)

    def test_duplicate_across_groups(self):
        self.assertEqual(
            duplicate_across_groups(["x", "x", "y"], ["s1", "s2", "s1"]), 1
        )
        self.assertEqual(
            duplicate_across_groups(["x", "y"], ["s1", "s1"]), 0
        )

    def test_md_escape(self):
        self.assertEqual(md_escape("a|b"), "a\\|b")


class RenderTests(unittest.TestCase):
    def fixture(self):
        durations = [600.0, 600.0, 600.0]
        return {
            "meta": {
                "repo_id": "example/data",
                "language": "English",
                "sha": "abc123",
                "last_modified": "2026-01-01",
                "python_version": "3.14.7",
                "datasets_version": "x",
                "hub_version": "y",
                "gated": False,
                "license": "cc-by-4.0",
                "tags": ["license:cc-by-4.0"],
            },
            "generated_at": "2026-11-02 10:00:00",
            "command": "python kaggle/inspect_datasets.py --dataset svarah",
            "splits": [
                {
                    "config": "default",
                    "split": "train",
                    "num_rows": 3,
                    "features": "{'audio': Audio(), 'text': Value('string')}",
                }
            ],
            "audio": {
                "columns": [
                    {
                        "name": "audio",
                        "measured": 3,
                        "failed": 0,
                        "sampling_rates": {"16000": 3},
                        "total_hours": 0.5,
                        "failure_examples": [],
                    }
                ],
                "duration_stats": duration_stats(durations),
                "primary_column": "audio",
            },
            "groups": [
                {
                    "column": "speaker",
                    "distinct": 2,
                    "top": group_stats(["s1", "s2", "s1"], durations),
                    "csv_path": None,
                    "clips_without_duration": 0,
                }
            ],
            "text": [
                {
                    "column": "text",
                    **text_stats(["a", "a", "b"]),
                    "group_column": "speaker",
                    "duplicates_across_groups": 1,
                }
            ],
            "summary": {
                "rows": 3,
                "group_column": "speaker",
                "total_hours": 0.5,
                "clips_measured": 3,
            },
            "label_columns": {},
            "warnings": [],
            "json_path": "docs/dataset_report_svarah.json",
        }

    def test_render_contains_measured_numbers(self):
        text = render_markdown([self.fixture()], "2026-11-02 10:00:00")
        self.assertIn("# Dataset report", text)
        self.assertIn("example/data", text)
        self.assertIn("16000 (3)", text)
        self.assertIn("0.5000", text)
        self.assertIn("Cross-group duplicate check uses column `speaker`", text)
        self.assertIn("Measured total: **0.5000 h** across 3 clips", text)

    def test_render_handles_missing_audio(self):
        report = self.fixture()
        report["audio"] = {"columns": [], "duration_stats": None}
        report["groups"] = []
        text = render_markdown([report], "2026-11-02 10:00:00")
        self.assertIn("No `Audio` feature column was found", text)
        self.assertIn("not measured", text)

    def test_render_duration_percentiles_p50_p90_p99(self):
        text = render_markdown([self.fixture()], "2026-11-02 10:00:00")
        self.assertIn(
            "| n | min | p5 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean | std |",
            text,
        )
        self.assertIn(
            "| 3 | 600.000 | 600.000 | 600.000 | 600.000 | 600.000 | 600.000"
            " | 600.000 | 600.000 | 600.000 | 600.000 | 0.000 |",
            text,
        )

    def test_render_sampling_rate_histogram(self):
        text = render_markdown([self.fixture()], "2026-11-02 10:00:00")
        self.assertIn("| Sampling rate (Hz) | Clips | Share |", text)
        self.assertIn("| 16000 | 3 | 100.0% |", text)

        report = self.fixture()
        report["audio"]["columns"][0]["sampling_rates"] = {"22050": 1, "16000": 2}
        text = render_markdown([report], "2026-11-02 10:00:00")
        self.assertIn("| 16000 | 2 | 66.7% |", text)
        self.assertIn("| 22050 | 1 | 33.3% |", text)

    def test_render_group_table_includes_hours(self):
        text = render_markdown([self.fixture()], "2026-11-02 10:00:00")
        self.assertIn("| Value | Clips | Hours |", text)
        self.assertIn("| s1 | 2 | 0.3333 |", text)
        self.assertIn("| s2 | 1 | 0.1667 |", text)

    def test_render_old_report_without_new_percentiles(self):
        report = self.fixture()
        for key in ("p50", "p90", "p99"):
            report["audio"]["duration_stats"].pop(key, None)
        text = render_markdown([report], "2026-11-02 10:00:00")
        self.assertIn("not measured", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
