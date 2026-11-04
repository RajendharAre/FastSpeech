"""Tests for the G2P name-dictionary builder (no g2p_en required)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

from build_g2p_names import (  # noqa: E402
    build_entries,
    extract_candidates,
    read_lexicon_words,
    read_manifest_transcripts,
    read_names,
    write_dictionary,
)


def fake_g2p(name):
    return {
        "Svarah": ["S", "V", "AH0", "R", "AH0"],
        "Rao": ["R", "AW1"],
    }.get(name, [])


class CandidateTests(unittest.TestCase):
    def test_excludes_common_words_and_lexicon_hits(self):
        transcripts = [
            "The Hyderabad team met Svarah.",
            "the team arrived.",
        ]

        candidates = extract_candidates(
            transcripts, lexicon_words={"hyderabad"}
        )

        self.assertEqual(candidates, [("Svarah", 1)])

    def test_min_count_filters(self):
        transcripts = ["Svarah spoke.", "Svarah left."]

        self.assertEqual(
            extract_candidates(transcripts, set(), min_count=2),
            [("Svarah", 2)],
        )
        self.assertEqual(
            extract_candidates(transcripts, set(), min_count=3), []
        )


class EntryTests(unittest.TestCase):
    def test_written_existing_and_missing(self):
        entries, report = build_entries(
            [("Svarah", 2), ("Hyderabad", 1), ("Xyz", 1)],
            lexicon_words={"hyderabad"},
            g2p=fake_g2p,
        )

        self.assertEqual(
            entries, [("Svarah", ["S", "V", "AH0", "R", "AH0"])]
        )
        statuses = {row["name"]: row["status"] for row in report}
        self.assertEqual(statuses["Svarah"], "written")
        self.assertEqual(statuses["Hyderabad"], "already_in_lexicon")
        self.assertEqual(statuses["Xyz"], "no_phones")

    def test_duplicate_name_is_counted_once(self):
        entries, report = build_entries(
            [("Svarah", 1), ("svarah", 1)], set(), g2p=fake_g2p
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(len(report), 1)

    def test_dictionary_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g2p_names.txt"
            write_dictionary(
                path, [("Rao", ["R", "AW1"]), ("Svarah", ["S", "V"])]
            )
            text = path.read_text(encoding="utf-8")

        self.assertEqual(text, "RAO\tR AW1\nSVARAH\tS V\n")


class InputTests(unittest.TestCase):
    def test_read_names_ignores_comments_and_blanks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "names.txt"
            path.write_text("# names\nSvarah\n\nRao\n", encoding="utf-8")
            self.assertEqual(read_names(path), ["Svarah", "Rao"])

    def test_read_manifest_transcripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            path.write_text(
                "clip_id,speaker,duration,transcript\n"
                "1,spk0,1.0,Meet Svarah.\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_manifest_transcripts([path]), ["Meet Svarah."]
            )

    def test_read_lexicon_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lexicon.txt"
            path.write_text(
                "HELLO\tHH AH0 L OW1\n;;; comment\nworld\tW ER1 L D\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_lexicon_words(path), {"hello", "world"}
            )


if __name__ == "__main__":
    unittest.main()
