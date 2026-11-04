"""Tests for loading the main and supplementary lexicons."""
import os
import sys
import tempfile
import unittest

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

from inference import read_lexicon  # noqa: E402


class LexiconMergeTests(unittest.TestCase):
    def write(self, content):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8", newline=""
        )
        handle.write(content)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_main_lexicon_wins_and_supplement_fills_gaps(self):
        main = self.write("SVARAH\tS V AA1 R AH0\n")
        supplement = self.write("SVARAH\tS V AA2 R AH0\nRAO\tR AW1\n")

        lexicon = read_lexicon(main, supplement)

        self.assertEqual(lexicon["svarah"], ["S", "V", "AA1", "R", "AH0"])
        self.assertEqual(lexicon["rao"], ["R", "AW1"])

    def test_blank_and_comment_lines_are_skipped(self):
        main = self.write("\n;;; comment\nHELLO\tHH AH0 L OW1\n")
        lexicon = read_lexicon(main)
        self.assertNotIn("", lexicon)
        self.assertEqual(lexicon["hello"], ["HH", "AH0", "L", "OW1"])

    def test_missing_supplement_raises(self):
        main = self.write("HELLO\tHH AH0 L OW1\n")
        with self.assertRaises(FileNotFoundError):
            read_lexicon(main, "does-not-exist.txt")


if __name__ == "__main__":
    unittest.main()
