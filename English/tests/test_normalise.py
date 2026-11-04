"""Rule-based English normalisation tests.

Run from English/:
    python -m unittest tests.test_normalise
"""
import os
import sys
import tempfile
import unittest

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

from text.normalise import (  # noqa: E402
    format_rules,
    load_acronyms,
    normalise_english,
)


class NormaliseTests(unittest.TestCase):
    def test_time_is_spoken(self):
        text, rules = normalise_english("Start at 10:45 AM.")
        self.assertEqual(text, "Start at ten forty-five a m.")
        self.assertIn("time", rules)

    def test_time_with_minutes_below_ten(self):
        text, rules = normalise_english("Start at 09:05.")
        self.assertEqual(text, "Start at nine oh five.")
        self.assertIn("time", rules)

    def test_iso_date(self):
        text, rules = normalise_english("On 2026-11-12.")
        self.assertEqual(
            text, "On the twelfth of november twenty twenty-six."
        )
        self.assertIn("date", rules)

    def test_currency_and_percent(self):
        text, rules = normalise_english("It costs $5.50, or 45% more.")
        self.assertEqual(
            text,
            "It costs five point five zero dollars, or forty-five percent more.",
        )
        self.assertIn("currency", rules)
        self.assertIn("percent", rules)

    def test_acronym_from_csv_changes_case_only_token(self):
        text, rules = normalise_english("ASR and NASA launched it.")
        self.assertEqual(text, "A S R and NASA launched it.")
        self.assertEqual(rules, ("acronym",))

    def test_word_mode_acronym_leaves_text_unchanged(self):
        text, rules = normalise_english("ISRO launched it.")
        self.assertEqual(text, "ISRO launched it.")
        self.assertEqual(rules, ())

    def test_dotted_acronym(self):
        text, rules = normalise_english("The U.S.A. team.")
        self.assertEqual(text, "The U S A team.")
        self.assertIn("acronym", rules)

    def test_abbreviation(self):
        text, rules = normalise_english("Dr. Rao arrived.")
        self.assertEqual(text, "doctor Rao arrived.")
        self.assertIn("abbreviation", rules)

    def test_decimal_and_ordinal(self):
        text, rules = normalise_english("Pi is 3.14 on the 12th day.")
        self.assertEqual(
            text, "Pi is three point one four on the twelfth day."
        )
        self.assertIn("decimal", rules)
        self.assertIn("number", rules)

    def test_plain_text_has_no_rules(self):
        text, rules = normalise_english("hello world")
        self.assertEqual(text, "hello world")
        self.assertEqual(rules, ())

    def test_format_rules(self):
        self.assertEqual(format_rules(("time", "number")), "time;number")
        self.assertEqual(format_rules(()), "")


class AcronymFileTests(unittest.TestCase):
    def write(self, content):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        )
        handle.write(content)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_loads_mapping(self):
        path = self.write("acronym,spoken,mode\nAI,A I,letters\nNASA,NASA,word\n")
        self.assertEqual(
            load_acronyms(path), {"AI": "A I", "NASA": "NASA"}
        )

    def test_invalid_mode_raises(self):
        path = self.write("acronym,spoken,mode\nAI,A I,shout\n")
        with self.assertRaises(ValueError):
            load_acronyms(path)

    def test_missing_column_raises(self):
        path = self.write("acronym,spoken\nAI,A I\n")
        with self.assertRaises(ValueError):
            load_acronyms(path)


if __name__ == "__main__":
    unittest.main()
