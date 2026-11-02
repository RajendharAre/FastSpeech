"""Unit tests for the Telugu text frontend. No audio, model or data needed.

Run from Telugu/:
    python -m unittest tests.test_telugu_frontend
"""
import os
import sys
import unittest

LANG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LANG_DIR)

from text import sequence_to_text, text_to_sequence  # noqa: E402
from text import telugu  # noqa: E402
from text.symbols import symbols  # noqa: E402


class NormalizationTests(unittest.TestCase):
    def test_nfc_composes_e_vowel_signs(self):
        self.assertEqual(telugu.normalize_text("\u0C46\u0C56"), "\u0C48")

    def test_whitespace_and_controls_are_normalized(self):
        raw = "  \u0C05\u00A0\u0C15\u200D\n\u200B"
        self.assertEqual(telugu.normalize_text(raw), "\u0C05 \u0C15")

    def test_typographic_punctuation_is_folded(self):
        raw = "\u201C\u0C05\u201D \u2013 \u0C15\u2026"
        self.assertEqual(telugu.normalize_text(raw), '"\u0C05" - \u0C15...')

    def test_danda_becomes_full_stop(self):
        self.assertEqual(telugu.normalize_text("\u0C05\u0964"), "\u0C05.")

    def test_normalization_is_idempotent(self):
        once = telugu.normalize_text(" \u0C05\u0C46\u0C56  \u0C15 ")
        self.assertEqual(telugu.normalize_text(once), once)

    def test_non_string_is_rejected(self):
        with self.assertRaises(TypeError):
            telugu.normalize_text(123)


class SymbolSetTests(unittest.TestCase):
    def test_symbols_are_unique(self):
        self.assertEqual(len(symbols), len(set(symbols)))

    def test_required_symbols_are_present(self):
        for symbol in ("_", "-", " ", "@sp", "@spn", "@sil", "\u0C05", "\u0C4D"):
            self.assertIn(symbol, symbols)

    def test_english_letters_are_not_symbols(self):
        for letter in "AZaz":
            self.assertNotIn(letter, symbols)


class EncodingTests(unittest.TestCase):
    def test_encode_decode_roundtrip(self):
        text = telugu.normalize_text("\u0C24\u0C46\u0C32\u0C41\u0C17\u0C41 \u0C15")
        sequence = telugu.encode_text(text)
        self.assertEqual(len(sequence), len(text))
        self.assertEqual(telugu.decode_text(sequence), text)

    def test_pipeline_roundtrip(self):
        text = "\u0C05\u0C2E\u0C4D\u0C2E \u0C05\u0C2A\u0C4D\u0C2A"
        sequence = text_to_sequence(text, ["telugu_cleaners"])
        self.assertEqual(sequence_to_text(sequence), text)

    def test_unknown_character_raises_with_code_point(self):
        with self.assertRaises(telugu.UnknownCharacterError) as caught:
            telugu.encode_text("\u0C05A")
        self.assertIn("U+0041", str(caught.exception))

    def test_find_unknown_characters_does_not_raise(self):
        self.assertEqual(telugu.find_unknown_characters("\u0C05AB"), ["A", "B"])

    def test_unassigned_and_private_use_characters_are_rejected(self):
        for character in ("\u0C0D", "\uE000"):
            with self.assertRaises(telugu.UnknownCharacterError):
                telugu.encode_text(character)

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            telugu.encode_text("   \u200D ")

    def test_unknown_symbol_id_raises(self):
        with self.assertRaises(telugu.UnknownSymbolError):
            telugu.decode_text([10 ** 6])


class AlignmentTokenTests(unittest.TestCase):
    def test_alignment_tokens_are_one_symbol_per_phone(self):
        sequence = text_to_sequence("{\u0C05 \u0C15}", ["telugu_cleaners"])
        self.assertEqual(sequence, telugu.encode_text("\u0C05\u0C15"))

    def test_silence_tokens_map_to_silence_symbols(self):
        sequence = text_to_sequence(
            "{\u0C05 sp \u0C15 sil spn}", ["telugu_cleaners"]
        )
        expected = (
            telugu.encode_text("\u0C05")
            + [symbols.index("@sp")]
            + telugu.encode_text("\u0C15")
            + [symbols.index("@sil"), symbols.index("@spn")]
        )
        self.assertEqual(sequence, expected)

    def test_silence_decode_uses_token_names(self):
        self.assertEqual(telugu.decode_text([symbols.index("@sp")]), "sp")

    def test_multi_character_token_raises(self):
        with self.assertRaises(telugu.AlignmentTokenError):
            text_to_sequence("{\u0C05 ka}", ["telugu_cleaners"])

    def test_empty_alignment_raises(self):
        with self.assertRaises(telugu.AlignmentTokenError):
            text_to_sequence("{}", ["telugu_cleaners"])

    def test_unbalanced_braces_raise(self):
        with self.assertRaises(telugu.AlignmentTokenError):
            text_to_sequence("{\u0C05", ["telugu_cleaners"])


if __name__ == "__main__":
    unittest.main()
