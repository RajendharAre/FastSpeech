"""Unit tests for scripts/render_report_pdf.py.

Run from the repository root:
    python -m unittest discover -s scripts/tests -v
"""
import os
import sys
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

from render_report_pdf import find_browser, telugu_chars, to_html  # noqa: E402


class TeluguCountTest(unittest.TestCase):
    def test_counts_only_telugu_codepoints(self):
        # The virama and vowel signs are Telugu-block codepoints, so a five-glyph
        # cluster counts as five, not as the two consonants a reader sees.
        self.assertEqual(telugu_chars("telugu పట్టు 12"), 5)

    def test_latin_only_counts_zero(self):
        self.assertEqual(telugu_chars("The rhythm of the orchestra."), 0)


class HtmlTest(unittest.TestCase):
    def setUp(self):
        try:
            import markdown_it  # noqa: F401
        except ImportError:
            self.skipTest("markdown-it-py is not installed")

    def test_table_is_rendered_as_html(self):
        html = to_html("| ID | Sentence |\n|---|---|\n| E3 | The rhythm. |\n")
        self.assertIn("<table>", html)
        self.assertIn("E3", html)

    def test_utf8_meta_and_telugu_survive(self):
        html = to_html("పులి అడవిలో నడిచింది.\n")
        self.assertIn("charset='utf-8'", html)
        self.assertIn("పులి", html)

    def test_css_placeholders_are_filled(self):
        html = to_html("text\n")
        self.assertNotIn("%(", html)
        self.assertIn("100%", html)

    def test_telugu_font_is_embedded_as_data_uri(self):
        # Only the base64 embedding is under test, so the bytes need not be a real font.
        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as handle:
            handle.write(b"\x00\x01\x00\x00fake font bytes")
            font = handle.name
        try:
            html = to_html("పులి\n", font)
        finally:
            os.remove(font)
        self.assertIn("ReportTelugu", html)
        self.assertIn("data:font/ttf;base64,", html)


class BrowserTest(unittest.TestCase):
    def test_found_browser_exists(self):
        browser = find_browser()
        if browser is not None:
            self.assertTrue(os.path.isfile(browser))


if __name__ == "__main__":
    unittest.main()
