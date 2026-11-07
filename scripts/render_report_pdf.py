#!/usr/bin/env python3
"""Render Report/Technical_Report.md to PDF with a headless Chromium-family browser.

Read-only for the Markdown. Writes only the PDF (and optionally the intermediate HTML).

    python scripts/render_report_pdf.py --telugu-font <path-to-telugu.ttf> --check

`--check` re-reads the produced PDF. Latin sentences must appear verbatim; Telugu
is checked by codepoint coverage instead, because a PDF content stream stores glyph
runs in visual order and extraction reorders the vowel signs. A machine with no
Telugu font draws blanks and fails the coverage test, which is the hazard this
check exists to catch.
"""

import argparse
import base64
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BROWSER_CANDIDATES = [
    os.environ.get("REPORT_BROWSER"),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]

CSS = """
@page { size: A4; margin: 14mm 15mm; }
body { font: 10pt/1.35 "Noto Serif", Georgia, serif; color: #111; margin: 0;
       font-family: "Noto Serif", Georgia, %(telugu_stack)s serif; }
h1 { font-size: 16pt; margin: 0 0 0.8em; }
h2 { font-size: 12.5pt; margin: 1.1em 0 0.35em; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 0.9em 0 0.3em; page-break-after: avoid; }
p, li { orphans: 2; widows: 2; }
p { margin: 0 0 0.55em; }
code { font-family: "DejaVu Sans Mono", "Courier New", monospace; font-size: 9pt;
       background: #f3f3f3; padding: 0 2px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 8px 10px; overflow-x: hidden;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%%; margin: 0.45em 0; font-size: 9.2pt; }
table.keep-together { page-break-inside: avoid; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3px 5px; text-align: left; vertical-align: top; }
th { background: #eee; }
blockquote { margin: 0.45em 0; padding: 4px 8px; border-left: 3px solid #888;
             background: #f7f7f2; color: #333; }
blockquote p { margin: 0; }
hr { border: none; border-top: 1px solid #ccc; margin: 0.8em 0; }
img { max-width: 100%%; height: auto; }
"""


def find_browser():
    for candidate in BROWSER_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def telugu_face(font_path):
    data = base64.b64encode(Path(font_path).read_bytes()).decode("ascii")
    face = (
        "@font-face { font-family: 'ReportTelugu'; "
        "src: url(data:font/ttf;base64,%s) format('truetype'); }" % data
    )
    return face, "'ReportTelugu',"


def keep_short_tables(body):
    """Stop a small table breaking across pages and leaving orphan rows behind."""
    def tag(match):
        block = match.group(0)
        if block.count("<tr>") <= 12:
            return block.replace("<table>", '<table class="keep-together">', 1)
        return block

    return re.sub(r"<table>.*?</table>", tag, body, flags=re.S)


def to_html(markdown_text, font_path=None, markdown_path=None):
    from markdown_it import MarkdownIt

    body = keep_short_tables(MarkdownIt("commonmark").enable("table").render(markdown_text))
    if markdown_path is not None:
        markdown_dir = Path(markdown_path).resolve().parent

        def embed_image(match):
            source = match.group(2)
            if source.startswith(("data:", "http:", "https:", "file:")):
                return match.group(0)
            image_path = (markdown_dir / source).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(
                    "report image not found: {} (referenced by {})".format(
                        image_path, markdown_path
                    )
                )
            media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            return '{}data:{};base64,{}{}'.format(
                match.group(1), media_type, encoded, match.group(3)
            )

        body = re.sub(
            r'(<img\b[^>]*\bsrc=["\'])([^"\']+)(["\'])',
            embed_image,
            body,
        )
    face, stack = telugu_face(font_path) if font_path else ("", "")
    css = CSS % {"telugu_stack": stack}
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Technical Report</title><style>{face}{css}</style></head>"
        f"<body>{body}</body></html>"
    )


def print_pdf(browser, html_path, pdf_path, work_dir):
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--user-data-dir={work_dir}",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if not Path(pdf_path).is_file():
        if "--headless=new" in cmd[1]:
            cmd[1] = "--headless"
            done = subprocess.run(cmd, capture_output=True, text=True)
    if not Path(pdf_path).is_file():
        sys.exit(f"browser produced no PDF.\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}")
    return done


def telugu_chars(text):
    return sum(1 for ch in text if 0x0C00 <= ord(ch) <= 0x0C7F)


def check_pdf(pdf_path, md_text, sentences):
    try:
        import pymupdf
    except ImportError:
        print("PyMuPDF is not installed; cannot verify the PDF. Install pymupdf to use --check.")
        return False
    text = "".join(page.get_text() for page in pymupdf.open(pdf_path))
    flat = re.sub(r"\s+", " ", text)
    problems = []

    # Latin text extracts in logical order, so an exact match is a fair test.
    for label, sentence in sentences:
        if telugu_chars(sentence) == 0:
            probe = re.sub(r"\s+", " ", sentence).strip()
            if probe and probe not in flat:
                problems.append(f"{label} is absent from the PDF")

    # Telugu does not: a PDF content stream stores the glyph runs in visual order, so
    # extracting it reorders or repeats the vowel signs and anusvara. Comparing whole
    # strings would fail even when the page renders perfectly, so compare coverage.
    wanted, drawn = telugu_chars(md_text), telugu_chars(text)
    if wanted and drawn / wanted < 0.9:
        problems.append(
            f"only {drawn} of {wanted} Telugu codepoints reached the PDF — "
            "install a Telugu font or pass --telugu-font"
        )
    if "\ufffd" in text:
        problems.append("the PDF contains U+FFFD replacement characters")

    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print("  -", problem)
        return False
    print(f"PROBLEMS: none — {wanted} Telugu codepoints drawn, all Latin sentences present")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", default=str(ROOT / "Report" / "Technical_Report.md"))
    ap.add_argument("--pdf", default=str(ROOT / "Report" / "Technical_Report.pdf"))
    ap.add_argument("--telugu-font", help="TTF to embed for Telugu glyphs")
    ap.add_argument("--keep-html", action="store_true")
    ap.add_argument("--check", action="store_true", help="verify every test sentence survived into the PDF")
    args = ap.parse_args()

    browser = find_browser()
    if not browser:
        sys.exit("No Chrome or Edge found. Set REPORT_BROWSER to the executable path.")

    md_path = Path(args.md)
    html = to_html(
        md_path.read_text(encoding="utf-8"),
        args.telugu_font,
        markdown_path=md_path,
    )

    work_dir = Path(tempfile.mkdtemp(prefix="report_pdf_"))
    html_path = work_dir / "Technical_Report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"browser: {browser}\nhtml: {html_path} ({len(html):,} bytes)")

    print_pdf(browser, html_path, args.pdf, work_dir)
    print(f"pdf: {args.pdf} ({Path(args.pdf).stat().st_size:,} bytes)")

    if args.keep_html:
        target = md_path.with_suffix(".html")
        shutil.copyfile(html_path, target)
        print(f"kept: {target}")

    ok = True
    if args.check:
        import json

        data = json.loads((ROOT / "docs" / "test_sentences.json").read_text(encoding="utf-8"))
        pairs = [(it["id"], it["final"]) for lang in ("English", "Telugu") for it in data[lang]]
        ok = check_pdf(args.pdf, md_path.read_text(encoding="utf-8"), pairs)
    shutil.rmtree(work_dir, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
