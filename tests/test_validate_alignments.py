import csv
from pathlib import Path

import pytest

from scripts.validate_alignments import (
    load_manifest,
    parse_textgrid,
    review_textgrid,
    validate,
)


def write_textgrid(path: Path) -> Path:
    path.write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.02
            text = "a"
        intervals [2]:
            xmin = 0.02
            xmax = 1
            text = "hello"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.03
            text = "AH"
        intervals [2]:
            xmin = 0.03
            xmax = 0.90
            text = "L"
""",
        encoding="utf-8",
    )
    return path


def test_parse_textgrid_reads_tier_names_and_intervals(tmp_path):
    path = write_textgrid(tmp_path / "clip_001.TextGrid")

    tiers = parse_textgrid(path)

    assert [tier.name for tier in tiers] == ["words", "phones"]
    assert tiers[0].intervals[0].text == "a"
    assert tiers[0].intervals[0].duration == pytest.approx(0.02)


def test_review_flags_short_word_and_phone_extremes(tmp_path):
    path = write_textgrid(tmp_path / "clip_001.TextGrid")

    findings = review_textgrid(path)
    reasons = {finding["reason"] for finding in findings}

    assert "short_word_interval" in reasons
    assert "very_short_phone_interval" in reasons
    assert "long_phone_interval" in reasons


def test_validate_reports_manifest_textgrid_mismatch(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "clip_id,speaker,duration,transcript\n"
        "clip_001,spk0,1.0,hello\n"
        "clip_002,spk0,1.0,world\n",
        encoding="utf-8",
    )

    textgrid_dir = tmp_path / "aligned"
    textgrid_dir.mkdir()
    write_textgrid(textgrid_dir / "clip_001.TextGrid")

    output = tmp_path / "review.csv"
    manifest_count, textgrid_count, finding_count = validate(
        manifest, textgrid_dir, output
    )

    assert manifest_count == 2
    assert textgrid_count == 1

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert any(
        row["clip_id"] == "clip_002"
        and row["reason"] == "manifest_clip_missing_textgrid"
        for row in rows
    )


def test_manifest_with_provenance_columns_is_accepted(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "clip_id,speaker,duration,transcript,"
        "transcript_normalised,rules_fired\n"
        "clip_001,spk0,1.0,Meet at 10:45 AM.,"
        "Meet at ten forty-five a m.,time\n",
        encoding="utf-8",
    )

    rows = load_manifest(manifest)

    assert rows["clip_001"]["transcript"] == "Meet at 10:45 AM."
    assert rows["clip_001"]["transcript_normalised"] == (
        "Meet at ten forty-five a m."
    )
