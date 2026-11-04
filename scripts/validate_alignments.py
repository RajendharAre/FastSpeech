#!/usr/bin/env python3
"""Flag suspicious MFA TextGrid intervals for manual alignment review.

Does not modify TextGrids or manifests, and does not determine whether
an alignment is acoustically correct.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


REQUIRED_MANIFEST_COLUMNS = {"clip_id", "speaker", "duration", "transcript"}

# Praat long-format TextGrid: one item [N] block per tier.
ITEM_RE = re.compile(
    r"(?ms)^\s*item\s*\[\d+\]\s*:\s*(.*?)(?=^\s*item\s*\[\d+\]\s*:|\Z)"
)
INTERVAL_RE = re.compile(
    r'(?ms)^\s*intervals\s*\[\d+\]\s*:[^\n]*\n'
    r'.*?^\s*xmin\s*=\s*([0-9.eE+-]+)[^\n]*\n'
    r'.*?^\s*xmax\s*=\s*([0-9.eE+-]+)[^\n]*\n'
    r'.*?^\s*text\s*=\s*"((?:[^"]|"")*)"[^\n]*(?:\n|\Z)'
)


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Tier:
    name: str
    tier_class: str
    intervals: tuple[Interval, ...]


def _field(block: str, field_name: str) -> str | None:
    match = re.search(
        rf'(?m)^\s*{re.escape(field_name)}\s*=\s*"((?:[^"]|"")*)"\s*$',
        block,
    )
    return match.group(1).replace('""', '"') if match else None


def parse_textgrid(path: Path) -> list[Tier]:
    """Parse interval tiers from a Praat long-format TextGrid."""
    content = path.read_text(encoding="utf-8-sig")
    tiers: list[Tier] = []

    for block_match in ITEM_RE.finditer(content):
        block = block_match.group(1)
        tier_class = _field(block, "class") or ""
        name = _field(block, "name") or ""

        if tier_class != "IntervalTier":
            continue

        intervals: list[Interval] = []
        for match in INTERVAL_RE.finditer(block):
            start, end = float(match.group(1)), float(match.group(2))
            label = match.group(3).replace('""', '"').strip()

            if end < start:
                raise ValueError(
                    f"{path}: interval end ({end}) precedes start ({start})"
                )

            intervals.append(Interval(start, end, label))

        tiers.append(Tier(name=name, tier_class=tier_class,
                          intervals=tuple(intervals)))

    if not tiers:
        raise ValueError(
            f"{path}: no interval tiers parsed; expected Praat long format"
        )

    return tiers


def _is_silence(label: str) -> bool:
    return label.strip().lower() in {
        "", "sil", "sp", "spn", "silence", "<sil>", "pau"
    }


def _is_phone_tier(tier: Tier) -> bool:
    name = tier.name.lower()
    return any(token in name for token in ("phone", "phoneme"))


def _is_word_tier(tier: Tier) -> bool:
    name = tier.name.lower()
    return any(token in name for token in ("word", "words"))


def review_textgrid(
    path: Path,
    *,
    short_word_seconds: float = 0.04,
    long_phone_seconds: float = 0.60,
    fast_phone_seconds: float = 0.055,
    gap_seconds: float = 0.12,
) -> list[dict[str, str | float]]:
    """Return one review row per flagged interval or adjacent-interval gap."""
    tiers = parse_textgrid(path)
    findings: list[dict[str, str | float]] = []

    def add(tier: Tier, interval: Interval, reason: str) -> None:
        findings.append({
            "clip_id": path.stem,
            "textgrid": str(path),
            "tier": tier.name,
            "start": round(interval.start, 4),
            "end": round(interval.end, 4),
            "duration": round(interval.duration, 4),
            "label": interval.text,
            "reason": reason,
        })

    for tier in tiers:
        intervals = list(tier.intervals)

        # A large gap between consecutive non-empty intervals can indicate
        # a possible boundary/alignment issue. Silence labels are excluded.
        non_silence = [i for i in intervals if not _is_silence(i.text)]
        for left, right in zip(non_silence, non_silence[1:]):
            gap = right.start - left.end
            if gap > gap_seconds:
                findings.append({
                    "clip_id": path.stem,
                    "textgrid": str(path),
                    "tier": tier.name,
                    "start": round(left.end, 4),
                    "end": round(right.start, 4),
                    "duration": round(gap, 4),
                    "label": f"{left.text} -> {right.text}",
                    "reason": "large_gap_between_non_silence_intervals",
                })

        for interval in intervals:
            label = interval.text.strip()
            if _is_silence(label):
                continue

            if _is_word_tier(tier) and interval.duration < short_word_seconds:
                add(tier, interval, "short_word_interval")
            elif _is_phone_tier(tier):
                if interval.duration > long_phone_seconds:
                    add(tier, interval, "long_phone_interval")
                elif interval.duration < fast_phone_seconds:
                    add(tier, interval, "very_short_phone_interval")

    return findings


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_MANIFEST_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{path}: missing required manifest columns: "
                f"{', '.join(sorted(missing))}"
            )

        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            clip_id = (row.get("clip_id") or "").strip()
            if not clip_id:
                continue
            if clip_id in rows:
                raise ValueError(f"{path}: duplicate clip_id {clip_id!r}")
            rows[clip_id] = row

    return rows


def validate(
    manifest_path: Path,
    textgrid_dir: Path,
    output_path: Path,
    *,
    short_word_seconds: float = 0.04,
    long_phone_seconds: float = 0.60,
    fast_phone_seconds: float = 0.055,
    gap_seconds: float = 0.12,
) -> tuple[int, int, int]:
    manifest = load_manifest(manifest_path)
    # Windows globbing is case-insensitive, so both patterns can return the
    # same path. Deduplicate while keeping distinct names on case-sensitive
    # filesystems.
    textgrids = sorted(set(
        list(textgrid_dir.rglob("*.TextGrid"))
        + list(textgrid_dir.rglob("*.textgrid"))
    ))

    if not textgrids:
        raise FileNotFoundError(f"No TextGrid files found under {textgrid_dir}")

    findings: list[dict[str, str | float]] = []
    seen_clip_ids: set[str] = set()

    for path in textgrids:
        clip_id = path.stem
        seen_clip_ids.add(clip_id)

        if clip_id not in manifest:
            findings.append({
                "clip_id": clip_id,
                "textgrid": str(path),
                "tier": "",
                "start": "",
                "end": "",
                "duration": "",
                "label": "",
                "reason": "textgrid_clip_id_not_in_manifest",
            })
            continue

        findings.extend(review_textgrid(
            path,
            short_word_seconds=short_word_seconds,
            long_phone_seconds=long_phone_seconds,
            fast_phone_seconds=fast_phone_seconds,
            gap_seconds=gap_seconds,
        ))

    missing_textgrids = sorted(set(manifest) - seen_clip_ids)
    for clip_id in missing_textgrids:
        findings.append({
            "clip_id": clip_id,
            "textgrid": "",
            "tier": "",
            "start": "",
            "end": "",
            "duration": "",
            "label": "",
            "reason": "manifest_clip_missing_textgrid",
        })

    columns = [
        "clip_id", "textgrid", "tier", "start", "end",
        "duration", "label", "reason",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(findings)

    print(f"Manifest clips: {len(manifest)}")
    print(f"TextGrids found: {len(textgrids)}")
    print(f"Flagged review rows: {len(findings)}")
    print(f"Report: {output_path}")

    return len(manifest), len(textgrids), len(findings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--textgrid-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("alignment_review.csv"))
    parser.add_argument("--short-word-seconds", type=float, default=0.04)
    parser.add_argument("--long-phone-seconds", type=float, default=0.60)
    parser.add_argument("--fast-phone-seconds", type=float, default=0.055)
    parser.add_argument("--gap-seconds", type=float, default=0.12)
    args = parser.parse_args()

    validate(
        args.manifest,
        args.textgrid_dir,
        args.output,
        short_word_seconds=args.short_word_seconds,
        long_phone_seconds=args.long_phone_seconds,
        fast_phone_seconds=args.fast_phone_seconds,
        gap_seconds=args.gap_seconds,
    )


if __name__ == "__main__":
    main()
