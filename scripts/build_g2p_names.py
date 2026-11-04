#!/usr/bin/env python3
"""Build a G2P supplementary pronunciation dictionary for proper names.

Reads explicit names (``--names``) or extracts capitalised candidates from
manifests, skips words already in the main lexicon, and runs ``g2p_en`` once
per name. The output is a LibriSpeech-style ``WORD  PHONES`` file that the
English frontend loads after the main lexicon.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
COMMENT_PREFIX = "#"


def read_names(path: Path) -> list[str]:
    """Read one name per line; blank lines and ``#`` comments are ignored."""
    names = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            name = line.strip()
            if not name or name.startswith(COMMENT_PREFIX):
                continue
            names.append(name)
    if not names:
        raise ValueError(f"{path}: no names found")
    return names


def read_manifest_transcripts(paths: list[Path]) -> list[str]:
    transcripts = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "transcript" not in reader.fieldnames:
                raise ValueError(f"{path}: missing 'transcript' column")
            for row in reader:
                text = (row.get("transcript") or "").strip()
                if text:
                    transcripts.append(text)
    if not transcripts:
        raise ValueError("no transcripts found in the supplied manifests")
    return transcripts


def read_lexicon_words(path: Path) -> set[str]:
    words = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0].startswith(";;;"):
                continue
            words.add(parts[0].casefold())
    return words


def extract_candidates(
    transcripts: list[str],
    lexicon_words: set[str],
    min_count: int = 1,
) -> list[tuple[str, int]]:
    """Return capitalised, non-common, out-of-lexicon tokens with counts.

    A token is rejected as a common word when its lowercase form occurs more
    often than its capitalised form, which is what sentence-initial ``The``
    looks like in a corpus that also contains lowercase ``the``.
    """
    lowercase_counts = Counter()
    capitalised_counts = Counter()

    for text in transcripts:
        for token in TOKEN_RE.findall(text):
            lowercase_counts[token.casefold()] += 1
            if token[0].isupper() and any(character.islower() for character in token):
                capitalised_counts[token] += 1

    candidates = []
    for name, count in capitalised_counts.items():
        if len(name) < 2 or count < min_count:
            continue
        key = name.casefold()
        if key in lexicon_words:
            continue
        if lowercase_counts[key] > count:
            continue
        candidates.append((name, count))

    return sorted(candidates, key=lambda item: (-item[1], item[0].casefold()))


def phones_for(name: str, g2p) -> list[str]:
    return [phone for phone in g2p(name) if phone and phone.strip()]


def build_entries(names, lexicon_words, g2p):
    """Return ``(entries, report)`` for ``[(name, count), ...]``."""
    entries = []
    report = []
    seen = set()

    for name, count in names:
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)

        if key in lexicon_words:
            report.append({
                "name": name,
                "count": count,
                "phones": "",
                "status": "already_in_lexicon",
            })
            continue

        phones = phones_for(name, g2p)
        if not phones:
            report.append({
                "name": name,
                "count": count,
                "phones": "",
                "status": "no_phones",
            })
            continue

        entries.append((name, phones))
        report.append({
            "name": name,
            "count": count,
            "phones": " ".join(phones),
            "status": "written",
        })

    return entries, report


def write_dictionary(path: Path, entries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for name, phones in sorted(entries, key=lambda item: item[0].casefold()):
            handle.write("{}\t{}\n".format(name.upper(), " ".join(phones)))


def write_report(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("name", "count", "phones", "status")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_g2p():
    try:
        from g2p_en import G2p
    except ImportError as exc:
        raise SystemExit(
            "error: g2p_en is required to generate name pronunciations; "
            "install requirements.txt first ({})".format(exc)
        )
    return G2p()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", type=Path, help="explicit names, one per line")
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="manifest to extract capitalised candidates from (repeatable)",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=Path("English/lexicon/librispeech-lexicon.txt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("English/lexicon/g2p_names.txt"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("English/lexicon/g2p_names_report.csv"),
    )
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args(argv)

    if not args.names and not args.manifest:
        raise SystemExit("error: pass --names or at least one --manifest")
    if args.min_count < 1:
        raise SystemExit("error: --min-count must be >= 1")

    lexicon_words = read_lexicon_words(args.lexicon)
    if args.names:
        names = [(name, 1) for name in read_names(args.names)]
    else:
        transcripts = read_manifest_transcripts(args.manifest)
        names = extract_candidates(transcripts, lexicon_words, args.min_count)

    if not names:
        raise SystemExit("error: no name candidates found")

    entries, report = build_entries(names, lexicon_words, load_g2p())
    write_dictionary(args.output, entries)
    write_report(args.report, report)

    written = sum(1 for row in report if row["status"] == "written")
    existing = sum(1 for row in report if row["status"] == "already_in_lexicon")
    missing = sum(1 for row in report if row["status"] == "no_phones")
    print(f"Names considered: {len(report)}")
    print(f"Written: {written}")
    print(f"Already in lexicon: {existing}")
    print(f"No phones: {missing}")
    print(f"Dictionary: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
