#!/usr/bin/env python3
"""Deterministic, duration-aware train/validation split for manifest CSVs.

Input CSV columns (extra columns are ignored):
    clip_id,speaker,duration,transcript
* ``clip_id`` is the WAV basename (without ``.wav``).
* ``duration`` is seconds of audio, measured before any trimming.
* ``transcript`` is the raw sentence text. When ``--normalise-english`` is
  used, ``transcript_normalised`` holds the text the frontend will speak and
  ``rules_fired`` lists the normalisation rules that changed it (semicolon
  separated). Both columns are omitted when no row was normalised.

The splitter:
* groups clips that share a normalized transcript and never separates a group,
  so the same sentence cannot appear in both train and validation;
* requires every validation speaker to keep at least one training clip;
* fills validation and training up to duration targets expressed in hours,
  allowing ``--tolerance`` overshoot;
* uses a seeded SHA-256 ordering instead of ``random``, so the result does not
  depend on Python's random implementation or on input row order.

Usage (from the repository root):
    python scripts/split_dataset.py \\
        --manifest manifests/all.csv \\
        --train-hours 8.0 --val-hours 0.5 \\
        --train-out English/manifests/train.csv \\
        --val-out English/manifests/val.csv
"""
import argparse
import csv
import hashlib
import math
import os
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace

REQUIRED_COLUMNS = ("clip_id", "speaker", "duration", "transcript")
OPTIONAL_COLUMNS = ("transcript_normalised", "rules_fired")


class ManifestError(ValueError):
    """Raised when a manifest is missing columns or has invalid rows."""


class SplitError(ValueError):
    """Raised when the requested split cannot be produced."""


def normalize_transcript(text):
    """Canonical form used for grouping and overlap checks."""
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


@dataclass(frozen=True)
class ManifestRow:
    clip_id: str
    speaker: str
    duration: float
    duration_text: str
    transcript: str
    line_number: int
    transcript_normalised: str = ""
    rules_fired: str = ""

    @property
    def hours(self):
        return self.duration / 3600.0

    @property
    def key(self):
        return (self.speaker, self.clip_id)

    def as_dict(self):
        return {
            "clip_id": self.clip_id,
            "speaker": self.speaker,
            "duration": self.duration_text,
            "transcript": self.transcript,
            "transcript_normalised": self.transcript_normalised,
            "rules_fired": self.rules_fired,
        }


@dataclass
class TranscriptGroup:
    key: str
    rows: list

    @property
    def hours(self):
        return sum(row.hours for row in self.rows)

    @property
    def speakers(self):
        return sorted({row.speaker for row in self.rows})


def load_manifest(path):
    """Read and validate a manifest CSV. Blank lines are ignored."""
    if not os.path.exists(path):
        raise ManifestError("manifest not found: {}".format(path))

    rows = []
    seen = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ManifestError("{}: empty manifest".format(path))
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ManifestError(
                "{}: missing column(s): {}".format(path, ", ".join(missing))
            )

        for line_number, raw in enumerate(reader, start=2):
            values = {
                column: (raw.get(column) or "").strip() for column in REQUIRED_COLUMNS
            }
            if not any(values.values()):
                continue
            for column in REQUIRED_COLUMNS:
                if not values[column]:
                    raise ManifestError(
                        "{}:{}: column '{}' is empty".format(path, line_number, column)
                    )
            for column in ("clip_id", "speaker"):
                value = values[column]
                if "/" in value or "\\" in value or value in (".", ".."):
                    raise ManifestError(
                        "{}:{}: '{}' must not be a path: {!r}".format(
                            path, line_number, column, value
                        )
                    )

            try:
                duration = float(values["duration"])
            except ValueError:
                raise ManifestError(
                    "{}:{}: duration {!r} is not a number".format(
                        path, line_number, values["duration"]
                    )
                )
            if not math.isfinite(duration) or duration <= 0:
                raise ManifestError(
                    "{}:{}: duration must be finite and > 0, got {!r}".format(
                        path, line_number, values["duration"]
                    )
                )

            if not normalize_transcript(values["transcript"]):
                raise ManifestError(
                    "{}:{}: transcript is empty after normalization".format(
                        path, line_number
                    )
                )

            key = (values["speaker"], values["clip_id"])
            if key in seen:
                raise ManifestError(
                    "{}:{}: duplicate clip {}/{} (first on line {})".format(
                        path,
                        line_number,
                        key[0],
                        key[1],
                        seen[key],
                    )
                )
            seen[key] = line_number

            rows.append(
                ManifestRow(
                    clip_id=values["clip_id"],
                    speaker=values["speaker"],
                    duration=duration,
                    duration_text=values["duration"],
                    transcript=values["transcript"],
                    line_number=line_number,
                    transcript_normalised=(raw.get("transcript_normalised") or "").strip(),
                    rules_fired=(raw.get("rules_fired") or "").strip(),
                )
            )

    if not rows:
        raise ManifestError("{}: manifest contains no rows".format(path))
    return rows


def write_manifest(path, rows):
    """Write a manifest in canonical column/row order.

    The provenance columns are only written when at least one row carries
    them, so un-normalised (for example Telugu) manifests keep four columns.
    """
    fieldnames = list(REQUIRED_COLUMNS)
    if any(row.transcript_normalised or row.rules_fired for row in rows):
        fieldnames.extend(OPTIONAL_COLUMNS)

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.speaker.casefold(), item.clip_id)):
            writer.writerow(row.as_dict())


def _group_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(normalize_transcript(row.transcript), []).append(row)
    return [
        TranscriptGroup(key=key, rows=grouped[key]) for key in sorted(grouped)
    ]


def _shuffle_key(seed, group_key):
    payload = "{}:{}".format(seed, group_key).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_rows(rows, train_hours, val_hours, seed=1234, tolerance=0.1):
    """Split manifest rows; returns ``(train_rows, val_rows, unused_rows, summary)``."""
    if train_hours <= 0 or val_hours <= 0:
        raise SplitError("--train-hours and --val-hours must both be > 0")
    if tolerance < 0:
        raise SplitError("--tolerance must be >= 0")

    groups = _group_rows(rows)
    total_hours = sum(group.hours for group in groups)
    requested = train_hours + val_hours
    if requested > total_hours + 1e-9:
        raise SplitError(
            "requested {:.3f} h (train {:.3f} + val {:.3f}) but the manifest "
            "contains {:.3f} h".format(requested, train_hours, val_hours, total_hours)
        )

    ordered = sorted(
        groups,
        key=lambda group: (_shuffle_key(seed, group.key), group.key),
    )

    groups_per_speaker = Counter()
    for group in groups:
        for speaker in group.speakers:
            groups_per_speaker[speaker] += 1

    # Never let validation consume hours needed by the training target.
    val_budget = min(val_hours * (1.0 + tolerance), total_hours - train_hours)
    selected_val = []
    val_hours_done = 0.0
    val_groups_per_speaker = Counter()

    for group in ordered:
        if val_hours_done >= val_hours:
            break
        if val_hours_done + group.hours > val_budget + 1e-12:
            continue
        if any(
            val_groups_per_speaker[speaker] + 1 >= groups_per_speaker[speaker]
            for speaker in group.speakers
        ):
            # Taking this group would leave at least one speaker without
            # any training clip.
            continue

        selected_val.append(group)
        val_hours_done += group.hours
        for speaker in group.speakers:
            val_groups_per_speaker[speaker] += 1

    if not selected_val:
        smallest = min(group.hours for group in groups)
        raise SplitError(
            "could not select a validation group: --val-hours={:.3f} with "
            "tolerance {:.0%} cannot fit the smallest transcript group "
            "({:.4f} h) while leaving a training clip for every speaker".format(
                val_hours, tolerance, smallest
            )
        )

    val_keys = {group.key for group in selected_val}
    train_budget = train_hours * (1.0 + tolerance)
    selected_train = []
    train_hours_done = 0.0
    for group in ordered:
        if group.key in val_keys:
            continue
        if train_hours_done + group.hours > train_budget + 1e-12:
            continue
        selected_train.append(group)
        train_hours_done += group.hours

    if not selected_train:
        raise SplitError("could not select any training group")

    used_keys = val_keys | {group.key for group in selected_train}
    unused_groups = [group for group in groups if group.key not in used_keys]

    train_rows = [row for group in selected_train for row in group.rows]
    val_rows = [row for group in selected_val for row in group.rows]
    unused_rows = [row for group in unused_groups for row in group.rows]

    train_speakers = {row.speaker for row in train_rows}
    val_speakers = {row.speaker for row in val_rows}
    missing_speakers = sorted(val_speakers - train_speakers)
    if missing_speakers:
        raise SplitError(
            "internal error: validation speakers missing from train: {}".format(
                ", ".join(missing_speakers)
            )
        )

    unused_hours = sum(row.hours for row in unused_rows)
    summary = {
        "total_hours": total_hours,
        "train_hours": train_hours_done,
        "val_hours": val_hours_done,
        "unused_hours": unused_hours,
        "train_clips": len(train_rows),
        "val_clips": len(val_rows),
        "unused_clips": len(unused_rows),
        "train_groups": len(selected_train),
        "val_groups": len(selected_val),
        "train_speakers": len(train_speakers),
        "val_speakers": len(val_speakers),
        "train_target_hours": train_hours,
        "val_target_hours": val_hours,
        "tolerance": tolerance,
        "seed": seed,
    }
    return train_rows, val_rows, unused_rows, summary


def _format_summary(summary):
    lines = [
        "total: {total_hours:.3f} h",
        "train: {train_clips} clips, {train_hours:.3f} h "
        "(target {train_target_hours:.3f} h), {train_speakers} speakers",
        "val:   {val_clips} clips, {val_hours:.3f} h "
        "(target {val_target_hours:.3f} h), {val_speakers} speakers",
        "unused: {unused_clips} clips, {unused_hours:.3f} h",
        "seed: {seed}, tolerance: {tolerance:.0%}",
    ]
    return "\n".join(line.format(**summary) for line in lines)


def normalise_rows(rows, normaliser):
    """Attach the normalised transcript and fired rules to each row."""
    normalised_rows = []
    for row in rows:
        text, rules = normaliser(row.transcript)
        if not normalize_transcript(text):
            raise SplitError(
                "clip {}/{}: transcript is empty after normalisation".format(
                    row.speaker, row.clip_id
                )
            )
        normalised_rows.append(
            replace(
                row,
                transcript_normalised=text,
                rules_fired=";".join(rules),
            )
        )
    return normalised_rows


def load_english_normaliser(english_root):
    """Import the English normaliser without affecting Telugu runs."""
    root = os.path.abspath(english_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from text.normalise import normalise_english

    return normalise_english


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Deterministic, duration-aware train/validation split."
    )
    parser.add_argument("--manifest", required=True, help="input manifest CSV")
    parser.add_argument("--train-hours", type=float, required=True)
    parser.add_argument("--val-hours", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="allowed overshoot of each duration target (default 0.1 = 10%%)",
    )
    parser.add_argument("--train-out", default="manifests/train.csv")
    parser.add_argument("--val-out", default="manifests/val.csv")
    parser.add_argument(
        "--unused-out",
        default=None,
        help="optional CSV for clips not selected by either target",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing existing output manifests",
    )
    parser.add_argument(
        "--normalise-english",
        action="store_true",
        help="add transcript_normalised and rules_fired using the English rules",
    )
    parser.add_argument(
        "--english-root",
        default="English",
        help="language folder containing text/normalise.py (default: English)",
    )
    args = parser.parse_args(argv)

    outputs = [args.train_out, args.val_out]
    if args.unused_out:
        outputs.append(args.unused_out)
    existing = [path for path in outputs if os.path.exists(path)]
    if existing and not args.overwrite:
        raise SystemExit(
            "refusing to overwrite existing manifest(s): {} (pass --overwrite)".format(
                ", ".join(existing)
            )
        )

    try:
        rows = load_manifest(args.manifest)
        if args.normalise_english:
            try:
                normaliser = load_english_normaliser(args.english_root)
                rows = normalise_rows(rows, normaliser)
            except (ImportError, OSError, ValueError) as exc:
                raise SystemExit(
                    "error: English normalisation failed: {}".format(exc)
                )
        train_rows, val_rows, unused_rows, summary = split_rows(
            rows,
            args.train_hours,
            args.val_hours,
            seed=args.seed,
            tolerance=args.tolerance,
        )
    except (ManifestError, SplitError) as exc:
        raise SystemExit("error: {}".format(exc))

    write_manifest(args.train_out, train_rows)
    write_manifest(args.val_out, val_rows)
    if args.unused_out:
        write_manifest(args.unused_out, unused_rows)

    print(_format_summary(summary))
    print("wrote train: {}".format(args.train_out))
    print("wrote val:   {}".format(args.val_out))
    if args.unused_out:
        print("wrote unused: {}".format(args.unused_out))

    lower = 1.0 - args.tolerance
    if summary["train_hours"] < args.train_hours * lower:
        print(
            "warning: train is below the target by more than the tolerance",
        )
    if summary["val_hours"] < args.val_hours * lower:
        print("warning: val is below the target by more than the tolerance")


if __name__ == "__main__":
    main()
