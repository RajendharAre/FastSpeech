#!/usr/bin/env python3
"""Explicit train/validation overlap checks for manifest CSVs.

Checks:
* duplicate ``(speaker, clip_id)`` inside a split;
* the same ``(speaker, clip_id)`` in both splits;
* the same normalized transcript in both splits;
* validation speakers that have no training clip;
* optional achieved-duration checks against expected hours.

Usage (from the repository root):
    python scripts/check_no_overlap.py \\
        --train English/manifests/train.csv \\
        --val English/manifests/val.csv \\
        --expect-train-hours 8.0 --expect-val-hours 0.5
"""
import argparse
from collections import Counter

from split_dataset import ManifestError, load_manifest, normalize_transcript

SAMPLE_SIZE = 5


def _duplicates(rows):
    counts = Counter(row.key for row in rows)
    return sorted(key for key, count in counts.items() if count > 1)


def _hours(rows):
    return sum(row.hours for row in rows)


def _sample(items):
    items = list(items)
    if len(items) <= SAMPLE_SIZE:
        return items
    return items[:SAMPLE_SIZE] + ["... ({} total)".format(len(items))]


def check_manifests(
    train_rows,
    val_rows,
    require_speaker_coverage=True,
    expected_train_hours=None,
    expected_val_hours=None,
    tolerance=0.1,
):
    """Return a report dictionary; ``report["violations"]`` is empty when clean."""
    violations = []

    duplicate_train = _duplicates(train_rows)
    duplicate_val = _duplicates(val_rows)
    if duplicate_train:
        violations.append(
            "duplicate clips inside train: {}".format(_sample(duplicate_train))
        )
    if duplicate_val:
        violations.append(
            "duplicate clips inside validation: {}".format(_sample(duplicate_val))
        )

    train_keys = {row.key for row in train_rows}
    val_keys = {row.key for row in val_rows}
    clips_in_both = sorted(train_keys & val_keys)
    if clips_in_both:
        violations.append(
            "clips present in both train and validation: {}".format(
                _sample(clips_in_both)
            )
        )

    train_transcripts = {normalize_transcript(row.transcript) for row in train_rows}
    val_transcripts = {normalize_transcript(row.transcript) for row in val_rows}
    shared_transcripts = sorted(train_transcripts & val_transcripts)
    if shared_transcripts:
        violations.append(
            "transcripts present in both train and validation: {}".format(
                _sample(shared_transcripts)
            )
        )

    train_speakers = {row.speaker for row in train_rows}
    val_speakers = {row.speaker for row in val_rows}
    missing_train_speakers = sorted(val_speakers - train_speakers)
    if require_speaker_coverage and missing_train_speakers:
        violations.append(
            "validation speakers with no training clip: {}".format(
                _sample(missing_train_speakers)
            )
        )

    if not train_rows:
        violations.append("training split is empty")
    if not val_rows:
        violations.append("validation split is empty")

    train_hours = _hours(train_rows)
    val_hours = _hours(val_rows)
    if expected_train_hours is not None:
        low = expected_train_hours * (1.0 - tolerance)
        high = expected_train_hours * (1.0 + tolerance)
        if not low - 1e-9 <= train_hours <= high + 1e-9:
            violations.append(
                "train duration {:.3f} h is outside {:.3f} h +/- {:.0%}".format(
                    train_hours, expected_train_hours, tolerance
                )
            )
    if expected_val_hours is not None:
        low = expected_val_hours * (1.0 - tolerance)
        high = expected_val_hours * (1.0 + tolerance)
        if not low - 1e-9 <= val_hours <= high + 1e-9:
            violations.append(
                "validation duration {:.3f} h is outside {:.3f} h +/- {:.0%}".format(
                    val_hours, expected_val_hours, tolerance
                )
            )

    return {
        "train_clips": len(train_rows),
        "val_clips": len(val_rows),
        "train_hours": train_hours,
        "val_hours": val_hours,
        "train_speakers": sorted(train_speakers),
        "val_speakers": sorted(val_speakers),
        "duplicate_train_clips": duplicate_train,
        "duplicate_val_clips": duplicate_val,
        "clips_in_both": clips_in_both,
        "shared_transcripts": shared_transcripts,
        "val_speakers_missing_from_train": missing_train_speakers,
        "violations": violations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check train/validation manifests for overlap."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--expect-train-hours", type=float, default=None)
    parser.add_argument("--expect-val-hours", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument(
        "--allow-unseen-val-speakers",
        action="store_true",
        help="do not require every validation speaker to appear in train",
    )
    args = parser.parse_args(argv)

    try:
        train_rows = load_manifest(args.train)
        val_rows = load_manifest(args.val)
    except ManifestError as exc:
        raise SystemExit("error: {}".format(exc))

    report = check_manifests(
        train_rows,
        val_rows,
        require_speaker_coverage=not args.allow_unseen_val_speakers,
        expected_train_hours=args.expect_train_hours,
        expected_val_hours=args.expect_val_hours,
        tolerance=args.tolerance,
    )

    print(
        "train: {} clips, {:.3f} h, {} speakers".format(
            report["train_clips"],
            report["train_hours"],
            len(report["train_speakers"]),
        )
    )
    print(
        "val:   {} clips, {:.3f} h, {} speakers".format(
            report["val_clips"],
            report["val_hours"],
            len(report["val_speakers"]),
        )
    )

    if report["violations"]:
        for violation in report["violations"]:
            print("FAIL: {}".format(violation))
        raise SystemExit(1)

    print("OK: no overlap or coverage violations")
    return 0


if __name__ == "__main__":
    main()
